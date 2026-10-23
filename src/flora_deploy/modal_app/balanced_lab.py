"""``flora-balanced-lab``: the balanced engines on real GPUs (P1b-1 validation). Ephemeral only.

Not the factory: this lab runs the engines and stages directly, on a few golden episodes from
``flora-golden:golden/v1/``, and measures four things per model kind (``docs/balanced-engines.md``):

(a) **L0 in-engine**: for captured requests, vLLM's ``prompt_token_ids`` (from the completion)
    and ``video_grid_thw`` (``LLM._preprocess_cmpl``, private; recheck on a vLLM bump) equal
    NVIDIA's HF processor on the same captured request (``nv.capture.replay``);
(b) segments (TempSeg, ``L40S``) and captions (Captioner, ``H100!``, on the TempSeg run's
    segments) end to end through ``stages.segment`` / ``stages.caption`` + the micro-batcher;
(c) throughput: video-seconds per GPU-second of the stage, and $/video-hour at list prices;
(d) **Gate A**: greedy generations with top-20 raw logits on the same requests, without and with
    the ``dt_bias`` FP32 plugin (``engines.patches.dt_bias_fp32``): max/mean |Δlogit| over the
    shared top-k at every position up to the first divergence, and greedy agreement.

Each engine runs in its own child process (``python -c ...``) so the plugin variant gets a fresh
engine-core with the plugin env, and GPU memory is released between them. Child args (a JSON
file, never argv), results and the Gate A requests go through ``/tmp`` in the container; the
TempSeg run also writes its segment
rows to ``flora-golden:balanced-lab/<run>/segments.json`` (private Volume) for the caption run.

A baseline run that sends zero requests (no episode ran) returns ``error`` and
``no_evidence`` and skips the plugin engine. The compile caches are committed after each child,
and each result records whether its cache was warm (``compile_cache``).

**Split calls** (``arm``): one call builds both engines (``"both"``, the default), or one engine
per call, each in its own container under its own timeout: ``"baseline"`` (L0, the stage, the
Gate A requests and baseline logits, which it stages to
``flora-golden:balanced-lab/<run>/<kind>-gate/``) and then ``"plugin"`` (``gate_rel`` names
that dir: the plugin engine's Gate A on the same requests, compared with the staged baseline
logits; ``gate_sha256`` binds it to the baseline call). ``scripts/wave_h.py`` splits the
Captioner: its two cold builds and the v4 caption stage do not fit one 20-minute container.

Run (each call is one GPU container, hard timeout 20 min)::

    uv run modal run src/flora_deploy/modal_app/balanced_lab.py::preflight          # CPU, cents
    uv run modal run src/flora_deploy/modal_app/balanced_lab.py --kind tempseg --run r1
    uv run modal run src/flora_deploy/modal_app/balanced_lab.py --kind captioner --run r1
    uv run python scripts/spend.py --app flora-balanced-lab
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import (
    CACHE_MOUNT,
    CACHE_VOLUME,
    GOLDEN_MOUNT,
    GOLDEN_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    derived_path,
    snapshot_path,
)
from flora_deploy.modal_app.images import cpu_image, vllm_image

app = modal.App("flora-balanced-lab")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2).read_only()
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)

MODEL_DIRS = {
    "tempseg": derived_path("tempseg-merged", PINS_BY_KEY["tempseg"]),
    "captioner": derived_path("captioner-bf16", PINS_BY_KEY["captioner"]),
}
GPUS = {"tempseg": "L40S", "captioner": "H100!"}
# Modal list prices ($/h): GPU + 8 cores x $0.0472 + memory GiB x $0.008.
GPU_USD_H = {"tempseg": 1.95, "captioner": 3.95}
CPU, MEM_GIB = 8.0, {"tempseg": 48, "captioner": 96}
TIMEOUT_S = 20 * 60
POOLS = ("human", "yam", "single_view_robot", "dex")  # one shortest episode each
GATE_A_REQUESTS = 6
L0_REQUESTS = 6
ENV = {"PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}


def usd_per_h(kind: str) -> float:
    return GPU_USD_H[kind] + CPU * 0.0472 + MEM_GIB[kind] * 0.008


def cache_env(kind: str, plugin: bool) -> dict[str, str]:
    base = Path(CACHE_MOUNT) / "balanced" / f"{kind}{'-dtbias' if plugin else ''}"
    return {"VLLM_CACHE_ROOT": str(base / "vllm"), "TRITON_CACHE_DIR": str(base / "triton"),
            "CUDA_CACHE_PATH": str(base / "cuda"), "XDG_CACHE_HOME": str(base / "xdg")}


def episode_inputs(manifest: dict, ids, mount: PurePosixPath | str = GOLDEN_MOUNT) -> list[dict]:
    """``golden_episode_inputs`` for ``ids`` with camera files under the golden mount.

    The manifest's ``path`` is Volume-relative (``golden/v1/<pool>/<dataset>/<slug>``,
    ``validate.golden.episode_relpath``), so the root is the mount itself. The 2026-10-22 lab
    joined ``<mount>/golden/v1`` + ``golden/v1/...``: every episode raised FileNotFoundError,
    the stage sent zero requests and the call still returned without an error."""
    from flora_deploy.stages.reference import golden_episode_inputs

    return golden_episode_inputs(manifest, ids, PurePosixPath(mount))


def volume_paths(inputs: list[dict], mount: PurePosixPath | str = GOLDEN_MOUNT) -> list[str]:
    """The Volume-relative path of every camera file the child will open (for a check before
    any container starts); raises if one is not under ``mount``."""
    out = []
    for e in inputs:
        for path in e["videos"].values():
            out.append(str(PurePosixPath(path).relative_to(PurePosixPath(mount))))
    return out


def evidence_problems(ids: list[str], episodes: list[dict], requests: int) -> list[str]:
    """Why a baseline run produced no L0 / Gate A evidence (empty: it did)."""
    out = []
    if not episodes:
        out.append(f"no episode ran (of {len(ids)} requested)")
    if requests == 0:
        out.append("the stage sent zero requests to the engine")
    return out


def pick_episodes(manifest: dict, pools=POOLS) -> list[str]:
    """The shortest oracle-eligible episode of each pool (manifest order breaks ties)."""
    out = []
    for pool in pools:
        eps = [e for e in manifest.get("episodes", [])
               if e.get("pool") == pool and e.get("oracle_eligible")]
        if eps:
            out.append(min(eps, key=lambda e: float(e.get("duration_s") or 1e9))["id"])
    return out


# ---------------------------------------------------------------------------------------------
# Child process: one engine
# ---------------------------------------------------------------------------------------------


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _l0(llm, proc, requests, completions) -> list[dict]:
    from flora_deploy.nv.capture import CapturedRequest, replay
    from flora_deploy.stages.segment import unwrap_prompt

    rows = []
    for r, c in zip(requests, completions, strict=True):
        cap = CapturedRequest(prompt=unwrap_prompt(r.prompt), videos=list(r.videos),
                              call_kwargs={"return_tensors": "pt"})
        hf = replay(proc, cap)
        hf_ids = hf["input_ids"][0].tolist()
        hf_grid = hf["video_grid_thw"].reshape(-1, 3).tolist()
        (ei,) = llm._preprocess_cmpl([r.prompt_dict()])
        grids = []
        for item in ei["mm_kwargs"]["video"]:
            data = item["video_grid_thw"].data
            grids += (data.cpu() if hasattr(data, "cpu") else data).reshape(-1, 3).tolist()
        rows.append({"key": r.key, "ids_equal": list(c.prompt_token_ids or []) == hf_ids,
                     "render_ids_equal": list(ei["prompt_token_ids"]) == hf_ids,
                     "grid_equal": grids == hf_grid, "prompt_tokens": len(hf_ids),
                     "videos": len(r.videos)})
    return rows


def engine_hash(spec: Any) -> str:
    """``certkey.engine_settings_hash`` of the engine ``spec`` (an ``EngineSpec``): the hash a
    certificate's L0 and ``scripts/wave_h.py``'s lab reuse compare with the current engine's
    (review 13 S3-1). The Gate A logprob arguments only change what requests return, so the
    lab's engines hash as the workers' do."""
    from flora_deploy.validate.certkey import engine_settings_hash

    return engine_settings_hash(spec.hashed())


def child(args: dict) -> None:
    """One engine in this process (``args`` from ``run_kind``); writes ``args['out']``."""
    import asyncio
    import pickle

    from flora_deploy.engines.microbatch import MicroBatcher
    from flora_deploy.engines.offline_llm import EngineSpec, OfflineEngine, gdn_param_dtypes
    from flora_deploy.engines.patches import dt_bias_fp32
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.stages import caption as C
    from flora_deploy.stages import segment as S

    kind, plugin = args["kind"], args["plugin"]
    out: dict[str, Any] = {"kind": kind, "plugin": plugin, "gpu": _gpu_name(),
                           "plugin_provenance": dt_bias_fp32.provenance()}
    t0 = time.time()
    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"])))
    model_dir = str(MODEL_DIRS[kind])
    proc = nv.models.processor(kind, Path(model_dir))
    spec = EngineSpec.for_gpu(kind, model_dir, args["gpu"], dtbias_fp32=plugin,
                              logprobs_mode="raw_logits", max_logprobs=20)
    engine = OfflineEngine(spec, keep_prompt_ids=True)  # verifies the GDN dtypes (M6)
    out["engine_build_s"] = round(time.time() - t0, 1)
    out["spec"] = spec.as_json()
    out["engine_hash"] = engine_hash(spec)
    out["gdn_dtypes"] = engine.gdn_dtypes or gdn_param_dtypes(engine.llm)
    eos = proc.tokenizer.eos_token_id

    if not args.get("gate_only"):
        manifest = json.loads((Path(GOLDEN_MOUNT) / "golden/v1/manifest.json").read_text())
        ids = args.get("episodes") or pick_episodes(manifest)
        inputs = episode_inputs(manifest, ids)  # raises on an id the manifest lacks
        out["episodes_requested"] = list(ids)
        segments_in = args.get("segments") or {}
        if kind == "captioner" and (miss := [i for i in ids if not segments_in.get(i)]):
            out["error"] = f"no segments for {len(miss)} of {len(ids)} episodes: {miss}"
            out["no_evidence"] = True
            Path(args["out"]).write_text(json.dumps(out, default=str))
            return
        seen: list = []

        def observer(reqs, comps, info):
            for r, c in zip(reqs, comps, strict=True):
                if len(seen) < max(L0_REQUESTS, GATE_A_REQUESTS):
                    seen.append((r, c))

        stage = S.SegmentStage(nv, proc) if kind == "tempseg" else C.CaptionStage(nv, proc)

        async def one(mb, e):
            async with mb.episode():
                t = time.time()
                try:
                    ep = await asyncio.to_thread(
                        S.prepare_videos, nv, e["videos"], e["profile"],
                        yam32=e["long_windows"], instruction=e["instruction"])
                    tp = time.time() - t
                    if kind == "tempseg":
                        r = await stage.segment(mb, ep)
                        res = {"rows": S.json_safe(r.rows), "apredict_rounds": r.apredict_rounds,
                               "calls": r.stats.as_json()}
                    else:
                        r = await stage.caption(mb, ep, segments_in[e["id"]])
                        res = {"rows": S.json_safe(r.rows), "calls": r.stats.as_json()}
                    return {"id": e["id"], "ok": True, "duration_s": ep.duration_s,
                            "probe_s": round(tp, 2), "wall_s": round(time.time() - t, 2), **res}
                except Exception as exc:  # recorded per episode
                    return {"id": e["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}

        async def main():
            async with MicroBatcher(engine, max_wait_s=2.0, observer=observer) as mb:
                t = time.time()
                eps = await asyncio.gather(*(one(mb, e) for e in inputs
                                             if kind == "tempseg" or e["id"] in segments_in))
                return eps, time.time() - t, mb.stats.as_json()

        eps, wall, bstats = asyncio.run(main())
        stage.close()
        video_s = sum(e.get("duration_s", 0.0) for e in eps if e["ok"])
        out.update(episodes=eps, stage_wall_s=round(wall, 2), video_s=round(video_s, 2),
                   batcher=bstats, engine_stats=dict(engine.stats))
        out["throughput"] = {
            "video_s_per_gpu_s": round(video_s / wall, 4) if wall else None,
            "usd_per_video_h": round(usd_per_h(kind) * wall / video_s, 2) if video_s else None,
            "generate_s": round(engine.stats["generate_s"], 2)}
        failed = [e for e in eps if not e["ok"]]
        if failed:  # evidence from the rest is kept; wave_h does not count this kind as ok
            out["episode_errors"] = {e["id"]: e["error"] for e in failed}
        problems = evidence_problems(list(ids), [e for e in eps if e["ok"]],
                                     int(bstats.get("requests") or 0))
        if problems:  # fail loudly: no L0, no Gate A requests, no plugin engine after this
            out["error"] = "; ".join(problems) + (
                f" (first episode error: {failed[0]['error']})" if failed else "")
            out["no_evidence"] = True
            out["l0"] = []
            Path(args["out"]).write_text(json.dumps(out, default=str))
            return
        out["l0"] = _l0(engine.llm, proc, [r for r, _ in seen[:L0_REQUESTS]],
                        [c for _, c in seen[:L0_REQUESTS]])
        with open(args["gate_requests"], "wb") as f:
            pickle.dump([r for r, _ in seen[:GATE_A_REQUESTS]], f, protocol=5)

    with open(args["gate_requests"], "rb") as f:
        gate_reqs = pickle.load(f)
    if not gate_reqs:
        out["error"], out["no_evidence"] = "no Gate A requests captured", True
        Path(args["out"]).write_text(json.dumps(out, default=str))
        return
    budget = 1024 if kind == "tempseg" else 512
    sp = S.nvidia_sampling(budget, eos, logprobs=20)
    comps = engine.generate_batch(gate_reqs, [sp] * len(gate_reqs))
    out["gate_a"] = [{"key": c.key, "ids": list(c.token_ids), "error": c.error,
                      "logits": [{str(k): v for k, v in pos.items()} for pos in (c.logprobs or ())]}
                     for c in comps]
    Path(args["out"]).write_text(json.dumps(out, default=str))


def gate_a_compare(base: list[dict], patched: list[dict]) -> dict:
    """max/mean |Δlogit| over shared top-k ids per position up to the first greedy divergence."""
    deltas, agree, first_div, positions = [], 0, [], 0
    for a, b in zip(base, patched, strict=True):
        ia, ib = a["ids"], b["ids"]
        n = 0
        while n < min(len(ia), len(ib)) and ia[n] == ib[n]:
            n += 1
        agree += int(ia == ib)
        if ia != ib:
            first_div.append(n)
        for pos in range(min(n + 1, len(a["logits"]), len(b["logits"]))):
            la, lb = a["logits"][pos], b["logits"][pos]
            for tok in set(la) & set(lb):
                deltas.append(abs(float(la[tok]) - float(lb[tok])))
            positions += 1
    return {"requests": len(base), "greedy_identical": agree, "first_divergence": first_div,
            "positions": positions,
            "max_abs_dlogit": round(max(deltas), 5) if deltas else None,
            "mean_abs_dlogit": round(sum(deltas) / len(deltas), 6) if deltas else None}


CHILD_CODE = ("import json,sys,pathlib; from flora_deploy.modal_app.balanced_lab import child; "
              "child(json.loads(pathlib.Path(sys.argv[1]).read_text()))")


def child_argv(args: dict) -> list[str]:
    """The child's command line. ``args`` go through a JSON file next to ``args['out']`` (its
    path in argv), never inline: with the deployed job's segments one argv string passed
    Linux's 128 KiB per-argument limit (E2BIG, 2026-10-22 captioner lab)."""
    path = Path(args["out"]).with_suffix(".args.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(args))
    return [sys.executable, "-c", CHILD_CODE, str(path)]


def _run_child(args: dict, env: dict[str, str]) -> dict:
    t = time.time()
    p = subprocess.run(child_argv(args), env={**os.environ, **env},
                       capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": f"child exit {p.returncode}", "stderr_tail": p.stderr[-6000:],
                "wall_s": round(time.time() - t, 1)}
    out = json.loads(Path(args["out"]).read_text())
    out["child_wall_s"] = round(time.time() - t, 1)
    return out


# ---------------------------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------------------------


@app.function(image=cpu_image(), cpu=2.0, memory=4096, timeout=600,
              volumes={str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden.read_only()})
def preflight() -> dict:
    """CPU: derived dirs exist with receipts; ``_extra_state`` key prefixes vLLM must skip
    (raises if any is under ``model.visual``: vLLM's vision loader would KeyError); EOS ids; the
    golden episodes the GPU runs would pick."""
    from flora_deploy.engines.offline_llm import check_extra_state

    out: dict[str, Any] = {}
    for kind, d in MODEL_DIRS.items():
        d = Path(d)
        idx = json.loads((d / "model.safetensors.index.json").read_text())
        extra = check_extra_state(idx["weight_map"])  # fails on model.visual (review 6 Q5)
        tok = json.loads((d / "tokenizer_config.json").read_text())
        out[kind] = {"dir": str(d), "receipt": (d / "derived_receipt.json").is_file(),
                     "extra_state_prefixes": extra, "eos_token": tok.get("eos_token"),
                     "files": sorted(p.name for p in d.iterdir())}
    manifest = json.loads((Path(GOLDEN_MOUNT) / "golden/v1/manifest.json").read_text())
    out["episodes"] = pick_episodes(manifest)
    return out


def compile_cache_state(kind: str, plugin: bool, root: Path | str = CACHE_MOUNT) -> dict:
    """The vLLM compile-cache entries under this engine's ``cache_env`` root: hash dirs, and
    which hold a finished ``vllm_compile_cache.py`` (an engine with that config hash loads
    instead of compiling). Taken before each child, so the result says warm or cold."""
    base = Path(root) / "balanced" / f"{kind}{'-dtbias' if plugin else ''}" / "vllm" \
        / "torch_compile_cache"
    hashes = sorted(p.name for p in base.iterdir() if p.is_dir()
                    and p.name != "torch_aot_compile") if base.is_dir() else []
    done = [h for h in hashes if any((base / h).glob("rank_*/*/vllm_compile_cache.py"))]
    return {"hashes": hashes, "complete": done, "warm": bool(done)}


def check_segments_from(segments: dict | None, segments_from: dict | None) -> str | None:
    """Why inline ``segments`` cannot be captioned (None: they can). Inline segments are a
    deployed job's records; the caller must say which job and segment stage hash they came from
    (``{"job_id", "segment_hash"}``, checked against the records by ``scripts/wave_h.py``), so a
    result is never captioned from another job's segments without saying so (review 12 F5)."""
    if not segments:
        return None
    sf = segments_from or {}
    if not (isinstance(sf, dict) and sf.get("job_id") and sf.get("segment_hash")):
        return ("inline segments without their provenance (segments_from: job_id and "
                "segment_hash); refusing to caption segments of an unknown job")
    return None


ARMS = ("both", "baseline", "plugin")
GATE_REQUESTS, GATE_BASELINE = "requests.pkl", "baseline.json"


def gate_dir_rel(run_id: str, kind: str) -> str:
    """Where a ``baseline`` call stages the Gate A requests and baseline logits (relative to
    the ``flora-golden`` mount) for the ``plugin`` call."""
    return f"balanced-lab/{run_id}/{kind}-gate"


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_gate(kind: str, run_id: str, requests: Path, baseline: dict,
               root: Path | str = GOLDEN_MOUNT) -> dict:
    """A ``baseline`` call's Gate A inputs for the ``plugin`` call: the pickled requests and
    the baseline logits, under :func:`gate_dir_rel`. Returns ``gate_rel`` and the requests'
    ``gate_sha256`` (the plugin call reports the digest it read)."""
    import shutil

    rel = gate_dir_rel(run_id, kind)
    d = Path(root) / rel
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(requests, d / GATE_REQUESTS)
    sha = _sha256_file(d / GATE_REQUESTS)
    (d / GATE_BASELINE).write_text(json.dumps({"gate_a": baseline.get("gate_a"),
                                               "engine_hash": baseline.get("engine_hash"),
                                               "gate_sha256": sha}, default=str))
    return {"gate_rel": rel, "gate_sha256": sha}


def load_gate(gate_rel: str, requests_out: Path, root: Path | str = GOLDEN_MOUNT) -> dict:
    """The ``plugin`` call's side of :func:`stage_gate`: copies the requests to
    ``requests_out`` and returns the staged baseline record (with the requests' digest as
    read, ``gate_sha256_read``). Raises ``OSError`` / ``ValueError`` when they are missing or
    unreadable."""
    import shutil

    d = Path(root) / gate_rel
    staged = json.loads((d / GATE_BASELINE).read_text())
    shutil.copyfile(d / GATE_REQUESTS, requests_out)
    staged["gate_sha256_read"] = _sha256_file(requests_out)
    return staged


def run_kind(kind: str, run_id: str, segments_rel: str = "",
             episodes: list[str] | None = None, gate_a: bool = True,
             segments: dict | None = None, segments_from: dict | None = None,
             arm: str = "both", gate_rel: str = "") -> dict:
    """One lab call. ``segments`` (golden id -> NVIDIA segment rows, e.g. the deployed job's
    records) takes precedence over ``segments_rel`` (a file on ``flora-golden``); inline
    segments need ``segments_from`` (:func:`check_segments_from`), echoed in the result.
    ``arm`` (module docstring, "Split calls"): ``both`` engines in this call, the ``baseline``
    alone (its Gate A inputs staged for the plugin call) or the ``plugin`` alone on the Gate A
    inputs a baseline call staged at ``gate_rel``."""
    if arm not in ARMS:
        return {"run": run_id, "kind": kind, "arm": arm, "no_evidence": True,
                "error": f"unknown arm {arm!r} (one of {', '.join(ARMS)})"}
    if arm == "plugin" and not gate_rel:
        return {"run": run_id, "kind": kind, "arm": arm, "no_evidence": True,
                "error": "plugin arm: no gate_rel (the baseline call's staged Gate A inputs)"}
    if arm != "plugin" and (why := check_segments_from(segments, segments_from)):
        return {"run": run_id, "kind": kind, "error": why, "no_evidence": True,
                "segments_from": segments_from}
    from flora_deploy.engines.patches import dt_bias_fp32
    from flora_deploy.modal_app import warm

    t0 = time.time()
    prefetch = warm.start_prefetch(str(MODEL_DIRS[kind]), threads=16)
    tmp = Path("/tmp/balanced-lab")
    tmp.mkdir(parents=True, exist_ok=True)
    gate_pkl = tmp / "gate.pkl"
    if arm == "plugin":
        return _run_plugin_arm(kind, run_id, gate_rel, gate_pkl, tmp, t0, prefetch, episodes)
    segments = dict(segments or {})
    if segments_rel and not segments:
        golden.reload()
        segments = json.loads((Path(GOLDEN_MOUNT) / segments_rel).read_text())
    base = {"kind": kind, "gpu": GPUS[kind], "episodes": episodes, "segments": segments,
            "gate_requests": str(gate_pkl)}
    for e in cache_env(kind, False).values():
        Path(e).mkdir(parents=True, exist_ok=True)
    caches = {"baseline": compile_cache_state(kind, False)}
    a = _run_child({**base, "plugin": False, "out": str(tmp / "a.json")},
                   cache_env(kind, False))
    cache.commit()  # keep the baseline compile even if the plugin child hits a deadline
    result: dict[str, Any] = {"run": run_id, "kind": kind, "baseline": a,
                              "segments_from": dict(segments_from) if segments else None,
                              # the engine-settings hash of each engine built (S3-1)
                              "engine_hashes": {"baseline": a.get("engine_hash")},
                              "compile_cache": caches,
                              "prefetch": (s.as_json() if (s := prefetch.wait(0)) else None)}
    if arm != "both":
        result["arm"] = arm
    if "error" in a:
        result["error"] = f"baseline: {a['error']}"
        result["no_evidence"] = True
    if arm == "baseline" and gate_a and "error" not in a:
        try:  # the plugin call's inputs, staged before this container goes
            result.update(stage_gate(kind, run_id, gate_pkl, a))
            golden.commit()
        except (OSError, ValueError) as e:
            result["error"] = f"baseline: staging the Gate A inputs: {e!r}"[:2000]
    if arm == "both" and gate_a and "error" not in a:
        env = {**cache_env(kind, True), **dt_bias_fp32.install_entry_point(tmp / "plugin")}
        for e in cache_env(kind, True).values():
            Path(e).mkdir(parents=True, exist_ok=True)
        caches["plugin"] = compile_cache_state(kind, True)
        b = _run_child({**base, "plugin": True, "gate_only": True,
                        "out": str(tmp / "b.json")}, env)
        cache.commit()
        result["dtbias"] = {k: v for k, v in b.items() if k != "gate_a"}
        result["engine_hashes"]["plugin"] = b.get("engine_hash")
        if "error" in b:
            result["error"] = f"plugin: {b['error']}"
        else:
            result["gate_a"] = gate_a_compare(a["gate_a"], b["gate_a"])
    ok_rows = {e["id"]: e["rows"] for e in a.get("episodes") or [] if e["ok"]}
    if kind == "tempseg" and ok_rows:  # never hand an empty segments file on
        rel = f"balanced-lab/{run_id}/segments.json"
        p = Path(GOLDEN_MOUNT) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ok_rows))
        golden.commit()
        result["segments_rel"] = rel
    result["baseline"].pop("gate_a", None)
    result["container_s"] = round(time.time() - t0, 1)
    result["usd_estimate"] = round(usd_per_h(kind) * result["container_s"] / 3600, 3)
    return result


def _run_plugin_arm(kind: str, run_id: str, gate_rel: str, gate_pkl: Path, tmp: Path,
                    t0: float, prefetch: Any, episodes: list[str] | None) -> dict:
    """The ``plugin`` arm of :func:`run_kind`: the ``dt_bias`` plugin engine's Gate A on the
    requests a ``baseline`` call staged at ``gate_rel``, compared with its staged logits."""
    from flora_deploy.engines.patches import dt_bias_fp32

    result: dict[str, Any] = {"run": run_id, "kind": kind, "arm": "plugin",
                              "gate_rel": gate_rel, "engine_hashes": {}, "compile_cache": {}}
    try:
        golden.reload()
        staged: dict | None = load_gate(gate_rel, gate_pkl)
    except (OSError, ValueError) as e:
        staged = None
        result.update(error=f"plugin: the baseline call's Gate A inputs: {e!r}"[:2000],
                      no_evidence=True)
    if staged is not None:
        result["gate_sha256"] = staged.get("gate_sha256_read")
        result["baseline_engine_hash"] = staged.get("engine_hash")
        env = {**cache_env(kind, True), **dt_bias_fp32.install_entry_point(tmp / "plugin")}
        for e in cache_env(kind, True).values():
            Path(e).mkdir(parents=True, exist_ok=True)
        result["compile_cache"]["plugin"] = compile_cache_state(kind, True)
        b = _run_child({"kind": kind, "gpu": GPUS[kind], "episodes": episodes, "segments": {},
                        "gate_requests": str(gate_pkl), "plugin": True, "gate_only": True,
                        "out": str(tmp / "b.json")}, env)
        cache.commit()
        result["dtbias"] = {k: v for k, v in b.items() if k != "gate_a"}
        result["engine_hashes"]["plugin"] = b.get("engine_hash")
        if "error" in b:
            result["error"] = f"plugin: {b['error']}"
        elif not staged.get("gate_a"):
            result["error"] = "plugin: the staged baseline has no Gate A logits"
        else:
            result["gate_a"] = gate_a_compare(staged["gate_a"], b["gate_a"])
    result["prefetch"] = s.as_json() if (s := prefetch.wait(0)) else None
    result["container_s"] = round(time.time() - t0, 1)
    result["usd_estimate"] = round(usd_per_h(kind) * result["container_s"] / 3600, 3)
    return result


_GPU_VOLUMES = {str(WEIGHTS_MOUNT): weights, str(GOLDEN_MOUNT): golden, str(CACHE_MOUNT): cache}


@app.function(image=vllm_image(), gpu=GPUS["tempseg"], cpu=CPU,
              memory=MEM_GIB["tempseg"] * 1024, timeout=TIMEOUT_S, volumes=_GPU_VOLUMES, env=ENV)
def run_tempseg(run_id: str, segments_rel: str = "", episodes: list[str] | None = None,
                gate_a: bool = True, segments: dict | None = None,
                segments_from: dict | None = None, arm: str = "both",
                gate_rel: str = "") -> dict:
    return run_kind("tempseg", run_id, segments_rel, episodes, gate_a, segments, segments_from,
                    arm, gate_rel)


@app.function(image=vllm_image(), gpu=GPUS["captioner"], cpu=CPU,
              memory=MEM_GIB["captioner"] * 1024, timeout=TIMEOUT_S, volumes=_GPU_VOLUMES,
              env=ENV)
def run_captioner(run_id: str, segments_rel: str = "", episodes: list[str] | None = None,
                  gate_a: bool = True, segments: dict | None = None,
                  segments_from: dict | None = None, arm: str = "both",
                  gate_rel: str = "") -> dict:
    return run_kind("captioner", run_id, segments_rel, episodes, gate_a, segments,
                    segments_from, arm, gate_rel)


@app.local_entrypoint()
def main(kind: str = "tempseg", run: str = "r1", gate_a: bool = True) -> None:
    fn = run_tempseg if kind == "tempseg" else run_captioner
    seg = f"balanced-lab/{run}/segments.json" if kind == "captioner" else ""
    result = fn.remote(run, seg, None, gate_a)
    out = Path("reports/balanced-lab") / f"{run}-{kind}.json"  # git-ignored reports dir
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    summary = {k: result.get(k) for k in ("gate_a", "container_s", "usd_estimate")}
    base = result.get("baseline", {})
    summary.update(throughput=base.get("throughput"), l0=base.get("l0"),
                   error=result.get("error") or base.get("error"))
    print(json.dumps(summary, indent=1, default=str))
