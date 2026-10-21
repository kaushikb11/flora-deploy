"""``flora-coldstart-lab`` (W3-B): measure cold-start costs; results in ``docs/cold-start.md``.

Ephemeral only (``modal run``), never deployed. Three experiments:

- ``throughput`` (CPU, ``img_cpu``): read the Rewarder (52 GiB) and Captioner FP32 (102 GiB)
  snapshots from ``flora-weights`` with 1/4/8/16 readers (``warm.read_files``), then re-read a
  subset in the same container (page cache?). Run on several containers, sequentially and
  concurrently, for variance and cross-container caching.
- ``manifest_nocopy`` (CPU, ``img_hfref``, network blocked): the no-copy model cache
  (``warm.build_linked_layout``) with NVIDIA's own ``models.snapshot`` / ``validate_manifest``,
  timed, optionally with a parallel prefetch running ahead of NVIDIA's sequential hasher, then
  a second ``validate_manifest`` (NVIDIA's per-process ``_VERIFIED_MANIFESTS`` reuse) and a
  safetensors read pass standing in for ``from_pretrained``.
- ``rewarder_engine`` (``H100!``, ``img_vllm``): NVIDIA's ``engine.load_llm`` with NVIDIA's
  arguments (``stages.reward.engine_spec``), one fresh **subprocess per variant** so every
  engine starts from an empty process, with ``VLLM_CACHE_ROOT`` on ``flora-cache``. Variants
  may add a background prefetch or vLLM 0.24.0 I/O-only load options
  (``safetensors_load_strategy``, ``model_loader_extra_config``); sampling, dtype and backend
  arguments are never touched. vLLM's own phase timings are parsed from its log.

CLI: ``uv run modal run src/flora_deploy/modal_app/coldstart_lab.py::<entrypoint>``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import (
    CACHE_MOUNT,
    CACHE_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    snapshot_path,
)
from flora_deploy.modal_app import warm
from flora_deploy.modal_app.images import cpu_image, hfref_image, vllm_image

APP_NAME = "flora-coldstart-lab"
app = modal.App(APP_NAME)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)

REWARDER = snapshot_path(PINS_BY_KEY["rewarder"])
CAPTIONER = snapshot_path(PINS_BY_KEY["captioner"])
LAB_CACHE = CACHE_MOUNT / "coldstart-lab"
GiB = 1 << 30
MiB = 1 << 20

H100_TIMEOUT_S = 20 * 60
H100_LIFETIME_S = 19 * 60  # watchdog, below the Modal timeout
CHILD_TIMEOUT_S = 14 * 60


def _watchdog(limit_s: float) -> None:
    def kill():
        print(f"coldstart-lab watchdog: {limit_s:.0f}s reached, exiting", flush=True)
        os._exit(17)

    t = threading.Timer(limit_s, kill)
    t.daemon = True
    t.start()


def _env_facts(mount: str) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "task_id": os.environ.get("MODAL_TASK_ID"),
        "region": os.environ.get("MODAL_REGION"),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "cpus": os.cpu_count(),
    }
    try:
        facts["mounts"] = [ln for ln in Path("/proc/mounts").read_text().splitlines()
                           if f" {mount}" in ln]
        mem = dict(ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())
        facts["mem_total"] = mem.get("MemTotal", "").strip()
        facts["mem_available"] = mem.get("MemAvailable", "").strip()
        facts["cached"] = mem.get("Cached", "").strip()
    except OSError as e:
        facts["error"] = repr(e)
    try:
        facts["kernel"] = os.uname().release
    except Exception:  # noqa: BLE001
        pass
    return facts


# ---------------------------------------------------------------------------
# 1. Volume read throughput (CPU)
# ---------------------------------------------------------------------------


def _step(label, snap, shards, threads, chunk_mib=256):
    return {"label": label, "snapshot": str(snap), "shards": list(shards), "threads": threads,
            "chunk_mib": chunk_mib}


# Disjoint shard sets per first read so every first read is "cold" for this container.
DEFAULT_PLAN: tuple[dict, ...] = (
    _step("rew 1-stream seq", REWARDER, range(0, 2), 1, None),
    _step("rew 4 readers", REWARDER, range(2, 4), 4),
    _step("rew 8 readers", REWARDER, range(4, 6), 8),
    _step("rew 16 readers", REWARDER, range(6, 8), 16),
    _step("rew 16 readers 64MiB", REWARDER, range(8, 11), 16, 64),
    _step("rew again 1-stream (same container)", REWARDER, range(0, 2), 1, None),
    _step("rew again 16 readers (same container)", REWARDER, range(2, 4), 16),
    _step("cap 1-stream seq", CAPTIONER, range(0, 1), 1, None),
    _step("cap 16 readers (rest)", CAPTIONER, range(1, 11), 16),
)
# Same shard sets with the reader counts reversed: separates "which shards" from "how many".
REVERSED_PLAN: tuple[dict, ...] = (
    _step("rew 16 readers", REWARDER, range(0, 2), 16),
    _step("rew 8 readers", REWARDER, range(2, 4), 8),
    _step("rew 4 readers", REWARDER, range(4, 6), 4),
    _step("rew 1-stream seq", REWARDER, range(6, 8), 1, None),
    _step("rew 32 readers 64MiB", REWARDER, range(8, 11), 32, 64),
    _step("cap 16 readers", CAPTIONER, range(0, 10), 16),
    _step("cap 1-stream seq", CAPTIONER, range(10, 11), 1, None),
)


@app.function(image=cpu_image(), volumes={str(WEIGHTS_MOUNT): weights.read_only()}, cpu=8.0,
              memory=32 * 1024, timeout=20 * 60,
              single_use_containers=True)  # a reused container would serve reads from its cache
def throughput(plan: list[dict], name: str = "") -> dict:
    t0 = time.monotonic()
    out = {"name": name, "env": _env_facts(str(WEIGHTS_MOUNT)), "steps": []}
    for step in plan:
        files = warm.snapshot_files(step["snapshot"])
        files = [files[i] for i in step["shards"] if i < len(files)]
        chunk = None if step["chunk_mib"] is None else step["chunk_mib"] * MiB
        stats = warm.read_files(files, threads=step["threads"], chunk_bytes=chunk)
        row = {**{k: step[k] for k in ("label", "shards", "threads")},
               "snapshot": PurePosixPath(step["snapshot"]).name.split("@")[0], **stats.as_json()}
        print(json.dumps(row), flush=True)
        out["steps"].append(row)
    out["env_after"] = _env_facts(str(WEIGHTS_MOUNT))
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


@app.local_entrypoint()
def run_throughput(out: str = "reports/w3b/throughput.json") -> None:
    """Fresh containers: A, B (default plan) and C (reversed) concurrently, then D (default)
    alone."""
    calls = [throughput.spawn(list(DEFAULT_PLAN), "A"), throughput.spawn(list(DEFAULT_PLAN), "B"),
             throughput.spawn(list(REVERSED_PLAN), "C")]
    results = [c.get() for c in calls]
    results.append(throughput.remote(list(DEFAULT_PLAN), "D"))
    _save(out, results)


@app.function(image=cpu_image(), volumes={str(WEIGHTS_MOUNT): weights.read_only()},
              cpu=1.0, memory=1024, timeout=5 * 60)
def mount_probe() -> dict:
    """Where a Volume mount really lives (symlink?) and which ancestors are writable."""
    out: dict[str, Any] = {}
    for p in ("/", str(WEIGHTS_MOUNT)):
        out[p] = {"islink": os.path.islink(p), "real": os.path.realpath(p)}
    parts = os.path.realpath(str(WEIGHTS_MOUNT)).split("/")
    for i in range(1, len(parts) + 1):
        d = "/".join(parts[:i]) or "/"
        probe = os.path.join(d, ".flora_probe")
        try:
            os.mkdir(probe)
            os.rmdir(probe)
            out[f"writable {d}"] = True
        except OSError as e:
            out[f"writable {d}"] = repr(e)
    out["mounts"] = [ln for ln in Path("/proc/mounts").read_text().splitlines()
                     if "flora" in ln or "modal" in ln or "volume" in ln][:20]
    return out


@app.local_entrypoint()
def run_mount_probe() -> None:
    print(json.dumps(mount_probe.remote(), indent=1))


# ---------------------------------------------------------------------------
# 2. No-copy reference model cache + NVIDIA's manifest validation (CPU)
# ---------------------------------------------------------------------------


def _st_read_pass(snapshot: Path) -> dict:
    """Materialise every tensor with safetensors (mmap, like ``from_pretrained``), discarding
    each one: a CPU stand-in for the weight read of the Captioner's GPU load."""
    import torch
    from safetensors import safe_open

    t0 = time.monotonic()
    total = 0
    for f in warm.snapshot_files(snapshot):
        with safe_open(str(f), framework="pt") as st:
            for key in st.keys():  # noqa: SIM118
                t = st.get_tensor(key)
                flat = t.reshape(-1).view(torch.uint8) if t.numel() else None
                if flat is not None:
                    flat[::4096].sum()  # touch every 4 KiB page even if the tensor is a lazy map
                total += t.numel() * t.element_size()
                del t, flat
    s = time.monotonic() - t0
    return {"bytes": total, "seconds": round(s, 1), "gbps": round(total / s / 1e9, 3)}


@app.function(
    image=hfref_image(),
    volumes={str(WEIGHTS_MOUNT): weights.read_only()},
    env=warm.linked_offline_env(),
    cpu=8.0,
    memory=96 * 1024,  # the ReferenceWorker's memory, so page-cache behaviour matches
    timeout=25 * 60,
    block_network=True,
    single_use_containers=True,
)
def manifest_nocopy(prefetch_threads: int = 0, read_pass: bool = True) -> dict:
    t0 = time.monotonic()
    import huggingface_hub.constants as hfc

    from flora_deploy.nv import Family, load_family
    from flora_deploy.stages.reference import (
        capture_runtime_events,
        check_nv_models,
    )

    assert hfc.HF_HUB_OFFLINE
    out: dict[str, Any] = {"env": _env_facts(str(WEIGHTS_MOUNT)),
                           "prefetch_threads": prefetch_threads}
    root, mount = Path(warm.LINKED_ROOT), Path(WEIGHTS_MOUNT)
    t = time.monotonic()
    lay = warm.build_linked_layout(root, mount)
    out["layout"] = {"seconds": round(time.monotonic() - t, 2), "stats": lay.stats}
    code_dir = Path(snapshot_path(PINS_BY_KEY["tempseg_code"], PurePosixPath(mount)))
    nv = load_family(Family.SEGCAP, code_dir)
    check_nv_models(nv.config.MODELS)
    assert Path(nv.config.MODEL_CACHE) == lay.model_cache, nv.config.MODEL_CACHE
    models = nv.models
    steps = []

    def timed(label, fn):
        t = time.monotonic()
        with capture_runtime_events() as events:
            value = fn()
        row = {"label": label, "seconds": round(time.monotonic() - t, 2),
               "events": [{k: v for k, v in e.items() if k != "utc"} for e in events]}
        print(json.dumps(row), flush=True)
        steps.append(row)
        return value

    ts = timed("snapshot(tempseg)", lambda: models.snapshot("tempseg"))
    assert Path(ts) == lay.paths["tempseg"].resolve(), ts
    timed("validate_manifest(tempseg)", lambda: models.validate_manifest(ts))
    from huggingface_hub import snapshot_download

    spec = nv.config.MODELS["tempseg_base"]
    base = timed("tempseg_base snapshot_download (models.py:130)", lambda: snapshot_download(
        spec["repo"], revision=spec["revision"], token=None,
        allow_patterns=["*.safetensors", "*.json", "*.jinja"]))
    out["tempseg_base"] = base
    cap = timed("snapshot(captioner)", lambda: models.snapshot("captioner"))
    assert Path(cap) == lay.paths["captioner"].resolve(), cap
    job = None
    if prefetch_threads:
        job = warm.start_prefetch(cap, threads=prefetch_threads)
    timed("validate_manifest(captioner) #1", lambda: models.validate_manifest(cap))
    if job is not None:
        stats = job.wait()
        out["prefetch"] = stats.as_json() if stats else None
    timed("validate_manifest(captioner) #2", lambda: models.validate_manifest(cap))
    timed("snapshot(captioner) + validate #3",
          lambda: models.validate_manifest(models.snapshot("captioner")))
    if read_pass:
        out["st_read_pass"] = _st_read_pass(Path(cap))
        print(json.dumps(out["st_read_pass"]), flush=True)
    out["verified_manifest_keys"] = sorted(models._VERIFIED_MANIFESTS)
    out["steps"] = steps
    out["env_after"] = _env_facts(str(WEIGHTS_MOUNT))
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


@app.local_entrypoint()
def run_manifest(out: str = "reports/w3b/manifest.json", prefetch_threads: int = 16,
                 both: bool = True) -> None:
    results = []
    if both:
        results.append(manifest_nocopy.remote(0))
    results.append(manifest_nocopy.remote(prefetch_threads))
    _save(out, results)


# ---------------------------------------------------------------------------
# 3. Rewarder vLLM engine startup on H100 (one subprocess per variant)
# ---------------------------------------------------------------------------

TIMING_PATTERNS: dict[str, str] = {
    "fs_type": r"Filesystem type for checkpoints: (\S+)\.",
    "loading_weights_s": r"Loading weights took ([\d.]+) seconds",
    "model_loading_s": r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds",
    "dynamo_transform_s": r"Dynamo bytecode transform time: ([\d.]+) s",
    "compile_graph_s": r"Compiling a graph for compile range .* takes ([\d.]+) s",
    "compile_cache_load_s": r"from the cache, took ([\d.]+) s",
    "torch_compile_total_s": r"torch\.compile took ([\d.]+) s in total",
    "compile_and_warmup_s": r"run together took ([\d.]+) s in total",
    "profiling_warmup_s": r"Initial profiling/warmup run took ([\d.]+) s",
    "graph_capture_s": r"Graph capturing finished in ([\d.]+) secs",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+)",
    "vllm_prefetch_s": r"Prefetching checkpoint files into page cache finished in ([\d.]+)s",
    "cache_dir": r"Using cache directory: (\S+) for vLLM's torch\.compile",
}
PROMPT = "List the steps to make a cup of coffee, one per line:"


def parse_vllm_timings(lines: list[str]) -> dict[str, Any]:
    """vLLM's own phase timings from its log (a phase may log several times; keep all)."""
    out: dict[str, Any] = {}
    for key, pat in TIMING_PATTERNS.items():
        rx = re.compile(pat)
        hits = [m.group(1) for line in lines for m in [rx.search(line)] if m]
        if not hits:
            continue
        if key in ("fs_type", "cache_dir"):
            out[key] = hits[0]
        else:
            vals = [float(h) for h in hits]
            out[key] = vals[0] if len(vals) == 1 else {"values": vals, "sum": round(sum(vals), 2)}
    return out


def _child(cfg: dict) -> dict:
    """Runs in a fresh interpreter: NVIDIA's load_llm with NVIDIA's arguments (+ I/O-only
    load options), one greedy text generation, timings."""
    t_start = time.perf_counter()
    import argparse

    from flora_deploy.nv import Family, load_family
    from flora_deploy.stages import reward

    nv = load_family(Family.REWARDER, str(REWARDER))
    spec = reward.engine_spec(nv, str(REWARDER))
    extra = cfg.get("extra") or {}
    forbidden = set(extra) & {"dtype", "gdn_prefill_backend", "enable_prefix_caching", "seed",
                              "max_model_len", "max_num_seqs", "gpu_memory_utilization",
                              "mm_processor_kwargs", "max_logprobs", "limit_mm_per_prompt"}
    if forbidden:
        raise ValueError(f"lab variants may only change I/O options, not {sorted(forbidden)}")
    t_link = time.perf_counter()
    import vllm  # noqa: F401  (import cost measured separately)

    t_import = time.perf_counter()
    llm = nv.engine.load_llm(spec.model_path, argparse.Namespace(**spec.args), **spec.kwargs,
                             **extra)
    t_engine = time.perf_counter()
    from vllm import SamplingParams

    res = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=32, seed=0),
                       use_tqdm=False)
    t_gen = time.perf_counter()
    return {
        "link_s": round(t_link - t_start, 2),
        "import_vllm_s": round(t_import - t_link, 2),
        "load_llm_s": round(t_engine - t_import, 2),
        "first_generate_s": round(t_gen - t_engine, 2),
        "child_total_s": round(t_gen - t_start, 2),
        "token_ids": list(res[0].outputs[0].token_ids),
        "text": res[0].outputs[0].text,
        "spec": spec.as_json(),
        "extra": extra,
    }


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, check=True).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001
        return None


def _run_variant(v: dict, cache_root: str) -> dict:
    env = {**os.environ, "VLLM_CACHE_ROOT": cache_root, "PYTHONUNBUFFERED": "1"}
    cfg = {"extra": v.get("extra") or {}}
    out_path = Path(f"/tmp/coldstart-{v['name']}.json")
    job = None
    t0 = time.perf_counter()
    if v.get("prefetch_threads"):
        job = warm.start_prefetch(str(REWARDER), threads=v["prefetch_threads"])
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "flora_deploy.modal_app.coldstart_lab", "child",
         json.dumps(cfg), str(out_path)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines: list[str] = []
    stamped: list[tuple[float, str]] = []

    def pump():
        for line in proc.stdout:  # type: ignore[union-attr]
            lines.append(line.rstrip())
            stamped.append((time.perf_counter() - t0, line.rstrip()))
            print(f"[{v['name']} {time.perf_counter() - t0:7.1f}s] {line.rstrip()}", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = "timeout"
    reader.join(10)
    wall = time.perf_counter() - t0
    row: dict[str, Any] = {"name": v["name"], "exit": code, "wall_s": round(wall, 1),
                           "cache_root": cache_root, "vllm": parse_vllm_timings(lines)}
    if job is not None:
        stats = job.wait(timeout=600)
        row["prefetch"] = stats.as_json() if stats else None
    if out_path.is_file():
        row["child"] = json.loads(out_path.read_text())
    else:
        row["log_tail"] = lines[-60:]
    # Wall-clock offsets of milestone lines (when each phase ended, from process start).
    marks = {}
    for key in ("loading_weights_s", "graph_capture_s", "init_engine_s", "torch_compile_total_s"):
        rx = re.compile(TIMING_PATTERNS[key])
        for ts, line in stamped:
            if rx.search(line):
                marks[key.removesuffix("_s") + "_at_s"] = round(ts, 1)
                break
    row["marks"] = marks
    # Wait for the GPU to be released before the next variant.
    for _ in range(60):
        used = _gpu_used_mib()
        if used is None or used < 2000:
            break
        time.sleep(2)
    row["gpu_used_after_mib"] = _gpu_used_mib()
    return row


@app.function(
    image=vllm_image(),
    gpu="H100!",
    cpu=8.0,
    memory=80 * 1024,  # room for 52 GiB of page cache next to the engine process
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(CACHE_MOUNT): cache},
    timeout=H100_TIMEOUT_S,
    max_containers=1,
)
def rewarder_engine(variants: list[dict], tag: str, name: str = "") -> dict:
    _watchdog(H100_LIFETIME_S)
    t0 = time.monotonic()
    cache_root = str(LAB_CACHE / tag)
    out: dict[str, Any] = {"name": name, "env": _env_facts(str(WEIGHTS_MOUNT)),
                           "cache_root": cache_root,
                           "cache_existed": Path(cache_root).is_dir(), "variants": []}
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    for v in variants:
        if time.monotonic() - t0 > H100_LIFETIME_S - 6 * 60:
            out["variants"].append({"name": v["name"], "skipped": "not enough time left"})
            continue
        row = _run_variant(v, cache_root)
        print(json.dumps({k: row[k] for k in ("name", "exit", "wall_s", "vllm", "marks")}),
              flush=True)
        out["variants"].append(row)
        cache.commit()
    out["env_after"] = _env_facts(str(WEIGHTS_MOUNT))
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


@app.local_entrypoint()
def run_engine(container: int = 1, tag: str = "w3b-1", out: str = "") -> None:
    """``--container 1``: cold compile cache, then the same cache warm (fresh processes).
    ``--container 2``: fresh container on the warm cache, with a parallel prefetch at t=0."""
    plans = {
        1: [{"name": "c1-cold-cache"}, {"name": "c1-warm-cache"}],
        2: [{"name": "c2-warm-cache+prefetch16", "prefetch_threads": 16},
            {"name": "c2-warm-cache+vllm-prefetch",
             "extra": {"safetensors_load_strategy": "prefetch"}}],
    }
    result = rewarder_engine.remote(plans[container], tag, f"container-{container}")
    _save(out or f"reports/w3b/engine-{container}.json", result)


# ---------------------------------------------------------------------------


def _save(path: str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "child":
    result = _child(json.loads(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result))
    print("coldstart-lab child done", flush=True)
