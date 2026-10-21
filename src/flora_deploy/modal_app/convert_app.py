"""``flora-convert``: build the BF16 Captioner and prove it loads bit-identically (OPT-2).

- ``convert`` (CPU, ``img_hfref``, network blocked): streams the FP32 Captioner snapshot from
  ``flora-weights`` and writes ``derived/captioner-bf16@<rev>`` plus its receipt
  (``prep/convert_captioner.py``). Writes nothing outside that directory.
- ``identity`` (H200, weights read-only, network blocked): loads the Captioner twice through
  NVIDIA's own ``Session.load`` (A: unmodified, from the FP32 snapshot copied to local disk as the
  reference worker does; B: with ``accel.captioner_load`` installed, from the BF16 copy on the
  Volume), compares every parameter and buffer bit for bit plus the logits of one fixed forward
  pass, and reports both load times.

Run (ephemeral; never deployed)::

    uv run modal run src/flora_deploy/modal_app/convert_app.py --step convert
    uv run modal run src/flora_deploy/modal_app/convert_app.py --step identity
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import modal

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import WEIGHTS_MOUNT, WEIGHTS_VOLUME, derived_path, snapshot_path
from flora_deploy.modal_app.images import hfref_image
from flora_deploy.stages.reference import OFFLINE_ROOT, offline_env

app = modal.App("flora-convert")
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)

CONVERT_CPU = 8.0
CONVERT_MEMORY_MIB = 56 * 1024  # 3 workers x one source shard (<= 10.7 GB) + casts
CONVERT_WORKERS = 3
CONVERT_THREADS = 4  # torch intra-op threads; 3 x 4 on 8 physical cores (16 vCPU)
CONVERT_TIMEOUT_S = 45 * 60
IDENTITY_TIMEOUT_S = 28 * 60
IDENTITY_MEMORY_MIB = 64 * 1024


def _watchdog(limit_s: float) -> None:
    def kill():
        print(f"flora-convert watchdog: {limit_s:.0f}s reached, exiting", flush=True)
        os._exit(17)

    t = threading.Timer(limit_s, kill)
    t.daemon = True
    t.start()


def _code_dir() -> Path:
    return Path(snapshot_path(PINS_BY_KEY["tempseg_code"], WEIGHTS_MOUNT))


def _source() -> Path:
    return Path(snapshot_path(PINS_BY_KEY["captioner"], WEIGHTS_MOUNT))


def _derived() -> Path:
    return Path(derived_path("captioner-bf16", PINS_BY_KEY["captioner"], WEIGHTS_MOUNT))


@app.function(
    image=hfref_image(),
    cpu=CONVERT_CPU,
    memory=CONVERT_MEMORY_MIB,
    volumes={str(WEIGHTS_MOUNT): weights},
    env=offline_env(OFFLINE_ROOT),
    timeout=CONVERT_TIMEOUT_S,
    block_network=True,
)
def convert(force: bool = False, provenance: dict | None = None) -> dict:
    _watchdog(CONVERT_TIMEOUT_S + 60)
    from flora_deploy.nv import Family, load_family
    from flora_deploy.prep.convert_captioner import convert as run

    nv = load_family(Family.SEGCAP, _code_dir())
    keep = nv.vendor("production_gdn_fix")._keep_native_gdn_fp32
    receipt = run(_source(), _derived(), keep, workers=CONVERT_WORKERS,
                  threads_per_worker=CONVERT_THREADS, force=force,
                  provenance=provenance, commit=weights.commit)
    return receipt


def _gpu_memory() -> dict:
    import torch

    return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)}


def _events(events: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k not in ("utc", "values")} for e in events
            if e.get("event", "").startswith(("model_", "manifest_"))]


@app.function(
    image=hfref_image(),
    gpu="H200",
    cpu=8.0,
    memory=IDENTITY_MEMORY_MIB,
    volumes={str(WEIGHTS_MOUNT): weights.read_only()},
    env=offline_env(OFFLINE_ROOT),
    timeout=IDENTITY_TIMEOUT_S,
    block_network=True,
)
def identity() -> dict:
    _watchdog(IDENTITY_TIMEOUT_S + 30)
    t0 = time.monotonic()
    import torch

    from flora_deploy.accel.captioner_load import (
        compare_models,
        fixed_forward,
        install,
        logits_digest,
        tensors_equal_bits,
    )
    from flora_deploy.nv import Family, load_family
    from flora_deploy.stages.reference import (
        build_offline_layout,
        capture_runtime_events,
        check_nv_models,
    )

    out: dict = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    nv = load_family(Family.SEGCAP, _code_dir())
    check_nv_models(nv.config.MODELS)

    # A: NVIDIA's path exactly as the reference worker runs it (local copy + manifest hash).
    t = time.monotonic()
    layout = build_offline_layout(OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("captioner",))
    out["a_copy"] = {**layout.copies.get("captioner", {}),
                     "seconds": round(time.monotonic() - t, 1)}
    assert Path(nv.models.snapshot("captioner")).resolve() == layout.paths["captioner"].resolve()
    sa = nv.models.Session()
    t = time.monotonic()
    with capture_runtime_events() as ev:
        sa.load("captioner")
    out["a_load_s"] = round(time.monotonic() - t, 1)
    out["a_events"] = _events(ev)
    out["a_gpu"] = _gpu_memory()
    print(f"A loaded in {out['a_load_s']}s", flush=True)

    # B: the same Session.load with the BF16 redirection installed.
    t = time.monotonic()
    patch = install(nv, _derived())
    sb = nv.models.Session()
    with capture_runtime_events() as ev:
        sb.load("captioner")
    out["b_load_s"] = round(time.monotonic() - t, 1)
    patch.uninstall()
    out["b_events"] = _events(ev)
    out["b_verifications"] = patch.verifications
    out["b_gpu"] = _gpu_memory()
    print(f"B loaded in {out['b_load_s']}s", flush=True)

    t = time.monotonic()
    out["compare"] = compare_models(sa.model, sb.model)
    ga, gb = sa.audit["captioner"], sb.audit["captioner"]
    out["audit"] = {
        "gdn_values_equal": [v["fp32_value_sha256"] for v in ga["gdn"]["values"]]
        == [v["fp32_value_sha256"] for v in gb["gdn"]["values"]],
        "gdn_tensors": ga["gdn"]["tensors"],
        "gdn_status": [ga["gdn"]["status"], gb["gdn"]["status"]],
        "gdn_source_dtypes": [ga["gdn"]["source_dtypes"], gb["gdn"]["source_dtypes"]],
        "parameter_dtypes": [ga["parameter_dtypes"], gb["parameter_dtypes"]],
        "parameter_devices": [ga["parameter_devices"], gb["parameter_devices"]],
        "loading_info_equal": {k: sorted(map(str, ga["loading_info"].get(k) or []))
                               == sorted(map(str, gb["loading_info"].get(k) or []))
                               for k in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                         "error_msgs")},
        "unexpected_keys": len(ga["loading_info"].get("unexpected_keys") or []),
    }
    la = fixed_forward(sa.model)
    lb = fixed_forward(sb.model)
    la2 = fixed_forward(sa.model)
    out["logits"] = {
        "shape": list(la.shape), "dtype": str(la.dtype),
        "a_vs_b_bit_identical": tensors_equal_bits(la, lb),
        "a_repeat_bit_identical": tensors_equal_bits(la, la2),
        "a_sha256": logits_digest(la), "b_sha256": logits_digest(lb),
        "max_abs_diff": float((la.float() - lb.float()).abs().max()),
        "argmax_equal": bool(torch.equal(la.argmax(-1), lb.argmax(-1))),
    }
    out["compare_s"] = round(time.monotonic() - t, 1)
    out["identical"] = bool(out["compare"]["identical"] and out["logits"]["a_vs_b_bit_identical"]
                            and out["audit"]["gdn_values_equal"])
    out["total_s"] = round(time.monotonic() - t0, 1)
    return out


def _git_provenance() -> dict:
    import subprocess

    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


@app.local_entrypoint()
def main(step: str = "convert", force: bool = False, out: str = "") -> None:
    if step == "convert":
        result = convert.remote(force=force, provenance=_git_provenance())
    elif step == "identity":
        result = identity.remote()
    else:
        raise SystemExit(f"unknown step {step!r} (convert | identity)")
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


# --- P1b-0: FP32 TempSeg LoRA merge + equivalence check (docs/tempseg-merge.md) ----------------
#
#   uv run modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step merge [--force]
#   uv run modal run src/flora_deploy/modal_app/convert_app.py::tempseg --step equivalence

golden = modal.Volume.from_name("flora-golden", version=2)

MERGE_CPU = 8.0
MERGE_MEMORY_MIB = 32 * 1024  # 2 workers x one base shard (<= 5.4 GB) + adapter + FP32 temps
MERGE_WORKERS = 2
MERGE_THREADS = 4
MERGE_TIMEOUT_S = 30 * 60
EQUIV_GPU = "L40S"
EQUIV_MEMORY_MIB = 48 * 1024  # the base is built on CPU in BF16 (~19 GB) before .to("cuda")
EQUIV_TIMEOUT_S = 25 * 60
EQUIV_GOLDEN_MOUNT = "/golden"
# One real window from each: a human single-view episode and a 3-view YAM episode.
EQUIV_EPISODES = ("human/egodex_add_remove_lid/episode_000014",
                  "yam/aloha_static_cups_open/episode_000000")
FIXED_PROMPTS = (
    "Describe what a robot arm does when it picks up a cup.",
    "List three steps to fold a towel, as JSON.",
    "What is 17 times 23?",
    "Segment this task into subtasks: open the drawer, put the block inside, close the drawer.",
)


def _tempseg_flora() -> Path:
    return Path(snapshot_path(PINS_BY_KEY["tempseg"], WEIGHTS_MOUNT))


def _tempseg_base() -> Path:
    return Path(snapshot_path(PINS_BY_KEY["tempseg_base"], WEIGHTS_MOUNT))


def _tempseg_merged() -> Path:
    return Path(derived_path("tempseg-merged", PINS_BY_KEY["tempseg"], WEIGHTS_MOUNT))


@app.function(
    image=hfref_image(),
    cpu=MERGE_CPU,
    memory=MERGE_MEMORY_MIB,
    volumes={str(WEIGHTS_MOUNT): weights},
    env=offline_env(OFFLINE_ROOT),
    timeout=MERGE_TIMEOUT_S,
    block_network=True,
)
def merge_tempseg(force: bool = False, provenance: dict | None = None) -> dict:
    """CPU: merge the FP32 TempSeg LoRA into the base (``prep/merge_tempseg.py``). Writes only
    ``derived/tempseg-merged@<tempseg rev>``."""
    _watchdog(MERGE_TIMEOUT_S + 60)
    from flora_deploy.nv import Family, load_family
    from flora_deploy.prep.merge_tempseg import merge

    nv = load_family(Family.SEGCAP, _code_dir())
    keep = nv.vendor("production_gdn_fix")._keep_native_gdn_fp32
    return merge(_tempseg_flora(), _tempseg_base(), _tempseg_merged(), keep,
                 workers=MERGE_WORKERS, threads_per_worker=MERGE_THREADS, force=force,
                 provenance=provenance, commit=weights.commit)


class _Enough(BaseException):  # BaseException: escapes any `except Exception` on the way
    """Raised from inside NVIDIA's segmenter once the window we want has been generated."""


class _OneModel:
    """Duck-types NVIDIA ``Session`` for ``pipeline.segment``: returns an already-loaded model."""

    def __init__(self, model, proc):
        self.pair = (model, proc)

    def load(self, kind: str):
        assert kind == "tempseg", kind
        return self.pair

    def close(self) -> None:
        pass


def _golden_episodes() -> list[dict]:
    manifest = json.loads((Path(EQUIV_GOLDEN_MOUNT) / "golden/v1/manifest.json").read_text())
    by_id = {e["id"]: e for e in manifest["episodes"]}
    out = []
    for eid in EQUIV_EPISODES:
        e = by_id[eid]
        root = Path(EQUIV_GOLDEN_MOUNT) / e["path"]
        out.append({"id": eid, "profile": e["profile"],
                    "videos": {role: str(root / f["file"]) for role, f in e["files"].items()}})
    return out


def _inputs_digest(inputs: dict) -> str:
    import hashlib

    import torch

    h = hashlib.sha256()
    for k in sorted(inputs):
        t = inputs[k].contiguous()
        h.update(f"{k}:{t.dtype}:{list(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _capture_window(nv, model, proc, ep: dict) -> dict:
    """Run NVIDIA's ``pipeline.segment`` on ``ep`` and stop after its first window: returns
    that window's model inputs (built by NVIDIA's ``temporal_inputs``), the text of NVIDIA's
    ``generate`` and the generated token ids."""
    import torch

    v = ep["videos"]
    nv_videos, canonical = nv.api.DataEnginePipeline._inputs(
        v["main"], ep["profile"], v.get("left_wrist"), v.get("right_wrist"), False, "")
    pool = nv.config.PROFILE_POOLS[canonical]
    rec: dict = {"episode": ep["id"], "pool": pool, "views": len(nv_videos)}
    nv_generate = nv.pipeline.generate

    def generate(m, p, inputs, budget):
        rec["inputs"] = {k: t.detach().cpu().clone() for k, t in dict(inputs).items()
                         if torch.is_tensor(t)}
        rec["input_digest"] = _inputs_digest(rec["inputs"])
        holder: dict = {}
        missing = object()
        prev = m.__dict__.get("generate", missing)
        inner = m.generate

        def recording(*a, **k):
            holder["out"] = inner(*a, **k)
            return holder["out"]

        m.generate = recording
        t = time.monotonic()
        try:
            rec["text"] = nv_generate(m, p, inputs, budget)
        except Exception as exc:  # recorded: an incomplete output is a result, not a crash
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if prev is missing:
                del m.generate
            else:
                m.generate = prev
        n = rec["inputs"]["input_ids"].shape[1]
        rec["input_tokens"] = n
        rec["ids"] = holder["out"][0, n:].cpu().tolist() if "out" in holder else []
        rec["generate_s"] = round(time.monotonic() - t, 1)
        raise _Enough

    nv.pipeline.generate = generate
    try:
        nv.pipeline.segment(nv_videos, pool, _OneModel(model, proc), "", lambda *a, **k: None)
    except _Enough:
        pass
    finally:
        nv.pipeline.generate = nv_generate
    if "inputs" not in rec:
        raise RuntimeError(f"{ep['id']}: NVIDIA's segmenter made no model call")
    return rec


def _forward_logits(model, inputs: dict, autocast: bool, logits_to_keep: int = 0):
    import contextlib

    import torch

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())
    dev = {k: t.to("cuda:0") for k, t in inputs.items()}
    with torch.inference_mode(), ctx:
        out = model(**dev, use_cache=False, logits_to_keep=logits_to_keep)
    return out.logits[0].detach().float().cpu()


def _prompt_inputs(proc) -> list[dict]:
    out = []
    for p in FIXED_PROMPTS:
        text = proc.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        out.append(dict(proc.tokenizer(text, return_tensors="pt")))
    return out


def _json_equal(a: str | None, b: str | None) -> bool | None:
    try:
        return json.loads(a) == json.loads(b)
    except (TypeError, ValueError):
        return None


@app.function(
    image=hfref_image(),
    gpu=EQUIV_GPU,
    cpu=8.0,
    memory=EQUIV_MEMORY_MIB,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), EQUIV_GOLDEN_MOUNT: golden.read_only()},
    env=offline_env(OFFLINE_ROOT),
    timeout=EQUIV_TIMEOUT_S,
    block_network=True,
)
def tempseg_equivalence() -> dict:
    """GPU: NVIDIA's unmerged TempSeg (``Session.load('tempseg')``: FP32 LoRA, BF16 autocast)
    vs the merged checkpoint as a plain BF16 HF model (FP32 ``A_log``/``dt_bias``), loaded one
    after the other on one GPU. Window inputs are built by NVIDIA's own code."""
    _watchdog(EQUIV_TIMEOUT_S + 30)
    t0 = time.monotonic()
    import gc
    from collections import Counter

    import torch

    from flora_deploy.accel.captioner_load import fixed_forward
    from flora_deploy.nv import Family, load_family
    from flora_deploy.prep import merge_tempseg as mt
    from flora_deploy.stages.reference import (
        build_offline_layout,
        capture_runtime_events,
        check_nv_models,
    )

    out: dict = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                 "episodes": list(EQUIV_EPISODES)}
    nv = load_family(Family.SEGCAP, _code_dir())
    check_nv_models(nv.config.MODELS)
    merged = _tempseg_merged()
    out["verify_merged"] = mt.verify_merged(merged)
    spec = mt.LoraSpec.from_config(
        json.loads((_tempseg_flora() / mt.ADAPTER_CONFIG).read_text()))
    episodes = _golden_episodes()
    prefix = "base_model.model."

    # --- A: NVIDIA's unmerged path, exactly Session.load('tempseg') --------------------------
    t = time.monotonic()
    build_offline_layout(OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("tempseg",))
    sa = nv.models.Session()
    with capture_runtime_events():
        model_a, proc_a = sa.load("tempseg")
    out["a_load_s"] = round(time.monotonic() - t, 1)
    audit = sa.audit["tempseg"]
    out["a_audit"] = {"gdn_status": audit["GDN_VALUE_VERIFICATION.json"]["status"],
                      "adapter_values_exact": audit["ADAPTER_VALUE_VERIFICATION.json"][
                          "all_saved_values_exact_before_and_after_placement"]}
    prompts = _prompt_inputs(proc_a)
    a_fixed = [_forward_logits(model_a, p, autocast=True) for p in prompts]
    a_repeat = _forward_logits(model_a, prompts[0], autocast=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a_random = fixed_forward(model_a)[0].float().cpu()
    a_params = mt.tensor_digests(
        ((n.removeprefix(prefix), p) for n, p in model_a.named_parameters()),
        skip=lambda n: ".lora_" in n or ".base_layer." in n)
    a_windows = []
    for ep in episodes:
        w = _capture_window(nv, model_a, proc_a, ep)
        if w["ids"]:
            ext, keep = mt.teacher_forced_inputs(w["inputs"], w["ids"])
            w["tf_logits"] = _forward_logits(model_a, ext, autocast=True, logits_to_keep=keep)
        a_windows.append(w)
        print(f"A {ep['id']}: {len(w['ids'])} tokens in {w['generate_s']}s", flush=True)
    out["a_gpu"] = _gpu_memory()
    del model_a
    sa.close()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- B: merged checkpoint, plain HF BF16 load (FP32 A_log/dt_bias via NVIDIA's patch) -----
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    t = time.monotonic()
    nv.vendor("production_gdn_fix")._keep_native_gdn_fp32(Qwen3_5PreTrainedModel)
    model_b, info = AutoModelForImageTextToText.from_pretrained(
        merged, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True)
    model_b.to("cuda:0").eval()
    out["b_load_s"] = round(time.monotonic() - t, 1)
    out["b_loading_info"] = {k: len(info.get(k) or []) for k in
                             ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    verify = nv.vendor("verify_gdn").verify
    out["b_gdn"] = {"vs_merged": verify(model_b, merged)["status"],
                    "vs_base_checkpoint": verify(model_b, _tempseg_base())["status"]}
    out["b_parameter_dtypes"] = dict(Counter(str(p.dtype) for p in model_b.parameters()))
    proc_b = nv.models.processor("tempseg", merged)
    b_params = mt.tensor_digests(model_b.named_parameters(),
                                 skip=lambda n: n.endswith(".weight") and spec.matches(n[:-7]))
    out["non_target_parameters"] = {
        "count_a": len(a_params), "count_b": len(b_params),
        "names_equal": set(a_params) == set(b_params),
        "bit_identical": sum(a_params[k] == b_params.get(k) for k in a_params)}

    b_prompts = _prompt_inputs(proc_b)
    out["fixed_prompts"] = []
    for p, pb, la in zip(prompts, b_prompts, a_fixed, strict=True):
        lb = _forward_logits(model_b, p, autocast=False)
        lbc = _forward_logits(model_b, p, autocast=True)
        out["fixed_prompts"].append({"tokens": int(p["input_ids"].shape[1]),
                                     "processor_ids_equal": torch.equal(p["input_ids"],
                                                                        pb["input_ids"]),
                                     "a_vs_b": mt.logit_diff(la, lb),
                                     "a_vs_b_autocast": mt.logit_diff(la, lbc)})
    out["a_repeat"] = mt.logit_diff(a_fixed[0], a_repeat)
    lb = fixed_forward(model_b)[0].float().cpu()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lbc = fixed_forward(model_b)[0].float().cpu()
    out["random_tokens"] = {"a_vs_b": mt.logit_diff(a_random, lb),
                            "a_vs_b_autocast": mt.logit_diff(a_random, lbc)}

    out["windows"] = []
    for ep, wa in zip(episodes, a_windows, strict=True):
        wb = _capture_window(nv, model_b, proc_b, ep)
        row = {"episode": ep["id"], "pool": wa["pool"], "views": wa["views"],
               "input_tokens": wa["input_tokens"],
               "inputs_identical": wa["input_digest"] == wb["input_digest"],
               "a_tokens": len(wa["ids"]), "b_tokens": len(wb["ids"]),
               "a_error": wa.get("error"), "b_error": wb.get("error"),
               "ids_identical": wa["ids"] == wb["ids"],
               "first_divergence": mt.first_divergence(wa["ids"], wb["ids"]),
               "text_identical": wa.get("text") == wb.get("text"),
               "json_equal": _json_equal(wa.get("text"), wb.get("text")),
               "a_generate_s": wa["generate_s"], "b_generate_s": wb["generate_s"]}
        if "tf_logits" in wa:
            ext, keep = mt.teacher_forced_inputs(wa["inputs"], wa["ids"])
            tb = _forward_logits(model_b, ext, autocast=False, logits_to_keep=keep)
            tbc = _forward_logits(model_b, ext, autocast=True, logits_to_keep=keep)
            gen = torch.tensor(wa["ids"])
            row["teacher_forced_on_a"] = {
                "a_vs_b": mt.logit_diff(wa["tf_logits"], tb),
                "a_vs_b_autocast": mt.logit_diff(wa["tf_logits"], tbc),
                "a_argmax_is_a_token": float((wa["tf_logits"].argmax(-1) == gen).double().mean()),
                "b_argmax_is_a_token": float((tb.argmax(-1) == gen).double().mean()),
            }
        out["windows"].append(row)
        print(f"B {ep['id']}: ids identical={row['ids_identical']}", flush=True)
    out["b_gpu"] = _gpu_memory()
    out["total_s"] = round(time.monotonic() - t0, 1)
    return out


@app.local_entrypoint()
def tempseg(step: str = "merge", force: bool = False, out: str = "") -> None:
    if step == "merge":
        result = merge_tempseg.remote(force=force, provenance=_git_provenance())
    elif step == "equivalence":
        result = tempseg_equivalence.remote()
    else:
        raise SystemExit(f"unknown step {step!r} (merge | equivalence)")
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if out:
        Path(out).write_text(text + "\n")
    print(text)
