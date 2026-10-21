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
