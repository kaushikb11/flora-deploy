"""``flora-perf-lab`` (Perf-1): why is NVIDIA's ``generate`` ~3x slower on the cu128 image?

Ephemeral only (``modal run``), never deployed. Results in ``docs/perf-1.md``.

One H200 container per image variant. Each loads TempSeg (and optionally the BF16 Captioner)
through NVIDIA's own ``Session.load`` (D-1's offline layout), captures fixed model inputs with
NVIDIA's own builders (``pipeline.segment`` windows of a WS-D episode, D-1's method, and one
``caption_single`` call on a fixed span), then times NVIDIA's unmodified ``models.generate`` on
exactly those inputs:

- per call: wall, NVIDIA's ``elapsed_s``, prefill (to the first ``lm_head`` output, synced) and
  decode tokens/s (synced per step; HF ``generate`` already syncs every step), thread CPU time,
  text and per-step logits digests (D-1's), the input digest;
- ``torch.profiler`` of one short call per model: top CUDA kernels, GPU busy fraction, the
  attention kernel names, CPU-side op totals;
- SDPA backend sweeps (``sdpa_kernel`` contexts), an SDPA decode-shape micro-benchmark over
  growing KV lengths per backend, kernel-launch and interpreter micro-benchmarks;
- ``nvidia-smi`` clocks / utilization / power sampled at 1 Hz, tagged by phase;
- the stack: torch / CUDA / cuDNN / NVRTC builds, Python build, CPU, env, SDPA flags.

Variants (``--variant``):

- ``cu128``: the production ``img_hfref`` (``images.hfref_image()``), unchanged.
- ``cu130``: the pre-Fix-images ``img_hfref`` rebuilt from git history (commit ``LEGACY_REV``:
  Modal ``debian_slim`` 3.12 at builder 2025.06 + that commit's ``locks/hfref.txt``, PyPI torch
  2.11.0 cu130). Lab-only; the production builders are not touched.
- ``cu130_slim``: the current ``python:3.12.10-slim-bookworm`` base by digest + the legacy cu130
  lock (separates the base switch from the torch switch).

CLI: ``uv run modal run src/flora_deploy/modal_app/perf_lab.py --variant cu128 --out r.json``.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import modal

from flora_deploy.config.pins import PINS_BY_KEY, RUNTIME
from flora_deploy.layout import GOLDEN_MOUNT, GOLDEN_VOLUME, WEIGHTS_MOUNT, WEIGHTS_VOLUME
from flora_deploy.modal_app import images
from flora_deploy.stages.reference import OFFLINE_ROOT, offline_env

APP_NAME = "flora-perf-lab"
app = modal.App(APP_NAME)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)

LIFETIME_S = 19 * 60  # hard cap per container (task rule: <= 20 min)
SAMPLES = GOLDEN_MOUNT / "samples" / "wsd" / "manifest.json"
# Last main commit before Fix-images (de655f1): W3-A and D-1 ran on this img_hfref.
LEGACY_REV = "0812b4d"
LEGACY_LOCK = "src/flora_deploy/modal_app/locks/hfref.txt"
_LAB_DIR = Path(tempfile.gettempdir()) / "flora-perf-lab"
BACKEND_MODES = ("cudnn", "flash", "efficient", "math", "nocudnn")
# Tagged into every nvidia-smi sample.
_PHASE = {"name": "start"}


# ------------------------------------------------------------------------------------ images
def _legacy_lock() -> Path:
    """The cu130 lock as committed at ``LEGACY_REV``, written to a local temp file (read only
    when the image is built locally; containers never need the file)."""
    path = _LAB_DIR / f"hfref-{LEGACY_REV}.txt"
    if modal.is_local() and not path.exists():
        root = Path(__file__).resolve().parents[3]
        text = subprocess.run(["git", "-C", str(root), "show", f"{LEGACY_REV}:{LEGACY_LOCK}"],
                              check=True, capture_output=True, text=True).stdout
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def _source(image: modal.Image) -> modal.Image:
    return image.env(images.ENV["img_hfref"]).add_local_python_source(
        "flora_deploy", ignore=~modal.FilePatternMatcher(*images.SOURCE_PATTERNS))


def legacy_cu130_image() -> modal.Image:
    """Exactly ``hfref_image()`` at ``LEGACY_REV`` (debian_slim + uv_pip_install of the lock)."""
    images._pin_builder_version()
    return _source(modal.Image.debian_slim(python_version=RUNTIME["python_image"]).uv_pip_install(
        requirements=[str(_legacy_lock())], extra_options="--require-hashes --no-build",
        uv_version=RUNTIME["uv"]))


def slim_cu130_image() -> modal.Image:
    """The current base (official python slim by digest) + the legacy cu130 lock, installed the
    way the current builder installs (uv by digest, same options)."""
    images._pin_builder_version()
    dest = "/.uv/hfref-cu130.txt"
    ctx = "/.lock_hfref-cu130.txt"
    steps = [
        f"COPY --from={images.UV_IMAGE} /uv /.uv/uv",
        f"COPY {ctx} {dest}",
        "RUN /.uv/uv pip install --python $(command -v python) --compile-bytecode "
        f"--require-hashes --no-build --requirements {dest}",
    ]
    return _source(modal.Image.from_registry(images.PY_BASE).dockerfile_commands(
        steps, context_files={ctx: str(_legacy_lock())}))


# ------------------------------------------------------------------------------------ helpers
class _Enough(Exception):
    pass


def _sha_tensor(t) -> str:
    import torch

    return hashlib.sha256(t.detach().contiguous().cpu().view(-1).view(torch.uint8)
                          .numpy().tobytes()).hexdigest()


def _digest(texts) -> str:
    return hashlib.sha256("|".join(texts).encode()).hexdigest()[:12]


class GpuSampler:
    """``nvidia-smi`` at ~1 Hz in a thread; every sample carries the current ``_PHASE``."""

    FIELDS = ("clocks.sm", "clocks.mem", "clocks.max.sm", "utilization.gpu", "power.draw",
              "temperature.gpu", "clocks_event_reasons.active")

    def __init__(self, period: float = 1.0):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.fields = self.FIELDS
        self.thread = threading.Thread(target=self._run, daemon=True, name="perf-gpu-sampler")

    def _query(self) -> list[str] | None:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(self.fields)}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10)
        if out.returncode != 0 and "clocks_event_reasons" in ",".join(self.fields):
            self.fields = (*self.FIELDS[:-1], "clocks_throttle_reasons.active")
            return None
        return [x.strip() for x in out.stdout.strip().split(",")] if out.returncode == 0 else None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = self._query()
                if row:
                    self.samples.append({"t": time.monotonic(), "phase": _PHASE["name"],
                                         **dict(zip(self.fields, row, strict=False))})
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            self._stop.wait(self.period)

    def start(self) -> GpuSampler:
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        by: dict[str, list[dict]] = {}
        for s in self.samples:
            by.setdefault(s["phase"], []).append(s)

        def mean(rows, key):
            vals = []
            for r in rows:
                with contextlib.suppress(ValueError, TypeError, KeyError):
                    vals.append(float(r[key]))
            return round(sum(vals) / len(vals), 1) if vals else None

        reason_key = self.fields[-1]
        return {phase: {"n": len(rows), "sm_mhz": mean(rows, "clocks.sm"),
                        "mem_mhz": mean(rows, "clocks.mem"),
                        "max_sm_mhz": mean(rows, "clocks.max.sm"),
                        "util_pct": mean(rows, "utilization.gpu"),
                        "power_w": mean(rows, "power.draw"),
                        "temp_c": mean(rows, "temperature.gpu"),
                        "reasons": sorted({r.get(reason_key, "") for r in rows})}
                for phase, rows in by.items()}


def stack_report() -> dict:
    import importlib.metadata as md
    import platform
    import sys
    import sysconfig

    import torch

    def smi(*args):
        with contextlib.suppress(Exception):
            return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                                  timeout=20).stdout.strip()[:1500]
        return None

    cpu_model: dict[str, Any] = {}
    with contextlib.suppress(OSError):
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            key = ln.split(":", 1)[0].strip()
            if key in ("vendor_id", "model name", "cpu family", "model", "flags") and \
                    key not in cpu_model:
                val = ln.split(":", 1)[1].strip()
                cpu_model[key] = ([f for f in val.split() if f.startswith(("avx", "amx"))]
                                  if key == "flags" else val)
    import torch._C as tc

    with contextlib.suppress(Exception):
        cpu_model["torch_cpu_capability"] = tc._get_cpu_capability()
    dists = {d.metadata["Name"]: d.version for d in md.distributions()
             if d.metadata["Name"] and (d.metadata["Name"].lower().startswith(
                 ("nvidia", "torch", "triton", "transformers", "numpy", "pillow", "accelerate",
                  "peft", "safetensors", "av")))}
    cuda = torch.backends.cuda
    return {
        "python": sys.version, "python_config_args": sysconfig.get_config_var("CONFIG_ARGS"),
        "python_cflags": sysconfig.get_config_var("PY_CFLAGS"), "libc": platform.libc_ver(),
        "executable": sys.executable,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "arch_list": torch.cuda.get_arch_list(),
        "nvidia_smi": smi("--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,"
                          "power.limit,persistence_mode,compute_mode,ecc.mode.current",
                          "--format=csv"),
        "cpu_model": cpu_model, "cpu_count": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "sdpa": {"flash": cuda.flash_sdp_enabled(),
                 "mem_efficient": cuda.mem_efficient_sdp_enabled(),
                 "math": cuda.math_sdp_enabled(), "cudnn": cuda.cudnn_sdp_enabled()},
        "matmul_allow_tf32": cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "env": {k: v for k, v in os.environ.items() if k.startswith(
            ("CUDA", "TORCH", "PYTORCH", "OMP", "MKL", "NCCL", "CUDNN", "CUBLAS", "NVIDIA",
             "LD_", "TRITON"))},
        "dists": dict(sorted(dists.items())),
        "torch_config": torch.__config__.show()[:2500],
        "parallel_info": torch.__config__.parallel_info()[:800],
    }


# ------------------------------------------------------------------------------------ micro
def micro_launch() -> dict:
    import torch

    x = torch.zeros(1, device="cuda")
    for _ in range(200):
        x.add_(1)
    torch.cuda.synchronize()
    n = 5000
    t = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    enq = time.perf_counter() - t
    torch.cuda.synchronize()
    tot = time.perf_counter() - t
    t = time.perf_counter()
    s = 0
    for i in range(2_000_000):
        s += i
    py = time.perf_counter() - t
    return {"enqueue_us_per_op": round(enq / n * 1e6, 2),
            "total_us_per_op": round(tot / n * 1e6, 2),
            "python_loop_2m_s": round(py, 3)}


def _backend(name: str):
    from torch.nn.attention import SDPBackend

    return {"cudnn": [SDPBackend.CUDNN_ATTENTION], "flash": [SDPBackend.FLASH_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION], "math": [SDPBackend.MATH],
            "nocudnn": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH]}[name]


def micro_sdpa(heads: int, kv_heads: int, head_dim: int, base_len: int = 1900,
               steps: int = 24) -> dict:
    """Decode-shaped SDPA (1 query vs a KV cache that grows by one per call, BF16) per backend:
    the first call at each new length, then repeats at an already-seen length; plus one causal
    prefill. ``default`` is whatever torch picks; its kernel names are profiled."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel

    out: dict[str, Any] = {"shape": {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
                                     "base_len": base_len}}
    q = torch.randn(1, heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, kv_heads, base_len + steps + 2, head_dim, device="cuda",
                     dtype=torch.bfloat16)
    rep = heads // kv_heads
    k_full = kv.repeat_interleave(rep, dim=1)
    pq = torch.randn(1, heads, base_len, head_dim, device="cuda", dtype=torch.bfloat16)

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3

    for name in ("default", "cudnn", "flash", "efficient", "math"):
        ctx = contextlib.nullcontext() if name == "default" else sdpa_kernel(_backend(name))
        res: dict[str, Any] = {}
        try:
            with ctx:
                new_len = []
                for i in range(steps):
                    L = base_len + i  # noqa: N806
                    k = k_full[:, :, :L]
                    new_len.append(timed(lambda k=k: F.scaled_dot_product_attention(q, k, k)))
                k = k_full[:, :, :base_len + steps // 2]
                again = [timed(lambda k=k: F.scaled_dot_product_attention(q, k, k))
                         for _ in range(10)]
                gqa = None
                with contextlib.suppress(Exception):
                    kk = kv[:, :, :base_len]
                    def gqa_call(kk=kk):
                        return F.scaled_dot_product_attention(q, kk, kk, enable_gqa=True)

                    timed(gqa_call)
                    gqa = round(timed(gqa_call), 3)
                pk = k_full[:, :, :base_len]
                def pre_call(pk=pk):
                    return F.scaled_dot_product_attention(pq, pk, pk, is_causal=True)

                timed(pre_call)
                pre = timed(pre_call)
            res = {"new_len_ms_first": round(new_len[0], 3),
                   "new_len_ms_median": round(sorted(new_len[1:])[len(new_len[1:]) // 2], 3),
                   "new_len_ms_mean": round(sum(new_len[1:]) / len(new_len[1:]), 3),
                   "seen_len_ms_median": round(sorted(again)[5], 3), "gqa_ms": gqa,
                   "prefill_ms": round(pre, 3)}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"[:300]}
        out[name] = res
    # which kernels does the default pick (decode + prefill)?
    from torch.profiler import ProfilerActivity, profile

    k = k_full[:, :, :base_len + 3]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(q, k, k)
        F.scaled_dot_product_attention(pq, k_full[:, :, :base_len], k_full[:, :, :base_len],
                                       is_causal=True)
        torch.cuda.synchronize()
    out["default_ops"] = sorted({e.key for e in prof.key_averages()})[:40]
    return out


def profile_summary(prof, wall_s: float, top: int = 30) -> dict:
    rows = []
    for e in prof.key_averages():
        dev = float(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
                    or 0)
        rows.append({"name": e.key[:140], "count": e.count,
                     "self_cpu_ms": round(e.self_cpu_time_total / 1e3, 2),
                     "self_dev_ms": round(dev / 1e3, 2)})
    kernels = [ev for ev in prof.events() if "CUDA" in str(getattr(ev, "device_type", ""))]
    dev_total = sum(ev.time_range.elapsed_us() for ev in kernels) / 1e6
    by_dev = sorted(rows, key=lambda r: -r["self_dev_ms"])[:top]
    by_cpu = sorted(rows, key=lambda r: -r["self_cpu_ms"])[:top]
    words = ("cudnn", "flash", "fmha", "attention", "sdpa", "mha", "nvrtc", "cutlass", "sm90",
             "efficient", "gemv", "gemm", "jit")
    attn = sorted({r["name"] for r in rows if any(w in r["name"].lower() for w in words)})[:60]
    return {"wall_s": round(wall_s, 3), "gpu_kernel_s": round(dev_total, 3),
            "gpu_busy_frac": round(dev_total / wall_s, 3) if wall_s else None,
            "n_kernels": len(kernels), "top_device": by_dev, "top_cpu": by_cpu,
            "matching_names": attn}


# ------------------------------------------------------------------------------------ lab
class _LabBase:
    def _load(self, captioner: bool) -> dict:
        t0 = time.monotonic()
        from flora_deploy.stages.reference import (
            build_offline_layout,
            capture_runtime_events,
            check_nv_models,
        )

        build_offline_layout(OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("tempseg",))
        from flora_deploy.config.pins import PINS_BY_KEY as PINS
        from flora_deploy.layout import snapshot_path
        from flora_deploy.nv import CachedSession, Family, load_family

        code_dir = Path(snapshot_path(PINS["tempseg_code"], WEIGHTS_MOUNT))
        self.nv = nv = load_family(Family.SEGCAP, code_dir)
        check_nv_models(nv.config.MODELS)
        info: dict[str, Any] = {}
        if captioner:
            from flora_deploy.accel import captioner_load

            derived = captioner_load.default_derived_dir(WEIGHTS_MOUNT)
            self.bf16 = captioner_load.install(nv, derived)
        self.session = CachedSession.from_family(nv)
        with capture_runtime_events() as events:
            self.ts = self.session.load("tempseg")
        info["tempseg_s"] = round(time.monotonic() - t0, 1)
        if captioner:
            t = time.monotonic()
            with capture_runtime_events() as ev2:
                self.cap = self.session.load("captioner")
            events += ev2
            info["captioner_s"] = round(time.monotonic() - t, 1)
        info["ready"] = [{k: v for k, v in e.items() if k in ("kind", "torch_version", "gpu",
                                                              "allocated_gib")}
                         for e in events if e.get("event") == "model_ready"]
        return info

    def _episode(self, episode: str):
        nv = self.nv
        manifest = json.loads(Path(SAMPLES).read_text())
        ep = next(e for e in manifest["episodes"] if e["episode"] == episode)
        vids, canonical = nv.api.DataEnginePipeline._inputs(
            ep["videos"]["main"], ep["profile"], ep["videos"].get("left_wrist"),
            ep["videos"].get("right_wrist"), False, "")
        return vids, canonical, ep

    def _recorder(self, captured: list, want: int, run_real_before_last: bool):
        import torch

        real = self.nv.pipeline.generate

        def recorder(model, proc, inputs, budget):
            if len(captured) >= want:
                raise _Enough
            captured.append({"budget": budget, "inputs": {
                k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in dict(inputs).items()}})
            if len(captured) >= want and not run_real_before_last:
                raise _Enough
            t = time.monotonic()
            out = real(model, proc, inputs, budget)
            captured[-1]["capture_call_s"] = round(time.monotonic() - t, 2)
            return out

        return real, recorder

    def _capture_tempseg(self, episode: str, windows: int) -> list[dict]:
        vids, canonical, _ = self._episode(episode)
        pool = self.nv.config.PROFILE_POOLS[canonical]
        captured: list[dict] = []
        real, rec = self._recorder(captured, windows, run_real_before_last=False)
        self.nv.pipeline.generate = rec
        try:
            self.nv.pipeline.segment(vids, pool, self.session, "", lambda *a, **k: None)
        except Exception as e:  # noqa: BLE001 - _Enough (possibly wrapped) ends capture
            print(f"tempseg capture ended: {type(e).__name__}", flush=True)
        finally:
            self.nv.pipeline.generate = real
        for c in captured:
            c["kind"] = "tempseg"
        return captured

    def _capture_caption(self, episode: str) -> list[dict]:
        vids, canonical, ep = self._episode(episode)
        captured: list[dict] = []
        real, rec = self._recorder(captured, 1, run_real_before_last=False)
        self.nv.pipeline.generate = rec
        dur = float(vids["main"].duration_s)
        spans = [(2.0, 6.0), (1.0, 4.0), (0.5, min(dur - 0.5, 8.0))]
        try:
            for a, b in spans:
                with contextlib.suppress(_Enough):
                    self.nv.pipeline.caption_single(
                        vids, [{"start_s": a, "end_s": b, "metadata": {}}], canonical,
                        self.session, "", lambda *x, **k: None)
                if captured:
                    captured[0]["span"] = [a, b]
                    break
        finally:
            self.nv.pipeline.generate = real
        for c in captured:
            c["kind"] = "captioner"
        return captured

    def _model(self, kind: str):
        return self.ts if kind == "tempseg" else self.cap

    def _lm_head(self, model):
        for name, m in model.named_modules():
            if name.endswith("lm_head"):
                return m
        raise RuntimeError("no lm_head")

    def _generate(self, item: dict, *, budget: int | None = None, steps_timing: bool = True,
                  ref: dict | None = None, keep_ref: bool = False) -> dict:
        """NVIDIA's unmodified ``models.generate`` on the captured inputs."""
        import torch

        from flora_deploy.stages.reference import capture_runtime_events

        model, proc = self._model(item["kind"])
        stamps: list[float] = []
        digests: list[str] = []
        kept: list = []
        diff: dict = {}

        def hook(_m, _i, out):
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
            t = out[:, -1] if out.dim() == 3 else out
            digests.append(_sha_tensor(t))
            j = len(digests) - 1
            if keep_ref:
                kept.append(t.detach().cpu())
            if ref and not diff and j < len(ref["steps"]) and digests[-1] != ref["steps"][j]:
                a, b = t.detach().float().cpu(), ref["logits"][j].float()
                diff.update(step=j, max_abs=float((a - b).abs().max()),
                            argmax_same=bool(int(a.argmax()) == int(b.argmax())))

        h = self._lm_head(model).register_forward_hook(hook) if steps_timing else None
        err = text = None
        torch.cuda.synchronize()
        cpu0, t0 = time.thread_time(), time.perf_counter()
        try:
            with capture_runtime_events() as events:
                text = self.nv.models.generate(model, proc, dict(item["inputs"]),
                                               budget or item["budget"])
        except Exception as e:  # noqa: BLE001 - incomplete (capped budget) is expected
            err = f"{type(e).__name__}: {e}"[:200]
        finally:
            if h is not None:
                h.remove()
        torch.cuda.synchronize()
        wall, cpu = time.perf_counter() - t0, time.thread_time() - cpu0
        start = next((e for e in events if e.get("event") == "generation_start"), {})
        end = next((e for e in events if e.get("event") == "generation_end"), {})
        n = end.get("generated_tokens")
        res = {"kind": item["kind"], "input_tokens": start.get("input_tokens"),
               "input_digest": str(start.get("input_tensor_digest"))[:16], "tokens": n,
               "complete": end.get("complete"), "nv_elapsed_s": end.get("elapsed_s"),
               "wall_s": round(wall, 3), "thread_cpu_s": round(cpu, 3), "error": err,
               "text_digest": hashlib.sha256(str(text).encode()).hexdigest()[:12]}
        if stamps:
            res["prefill_s"] = round(stamps[0] - t0, 3)
            if len(stamps) > 1:
                dec = stamps[-1] - stamps[0]
                res["decode_s"] = round(dec, 3)
                res["decode_tok_s"] = round((len(stamps) - 1) / dec, 2)
                gaps = sorted(b - a for a, b in zip(stamps, stamps[1:], strict=False))
                res["step_ms_median"] = round(gaps[len(gaps) // 2] * 1e3, 2)
                res["step_ms_max"] = round(gaps[-1] * 1e3, 2)
                res["step_ms_first5"] = [round((b - a) * 1e3, 1) for a, b in
                                         zip(stamps[:6], stamps[1:6], strict=False)]
            res["logit_seq_digest"] = _digest(digests)
            if ref is not None:
                k = next((j for j, (a, b) in enumerate(zip(digests, ref["steps"], strict=False))
                          if a != b), None)
                if k is None and len(digests) != len(ref["steps"]):
                    k = min(len(digests), len(ref["steps"]))
                res["first_diff_step_vs_ref"] = k
                res["diff"] = diff
                res["text_equals_ref"] = text == ref["text"]
        if keep_ref:
            res["_ref"] = {"steps": digests, "logits": kept, "text": text}
        return res

    def _profile(self, item: dict, budget: int) -> dict:
        import torch
        from torch.profiler import ProfilerActivity, profile

        model, proc = self._model(item["kind"])
        torch.cuda.synchronize()
        t = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with contextlib.suppress(Exception):
                self.nv.models.generate(model, proc, dict(item["inputs"]), budget)
            torch.cuda.synchronize()
        return profile_summary(prof, time.perf_counter() - t)

    @staticmethod
    def _png_load(stop: threading.Event) -> None:
        """CPU contention like the worker's episode threads (PIL PNG encode holds the GIL in
        parts)."""
        import io

        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1080, 1920, 3),
                                                                dtype=np.uint8))
        while not stop.is_set():
            img.save(io.BytesIO(), format="PNG")

    def run(self, episode: str, windows: int, captioner: bool, deadline_s: float,
            backends: list[str], reps: int, profile: bool = False,
            threads: bool = True) -> dict:
        import torch
        import torch.nn.attention  # noqa: F401 - sdpa_kernel below

        t_start = time.monotonic()
        sampler = GpuSampler().start()
        out: dict[str, Any] = {"variant": self.VARIANT, "episode": episode}

        def left() -> float:
            return deadline_s - (time.monotonic() - t_start)

        def phase(name):
            _PHASE["name"] = name
            print(f"[{self.VARIANT} {time.monotonic() - t_start:7.1f}s] {name}", flush=True)

        try:
            phase("load")
            out["load"] = self._load(captioner)
            out["stack"] = stack_report()
            print(json.dumps({k: out["stack"][k] for k in ("torch", "torch_cuda", "cudnn",
                                                           "python", "sdpa")}), flush=True)
            phase("capture")
            items = self._capture_tempseg(episode, windows)
            if captioner:
                try:
                    items += self._capture_caption(episode)
                except Exception as e:  # noqa: BLE001 - TempSeg timings still count
                    out["caption_capture_error"] = f"{type(e).__name__}: {e}"[:300]
            out["captured"] = [{"kind": i["kind"], "budget": i["budget"],
                                "capture_call_s": i.get("capture_call_s"),
                                "span": i.get("span")} for i in items]
            out["capture_s"] = round(time.monotonic() - t_start, 1)
            calls: list[dict] = []
            refs: dict[int, dict] = {}

            def call(mode: str, idx: int, runner=None, **kw):
                ctx = (torch.nn.attention.sdpa_kernel(_backend(mode)) if mode in BACKEND_MODES
                       else contextlib.nullcontext())
                with ctx:
                    def go():
                        return self._generate(items[idx], ref=refs.get(idx), **kw)

                    r = runner(go) if runner is not None else go()
                r.update(mode=mode, item=idx)
                if "_ref" in r:
                    refs[idx] = r.pop("_ref")
                calls.append(r)
                print(f"  [{self.VARIANT}] {mode} item{idx}: " + json.dumps(
                    {k: r.get(k) for k in ("tokens", "wall_s", "prefill_s", "decode_tok_s",
                                           "step_ms_median", "text_equals_ref",
                                           "first_diff_step_vs_ref", "error")}), flush=True)
                return r

            # 1. NVIDIA as is, on every captured input
            phase("nvidia")
            for i in range(len(items)):
                call("nvidia", i, keep_ref=True)
            out["calls"] = calls
            # 1b. thread-local cuDNN graph cache: same input on a brand-new thread (cold), then
            # twice on one persistent accel.gpu_thread thread (cold, then warm)
            if threads:
                from flora_deploy.accel.gpu_thread import GpuThread

                phase("threads")
                gt = GpuThread()

                def fresh(fn):
                    box: dict = {}
                    t = threading.Thread(target=lambda: box.update(r=fn()))
                    t.start()
                    t.join()
                    return box["r"]

                for mode, runner in (("fresh_thread", fresh), ("gpu_thread_1", gt.run),
                                     ("gpu_thread_2", gt.run), ("main_thread_again", None)):
                    if left() > next((c["wall_s"] for c in calls if c["item"] == 0), 30) + 30:
                        call(mode, 0, runner=runner)
                gt.close()
            # 2. one profiled short call per model (costly: ~70 s per call under the profiler)
            phase("profile")
            prof = {}
            for i, it in enumerate(items):
                if not profile or it["kind"] in prof or left() < 60:
                    continue
                prof[it["kind"]] = self._profile(it, 16 if it["kind"] == "tempseg" else 8)
                prof[it["kind"]]["item"] = i
            out["profile"] = prof
            # 3. micro-benchmarks
            phase("micro")
            out["micro_launch"] = micro_launch()
            tc = self.ts[0].config.get_text_config()
            out["micro_sdpa"] = micro_sdpa(tc.num_attention_heads, tc.num_key_value_heads,
                                           getattr(tc, "head_dim", None) or
                                           tc.hidden_size // tc.num_attention_heads)
            # 4. SDPA backend sweeps (outputs compared with the NVIDIA r0 of the same input)
            for mode in backends:
                phase(f"sdpa_{mode}")
                for i in range(len(items)):
                    est = next((c["wall_s"] for c in calls if c["item"] == i), 30) * 1.5 + 10
                    if left() < est:
                        out.setdefault("skipped", []).append(f"{mode}:{i}")
                        continue
                    try:
                        call(mode, i)
                    except Exception as e:  # noqa: BLE001
                        calls.append({"mode": mode, "item": i,
                                      "error": f"{type(e).__name__}: {e}"[:300]})
            # 5. repeats of NVIDIA as is (run-to-run spread) and under CPU contention
            phase("nvidia_repeat")
            for _ in range(max(0, reps - 1)):
                for i in range(len(items)):
                    if left() > next((c["wall_s"] for c in calls if c["item"] == i), 30) + 20:
                        call("nvidia", i)
            phase("png_contention")
            if left() > 90:
                stop = threading.Event()
                ts = [threading.Thread(target=self._png_load, args=(stop,), daemon=True)
                      for _ in range(3)]
                for t in ts:
                    t.start()
                try:
                    call("png_contention", 0)
                finally:
                    stop.set()
        except Exception as e:  # noqa: BLE001 - always return what was measured
            import traceback

            out["error"] = f"{type(e).__name__}: {e}"[:600]
            out["traceback"] = traceback.format_exc()[-3000:]
        finally:
            sampler.stop()
            out["gpu_samples"] = sampler.summary()
            out["seconds"] = round(time.monotonic() - t_start, 1)
            # Also in the log, so a failed result upload never loses the numbers.
            print(f"PERF_RESULT {self.VARIANT} " + json.dumps(
                {k: v for k, v in out.items() if k not in ("traceback",)}, default=str),
                flush=True)
        return out


_CLS = dict(
    gpu="H200",
    cpu=4.0,
    memory=64 * 1024,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden.read_only()},
    env=offline_env(OFFLINE_ROOT),
    timeout=LIFETIME_S,
    startup_timeout=5 * 60,
    scaledown_window=2,
    max_containers=1,
    # Network on: Modal uploads large method results as blobs. HF stays offline (env).
    block_network=False,
)


@app.cls(image=images.hfref_image(), **_CLS)
class LabCu128(_LabBase):
    VARIANT = "cu128"

    @modal.method()
    def measure(self, **kw) -> dict:
        return self.run(**kw)


@app.cls(image=legacy_cu130_image(), **_CLS)
class LabCu130(_LabBase):
    VARIANT = "cu130"

    @modal.method()
    def measure(self, **kw) -> dict:
        return self.run(**kw)


@app.cls(image=slim_cu130_image(), **_CLS)
class LabCu130Slim(_LabBase):
    VARIANT = "cu130_slim"

    @modal.method()
    def measure(self, **kw) -> dict:
        return self.run(**kw)


VARIANTS = {"cu128": LabCu128, "cu130": LabCu130, "cu130_slim": LabCu130Slim}


@app.local_entrypoint()
def main(variant: str = "cu128,cu130", episode: str = "homer_50", windows: int = 2,
         captioner: bool = True, deadline_min: float = 14.0,
         backends: str = "nocudnn,flash", reps: int = 2, profile: bool = False,
         threads: bool = True, out: str = "") -> None:
    """Runs the variants in parallel (one container each) and writes one JSON per variant."""
    kw = dict(episode=episode, windows=windows, captioner=captioner,
              deadline_s=deadline_min * 60, backends=[b for b in backends.split(",") if b],
              reps=reps, profile=profile, threads=threads)
    names = [v for v in variant.split(",") if v]
    calls = {v: VARIANTS[v]().measure.spawn(**kw) for v in names}
    for v, fc in calls.items():
        res = fc.get()
        text = json.dumps(res, indent=1, default=str)
        if out:
            path = Path(out.replace(".json", f"-{v}.json"))
            path.write_text(text)
            print(f"wrote {path}")
        print(text[:6000])
    _ = (GOLDEN_VOLUME, PINS_BY_KEY)
