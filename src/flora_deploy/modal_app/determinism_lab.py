"""``flora-determinism-lab`` (D-1): is NVIDIA's TempSeg ``generate`` reproducible run to run?

Ephemeral only (``modal run``), never deployed. Results in ``docs/determinism.md``.

One GPU container loads **only TempSeg** through NVIDIA's own ``Session.load`` (same offline
layout as ``reference_app``), runs NVIDIA's ``pipeline.segment`` on one staged WS-D episode just
long enough to capture the model inputs of its first ``--windows`` TempSeg calls, then calls
NVIDIA's unmodified ``models.generate`` on those exact inputs ``--reps`` times per *mode*:

- ``baseline``: NVIDIA's settings as loaded.
- ``dirty``: baseline, but before each call the caching allocator is filled with random bytes and
  a random-sized tensor shifts later allocations (tests reads of uninitialized memory /
  address-dependent kernel choice).
- ``threads1``: torch intra-op threads 1 (CPU side only).
- ``cudnn_notf32``: ``torch.backends.cudnn.allow_tf32=False``.
- ``sdpa_math``: all SDPA calls forced to the math backend.
- ``sdpa_nocudnn``: SDPA's cuDNN backend disabled (flash / efficient / math stay allowed).
- ``det_warn``: ``use_deterministic_algorithms(True, warn_only=True)``: lists flagged ops.
- ``det``: strict ``use_deterministic_algorithms(True)`` + cuDNN deterministic (SDPA backends
  left to torch). ``det_nofill`` is the same without NaN-filling uninitialized memory.
  (``accel.determinism``'s mode additionally disables cuDNN SDPA explicitly.)

Each call records NVIDIA's ``generation_start/_end`` events (input digest, token count, elapsed),
the text, and a sha256 per decode step of the ``lm_head`` output. ``prefill`` modes run one
forward pass with hooks on the vision tower and every decoder sub-module and report the first
module whose output bytes differ between repeats.

CLI: ``uv run modal run src/flora_deploy/modal_app/determinism_lab.py --gpu L40S``.
The container sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` from the start (required by strict
deterministic mode; it only fixes the workspace size, which is torch's default on Hopper).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import modal

from flora_deploy.accel.determinism import container_env
from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import (
    GOLDEN_MOUNT,
    GOLDEN_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    snapshot_path,
)
from flora_deploy.modal_app.images import hfref_image
from flora_deploy.stages.reference import OFFLINE_ROOT, offline_env

APP_NAME = "flora-determinism-lab"
app = modal.App(APP_NAME)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
golden = modal.Volume.from_name(GOLDEN_VOLUME, create_if_missing=True, version=2)

LIFETIME_S = 28 * 60  # watchdog: spend guard
SAMPLES = GOLDEN_MOUNT / "samples" / "wsd" / "manifest.json"
DEFAULT_MODES = ("baseline", "prefill", "dirty", "det_warn", "det", "prefill_det",
                 "sdpa_math", "cudnn_notf32", "det_nofill")


def _watchdog(limit_s: float) -> None:
    def kill():
        print(f"determinism-lab watchdog: {limit_s:.0f}s reached, exiting", flush=True)
        os._exit(17)

    t = threading.Timer(limit_s, kill)
    t.daemon = True
    t.start()


class _Enough(Exception):
    pass


def _sha(t) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(-1).view(__import__("torch").uint8)
                          .numpy().tobytes()).hexdigest()


def _first_tensor(out):
    import torch

    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            t = _first_tensor(x)
            if t is not None:
                return t
        return None
    for name in ("last_hidden_state", "pooler_output", "logits"):
        v = getattr(out, name, None)
        if torch.is_tensor(v):
            return v
    return None


@app.cls(
    image=hfref_image(),
    gpu="L40S",
    cpu=8.0,
    memory=64 * 1024,
    volumes={str(WEIGHTS_MOUNT): weights.read_only(), str(GOLDEN_MOUNT): golden.read_only()},
    env={**offline_env(OFFLINE_ROOT), **container_env("deterministic")},
    timeout=LIFETIME_S,
    startup_timeout=15 * 60,
    scaledown_window=10,
    max_containers=1,
    block_network=True,
)
class Lab:
    @modal.enter()
    def load(self) -> None:
        _watchdog(LIFETIME_S - 30)
        t0 = time.monotonic()
        from flora_deploy.stages.reference import build_offline_layout, check_nv_models

        build_offline_layout(OFFLINE_ROOT, WEIGHTS_MOUNT, kinds=("tempseg",))
        import torch

        from flora_deploy.nv import CachedSession, Family, load_family
        from flora_deploy.stages.reference import capture_runtime_events

        code_dir = Path(snapshot_path(PINS_BY_KEY["tempseg_code"], WEIGHTS_MOUNT))
        self.nv = nv = load_family(Family.SEGCAP, code_dir)
        check_nv_models(nv.config.MODELS)
        self.session = CachedSession.from_family(nv)
        with capture_runtime_events() as events:
            self.model, self.proc = self.session.load("tempseg")
        self.load_events = [e for e in events if e.get("event") == "model_ready"]
        self.load_s = round(time.monotonic() - t0, 1)
        self.gpu = torch.cuda.get_device_name(0)
        print(f"TempSeg loaded in {self.load_s}s on {self.gpu}", flush=True)

    # ---------------------------------------------------------------- capture
    def _capture(self, episode: str, windows: int) -> list[dict]:
        import torch

        nv = self.nv
        manifest = json.loads(Path(SAMPLES).read_text())
        ep = next(e for e in manifest["episodes"] if e["episode"] == episode)
        nv_videos, canonical = nv.api.DataEnginePipeline._inputs(
            ep["videos"]["main"], ep["profile"], ep["videos"].get("left_wrist"),
            ep["videos"].get("right_wrist"), False, "")
        pool = nv.config.PROFILE_POOLS[canonical]
        captured: list[dict] = []
        real = nv.pipeline.generate

        def recorder(model, proc, inputs, budget):
            if len(captured) >= windows:
                raise _Enough
            captured.append({
                "budget": budget,
                "inputs": {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                           for k, v in dict(inputs).items()},
            })
            return real(model, proc, inputs, budget)

        nv.pipeline.generate = recorder
        try:
            nv.pipeline.segment(nv_videos, pool, self.session, "", lambda *a, **k: None)
        except Exception as e:  # noqa: BLE001 - _Enough (or a segmenter wrapping it) ends capture
            print(f"capture ended: {type(e).__name__}", flush=True)
        finally:
            nv.pipeline.generate = real
        return captured

    # ---------------------------------------------------------------- one call
    def _lm_head(self):
        for name, m in self.model.named_modules():
            if name.endswith("lm_head"):
                return m
        raise RuntimeError("no lm_head")

    def _generate(self, item: dict, keep_logits: bool = False, ref: dict | None = None) -> dict:
        import torch

        from flora_deploy.stages.reference import capture_runtime_events

        steps: list[str] = []
        kept: list = []
        diff: dict = {}

        def hook(_m, _i, out):
            t = out[:, -1] if out.dim() == 3 else out
            j = len(steps)
            steps.append(_sha(t))
            if keep_logits:
                kept.append(t.detach().cpu())
            if ref and not diff and j < len(ref["steps"]) and steps[-1] != ref["steps"][j]:
                a, b = t.detach().float().cpu(), ref["logits"][j].float()
                diff.update(step=j, max_abs=float((a - b).abs().max()),
                            argmax_same=bool(int(a.argmax()) == int(b.argmax())),
                            ref_top2_margin=float(b.topk(2).values[..., 0].max()
                                                  - b.topk(2).values[..., 1].max()))

        h = self._lm_head().register_forward_hook(hook)
        err = None
        t = time.monotonic()
        try:
            with capture_runtime_events() as events:
                text = self.nv.models.generate(self.model, self.proc, dict(item["inputs"]),
                                               item["budget"])
        except Exception as e:  # noqa: BLE001
            text, err = None, f"{type(e).__name__}: {e}"[:400]
        finally:
            h.remove()
        torch.cuda.synchronize()
        start = next((e for e in events if e.get("event") == "generation_start"), {})
        end = next((e for e in events if e.get("event") == "generation_end"), {})
        return {"text": text, "error": err, "steps": steps, "logits": kept, "diff": diff,
                "input_digest": start.get("input_tensor_digest"),
                "tokens": end.get("generated_tokens"), "complete": end.get("complete"),
                "nv_elapsed_s": end.get("elapsed_s"), "wall_s": round(time.monotonic() - t, 3)}

    def _prefill(self, item: dict) -> dict:
        """One forward pass over the prompt (as generate's prefill: autocast BF16), hashing the
        output of the vision tower and every decoder sub-module."""
        import torch

        hashes: dict[str, str] = {}
        handles = []
        targets = []
        for name, m in self.model.named_modules():
            parts = name.split(".")
            if "visual" in parts:
                if parts[-1] == "visual":
                    targets.append((name, m))
                continue
            if parts[-1] == "lm_head" or (len(parts) > 1 and parts[-2] == "layers") or (
                    len(parts) > 2 and parts[-3] == "layers"
                    and parts[-1] in ("linear_attn", "self_attn", "mlp")):
                targets.append((name, m))
        for name, m in targets:
            def hook(_m, _i, out, name=name):
                t = _first_tensor(out)
                if t is not None:
                    hashes[name] = _sha(t)
            handles.append(m.register_forward_hook(hook))
        inputs = dict(item["inputs"])
        inputs.pop("video_metadata", None)
        device = next(self.model.parameters()).device
        args = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.model(**args, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        torch.cuda.synchronize()
        return {"order": [n for n, _ in targets], "hashes": hashes}

    # ---------------------------------------------------------------- modes
    def _set_mode(self, mode: str):
        """Return (enter, exit) callables for ``mode``."""
        import contextlib
        import warnings

        import torch

        state: dict[str, Any] = {}
        stack = contextlib.ExitStack()

        def enter():
            if mode in ("det", "det_nofill", "prefill_det", "det_warn"):
                state["warn"] = stack.enter_context(warnings.catch_warnings(record=True))
                warnings.simplefilter("always")
                torch.use_deterministic_algorithms(True, warn_only=(mode == "det_warn"))
                torch.backends.cudnn.deterministic = True
                if mode == "det_nofill":
                    torch.utils.deterministic.fill_uninitialized_memory = False
            elif mode == "cudnn_notf32":
                torch.backends.cudnn.allow_tf32 = False
            elif mode == "sdpa_math":
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(sdpa_kernel([SDPBackend.MATH]))
            elif mode == "sdpa_nocudnn":
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                                 SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]))
            elif mode == "threads1":
                state["threads"] = torch.get_num_threads()
                torch.set_num_threads(1)

        def leave():
            stack.close()
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.utils.deterministic.fill_uninitialized_memory = True
            if "threads" in state:
                torch.set_num_threads(state["threads"])
            warns = sorted({str(w.message)[:300] for w in state.get("warn") or []})
            return warns

        return enter, leave

    def _dirty(self, rng) -> Any:
        import torch

        free, _ = torch.cuda.mem_get_info()
        n = int(min(free * 0.5, 12 * 2**30) // 4)
        junk = torch.empty(n, dtype=torch.float32, device="cuda").uniform_(-1e4, 1e4)
        del junk  # back to the caching allocator, contents left as random bytes
        shift = torch.empty(int(rng.integers(1, 64)) * 4096 + int(rng.integers(0, 512)),
                            dtype=torch.uint8, device="cuda")
        return shift

    @modal.method()
    def run(self, episode: str = "homer_50", windows: int = 2, reps: int = 10,
            modes: list[str] | None = None, deadline_s: float = 18 * 60) -> dict:
        import numpy as np
        import torch

        from flora_deploy.accel.determinism import flags

        t0 = time.monotonic()
        modes = list(modes or DEFAULT_MODES)
        out: dict[str, Any] = {"gpu": self.gpu, "load_s": self.load_s, "episode": episode,
                               "flags_at_start": flags(), "load_events": self.load_events,
                               "modes": {}}
        items = self._capture(episode, windows)
        out["captured"] = len(items)
        out["capture_s"] = round(time.monotonic() - t0, 1)
        refs: dict[int, dict] = {}
        rng = np.random.default_rng(0)
        for mode in modes:
            if time.monotonic() - t0 > deadline_s:
                out["modes"][mode] = {"skipped": "deadline"}
                continue
            enter, leave = self._set_mode(mode)
            enter()
            res: dict[str, Any] = {"flags": flags(), "inputs": {}}
            try:
                for i, item in enumerate(items):
                    runs = []
                    n = 1 if mode == "det_warn" else (reps if not mode.startswith("prefill")
                                                      else max(4, reps // 2))
                    for r in range(n):
                        if time.monotonic() - t0 > deadline_s:
                            break
                        if mode.startswith("prefill"):
                            runs.append(self._prefill(item))
                            continue
                        shift = self._dirty(rng) if mode == "dirty" else None
                        keep = mode == "baseline" and r == 0
                        g = self._generate(item, keep_logits=keep, ref=refs.get(i))
                        del shift
                        if keep:
                            refs[i] = {"steps": g["steps"], "logits": g["logits"],
                                       "text": g["text"]}
                        g.pop("logits")
                        runs.append(g)
                    res["inputs"][str(i)] = (self._summ_prefill(runs) if mode.startswith("prefill")
                                             else self._summ(runs, refs.get(i)))
            except Exception as e:  # noqa: BLE001
                res["error"] = f"{type(e).__name__}: {e}"[:600]
            res["warnings"] = leave()
            out["modes"][mode] = res
            print(f"mode {mode}: {json.dumps(res)}", flush=True)
        out["flags_at_end"] = flags()
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return out

    @staticmethod
    def _summ(runs: list[dict], ref: dict | None) -> dict:
        texts = [r["text"] for r in runs]
        tdig = [hashlib.sha256(str(t).encode()).hexdigest()[:12] for t in texts]
        sdig = [hashlib.sha256("|".join(r["steps"]).encode()).hexdigest()[:12] for r in runs]
        first_diff = []
        for r in runs:
            if ref is None:
                first_diff.append(None)
                continue
            k = next((j for j, (a, b) in enumerate(zip(r["steps"], ref["steps"], strict=False))
                      if a != b), None)
            if k is None and len(r["steps"]) != len(ref["steps"]):
                k = min(len(r["steps"]), len(ref["steps"]))
            first_diff.append(k)
        return {
            "n": len(runs),
            "input_digests": sorted({str(r["input_digest"])[:16] for r in runs}),
            "text_digests": tdig, "distinct_texts": len(set(tdig)),
            "text_equals_baseline_r0": [ref is not None and t == ref["text"] for t in texts],
            "logit_seq_digests": sdig, "distinct_logit_seqs": len(set(sdig)),
            "first_diff_step_vs_baseline_r0": first_diff,
            "diff_at_first_step": [r["diff"] for r in runs if r["diff"]][:6],
            "tokens": [r["tokens"] for r in runs],
            "complete": [r["complete"] for r in runs],
            "errors": [r["error"] for r in runs if r["error"]],
            "nv_elapsed_s": [r["nv_elapsed_s"] for r in runs],
            "mean_elapsed_s": round(sum(r["wall_s"] for r in runs) / max(1, len(runs)), 3),
        }

    @staticmethod
    def _summ_prefill(runs: list[dict]) -> dict:
        if not runs:
            return {"n": 0}
        order = runs[0]["order"]
        differing = [n for n in order
                     if len({r["hashes"].get(n) for r in runs}) > 1]
        return {"n": len(runs), "modules_hashed": len(runs[0]["hashes"]),
                "differing_modules": len(differing), "first_differing": differing[:8],
                "distinct_final": len({r["hashes"].get(order[-1]) for r in runs})}


@app.local_entrypoint()
def main(gpu: str = "L40S", episode: str = "homer_50", windows: int = 2, reps: int = 10,
         modes: str = "", deadline_min: float = 18.0, out: str = "", cpu: float = 8.0,
         memory_gib: int = 64) -> None:
    lab = Lab.with_options(gpu=gpu, cpu=cpu, memory=memory_gib * 1024)()
    res = lab.run.remote(episode=episode, windows=windows, reps=reps,
                         modes=[m for m in modes.split(",") if m] or None,
                         deadline_s=deadline_min * 60)
    text = json.dumps(res, indent=1, default=str)
    if out:
        Path(out).write_text(text)
    print(text[:20000])
    _ = GOLDEN_VOLUME
