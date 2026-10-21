"""Run every NVIDIA model call on one long-lived thread (Perf-1, ``docs/perf-1.md``). Opt-in.

Why: on H200 PyTorch's SDPA picks the cuDNN attention backend, and torch keeps the cuDNN
attention graphs it builds in a **thread-local** cache keyed by the exact shapes
(``aten/src/ATen/native/cudnn/MHA.cpp``: ``thread_local MHAGraphCache``; ``s_q``, ``s_kv``,
strides, ...). Every decode step has a new KV length, so a thread that has not seen that length
builds a new graph. NVIDIA's ``pipeline.segment`` calls ``generate`` from
``asyncio.to_thread`` workers of a fresh event loop per call, and our episode pool adds more
threads, so the cache is almost always cold. With torch 2.11.0+cu128 (NVIDIA's validated stack)
a build costs ~0.4 s, which makes a decode step ~450 ms instead of ~65 ms (measured, Perf-1).

This module changes only *which OS thread* calls NVIDIA's unmodified ``generate``: the same
function, the same tensors, the same backend selection, the same cuDNN graph keys, so each
graph is built the way it would be built on the caller's thread. The cache then persists for the
container's lifetime and each (shape) graph is built once.

Usage (in ``@modal.enter``, after NVIDIA's code is linked; composes with ``install_gpu_lock``)::

    from flora_deploy.accel import gpu_thread
    gt = gpu_thread.install(nv, patcher)          # swaps nv.pipeline.generate
    record["gpu_thread"] = gt.provenance()

``patcher`` is any object with ``patch(obj, name, wrap)`` (``observe.Patcher``); restoring it
restores NVIDIA's attribute. Context variables (stdout event routing, observe recorders,
session binding) are carried into the GPU thread with ``contextvars.copy_context``; exceptions
propagate unchanged. A call made from the GPU thread itself runs inline (no deadlock).

Guard: the caller's thread-local torch modes (autocast, inference mode, grad mode) do not cross
threads. NVIDIA's call sites set none of them (``models.generate`` enters ``inference_mode``
itself; TempSeg's autocast is inside ``model.generate``), so ``run`` refuses a call whose
caller has any non-default mode instead of silently dropping it.
"""

from __future__ import annotations

import contextvars
import functools
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

SCHEMA = "flora_gpu_thread_v1"
MARK = "__flora_gpu_thread__"
THREAD_NAME = "flora-gpu"


class GpuThreadError(RuntimeError):
    """A call cannot be moved to the GPU thread without changing what it computes."""


def _caller_modes(torch: Any) -> dict[str, bool]:
    if torch is None:
        return {}
    modes = {"grad_enabled": bool(torch.is_grad_enabled()),
             "inference_mode": bool(torch.is_inference_mode_enabled())}
    is_autocast = getattr(torch, "is_autocast_enabled", None)
    if is_autocast is not None:
        try:
            modes["autocast_cuda"] = bool(is_autocast("cuda"))
        except TypeError:  # older signature without a device type
            modes["autocast_cuda"] = bool(is_autocast())
    return modes


DEFAULT_MODES = {"grad_enabled": True, "inference_mode": False, "autocast_cuda": False}


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - CPU-only envs without torch
        return None
    return torch


class GpuThread:
    """A single persistent worker thread; ``run(fn, ...)`` executes ``fn`` on it and waits."""

    def __init__(self, name: str = THREAD_NAME, *, torch: Any = "auto"):
        self.name = name
        self._torch = _torch() if torch == "auto" else torch
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._ident: int | None = None
        self._lock = threading.Lock()
        self.calls = 0
        self.inline_calls = 0
        self.busy_s = 0.0
        self._closed = False
        # Start the thread now so its identity is known (and stable) before the first call.
        self._ident = self._pool.submit(threading.get_ident).result()

    @property
    def ident(self) -> int | None:
        return self._ident

    def _check_modes(self) -> None:
        modes = _caller_modes(self._torch)
        bad = {k: v for k, v in modes.items() if DEFAULT_MODES.get(k, v) != v}
        if bad:
            raise GpuThreadError(
                f"caller has thread-local torch modes {bad} that would not reach the GPU thread")

    def run(self, fn: Callable, *args, **kwargs):
        if threading.get_ident() == self._ident:
            with self._lock:
                self.inline_calls += 1
            return fn(*args, **kwargs)
        if self._closed:
            raise GpuThreadError("GPU thread is closed")
        self._check_modes()
        ctx = contextvars.copy_context()

        def job():
            t = time.monotonic()
            try:
                return ctx.run(fn, *args, **kwargs)
            finally:
                with self._lock:
                    self.calls += 1
                    self.busy_s += time.monotonic() - t

        return self._pool.submit(job).result()

    def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=True)

    def provenance(self) -> dict:
        return {"schema": SCHEMA, "enabled": True, "thread": self.name, "ident": self._ident,
                "calls": self.calls, "inline_calls": self.inline_calls,
                "busy_s": round(self.busy_s, 3)}


def pinned(generate: Callable, gpu: GpuThread) -> Callable:
    """``generate`` executed on ``gpu``'s thread (same arguments, same return, same errors)."""

    @functools.wraps(generate)
    def generate_on_gpu_thread(*args, **kwargs):
        return gpu.run(generate, *args, **kwargs)

    setattr(generate_on_gpu_thread, MARK, gpu)
    return generate_on_gpu_thread


def install(nv, patcher, gpu: GpuThread | None = None) -> GpuThread:
    """Route NVIDIA's model calls (``nv.pipeline.generate``, the name ``pipeline.py`` imported
    from ``.models``) through one persistent thread. Refuses a second install."""
    if _find_mark(nv.pipeline.generate) is not None:
        raise GpuThreadError("NVIDIA generate() already runs on a GPU thread")
    gpu = gpu or GpuThread()
    patcher.patch(nv.pipeline, "generate", lambda orig: pinned(orig, gpu))
    return gpu


def _find_mark(fn: Any) -> GpuThread | None:
    seen = 0
    while fn is not None and seen < 16:
        mark = getattr(fn, MARK, None)
        if mark is not None:
            return mark
        fn = getattr(fn, "__wrapped__", None)
        seen += 1
    return None


def disabled_provenance() -> dict:
    return {"schema": SCHEMA, "enabled": False}
