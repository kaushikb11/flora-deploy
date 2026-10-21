"""``accel.gpu_thread`` (Perf-1): NVIDIA's model calls on one persistent thread."""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from flora_deploy.accel import gpu_thread as g
from flora_deploy.observe.timers import Patcher


class FakeTorch:
    def __init__(self, grad=True, inference=False, autocast=False):
        self.grad, self.inference, self.autocast = grad, inference, autocast

    def is_grad_enabled(self):
        return self.grad

    def is_inference_mode_enabled(self):
        return self.inference

    def is_autocast_enabled(self, device):
        assert device == "cuda"
        return self.autocast


def _fake_nv(record: list):
    def generate(model, proc, inputs, budget):
        record.append(threading.get_ident())
        if budget < 0:
            raise ValueError("bad budget")
        return f"{model}:{inputs['x']}:{budget}"

    return SimpleNamespace(pipeline=SimpleNamespace(generate=generate))


def test_all_calls_run_on_one_persistent_thread_from_many_callers():
    seen: list[int] = []
    nv = _fake_nv(seen)
    original = nv.pipeline.generate
    with Patcher() as patcher:
        gt = g.install(nv, patcher, g.GpuThread(torch=FakeTorch()))
        # like NVIDIA: asyncio.to_thread workers and episode-pool threads, each short-lived
        for _ in range(3):
            with ThreadPoolExecutor(4) as pool:
                outs = list(pool.map(lambda i: nv.pipeline.generate("m", None, {"x": i}, 7),
                                     range(8)))
            assert outs == [f"m:{i}:7" for i in range(8)]
        assert set(seen) == {gt.ident}
        assert gt.ident != threading.get_ident()
        assert gt.provenance()["calls"] == 24
    assert nv.pipeline.generate is original  # restore gives NVIDIA's attribute back
    gt.close()


def test_results_and_exceptions_pass_through_unchanged():
    nv = _fake_nv([])
    with Patcher() as patcher:
        gt = g.install(nv, patcher, g.GpuThread(torch=FakeTorch()))
        assert nv.pipeline.generate("a", None, {"x": 1}, 2) == "a:1:2"
        with pytest.raises(ValueError, match="bad budget"):
            nv.pipeline.generate("a", None, {"x": 1}, -1)
    gt.close()


def test_context_variables_follow_the_call():
    var: contextvars.ContextVar[str] = contextvars.ContextVar("route", default="none")
    gt = g.GpuThread(torch=FakeTorch())

    def caller(tag):
        var.set(tag)
        return gt.run(var.get)

    with ThreadPoolExecutor(3) as pool:
        assert list(pool.map(caller, ["a", "b", "c"])) == ["a", "b", "c"]
    assert var.get() == "none"
    gt.close()


def test_reentrant_call_runs_inline():
    gt = g.GpuThread(torch=FakeTorch())
    assert gt.run(lambda: gt.run(threading.get_ident)) == gt.ident
    assert gt.provenance()["inline_calls"] == 1
    gt.close()


@pytest.mark.parametrize("modes", [dict(grad=False), dict(inference=True), dict(autocast=True)])
def test_refuses_caller_thread_local_torch_modes(modes):
    gt = g.GpuThread(torch=FakeTorch(**modes))
    with pytest.raises(g.GpuThreadError, match="thread-local torch modes"):
        gt.run(lambda: 1)
    gt.close()


def test_double_install_refused_and_composes_with_other_wrappers():
    import functools

    nv = _fake_nv([])
    with Patcher() as patcher:
        gt = g.install(nv, patcher, g.GpuThread(torch=FakeTorch()))
        inner = nv.pipeline.generate

        @functools.wraps(inner)
        def locked(*a, **k):  # e.g. stages.reference.locked_generate on top
            return inner(*a, **k)

        nv.pipeline.generate = locked
        with pytest.raises(g.GpuThreadError, match="already"):
            g.install(nv, patcher)
        nv.pipeline.generate = inner
    gt.close()


def test_closed_thread_refuses_calls():
    gt = g.GpuThread(torch=FakeTorch())
    gt.close()
    with pytest.raises(g.GpuThreadError, match="closed"):
        gt.run(lambda: 1)
    assert g.disabled_provenance() == {"schema": g.SCHEMA, "enabled": False}


def test_real_torch_default_modes_pass():
    torch = pytest.importorskip("torch")
    gt = g.GpuThread()
    assert gt.run(lambda: torch.ones(2).sum().item()) == 2.0
    with torch.no_grad(), pytest.raises(g.GpuThreadError):
        gt.run(lambda: 1)
    gt.close()
