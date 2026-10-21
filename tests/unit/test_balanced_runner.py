"""BalancedShardRunner error taxonomy (review 6 M1): segcap's classifier, phase by phase."""

from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace

import pytest

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.stages import segment as S
from flora_deploy.workers_impl.reward import ShardInfraError
from flora_deploy.workers_impl.segment import (
    BalancedShardRunner,
    BalancedWorkerSettings,
    EpisodeInputs,
    SegmentEpisodeJob,
    StageInfo,
)
from tests.conformance.worker_suite import WorkerSpec, World
from tests.fakes import fake_nvidia


class NullEngine:
    name = "null"

    def generate_batch(self, requests, samplings):
        return [Completion(key=r.key, token_ids=(1,), finish_reason="stop", prompt_tokens=1)
                for r in requests]


def inputs_ok(ref):
    return EpisodeInputs(uid=ref.uid, dataset=ref.dataset, episode_id=ref.episode_id,
                         paths={"main": "/x.mp4"}, profile="human", yam32=False,
                         instruction="")


class FakeStage:
    nv = None

    def __init__(self, exc_for):
        self.exc_for = exc_for

    async def segment(self, batcher, ep):
        out = await batcher.generate(VllmRequest(prompt="<p>", videos=(), key=ep.uid),
                                     Sampling(max_tokens=1))
        assert out.ok
        exc = self.exc_for(ep.uid)
        if exc is not None:
            raise exc
        return SimpleNamespace(rows=[dict(start_s=0.0, end_s=1.0, metadata={})], calls=[],
                               apredict_rounds=1, stats=SimpleNamespace(as_json=lambda: {}))


def world(tmp_path, monkeypatch, *, inputs=inputs_ok, exc_for=lambda uid: None, fatal=None):
    monkeypatch.setattr(S, "prepare_videos", lambda nv, paths, profile, **kw: SimpleNamespace(
        uid=next(iter(paths.values())), pool="human", profile=profile, duration_s=5.0))

    def factory(env):
        job = SegmentEpisodeJob(FakeStage(exc_for), StageInfo("vllm-test"),
                                lambda ref: _with_uid(inputs(ref), ref), budget=1024)
        settings = BalancedWorkerSettings(max_inflight_episodes=1, flush_every=1,
                                          max_wait_s=0.01, container_id=env.container_id,
                                          local_dir=tempfile.mkdtemp(dir=tmp_path))
        kw = {"fatal": fatal} if fatal is not None else {}
        return BalancedShardRunner(env.store, NullEngine(), job, settings, clock=env.clock,
                                   **kw)

    return World(WorkerSpec("balanced-test", "segment", factory), tmp_path / "w", n=3)


def _with_uid(inp, ref):
    # prepare_videos is faked: carry the uid through the paths so the stage can see it
    return EpisodeInputs(**{**inp.__dict__, "paths": {"main": ref.uid}})


def test_input_plumbing_value_error_is_infra_not_excused(tmp_path, monkeypatch):
    def inputs(ref):
        raise ValueError(f"episode {ref.uid!r}: no NVIDIA profile for pool 'z'")

    w = world(tmp_path, monkeypatch, inputs=inputs)
    with pytest.raises(ShardInfraError):
        w.run("ta-1")
    assert w.receipt() is None
    assert {e.kind for e in w.errors()} == {"infra"}
    assert {e.diagnostics["phase"] for e in w.errors()} == {"inputs"}


def test_truncated_index_receipt_is_infra(tmp_path, monkeypatch):
    def inputs(ref):
        return json.loads("{")

    w = world(tmp_path, monkeypatch, inputs=inputs)
    with pytest.raises(ShardInfraError):
        w.run("ta-1")
    assert {e.kind for e in w.errors()} == {"infra"}


def test_nvidia_builtin_runtime_error_rejection_is_terminal(tmp_path, monkeypatch):
    bad = []
    def exc_for(uid):
        if uid not in bad:
            return None
        try:  # raised by NVIDIA's (fake) code: a builtin RuntimeError from its frame
            fake_nvidia.nvidia_raise(RuntimeError(
                "The pipeline returned an invalid caption interval."))
        except RuntimeError as e:
            return e

    w = world(tmp_path, monkeypatch, exc_for=exc_for)
    bad.append(w.uids[1])
    rec = w.run("ta-1")
    assert rec is not None and rec.episodes_failed == (w.uids[1],)
    assert [e.kind for e in w.errors()] == ["nv_failure"]


def test_records_phase_value_error_is_infra(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise ValueError("Out of range float values are not JSON compliant")

    monkeypatch.setattr(S, "segment_record", boom)
    w = world(tmp_path, monkeypatch)
    with pytest.raises(ShardInfraError):
        w.run("ta-1")
    assert {e.kind for e in w.errors()} == {"infra"}
    assert {e.diagnostics["phase"] for e in w.errors()} == {"records"}


def test_cuda_fault_fails_fast_and_exits_container(tmp_path, monkeypatch):
    calls, seen = [], []

    def exc_for(uid):
        seen.append(uid)
        return RuntimeError("CUDA error: an illegal memory access was encountered")

    w = world(tmp_path, monkeypatch, exc_for=exc_for, fatal=calls.append)
    with pytest.raises(ShardInfraError):
        w.run("ta-1")
    assert len(calls) == 1
    assert len(seen) == 1, "episodes kept running on a dead engine"
    m = w.marker("ta-1")
    assert m is not None and m.ended_utc and not m.preempted


@pytest.mark.parametrize("exc,phase,src,want", [
    (RuntimeError("No common caption input geometry fits"), "nvidia", "nv", "nv_failure"),
    (RuntimeError("The pipeline returned an invalid caption interval."), "nvidia", "nv",
     "nv_failure"),
    (RuntimeError("The pipeline returned an invalid caption interval."), "nvidia", "ours",
     "python"),
    (fake_nvidia.InputError("NVIDIA InputError"), "nvidia", "ours", "nv_failure"),
    (ValueError("our ValueError"), "nvidia", "ours", "python"),
    (ValueError("no NVIDIA profile"), "inputs", "ours", "infra"),
    (ValueError("nan"), "records", "nv", "python"),
    (RuntimeError("CUDA error: device-side assert triggered"), "nvidia", "ours", "cuda_fatal"),
    (RuntimeError("CUDA out of memory"), "nvidia", "ours", "oom"),
    (KeyError("x"), "nvidia", "ours", "python"),
    (KeyError("x"), "nvidia", "nv", "nv_failure"),
])
def test_classify_categories(exc, phase, src, want):
    """By phase, then by origin (``nv.errors``): ``src`` is the frame that raised ``exc``,
    NVIDIA's (fake) code or this test's."""
    from flora_deploy.workers_impl.segment import _classify

    try:
        if src == "nv":
            fake_nvidia.nvidia_raise(exc)
        raise exc
    except BaseException as e:  # noqa: BLE001
        e.flora_phase = phase
        assert _classify(e) == want


def test_sigint_in_main_thread_marks_preempted_promptly(tmp_path, monkeypatch):
    """Review 6 M4: real SIGINT lands in the main thread (``asyncio.run`` turns it into a
    cancel) while ``llm.generate`` is running in the batcher's thread. The preempt mark must be
    written without waiting for that generate to finish."""
    import _thread
    import signal
    import time

    from tests.unit.test_engines_microbatch import SlowEngine

    eng = SlowEngine(hold=6.0, on_start=lambda: _thread.interrupt_main(signal.SIGINT))
    w = world(tmp_path, monkeypatch)
    w.spec = WorkerSpec(w.spec.name, w.spec.kind, _swap_engine(w.spec.factory, eng))
    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        w.run("ta-1")
    took = time.perf_counter() - t0
    m = w.marker("ta-1")
    assert m is not None and m.preempted
    assert took < 3.0, f"preempt mark took {took:.1f}s (waited for the in-flight generate)"
    eng.release.set()


def _swap_engine(factory, eng):
    def f(env):
        runner = factory(env)
        runner.engine = eng
        return runner
    return f
