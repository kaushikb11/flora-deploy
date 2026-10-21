"""Workers and stores that must pass the conformance suites.

A new ``ShardWorker`` (RewardWorker, SegcapWorker, ...) adds a ``WorkerSpec`` here whose factory
builds it over the suite's store with its engine faked to follow ``WorkerEnv.script``; see
``worker_suite.py``. A new ``Store`` adds a factory to ``STORES`` in ``store_suite.py``.
"""

from __future__ import annotations

from tests.conformance.reference_worker import fake_spec
from tests.conformance.worker_suite import WorkerSpec


def _canary_spec() -> WorkerSpec:
    from flora_deploy.modal_app.canary_app import conformance_spec

    return conformance_spec()


def _reward_specs() -> list[WorkerSpec]:
    """The real ``RewardShardRunner`` (``workers_impl.reward``) over the suite's store, with
    ``stages.reward`` driven by the fake NVIDIA core / LLM of ``tests/unit/test_reward_stage``
    (a fake ``vllm`` module is installed only for the duration of each call). The fake LLM
    answers garbage for ``script.nv_fail`` episodes (NVIDIA rejects it: terminal parse / dense
    stitch) and raises ``KeyboardInterrupt`` on the request of ``script.interrupt_before``.
    Rounds hold one episode
    (``nvidia_group=1``), so every finished episode is flushed before the next one starts."""
    import sys
    import types
    from unittest import mock

    from flora_deploy.stages import reward as R
    from flora_deploy.workers_impl.reward import (
        RewardEngine,
        RewardShardRunner,
        RewardWorkerSettings,
    )
    from tests.unit.test_reward_stage import FakeLLM, Params, make_nv

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = type("SamplingParams", (Params,), {})
    sp = types.ModuleType("vllm.sampling_params")
    sp.StructuredOutputsParams = type("StructuredOutputsParams", (Params,), {})

    def tag(uid: str) -> str:
        return f"<{uid}>"

    class ScriptedLLM(FakeLLM):
        def __init__(self, script):
            super().__init__(answers={tag(u): "no score STITCH_FAIL" for u in script.nv_fail})
            self.interrupt = tag(script.interrupt_before) if script.interrupt_before else None

        def generate(self, requests, params, use_tqdm=True):
            if self.interrupt and any(self.interrupt in r["prompt"] for r in requests):
                raise KeyboardInterrupt("SIGINT")
            return super().generate(requests, params, use_tqdm)

    class Worker:
        def __init__(self, env, kind):
            settings = RewardWorkerSettings(
                decoder="inline", container_id=env.container_id, gpu="H100",
                models={"rewarder": "test"}, limits=R.RoundLimits(nvidia_group=1),
                terminal_mode="terminal")
            self.runner = RewardShardRunner(
                env.store, RewardEngine(ScriptedLLM(env.script), "PROC"), make_nv(), settings,
                call_id=lambda: env.call_id, clock=env.clock,
                episodes=lambda ref: R.Episode(ref.episode_id, f"task {tag(ref.uid)}",
                                               ("/v/vvvv",), ref.dataset, ref.envelope_ok))

        def run_shard(self, job_id, shard_id, config_hash, token=""):
            with mock.patch.dict(sys.modules, {"vllm": vllm, "vllm.sampling_params": sp}):
                return self.runner.run_shard(job_id, shard_id, config_hash, token)

        def on_exit(self):
            self.runner.on_exit()

    return [WorkerSpec(f"reward-worker[{kind}]", kind, lambda env, k=kind: Worker(env, k))
            for kind in ("reward.terminal", "reward.dense")]


def segcap_fake_engine(script, *, concurrency: int = 2, raises=None, sleeps=None):
    """The production ``FastExactEngine`` composite over the fake NVIDIA SEGCAP family of
    ``tests/unit/test_reference_concurrency`` (NVIDIA's call structure: ``api`` entry point,
    ``pipeline.Session``, ``asyncio.to_thread`` predictor, ``FLORA_RUNTIME`` events): resident
    ``CachedSession``, ``install_nvidia_entry`` + GPU lock (``generate_output`` lines on) +
    ``accel.gpu_thread``. ``script.interrupt_before`` raises ``KeyboardInterrupt`` inside NVIDIA's
    ``pipeline.annotate`` for that episode (its main camera path names the uid). ``raises``
    (uid substring -> exception factory) makes NVIDIA's ``annotate`` raise for that episode;
    ``sleeps`` (uid substring -> seconds) makes it take that long first."""
    import time

    from flora_deploy.accel import gpu_thread
    from flora_deploy.nv.session import CachedSession
    from flora_deploy.observe import Patcher, TimedLock
    from flora_deploy.stages import reference as ref
    from flora_deploy.workers_impl.segcap import FastExactEngine
    from tests.unit.test_reference_concurrency import FakeNvSession, GpuProbe, fake_nv

    nv = fake_nv(GpuProbe())
    original = nv.pipeline.annotate

    def annotate(videos, *a, **k):
        main = str(videos["main"].path)
        if script.interrupt_before and script.interrupt_before in main:
            raise KeyboardInterrupt("SIGINT")
        for key, seconds in (sleeps or {}).items():
            if key in main:
                time.sleep(seconds)
        for key, make in (raises or {}).items():
            if key in main:
                raise make()
        return original(videos, *a, **k)

    nv.pipeline.annotate = annotate
    nv.audits = []
    nv.audit_report = lambda: nv.audits.append(1) or {"ok": True, "audits": len(nv.audits)}
    patcher, lock = Patcher(), TimedLock(ref.GPU_LOCK_NAME)
    # one Patcher patches an attribute once: the GPU thread has its own (innermost) layer
    gt = gpu_thread.install(nv, Patcher(), gpu_thread.GpuThread(torch=None))
    ref.install_gpu_lock(nv, lock, patcher, emit_output=True)
    ref.install_nvidia_entry(nv, patcher, request_lock=False)
    return FastExactEngine(nv, CachedSession(FakeNvSession), gpu_lock=lock,
                           clones=ref.ProcessorClones(), gpu_thread=gt,
                           concurrency=concurrency, profile=ref.ExactProfile().provenance(),
                           image_fingerprint="sha256:fake")


def _segcap_spec() -> WorkerSpec:
    """The real ``SegcapShardRunner`` (``workers_impl.segcap``) over the suite's store, running
    the real fast-exact composite (``segcap_fake_engine``) with two episode threads. An
    ``script.nv_fail`` episode asks NVIDIA for the 32-second profile on one camera, which
    NVIDIA's ``_inputs`` rejects (``InputError``)."""
    from flora_deploy.workers_impl.segcap import (
        SegcapInputs,
        SegcapShardRunner,
        SegcapWorkerSettings,
    )

    class Worker:
        def __init__(self, env):
            def inputs(ref):
                return SegcapInputs(uid=ref.uid, profile="human",
                                    long_windows=ref.uid in env.script.nv_fail,
                                    instruction="", videos={"main": f"/v/{ref.uid}.mp4"},
                                    sources={}, duration_s=ref.duration_s)

            settings = SegcapWorkerSettings(container_id=env.container_id, gpu="H200",
                                            models={"tempseg": "test"})
            self.runner = SegcapShardRunner(env.store, segcap_fake_engine(env.script), settings,
                                            call_id=lambda: env.call_id, clock=env.clock,
                                            inputs=inputs)

        def run_shard(self, job_id, shard_id, config_hash, token=""):
            return self.runner.run_shard(job_id, shard_id, config_hash, token)

        def on_exit(self):
            self.runner.on_exit()

    return WorkerSpec("segcap-worker", "segcap", Worker)


WORKERS: list[WorkerSpec] = [
    fake_spec("segcap"),
    fake_spec("reward.terminal"),
    _canary_spec(),
    *_reward_specs(),
    _segcap_spec(),
]
