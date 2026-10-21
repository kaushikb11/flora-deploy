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


def _balanced_specs() -> list[WorkerSpec]:
    """The real balanced ``BalancedShardRunner`` + ``MicroBatcher`` (``workers_impl.segment`` /
    ``workers_impl.caption``) with a scripted per-episode job and engine
    (``tests/conformance/balanced_workers.py``)."""
    from tests.conformance.balanced_workers import balanced_specs

    return balanced_specs()


WORKERS: list[WorkerSpec] = [
    fake_spec("segcap"),
    fake_spec("reward.terminal"),
    _canary_spec(),
    *_reward_specs(),
    *_balanced_specs(),
]
