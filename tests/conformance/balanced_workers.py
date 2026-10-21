"""Conformance factories for the balanced segment / caption workers (``workers_impl.segment``,
``workers_impl.caption``).

The real ``BalancedShardRunner`` and the real ``MicroBatcher`` run over the suite's store; only
the per-episode job and the engine are scripted (``WorkerEnv.script``):

* the job sends one model call per episode through the micro-batcher and turns the answer into
  a real ``segment`` / ``caption`` record (``stages.segment.segment_record`` /
  ``stages.caption.caption_record``); for ``script.nv_fail`` episodes it raises a ``ValueError``
  the way NVIDIA's code does (``media.InputError``), which the runner must classify as
  ``nv_failure``;
* the engine raises ``KeyboardInterrupt`` (SIGINT) inside ``generate_batch`` when it sees
  ``script.interrupt_before``'s call, i.e. from the batcher's worker thread, which is how a real
  preemption reaches a running ``llm.generate``.

Episodes run one at a time (``max_inflight_episodes=1``, flush after every episode), so every
episode before the interrupted one is finished and flushed, as the preemption check expects.
"""

from __future__ import annotations

import tempfile

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.records import Provenance
from flora_deploy.stages import caption as C
from flora_deploy.stages import segment as S
from flora_deploy.workers_impl.segment import (
    BalancedShardRunner,
    BalancedWorkerSettings,
    EpisodeOutcome,
)
from tests.conformance.worker_suite import WorkerSpec


class ScriptedEngine:
    name = "scripted-vllm"

    def __init__(self, interrupt: str | None):
        self.interrupt = interrupt
        self.batches: list[list[str]] = []

    def generate_batch(self, requests, samplings):
        keys = [r.key for r in requests]
        self.batches.append(keys)
        if self.interrupt is not None and self.interrupt in keys:
            raise KeyboardInterrupt("SIGINT")
        return [Completion(key=k, token_ids=(1, 2), finish_reason="stop", prompt_tokens=3)
                for k in keys]


class ScriptedJob:
    nv = None

    def __init__(self, stage: str, nv_fail):
        self.stage = stage
        self.nv_fail = set(nv_fail)

    async def run(self, ref, batcher, ctx):
        out = await batcher.generate(VllmRequest(prompt=f"<{ref.uid}>", videos=(), key=ref.uid),
                                     Sampling(max_tokens=8))
        assert out.ok
        if ref.uid in self.nv_fail:
            raise ValueError("NVIDIA rejected the input (InputError)")
        prov = Provenance(models={"tempseg": "test"}, engine="vllm-0.24.0-offline",
                          preset="balanced", config_hash="stage", logic_version="test",
                          framework="flora-deploy test")
        if self.stage == "segment":
            rec = S.segment_record([dict(start_s=0.0, end_s=1.0, metadata={})], uid=ref.uid,
                                   dataset=ref.dataset, episode_id=ref.episode_id,
                                   provenance=prov)
        else:
            rec = C.caption_record([dict(start_s=0.0, end_s=1.0, caption="c", status="complete",
                                         segmentation_provenance="fine")], uid=ref.uid,
                                   dataset=ref.dataset, episode_id=ref.episode_id,
                                   provenance=prov)
        return EpisodeOutcome([rec], [{"uid": ref.uid, "stage": self.stage}])


class Worker:
    def __init__(self, env, kind: str):
        settings = BalancedWorkerSettings(max_inflight_episodes=1, flush_every=1, max_wait_s=5.0,
                                          container_id=env.container_id, gpu="L40S",
                                          local_dir=tempfile.mkdtemp(prefix="flora-balanced-"))
        self.runner = BalancedShardRunner(
            env.store, ScriptedEngine(env.script.interrupt_before),
            ScriptedJob(kind, env.script.nv_fail), settings, call_id=lambda: env.call_id,
            clock=env.clock)

    def run_shard(self, job_id, shard_id, config_hash, token=""):
        return self.runner.run_shard(job_id, shard_id, config_hash, token)

    def on_exit(self):
        self.runner.on_exit()


def balanced_specs() -> list[WorkerSpec]:
    return [WorkerSpec(f"balanced-worker[{kind}]", kind, lambda env, k=kind: Worker(env, k))
            for kind in ("segment", "caption")]
