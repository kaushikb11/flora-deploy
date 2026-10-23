"""Fake ``balanced`` segment / caption workers (``--fake-models``, local and on Modal).

The real ``workers_impl.segment.BalancedShardRunner`` (attempt markers, upstream exclusions,
concurrent episode coroutines, the real ``MicroBatcher``, part sinks, receipts) runs with a
fake per-episode job and a fake ``BatchEngine`` at the model boundary, exactly where the vLLM
TempSeg / Captioner engines sit on a GPU:

* inputs come from the real ``workers_impl.segment.InputResolver`` (index receipt -> cuts on
  local disk, NVIDIA profile, instruction);
* :class:`FakeSegmentJob` decodes the episode's videos, hashes the frames and sends one model
  call per episode through the micro-batcher; its segments are ``local.engines.fake_segments``
  of that hash (the same boundaries the fake ``segcap`` engine gives);
* :class:`FakeCaptionJob` reads the episode's **upstream segment records** with the real
  ``workers_impl.caption.UpstreamSegments`` (which caption shards depend on), sends one model
  call per segment and writes ``"<verb> <object> (task: <instruction>)"`` captions over those
  exact spans, so a caption record proves which segment record it came from.

Both run in two phases as the real jobs do (``workers_impl.segment.EpisodeGates``): decoding
and hashing under ``gates.prep()``, outside the batcher's episode, the model calls under
``gates.infer(batcher)``. Both are pure functions of the decoded frames and the instruction,
so a preempted shard, a bisection child or a re-run produce byte-identical records. Faults are
hooks: ``before(uid)`` runs before each episode (``local.faults.checkpoint`` locally,
``workers_app.FakeFaultHook`` on Modal) and ``rejects(uid)`` turns an episode into NVIDIA's input
rejection (``nv_failure``).

On Modal (``MODAL_TASK_ID`` set) the runner samples container resources as the real workers do
(``sample_resources``) and, before its first shard of each job, writes its startup facts
(``BalancedShardRunner.startup_facts``) to the job store (``jobs.containers.write_startup_facts``):
the real workers only print them, and the H0 rehearsal checks them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
from collections.abc import Callable
from typing import Any

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.jobs.types import EpisodeRef
from flora_deploy.local.engines import FAKE_ENGINE, _h, decode, fake_segments
from flora_deploy.local.fake_nvidia import FakeNvidiaInputError
from flora_deploy.records import Provenance
from flora_deploy.workers_impl.segment import (
    BalancedShardRunner,
    BalancedWorkerSettings,
    EpisodeInputs,
    EpisodeOutcome,
    ShardContext,
    gates_of,
    phase,
)

FAKE_BALANCED_MODELS = {"segment": {"tempseg": "fake"}, "caption": {"captioner": "fake"}}


def _never(_uid: str) -> bool:
    return False


def _nothing(_uid: str) -> None:
    return None


class FakeBalancedEngine:
    """``BatchEngine``: every request completes with one token and EOS (``finish_reason`` stop).
    Counts batches, so tests see that episodes shared rounds."""

    name = FAKE_ENGINE

    def __init__(self) -> None:
        self.batches: list[int] = []

    def generate_batch(self, requests, samplings) -> list[Completion]:
        self.batches.append(len(requests))
        return [Completion(key=r.key, token_ids=(1,), finish_reason="stop",
                           prompt_tokens=8 + len(r.prompt) // 4) for r in requests]


class _FakeJob:
    stage = ""
    nv = None
    phased = True  # the runner's classifier: fake rejections are NVIDIA's by origin (fake_nvidia)

    def __init__(self, inputs: Callable[[EpisodeRef], EpisodeInputs], *, gpu: str | None,
                 certificate: str, before: Callable[[str], Any] = _nothing,
                 rejects: Callable[[str], bool] = _never, before_in_thread: bool = False,
                 on_shard: Callable[[int], Any] | None = None):
        """``before_in_thread``: run ``before`` on a worker thread (the Modal hook, which may
        block until a SIGINT arrives while the event loop stays free to act on it); else on the
        loop's own thread (``local.faults`` reads a thread-local container context)."""
        self.inputs, self.gpu, self.certificate = inputs, gpu, certificate
        self.before, self.rejects = before, rejects
        self.before_in_thread = before_in_thread
        self.on_shard = on_shard  # once per shard start, with its episode count

    def provenance(self, inp: EpisodeInputs) -> Provenance:
        from flora_deploy import LOGIC_VERSION, __version__

        return Provenance(models=dict(FAKE_BALANCED_MODELS[self.stage]), engine=FAKE_ENGINE,
                          preset="balanced", config_hash="fake", logic_version=LOGIC_VERSION,
                          framework=f"flora-deploy {__version__}", gpu=self.gpu,
                          certificate=self.certificate, source=dict(inp.sources))

    def digest(self, inp: EpisodeInputs) -> tuple[str, float, list[int]]:
        decoded = [decode(p) for _role, p in sorted(inp.paths.items())]
        h = hashlib.sha256("".join(d.digest for d in decoded).encode()
                           + inp.instruction.encode() + inp.profile.encode()).hexdigest()
        return h, min(d.duration_s for d in decoded), [d.num_frames for d in decoded]

    async def _prepare(self, ref: EpisodeRef, ctx: ShardContext
                       ) -> tuple[EpisodeInputs, str, float, list[int]]:
        if self.on_shard is not None and not ctx.cache.get("fake_started"):
            ctx.cache["fake_started"] = True
            self.on_shard(len(ctx.shard.episodes))
        with phase("inputs"):
            inp = await asyncio.to_thread(self.inputs, ref)
        with phase("nvidia"):
            if self.before_in_thread:
                await asyncio.to_thread(self.before, ref.uid)
            else:
                self.before(ref.uid)
            if self.rejects(ref.uid):
                raise FakeNvidiaInputError(f"fake model rejected episode {ref.uid}")
            digest, duration, frames = await asyncio.to_thread(self.digest, inp)
        return inp, digest, duration, frames


class FakeSegmentJob(_FakeJob):
    stage = "segment"

    async def run(self, ref: EpisodeRef, batcher, ctx: ShardContext) -> EpisodeOutcome:
        from flora_deploy.stages import segment as S

        gates = gates_of(ctx)
        async with gates.prep():
            inp, digest, duration, frames = await self._prepare(ref, ctx)
        async with gates.infer(batcher):
            with phase("nvidia"):
                out = await batcher.generate(
                    VllmRequest(prompt=f"<tempseg {digest}>", videos=(), key=ref.uid),
                    Sampling(max_tokens=1024))
                if not out.ok:
                    raise RuntimeError(f"fake engine failed: {out.error}")
        rows = [{"start_s": s["start_s"], "end_s": s["end_s"],
                 "metadata": {"segmentation_provenance": "fake", "fake_digest": digest}}
                for s in fake_segments(digest, duration, "")]
        with phase("records"):
            rec = S.segment_record(rows, uid=ref.uid, dataset=ref.dataset,
                                   episode_id=ref.episode_id, provenance=self.provenance(inp))
            audit = {"uid": ref.uid, "stage": "segment", "profile": inp.profile,
                     "duration_s": round(duration, 6), "frames": frames, "fake": True,
                     "instruction": inp.instruction}
            return EpisodeOutcome([rec], [audit])


class FakeCaptionJob(_FakeJob):
    stage = "caption"

    def __init__(self, *args, segments=None, **kw):
        super().__init__(*args, **kw)
        from flora_deploy.workers_impl.caption import segments_for

        # pinned oracle segments when the env pins them (as the real worker), else upstream
        self.segments = segments or segments_for(os.environ)

    async def run(self, ref: EpisodeRef, batcher, ctx: ShardContext) -> EpisodeOutcome:
        from flora_deploy.local.engines import _OBJECTS, _VERBS
        from flora_deploy.stages import caption as C
        from flora_deploy.stages import segment as S

        gates = gates_of(ctx)
        async with gates.prep():
            with phase("inputs"):
                seg = await asyncio.to_thread(self.segments, ref, ctx)
                from flora_deploy.workers_impl.caption import PinnedSegments, pinned_rows

                pinned = isinstance(self.segments, PinnedSegments)
                spans = pinned_rows(seg) if pinned else S.rows_from_record(seg)
            inp, digest, _duration, _frames = await self._prepare(ref, ctx)
        async with gates.infer(batcher):
            with phase("nvidia"):
                outs = await batcher.generate_many(
                    [VllmRequest(prompt=f"<captioner {digest} {i}>", videos=(),
                                 key=f"{ref.uid}#{i}") for i in range(len(spans))],
                    [Sampling(max_tokens=512)] * len(spans))
                if not all(o.ok for o in outs):
                    raise RuntimeError("fake engine failed")
        rows = []
        for i, sp in enumerate(spans):
            caption = (f"{_VERBS[_h(digest, 'verb', i) % len(_VERBS)]} "
                       f"{_OBJECTS[_h(digest, 'obj', i) % len(_OBJECTS)]}")
            if inp.instruction:
                caption += f" (task: {inp.instruction})"
            rows.append({"start_s": sp["start_s"], "end_s": sp["end_s"], "caption": caption,
                         "status": "complete", "segmentation_provenance": "fake"})
        with phase("records"):
            prov = self.provenance(inp)
            if pinned:
                prov = dataclasses.replace(prov, segments_from=self.segments.segments_from)
            rec = C.caption_record(rows, uid=ref.uid, dataset=ref.dataset,
                                   episode_id=ref.episode_id, provenance=prov)
            audit = {"uid": ref.uid, "stage": "caption", "segments": len(rows), "fake": True,
                     "upstream_engine_hash": seg.provenance.engine_hash,
                     "instruction": inp.instruction}
            return EpisodeOutcome([rec], [audit])


def fake_balanced_runner(kind: str, store, *, media_root: str | None, gpu: str | None,
                         certificate: str, default_task: str | None = None,
                         local_dir: str = "/tmp/flora-balanced",
                         container_id: str | None = None,
                         before: Callable[[str], Any] = _nothing,
                         rejects: Callable[[str], bool] = _never,
                         before_in_thread: bool = False,
                         on_shard: Callable[[int], Any] | None = None,
                         call_id: Callable[[], str | None] = lambda: "",
                         fatal: Callable[[BaseException], None] | None = None,
                         stamp: Any = None, max_inflight_episodes: int = 4,
                         prefetch_episodes: int | None = None,
                         flush_every: int = 1, sample_resources: bool | None = None
                         ) -> BalancedShardRunner:
    """The real ``BalancedShardRunner`` for ``kind`` (``segment`` | ``caption``) with the fake
    job and engine (module docstring). ``stamp``: a ``workers_impl.certstamp.CertStamp``.
    ``sample_resources`` (default: on Modal only) also writes the startup facts per job."""
    from flora_deploy.workers_impl.balanced import StampedJob
    from flora_deploy.workers_impl.segment import InputResolver

    if kind not in FAKE_BALANCED_MODELS:
        raise ValueError(f"no fake balanced job for kind {kind!r}")
    inputs = InputResolver(store, media_root=media_root, local_dir=local_dir,
                           default_task=default_task)
    cls = FakeSegmentJob if kind == "segment" else FakeCaptionJob
    job: Any = cls(inputs, gpu=gpu, certificate=certificate, before=before, rejects=rejects,
                   before_in_thread=before_in_thread, on_shard=on_shard)
    if stamp is not None:
        job = StampedJob(job, stamp)
    kw: dict[str, Any] = {} if container_id is None else {"container_id": container_id}
    on_modal = bool(os.environ.get("MODAL_TASK_ID")) if sample_resources is None \
        else bool(sample_resources)
    settings = BalancedWorkerSettings(max_inflight_episodes=max_inflight_episodes,
                                      prefetch_episodes=prefetch_episodes or
                                      max_inflight_episodes,
                                      flush_every=flush_every, max_wait_s=0.05,
                                      local_dir=local_dir, gpu=gpu,
                                      sample_resources=on_modal, **kw)
    cls_ = FactsShardRunner if on_modal else BalancedShardRunner
    return cls_(store, FakeBalancedEngine(), job, settings, call_id=call_id, fatal=fatal)


class FactsShardRunner(BalancedShardRunner):
    """``BalancedShardRunner`` that writes its startup facts before its first shard of each job
    (module docstring). Diagnostics only: a failed write never fails the shard."""

    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        written = self.__dict__.setdefault("_facts_jobs", set())
        if job_id not in written:
            written.add(job_id)
            try:
                from flora_deploy.jobs.containers import write_startup_facts

                write_startup_facts(self.store, job_id, self.container_id, self.job.stage,
                                    self.startup_facts())
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                print(f"startup facts not written: {exc!r}", flush=True)
        return super().run_shard(job_id, shard_id, config_hash, token=token)
