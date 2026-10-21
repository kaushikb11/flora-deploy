"""Balanced CaptionWorker core (P1b-1): ``workers.ShardWorker`` around ``stages.caption``.

Same runner as the segment worker (``workers_impl.segment.BalancedShardRunner``); the per-episode
job reads the episode's ``segment`` record from the upstream shard's parts, turns it back into
NVIDIA's ``pipeline.segment`` rows (``stages.segment.rows_from_record``: span plus NVIDIA's full
segment metadata), probes the cameras with NVIDIA's code and runs
``stages.caption.CaptionStage.caption``.

Upstream records (``UpstreamSegments``): every part under the shard's ``depends_on`` segment
shards and under their single-episode bisection children (``attempts.child_shard_id``), filtered
to this config (``JobFiles.accepts("segment")``: the segment stage hash or the job hash); read
once per shard. An episode without one is an ``infra`` ``EpisodeError`` (the orchestrator
chains caption shards on segment receipts, so this means a lost or foreign part). Upstream-poisoned episodes never get here (runner step 3).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.types import EpisodeRef, Shard
from flora_deploy.records import StageRecord
from flora_deploy.workers_impl.segment import (
    EpisodeInputs,
    EpisodeOutcome,
    ShardContext,
    StageInfo,
)


class MissingUpstream(RuntimeError):
    """No segment record for this episode in the upstream shards' parts."""


class UpstreamSegments:
    """uid -> ``segment`` StageRecord of this job, from the caption shard's upstream parts."""

    def __call__(self, ref: EpisodeRef, ctx: ShardContext) -> StageRecord:
        cache = ctx.cache.get("upstream")
        if cache is None:
            cache = ctx.cache["upstream"] = self.load(ctx.jf, ctx.shard)
        rec = cache.get(ref.uid)
        if rec is None:
            raise MissingUpstream(f"episode {ref.uid!r}: no segment record upstream of "
                                  f"{ctx.shard.shard_id!r}")
        return rec

    @staticmethod
    def load(jf: A.JobFiles, shard: Shard) -> dict[str, StageRecord]:
        ids = list(shard.depends_on) + [A.child_shard_id("segment", u) for u in shard.episodes]
        out: dict[str, StageRecord] = {}
        # The sink stamps segment records with the segment *stage* hash (``JobFiles.key``);
        # accept exactly what ``progress.shard_parts`` accepts (review 6 H1).
        accepted = jf.accepts("segment")
        for sid in dict.fromkeys(ids):
            prefix = P.part_prefix(jf, Shard(sid, "segment", (), 0.0))
            for path in sorted(p for p in jf.store.list(prefix) if p.endswith(".jsonl")):
                for rec in P.read_part(jf, path):
                    if (rec.stage == "segment" and rec.provenance.config_hash in accepted
                            and rec.uid in shard.episodes):
                        out.setdefault(rec.uid, rec)
        return out


class CaptionEpisodeJob:
    """One episode: upstream segments, NVIDIA's probe, ``CaptionStage.caption``."""

    stage = "caption"

    def __init__(self, stage, info: StageInfo, inputs: Callable[[EpisodeRef], EpisodeInputs],
                 *, segments: Callable[[EpisodeRef, ShardContext], StageRecord] | None = None,
                 budget: int | None = None):
        self.cap = stage
        self.nv = stage.nv
        self.info = info
        self.inputs = inputs
        self.segments = segments or UpstreamSegments()
        if budget is None:
            from flora_deploy.nv.budget import budget_for

            budget = budget_for(self.nv, "captioner")
        self.budget = budget

    async def run(self, ref: EpisodeRef, batcher, ctx: ShardContext) -> EpisodeOutcome:
        from flora_deploy.stages import caption as C
        from flora_deploy.stages import segment as S

        seg_record = await asyncio.to_thread(self.segments, ref, ctx)
        rows = S.rows_from_record(seg_record)
        inp = await asyncio.to_thread(self.inputs, ref)
        t0 = time.perf_counter()
        ep = await asyncio.to_thread(S.prepare_videos, self.nv, inp.paths, inp.profile,
                                     yam32=inp.yam32, instruction=inp.instruction)
        t1 = time.perf_counter()
        res = await self.cap.caption(batcher, ep, rows)
        t2 = time.perf_counter()
        prov = S.provenance("captioner", engine_label=self.info.engine_label,
                            spec=self.info.engine_spec, models=self.info.models,
                            gpu=self.info.gpu, budget=self.budget,
                            certificate=self.info.certificate, extra=self.info.config_extra,
                            source=inp.sources)
        rec = C.caption_record(res.rows, uid=ref.uid, dataset=ref.dataset,
                               episode_id=ref.episode_id, provenance=prov)
        audit = {"uid": ref.uid, "stage": "caption", "config_hash": prov.config_hash,
                 "profile": ep.profile, "duration_s": ep.duration_s,
                 "envelope_ok": inp.envelope_ok, "caption_inputs": S.json_safe(res.audit),
                 "calls": res.stats.as_json(),
                 "timings_s": {"probe": round(t1 - t0, 3), "caption": round(t2 - t1, 3)}}
        return EpisodeOutcome([rec], [audit])
