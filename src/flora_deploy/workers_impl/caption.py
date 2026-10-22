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
chains caption shards on segment receipts, so this means a lost or foreign part).
Upstream-poisoned episodes never get here (runner step 3).

Pinned segments (``PinnedSegments``, stage-isolated caption certification only:
``pipeline.caption_segments``, ``docs/certification.md``): the shard captions the **exact
oracle's** ``segment`` records from one staged JSONL file in the job store instead of its own
job's segments. The file is read once per shard and its sha256 must be the pinned one; every
record must be the oracle's (preset ``exact``, stage ``segment``, ``config_hash`` the pinned
oracle stage hash). Oracle segment records carry no NVIDIA metadata, so the rows are rebuilt
as ``{start_s, end_s, metadata: {segmentation_provenance[, fine_event_id]}}``: NVIDIA's caption
plan builder needs only the spans (and copies the provenance), which
``tests/nvlink/test_balanced_nvlink.py`` checks, and every call's input digest is compared with
the oracle's at certification (``inputs_not_isolated``). Each record is stamped
``provenance.segments_from``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.types import EpisodeRef, Shard
from flora_deploy.records import StageRecord
from flora_deploy.workers_impl.segment import (
    EpisodeInputs,
    EpisodeOutcome,
    ShardContext,
    StageInfo,
    gates_of,
    phase,
    timings,
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


# Worker env of a stage-isolated caption job (``modal_app.factory.plan_app``): JSON
# ``{"key": <store key of the staged oracle segment JSONL>, "sha256", "oracle_stage_hash"}``.
CAPTION_SEGMENTS_ENV = "FLORA_CAPTION_SEGMENTS"
_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")


class PinnedSegmentsError(RuntimeError):
    """The staged oracle segment file is missing, altered or not the oracle's (infra)."""


@dataclass(frozen=True)
class PinnedSpec:
    key: str  # store key of the JSONL (relative to the job store root)
    sha256: str
    oracle_stage_hash: str

    def __post_init__(self) -> None:
        if not self.key or self.key.startswith("/") or ".." in self.key.split("/"):
            raise ValueError(f"pinned segments key {self.key!r} is not a store-relative key")
        for f in ("sha256", "oracle_stage_hash"):
            if not _HEX64.match(str(getattr(self, f))):
                raise ValueError(f"pinned segments {f} must be 64 lowercase hex")

    @property
    def segments_from(self) -> dict[str, str]:
        return {"sha256": self.sha256, "oracle_stage_hash": self.oracle_stage_hash}

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> PinnedSpec | None:
        raw = env.get(CAPTION_SEGMENTS_ENV)
        if not raw:
            return None
        d = json.loads(raw)
        return cls(key=str(d["key"]), sha256=str(d["sha256"]),
                   oracle_stage_hash=str(d["oracle_stage_hash"]))


class PinnedSegments:
    """uid -> the exact oracle's ``segment`` StageRecord, from the staged file ``spec.key``."""

    def __init__(self, spec: PinnedSpec):
        self.spec = spec

    @property
    def segments_from(self) -> dict[str, str]:
        return self.spec.segments_from

    def __call__(self, ref: EpisodeRef, ctx: ShardContext) -> StageRecord:
        cache = ctx.cache.get("pinned")
        if cache is None:
            cache = ctx.cache["pinned"] = self.load(ctx.jf.store)
        rec = cache.get(ref.uid)
        if rec is None:
            raise MissingUpstream(f"episode {ref.uid!r}: not in the pinned oracle segments "
                                  f"{self.spec.key!r}")
        return rec

    def load(self, store: Any) -> dict[str, StageRecord]:
        try:
            data = store.get_bytes(self.spec.key)
        except Exception as exc:  # noqa: BLE001 - a lost staging step: infra
            raise PinnedSegmentsError(f"pinned oracle segments {self.spec.key!r}: {exc!r}"
                                      ) from exc
        got = hashlib.sha256(data).hexdigest()
        if got != self.spec.sha256:
            raise PinnedSegmentsError(f"pinned oracle segments {self.spec.key!r}: sha256 {got} "
                                      f"!= pinned {self.spec.sha256}")
        out: dict[str, StageRecord] = {}
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            rec = StageRecord.from_json(line)
            p = rec.provenance
            if (rec.stage != "segment" or p.preset != "exact"
                    or p.config_hash != self.spec.oracle_stage_hash):
                raise PinnedSegmentsError(
                    f"pinned oracle segments: a {p.preset!r} {rec.stage!r} record of stage hash "
                    f"{str(p.config_hash)[:16]}; expected the exact oracle's segment records "
                    f"({self.spec.oracle_stage_hash[:16]})")
            if rec.uid in out:
                raise PinnedSegmentsError(f"pinned oracle segments: two records for {rec.uid!r}")
            out[rec.uid] = rec
        return out


def pinned_rows(record: StageRecord) -> list[dict]:
    """NVIDIA's ``pipeline.segment`` rows rebuilt from an oracle segment record (which has no
    ``nv_metadata``): the span, ``segmentation_provenance`` and, when set, ``fine_event_id``."""
    if record.stage != "segment":
        raise ValueError(f"expected a segment record, got {record.stage!r}")
    rows = []
    for s in record.segments:
        md: dict[str, Any] = {"segmentation_provenance": s.segmentation_provenance or "fine"}
        if s.fine_event_id is not None:
            md["fine_event_id"] = s.fine_event_id
        rows.append(dict(start_s=s.start_s, end_s=s.end_s, metadata=md))
    return rows


def segments_for(env: Mapping[str, str] = os.environ
                 ) -> Callable[[EpisodeRef, ShardContext], StageRecord]:
    """The caption job's segment provider: :class:`PinnedSegments` when the job pins the
    oracle's segments (:data:`CAPTION_SEGMENTS_ENV`), else :class:`UpstreamSegments`."""
    spec = PinnedSpec.from_env(env)
    return PinnedSegments(spec) if spec is not None else UpstreamSegments()


class CaptionEpisodeJob:
    """One episode: upstream segments, NVIDIA's probe, ``CaptionStage.caption``."""

    stage = "caption"
    phased = True  # CPU preparation outside the batcher's episode (segment.EpisodeGates)

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
        from flora_deploy.stages import segment as S

        gates = gates_of(ctx)
        async with gates.prep():  # phase 1: CPU, outside the batcher's episode
            with phase("inputs"):
                seg_record = await asyncio.to_thread(self.segments, ref, ctx)
                rows = (pinned_rows(seg_record) if isinstance(self.segments, PinnedSegments)
                        else S.rows_from_record(seg_record))
                inp = await asyncio.to_thread(self.inputs, ref)
            t0 = time.perf_counter()
            with phase("nvidia"):
                ep = await asyncio.to_thread(S.prepare_videos, self.nv, inp.paths, inp.profile,
                                             yam32=inp.yam32, instruction=inp.instruction)
            t1 = time.perf_counter()
        async with gates.infer(batcher):  # phase 2: inference, inside it
            tq = time.perf_counter()
            with phase("nvidia"):
                res = await self.cap.caption(batcher, ep, rows)
        t2 = time.perf_counter()
        with phase("records"):
            return self._outcome(ref, inp, ep, res, t0, t1, t2, tq)

    def _outcome(self, ref, inp, ep, res, t0, t1, t2, tq=None) -> EpisodeOutcome:
        from flora_deploy.stages import caption as C
        from flora_deploy.stages import segment as S

        prov = S.provenance("captioner", engine_label=self.info.engine_label,
                            spec=self.info.engine_spec, models=self.info.models,
                            gpu=self.info.gpu, budget=self.budget,
                            certificate=self.info.certificate, extra=self.info.config_extra,
                            source=inp.sources)
        pinned = getattr(self.segments, "segments_from", None)
        if pinned is not None:
            prov = dataclasses.replace(prov, segments_from=dict(pinned))
        rec = C.caption_record(res.rows, uid=ref.uid, dataset=ref.dataset,
                               episode_id=ref.episode_id, provenance=prov)
        audit = {"uid": ref.uid, "stage": "caption", "config_hash": prov.config_hash,
                 "profile": ep.profile, "duration_s": ep.duration_s,
                 "envelope_ok": inp.envelope_ok, "caption_inputs": S.json_safe(res.audit),
                 "calls": res.stats.as_json(),
                 # per caption row: NVIDIA's input digest of its Captioner call (None: no call)
                 "input_digests": list(getattr(res, "input_digests", None) or []),
                 "timings_s": timings(t0, t1, t2, tq, "caption")}
        errors = list(getattr(res, "digest_errors", None) or [])
        if errors:
            audit["input_digest_error"] = errors[0]
        return EpisodeOutcome([rec], [audit])
