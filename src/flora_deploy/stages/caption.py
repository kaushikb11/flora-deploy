"""``balanced`` caption stage: NVIDIA's caption plans and input builders, batched across episodes.

For one episode the stage runs exactly what ``pipeline.annotate`` runs after segmentation:

- ``caption_three(videos, segments, session, instruction, progress, profile=profile)`` for three
  views, else ``caption_single(videos, segments, profile, session, instruction, progress)``.
  NVIDIA's code builds every plan (``plans_for_episode`` / ``caption_plans`` / ``native_blocks``),
  decodes and composes frames, normalizes geometry, writes and reads its PNGs and builds every
  prompt. The session hands it a ``CaptureProcessor`` instead of the real Captioner processor,
  and ``pipeline.generate`` is tapped (``stages.segment.GenerateTap``), so each caption call
  becomes a ``CapturedRequest`` plus a sentinel caption in NVIDIA's rows;
- all calls of all live episodes go to the micro-batcher (Gate C wrap, greedy, 512 tokens, stop
  on ``<|im_end|>`` only), then each completion is accepted by NVIDIA's ``models.generate`` rules
  in NVIDIA's call order (32k guard, EOS, non-empty) and replaces its sentinel. The first failing
  call fails the episode, as NVIDIA's sequential loop would;
- ``pipeline.annotate``'s ``segmentation_provenance`` copy and ``api.annotate``'s interval check
  follow.

LICENSE REVIEW FLAG (PLAN.md §11): ``CaptionStage.caption`` restates the branch, the provenance copy
(``segment_row['metadata'].get('segmentation_provenance', 'fine')``) and the interval check of
``pipeline.annotate`` / ``api.DataEnginePipeline.annotate``.
``tests/nvlink/test_balanced_nvlink.py`` compares the result with NVIDIA's own ``caption_*`` on
canned outputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from flora_deploy.nv.capture import CaptureProcessor
from flora_deploy.nv.errors import NvidiaRuntimeError  # NVIDIA's rejection (api.py:55)
from flora_deploy.stages.segment import (
    UNCERTIFIED,
    CallStats,
    EpisodeVideos,
    GenerateTap,
    run_tapped,
    substitute,
)


class _CaptureSession:
    """What ``caption_*`` see as ``session``: ``load('captioner')`` -> (no model, capture)."""

    def __init__(self, proc: CaptureProcessor):
        self.proc = proc
        self.audit: dict = {}

    def load(self, kind: str):
        if kind != "captioner":
            raise RuntimeError(f"caption stage session asked for {kind!r}")
        return None, self.proc

    def close(self) -> None:
        return None


@dataclass
class CaptionResult:
    rows: list[dict]  # NVIDIA's caption rows (+ segmentation_provenance), in segment order
    audit: list[dict]  # NVIDIA's caption_inputs audit
    stats: CallStats


def _noop_progress(*_a, **_k) -> None:
    return None


class CaptionStage:
    """The Captioner on a micro-batched engine. One per container; ``caption`` per episode."""

    def __init__(self, nv, processor):
        self.nv = nv
        self.processor = processor
        self.tokenizer = processor.tokenizer
        eos = self.tokenizer.eos_token_id
        if type(eos) is not int:
            raise ValueError(f"Captioner tokenizer EOS must be one id, got {eos!r}")
        self.eos = eos
        self.tap = GenerateTap.acquire(nv)

    def close(self) -> None:
        self.tap.release()

    def _nvidia(self, ep: EpisodeVideos, segments: list[dict], proc: CaptureProcessor):
        pipeline = self.nv.pipeline
        session = _CaptureSession(proc)
        if len(ep.videos) == 3:
            return pipeline.caption_three(ep.videos, segments, session, ep.instruction,
                                          _noop_progress, profile=ep.profile)
        return pipeline.caption_single(ep.videos, segments, ep.profile, session, ep.instruction,
                                       _noop_progress)

    async def caption(self, batcher, ep: EpisodeVideos, segments: Sequence[Mapping]
                      ) -> CaptionResult:
        segs = [dict(s) for s in segments]
        proc = CaptureProcessor.for_kind(self.nv, self.processor, "captioner")
        stats = CallStats()
        (rows, audit), texts = await run_tapped(
            self.nv, batcher, lambda: self._nvidia(ep, segs, proc), tokenizer=self.tokenizer,
            eos=self.eos, stats=stats)
        rows = [dict(r) for r in rows]
        for row in rows:
            row["caption"] = substitute(row.get("caption"), texts)
        for row, seg in zip(rows, segs, strict=True):
            row["segmentation_provenance"] = (seg.get("metadata") or {}).get(
                "segmentation_provenance", "fine")
        duration = ep.duration_s
        for row in rows:
            s, e = row["start_s"], row["end_s"]
            if not (math.isfinite(s) and math.isfinite(e) and 0 <= s < e <= duration + 0.001):
                raise NvidiaRuntimeError("The pipeline returned an invalid caption interval.")
        return CaptionResult(rows, list(audit), stats)


def caption_record(rows: Sequence[Mapping], *, uid: str, dataset: str, episode_id: str,
                   provenance, certificate_ok: bool = False):
    """NVIDIA's caption rows as a ``caption`` ``StageRecord`` (shaped like the ``exact``
    preset's ``stages.reference.stage_records``)."""
    from flora_deploy.records import Segment, StageRecord

    segs = tuple(
        Segment(event_id=f"event_{i:04d}", start_s=float(r["start_s"]), end_s=float(r["end_s"]),
                status="complete" if r.get("status") == "complete"
                else "unsupported_target_frames",
                caption=r.get("caption"), segmentation_provenance=r.get("segmentation_provenance"))
        for i, r in enumerate(rows))
    return StageRecord(uid=uid, dataset=dataset, episode_id=episode_id, stage="caption",
                       provenance=provenance, certified=certificate_ok,
                       uncertified_reason=None if certificate_ok else UNCERTIFIED,
                       segments=segs)


def summary(result: CaptionResult) -> dict[str, Any]:
    return {"captions": len(result.rows),
            "complete": sum(r.get("status") == "complete" for r in result.rows),
            **result.stats.as_json()}
