"""The two segmenter constructions NVIDIA's ``pipeline.segment()`` uses, built from NVIDIA classes.

LICENSE REVIEW FLAG (PLAN.md §11). This module restates these NVIDIA values (from
``batch_utils/pipeline.py`` at nvidia/Flora-TempSeg@78a0e4ac); everything else calls NVIDIA's
functions:

- ``three_view_segmenter``: the constructor arguments of NVIDIA's inline 3-view construction
  (``pipeline.py:47-49``);
- ``source_id_for``: the ``"upload_"`` prefix and the 24-hex-digit truncation of the main video's
  sha256 (``pipeline.py:44``);
- ``model_binding``: ``candidate_id="flora_final"`` and the keys of the ``bound`` dict
  (``pipeline.py:45``).

(``nv/budget.py`` restates one more value, the 32768-token context limit, flagged there.) The
nvlink test ``test_nv_link_segmenter.py`` proves that the binding, the source and both
constructions equal what NVIDIA's own ``pipeline.segment()`` builds: it captures the arguments
NVIDIA passes to its own classes and compares them with ours.

``segmenter_for`` mirrors the branch in ``pipeline.py:47-51``: three camera views use the inline
construction, anything else uses ``adapter.production_segmenter`` (``adapter.py:44-47``), including
single-view robot episodes. It does NOT apply ``pipeline.annotate``'s profile and view guards
(``pipeline.py:142-152``; e.g. NVIDIA rejects the human pool with three views). Callers must
enforce them.
"""

from __future__ import annotations

from typing import Any

from flora_deploy.nv import Family


def _adapter(nv):
    if nv.family is not Family.SEGCAP:
        raise ValueError("segmenters come from the SEGCAP family")
    return nv.production


def source_id_for(sha256: str) -> str:
    """``AnnotationSource.source_id`` as NVIDIA derives it from the main video's sha256
    (``pipeline.py:44``). Any stable id works (NVIDIA uses it for ids and hashes only)."""
    return "upload_" + sha256[:24]


def annotation_source(nv, *, source_id: str, location: str, duration_s: float, fps: float):
    """NVIDIA ``AnnotationSource`` with the fields ``pipeline.segment()`` sets
    (``pipeline.py:44``)."""
    a = _adapter(nv)
    return a.AnnotationSource(
        source_id=source_id, location=location, duration_s=duration_s, fps=fps
    )


def model_binding(nv, pool: str) -> dict:
    """The ``bound`` dict of ``pipeline.py:45``; NVIDIA's segmenter and predictor both check it."""
    a = _adapter(nv)
    return dict(
        candidate_id="flora_final",
        checkpoint_sha256=nv.config.MODELS["tempseg"]["sha256"],
        model_io_contract=a.fine_window_profile(a.PROFILES[pool]).model_io_contract,
    )


def production_segmenter(nv, predictor: Any, binding: dict, pool: str):
    """Human and single-view construction: NVIDIA's own ``adapter.production_segmenter``."""
    return _adapter(nv).production_segmenter(predictor, binding, pool)


def three_view_segmenter(nv, predictor: Any, binding: dict, pool: str):
    """3-view construction, re-expressed from ``pipeline.py:47-49`` (flagged for license review).

    Differs from ``production_segmenter`` in ``profile_id`` and in using NVIDIA's *default*
    ``FineSegmentationQualityPolicy`` apart from ``drop_guard_only_segments=True``.
    """
    a = _adapter(nv)
    fine = a.FineSubtaskSegmenter(
        predictor=predictor,
        model_binding=binding,
        profile_id="robot_atomic_intent_v1",
        window_profile_id=a.PROFILES[pool],
        fail_on_unresolved=True,
        batch_concurrency=1,
        unresolved_stitch_tolerance_ticks=2,
        quality_policy=a.FineSegmentationQualityPolicy(drop_guard_only_segments=True),
    )
    return a.HierarchicalSubtaskSegmenter(
        coarse_segmenter=a.NoCoarse(),
        fine_segmenter=fine,
        fine_prior_mode="none",
        fine_failure_policy="adaptive_fine_then_conservative_completion",
        fine_output_coverage_policy="allow_gaps",
        adaptive_fine_retry_max_windows=32,
    )


def segmenter_for(nv, predictor: Any, binding: dict, pool: str, n_views: int):
    """Pick the construction exactly as ``pipeline.segment()`` does (``len(videos) == 3``)."""
    if n_views == 3:
        return three_view_segmenter(nv, predictor, binding, pool)
    return production_segmenter(nv, predictor, binding, pool)
