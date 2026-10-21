"""Output record schema ``flora_labels_v1`` (PLAN.md §6.6): one record per episode per stage.

Records are JSON-serializable dataclasses. Workers append them to local part-files (JSONL) and copy
the parts whole to the store; ``compose`` dedupes by ``(episode_id, stage)``. NVIDIA-specific
payloads (segment metadata, rationales) are carried as opaque dicts from NVIDIA's own output and
are never interpreted here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA = "flora_labels_v1"

Stage = Literal["segment", "caption", "reward.terminal", "reward.dense", "reward.rubric"]
SegmentStatus = Literal["complete", "unsupported_target_frames"]
UncertifiedReason = Literal[
    "outside_reference_envelope",
    "uncertified_stack",
    "uncovered_pool",
    "oracle_ineligible_source",
]


@dataclass(frozen=True)
class Provenance:
    models: dict[str, str]  # pin key -> revision
    engine: str  # e.g. "hf-ref-5.8.1" | "vllm-0.24.0-offline"
    preset: str  # exact | balanced | budget | interactive
    config_hash: str
    logic_version: str
    framework: str  # "flora-deploy <version>"
    gpu: str | None = None  # from nvidia-smi; not part of config_hash (D11)
    certificate: str | None = None  # None for `exact` (reference by construction)
    source: dict[str, dict[str, Any]] = field(default_factory=dict)  # role -> {uri, range_s}


@dataclass(frozen=True)
class Segment:
    event_id: str
    start_s: float
    end_s: float
    status: SegmentStatus = "complete"
    caption: str | None = None  # None in segment-stage records
    fine_event_id: str | None = None
    segmentation_provenance: str | None = None  # one of NVIDIA's values, passed through
    degradation: dict[str, Any] = field(default_factory=dict)
    nv_metadata: dict[str, Any] = field(default_factory=dict)  # NVIDIA's segment metadata, opaque


@dataclass(frozen=True)
class TerminalReward:
    score: int  # 1..5, decoded token
    expected_score: float  # E[s] from the renormalized top-k score-token distribution
    probabilities: tuple[float, float, float, float, float]
    rationale: str | None = None


@dataclass(frozen=True)
class DenseReward:
    segments: tuple[dict[str, Any], ...]  # NVIDIA stitched segments (label, span, rationale)
    per_step: tuple[int, ...]  # 1 failure / 2 suboptimal / 3 normal per 0.5 s group


@dataclass(frozen=True)
class RubricReward:
    scores: tuple[float, ...]
    total: float
    samples: int
    medoid_index: int


@dataclass(frozen=True)
class StageRecord:
    episode_id: str
    stage: Stage
    provenance: Provenance
    certified: bool
    uncertified_reason: UncertifiedReason | None = None
    segments: tuple[Segment, ...] = ()
    terminal: TerminalReward | None = None
    dense: DenseReward | None = None
    rubric: RubricReward | None = None
    audit_ref: str | None = None  # store path of the full NVIDIA audit for this record
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.certified == (self.uncertified_reason is not None):
            raise ValueError("certified records have no reason; uncertified records need one")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_key(rec: StageRecord) -> tuple[str, str]:
    """Dedupe key used by compose."""
    return (rec.episode_id, rec.stage)
