"""Output record schema ``flora_labels_v1`` (docs/output-format.md): one record per episode
per stage.

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
    "fake_models",  # local runs (`flora run --local --fake-models`): never real labels
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
    # The producing stage's own hash (e.g. the balanced engine spec + sampling + pins hash of
    # ``stages.segment.config_hash``), kept when the part sink stamps ``config_hash`` with the
    # job's stage key (review 6 M3). ``None`` when the stage set none.
    engine_hash: str | None = None
    # Certification binding (docs/certification.md, "Provenance a record must carry"): the
    # job's embodiment pool, and ``validate.certkey.key_document`` of the stack that produced
    # the record. ``flora certify`` refuses records without them.
    embodiment: str | None = None
    cert_key: dict[str, Any] | None = None
    # Balanced vLLM stages (review 13): the compile cache the container ran
    # (``engines.compile_cache.CacheStatus.identity``: ``warm:<key>:<manifest sha256[:16]>``,
    # ``legacy:<key>``, ``cold`` or ``cold:forced``; a warm start replays the publisher's
    # Inductor autotune picks), and vLLM's preemption counter over the episode's inference
    # window (``vllm:num_preemptions``; > 0: some request in flight then was preempted and
    # recomputed; ``None``: unknown). ``flora certify`` reads both.
    compile_cache: str | None = None
    engine_preemptions: int | None = None
    # Stage-isolated caption certification only (docs/certification.md): the caption stage
    # captioned the pinned oracle segment records ``{"sha256": <file sha256>,
    # "oracle_stage_hash": <exact segcap stage hash>}`` instead of its own job's segments.
    # Omitted from the JSON when None, so every other record's bytes are unchanged.
    segments_from: dict[str, str] | None = None


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
    scores: tuple[float, ...]  # NVIDIA's medoid sample, one score per rubric step
    total: float  # round(sum(scores), 4), as NVIDIA's rubric_reward.py writes it
    samples: int  # answers sampled (NVIDIA's --num-samples, 9)
    medoid_index: int  # index of the medoid among all samples
    parsed_samples: int | None = None  # samples NVIDIA's parse_rubric accepted (medoid pool)


@dataclass(frozen=True)
class StageRecord:
    uid: str  # jobs.types.episode_uid(dataset, episode_id): unique within a job
    dataset: str
    episode_id: str  # as the source dataset names it (not unique across datasets)
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
        """Canonical JSON. NaN/inf are rejected (ValueError): stages must not emit them.
        ``provenance.segments_from`` is left out when None (records predating it, and every
        record but a stage-isolated caption record, keep their bytes)."""
        d = asdict(self)
        if d["provenance"].get("segments_from") is None:
            d["provenance"].pop("segments_from", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageRecord:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unsupported record schema {d.get('schema')!r}")
        d = dict(d)
        d["provenance"] = Provenance(**d["provenance"])
        d["segments"] = tuple(Segment(**s) for s in d.get("segments", ()))
        if d.get("terminal"):
            t = dict(d["terminal"])
            t["probabilities"] = tuple(t["probabilities"])
            d["terminal"] = TerminalReward(**t)
        if d.get("dense"):
            d["dense"] = DenseReward(segments=tuple(d["dense"]["segments"]),
                                     per_step=tuple(d["dense"]["per_step"]))
        if d.get("rubric"):
            r = dict(d["rubric"])
            r["scores"] = tuple(r["scores"])
            d["rubric"] = RubricReward(**r)
        return cls(**d)

    @classmethod
    def from_json(cls, line: str) -> StageRecord:
        return cls.from_dict(json.loads(line))


def certification(envelope_ok: bool | None, *, stack_certified: bool = True,
                  stack_reason: UncertifiedReason | None = "uncertified_stack"
                  ) -> tuple[bool, UncertifiedReason | None]:
    """The one rule for a record's ``certified`` / ``uncertified_reason`` (every worker kind):
    certified only when the episode's **measured** envelope verdict is ``True`` (D13: the
    envelope policy decides which episodes run, never ``certified``; no certificate's evidence
    covers an out-of-envelope input) **and** the stack is certified: the reference (``exact``,
    and kinds that are the reference by identity) or a matching committed certificate.
    ``stack_reason``: the reason when only the stack is not certified."""
    if envelope_ok is False:
        return False, "outside_reference_envelope"
    if envelope_ok is not True:
        return False, "oracle_ineligible_source"  # no envelope verdict
    if not stack_certified:
        return False, stack_reason
    return True, None


def record_key(rec: StageRecord) -> tuple[str, str]:
    """Dedupe key used by compose."""
    return (rec.uid, rec.stage)
