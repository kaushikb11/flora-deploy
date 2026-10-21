"""Media contracts shared by the index pass, decoders and stages (PLAN.md D7).

``StreamIndex`` mirrors the fields NVIDIA's ``batch_utils.media.Video`` exposes. For any episode,
its values must equal what NVIDIA's ``media.probe`` returns on a stream-copy per-episode cut of the
same frames. That equality is the L0(a) parity test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

CameraRole = Literal["main", "left_wrist", "right_wrist"]


@dataclass(frozen=True)
class SourceRef:
    """Where an episode's frames for one camera live in the source dataset."""

    uri: str  # local path, Volume path or bucket key of the video file
    # Episode range inside the file, in the file's own seconds (None = whole file).
    start_s: float | None = None
    end_s: float | None = None


@dataclass(frozen=True)
class StreamIndex:
    """Probe-equivalent metadata for one camera of one episode."""

    source: SourceRef
    rate: Fraction  # == NVIDIA Video.rate (stream.average_rate of the per-episode cut)
    pts_us: tuple[Fraction, ...]  # relative to the episode's first frame; == Video.pts_us
    duration_us: int  # == Video.duration_us
    size: tuple[int, int]  # (width, height)
    cfr: bool  # == Video.cfr
    codec: str  # e.g. "h264", "av1"
    pix_fmt: str
    starts_on_keyframe: bool  # a stream-copy cut decodes exactly like the source -> eligible
    # Exact source timestamps (seconds in the source file) of every frame, for write-back.
    source_timestamps_s: tuple[float, ...] = ()
    # Added by WS-C (all defaulted, so existing constructors keep working). They carry what the
    # envelope checks and decoders need without reopening the file:
    # Source track time_base; pts_us are exact multiples. The cut keeps it except below a
    # timescale of 10000, where movenc doubles the timescale (1/30 -> 1/15360): same rationals.
    time_base: Fraction | None = None
    byte_size: int = 0  # sum of the episode's packet payloads (lower bound of the cut's size)
    video_streams: int = 1  # video streams NVIDIA's probe would see (the cut: always 1)
    missing_pts: bool = False  # a packet without a PTS was seen (probe rejects that)
    # Codec-parameter colour tags: advisory only. NVIDIA decides the canonical_rgb branch per
    # decoded frame (media.py:85); never pre-select the branch from these.
    colorspace: int | None = None
    color_range: int | None = None
    # Absolute source pts (time_base units) of every frame, in presentation order: range decode
    # checks decoded frames against these (not against the first decoded frame).
    source_pts: tuple[int, ...] = ()
    # Why a stream-copy cut is not exact (index.INELIGIBILITY_REASONS); () when eligible.
    ineligible: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return self.duration_us / 1_000_000


@dataclass(frozen=True)
class EnvelopeVerdict:
    """NVIDIA's validated operating envelope, evaluated per episode (PLAN.md D13; PLAN-v3 §1.1)."""

    ok: bool
    reasons: tuple[str, ...] = ()  # machine-readable, e.g. "duration_gt_120s", "not_cfr"


@dataclass(frozen=True)
class EpisodeIndex:
    episode_id: str
    dataset: str
    streams: dict[CameraRole, StreamIndex]
    task: str | None = None  # episode instruction text, if the dataset has one
    envelope: EnvelopeVerdict = field(default_factory=lambda: EnvelopeVerdict(ok=True))

    @property
    def oracle_eligible(self) -> bool:
        return self.envelope.ok and all(s.starts_on_keyframe for s in self.streams.values())

    @property
    def ineligibility(self) -> tuple[str, ...]:
        """``role:reason`` for every camera whose cut would not be exact."""
        return tuple(f"{r}:{x}" for r, s in sorted(self.streams.items()) for x in s.ineligible)
