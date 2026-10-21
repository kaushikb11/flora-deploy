"""Per-episode probe equivalent and NVIDIA envelope checks (PLAN.md D7, D13, §1.1).

Contract: for an episode range of a (possibly concatenated) MP4, ``index_stream(source)`` returns
exactly what NVIDIA's ``batch_utils.media.probe`` (``media.py:25-58``) returns when run on
``cuts.stream_copy_cut(source, ...)``. NVIDIA's code is never imported here; the rules are
re-expressed and cited by file:line of the pinned Flora-TempSeg ``batch_utils`` snapshot. The
``-m nvlink`` tests check the equality field by field against the real ``probe``.

How each value of the cut is predicted from the source's packets (evidence: docs/media-notes.md):

* The cut holds exactly the episode's packets, in decode order, with the source ``time_base``,
  timestamps shifted by a constant, and packet durations copied.
* ``rate``: FFmpeg's mov demuxer sets ``avg_frame_rate = av_reduce(timescale * n_samples,
  sum(stts durations), INT_MAX)``. The MP4 muxer writes each sample's stts duration as the dts
  delta to the next sample, and the last one as that packet's ``duration``. All of those equal the
  ``duration`` values the source demuxer reports for the same packets, so the per-episode rate is
  the formula over the episode's own packet durations - not the concatenated file's
  ``stream.average_rate``.
* ``pts_us``: presentation order is ascending pts; relative to the first presented frame.
* Tail: the last presented frame's ``frame.duration`` is the duration of the packet that carried
  it (the source packet with the largest pts), else ``1/rate`` (``media.py:48,51``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import av

from flora_deploy.media.types import (
    CameraRole,
    EnvelopeVerdict,
    EpisodeIndex,
    SourceRef,
    StreamIndex,
)

# NVIDIA's envelope constants (batch_utils/config.py:8-11). Numbers only, re-stated.
MAX_DURATION_S = 120
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_FRAME_COUNT = 7200
MIN_RATE = 4
INSTRUCTION_MAX_CHARS = 2000  # api.py:24-25, preprocessing.py:11-12

INT_MAX = 2**31 - 1
US = 1_000_000

# NVIDIA's opening options (media.py:26, :76) - applied to the format and codec contexts.
NV_OPEN_OPTIONS = {"err_detect": "explode"}

THREE_VIEW_ROLES: frozenset[str] = frozenset({"main", "left_wrist", "right_wrist"})


class MediaError(ValueError):
    """A source cannot be indexed or cut the way the contract requires."""


class NotKeyframeAligned(MediaError):
    """The episode range does not start on a keyframe, so no stream-copy cut exists."""


# ---------------------------------------------------------------------------------------------
# Packet scan (one demux pass per file, shared by every episode in a concatenated file)


@dataclass(frozen=True)
class Packet:
    pts: int
    dts: int
    duration: int
    keyframe: bool
    size: int


@dataclass(frozen=True)
class StreamScan:
    """Demux-only view of a file's first video stream (no decoding)."""

    path: str
    time_base: Fraction
    codec: str
    pix_fmt: str
    width: int
    height: int
    video_streams: int
    colorspace: int | None
    color_range: int | None
    packets: tuple[Packet, ...]  # decode (file) order, payload-bearing packets only
    missing_pts: bool  # a payload packet had no pts


def scan_stream(path: str | os.PathLike[str]) -> StreamScan:
    """Demux every packet of the first video stream of ``path`` (MP4/MOV)."""
    with av.open(str(path), options=NV_OPEN_OPTIONS) as container:
        if not container.streams.video:
            raise MediaError(f"{path}: no video stream")
        stream = container.streams.video[0]
        tb = stream.time_base
        if tb is None or tb.numerator != 1:
            # MP4/MOV tracks always use 1/timescale; other containers are not supported.
            raise MediaError(f"{path}: unsupported time_base {tb} (MP4/MOV sources only)")
        cc = stream.codec_context
        packets: list[Packet] = []
        missing = False
        for pkt in container.demux(stream):
            if not pkt.size:
                continue  # demuxer flush packets
            if pkt.pts is None or pkt.dts is None:
                missing = True
                continue
            packets.append(Packet(pkt.pts, pkt.dts, pkt.duration or 0, pkt.is_keyframe, pkt.size))
        par = stream.codec_context
        return StreamScan(
            path=str(path),
            time_base=Fraction(tb),
            codec=cc.codec.canonical_name,
            pix_fmt=str(par.pix_fmt or ""),
            width=cc.width,
            height=cc.height,
            video_streams=len(container.streams.video),
            colorspace=_int_or_none(getattr(par, "colorspace", None)),
            color_range=_int_or_none(getattr(par, "color_range", None)),
            packets=tuple(packets),
            missing_pts=missing,
        )


def _int_or_none(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# Episode selection


@dataclass(frozen=True)
class EpisodeSlice:
    """The episode's packets inside a scan: ``scan.packets[first:last + 1]`` in decode order."""

    first: int
    last: int
    contiguous: bool  # the range's packets are one contiguous run in decode order


def select_episode(scan: StreamScan, start_s: float | None, end_s: float | None) -> EpisodeSlice:
    """Packets whose presentation time lies in ``[start_s, end_s)`` of the file.

    LeRobot's ``from_timestamp``/``to_timestamp`` are float sums of per-episode durations, so the
    bounds are matched with a tolerance of half the file's median packet duration: a frame at
    ``t`` belongs to the episode iff ``start_s - h <= t < end_s - h``.
    """
    pk = scan.packets
    if not pk:
        raise MediaError(f"{scan.path}: no packets")
    if start_s is None and end_s is None:
        return EpisodeSlice(0, len(pk) - 1, True)
    tb = scan.time_base
    durs = sorted(p.duration for p in pk if p.duration > 0)
    half = Fraction(durs[len(durs) // 2], 2) * tb if durs else Fraction(1, 10_000)
    lo = Fraction(start_s) - half if start_s is not None else None
    hi = Fraction(end_s) - half if end_s is not None else None
    idx = [
        i
        for i, p in enumerate(pk)
        if (lo is None or p.pts * tb >= lo) and (hi is None or p.pts * tb < hi)
    ]
    if not idx:
        raise MediaError(f"{scan.path}: no frames in [{start_s}, {end_s})")
    first, last = idx[0], idx[-1]
    return EpisodeSlice(first, last, len(idx) == last - first + 1)


# ---------------------------------------------------------------------------------------------
# FFmpeg rational reduction


def av_reduce(num: int, den: int, maximum: int = INT_MAX) -> Fraction:
    """Best rational approximation with numerator and denominator <= ``maximum``.

    Behaves like libavutil ``av_reduce`` (rational.c), which the mov demuxer uses for
    ``avg_frame_rate``: exact when the reduced fraction fits, otherwise the convergent or the
    larger semiconvergent of the continued fraction, chosen by the same closeness test.
    Re-implemented from the continued-fraction definition (not copied).
    """
    if den == 0:
        raise ZeroDivisionError("av_reduce with zero denominator")
    sign = -1 if (num < 0) != (den < 0) else 1
    num, den = abs(num), abs(den)
    g = math.gcd(num, den)
    if g:
        num //= g
        den //= g
    if num <= maximum and den <= maximum:
        return Fraction(sign * num, den)
    p0, q0, p1, q1 = 0, 1, 1, 0  # previous two convergents
    while den:
        x = num // den
        rem = num - den * x
        p2, q2 = x * p1 + p0, x * q1 + q0
        if p2 > maximum or q2 > maximum:
            # Largest admissible partial quotient; take the semiconvergent when it is closer.
            xs = (maximum - p0) // p1 if p1 else x
            if q1:
                xs = min(xs, (maximum - q0) // q1)
            if den * (2 * xs * q1 + q0) > num * q1:
                p1, q1 = xs * p1 + p0, xs * q1 + q0
            break
        p0, q0, p1, q1 = p1, q1, p2, q2
        num, den = den, rem
    return Fraction(sign * p1, q1)


def cut_timescale(time_base: Fraction) -> int:
    """Track timescale the MP4 muxer gives a stream-copy cut that keeps ``time_base``.

    movenc uses ``time_base.den`` and doubles it while below 10000. Scaling every duration and
    the timescale by the same factor leaves all rationals unchanged; only ``av_reduce``'s
    INT_MAX bound sees the scaled values.
    """
    ts = time_base.denominator
    while ts < 10_000:
        ts *= 2
    return ts


# ---------------------------------------------------------------------------------------------
# Index


def index_stream(
    source: SourceRef,
    *,
    scan: StreamScan | None = None,
    decode: bool = False,
) -> StreamIndex:
    """Probe-equivalent ``StreamIndex`` for one camera of one episode.

    ``scan``: reuse a ``scan_stream`` result for the same file (one scan per concatenated file).
    ``decode``: take PTS and the tail from actually decoded frames (as ``probe`` does) instead of
    predicting them from packets. Both modes are checked against NVIDIA in the nvlink tests;
    packet mode is the fast default.
    """
    scan = scan if scan is not None else scan_stream(source.uri)
    sl = select_episode(scan, source.start_s, source.end_s)
    pk = scan.packets[sl.first : sl.last + 1]
    tb = scan.time_base
    n = len(pk)

    # rate: mov demuxer formula over the cut's stts (see module docstring).
    scale = cut_timescale(tb) // tb.denominator
    total = sum(p.duration for p in pk) * scale
    rate = av_reduce(cut_timescale(tb) * n, total) if total > 0 else Fraction(0)

    first = pk[0]
    starts_on_keyframe = (
        first.keyframe
        and sl.contiguous
        and first.pts == min(p.pts for p in pk)  # no leading (open-GOP) pictures
    )

    if decode:
        if not starts_on_keyframe:
            raise NotKeyframeAligned(f"{source.uri}: decode mode needs a keyframe-aligned range")
        from flora_deploy.media.rangedecode import iter_frames

        pts_abs: list[int] = []
        last_dur = 0
        for frame in iter_frames(source, scan=scan):
            pts_abs.append(frame.pts)
            last_dur = frame.duration or 0
        presented = pts_abs
    else:
        presented = sorted(p.pts for p in pk)
        last_dur = max(pk, key=lambda p: p.pts).duration

    zero = presented[0]
    pts_us = tuple(Fraction(t - zero) * tb * US for t in presented)
    if last_dur:
        tail = Fraction(last_dur) * tb * US
    elif rate:
        tail = Fraction(US) / rate
    else:
        tail = Fraction(0)
    duration_us = round(pts_us[-1] + tail)  # Fraction round = banker's rounding, as media.py:52
    cfr = bool(rate) and all(abs(t - Fraction(i * US) / rate) <= 1 for i, t in enumerate(pts_us))

    return StreamIndex(
        source=source,
        rate=rate,
        pts_us=pts_us,
        duration_us=duration_us,
        size=(scan.width, scan.height),
        cfr=cfr,
        codec=scan.codec,
        pix_fmt=scan.pix_fmt,
        starts_on_keyframe=starts_on_keyframe,
        source_timestamps_s=tuple(float(t * tb) for t in presented),
        time_base=tb,
        byte_size=sum(p.size for p in pk),
        video_streams=1,  # a stream-copy cut carries exactly the one video stream
        missing_pts=scan.missing_pts,
        colorspace=scan.colorspace,
        color_range=scan.color_range,
    )


def estimated_cut_bytes(stream: StreamIndex) -> int:
    """Conservative upper estimate of the cut file's size (payload + MP4 boxes).

    Exact payload; boxes bounded by 4 KiB fixed + 32 bytes per sample (stts/ctts/stsz/stco/stss
    entries plus chunking), which over-covers what movenc writes for a single-track file.
    """
    return stream.byte_size + 4096 + 32 * len(stream.pts_us)


# ---------------------------------------------------------------------------------------------
# Envelope


def stream_checks(stream: StreamIndex) -> list[str]:
    """Per-file reasons NVIDIA would reject this camera's cut (machine-readable ids).

    Mirrors, in this order: ``api.py:34-35`` (0 < size <= 256 MiB), ``media.py:27-28`` (one video
    stream), ``:31-33`` (rate >= 4), ``:34-36`` (1..4096 px), ``:39-40`` (every frame has a
    PTS), ``:43-44`` (strictly increasing PTS), ``:45-46`` (t < 120 s and at most 7200 frames),
    ``:49-50`` (>= 2 frames), ``:53-54`` (duration <= 120 s).
    """
    reasons: list[str] = []
    size = estimated_cut_bytes(stream)
    if stream.byte_size <= 0:
        reasons.append("file_empty")
    elif size > MAX_UPLOAD_BYTES:
        reasons.append("file_gt_256MiB")
    if stream.video_streams != 1:
        reasons.append("video_streams_ne_1")
    if not stream.rate or stream.rate < MIN_RATE:
        reasons.append("fps_lt_4")
    if min(stream.size) <= 0 or max(stream.size) > MAX_DIMENSION:
        reasons.append("dimension_out_of_range")
    if stream.missing_pts:
        reasons.append("missing_pts")
    pts = stream.pts_us
    if any(b <= a for a, b in zip(pts, pts[1:], strict=False)):
        reasons.append("pts_not_increasing")
    if len(pts) > MAX_FRAME_COUNT:
        reasons.append("frames_gt_7200")
    if any(t >= MAX_DURATION_S * US for t in pts) or stream.duration_us > MAX_DURATION_S * US:
        reasons.append("duration_gt_120s")
    if len(pts) < 2:
        reasons.append("too_few_frames")
    return reasons


_PROFILES = {"human": "Human", "yam": "YAM", "dex": "Dex", "tianji": "Tianji"}  # api.py:6


def envelope_checks(
    streams: Mapping[str, StreamIndex],
    *,
    profile: str | None = None,
    yam32: bool = False,
    instruction: str = "",
) -> EnvelopeVerdict:
    """Every NVIDIA input check for one episode, collected instead of raised.

    Stream checks: ``stream_checks`` per camera. Episode checks:

    * camera roles: ``main`` alone, or exactly ``main/left_wrist/right_wrist`` - wrists come both
      or neither (``api.py:16-17, 26-28``; mosaics index those keys, ``media.py:103-105``);
    * 3-view sync: identical ``rate``, full ``pts_us`` tuple and ``duration_us`` across cameras
      (``media.py:63-65``) and ``main.cfr`` (``media.py:66-67``).

    Profile checks (only when ``profile`` is given; api-style names human/yam/dex/tianji):
    profile name (``api.py:13-14``), human takes one view (``api.py:18-19``,
    ``pipeline.py:146-147``), yam32 needs YAM + 3 views (``api.py:22-23``,
    ``pipeline.py:148-151``), instruction <= 2000 chars after strip (``api.py:24-25``), and
    single-view robot captioning needs CFR (``pipeline.py:73-74``; raised at caption time).
    """
    reasons: list[str] = []
    for role in sorted(streams):
        reasons += [f"{role}:{r}" for r in stream_checks(streams[role])]
    roles = set(streams)
    if roles not in ({"main"}, THREE_VIEW_ROLES):
        reasons.append("camera_roles_invalid")
    main = streams.get("main")
    if main is not None and len(streams) == 3:
        for s in streams.values():
            if s.rate != main.rate or s.pts_us != main.pts_us or s.duration_us != main.duration_us:
                reasons.append("views_not_synchronized")
                break
        if not main.cfr:
            reasons.append("three_view_not_cfr")
    if profile is not None:
        canonical = _PROFILES.get(profile.lower()) if isinstance(profile, str) else None
        if canonical is None:
            reasons.append("profile_invalid")
        if canonical == "Human" and len(streams) > 1:
            reasons.append("human_profile_multi_view")
        if yam32 and (canonical != "YAM" or len(streams) != 3):
            reasons.append("yam32_requires_yam_three_view")
        if len(instruction.strip()) > INSTRUCTION_MAX_CHARS:
            reasons.append("instruction_too_long")
        if canonical not in (None, "Human") and len(streams) == 1 and main and not main.cfr:
            reasons.append("single_view_robot_not_cfr")
    return EnvelopeVerdict(ok=not reasons, reasons=tuple(reasons))


def index_episode(
    episode_id: str,
    dataset: str,
    cameras: Mapping[CameraRole, SourceRef],
    *,
    task: str | None = None,
    profile: str | None = None,
    yam32: bool = False,
    scans: dict[str, StreamScan] | None = None,
    decode: bool = False,
) -> EpisodeIndex:
    """Index every camera of an episode and attach the envelope verdict.

    ``scans`` is a per-file cache (``uri -> StreamScan``) so a concatenated file is demuxed once
    for all of its episodes; pass the same dict across calls.
    """
    scans = scans if scans is not None else {}
    streams: dict[CameraRole, StreamIndex] = {}
    for role, ref in cameras.items():
        if ref.uri not in scans:
            scans[ref.uri] = scan_stream(ref.uri)
        streams[role] = index_stream(ref, scan=scans[ref.uri], decode=decode)
    verdict = envelope_checks(streams, profile=profile, yam32=yam32, instruction=task or "")
    return EpisodeIndex(
        episode_id=episode_id, dataset=dataset, streams=streams, task=task, envelope=verdict
    )


class SeekOvershoot(MediaError):
    """A seek landed after the episode's first packet; callers retry without seeking."""


def demux_range(container, stream, first_dts: int, last_dts: int, *, seek: bool):
    """Yield the source packets with ``first_dts <= dts <= last_dts`` (decode order).

    With ``seek`` the demuxer is first positioned on the keyframe at or before ``first_dts``;
    if that lands past the range, ``SeekOvershoot`` is raised before anything is yielded.
    """
    if seek:
        container.seek(first_dts, stream=stream, backward=True, any_frame=False)
    started = False
    for pkt in container.demux(stream):
        if not pkt.size or pkt.dts is None:
            continue
        if not started:
            if pkt.dts < first_dts:
                continue
            if pkt.dts > first_dts:
                raise SeekOvershoot(f"{container.name}: positioned past dts {first_dts}")
            started = True
        done = pkt.dts == last_dts
        yield pkt
        if done:
            return
    if not started:
        raise MediaError(f"{container.name}: packet with dts {first_dts} not found")
