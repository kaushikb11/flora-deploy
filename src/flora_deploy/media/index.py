"""Per-episode probe equivalent and NVIDIA envelope checks (PLAN.md D7, D13; PLAN-v3.md §1.1).

Contract: for an **oracle-eligible** episode range of a (possibly concatenated) MP4,
``index_stream(source)`` returns exactly what NVIDIA's ``batch_utils.media.probe``
(``media.py:25-58``) returns when run on ``cuts.stream_copy_cut(source, ...)``, and
``rangedecode.decode_selected`` returns the frames NVIDIA's ``decode_selected`` returns on that cut.
NVIDIA's code is never imported here; the rules are re-expressed and cited by file:line of the
pinned Flora-TempSeg ``batch_utils`` snapshot. The ``-m nvlink`` tests check the equality field by
field against the real ``probe``.

An episode is oracle-eligible (``StreamIndex.starts_on_keyframe``, empty ``ineligible``) only when
a stream-copy cut is guaranteed to decode to the same frames as the source. Everything else fails
closed with a machine-readable reason (``INELIGIBILITY_REASONS``):

* ``not_keyframe``: the first packet in decode order is not a sync sample;
* ``not_random_access``: it is a sync sample but not a real random-access point (H.264 non-IDR
  recovery point / intra refresh, HEVC non-IRAP, AV1 frame that is not a shown ``KEY_FRAME``,
  or an uninspectable codec whose decode cross-check failed);
* ``leading_pictures``: a packet of the range presents before the first one (open GOP);
* ``non_contiguous``: other episodes' packets sit between the range's packets in decode order;
* ``edit_list``: a packet in the cut's span carries ``AV_PKT_FLAG_DISCARD`` (MP4 edit-list
  trimming), which a remux cannot reproduce;
* ``decode_error`` / ``decoder_mismatch``: only from ``decode=True``: the decoder raised, or its
  output differs from the packet-level prediction.

How each value of the cut is predicted from the source's packets (evidence: docs/media-notes.md):

* The cut holds exactly the episode's packets, in decode order, timestamps shifted by a constant,
  and packet durations copied (time base: the source's, scaled by movenc below 10000).
* ``rate``: FFmpeg's mov demuxer sets ``avg_frame_rate = av_reduce(timescale * n_samples,
  sum(stts durations), INT_MAX)``. The MP4 muxer writes each sample's stts duration as the dts
  delta to the next sample, and the last one as that packet's ``duration``. All of those equal the
  ``duration`` values the source demuxer reports for the same packets, so the per-episode rate is
  the formula over the episode's own packet durations - not the concatenated file's
  ``stream.average_rate``.
* ``pts_us``: presentation order is ascending pts; relative to the first presented frame.
* Tail: the last presented frame's ``frame.duration`` is the duration of the packet that carried
  it (the source packet with the largest pts), else ``1/rate`` (``media.py:48,51``).

For ineligible episodes the index describes the frames the *source* decoder outputs for the range
(discarded packets excluded; when the file has no random-access point before the range, the
decoder's actual output), which is what ``rangedecode`` produces from a pre-roll.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from fractions import Fraction

import av
import numpy as np

from flora_deploy.media.avsafe import open_input
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

# Oracle-ineligibility reasons (StreamIndex.ineligible).
NOT_KEYFRAME = "not_keyframe"
NOT_RANDOM_ACCESS = "not_random_access"
LEADING_PICTURES = "leading_pictures"
NON_CONTIGUOUS = "non_contiguous"
EDIT_LIST = "edit_list"
DECODE_ERROR = "decode_error"
DECODER_MISMATCH = "decoder_mismatch"
INELIGIBILITY_REASONS = (
    NOT_KEYFRAME,
    NOT_RANDOM_ACCESS,
    LEADING_PICTURES,
    NON_CONTIGUOUS,
    EDIT_LIST,
    DECODE_ERROR,
    DECODER_MISMATCH,
)

# Per-packet flags in StreamScan.flags.
FLAG_KEY = 1  # container sync sample (AV_PKT_FLAG_KEY / stss)
FLAG_DISCARD = 2  # AV_PKT_FLAG_DISCARD: decoded but never output (edit-list trimming)
FLAG_RAP = 4  # keyframe verified as a random-access point by bitstream inspection
FLAG_RAP_UNKNOWN = 8  # keyframe of a codec we do not inspect; a decode cross-check decides


class MediaError(ValueError):
    """A source cannot be indexed, cut or decoded the way the contract requires."""


class NotKeyframeAligned(MediaError):
    """No stream-copy cut of the range decodes like the source: the episode is not
    oracle-eligible. ``reasons`` holds the ``INELIGIBILITY_REASONS`` that apply."""

    def __init__(self, message: str, reasons: tuple[str, ...] = ()):
        super().__init__(message)
        self.reasons = tuple(reasons)


class SeekOvershoot(MediaError):
    """A seek landed after the span's first packet; callers retry without seeking."""


# ---------------------------------------------------------------------------------------------
# Random-access-point inspection (bitstream; cheap, keyframe packets only)


def _nal_units(data: bytes, length_size: int | None) -> Iterator[bytes]:
    if length_size:  # ISO/IEC 14496-15 length-prefixed NAL units (avcC / hvcC)
        i, n = 0, len(data)
        while i + length_size <= n:
            size = int.from_bytes(data[i : i + length_size], "big")
            i += length_size
            if size <= 0 or i + size > n:
                return
            yield data[i : i + size]
            i += size
    else:  # Annex B start codes
        for chunk in data.split(b"\x00\x00\x01")[1:]:
            if chunk:
                yield chunk


def _h264_is_rap(data: bytes, length_size: int | None) -> bool:
    """IDR (nal_unit_type 5) before any non-IDR slice. Recovery-point I-frames are not RAPs."""
    for nal in _nal_units(data, length_size):
        t = nal[0] & 0x1F
        if t == 5:
            return True
        if 1 <= t <= 4:
            return False
    return False


def _hevc_is_rap(data: bytes, length_size: int | None) -> bool:
    """First VCL NAL is an IRAP (BLA/IDR/CRA, types 16-21). Leading pictures of a CRA/BLA are
    rejected separately (they present before the IRAP: ``leading_pictures``)."""
    for nal in _nal_units(data, length_size):
        if len(nal) < 2:
            continue
        t = (nal[0] >> 1) & 0x3F
        if t <= 31:
            return 16 <= t <= 21
    return False


def _leb128(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    for k in range(8):
        if i >= len(data):
            break
        b = data[i]
        i += 1
        value |= (b & 0x7F) << (7 * k)
        if not b & 0x80:
            break
    return value, i


def _av1_obus(data: bytes) -> Iterator[tuple[int, bytes]]:
    i = 0
    while i < len(data):
        h = data[i]
        obu_type, ext, has_size = (h >> 3) & 0xF, (h >> 2) & 1, (h >> 1) & 1
        i += 1 + ext
        if has_size:
            size, i = _leb128(data, i)
        else:
            size = len(data) - i
        yield obu_type, data[i : i + size]
        i += size


def _av1_reduced_still(obus: bytes) -> bool:
    for t, payload in _av1_obus(obus):
        if t == 1 and payload:  # sequence header: seq_profile(3) still(1) reduced_still(1)
            return bool(payload[0] & 0x08)
    return False


def _av1_is_rap(data: bytes, reduced_still: bool) -> bool:
    """The temporal unit's first frame is a shown ``KEY_FRAME``. ``show_existing_frame`` (incl.
    forward-key-frame display), hidden key frames and intra-only frames are not RAPs."""
    for t, payload in _av1_obus(data):
        if t == 1 and payload:
            reduced_still = bool(payload[0] & 0x08)
        elif t in (3, 6):  # OBU_FRAME_HEADER, OBU_FRAME
            if reduced_still:
                return True
            if not payload:
                return False
            b = payload[0]
            if b & 0x80:  # show_existing_frame
                return False
            return (b >> 5) & 3 == 0 and (b >> 4) & 1 == 1  # KEY_FRAME and show_frame
    return False


def rap_classifier(codec: str, extradata: bytes | None) -> Callable[[bytes], bool] | None:
    """Packet -> "is a random-access point" for codecs we inspect; None for the others."""
    ex = bytes(extradata or b"")
    if codec == "h264":
        ls = (ex[4] & 3) + 1 if len(ex) > 4 and ex[0] == 1 else None
        return lambda d: _h264_is_rap(d, ls)
    if codec == "hevc":
        ls = (ex[21] & 3) + 1 if len(ex) > 22 and ex[0] == 1 else None
        return lambda d: _hevc_is_rap(d, ls)
    if codec == "av1":
        reduced = _av1_reduced_still(ex[4:]) if len(ex) > 4 and ex[0] & 0x80 else False
        return lambda d: _av1_is_rap(d, reduced)
    return None


# ---------------------------------------------------------------------------------------------
# Packet scan (one demux pass per file, shared by every episode in a concatenated file)


@dataclass(frozen=True)
class Packet:
    pts: int
    dts: int
    duration: int
    keyframe: bool
    size: int
    discard: bool = False


@dataclass(frozen=True, eq=False)
class StreamScan:
    """Demux-only view of a file's video stream (no decoding), as parallel arrays.

    Arrays are in decode (file) order and hold payload-bearing packets that have pts and dts.
    About 50 bytes per packet including the derived lookup arrays.
    """

    path: str
    time_base: Fraction
    codec: str
    pix_fmt: str
    width: int
    height: int
    video_streams: int
    colorspace: int | None
    color_range: int | None
    pts: np.ndarray  # int64
    dts: np.ndarray  # int64
    duration: np.ndarray  # int64
    size: np.ndarray  # int64
    flags: np.ndarray  # uint8, FLAG_*
    missing_pts: bool  # a payload packet had no pts
    stream_index: int = 0  # container stream index of the scanned video stream
    # Derived (filled in __post_init__).
    order: np.ndarray = field(init=False, repr=False)  # decode indices sorted by pts
    sorted_pts: np.ndarray = field(init=False, repr=False)
    half_frame: Fraction = field(init=False, repr=False)  # selection tolerance, seconds
    discard_cum: np.ndarray = field(init=False, repr=False)  # prefix count of discards
    rap_index: np.ndarray = field(init=False, repr=False)  # decode indices of RAP keyframes
    _checks: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.pts)
        if not all(len(a) == n for a in (self.dts, self.duration, self.size, self.flags)):
            raise MediaError("StreamScan arrays differ in length")
        order = np.argsort(self.pts, kind="stable")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "sorted_pts", self.pts[order])
        pos = self.duration[self.duration > 0]
        if len(pos):
            med = int(np.sort(pos)[len(pos) // 2])
            half = Fraction(med, 2) * self.time_base
        else:
            half = Fraction(1, 10_000)
        object.__setattr__(self, "half_frame", half)
        cum = np.zeros(n + 1, dtype=np.int64)
        np.cumsum((self.flags & FLAG_DISCARD) != 0, out=cum[1:])
        object.__setattr__(self, "discard_cum", cum)
        rap = np.flatnonzero(
            ((self.flags & FLAG_KEY) != 0) & ((self.flags & (FLAG_RAP | FLAG_RAP_UNKNOWN)) != 0)
        )
        object.__setattr__(self, "rap_index", rap)

    def __len__(self) -> int:
        return len(self.pts)

    @property
    def nbytes(self) -> int:
        arrays = (self.pts, self.dts, self.duration, self.size, self.flags)
        derived = (self.order, self.sorted_pts, self.discard_cum, self.rap_index)
        return sum(a.nbytes for a in arrays + derived)

    @property
    def packets(self) -> tuple[Packet, ...]:
        """Per-packet view (convenience; builds Python objects - avoid on large files)."""
        return tuple(
            Packet(int(p), int(d), int(u), bool(f & FLAG_KEY), int(s), bool(f & FLAG_DISCARD))
            for p, d, u, s, f in zip(
                self.pts, self.dts, self.duration, self.size, self.flags, strict=True
            )
        )


def _video_stream(container, path) -> tuple[av.video.stream.VideoStream, int]:
    disp = av.stream.Disposition
    real = [s for s in container.streams.video if not (s.disposition & disp.attached_pic)]
    if not real:
        raise MediaError(f"{path}: no video stream")
    if len(real) > 1:
        raise MediaError(
            f"{path}: {len(real)} video streams; one camera per file is required (which stream "
            "is the camera is ambiguous)"
        )
    return real[0], len(real)


def scan_stream(path: str | os.PathLike[str]) -> StreamScan:
    """Demux every packet of the video stream of ``path`` (MP4/MOV); attached cover art is
    ignored, several real video streams are an error."""
    with open_input(path, options=NV_OPEN_OPTIONS) as container:
        stream, n_video = _video_stream(container, path)
        tb = stream.time_base
        if tb is None or tb.numerator != 1:
            # MP4/MOV tracks always use 1/timescale; other containers are not supported.
            raise MediaError(f"{path}: unsupported time_base {tb} (MP4/MOV sources only)")
        cc = stream.codec_context
        codec = cc.codec.canonical_name
        is_rap = rap_classifier(codec, cc.extradata)
        pts, dts, dur, size, flags = [], [], [], [], []
        missing = False
        for pkt in container.demux(stream):
            if not pkt.size:
                continue  # demuxer flush packets
            if pkt.pts is None or pkt.dts is None:
                missing = True
                continue
            f = FLAG_DISCARD if pkt.is_discard else 0
            if pkt.is_keyframe:
                f |= FLAG_KEY
                if is_rap is None:
                    f |= FLAG_RAP_UNKNOWN
                elif is_rap(bytes(pkt)):
                    f |= FLAG_RAP
            pts.append(pkt.pts)
            dts.append(pkt.dts)
            dur.append(pkt.duration or 0)
            size.append(pkt.size)
            flags.append(f)
        return StreamScan(
            path=str(path),
            time_base=Fraction(tb),
            codec=codec,
            pix_fmt=str(cc.pix_fmt or ""),
            width=cc.width,
            height=cc.height,
            video_streams=n_video,
            colorspace=_int_or_none(getattr(cc, "colorspace", None)),
            color_range=_int_or_none(getattr(cc, "color_range", None)),
            pts=np.array(pts, dtype=np.int64),
            dts=np.array(dts, dtype=np.int64),
            duration=np.array(dur, dtype=np.int64),
            size=np.array(size, dtype=np.int64),
            flags=np.array(flags, dtype=np.uint8),
            missing_pts=missing,
            stream_index=stream.index,
        )


def _int_or_none(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class ScanCache:
    """Bounded ``uri -> StreamScan`` cache (LRU by retained array bytes).

    Episodes of a concatenated file are usually indexed consecutively, so a budget of a few
    files' scans gives one demux pass per file without holding a whole dataset in memory.
    """

    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._scans: OrderedDict[str, StreamScan] = OrderedDict()
        self.scans_done = 0  # number of demux passes performed (for tests / stats)

    def get(self, uri: str) -> StreamScan:
        scan = self._scans.get(uri)
        if scan is not None:
            self._scans.move_to_end(uri)
            return scan
        scan = scan_stream(uri)
        self.scans_done += 1
        self._scans[uri] = scan
        while len(self._scans) > 1 and self.nbytes > self.max_bytes:
            self._scans.popitem(last=False)
        return scan

    def drop(self, uri: str) -> None:
        self._scans.pop(uri, None)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._scans.values())

    def __contains__(self, uri: object) -> bool:
        return uri in self._scans

    def __len__(self) -> int:
        return len(self._scans)


ScanStore = ScanCache | MutableMapping[str, StreamScan]


def get_scan(scans: ScanStore | None, uri: str) -> StreamScan:
    if scans is None:
        return scan_stream(uri)
    if isinstance(scans, ScanCache):
        return scans.get(uri)
    if uri not in scans:
        scans[uri] = scan_stream(uri)
    return scans[uri]


# ---------------------------------------------------------------------------------------------
# Episode selection and plan


@dataclass(frozen=True)
class EpisodeSlice:
    """Decode-order span of the episode's packets: ``first..last`` (inclusive)."""

    first: int
    last: int
    contiguous: bool  # the range's packets are one contiguous run in decode order


@dataclass(frozen=True, eq=False)
class EpisodePlan:
    """Everything index, cut and range decode need to agree on for one episode range."""

    selected: np.ndarray  # decode indices whose pts lies in the range (ascending)
    first: int
    last: int
    contiguous: bool
    frames: np.ndarray  # decode indices of the output frames (no discards), in pts order
    preroll: int | None  # where source decoding starts (None: from 0, output must be measured)
    reasons: tuple[str, ...]  # oracle-ineligibility reasons; () = a cut is exact

    @property
    def eligible(self) -> bool:
        return not self.reasons

    @property
    def slice(self) -> EpisodeSlice:
        return EpisodeSlice(self.first, self.last, self.contiguous)


def _select(scan: StreamScan, start_s: float | None, end_s: float | None) -> np.ndarray:
    """Decode indices of packets presented in ``[start_s - h, end_s - h)`` (h = half frame).

    O(log N + k): integer bounds in time-base units and a bisection over the sorted pts.
    ``p.pts * tb >= x`` iff ``p.pts >= ceil(x / tb)``; ``p.pts * tb < y`` iff
    ``p.pts < ceil(y / tb)``.
    """
    n = len(scan)
    if not n:
        raise MediaError(f"{scan.path}: no packets")
    if start_s is None and end_s is None:
        return np.arange(n)
    tb, h = scan.time_base, scan.half_frame
    a = 0
    b = n
    if start_s is not None:
        lo = math.ceil((Fraction(start_s) - h) / tb)
        a = int(np.searchsorted(scan.sorted_pts, lo, side="left"))
    if end_s is not None:
        hi = math.ceil((Fraction(end_s) - h) / tb)
        b = int(np.searchsorted(scan.sorted_pts, hi, side="left"))
    if b <= a:
        raise MediaError(f"{scan.path}: no frames in [{start_s}, {end_s})")
    return np.sort(scan.order[a:b])


def select_episode(scan: StreamScan, start_s: float | None, end_s: float | None) -> EpisodeSlice:
    """Decode-order span of the packets presented in ``[start_s, end_s)`` of the file.

    LeRobot's ``from_timestamp``/``to_timestamp`` are float sums of per-episode durations, so the
    bounds are matched with a tolerance of half the file's median packet duration: a frame at
    ``t`` belongs to the episode iff ``start_s - h <= t < end_s - h``.
    """
    sel = _select(scan, start_s, end_s)
    first, last = int(sel[0]), int(sel[-1])
    return EpisodeSlice(first, last, len(sel) == last - first + 1)


def plan_episode(
    scan: StreamScan, start_s: float | None, end_s: float | None, *, cross_check: bool = True
) -> EpisodePlan:
    """Select the range and decide whether a stream-copy cut of it is exact (fail closed)."""
    sel = _select(scan, start_s, end_s)
    first, last = int(sel[0]), int(sel[-1])
    contiguous = len(sel) == last - first + 1
    keep = sel[(scan.flags[sel] & FLAG_DISCARD) == 0]
    if not len(keep):
        raise MediaError(f"{scan.path}: every frame in [{start_s}, {end_s}) is discarded")
    frames = keep[np.argsort(scan.pts[keep], kind="stable")]

    reasons: list[str] = []
    f = int(scan.flags[first])
    rap = False
    if not f & FLAG_KEY:
        reasons.append(NOT_KEYFRAME)
    elif f & FLAG_RAP:
        rap = True
    elif f & FLAG_RAP_UNKNOWN and cross_check:
        rap = _cross_check_rap(scan, first, last)
        if not rap:
            reasons.append(NOT_RANDOM_ACCESS)
    else:
        reasons.append(NOT_RANDOM_ACCESS)
    leading = int(scan.pts[first]) != int(scan.pts[sel].min())
    if leading:
        reasons.append(LEADING_PICTURES)
    if not contiguous:
        reasons.append(NON_CONTIGUOUS)
    if scan.discard_cum[last + 1] - scan.discard_cum[first]:
        reasons.append(EDIT_LIST)

    # Where source decoding starts for production range decode: the range's own start when it is
    # a clean random-access point, else the last random-access point strictly before it (None:
    # the file has none, so the decoder's real output must be measured from packet 0).
    if rap and not leading:
        preroll: int | None = first
    else:
        k = int(np.searchsorted(scan.rap_index, first, side="left"))
        preroll = int(scan.rap_index[k - 1]) if k else None
    return EpisodePlan(sel, first, last, contiguous, frames, preroll, tuple(reasons))


def _frame_digest(frame: av.VideoFrame) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for plane in frame.planes:
        h.update(bytes(plane))
    return h.digest()


def _cross_check_rap(scan: StreamScan, first: int, last: int) -> bool:
    """Decode cross-check for keyframes of codecs we do not parse: decoding the first GOP from
    ``first`` with a fresh decoder must output every packet's frame, and the same pixels as
    decoding from the previous keyframe."""
    key = ("rap", first, last)
    if key in scan._checks:
        return scan._checks[key]
    keys = np.flatnonzero(scan.flags & FLAG_KEY)
    k = int(np.searchsorted(keys, first, side="right"))
    end = min(last, int(keys[k]) - 1) if k < len(keys) else last
    end = max(end, first)
    span = np.arange(first, end + 1)
    span = span[(scan.flags[span] & FLAG_DISCARD) == 0]
    want = sorted(int(p) for p in scan.pts[span])
    ok = False
    try:
        fresh = [(f.pts, _frame_digest(f)) for f in decode_span(scan, first, end)]
        ok = [p for p, _ in fresh] == want
        j = int(np.searchsorted(keys, first, side="left"))
        if ok and j:
            wanted = set(want)
            ref = {
                f.pts: _frame_digest(f)
                for f in decode_span(scan, int(keys[j - 1]), end)
                if f.pts in wanted
            }
            ok = all(ref.get(p) == d for p, d in fresh)
    except (av.FFmpegError, MediaError):
        ok = False
    scan._checks[key] = ok
    return ok


# ---------------------------------------------------------------------------------------------
# Demux / decode of a packet span (shared by cuts, decode-mode index and range decode)


def demux_range(container, stream, scan: StreamScan, start: int, last: int, *, seek: bool):
    """Yield the source packets ``scan[start..last]`` (decode order).

    With ``seek`` the demuxer is first positioned on the keyframe at or before the start packet's
    **pts** (the mov demuxer seeks by presentation time; seeking by dts lands a GOP early with
    B-frames). If that lands past the span, ``SeekOvershoot`` is raised before anything is
    yielded.
    """
    first_dts, last_dts = int(scan.dts[start]), int(scan.dts[last])
    if seek:
        container.seek(int(scan.pts[start]), stream=stream, backward=True, any_frame=False)
    started = False
    count = 0
    for pkt in container.demux(stream):
        if not pkt.size or pkt.dts is None:
            continue
        if not started:
            if pkt.dts < first_dts:
                continue
            if pkt.dts > first_dts:
                raise SeekOvershoot(f"{container.name}: positioned past dts {first_dts}")
            started = True
        if pkt.pts is None:
            raise MediaError(f"{container.name}: a packet in the range has no pts")
        count += 1
        done = pkt.dts >= last_dts
        yield pkt
        if done:
            break
    if not started:
        raise MediaError(f"{container.name}: packet with dts {first_dts} not found")
    if count != last - start + 1:
        raise MediaError(f"{container.name}: demuxed {count} packets, expected {last - start + 1}")


def decode_span(scan: StreamScan, start: int, last: int) -> Iterator[av.VideoFrame]:
    """A fresh decoder (``err_detect=explode``, ``thread_count=1``, as ``media.py:76-78``) fed
    the source packets ``scan[start..last]`` with their flags, then drained."""
    for seek in (start > 0, False):
        with open_input(scan.path, options=NV_OPEN_OPTIONS) as container:
            stream = container.streams[scan.stream_index]
            stream.codec_context.thread_count = 1
            packets = demux_range(container, stream, scan, start, last, seek=seek)
            try:
                head = next(packets)
            except SeekOvershoot:
                continue
            yield from stream.decode(head)
            for pkt in packets:
                yield from stream.decode(pkt)
            yield from stream.decode(None)  # drain, like the cut's EOF
            return
    raise MediaError(f"{scan.path}: could not position the decoder on packet {start}")


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

    movenc uses ``time_base.den`` and doubles it while below 10000, so the cut's ``time_base``
    differs from the source's for timescales < 10000 (1/30 -> 1/15360). Scaling every duration
    and the timescale by the same factor leaves all rationals unchanged; only ``av_reduce``'s
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
    ``decode``: decode the cut's packets with a fresh decoder (as ``probe`` does on the cut) and
    take PTS and the tail from the output; an exception becomes ``decode_error`` and output that
    differs from the packet prediction becomes ``decoder_mismatch`` (both make the episode
    ineligible). Only for eligible ranges (``NotKeyframeAligned`` otherwise). Packet mode is the
    fast default.
    """
    scan = scan if scan is not None else scan_stream(source.uri)
    plan = plan_episode(scan, source.start_s, source.end_s)
    tb = scan.time_base
    reasons = list(plan.reasons)
    if decode and reasons:
        raise NotKeyframeAligned(
            f"{source.uri}: decode mode needs an oracle-eligible range ({', '.join(reasons)})",
            plan.reasons,
        )

    frames = plan.frames
    if plan.preroll is None:
        # No random-access point at or before the range: which frames the decoder outputs is
        # decoder behaviour (frames before the first decodable picture are dropped). Measure it.
        wanted = {int(p): int(i) for p, i in zip(scan.pts[frames], frames, strict=True)}
        got = [wanted[f.pts] for f in decode_span(scan, 0, plan.last) if f.pts in wanted]
        if not got:
            raise MediaError(f"{source.uri}: the decoder outputs no frame of the range")
        frames = np.array(got, dtype=np.int64)
    presented = [int(p) for p in scan.pts[frames]]
    last_dur = int(scan.duration[frames[-1]])
    samples = plan.selected if plan.eligible else frames  # the cut's samples (stts)

    if decode:
        try:
            out = [(f.pts, f.duration or 0) for f in decode_span(scan, plan.first, plan.last)]
        except av.FFmpegError:
            reasons.append(DECODE_ERROR)
        else:
            dec = [p for p, _ in out]
            if dec != presented:
                reasons.append(DECODER_MISMATCH)
                if dec:
                    presented = dec
            if out:
                last_dur = out[-1][1]

    # rate: mov demuxer formula over the cut's stts (see module docstring).
    ts = cut_timescale(tb)
    total = int(scan.duration[samples].sum()) * (ts // tb.denominator)
    rate = av_reduce(ts * len(samples), total) if total > 0 else Fraction(0)

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
        starts_on_keyframe=not reasons,
        source_timestamps_s=tuple(float(t * tb) for t in presented),
        time_base=tb,
        byte_size=int(scan.size[plan.selected].sum()),
        video_streams=1,  # a stream-copy cut carries exactly the one video stream
        missing_pts=scan.missing_pts,
        colorspace=scan.colorspace,
        color_range=scan.color_range,
        source_pts=tuple(presented),
        ineligible=tuple(reasons),
    )


# Bounds of the boxes movenc writes around a single-track cut's payload.
CUT_MIN_OVERHEAD = 8 + 8 + 100  # ftyp, mdat header and a minimal moov - a strict lower bound
CUT_FIXED_OVERHEAD = 4096
CUT_PER_SAMPLE_OVERHEAD = 32


def estimated_cut_bytes(stream: StreamIndex) -> int:
    """Conservative upper estimate of the cut file's size (payload + MP4 boxes).

    Exact payload; boxes bounded by 4 KiB fixed + 32 bytes per sample (stts/ctts/stsz/stco/stss
    entries plus chunking), which over-covers what movenc writes for a single-track file.
    """
    return stream.byte_size + CUT_FIXED_OVERHEAD + CUT_PER_SAMPLE_OVERHEAD * len(stream.pts_us)


def exact_cut_bytes(stream: StreamIndex, *, scan: StreamScan | None = None) -> int:
    """Size of the real cut, produced in a temporary directory (only for eligible ranges)."""
    import tempfile
    from pathlib import Path

    from flora_deploy.media.cuts import stream_copy_cut

    with tempfile.TemporaryDirectory(prefix="flora-cut-size-") as d:
        return stream_copy_cut(stream.source, Path(d) / "cut.mp4", scan=scan).stat().st_size


# ---------------------------------------------------------------------------------------------
# Envelope


def stream_checks(
    stream: StreamIndex, *, max_bytes: int = MAX_UPLOAD_BYTES, exact_size: bool = True
) -> list[str]:
    """Per-file reasons NVIDIA would reject this camera's cut (machine-readable ids).

    Mirrors, in this order: ``api.py:34-35`` (0 < size <= 256 MiB), ``media.py:27-28`` (one video
    stream), ``:31-33`` (rate >= 4), ``:34-36`` (1..4096 px), ``:39-40`` (every frame has a
    PTS), ``:43-44`` (strictly increasing PTS), ``:45-46`` (t < 120 s and at most 7200 frames),
    ``:49-50`` (>= 2 frames), ``:53-54`` (duration <= 120 s).

    Size: accepted when the upper estimate fits, rejected when even the payload plus the minimal
    boxes does not; in between (a few KiB below the cap) the cut of an eligible range is made in
    a temporary directory and measured (``exact_size``), otherwise rejected.
    """
    reasons: list[str] = []
    if stream.byte_size <= 0:
        reasons.append("file_empty")
    elif estimated_cut_bytes(stream) > max_bytes:
        too_big = True
        if (
            exact_size
            and stream.byte_size + CUT_MIN_OVERHEAD <= max_bytes
            and stream.starts_on_keyframe
        ):
            try:
                too_big = exact_cut_bytes(stream) > max_bytes
            except (MediaError, OSError):
                too_big = True
        if too_big:
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
    scans: ScanStore | None = None,
    decode: bool = False,
) -> EpisodeIndex:
    """Index every camera of an episode and attach the envelope verdict.

    ``scans``: a ``ScanCache`` (bounded LRU; recommended) or a plain dict (unbounded, caller
    manages it), shared across calls so a concatenated file is demuxed once for all of its
    episodes. ``decode``: decode-mode index for eligible cameras (oracle candidates), turning
    decoder errors into the ``decode_error`` ineligibility reason.
    """
    scans = scans if scans is not None else {}
    streams: dict[CameraRole, StreamIndex] = {}
    for role, ref in cameras.items():
        scan = get_scan(scans, ref.uri)
        s = index_stream(ref, scan=scan)
        if decode and s.starts_on_keyframe:
            s = index_stream(ref, scan=scan, decode=True)
        streams[role] = s
    verdict = envelope_checks(streams, profile=profile, yam32=yam32, instruction=task or "")
    return EpisodeIndex(
        episode_id=episode_id, dataset=dataset, streams=streams, task=task, envelope=verdict
    )
