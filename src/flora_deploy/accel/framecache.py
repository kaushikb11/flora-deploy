"""``decode_once``: decode each source frame at most once per episode (PLAN.md D12, D17).

NVIDIA's ``media.decode_selected`` (``batch_utils/media.py:69-95``) opens the file and decodes
from frame 0 to ``max(wanted)`` on every call, and ``frames_for`` (``media.py:97-106``) calls it
once per camera; ``Predictor.predict`` calls ``frames_for`` once per TempSeg window and
``caption_three`` once per caption plan, so an episode decodes its prefix O(windows) times.

The replacement keeps NVIDIA's per-frame semantics and only changes *how often* the decoder
runs:

- **Persistent forward cursor** per source file: one decoder, opened exactly as NVIDIA opens it
  (``err_detect=explode``, ``thread_count=1``, ``container.decode(stream)``), whose position is
  kept across calls; a request for frames at or after its position continues decoding instead of
  restarting.
- **Random-access restart** for requests behind the cursor: decoding restarts at the last
  bitstream-verified random-access point (``media.index`` FLAG_RAP: H.264 IDR, HEVC IRAP, AV1
  shown key frame) at or before every requested frame, instead of at frame 0. This is used only
  when the file's demuxed packets map one-to-one onto the probed ``Video.pts_us`` (no edit-list
  discards, same count, same relative timestamps); every frame the restarted decoder outputs is
  matched to its index by its exact pts, and any surprise (unknown pts, a skipped requested
  frame, a demux/decode error, a seek overshoot) falls back to NVIDIA's own decode-from-0 path.
- **Byte-bounded LRU** of converted frames (1 view) or composed mosaics (3 views), so frames
  shared by overlapping windows / caption plans are converted once.
- The source file is hashed once and must equal ``Video.sha256`` (the probe's hash); otherwise
  the call is delegated to NVIDIA's original ``frames_for`` untouched.

Per frame the output is NVIDIA's: the same timestamp check, the same canonical-RGB vs
``to_image().convert('RGB')`` branch (NVIDIA's own ``canonical_rgb`` is called), NVIDIA's own
composers, the same dict keys and order, and the same ``InputError`` messages for bad requests.
"""

from __future__ import annotations

import bisect
import contextvars
import hashlib
import inspect
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import av
import numpy as np

from flora_deploy.media.index import (
    FLAG_DISCARD,
    FLAG_RAP,
    MediaError,
    SeekOvershoot,
    demux_range,
    scan_stream,
)
from flora_deploy.media.rangedecode import canonical_eligible, swscale_rgb

US = 1_000_000
NV_OPEN_OPTIONS = {"err_detect": "explode"}  # media.py:76
SEEK_OVERHEAD_FRAMES = 2  # a restart costs about this many frames of decode beyond the GOP
MIN_CHUNK_FRAMES = 48  # smallest run worth its own decoder in GOP-parallel decode

MSG_OUTSIDE = "A requested source frame is outside the video."  # media.py:75
MSG_PTS = "Source frame timestamps changed while decoding."  # media.py:84
MSG_MISSING = "A requested source frame could not be decoded."  # media.py:94
MSG_LAYOUT = "Choose a supported camera layout."  # media.py:106


class _Fallback(Exception):
    """The random-access cursor cannot serve the request; use the decode-from-0 path."""


@dataclass(frozen=True)
class SeekMap:
    """Frame index <-> packet mapping of a file whose packets are exactly the probed frames."""

    zero: int  # pts of frame 0 (smallest pts)
    sorted_pts: np.ndarray  # pts of frame i
    decode_index: np.ndarray  # decode-order packet index of frame i
    pts_index: dict[int, int]  # pts -> frame index
    raps: list[int]  # decode indices of verified random-access points (ascending)
    rap_pts: list[int]
    scan: Any


def build_seekmap(path: str, video) -> SeekMap | None:
    """``SeekMap`` when random-access restarts are provably the same frames as decoding from 0
    (else ``None``)."""
    try:
        scan = scan_stream(path)
    except (MediaError, av.FFmpegError, OSError):
        return None
    n = len(video.pts_us)
    if scan.missing_pts or scan.video_streams != 1 or len(scan) != n or n == 0:
        return None
    if (scan.flags & FLAG_DISCARD).any():
        return None
    sp = scan.sorted_pts
    if len(np.unique(sp)) != n:
        return None
    zero = int(sp[0])
    tb = scan.time_base
    for i in range(n):
        if Fraction(int(sp[i]) - zero) * tb * US != video.pts_us[i]:
            return None
    raps = [int(s) for s in scan.rap_index if scan.flags[s] & FLAG_RAP]
    return SeekMap(
        zero=zero,
        sorted_pts=sp,
        decode_index=scan.order,
        pts_index={int(p): i for i, p in enumerate(sp)},
        raps=raps,
        rap_pts=[int(scan.pts[s]) for s in raps],
        scan=scan,
    )


class _Cursor:
    """One open decoder positioned after frame ``next - 1``."""

    def __init__(self, path: str, seekmap: SeekMap | None = None, start_packet: int = 0):
        self.container = av.open(path, options=NV_OPEN_OPTIONS)
        try:
            self.stream = self.container.streams.video[0]
            self.stream.codec_context.thread_count = 1
            self.time_base = self.stream.time_base
            self.seeked = start_packet > 0
            self.next = 0
            self.decoded = 0  # frames pulled from the decoder
            if self.seeked:
                assert seekmap is not None
                if self.stream.index != seekmap.scan.stream_index:
                    raise _Fallback("stream index changed")
                self.seekmap = seekmap
                self.zero: int | None = seekmap.zero
                self._it = self._from_packet(seekmap.scan, start_packet)
            else:
                self.seekmap = None
                self.zero = None
                self._it = iter(self.container.decode(self.stream))
        except BaseException:
            self.container.close()
            raise

    def _from_packet(self, scan, start: int):
        packets = demux_range(self.container, self.stream, scan, start, len(scan) - 1, seek=True)
        for pkt in packets:
            yield from self.stream.decode(pkt)
        yield from self.stream.decode(None)

    def pull(self) -> tuple[int, av.VideoFrame]:
        """Next decoded frame and its index (``StopIteration`` at the end)."""
        if not self.seeked:
            frame = next(self._it)
            i = self.decoded
            self.decoded += 1
            self.zero = frame.pts if self.zero is None else self.zero
            self.next = i + 1
            return i, frame
        try:
            frame = next(self._it)
        except StopIteration:
            raise
        except (SeekOvershoot, MediaError, av.FFmpegError) as exc:
            raise _Fallback(str(exc)) from exc
        self.decoded += 1
        i = self.seekmap.pts_index.get(frame.pts) if frame.pts is not None else None
        if i is None or i < self.next - 1:
            raise _Fallback("unexpected frame from a random-access restart")
        self.next = i + 1
        return i, frame

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:  # noqa: BLE001 - closing a failed decoder
            pass


def nv_convert(frame: av.VideoFrame, canonical: bool, canonical_rgb: Callable | None):
    """NVIDIA's colour branch (``media.py:85-90``) with NVIDIA's own ``canonical_rgb``."""
    if canonical and canonical_eligible(frame):
        return canonical_rgb(frame)
    return swscale_rgb(frame)


class SourceDecoder:
    """Frames of one probed source file (thread-safe; one request at a time)."""

    def __init__(self, video, state, canonical_rgb: Callable[[], Callable], errors):
        self.video = video
        self.path = str(video.path)
        self.n = len(video.pts_us)
        self.state = state
        self._canonical_rgb = canonical_rgb
        self.InputError = errors
        self.lock = threading.Lock()
        self._cursor: _Cursor | None = None
        self._seekmap: SeekMap | None | bool = False  # False = not built yet
        self.trusted = _file_sha256(self.path) == video.sha256

    def close(self) -> None:
        with self.lock:
            if self._cursor is not None:
                self._cursor.close()
                self._cursor = None
        self.state.note_closed(self)

    def close_if_idle(self) -> bool:
        """Close the open file unless a request is running (``AccelState.max_open_decoders``);
        a later request reopens it."""
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self._cursor is None:
                return False
            self._cursor.close()
            self._cursor = None
        finally:
            self.lock.release()
        self.state.note_closed(self)
        return True

    @property
    def seekmap(self) -> SeekMap | None:
        if self._seekmap is False:
            self._seekmap = build_seekmap(self.path, self.video) if self.state.seek else None
            self.state.count("seekmap_built" if self._seekmap else "seekmap_unavailable")
        return self._seekmap  # type: ignore[return-value]

    def _start_for(self, missing: list[int]) -> tuple[int, int]:
        """(start packet, its frame index) for a restart serving ``missing`` (sorted)."""
        sm = self.seekmap
        if sm is None or not sm.raps:
            return 0, 0
        dmin = int(min(sm.decode_index[i] for i in missing))
        p0 = int(sm.sorted_pts[missing[0]])
        k = bisect.bisect_right(sm.raps, dmin) - 1
        while k >= 0 and sm.rap_pts[k] > p0:
            k -= 1
        if k < 0 or sm.raps[k] == 0:
            return 0, 0
        return sm.raps[k], sm.pts_index[sm.rap_pts[k]]

    def _open(self, start_packet: int) -> _Cursor:
        if start_packet > 0:
            try:
                cur = _Cursor(self.path, self.seekmap, start_packet)
                self.state.count("decoder_restarts_random_access")
                return cur
            except _Fallback:
                pass
        self.state.count("decoder_restarts_from_0")
        return _Cursor(self.path)

    def decode(self, missing: list[int], canonical: bool) -> dict[int, Any]:
        """Converted frames for sorted, in-range, distinct ``missing`` (NVIDIA semantics)."""
        try:
            with self.lock:
                return self._decode(missing, canonical)
        finally:
            if self._cursor is not None:
                self.state.note_open(self)
            else:
                self.state.note_closed(self)

    def _decode(self, missing: list[int], canonical: bool) -> dict[int, Any]:
        crgb = self._canonical_rgb() if canonical else None
        cur, self._cursor = self._cursor, None
        first = missing[0]
        forward = first - cur.next if cur is not None and cur.next <= first else None
        start, start_frame = self._start_for(missing)
        restart = first - start_frame + (SEEK_OVERHEAD_FRAMES if start else 0)
        fresh = forward is None or restart < forward
        chunks = self._chunks(missing) if fresh else None
        futures: dict[int, Any] = {}
        if chunks is not None and len(chunks) > 1:
            # GOP-parallel: independent decoders, each from a verified random-access point
            if cur is not None:
                cur.close()
            self.state.count("parallel_chunks", len(chunks))
            futs = [self.state.submit_decode(self._run_chunk, c, canonical, crgb) for c in chunks]
            last_cur = None
            error: BaseException | None = None
            for fut in futs:
                try:
                    chunk_cur, chunk_futures = fut.result()
                except BaseException as exc:  # noqa: BLE001 - re-raised in chunk order
                    error = error or exc
                    continue
                futures.update(chunk_futures)
                if last_cur is not None:
                    last_cur.close()
                last_cur = chunk_cur
            if error is not None:
                if last_cur is not None:
                    last_cur.close()
                raise error
            self._cursor = last_cur
        else:
            if fresh:
                if cur is not None:
                    cur.close()
                cur = self._open(start)
            self._cursor = self._run(cur, missing, canonical, crgb, futures)
        self.state.count("frames_converted", len(futures))
        return {i: futures[i].result() for i in sorted(futures)}

    def _chunks(self, missing: list[int]) -> list[list[int]] | None:
        """Split a request into runs that each start at their own random-access point."""
        k = self.state.workers
        if k <= 1 or len(missing) < 2 * MIN_CHUNK_FRAMES:
            return None
        sm = self.seekmap
        if sm is None:
            return None
        span = missing[-1] - missing[0] + 1
        k = min(k, span // MIN_CHUNK_FRAMES)
        if k <= 1:
            return None
        chunks: list[list[int]] = []
        bound = 0
        for j in range(1, k):
            m = bisect.bisect_left(missing, missing[0] + j * span // k, lo=bound)
            if m <= bound or m >= len(missing):
                continue
            if self._start_for(missing[m:])[0] == 0:
                continue  # no random-access point to start this run from
            chunks.append(missing[bound:m])
            bound = m
        chunks.append(missing[bound:])
        return chunks

    def _run_chunk(self, chunk: list[int], canonical: bool, crgb):
        futures: dict[int, Any] = {}
        start, _ = self._start_for(chunk)
        cur = self._run(self._open(start), chunk, canonical, crgb, futures)
        return cur, futures

    def _run(self, cur: _Cursor, wanted: list[int], canonical: bool, crgb,
             futures: dict[int, Any]) -> _Cursor:
        """Pull frames until every index of sorted ``wanted`` is converted (or raise); returns
        the cursor, positioned after the last one."""
        pending = set(wanted)
        lo = 0  # wanted[lo] is the lowest pending index
        try:
            while pending:
                try:
                    i, frame = cur.pull()
                except StopIteration:
                    if not cur.seeked:
                        break
                    cur.close()
                    cur = self._open(0)
                    continue
                except _Fallback:
                    self.state.count("random_access_fallbacks")
                    cur.close()
                    cur = self._open(0)
                    continue
                self.state.count("frames_decoded")
                if i in pending:
                    if Fraction(frame.pts - cur.zero) * cur.time_base * US != self.video.pts_us[i]:
                        raise self.InputError(MSG_PTS)
                    futures[i] = self.state.submit(nv_convert, frame, canonical, crgb)
                    pending.discard(i)
                    while lo < len(wanted) and wanted[lo] not in pending:
                        lo += 1
                if cur.seeked and pending and wanted[lo] < i:
                    # the restarted decoder skipped a requested frame: NVIDIA's path from 0
                    self.state.count("random_access_fallbacks")
                    cur.close()
                    cur = self._open(0)
            if pending:
                raise self.InputError(MSG_MISSING)
        except BaseException:
            cur.close()
            raise
        return cur

def _file_sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _video_key(video) -> tuple:
    return (str(video.path), video.sha256, video.rate, video.duration_us, video.pts_us)


@dataclass(frozen=True)
class SourceKey:
    """Cheap-to-hash token for one distinct probed ``Video`` (hashing ``pts_us`` is not)."""

    path: str
    sha256: str
    token: int


class FrameStore:
    """Per-episode-scope registry of ``SourceDecoder``s plus the scope's frame LRU
    (``AccelState.frame_store`` hands out a router to the current scope's store)."""

    def __init__(self, nv, state, scope=None):
        self.nv = nv
        self.state = state
        self.scope = scope if scope is not None else state.scope()
        self._lock = threading.Lock()
        self._keys: dict[int, tuple[Any, SourceKey]] = {}
        self._tokens: dict[tuple, SourceKey] = {}
        self._demand: dict[tuple, Counter] = {}  # unit -> announced future uses per index

    def _canonical_rgb(self) -> Callable:
        # NVIDIA imports it lazily as a top-level module (media.py:87); so do we, from the
        # same snapshot.
        return self.nv.vendor("canonical_rgb").canonical_rgb

    def key(self, video) -> SourceKey:
        with self._lock:
            memo = self._keys.get(id(video))
            if memo is not None and memo[0] is video:
                return memo[1]
            full = _video_key(video)
            k = self._tokens.get(full)
            if k is None:
                k = SourceKey(str(video.path), video.sha256, len(self._tokens))
                self._tokens[full] = k
            self._keys[id(video)] = (video, k)
            return k

    def decoder(self, video) -> SourceDecoder:
        key = self.key(video)
        with self._lock:
            dec = self.scope.decoders.get(key)
            if dec is None:
                errors = self.nv.media.InputError
                dec = SourceDecoder(video, self.state, self._canonical_rgb, errors)
                self.scope.decoders[key] = dec
                self.state.count("sources_opened")
            return dec

    def reset_keys(self) -> None:
        with self._lock:
            self._keys.clear()
            self._tokens.clear()
            self._demand.clear()

    # -- units: what one frames_for call returns per index ---------------------------------
    def unit(self, videos, mosaic: str) -> tuple | None:
        """Cache namespace of ``frames_for(videos, ., mosaic)`` outputs (None: no caching)."""
        canonical = len(videos) == 3  # media.py:98
        if len(videos) == 1:
            video = videos.get("main")
            return None if video is None else ("frame", self.key(video), canonical)
        if mosaic not in ("YAM", "Dex"):
            return None
        return ("mosaic", tuple((r, self.key(v)) for r, v in videos.items()), mosaic, canonical)

    @staticmethod
    def unit_bytes(videos, mosaic: str) -> int:
        if len(videos) == 1:
            w, h = next(iter(videos.values())).size
            return w * h * 3
        return 768 * 648 * 3 if mosaic == "YAM" else 640 * 720 * 3

    # -- lookahead -------------------------------------------------------------------------
    def hint(self, videos, mosaic: str, requests) -> None:
        """Announce frames_for requests that NVIDIA's code is about to make (the windows of one
        TempSeg request, the plans of one caption_three call). Only steers what one decoder
        pass prefetches; never changes a returned frame."""
        unit = self.unit(videos, mosaic)
        if unit is None:
            return
        n = min(len(v.pts_us) for v in videos.values())
        with self._lock:
            demand = self._demand.setdefault(unit, Counter())
            for wanted in requests:
                for i in set(wanted):
                    if isinstance(i, int) and 0 <= i < n:
                        demand[i] += 1
        self.state.count("lookahead_requests", len(requests))

    def _lookahead(self, unit: tuple, missing: list[int], budget: int) -> list[int]:
        with self._lock:
            demand = self._demand.get(unit)
            if not demand:
                return []
            ahead = sorted(i for i, c in demand.items() if c > 0 and i >= missing[0])
        have = set(missing)
        extra = []
        for i in ahead:
            if len(have) + len(extra) >= budget:
                break
            if i not in have and self.scope.frames.get((unit, i)) is None:
                extra.append(i)
        return extra

    def _consume(self, unit: tuple, want) -> None:
        with self._lock:
            demand = self._demand.get(unit)
            if not demand:
                return
            done = []
            for i in want:
                c = demand.get(i, 0)
                if c > 0:
                    demand[i] = c - 1
                    if c == 1:
                        done.append(i)
                        del demand[i]
        for i in done:  # no announced use left: first in line for eviction
            self.scope.frames.demote((unit, i))

    def serve(self, unit: tuple, want: set[int], unit_bytes: int,
              produce: Callable[[list[int]], dict[int, Any]]) -> dict[int, Any]:
        """Cached-or-produced outputs for ``want`` (shared objects: callers copy)."""
        out: dict[int, Any] = {}
        missing = []
        for i in sorted(want):
            hit = self.scope.frames.get((unit, i))
            if hit is None:
                missing.append(i)
            else:
                out[i] = hit
        self.state.count("frame_cache_hits", len(out))
        if missing:
            budget = max(len(missing), self.state.cache_bytes // 2 // max(unit_bytes, 1))
            extra = self._lookahead(unit, missing, budget)
            todo = sorted(set(missing).union(extra))
            if extra:
                self.state.count("lookahead_prefetched", len(extra))
            produced = produce(todo)
            for i in todo:
                self.scope.frames.put((unit, i), produced[i])
            for i in missing:
                out[i] = produced[i]
        self._consume(unit, want)
        return out


def make_frames_for(nv, original: Callable, state) -> Callable:
    """Replacement for NVIDIA ``frames_for(videos, wanted, mosaic='YAM')``."""
    store = state.frame_store(nv)  # routes to the current episode scope's FrameStore
    errors = nv.media.InputError

    def frames_for(videos, wanted, mosaic="YAM"):
        state.count("frames_for_calls")
        want = set(wanted)
        for video in videos.values():
            if want and (min(want) < 0 or max(want) >= len(video.pts_us)):
                raise errors(MSG_OUTSIDE)
        if not want:
            return original(videos, wanted, mosaic=mosaic)
        if not all(store.decoder(v).trusted for v in videos.values()):
            state.count("untrusted_delegations")
            return original(videos, wanted, mosaic=mosaic)
        canonical = len(videos) == 3  # media.py:98
        if len(videos) == 1:
            if "main" not in videos:
                return original(videos, wanted, mosaic=mosaic)  # NVIDIA's KeyError
            video = videos["main"]
            unit = store.unit(videos, mosaic)
            frames = store.serve(
                unit, want, store.unit_bytes(videos, mosaic),
                lambda todo: store.decoder(video).decode(todo, canonical),
            )
            return {i: frames[i].copy() for i in sorted(want)}  # media.py:91-95 order
        if mosaic not in ("YAM", "Dex"):
            raise errors(MSG_LAYOUT)
        missing_role = next((r for r in ROLES if r not in videos), None)
        if missing_role is not None:
            return original(videos, wanted, mosaic=mosaic)  # NVIDIA's KeyError
        unit = store.unit(videos, mosaic)
        mosaics = store.serve(
            unit, want, store.unit_bytes(videos, mosaic),
            lambda todo: _compose(nv, store, state, videos, todo, mosaic, canonical),
        )
        return {i: mosaics[i].copy() for i in wanted}  # media.py:102-105 order

    frames_for.__wrapped__ = original  # type: ignore[attr-defined]
    return frames_for


ROLES = ("main", "left_wrist", "right_wrist")


def _compose(nv, store, state, videos, todo: list[int], mosaic: str, canonical: bool) -> dict:
    composers = nv.vendor("composers")
    roles = list(videos.items())
    raw_list = state.map_io(lambda rv: store.decoder(rv[1]).decode(todo, canonical), roles)
    frames = {role: raw for (role, _), raw in zip(roles, raw_list, strict=True)}

    def compose(i):
        if mosaic == "YAM":  # media.py:102-103
            return composers.compose_top_dominant_mosaic(
                frames["main"][i], frames["left_wrist"][i], frames["right_wrist"][i]
            )
        return composers.compose_frame(  # media.py:104-105
            dict(
                ego=frames["main"][i],
                left_wrist=frames["left_wrist"][i],
                right_wrist=frames["right_wrist"][i],
            ),
            ("ego", "left_wrist", "right_wrist"),
        )

    composed = state.map(compose, todo)
    state.count("mosaics_composed", len(todo))
    return dict(zip(todo, composed, strict=True))


# ---------------------------------------------------------------------------------------------
# lookahead wrappers: tell the store which frames NVIDIA is about to request

_CAPTION_THREE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "flora_accel_caption_three", default=None
)


def make_predict(nv, original: Callable, state) -> Callable:
    """``Predictor.predict`` (pipeline.py:21-38), unchanged, after announcing the frame plans of
    every window in the request (computed with NVIDIA's own ``window_from_payload`` and
    ``temporal_frame_plan``, which are pure)."""
    store = state.frame_store(nv)

    def predict(self, request):
        try:
            a = nv.production
            plans = []
            for payload in request["windows"]:
                window = a.window_from_payload(payload, self.source)
                plan = a.temporal_frame_plan(window, list(self.videos["main"].pts_us))
                plans.append(set(plan["source_frame_indices"]))
            mosaic = "Dex" if self.pool in ("dex", "tianji") else "YAM"  # pipeline.py:28
            store.hint(self.videos, mosaic, plans)
        except Exception:  # noqa: BLE001 - NVIDIA's predict raises its own errors below
            state.count("lookahead_errors")
        return original(self, request)

    predict.__wrapped__ = original  # type: ignore[attr-defined]
    return predict


def make_caption_three(original: Callable, state) -> Callable:
    sig = inspect.signature(original)

    def caption_three(*args, **kwargs):
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            profile = bound.arguments["profile"]
            group = "Dex" if profile in ("Dex", "Tianji") else "YAM"  # pipeline.py:116
            ctx = (bound.arguments["videos"], group)
        except Exception:  # noqa: BLE001 - NVIDIA raises its own TypeError
            ctx = None
        token = _CAPTION_THREE.set(ctx)
        try:
            return original(*args, **kwargs)
        finally:
            _CAPTION_THREE.reset(token)

    caption_three.__wrapped__ = original  # type: ignore[attr-defined]
    return caption_three


def make_plans_for_episode(nv, original: Callable, state) -> Callable:
    """``plans.plans_for_episode``, unchanged; inside ``caption_three`` its plans are announced
    (caption_three builds every plan before it requests any frame, pipeline.py:119-125)."""
    store = state.frame_store(nv)

    def plans_for_episode(ep, clocks):
        plans = original(ep, clocks)
        ctx = _CAPTION_THREE.get()
        if ctx is not None:
            videos, group = ctx
            try:
                store.hint(videos, group, [
                    {j for block in plan["blocks"] for j in block["indices"]} for plan in plans
                ])
            except Exception:  # noqa: BLE001 - announcing is best effort
                state.count("lookahead_errors")
        return plans

    plans_for_episode.__wrapped__ = original  # type: ignore[attr-defined]
    return plans_for_episode


DECODE_ONCE_TARGETS = (
    # pipeline.py binds ``frames_for`` at import (``from .media import frames_for``)
    "batch_utils.media.frames_for",
    "batch_utils.pipeline.frames_for",
    "batch_utils.pipeline.Predictor.predict",
    "batch_utils.pipeline.caption_three",
    "vendor.plans.plans_for_episode",
)


def frames_for_targets() -> tuple[str, ...]:
    return DECODE_ONCE_TARGETS


def decode_once_factory(nv, current: Mapping[str, Any], state) -> dict[str, Any]:
    t = DECODE_ONCE_TARGETS
    return {
        t[0]: make_frames_for(nv, current[t[0]], state),
        t[1]: make_frames_for(nv, current[t[1]], state),
        t[2]: make_predict(nv, current[t[2]], state),
        t[3]: make_caption_three(current[t[3]], state),
        t[4]: make_plans_for_episode(nv, current[t[4]], state),
    }
