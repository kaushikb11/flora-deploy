"""Decode an episode's frames straight from the source range (PLAN.md D7).

Produces, for episode-relative frame indices, the same images NVIDIA's ``decode_selected``
(``media.py:69-95``) produces on the stream-copy cut of that range:

* oracle-eligible ranges: the decoder sees exactly the cut's packets (from the episode's first
  packet in decode order to its last), opened with ``err_detect=explode`` and ``thread_count=1``
  (``media.py:76-78``), then flushed - so frame ``i`` is the ``i``-th decoded output, as in the
  cut, and every output must be the indexed frame (no filtering);
* other ranges (``StreamIndex.ineligible``): decoding starts at the previous random-access point
  (or the file start), and the output is filtered to the indexed frames by absolute pts;
* every frame's **absolute** source pts is checked against ``StreamIndex.source_pts`` and its
  relative PTS against ``pts_us`` (``media.py:83-84``), so a decoder that drops or shifts frames
  raises instead of returning neighbouring frames;
* decoding stops once the largest wanted index is reached (``media.py:91-92``);
* colour conversion branch (``media.py:85-90``): when ``canonical`` (NVIDIA passes it for 3-view
  inputs, ``media.py:98``) and the frame is ``yuv420p`` with ``width % 16 == 0``,
  ``height % 2 == 0``, ``colorspace in (2, 5, 6)`` and ``color_range in (0, 1)``, the frame goes
  through the injected ``canonical_rgb``; otherwise ``frame.to_image().convert("RGB")``
  (swscale).

``canonical_rgb`` is injected, never imported: NVIDIA's implementation lives in its noncommercial
vendor tree (``vendor/support/canonical_rgb.py``), so linked mode (WS-B's ``nv`` linker) passes
NVIDIA's function, and this module refuses to guess one. Decoding a canonical-eligible frame with
``canonical=True`` and no converter raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from fractions import Fraction

import av
from PIL import Image

from flora_deploy.media.index import (
    US,
    MediaError,
    ScanStore,
    StreamScan,
    decode_span,
    get_scan,
    plan_episode,
    scan_stream,
)
from flora_deploy.media.types import SourceRef, StreamIndex

ColorConvert = Callable[[av.VideoFrame], Image.Image]


def canonical_eligible(frame: av.VideoFrame) -> bool:
    """Frame-level conditions of NVIDIA's canonical branch (``media.py:85``)."""
    return (
        frame.format.name == "yuv420p"
        and frame.width % 16 == 0
        and frame.height % 2 == 0
        and int(frame.colorspace) in (2, 5, 6)
        and int(frame.color_range) in (0, 1)
    )


def swscale_rgb(frame: av.VideoFrame) -> Image.Image:
    """NVIDIA's non-canonical branch (``media.py:90``)."""
    return frame.to_image().convert("RGB")


def iter_frames(
    source: SourceRef,
    *,
    scan: StreamScan | None = None,
    presented: Sequence[int] | None = None,
) -> Iterator[av.VideoFrame]:
    """Decoded frames of the episode in output order, each checked against its absolute pts.

    ``presented``: the absolute pts the index promised (``StreamIndex.source_pts``); computed
    from the scan when omitted (packet prediction, which needs a random-access point at or
    before the range).
    """
    scan = scan if scan is not None else scan_stream(source.uri)
    plan = plan_episode(scan, source.start_s, source.end_s)
    expected = list(presented) if presented else [int(p) for p in scan.pts[plan.frames]]
    if plan.eligible:
        start, strict = plan.first, True
    else:
        start, strict = (plan.preroll if plan.preroll is not None else 0), False
    wanted = set(expected)
    k = 0
    for frame in decode_span(scan, start, plan.last):
        if not strict and frame.pts not in wanted:
            continue
        if k >= len(expected) or frame.pts != expected[k]:
            raise MediaError(
                f"{source.uri}: decoded frame pts {frame.pts} where the index has "
                f"{expected[k] if k < len(expected) else 'no more frames'}"
            )
        k += 1
        yield frame
    if k != len(expected):
        raise MediaError(f"{source.uri}: decoder output {k} of {len(expected)} indexed frames")


def decode_selected(
    stream: StreamIndex,
    wanted: Iterable[int],
    *,
    canonical: bool = False,
    canonical_rgb: ColorConvert | None = None,
    scan: StreamScan | None = None,
    scans: ScanStore | None = None,
) -> dict[int, Image.Image]:
    """Images for episode-relative frame indices; == NVIDIA ``decode_selected`` on the cut.

    ``scan``/``scans``: reuse the index pass's scan (or scan cache) instead of re-demuxing.
    """
    want = set(wanted)
    result: dict[int, Image.Image] = {}
    if not want:
        return result
    if min(want) < 0 or max(want) >= len(stream.pts_us):
        raise MediaError("A requested source frame is outside the video.")
    if scan is None:
        scan = get_scan(scans, stream.source.uri) if scans is not None else None
    tb = stream.time_base
    absolute = stream.source_pts
    zero = absolute[0] if absolute else None
    frames = iter_frames(stream.source, scan=scan, presented=absolute or None)
    for i, frame in enumerate(frames):
        zero = frame.pts if zero is None else zero
        if i in want:
            rel = Fraction(frame.pts - zero) * (tb or frame.time_base) * US
            if (absolute and frame.pts != absolute[i]) or rel != stream.pts_us[i]:
                raise MediaError("Source frame timestamps changed while decoding.")
            if canonical and canonical_eligible(frame):
                if canonical_rgb is None:
                    raise MediaError(
                        "canonical-RGB frame but no converter injected (linked mode passes "
                        "NVIDIA's canonical_rgb)"
                    )
                result[i] = canonical_rgb(frame)
            else:
                result[i] = swscale_rgb(frame)
        if i >= max(want):
            break
    if set(result) != want:
        raise MediaError("A requested source frame could not be decoded.")
    return result
