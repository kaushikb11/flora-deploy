"""Decode an episode's frames straight from the source range (PLAN.md D7).

Produces, for episode-relative frame indices, the same images NVIDIA's ``decode_selected``
(``media.py:69-95``) produces on the stream-copy cut of that range:

* the decoder sees exactly the cut's packets (from the episode's first packet in decode order to
  its last), opened with ``err_detect=explode`` and ``thread_count=1`` (``media.py:76-78``), then
  flushed - so frame ``i`` is the ``i``-th decoded output, as in the cut;
* each wanted frame's relative PTS is checked against the index (``media.py:83-84``);
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

from collections.abc import Callable, Iterable, Iterator
from fractions import Fraction

import av
from PIL import Image

from flora_deploy.media.index import (
    NV_OPEN_OPTIONS,
    US,
    MediaError,
    SeekOvershoot,
    StreamScan,
    demux_range,
    scan_stream,
    select_episode,
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


def iter_frames(source: SourceRef, *, scan: StreamScan | None = None) -> Iterator[av.VideoFrame]:
    """Decoded frames of the episode, in output order, as the cut's decoder would emit them."""
    scan = scan if scan is not None else scan_stream(source.uri)
    sl = select_episode(scan, source.start_s, source.end_s)
    if not sl.contiguous:
        raise MediaError(f"{source.uri}: episode packets are not contiguous in decode order")
    first_dts = scan.packets[sl.first].dts
    last_dts = scan.packets[sl.last].dts
    for seek in (sl.first > 0, False):
        with av.open(str(source.uri), options=NV_OPEN_OPTIONS) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            packets = demux_range(container, stream, first_dts, last_dts, seek=seek)
            try:
                first = next(packets)
            except SeekOvershoot:
                continue
            yield from stream.decode(first)
            for pkt in packets:
                yield from stream.decode(pkt)
            yield from stream.decode(None)  # drain, like the cut's EOF
            return
    raise MediaError(f"{source.uri}: could not position the decoder on the episode")


def decode_selected(
    stream: StreamIndex,
    wanted: Iterable[int],
    *,
    canonical: bool = False,
    canonical_rgb: ColorConvert | None = None,
    scan: StreamScan | None = None,
) -> dict[int, Image.Image]:
    """Images for episode-relative frame indices; == NVIDIA ``decode_selected`` on the cut."""
    want = set(wanted)
    result: dict[int, Image.Image] = {}
    if not want:
        return result
    if min(want) < 0 or max(want) >= len(stream.pts_us):
        raise MediaError("A requested source frame is outside the video.")
    tb = stream.time_base
    zero = None
    for i, frame in enumerate(iter_frames(stream.source, scan=scan)):
        zero = frame.pts if zero is None else zero
        if i in want:
            rel = Fraction(frame.pts - zero) * (tb or frame.time_base) * US
            if rel != stream.pts_us[i]:
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
