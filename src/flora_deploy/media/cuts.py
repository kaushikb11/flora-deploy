"""Stream-copy per-episode cuts (PLAN.md D7): the oracle's per-episode input files.

The cut is a PyAV remux (no ffmpeg CLI, no re-encode) of exactly the episode's packets:

* same codec parameters (extradata, colour tags) via ``add_stream_from_template``;
* same ``time_base`` as the source track, so PTS stay exact multiples (a different timescale
  quantises them and breaks NVIDIA's 1 µs CFR test). movenc still doubles timescales below 10000
  (1/30 -> 1/15360), which keeps every rational unchanged;
* packets in decode order with ``pts``/``dts`` shifted so the first presented frame has
  ``pts == 0``, and ``duration`` copied - the muxer turns those into the cut's ``stts``;
* MP4 output only (``.mp4``), video stream only;
* written to a temporary name next to ``dest`` and renamed when complete (no partial files).

Only oracle-eligible ranges are cut (``index.plan_episode``: true random-access start, no leading
pictures, contiguous in decode order, no discarded packets); otherwise ``NotKeyframeAligned`` is
raised with the reasons. ``index.index_stream`` predicts what NVIDIA's ``probe`` reports on the
file.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import av

from flora_deploy.media.index import (
    MediaError,
    NotKeyframeAligned,
    SeekOvershoot,
    StreamScan,
    demux_range,
    plan_episode,
    scan_stream,
)
from flora_deploy.media.types import SourceRef


def stream_copy_cut(source: SourceRef, dest: Path, *, scan: StreamScan | None = None) -> Path:
    """Write the episode range of ``source`` to ``dest`` (MP4) without re-encoding."""
    dest = Path(dest)
    if dest.suffix.lower() != ".mp4":
        raise MediaError(f"cut destination must be .mp4, got {dest.name}")
    scan = scan if scan is not None else scan_stream(source.uri)
    plan = plan_episode(scan, source.start_s, source.end_s)
    if not plan.eligible:
        raise NotKeyframeAligned(
            f"{source.uri} [{source.start_s}, {source.end_s}): a stream-copy cut would not decode "
            f"like the source ({', '.join(plan.reasons)}). The episode is oracle-ineligible.",
            plan.reasons,
        )
    first, last = plan.first, plan.last
    shift = int(scan.pts[plan.selected].min())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.stem}.partial.mp4")
    written = 0
    try:
        for seek in (first > 0, False):
            written = 0
            with av.open(scan.path) as ic:
                ist = ic.streams[scan.stream_index]
                packets = demux_range(ic, ist, scan, first, last, seek=seek)
                try:
                    head = next(packets)
                except SeekOvershoot:
                    continue
                with av.open(str(tmp), "w", format="mp4") as oc:
                    ost = oc.add_stream_from_template(template=ist, opaque=True)
                    ost.time_base = ist.time_base
                    for pkt in itertools.chain([head], packets):
                        pkt.pts -= shift
                        pkt.dts -= shift
                        pkt.stream = ost
                        oc.mux(pkt)
                        written += 1
            break
        if written != last - first + 1:
            raise MediaError(f"{source.uri}: wrote {written} packets, expected {last - first + 1}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
