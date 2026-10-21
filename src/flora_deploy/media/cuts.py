"""Stream-copy per-episode cuts (PLAN.md D7): the oracle's per-episode input files.

The cut is a PyAV remux (no ffmpeg CLI, no re-encode) of exactly the episode's packets:

* same codec parameters (extradata, colour tags) via ``add_stream_from_template``;
* same ``time_base`` as the source track, so PTS stay exact multiples (a different timescale
  quantises them and breaks NVIDIA's 1 µs CFR test);
* packets in decode order with ``pts``/``dts`` shifted so the first presented frame has
  ``pts == 0``, and ``duration`` copied - the muxer turns those into the cut's ``stts``;
* MP4 output only (``.mp4``), video stream only.

The range must start on a keyframe with no leading pictures and be contiguous in decode order;
otherwise ``NotKeyframeAligned`` is raised (such episodes are oracle-ineligible, PLAN.md §9).
``index.index_stream`` predicts what NVIDIA's ``probe`` reports on this file.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import av

from flora_deploy.media.index import (
    MediaError,
    NotKeyframeAligned,
    SeekOvershoot,
    StreamScan,
    demux_range,
    index_stream,
    scan_stream,
    select_episode,
)
from flora_deploy.media.types import SourceRef


def stream_copy_cut(source: SourceRef, dest: Path, *, scan: StreamScan | None = None) -> Path:
    """Write the episode range of ``source`` to ``dest`` (MP4) without re-encoding."""
    dest = Path(dest)
    if dest.suffix.lower() != ".mp4":
        raise MediaError(f"cut destination must be .mp4, got {dest.name}")
    scan = scan if scan is not None else scan_stream(source.uri)
    idx = index_stream(source, scan=scan)
    if not idx.starts_on_keyframe:
        raise NotKeyframeAligned(
            f"{source.uri} [{source.start_s}, {source.end_s}): the range does not start on a "
            "keyframe (or has leading/non-contiguous packets); a stream-copy cut would not "
            "decode identically. The episode is oracle-ineligible."
        )
    sl = select_episode(scan, source.start_s, source.end_s)
    pk = scan.packets[sl.first : sl.last + 1]
    first_dts, last_dts = pk[0].dts, pk[-1].dts
    shift = min(p.pts for p in pk)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for seek in (sl.first > 0, False):
        written = 0
        with av.open(str(source.uri)) as ic:
            ist = ic.streams.video[0]
            packets = demux_range(ic, ist, first_dts, last_dts, seek=seek)
            try:
                head = next(packets)
            except SeekOvershoot:
                continue
            with av.open(str(dest), "w") as oc:
                ost = oc.add_stream_from_template(template=ist, opaque=True)
                ost.time_base = ist.time_base
                for pkt in itertools.chain([head], packets):
                    pkt.pts -= shift
                    pkt.dts -= shift
                    pkt.stream = ost
                    oc.mux(pkt)
                    written += 1
        break
    if written != len(pk):
        raise MediaError(f"{source.uri}: wrote {written} packets, expected {len(pk)}")
    return dest
