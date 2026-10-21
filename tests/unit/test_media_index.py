"""Index / cut / range-decode rules that need no NVIDIA code (adversarial sources, RAP parsing,
selection cost, scan cache, seeking, atomic cuts, exact size near the cap)."""

from __future__ import annotations

import time
from dataclasses import replace
from fractions import Fraction

import av
import numpy as np
import pytest

from flora_deploy.media import synth
from flora_deploy.media.cuts import stream_copy_cut
from flora_deploy.media.index import (
    FLAG_DISCARD,
    FLAG_KEY,
    FLAG_RAP,
    MediaError,
    NotKeyframeAligned,
    ScanCache,
    StreamScan,
    _av1_is_rap,
    _h264_is_rap,
    _hevc_is_rap,
    estimated_cut_bytes,
    index_episode,
    index_stream,
    scan_stream,
    select_episode,
    stream_checks,
)
from flora_deploy.media.rangedecode import decode_selected, swscale_rgb
from flora_deploy.media.synth import Spec
from flora_deploy.media.types import SourceRef


def _px(img):
    return np.asarray(img.convert("RGB"))


def _truth(path, pts):
    with av.open(str(path)) as c:
        return {f.pts: _px(swscale_rgb(f)) for f in c.decode(video=0) if f.pts in pts}


# ---------------------------------------------------------------------------------------------
# Random-access-point parsers


def _avcc(*nals: bytes) -> bytes:
    return b"".join(len(n).to_bytes(4, "big") + n for n in nals)


def test_h264_rap():
    sei, idr, non_idr = b"\x06\x05\x00", b"\x65\x88", b"\x41\x9a"
    assert _h264_is_rap(_avcc(sei, idr), 4)
    assert not _h264_is_rap(_avcc(sei, non_idr), 4)  # recovery-point I/P slice
    assert _h264_is_rap(b"\x00\x00\x00\x01\x67\x42\x00\x00\x01\x65\x88", None)  # Annex B
    assert not _h264_is_rap(b"", 4)


def test_hevc_rap():
    def nal(t):
        return bytes([t << 1, 1, 0])

    for t in (16, 19, 20, 21):  # BLA, IDR_W_RADL, IDR_N_LP, CRA
        assert _hevc_is_rap(_avcc(nal(39), nal(t)), 4)  # prefix SEI first
    for t in (0, 1, 8, 9):  # TRAIL, RASL
        assert not _hevc_is_rap(_avcc(nal(t)), 4)


def test_av1_rap():
    td = b"\x12\x00"  # temporal delimiter, size 0

    def frame(b0):
        return bytes([(6 << 3) | 2, 1, b0])  # OBU_FRAME with size field

    assert _av1_is_rap(td + frame(0b0001_0000), False)  # KEY_FRAME, show_frame
    assert not _av1_is_rap(td + frame(0b0000_0000), False)  # hidden key frame
    assert not _av1_is_rap(td + frame(0b1000_0000), False)  # show_existing_frame
    assert not _av1_is_rap(td + frame(0b0101_0000), False)  # INTER_FRAME
    assert not _av1_is_rap(td + frame(0b0101_0000), False)
    assert not _av1_is_rap(td + frame(0b0011_0000), False)  # INTRA_ONLY_FRAME
    assert _av1_is_rap(td + frame(0), True)  # reduced still-picture header


def test_real_streams_flag_keyframes_as_raps(tmp_path):
    for codec, size in (("libx264", (64, 48)), ("libsvtav1", (64, 64))):
        p = synth.encode(tmp_path / f"{codec}.mp4", Spec(12, codec=codec, size=size, gop=4))
        s = scan_stream(p)
        keys = (s.flags & FLAG_KEY) != 0
        assert keys.sum() >= 3
        assert np.array_equal(keys, (s.flags & FLAG_RAP) != 0), codec


# ---------------------------------------------------------------------------------------------
# Adversarial sources (fail closed, correct counts, source frames)


def test_intra_refresh_sync_samples_are_not_random_access(tmp_path):
    p = synth.encode(
        tmp_path / "ir.mp4",
        Spec(90, seed=2, bframes=0, x264_params="intra-refresh=1:keyint=30:scenecut=0"),
    )
    s = scan_stream(p)
    assert list(np.flatnonzero(s.flags & FLAG_KEY)) == [0, 30, 60]
    assert list(np.flatnonzero(s.flags & FLAG_RAP)) == [0]
    ref = SourceRef(str(p), 1.0, 2.0)
    idx = index_stream(ref, scan=s)
    assert idx.ineligible == ("not_random_access",) and not idx.starts_on_keyframe
    with pytest.raises(NotKeyframeAligned) as e:
        stream_copy_cut(ref, tmp_path / "c.mp4")
    assert e.value.reasons == ("not_random_access",)
    out = decode_selected(idx, [0, 29], scan=s)  # pre-rolls from the IDR at 0
    truth = _truth(p, set(idx.source_pts))
    assert np.array_equal(_px(out[0]), truth[idx.source_pts[0]])
    assert idx.source_timestamps_s[0] == 1.0


def test_edit_list_discards_are_not_frames(tmp_path):
    base = synth.encode(tmp_path / "b.mp4", Spec(90, seed=3, gop=30))
    trim = synth.remux_trimmed(base, tmp_path / "t.mp4", 1, 45 * 512)
    s = scan_stream(trim)
    assert int(((s.flags & FLAG_DISCARD) != 0).sum()) == 15
    idx = index_stream(SourceRef(str(trim)), scan=s)
    assert idx.ineligible == ("edit_list",)
    assert len(idx.pts_us) == 45 and idx.source_pts[0] == 0
    with pytest.raises(NotKeyframeAligned):
        stream_copy_cut(idx.source, tmp_path / "c.mp4")
    out = decode_selected(idx, [0, 44], scan=s)
    truth = _truth(trim, {0, idx.source_pts[44]})
    assert np.array_equal(_px(out[0]), truth[0])
    assert np.array_equal(_px(out[44]), truth[idx.source_pts[44]])


def test_whole_file_starting_mid_gop_counts_decoded_frames(tmp_path):
    g = synth.encode(tmp_path / "g.mp4", Spec(90, seed=4, gop=30, bframes=0))
    p = synth.remux_drop_head(g, tmp_path / "n.mp4", 10)
    idx = index_stream(SourceRef(str(p)))
    assert idx.ineligible == ("not_keyframe",)
    with av.open(str(p)) as c:
        decoded = [f.pts for f in c.decode(video=0)]
    assert len(idx.pts_us) == len(decoded) == 60
    assert list(idx.source_pts) == decoded
    assert set(decode_selected(idx, [0, 59])) == {0, 59}


def test_non_contiguous_range_uses_only_selected_frames(tmp_path):
    p = synth.encode(tmp_path / "b.mp4", Spec(90, seed=1, gop=30))
    s = scan_stream(p)
    ref = SourceRef(str(p), 1.0, 50 / 30)
    sl = select_episode(s, ref.start_s, ref.end_s)
    assert not sl.contiguous
    idx = index_stream(ref, scan=s)
    assert idx.ineligible == ("non_contiguous",)
    assert len(idx.pts_us) == 20
    assert idx.source_timestamps_s == tuple((30 + i) / 30 for i in range(20))
    assert idx.rate == 30 and idx.cfr
    assert set(decode_selected(idx, [0, 19], scan=s)) == {0, 19}


def test_absolute_pts_guard(tmp_path):
    p = synth.encode(tmp_path / "a.mp4", Spec(20, seed=5))
    idx = index_stream(SourceRef(str(p)))
    shifted = replace(idx, source_pts=tuple(t + 512 for t in idx.source_pts))
    with pytest.raises(MediaError):
        decode_selected(shifted, [0])


def test_two_video_streams_are_refused(tmp_path):
    path = tmp_path / "two.mp4"
    with av.open(str(path), "w") as out:
        streams = [out.add_stream("libx264", rate=30) for _ in range(2)]
        for s in streams:
            s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(3):
            for s in streams:
                f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 9, np.uint8), "rgb24")
                for pkt in s.encode(f.reformat(format="yuv420p")):
                    out.mux(pkt)
        for s in streams:
            for pkt in s.encode():
                out.mux(pkt)
    with pytest.raises(MediaError, match="2 video streams"):
        scan_stream(path)


# ---------------------------------------------------------------------------------------------
# Seeking (L1), atomic cuts (L7), exact size (L6)


def test_seek_by_pts_lands_on_the_keyframe(tmp_path):
    p = synth.encode(tmp_path / "b.mp4", Spec(120, seed=6, gop=30))  # B-frames: pts > dts
    s = scan_stream(p)
    for k in np.flatnonzero(s.flags & FLAG_KEY)[1:]:
        with av.open(str(p)) as c:
            st = c.streams.video[0]
            c.seek(int(s.pts[k]), stream=st, backward=True, any_frame=False)
            first = next(pkt for pkt in c.demux(st) if pkt.size)
        assert first.dts == s.dts[k], k  # (seeking by dts lands one GOP early)


def test_cut_leaves_no_partial_file(tmp_path, monkeypatch):
    p = synth.encode(tmp_path / "a.mp4", Spec(20, seed=7))
    from flora_deploy.media import cuts

    real = cuts.demux_range

    def short(*a, **kw):
        it = real(*a, **kw)
        yield next(it)
        raise MediaError("boom")

    monkeypatch.setattr(cuts, "demux_range", short)
    with pytest.raises(MediaError):
        stream_copy_cut(SourceRef(str(p)), tmp_path / "out" / "c.mp4")
    assert list((tmp_path / "out").iterdir()) == []


def test_size_near_cap_uses_the_real_cut(tmp_path):
    p = synth.encode(tmp_path / "a.mp4", Spec(30, seed=8))
    idx = index_stream(SourceRef(str(p)))
    real = stream_copy_cut(idx.source, tmp_path / "c.mp4").stat().st_size
    assert idx.byte_size < real < estimated_cut_bytes(idx)
    assert stream_checks(idx, max_bytes=real) == []  # estimate says too big, real cut fits
    assert stream_checks(idx, max_bytes=real - 1) == ["file_gt_256MiB"]
    assert stream_checks(idx, max_bytes=real, exact_size=False) == ["file_gt_256MiB"]
    assert stream_checks(idx, max_bytes=idx.byte_size) == ["file_gt_256MiB"]  # no I/O needed


# ---------------------------------------------------------------------------------------------
# Scan cost and cache (M3, M4)


def _synthetic_scan(n: int) -> StreamScan:
    pts = np.arange(n, dtype=np.int64) * 512
    flags = np.where(np.arange(n) % 2 == 0, FLAG_KEY | FLAG_RAP, 0).astype(np.uint8)
    return StreamScan(
        path="synthetic.mp4",
        time_base=Fraction(1, 15360),
        codec="av1",
        pix_fmt="yuv420p",
        width=640,
        height=480,
        video_streams=1,
        colorspace=None,
        color_range=None,
        pts=pts,
        dts=pts.copy(),
        duration=np.full(n, 512, np.int64),
        size=np.full(n, 1000, np.int64),
        flags=flags,
        missing_pts=False,
    )


def _old_select(scan, start_s, end_s):
    """The previous O(N) per-episode selection (Fraction comparisons over the whole file)."""
    tb = scan.time_base
    durs = sorted(int(d) for d in scan.duration if d > 0)
    half = Fraction(durs[len(durs) // 2], 2) * tb
    lo, hi = Fraction(start_s) - half, Fraction(end_s) - half
    return [i for i, p in enumerate(scan.pts.tolist()) if p * tb >= lo and p * tb < hi]


def test_benchmark_100k_packets_500_episodes():
    n, episodes, per = 100_000, 500, 200
    scan = _synthetic_scan(n)
    rate = Fraction(30)
    refs = [
        SourceRef("synthetic.mp4", float(k * per / rate), float((k + 1) * per / rate))
        for k in range(episodes)
    ]
    t0 = time.perf_counter()
    _old_select(scan, refs[250].start_s, refs[250].end_s)
    old_per_episode = time.perf_counter() - t0
    t0 = time.perf_counter()
    for ref in refs:
        idx = index_stream(ref, scan=scan)
        assert len(idx.pts_us) == per and idx.starts_on_keyframe and idx.rate == 30
    new_total = time.perf_counter() - t0
    print(
        f"\n100k packets / 500 episodes: new {new_total:.3f} s total "
        f"({new_total / episodes * 1e3:.2f} ms/episode); old selection alone "
        f"{old_per_episode * 1e3:.0f} ms/episode (~{old_per_episode * episodes:.0f} s total)"
    )
    assert new_total / episodes < old_per_episode / 10
    assert scan.nbytes / n < 64  # bytes per packet retained (was ~335 with Packet objects)


def test_scan_cache_is_bounded_and_reused(tmp_path):
    paths = [synth.encode(tmp_path / f"{k}.mp4", Spec(10, seed=k)) for k in range(3)]
    one = scan_stream(paths[0]).nbytes
    cache = ScanCache(max_bytes=int(one * 2.5))
    for p in paths:
        cache.get(str(p))
    assert len(cache) == 2 and str(paths[0]) not in cache and cache.scans_done == 3
    ix = index_episode("e", "d", {"main": SourceRef(str(paths[2]))}, scans=cache)
    decode_selected(ix.streams["main"], [0, 9], scans=cache)
    assert cache.scans_done == 3  # index and decode reused the cached scan


def test_decode_mode_turns_decoder_errors_into_a_reason(tmp_path):
    p = synth.encode(tmp_path / "a.mp4", Spec(30, seed=9, bframes=0))
    with av.open(str(p)) as c:
        pkts = [(k.pos, k.size) for k in c.demux(video=0) if k.size]
    pos, size = pkts[12]
    data = bytearray(p.read_bytes())
    data[pos + 4 : pos + size] = bytes((i * 37 + 11) % 256 for i in range(size - 4))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(bytes(data))
    ref = SourceRef(str(bad))
    assert index_stream(ref).starts_on_keyframe  # packets alone look fine (review L4)
    idx = index_stream(ref, decode=True)
    assert not idx.starts_on_keyframe
    assert set(idx.ineligible) & {"decode_error", "decoder_mismatch"}
