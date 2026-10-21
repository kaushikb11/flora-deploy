from dataclasses import replace
from fractions import Fraction

import pytest

from flora_deploy.media import synth
from flora_deploy.media.cuts import stream_copy_cut
from flora_deploy.media.index import (
    NotKeyframeAligned,
    av_reduce,
    envelope_checks,
    index_stream,
    stream_checks,
)
from flora_deploy.media.synth import Spec
from flora_deploy.media.types import SourceRef, StreamIndex


def _stream(n=30, rate=Fraction(30), **kw) -> StreamIndex:
    pts = tuple(Fraction(i * 1_000_000) / rate for i in range(n))
    base = dict(
        source=SourceRef("x.mp4"),
        rate=rate,
        pts_us=pts,
        duration_us=round(pts[-1] + Fraction(1_000_000) / rate) if pts else 0,
        size=(64, 48),
        cfr=True,
        codec="h264",
        pix_fmt="yuv420p",
        starts_on_keyframe=True,
        byte_size=1000,
    )
    base.update(kw)
    return StreamIndex(**base)


def test_ok_stream():
    assert stream_checks(_stream()) == []
    assert envelope_checks({"main": _stream()}, profile="human").ok


@pytest.mark.parametrize(
    "kw,reason",
    [
        (dict(rate=Fraction(3)), "fps_lt_4"),
        (dict(size=(4097, 16)), "dimension_out_of_range"),
        (dict(size=(0, 16)), "dimension_out_of_range"),
        (dict(byte_size=0), "file_empty"),
        (dict(byte_size=256 * 1024 * 1024), "file_gt_256MiB"),
        (dict(video_streams=2), "video_streams_ne_1"),
        (dict(missing_pts=True), "missing_pts"),
        (dict(n=1), "too_few_frames"),
        (dict(n=7201, rate=Fraction(100)), "frames_gt_7200"),
        (dict(n=3601, rate=Fraction(30)), "duration_gt_120s"),
    ],
)
def test_stream_reasons(kw, reason):
    n = kw.pop("n", 30)
    s = _stream(n=n, **{k: v for k, v in kw.items() if k == "rate"})
    s = replace(s, **{k: v for k, v in kw.items() if k != "rate"})
    assert reason in stream_checks(s)


def test_zero_rate():
    assert "fps_lt_4" in stream_checks(replace(_stream(), rate=Fraction(0)))


def test_boundaries_accept():
    assert stream_checks(_stream(n=7200, rate=Fraction(100))) == []
    assert stream_checks(_stream(n=3600, rate=Fraction(30))) == []  # exactly 120 s
    assert stream_checks(_stream(rate=Fraction(4))) == []


def test_non_increasing_pts():
    s = _stream()
    s = replace(s, pts_us=s.pts_us[:5] + (s.pts_us[4],) + s.pts_us[6:])
    assert "pts_not_increasing" in stream_checks(s)


def _three(**main_kw):
    s = _stream()
    return {"main": replace(s, **main_kw), "left_wrist": s, "right_wrist": s}


def test_three_view_rules():
    assert envelope_checks(_three(), profile="yam", yam32=True).ok
    v = envelope_checks(_three(duration_us=1))
    assert "views_not_synchronized" in v.reasons
    v = envelope_checks(_three(cfr=False))
    assert "three_view_not_cfr" in v.reasons
    two = {"main": _stream(), "left_wrist": _stream()}
    assert "camera_roles_invalid" in envelope_checks(two).reasons
    assert "camera_roles_invalid" in envelope_checks({"left_wrist": _stream()}).reasons


def test_profile_rules():
    assert "human_profile_multi_view" in envelope_checks(_three(), profile="human").reasons
    assert "profile_invalid" in envelope_checks({"main": _stream()}, profile="cat").reasons
    v = envelope_checks({"main": _stream()}, profile="yam", yam32=True)
    assert "yam32_requires_yam_three_view" in v.reasons
    v = envelope_checks({"main": _stream()}, profile="human", instruction="x" * 2001)
    assert "instruction_too_long" in v.reasons
    assert envelope_checks({"main": _stream()}, profile="human", instruction=" x" * 1000 + " ").ok
    vfr = {"main": _stream(cfr=False)}
    assert envelope_checks(vfr, profile="human").ok  # human accepts VFR
    assert "single_view_robot_not_cfr" in envelope_checks(vfr, profile="dex").reasons


def test_av_reduce():
    assert av_reduce(15360 * 30, 512 * 30) == 30
    assert av_reduce(-6, 4) == Fraction(-3, 2)
    big = av_reduce(10**12 + 1, 10**10)
    assert big.numerator <= 2**31 - 1 and big.denominator <= 2**31 - 1
    assert abs(big - Fraction(10**12 + 1, 10**10)) < Fraction(1, 10**6)
    # 355/113 = [3; 7, 16]: within 100 the convergent 22/7 beats the semiconvergent 91/29
    assert av_reduce(355, 113, 1000) == Fraction(355, 113)
    assert av_reduce(355, 113, 100) == Fraction(22, 7)
    # 3/2 approximations of 1.6 within max 4: convergents 1, 2, 3/2, 8/5 -> 3/2
    assert av_reduce(8, 5, 4) == Fraction(3, 2)


def test_concatenated_episode_ranges(tmp_path):
    parts = [synth.encode(tmp_path / f"e{i}.mp4", Spec(n, seed=i)) for i, n in enumerate((9, 13))]
    cat = synth.concat_like_lerobot(parts, tmp_path / "cat.mp4")
    d0 = synth.video_duration_s(parts[0])
    d1 = synth.video_duration_s(parts[1])
    a = index_stream(SourceRef(str(cat), 0.0, d0))
    b = index_stream(SourceRef(str(cat), d0, d0 + d1))
    assert len(a.pts_us) == 9 and len(b.pts_us) == 13
    assert a.starts_on_keyframe and b.starts_on_keyframe
    assert b.pts_us[0] == 0 and b.source_timestamps_s[0] == pytest.approx(d0)


def test_cut_refuses_non_keyframe_start(tmp_path):
    p = synth.encode(tmp_path / "g.mp4", Spec(30, gop=1000, bframes=0))
    ref = SourceRef(str(p), 10 / 30, 1.0)
    assert not index_stream(ref).starts_on_keyframe
    with pytest.raises(NotKeyframeAligned):
        stream_copy_cut(ref, tmp_path / "c.mp4")
