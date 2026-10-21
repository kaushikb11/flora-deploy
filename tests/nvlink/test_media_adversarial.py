"""Adversarial sources vs NVIDIA (review of WS-C): edit lists, intra refresh, mid-GOP file
starts, non-contiguous ranges, .mov colour boxes, low timescales, HEVC, many-episode files.

For every case the index either is oracle-eligible and equals NVIDIA ``probe`` on the cut (and
range decode equals NVIDIA ``decode_selected`` on the cut, bit-exactly), or it fails closed with
the expected reasons - and production range decode still returns the source's real frames.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from flora_deploy.media.cuts import stream_copy_cut
from flora_deploy.media.index import (
    NotKeyframeAligned,
    ScanCache,
    cut_timescale,
    index_stream,
)
from flora_deploy.media.rangedecode import decode_selected, swscale_rgb

try:
    from .media_fixtures import build_adversarial, build_many_episodes, load_nv_media
except ImportError:  # tests/nvlink is not a package
    from media_fixtures import build_adversarial, build_many_episodes, load_nv_media

pytestmark = pytest.mark.nvlink

FIELDS = ("rate", "pts_us", "duration_us", "size", "cfr")


@pytest.fixture
def nvm(nv_code, monkeypatch):
    m = load_nv_media(nv_code("tempseg_code"))
    monkeypatch.setitem(__import__("sys").modules, "canonical_rgb", m.canonical_rgb)
    return m


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("adv")
    return root, build_adversarial(root) + build_many_episodes(root / "many")


def _wanted(n):
    return sorted({0, 1, n // 3, n // 2, n - 2, n - 1})


def _px(img):
    return np.asarray(img.convert("RGB"))


def _source_truth(path: str, pts: set[int]) -> dict[int, np.ndarray]:
    """Frames of the whole source file decoded from its start, by absolute pts."""
    with av.open(path) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 1
        return {f.pts: _px(swscale_rgb(f)) for f in c.decode(s) if f.pts in pts}


def test_eligibility_reasons_and_frame_counts(cases):
    _, eps = cases
    for c in eps:
        idx = index_stream(c.ref)
        assert set(idx.ineligible) == set(c.reasons), (c.name, idx.ineligible)
        assert idx.starts_on_keyframe == (not c.reasons), c.name
        if c.frames is not None:
            assert len(idx.pts_us) == c.frames, c.name
        assert len(idx.source_pts) == len(idx.pts_us)


@pytest.mark.parametrize("canonical", [False, True])
def test_eligible_cases_equal_nvidia_on_cut(nvm, cases, canonical):
    root, eps = cases
    cache = ScanCache()
    checked = 0
    for c in eps:
        if c.reasons:
            continue
        scan = cache.get(c.ref.uri)
        cut = stream_copy_cut(c.ref, root / "cuts" / f"{c.name}.mp4", scan=scan)
        v = nvm.media.probe(cut)
        for decode in (False, True):
            idx = index_stream(c.ref, scan=scan, decode=decode)
            assert idx.starts_on_keyframe, (c.name, idx.ineligible)
            for f in FIELDS:
                assert getattr(idx, f) == getattr(v, f), (c.name, f, decode)
        with av.open(str(cut)) as cc:
            assert cc.streams.video[0].time_base == Fraction(1, cut_timescale(idx.time_base))
        wanted = _wanted(len(v.pts_us))
        theirs = nvm.media.decode_selected(v, wanted, canonical=canonical)
        ours = decode_selected(
            idx,
            wanted,
            canonical=canonical,
            canonical_rgb=nvm.canonical_rgb.canonical_rgb,
            scans=cache,
        )
        for i in wanted:
            assert np.array_equal(_px(ours[i]), _px(theirs[i])), (c.name, i)
        checked += 1
    assert checked >= 30


def test_ineligible_cases_fail_closed_and_decode_source_frames(cases, tmp_path):
    _, eps = cases
    seen = set()
    for c in eps:
        if not c.reasons:
            continue
        idx = index_stream(c.ref)
        with pytest.raises(NotKeyframeAligned) as e:
            stream_copy_cut(c.ref, tmp_path / f"{c.name}.mp4")
        assert set(e.value.reasons) == set(c.reasons)
        assert not (tmp_path / f"{c.name}.mp4").exists()
        with pytest.raises(NotKeyframeAligned):
            index_stream(c.ref, decode=True)
        # Production range decode: the indexed frames, by absolute source pts (pre-roll from
        # the previous random-access point), equal the source decoded from its start.
        wanted = _wanted(len(idx.pts_us))
        ours = decode_selected(idx, wanted)
        truth = _source_truth(c.ref.uri, {idx.source_pts[i] for i in wanted})
        for i in wanted:
            assert np.array_equal(_px(ours[i]), truth[idx.source_pts[i]]), (c.name, i)
        seen |= set(c.reasons)
    assert {"edit_list", "not_random_access", "not_keyframe", "non_contiguous"} <= seen


def test_whole_file_index_equals_nvidia_probe_of_the_original(nvm, cases):
    """Whole-file sources NVIDIA would read directly: index == probe(original), and range
    decode == decode_selected(original), including the edit-list and mid-GOP-start files."""
    _, eps = cases
    for c in eps:
        if not c.original_fields:
            continue
        idx = index_stream(c.ref)
        v = nvm.media.probe(Path(c.ref.uri))
        for f in c.original_fields:
            assert getattr(idx, f) == getattr(v, f), (c.name, f)
        wanted = _wanted(len(v.pts_us))
        theirs = nvm.media.decode_selected(v, wanted)
        ours = decode_selected(idx, wanted)
        for i in wanted:
            assert np.array_equal(_px(ours[i]), _px(theirs[i])), (c.name, i)


def test_mov_colour_tags_differ_in_cut_but_pixels_do_not(nvm, cases):
    root, eps = cases
    c = next(e for e in eps if e.name == "mov_nclc")
    cut = stream_copy_cut(c.ref, root / "cuts" / "mov_tags.mp4")
    with av.open(c.ref.uri) as a, av.open(str(cut)) as b:
        fa, fb = next(a.decode(video=0)), next(b.decode(video=0))
    assert int(fa.colorspace) == int(fb.colorspace) == 5
    assert {int(fa.color_range), int(fb.color_range)} <= {0, 1}  # nclc 0 -> nclx 1 (advisory)
    idx = index_stream(c.ref)
    v = nvm.media.probe(cut)
    theirs = nvm.media.decode_selected(v, [0, 5], canonical=True)
    ours = decode_selected(
        idx, [0, 5], canonical=True, canonical_rgb=nvm.canonical_rgb.canonical_rgb
    )
    for i in (0, 5):
        assert np.array_equal(_px(ours[i]), _px(theirs[i]))


def test_low_timescale_cut_time_base_differs_from_source(cases):
    root, eps = cases
    c = next(e for e in eps if e.name == "timescale_30")
    idx = index_stream(c.ref)
    assert idx.time_base == Fraction(1, 30)
    cut = stream_copy_cut(c.ref, root / "cuts" / "ts30_tb.mp4")
    with av.open(str(cut)) as cc:
        assert cc.streams.video[0].time_base == Fraction(1, 15360)


def test_intra_refresh_decode_is_not_shifted(nvm, cases, tmp_path):
    """The H2 failure: a cut started at a recovery point decodes 0.2 s late. We refuse the cut,
    and production decode returns the frames at the indexed absolute timestamps."""
    _, eps = cases
    c = next(e for e in eps if e.name == "intra_refresh_ep1")
    idx = index_stream(c.ref)
    assert idx.source_timestamps_s[0] == 1.0
    ours = decode_selected(idx, [0, 5, 29])
    truth = _source_truth(c.ref.uri, set(idx.source_pts))
    for i in (0, 5, 29):
        assert np.array_equal(_px(ours[i]), truth[idx.source_pts[i]])
