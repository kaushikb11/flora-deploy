"""L0(a)/(b): index pass == NVIDIA ``probe`` on the stream-copy cut; range decode ==
NVIDIA ``decode_selected`` on the cut; envelope verdicts == NVIDIA accept/raise."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from flora_deploy.media.cuts import stream_copy_cut
from flora_deploy.media.index import (
    NotKeyframeAligned,
    cut_timescale,
    envelope_checks,
    estimated_cut_bytes,
    index_stream,
    scan_stream,
)
from flora_deploy.media.rangedecode import decode_selected

try:
    from .media_fixtures import build_envelope_fixtures, build_fixtures, load_nv_media
except ImportError:  # tests/nvlink is not a package
    from media_fixtures import build_envelope_fixtures, build_fixtures, load_nv_media

pytestmark = pytest.mark.nvlink


@pytest.fixture
def nvm(nv_code):
    return load_nv_media(nv_code("tempseg_code"))


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    return root, build_fixtures(root)


def _cut(ep_name, role, ref, root: Path) -> Path:
    return stream_copy_cut(ref, root / "cuts" / f"{ep_name}_{role}.mp4")


def test_index_equals_probe_on_cut(nvm, fixtures):
    root, eps = fixtures
    scans: dict[str, object] = {}
    checked = 0
    for ep in eps:
        for role, ref in ep.cameras.items():
            scan = scans.setdefault(ref.uri, scan_stream(ref.uri))
            idx = index_stream(ref, scan=scan)
            assert idx.starts_on_keyframe == ep.keyframe_aligned, ep.name
            if not ep.keyframe_aligned:
                with pytest.raises(NotKeyframeAligned):
                    index_stream(ref, scan=scan, decode=True)
                continue
            for decode in (False, True):
                idx = index_stream(ref, scan=scan, decode=decode)
                cut = _cut(ep.name, role, ref, root)
                v = nvm.media.probe(cut)
                where = f"{ep.name}/{role} decode={decode}"
                assert idx.rate == v.rate, where
                assert idx.pts_us == v.pts_us, where
                assert idx.duration_us == v.duration_us, where
                assert idx.size == v.size, where
                assert idx.cfr == v.cfr, where
                with av.open(str(cut)) as c:
                    s = c.streams.video[0]
                    assert idx.codec == s.codec_context.codec.canonical_name, where
                    assert idx.pix_fmt == s.codec_context.pix_fmt, where
                    # movenc keeps the source timescale only when >= 10000 (docs/media-notes.md)
                    assert s.time_base == Fraction(1, cut_timescale(idx.time_base)), where
                assert estimated_cut_bytes(idx) >= cut.stat().st_size, where
                checked += 1
    assert checked >= 2 * 25


def test_source_timestamps_are_the_source_frames(fixtures):
    _, eps = fixtures
    ep = next(e for e in eps if e.name == "h264_cfr_ep1")
    ref = ep.cameras["main"]
    idx = index_stream(ref)
    with av.open(ref.uri) as c:
        times = [float(f.pts * f.time_base) for f in c.decode(video=0)]
    first = times.index(idx.source_timestamps_s[0])
    assert list(idx.source_timestamps_s) == times[first : first + len(idx.pts_us)]


def test_concatenated_rate_differs_but_index_matches_cut(nvm, fixtures):
    """The rule that motivates the index: whole-file average_rate != per-episode cut rate."""
    root, eps = fixtures
    mixed = [e for e in eps if e.name.startswith("mixed_rate")]
    with av.open(mixed[0].cameras["main"].uri) as c:
        whole = c.streams.video[0].average_rate
    rates = []
    for ep in mixed:
        ref = ep.cameras["main"]
        v = nvm.media.probe(_cut(ep.name, "main", ref, root))
        assert index_stream(ref).rate == v.rate
        rates.append(v.rate)
    assert all(r != whole for r in rates), (whole, rates)
    assert rates[0] == 30 and rates[1] == 15


def test_non_keyframe_episode_cannot_be_cut(fixtures, tmp_path):
    _, eps = fixtures
    ep = next(e for e in eps if e.name == "one_gop_ep1")
    with pytest.raises(NotKeyframeAligned):
        stream_copy_cut(ep.cameras["main"], tmp_path / "x.mp4")


def _pixels(img):
    return np.asarray(img.convert("RGB"))


@pytest.mark.parametrize("canonical", [False, True])
def test_range_decode_equals_nvidia_decode_selected(nvm, fixtures, monkeypatch, canonical):
    root, eps = fixtures
    monkeypatch.setitem(__import__("sys").modules, "canonical_rgb", nvm.canonical_rgb)
    branches = {"canonical": 0, "swscale": 0}
    for ep in eps:
        if not ep.keyframe_aligned:
            continue
        for role, ref in ep.cameras.items():
            idx = index_stream(ref)
            n = len(idx.pts_us)
            wanted = sorted({0, 1, n // 3, n // 2, n - 2, n - 1})
            v = nvm.media.probe(_cut(ep.name, role, ref, root))
            theirs = nvm.media.decode_selected(v, wanted, canonical=canonical)
            ours = decode_selected(
                idx, wanted, canonical=canonical, canonical_rgb=nvm.canonical_rgb.canonical_rgb
            )
            assert set(ours) == set(theirs)
            for i in wanted:
                assert ours[i].mode == theirs[i].mode and ours[i].size == theirs[i].size
                assert np.array_equal(_pixels(ours[i]), _pixels(theirs[i])), (ep.name, role, i)
            with av.open(ref.uri) as c:
                f = next(c.decode(video=0))
                from flora_deploy.media.rangedecode import canonical_eligible

                branches["canonical" if canonical and canonical_eligible(f) else "swscale"] += 1
    if canonical:
        assert branches["canonical"] > 0 and branches["swscale"] > 0, branches


def test_range_decode_pts_guard(nvm, fixtures):
    from dataclasses import replace
    from fractions import Fraction

    from flora_deploy.media.index import MediaError

    _, eps = fixtures
    ref = next(e for e in eps if e.name == "h264_cfr_ep1").cameras["main"]
    idx = index_stream(ref)
    bad = replace(idx, pts_us=(idx.pts_us[0] + Fraction(1),) + idx.pts_us[1:])
    with pytest.raises(MediaError):
        decode_selected(bad, [0])


# ---------------------------------------------------------------------------------------------
# Envelope verdicts vs NVIDIA's DataEnginePipeline._inputs (api.py:11-38: size, probe, sync)


@pytest.fixture(scope="module")
def envelope_cases(tmp_path_factory):
    return build_envelope_fixtures(tmp_path_factory.mktemp("envelope"))


def _nv_accepts(nvm, paths: dict[str, Path], profile: str) -> tuple[bool, str]:
    try:
        nvm.api.DataEnginePipeline._inputs(
            str(paths["main"]),
            profile,
            paths.get("left_wrist"),
            paths.get("right_wrist"),
            False,
            "",
        )
        return True, ""
    except nvm.media.InputError as e:
        return False, str(e)


def test_envelope_agrees_with_nvidia(nvm, envelope_cases, fixtures, tmp_path):
    _, eps = fixtures
    cases = dict(envelope_cases)
    for ep in eps:
        if ep.keyframe_aligned:
            cases[ep.name] = ep.cameras
    seen = {True: 0, False: 0}
    for name, cams in cases.items():
        streams = {role: index_stream(ref) for role, ref in cams.items()}
        paths = {role: _cut(name, role, ref, tmp_path) for role, ref in cams.items()}
        profile = "yam" if len(cams) == 3 else "human"
        verdict = envelope_checks(streams, profile=profile)
        ok, msg = _nv_accepts(nvm, paths, profile)
        assert verdict.ok == ok, (name, verdict.reasons, msg)
        seen[ok] += 1
    assert seen[True] and seen[False] >= 6, seen


@pytest.mark.parametrize(
    "kw,all_ok",
    [
        # x264 with B-frames: some per-episode files end on a short last sample, so after
        # concatenation one camera's episode cut has rate != 30 and NVIDIA rejects the 3-view
        # sync. We must agree with that, not accept (docs/media-notes.md).
        ({}, None),
        ({"bframes": 0}, True),
        ({"codec": "libsvtav1", "size": (64, 64)}, True),
    ],
)
def test_lerobot_v3_source_episodes_match_probe_and_nvidia_inputs(nvm, tmp_path, kw, all_ok):
    from flora_deploy.media import synth
    from flora_deploy.media.index import index_episode
    from flora_deploy.media.sources import open_source
    from flora_deploy.media.synth import Spec

    keys = ("observation.images.top", "observation.images.left", "observation.images.right")
    cmap = dict(zip(keys, ("main", "left_wrist", "right_wrist"), strict=True))
    episodes = [
        {k: Spec(n, seed=e * 3 + c, **kw) for c, k in enumerate(keys)}
        for e, n in enumerate((11, 17, 9))
    ]
    root = synth.lerobot_v3_tree(tmp_path / "ds", episodes, ["stack"], scratch=tmp_path / "s")
    src = open_source(root, camera_map=cmap)
    scans: dict = {}
    for ep in src.list_episodes():
        ix = index_episode(ep.episode_id, src.name, ep.cameras, scans=scans, profile="yam")
        paths = {}
        for role, ref in ep.cameras.items():
            paths[role] = stream_copy_cut(ref, tmp_path / "cuts" / f"{ep.episode_id}_{role}.mp4")
            v = nvm.media.probe(paths[role])
            s = ix.streams[role]
            ours = (s.rate, s.pts_us, s.duration_us, s.size, s.cfr)
            assert ours == (v.rate, v.pts_us, v.duration_us, v.size, v.cfr)
        ok, msg = _nv_accepts(nvm, paths, "yam")
        assert ok == ix.envelope.ok, (msg, ix.envelope.reasons)
        if all_ok:
            assert ok, msg
