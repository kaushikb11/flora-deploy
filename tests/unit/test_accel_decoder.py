"""``decode_once``'s decoder against a plain decode-from-frame-0 reference (no NVIDIA code).

The reference reproduces what NVIDIA's ``decode_selected`` does per frame (open with
``err_detect=explode``, ``thread_count=1``, enumerate decoder output from frame 0, swscale RGB);
the nvlink parity suite compares against NVIDIA's function itself.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import pytest

from flora_deploy.accel.framecache import SourceDecoder, build_seekmap
from flora_deploy.accel.state import AccelState
from flora_deploy.media import synth
from flora_deploy.media.synth import Spec


@dataclass(frozen=True)
class FakeVideo:
    path: Path
    sha256: str
    rate: Fraction
    pts_us: tuple
    duration_us: int
    size: tuple
    cfr: bool = True


def reference(path: Path):
    """(pts_us, [sha of RGB bytes per frame]) decoding from frame 0."""
    pts, digests = [], []
    with av.open(str(path), options={"err_detect": "explode"}) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 1
        zero = None
        for f in c.decode(s):
            zero = f.pts if zero is None else zero
            pts.append(Fraction(f.pts - zero) * s.time_base * 1_000_000)
            digests.append(hashlib.sha256(f.to_image().convert("RGB").tobytes()).hexdigest())
    return tuple(pts), digests


SPECS = {
    "h264_b": Spec(150, size=(64, 48), gop=12, bframes=3, seed=1),
    "h264_vfr": Spec(90, size=(48, 32), gop=9, time_base=Fraction(1, 90000),
                     pts=[i * 3000 + (i % 4) * 400 for i in range(90)], seed=2),
    "h264_intra_refresh": Spec(
        80, size=(64, 48), x264_params="intra-refresh=1:keyint=20:scenecut=0", bframes=0, seed=3
    ),
}


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("dec")
    out = {}
    for name, spec in SPECS.items():
        path = synth.encode(root / f"{name}.mp4", spec)
        pts, digests = reference(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        video = FakeVideo(path, sha, Fraction(spec.rate), pts, 0, spec.size)
        out[name] = (video, digests)
    return out


def _requests(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    reqs = [[0], [n - 1], list(range(n)), list(range(n // 2, n // 2 + 5))]
    for _ in range(25):
        a = rng.randrange(n)
        b = min(n, a + rng.randrange(1, 40))
        step = rng.randrange(1, 5)
        reqs.append(list(range(a, b, step)))
    return reqs


@pytest.mark.parametrize("workers,seek", [(1, False), (1, True), (4, True)])
@pytest.mark.parametrize("name", sorted(SPECS))
def test_decoder_matches_decode_from_zero(clips, name, workers, seek):
    video, digests = clips[name]
    n = len(video.pts_us)
    state = AccelState(workers=workers, seek=seek)
    dec = SourceDecoder(video, state, lambda: None, ValueError)
    assert dec.trusted
    for req in _requests(n, seed=len(name)):
        got = dec.decode(sorted(set(req)), canonical=False)
        assert list(got) == sorted(set(req))
        for i, im in got.items():
            assert im.mode == "RGB"
            assert hashlib.sha256(im.tobytes()).hexdigest() == digests[i], (name, i)
    stats = state.stats()
    if seek and name != "h264_intra_refresh":
        assert stats.get("decoder_restarts_random_access", 0) > 0
    if name == "h264_intra_refresh":
        # x264 intra refresh: keyframes in the container are not IDR random-access points
        assert stats.get("decoder_restarts_random_access", 0) == 0
    assert stats.get("random_access_fallbacks", 0) == 0
    dec.close()
    state.close()


def test_seekmap_refuses_mismatched_timestamps(clips):
    video, _ = clips["h264_b"]
    assert build_seekmap(str(video.path), video) is not None
    pts = list(video.pts_us)
    pts[3] += 1
    bad = FakeVideo(video.path, video.sha256, video.rate, tuple(pts), 0, video.size)
    assert build_seekmap(str(video.path), bad) is None
    short = FakeVideo(video.path, video.sha256, video.rate, video.pts_us[:-1], 0, video.size)
    assert build_seekmap(str(video.path), short) is None


def test_decoder_errors_like_nvidia(clips):
    video, _ = clips["h264_b"]
    pts = list(video.pts_us)
    pts[10] += 1
    shifted = FakeVideo(video.path, video.sha256, video.rate, tuple(pts), 0, video.size)
    state = AccelState(workers=1)
    dec = SourceDecoder(shifted, state, lambda: None, ValueError)
    assert len(dec.decode([3, 9], canonical=False)) == 2
    with pytest.raises(ValueError, match="Source frame timestamps changed while decoding."):
        dec.decode([10], canonical=False)
    longer = FakeVideo(video.path, video.sha256, video.rate, video.pts_us + (10**9,), 0,
                       video.size)
    dec2 = SourceDecoder(longer, state, lambda: None, ValueError)
    with pytest.raises(ValueError, match="A requested source frame could not be decoded."):
        dec2.decode([len(video.pts_us)], canonical=False)
    forged = FakeVideo(video.path, "0" * 64, video.rate, video.pts_us, 0, video.size)
    assert not SourceDecoder(forged, state, lambda: None, ValueError).trusted
    state.close()


def test_overlapping_episodes_are_scoped_and_bounded(tmp_path):
    """W3-A: episodes overlap on threads sharing one state. Each episode's decoders close when
    *it* exits (not when all do), open decoders and cache bytes stay bounded across episodes,
    and every frame still equals a decode from frame 0."""
    import os
    import threading
    import types
    from concurrent.futures import ThreadPoolExecutor

    from flora_deploy.accel.framecache import make_frames_for

    def clip(name, spec):
        path = synth.encode(tmp_path / f"{name}.mp4", spec)
        pts, digests = reference(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return FakeVideo(path, sha, Fraction(spec.rate), pts, 0, spec.size), digests

    long_video, long_digests = clip("long", Spec(120, size=(64, 48), gop=12, seed=99))
    shorts = [clip(f"s{i}", Spec(30, size=(48, 32), gop=10, seed=i)) for i in range(12)]
    nv = types.SimpleNamespace(media=types.SimpleNamespace(InputError=ValueError),
                               vendor=lambda name: None)
    state = AccelState(workers=2, cache_bytes=60_000, max_total_cache_bytes=100_000,
                       max_open_decoders=3)
    frames_for = make_frames_for(nv, lambda *a, **k: pytest.fail("delegated"), state)

    def nfds():
        return len(os.listdir("/dev/fd"))

    def same(got, digests):
        return all(hashlib.sha256(im.tobytes()).hexdigest() == digests[i]
                   for i, im in got.items())

    fd0 = nfds()
    started, release = threading.Event(), threading.Event()
    long_errors: list = []

    def long_episode():
        try:
            with state.episode():
                for k, wanted in enumerate(([0, 5, 30], list(range(40, 80, 3)), [119])):
                    assert same(frames_for({"main": long_video}, wanted), long_digests)
                    if k == 0:
                        started.set()
                release.wait(60)
                # still usable after the other episodes closed theirs
                assert same(frames_for({"main": long_video}, [3, 100]), long_digests)
        except BaseException as exc:  # noqa: BLE001 - reported below
            long_errors.append(exc)
            started.set()

    def short_episode(item):
        video, digests = item
        with state.episode():
            for wanted in ([1, 7, 20], list(range(0, 30, 4)), [29, 2]):
                got = frames_for({"main": video}, wanted)
                assert list(got) == sorted(set(wanted)) and same(got, digests)
            return state.scope()

    t = threading.Thread(target=long_episode)
    t.start()
    assert started.wait(60)
    with ThreadPoolExecutor(4) as pool:
        scopes = list(pool.map(short_episode, shorts))
    stats = state.stats()
    # every short episode is closed although the long one is still running
    assert all(s.closed and not s.decoders and s.nbytes() == 0 for s in scopes)
    assert stats["active_episodes"] == 1
    assert stats["open_decoders"] <= 1  # only the long episode's decoder may stay open
    assert nfds() - fd0 <= 1
    # bounded across concurrent episodes: 3 idle + one busy decoder per running request
    assert stats["peak_open_decoders"] <= 3 + 5
    assert stats["cache_bytes_total_peak"] <= 100_000
    release.set()
    t.join(60)
    assert not long_errors, long_errors
    stats = state.stats()
    assert stats["active_episodes"] == 0 and stats["open_decoders"] == 0
    assert stats["cache_bytes_total"] == 0
    assert stats["episodes"] == 13
    state.close()
    assert nfds() == fd0
