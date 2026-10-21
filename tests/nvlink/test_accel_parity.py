"""OPT-1 parity: every ``flora_deploy.accel`` patch vs NVIDIA's original function.

NVIDIA's unmodified code is the oracle (PLAN.md D17). Each test runs the original NVIDIA
function and the patched one on the same generated inputs and asserts bit-identical results:

- ``frames_for`` (``decode_once``): H.264 (B-frames, odd widths, BT.709, full range), VFR,
  AV1 and HEVC open-GOP (CRA + RASL) sources, 1-view and 3-view (YAM and Dex mosaics),
  forward / overlapping / backward / edge / duplicate / empty requests, eviction under a tiny
  cache, serial and threaded, with and without random-access restarts, plus NVIDIA's errors;
- ``human_transport`` (``transport_pool``), ``caption_single``'s PNG receipts and
  ``load_frames`` (``png_inmem``), ``normalized`` / ``source_frames`` (``geometry_pool``);
- end to end: ``pipeline.segment`` + ``caption_single`` / ``caption_three`` through the real
  Flora processors with a canned ``generate`` whose TempSeg answers are valid, non-empty
  predictions (so segments, multi-window stitching and caption plans run): every tensor
  NVIDIA's ``generate`` hashes (``mm_token_type_ids`` included), segments, captions, audits;
- W3-A: ``ALL_PATCHES`` sharing one state across overlapping episode threads, against serial
  unpatched NVIDIA, with bounded caches / open decoders and per-episode release.

Nothing NVIDIA ships is copied: the code is imported from the local pin cache at test time and
the fixtures are generated with ``flora_deploy.media.synth``.
"""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flora_deploy.accel import ALL_PATCHES, apply_patches
from flora_deploy.accel.pngmem import PrefetchedPNG
from flora_deploy.media import synth
from flora_deploy.media.synth import Spec
from flora_deploy.nv import Family
from flora_deploy.nv.linker import load_family, verify_code_manifest
from flora_deploy.nv.session import CachedSession

pytestmark = pytest.mark.nvlink

ROLES3 = ("main", "left_wrist", "right_wrist")
CONFIGS = {
    "default": dict(),
    "serial-noseek": dict(workers=1, seek=False),
    "tiny-cache": dict(workers=4, cache_bytes=300_000),
}


# ---------------------------------------------------------------------------------------------
# fixtures


@pytest.fixture(scope="module")
def nv_dirs():
    import os

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[k]) for k in ("tempseg_code", "captioner_code")}
    if not (dirs["tempseg_code"] / "batch_utils").is_dir():
        pytest.skip("NVIDIA code pin not fetched (scripts/fetch_nv_code.py)")
    return dirs


@pytest.fixture(scope="module")
def nv(nv_dirs):
    return load_family(Family.SEGCAP, nv_dirs["tempseg_code"])


@pytest.fixture(scope="module")
def procs(nv, nv_dirs):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    for key, path in nv_dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    return {
        "tempseg": nv.models.processor("tempseg", nv_dirs["tempseg_code"]),
        "captioner": nv.models.processor("captioner", nv_dirs["captioner_code"]),
    }


def _vfr(n: int, seed: int) -> Spec:
    return Spec(
        n,
        seed=seed,
        size=(96, 64),
        time_base=Fraction(1, 90000),
        pts=[i * 3000 + ((i * 7 + seed) % 5) * 330 for i in range(n)],
        gop=12,
    )


_X265_OPEN = "open-gop=1:bframes=4:keyint=24:min-keyint=24:scenecut=0:log-level=0"

SOURCES = {
    # name: (specs per role, tags)
    "h264_bframes_1v": [Spec(300, size=(112, 64), gop=24, bframes=2, seed=1)],
    "h264_odd_1v": [Spec(150, size=(72, 40), gop=10, seed=2)],
    "h264_vfr_1v": [_vfr(240, 3)],
    "av1_1v": [Spec(180, codec="libsvtav1", size=(64, 64), gop=30, seed=4)],
    "h264_3v": [Spec(300, size=(112, 64), gop=20, bframes=2, seed=10 + k) for k in range(3)],
    "h264_odd_bt709_3v": [
        Spec(120, size=(72, 40), gop=16, seed=20),
        Spec(120, size=(72, 40), gop=16, seed=21, colorspace=1),
        Spec(120, size=(80, 48), gop=16, seed=22, color_range=2),
    ],
    "av1_3v": [Spec(150, codec="libsvtav1", size=(64, 64), gop=25, seed=30 + k) for k in range(3)],
    # HEVC open GOP: CRA random-access points with leading RASL pictures (review p1)
    "hevc_open_gop_1v": [Spec(240, codec="libx265", size=(128, 64), gop=24, seed=40,
                              codec_options={"x265-params": _X265_OPEN})],
    "hevc_3v": [Spec(200, codec="libx265", size=(128, 64), gop=24, seed=41 + k,
                     codec_options={"x265-params": _X265_OPEN}) for k in range(3)],
    # long enough for several TempSeg windows (stitching, overlapping frame plans)
    "h264_long_1v": [Spec(900, size=(64, 48), gop=30, bframes=2, seed=50)],
}


@pytest.fixture(scope="module")
def sources(nv, tmp_path_factory):
    root = tmp_path_factory.mktemp("accel")
    out = {}
    for name, specs in SOURCES.items():
        roles = ROLES3 if len(specs) == 3 else ("main",)
        out[name] = {
            role: nv.media.probe(synth.encode(root / f"{name}_{role}.mp4", spec))
            for role, spec in zip(roles, specs, strict=True)
        }
    return out


# ---------------------------------------------------------------------------------------------
# helpers


def image_sig(im):
    return (type(im), im.mode, im.size, hashlib.sha256(im.tobytes()).hexdigest(), dict(im.info),
            im.readonly)


def assert_same_frames(ref: dict, got: dict, where=""):
    assert list(ref) == list(got), where
    for k in ref:
        assert image_sig(ref[k]) == image_sig(got[k]), (where, k)


def outcome(fn, *args, **kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - compared between oracle and patch
        return "raise", (type(exc), str(exc))


def assert_same_outcome(ref, got, where=""):
    assert ref[0] == got[0], (where, ref, got)
    if ref[0] == "raise":
        assert ref[1] == got[1], where
    else:
        assert_same_frames(ref[1], got[1], where)


def requests_for(n: int) -> list:
    return [
        {0},
        set(range(0, n, 7)),
        set(range(n // 3, n // 2)),  # backward of the cursor: random-access restart
        {n - 1},  # last frame
        set(range(5, min(n, 40), 3)),  # backward again, overlaps the first request
        {n // 2, 2, n - 2},
        [9, 3, 9, 1],  # list with a duplicate: NVIDIA's dict order follows iteration
        set(range(n // 2, n)),  # overlapping tail (cache hits + forward decode)
        set(range(n)),  # everything
        set(),
        {-1},
        {n},
        {0, n},
    ]


# ---------------------------------------------------------------------------------------------
# decode_once


@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_decode_once_frames_identical(nv, sources, source, config):
    videos = sources[source]
    original = nv.media.frames_for
    assert nv.pipeline.frames_for is original
    n = len(videos["main"].pts_us)
    mosaics = ("YAM", "Dex", "Bad") if len(videos) == 3 else ("YAM", "Bad")
    with apply_patches(nv, ["decode_once"], **CONFIGS[config]) as active:
        assert nv.pipeline.frames_for is not original
        assert nv.media.frames_for is not original
        with active.episode():
            for mosaic in mosaics:
                for k, wanted in enumerate(requests_for(n)):
                    where = f"{source}/{config}/{mosaic}/#{k}"
                    ref = outcome(original, videos, wanted, mosaic=mosaic)
                    got = outcome(nv.pipeline.frames_for, videos, wanted, mosaic=mosaic)
                    assert_same_outcome(ref, got, where)
            stats = active.stats()
    assert nv.pipeline.frames_for is original and nv.media.frames_for is original
    assert stats["frames_for_calls"] > 0
    assert stats.get("untrusted_delegations", 0) == 0
    if config != "serial-noseek":
        # every source here starts on a verified random-access point, has several GOPs, and
        # its restarts from them never needed NVIDIA's decode-from-0 fallback
        assert stats.get("seekmap_built", 0) == len(videos), stats
        assert stats.get("decoder_restarts_random_access", 0) > 0, stats
        assert stats.get("random_access_fallbacks", 0) == 0, stats


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_decode_once_gop_parallel_identical(nv, sources, source):
    """A cold request spanning the file is split across decoders started at verified
    random-access points; the frames are NVIDIA's."""
    videos = sources[source]
    original = nv.media.frames_for
    n = len(videos["main"].pts_us)
    with apply_patches(nv, ["decode_once"], workers=4) as active:
        for wanted in (set(range(n)), set(range(1, n, 3))):
            with active.episode():
                assert_same_frames(original(videos, wanted), nv.pipeline.frames_for(videos, wanted))
        stats = active.stats()
    assert stats["parallel_chunks"] >= 2 * len(videos), stats
    assert stats.get("random_access_fallbacks", 0) == 0, stats


def test_decode_once_default_mosaic_and_missing_roles(nv, sources):
    original = nv.media.frames_for
    videos = sources["h264_3v"]
    two = {k: videos[k] for k in ("main", "left_wrist")}
    other = {"cam": sources["h264_odd_1v"]["main"]}
    with apply_patches(nv, ["decode_once"]):
        patched = nv.pipeline.frames_for
        assert_same_outcome(outcome(original, videos, {3, 4}), outcome(patched, videos, {3, 4}))
        for vids in (two, other):  # NVIDIA raises KeyError for missing roles
            assert_same_outcome(outcome(original, vids, {1, 2}), outcome(patched, vids, {1, 2}))
            assert_same_outcome(outcome(original, vids, set()), outcome(patched, vids, set()))


def test_decode_once_decodes_each_frame_once_for_monotone_windows(nv, sources):
    videos = sources["h264_bframes_1v"]
    n = len(videos["main"].pts_us)
    windows = [set(range(a, min(n, a + 40), 2)) for a in range(0, n, 40)]
    with apply_patches(nv, ["decode_once"], workers=1) as active:
        with active.episode():
            for w in windows:
                nv.pipeline.frames_for(videos, w)
            stats = active.stats()
    # one decoder pass over the file (NVIDIA: one pass from frame 0 per window)
    assert stats["decoder_restarts_from_0"] == 1
    assert stats["frames_decoded"] <= max(max(w) for w in windows) + 1


def test_decode_once_untrusted_file_delegates(nv, sources):
    original = nv.media.frames_for
    video = sources["h264_odd_1v"]["main"]
    forged = {"main": dataclasses.replace(video, sha256="0" * 64)}
    with apply_patches(nv, ["decode_once"]) as active:
        got = nv.pipeline.frames_for(forged, {1, 5})
        assert active.stats()["untrusted_delegations"] == 1
    assert_same_frames(original(forged, {1, 5}), got)


def test_decode_once_changed_timestamps_raise_like_nvidia(nv, sources):
    original = nv.media.frames_for
    video = sources["h264_odd_1v"]["main"]
    pts = list(video.pts_us)
    pts[7] += 1
    # same file (sha matches), different probed timestamps: NVIDIA's pts check must fire
    shifted = {"main": dataclasses.replace(video, pts_us=tuple(pts))}
    with apply_patches(nv, ["decode_once"]):
        for wanted in ({3, 7}, {7}, {3, 6}):
            assert_same_outcome(
                outcome(original, shifted, wanted), outcome(nv.pipeline.frames_for, shifted, wanted)
            )


def test_patches_keep_nvidia_manifests_and_restore(nv):
    before = {
        t: getattr(*_resolve(nv, t)) for p in _registered().values() for t in p.targets
    }
    active = apply_patches(nv, ALL_PATCHES)
    prov = active.provenance()
    assert [p["name"] for p in prov["patches"]] == list(ALL_PATCHES)
    assert all(p["parity_test"].startswith("tests/nvlink/test_accel_parity.py::") for p in
               prov["patches"])
    # NVIDIA's own integrity checks still hold: files untouched
    assert verify_code_manifest(nv.code_dir) > 0
    vendor = nv.package_dir / "vendor"
    manifest = json.loads((vendor / "MANIFEST.json").read_text())
    for rel, digest in manifest.items():
        assert hashlib.sha256((vendor / rel).read_bytes()).hexdigest() == digest, rel
    active.restore()
    after = {t: getattr(*_resolve(nv, t)) for t in before}
    assert before == after


def _registered():
    from flora_deploy.accel import registered

    return registered()


def _resolve(nv, target):
    from flora_deploy.accel.registry import resolve_target

    return resolve_target(nv, target)


# ---------------------------------------------------------------------------------------------
# transport_pool


def _noise(size, seed, mode="RGB"):
    rng = np.random.default_rng(seed)
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 3 + seed) % 256, (y * 5) % 256, (x + y) % 256], axis=-1)
    arr = np.clip(base + rng.integers(-30, 31, size=(h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert(mode)


@pytest.mark.parametrize("workers", [1, 4])
def test_transport_pool_identical(nv, workers):
    original = nv.pipeline.human_transport
    frames = {i: _noise((97 + i % 3, 61), i) for i in (5, 2, 9, 1)}
    frames[7] = frames[2].copy()  # duplicate content: a memo hit
    frames[8] = _noise((64, 48), 8, mode="L")
    with apply_patches(nv, ["transport_pool"], workers=workers) as active:
        for _ in range(2):  # second round is all hits
            assert_same_frames(original(frames), nv.pipeline.human_transport(frames))
        # round 2 is all hits (round 1 may race on the duplicate when threaded)
        assert active.stats()["transport_cache_hits"] >= len(frames)
        bad = {1: frames[1], 2: "not an image"}
        ref = outcome(original, bad)
        got = outcome(nv.pipeline.human_transport, bad)
        assert ref[0] == got[0] == "raise" and ref[1][0] is got[1][0]


# ---------------------------------------------------------------------------------------------
# png_inmem


def test_png_prefetched_bytes_identical(nv, tmp_path):
    from flora_deploy.accel.pngmem import _encode_job
    from flora_deploy.accel.state import AccelState

    state = AccelState(workers=1)
    for k, im in enumerate([_noise((97, 61), 1), _noise((640, 360), 2), _noise((33, 17), 3)]):
        ref_path = tmp_path / f"ref{k}.png"
        im.save(ref_path, format="PNG")  # pipeline.py:99
        got = _encode_job(state, im)
        assert got == ref_path.read_bytes()
        tagged = im.copy()
        tagged.__class__ = PrefetchedPNG
        fut = state.submit(_encode_job, state, tagged)
        tagged._flora_png = fut
        out = tmp_path / f"got{k}.png"
        tagged.save(out, format="PNG")
        assert out.read_bytes() == ref_path.read_bytes()
        tagged.save(tmp_path / f"other{k}.png", format="PNG", compress_level=1)  # passthrough
        im.save(tmp_path / f"ref_other{k}.png", format="PNG", compress_level=1)
        assert (tmp_path / f"other{k}.png").read_bytes() == (
            tmp_path / f"ref_other{k}.png"
        ).read_bytes()
    state.close()


def _caption_sample_spy(nv, monkeypatch):
    receipts = []
    real = nv.production.caption_sample

    def spy(plan, rec, roles):
        receipts.append(copy.deepcopy(rec))
        return real(plan, rec, roles)

    monkeypatch.setattr(nv.production, "caption_sample", spy)
    return receipts


def test_png_inmem_receipts_identical(nv, procs, sources, monkeypatch, tmp_path):
    """Receipts (``png_sha256``, size, path) and load_frames output equal NVIDIA's."""
    run = _Runner(nv, procs, monkeypatch)
    for name in ("h264_bframes_1v", "h264_vfr_1v"):
        videos = sources[name]
        segs = _segments(videos)
        for instruction in ("", "Fold the towel."):
            receipts_ref = _caption_sample_spy(nv, monkeypatch)
            ref = run.caption_single(videos, segs, "Human", instruction)
            monkeypatch.undo()
            receipts_got = _caption_sample_spy(nv, monkeypatch)
            with apply_patches(nv, ["png_inmem"]) as active:
                got = run.caption_single(videos, segs, "Human", instruction)
                stats = active.stats()
            monkeypatch.undo()
            assert receipts_ref == receipts_got and receipts_ref
            assert ref == got, name
            assert stats["png_prefetched"] > 0
            assert stats["png_decode_cache_hits"] > 0  # frames shared by several blocks


def test_png_inmem_load_frames_errors_identical(nv, tmp_path):
    ui = nv.vendor("flora_unified.unified_input")
    original = ui.load_frames
    im = _noise((640, 480), 5)
    path = tmp_path / "000000000001.png"
    im.save(path, format="PNG")
    good = dict(png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=[640, 480])

    def sample(paths, receipts, camera_count=1):
        target = "pick"
        block = dict(
            role="TARGET", source_start_frame=0, source_end_frame_exclusive=2,
            block_local_timestamp_indices=[0, 1], paths=paths, frames=receipts,
            clock_fps_num=30, clock_fps_den=1, effective_fps_num=30, effective_fps_den=1,
        )
        return dict(
            blocks=[block], camera_count=camera_count,
            request=dict(target=dict(
                native_span=dict(start_frame=None), text=target,
                sha256=hashlib.sha256(target.encode()).hexdigest(),
            )),
        )

    cases = {
        "ok": sample([path.name, path.name], [good, good]),
        "ok-3view-canvas": sample([path.name, path.name], [good, good], camera_count=3),
        "changed": sample([path.name, path.name], [good, dict(good, png_sha256="0" * 64)]),
        "dims": sample([path.name, path.name], [good, dict(good, size=[1, 1])]),
        "unowned": sample(["../x.png", path.name], [good, good]),
        "invalid": sample([path.name], [good]),
    }
    with apply_patches(nv, ["png_inmem"]):
        patched = ui.load_frames
        assert patched is not original
        for name, s in cases.items():
            for canvas in (True, False):
                ref = outcome(original, s, tmp_path, canvas)
                got = outcome(patched, s, tmp_path, canvas)
                assert ref[0] == got[0], name
                if ref[0] == "raise":
                    assert ref[1] == got[1], name
                    continue
                (rv, rs), (gv, gs) = ref[1], got[1]
                assert rs == gs, name
                for a, b in zip(rv, gv, strict=True):
                    assert [image_sig(x) for x in a] == [image_sig(x) for x in b], name


# ---------------------------------------------------------------------------------------------
# geometry_pool


@pytest.mark.parametrize("workers", [1, 4])
def test_geometry_pool_identical(nv, monkeypatch, workers):
    rcg = nv.vendor("robot_caption_geometry")
    orig_norm, orig_src = rcg.normalized, rcg.source_frames
    frames = {i: _noise((768, 648), 100 + i) for i in range(12)}
    plan = dict(
        target_id="event_0000",
        blocks=[
            dict(role="BEFORE", indices=[0, 1, 2, 3]),
            dict(role="TARGET", indices=[4, 4]),
            dict(role="AFTER", indices=[5, 7, 9, 11, 10, 6]),
        ],
    )
    with apply_patches(nv, ["geometry_pool"], workers=workers):
        patched_norm, patched_src = rcg.normalized, rcg.source_frames
        assert patched_norm is not orig_norm
        for limit in (rcg.PACKED_IMAGE_LIMIT_BYTES, 4_000_000, 1_500_000, 10):
            monkeypatch.setattr(rcg, "PACKED_IMAGE_LIMIT_BYTES", limit)
            ref = outcome(orig_norm, plan, frames)
            got = outcome(patched_norm, plan, frames)
            assert ref[0] == got[0], limit
            if ref[0] == "raise":
                assert ref[1] == got[1]
                continue
            assert ref[1] == got[1], limit  # dicts incl. PNG blobs, scale, byte counts
            assert_same_frames(orig_src(ref[1]), patched_src(got[1]))
            # NVIDIA's re-check path (expected geometry)
            again_ref = orig_norm(plan, frames, expected=ref[1])
            assert again_ref == patched_norm(plan, frames, expected=ref[1])
        monkeypatch.undo()
        missing = dict(plan, blocks=[dict(role="TARGET", indices=[0, 99])])
        ref, got = outcome(orig_norm, missing, frames), outcome(patched_norm, missing, frames)
        assert ref[0] == got[0] == "raise" and ref[1] == got[1]
        tampered = orig_norm(plan, frames)
        tampered["frame_blocks"][0]["images"][0]["rgb_sha256"][0] = "0" * 64
        ref, got = outcome(orig_src, tampered), outcome(patched_src, tampered)
        assert ref[0] == got[0] == "raise" and ref[1] == got[1]


# ---------------------------------------------------------------------------------------------
# end to end through NVIDIA's pipeline


class _Runner:
    """Drives NVIDIA's pipeline with the real processors and a canned ``generate``.

    The canned TempSeg answers are *valid, non-empty* predictions: fixed events in global ticks
    (``EVENTS_S``), clipped to each window's valid range with NVIDIA's ``lo`` / ``ro`` rules, so
    segments, multi-window stitching and caption plans downstream of them are exercised. Every
    tensor NVIDIA's ``generate`` hashes (incl. ``mm_token_type_ids``) is logged, with NVIDIA's
    own ``input_tensor_digest`` formula (``models.py:171-172``)."""

    def __init__(self, nv, procs, monkeypatch):
        self.nv, self.procs, self.monkeypatch = nv, procs, monkeypatch

    def _install(self):
        # (re)applied per call: tests call ``monkeypatch.undo()`` between runs
        pipeline = self.nv.pipeline
        if not getattr(pipeline.temporal_inputs, "_parity", False):
            original = pipeline.temporal_inputs

            def temporal_inputs(frames, window, *args, **kwargs):
                _WINDOW.set(window)  # predict() calls generate right after, in this context
                return original(frames, window, *args, **kwargs)

            temporal_inputs._parity = True
            self.monkeypatch.setattr(pipeline, "temporal_inputs", temporal_inputs)
        if pipeline.generate is not _canned_generate:
            self.monkeypatch.setattr(pipeline, "generate", _canned_generate)

    def _session(self, procs=None):
        procs = procs or self.procs

        class Fake:
            audit: dict = {}

            def load(self, kind):
                return object(), procs[kind]

            def close(self):
                pass

        return CachedSession(Fake)

    def _run(self, fn, procs=None):
        self._install()
        log: list = []
        token = _LOG.set(log)
        try:
            out = fn(self._session(procs))
        finally:
            _LOG.reset(token)
        return out, [{k: v for k, v in d.items() if k != "digest"} for d in log]

    def segment(self, videos, pool, instruction="", procs=None):
        return self._run(lambda s: self.nv.pipeline.segment(
            videos, pool, s, instruction, lambda *a, **k: None), procs)

    def caption_single(self, videos, segments, profile, instruction="", procs=None):
        return self._run(lambda s: self.nv.pipeline.caption_single(
            videos, segments, profile, s, instruction, lambda *a, **k: None), procs)

    def caption_three(self, videos, segments, profile, instruction="", procs=None):
        return self._run(lambda s: self.nv.pipeline.caption_three(
            videos, segments, s, instruction, lambda *a, **k: None, profile=profile), procs)


_LOG: contextvars.ContextVar[list] = contextvars.ContextVar("parity_log")
_WINDOW: contextvars.ContextVar = contextvars.ContextVar("parity_window", default=None)
TICKS_PER_S = 20  # FineWindow ticks (probed: a 10 s window spans 200 ticks)
EVENTS_S = ((0.5, 2.5), (3.0, 5.5), (6.5, 9.0))  # repeated every 10 s


def nvidia_input_digest(inputs) -> tuple[dict, str]:
    """NVIDIA's ``input_tensors`` / ``input_tensor_digest`` (``models.py:166-172``)."""
    import torch

    inputs = dict(inputs)
    inputs.pop("video_metadata", None)
    tensors = {k: v for k, v in inputs.items() if torch.is_tensor(v)}
    meta = {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": hashlib.sha256(
        v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for k, v in tensors.items()}
    digest = hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return tensors, digest


def tempseg_answer(window) -> str:
    """A valid prediction for ``window``: the fixed global events, clipped to it."""
    off, lo_t, hi_t = window.global_tick_offset, window.valid_start_tick, window.valid_end_tick
    rows = []
    for base in range(0, 10_000, 10):
        g0 = (base + EVENTS_S[0][0]) * TICKS_PER_S
        if g0 - off > hi_t:
            break
        for a, b in EVENTS_S:
            s, e = round((base + a) * TICKS_PER_S) - off, round((base + b) * TICKS_PER_S) - off
            cs, ce = max(s, lo_t), min(e, hi_t)
            if cs >= ce or not (cs < window.core_end_tick and ce > window.core_start_tick):
                continue
            rows.append({"s": cs, "e": ce, "lo": s < lo_t, "ro": e > hi_t})
    rows.sort(key=lambda r: (r["s"], r["e"], r["lo"], r["ro"]))
    return json.dumps({"segments": rows})


def _canned_generate(model, proc, inputs, budget):
    log = _LOG.get()
    tensors, digest = nvidia_input_digest(inputs)
    log.append({**{k: (v.dtype, tuple(v.shape), v.numpy().tobytes())
                   for k, v in tensors.items()}, "digest": digest})
    if budget == 1024:
        window = _WINDOW.get()
        return tempseg_answer(window) if window is not None else '{"segments":[]}'
    return f"caption {len(log)}"


def _segments(videos):
    d = videos["main"].duration_s
    return [
        dict(start_s=0.0, end_s=min(1.5, d / 4), metadata={}),
        dict(start_s=d / 4, end_s=d / 2, metadata={}),
        dict(start_s=d / 2 - 0.5, end_s=d - 0.2, metadata={}),
    ]


E2E = {
    # id: (source, segment pool, caption kind, caption profile, instruction)
    "human-h264": ("h264_bframes_1v", "human", "single", "Human", ""),
    "human-vfr-instr": ("h264_vfr_1v", "human", "single", "Human", "Fold the towel."),
    "human-long-multiwindow": ("h264_long_1v", "human", "single", "Human", ""),
    "human-hevc-open-gop": ("hevc_open_gop_1v", "human", "single", "Human", ""),
    "robot-1v-av1": ("av1_1v", "yam", "single", "YAM", ""),
    "yam-3v-h264": ("h264_3v", "yam", "three", "YAM", ""),
    "dex-3v-av1": ("av1_3v", "dex", "three", "Dex", "Fold the towel."),
    "yam-3v-odd-bt709": ("h264_odd_bt709_3v", "yam", "three", "YAM", ""),
    "yam-3v-hevc": ("hevc_3v", "yam", "three", "YAM", ""),
}


def _e2e(run, videos, pool, kind, profile, instruction, procs=None):
    seg = run.segment(videos, pool, instruction, procs=procs)
    segs = _segments(videos)
    if kind == "single":
        cap = run.caption_single(videos, segs, profile, instruction, procs=procs)
    else:
        cap = run.caption_three(videos, segs, profile, instruction, procs=procs)
    return seg, cap


@pytest.mark.parametrize("case", sorted(E2E))
def test_pipeline_end_to_end_identical(nv, procs, sources, monkeypatch, case):
    source, pool, kind, profile, instruction = E2E[case]
    videos = sources[source]
    run = _Runner(nv, procs, monkeypatch)
    ref = _e2e(run, videos, pool, kind, profile, instruction)
    with apply_patches(nv, ALL_PATCHES) as active:
        with active.episode():
            got = _e2e(run, videos, pool, kind, profile, instruction)
            stats = active.stats()
    (ref_seg, ref_seg_t), (ref_cap, ref_cap_t) = ref
    (got_seg, got_seg_t), (got_cap, got_cap_t) = got
    assert ref_seg == got_seg  # segments + per-window audit (frame plans, prompt hashes)
    assert ref_cap == got_cap  # captions + audit (blocks, geometry, prompt hashes)
    assert len(ref_seg_t) >= 1 and len(ref_cap_t) >= 1
    assert ref_seg_t == got_seg_t  # every processor tensor, bit for bit
    assert ref_cap_t == got_cap_t
    assert any("mm_token_type_ids" in t for t in ref_seg_t + ref_cap_t)
    assert ref_seg[0], "the canned TempSeg answers must produce segments"
    if case == "human-long-multiwindow":
        assert len(ref_seg[1]) >= 2  # several windows, stitched
    assert stats["frames_for_calls"] >= 2
    assert stats.get("lookahead_requests", 0) >= 1  # windows / caption plans were announced
    assert stats.get("lookahead_errors", 0) == 0


@pytest.mark.parametrize("case", ["human-h264", "yam-3v-h264"])
def test_pipeline_end_to_end_identical_tiny_cache(nv, procs, sources, monkeypatch, case):
    """Same as above with a cache too small for any lookahead: every path recomputes."""
    source, pool, kind, profile, instruction = E2E[case]
    videos = sources[source]
    run = _Runner(nv, procs, monkeypatch)
    ref = _e2e(run, videos, pool, kind, profile, instruction)
    with apply_patches(nv, ALL_PATCHES, cache_bytes=100_000, workers=2):
        got = _e2e(run, videos, pool, kind, profile, instruction)
    assert ref == got


@pytest.mark.parametrize("config", [dict(), dict(workers=4, cache_bytes=150_000,
                                                 max_total_cache_bytes=200_000,
                                                 max_open_decoders=2)])
def test_overlapping_episode_threads_identical_and_bounded(nv, procs, sources, monkeypatch,
                                                           config):
    """W3-A: ALL_PATCHES with one shared state, episodes overlapping on threads (the same source
    in two threads at once, 1-view and 3-view). Every generate() input digest equals unpatched
    serial NVIDIA's; every episode's scope is released at its own exit; memory and open
    decoders stay bounded across episodes (review M2, p2 / p5)."""
    import copy
    from concurrent.futures import ThreadPoolExecutor

    jobs = [("A", "h264_bframes_1v", "human", "single", "Human"),
            ("B", "h264_bframes_1v", "human", "single", "Human"),
            ("C", "h264_long_1v", "human", "single", "Human"),
            ("D", "h264_3v", "yam", "three", "YAM"),
            ("E", "hevc_3v", "dex", "three", "Dex")]
    run = _Runner(nv, procs, monkeypatch)

    def job(spec, procs_for_job=None):
        _, source, pool, kind, profile = spec
        out = _e2e(run, sources[source], pool, kind, profile, "", procs=procs_for_job)
        (seg, seg_t), (cap, cap_t) = out
        return seg, cap, seg_t + cap_t

    ref = {spec[0]: job(spec) for spec in jobs}
    with apply_patches(nv, ALL_PATCHES, **config) as active:
        def episode(spec):
            clones = {k: copy.deepcopy(p) for k, p in procs.items()}  # W3-A's per-episode clones
            with active.episode():
                out = job(spec, clones)
                scope = active.state.scope()
            return spec[0], out, scope

        with ThreadPoolExecutor(len(jobs)) as pool:
            results = list(pool.map(episode, jobs * 2))  # each job twice, all concurrent
        stats = active.stats()
    for key, out, scope in results:
        assert out[0] == ref[key][0] and out[1] == ref[key][1], key
        assert out[2] == ref[key][2], key  # every tensor of every generate() call
        assert scope.closed and not scope.decoders and scope.nbytes() == 0
    assert stats["episodes"] == 2 * len(jobs) and stats["active_episodes"] == 0
    assert stats["open_decoders"] == 0 and stats["cache_bytes_total"] == 0
    assert stats["cache_bytes_total_peak"] <= stats["cache_bytes_total_limit"]
    assert stats["peak_active_episodes"] >= 2  # they did overlap
    if "max_open_decoders" in config:
        # idle decoders beyond the cap are closed; only busy ones (one per running request,
        # three cameras per 3-view request) can exceed it
        assert stats["peak_open_decoders"] <= config["max_open_decoders"] + 3 * 2 * len(jobs)
