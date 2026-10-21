"""CaptureProcessor parity through NVIDIA's own input builders and the real Flora processors.

Each case drives an NVIDIA pipeline function (``pipeline.segment``, ``caption_three``,
``caption_single``) end to end, with only two substitutions: ``media.decode_selected`` returns
synthetic frames (no video files), and ``models.generate`` records its inputs instead of running a
model. The processors are built by NVIDIA's ``models.processor(kind, path)`` from the fetched
tokenizer/processor/chat-template files (no weights).

Every case runs twice: once with a spy around the real processor (the HF reference) and once with
``CaptureProcessor``. We then check that

1. the captured prompt, frames and metadata are exactly what NVIDIA passed to the real processor;
2. replaying the captured request through the real processor, with **uint8 ndarray** frames as
   vLLM would receive them, reproduces ``input_ids``, ``attention_mask``, ``video_grid_thw`` and
   ``pixel_values_videos`` of NVIDIA's original PIL-input call bit for bit.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flora_deploy.nv import Family
from flora_deploy.nv.capture import CaptureProcessor, metadata_to_dict, replay
from flora_deploy.nv.linker import load_family
from flora_deploy.nv.session import CachedSession

pytestmark = pytest.mark.nvlink

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

FPS = 30
DURATION_S = 20.0
ROLES3 = ("main", "left_wrist", "right_wrist")
TENSOR_KEYS = ("input_ids", "attention_mask", "video_grid_thw", "pixel_values_videos")


@pytest.fixture(scope="module")
def nv_dirs():
    # module-scoped mirror of the nv_code fixture (which is function-scoped)
    import os

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[k]) for k in ("tempseg_code", "captioner_code")}
    for key, path in dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    return dirs


@pytest.fixture(scope="module")
def nv(nv_dirs):
    return load_family(Family.SEGCAP, nv_dirs["tempseg_code"])


@pytest.fixture(scope="module")
def procs(nv, nv_dirs):
    return {
        "tempseg": nv.models.processor("tempseg", nv_dirs["tempseg_code"]),
        "captioner": nv.models.processor("captioner", nv_dirs["captioner_code"]),
    }


def synthetic_frame(role: str, index: int, size=(640, 480)):
    """Deterministic RGB frame: a moving gradient plus seeded noise (sharp detail on purpose,
    so any resampling difference between input types would show)."""
    from PIL import Image

    w, h = size
    seed = (ROLES3.index(role) if role in ROLES3 else 7) * 100_003 + index
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x + 3 * index) % 256, (y + 5 * index) % 256, (x + y) % 256], axis=-1)
    noise = rng.integers(-40, 41, size=(h, w, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


def make_videos(nv, roles):
    n = int(DURATION_S * FPS)
    return {
        role: nv.media.Video(
            path=Path(f"{role}.mp4"), sha256=f"{i:02d}" * 32, rate=Fraction(FPS),
            pts_us=tuple(Fraction(k * 1_000_000, FPS) for k in range(n)),
            duration_us=int(DURATION_S * 1_000_000), size=(640, 480), cfr=True,
        )
        for i, role in enumerate(roles)
    }


class SpyProcessor:
    """Records exactly what NVIDIA passes to the real processor, then calls it."""

    def __init__(self, real):
        self._real = real
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __call__(self, **kwargs):
        frames = [np.stack([np.asarray(im) for im in video]) for video in kwargs["videos"]]
        modes = {im.mode for video in kwargs["videos"] for im in video}
        record = dict(
            kwargs={k: v for k, v in kwargs.items() if k not in ("videos", "video_metadata")},
            frames=frames, modes=modes,
            metadata=copy.deepcopy(list(kwargs["video_metadata"])),
        )
        out = self._real(**kwargs)
        record["output"] = {k: out[k].clone() for k in TENSOR_KEYS if k in out}
        self.calls.append(record)
        return out


class FakeNvSession:
    """Stands in for NVIDIA ``models.Session`` (no weights): returns the given processor."""

    def __init__(self, procs):
        self.procs, self.audit = procs, {}

    def load(self, kind):
        return object(), self.procs[kind]

    def close(self):
        pass


def drive(nv, monkeypatch, case, procs):
    """Run one NVIDIA pipeline function with the given processors; return #generate calls."""
    pipeline = nv.pipeline

    def fake_decode(video, wanted, canonical=False):
        return {i: synthetic_frame(video.path.stem, i) for i in wanted}

    generated = []

    def fake_generate(model, proc, inputs, budget):
        generated.append(budget)
        return '{"segments":[]}' if budget == 1024 else "pick up the cup"

    monkeypatch.setattr(nv.media, "decode_selected", fake_decode)
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    session = CachedSession(lambda: FakeNvSession(procs))
    progress = lambda *a, **k: None  # noqa: E731
    kind, pool_or_profile, n_views, instruction = case
    videos = make_videos(nv, ROLES3[:n_views])
    segments = [dict(start_s=2.0, end_s=5.0, metadata={}),
                dict(start_s=6.0, end_s=9.5, metadata={})]
    if kind == "segment":
        pipeline.segment(videos, pool_or_profile, session, instruction, progress)
    elif kind == "caption_three":
        pipeline.caption_three(videos, segments, session, instruction, progress,
                               profile=pool_or_profile)
    else:
        pipeline.caption_single(videos, segments, pool_or_profile, session, instruction,
                                progress)
    monkeypatch.undo()
    return len(generated)


CASES = {
    # id: (path, pool/profile, views, instruction)
    "temporal-human": ("segment", "human", 1, ""),
    "temporal-human-instruction": ("segment", "human", 1, "Fold the towel."),
    "temporal-yam-1view": ("segment", "yam", 1, ""),
    "temporal-yam-3view": ("segment", "yam", 3, ""),
    "temporal-dex-3view": ("segment", "dex", 3, ""),
    "caption-three-yam": ("caption_three", "YAM", 3, ""),
    "caption-three-dex": ("caption_three", "Dex", 3, "Fold the towel."),
    "caption-single-human": ("caption_single", "Human", 1, ""),
    "caption-single-human-instruction": ("caption_single", "Human", 1, "Fold the towel."),
    "caption-single-robot": ("caption_single", "YAM", 1, ""),
}


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_capture_parity(nv, procs, monkeypatch, case_id):
    case = CASES[case_id]
    spies = {k: SpyProcessor(p) for k, p in procs.items()}
    n_ref = drive(nv, monkeypatch, case, spies)
    reference = spies["tempseg"].calls + spies["captioner"].calls
    caps = {k: CaptureProcessor(p) for k, p in procs.items()}
    n_cap = drive(nv, monkeypatch, case, caps)
    captured = caps["tempseg"].requests + caps["captioner"].requests
    kind = "tempseg" if case[0] == "segment" else "captioner"

    assert n_ref == n_cap == len(reference) == len(captured) >= 2
    for ref, req in zip(reference, captured, strict=True):
        # 1. exactly what NVIDIA handed to the real processor
        assert ref["modes"] == {"RGB"}
        assert req.prompt == ref["kwargs"]["text"][0]
        assert len(req.videos) == len(ref["frames"])
        for (frames, md), ref_frames, ref_md in zip(req.videos, ref["frames"], ref["metadata"],
                                                    strict=True):
            assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3
            assert np.array_equal(frames, ref_frames)
            assert md == metadata_to_dict(ref_md)
            assert md["do_sample_frames"] is False
            assert md["frames_indices"] == list(ref_md.frames_indices)
            assert md["fps"] == ref_md.fps and md["total_num_frames"] == ref_md.total_num_frames
        assert req.call_kwargs == {k: v for k, v in ref["kwargs"].items()
                                   if k not in ("text", "do_sample_frames")}
        # 2. ndarray replay == NVIDIA's PIL call, bit for bit
        out = replay(procs[kind], req)
        # every case's frames are off the 32-px grid, so the HF processor really resizes them
        patch = procs[kind].video_processor.patch_size
        for (frames, _), thw in zip(req.videos, out["video_grid_thw"].tolist(), strict=True):
            assert frames.shape[1:3] != (thw[1] * patch, thw[2] * patch)
        for key in TENSOR_KEYS:
            assert key in ref["output"], key
            a, b = ref["output"][key], out[key]
            assert a.dtype == b.dtype and a.shape == b.shape, key
            assert torch.equal(a, b), f"{key}: max |diff| {(a.float() - b.float()).abs().max()}"


def test_video_metadata_pop_is_the_only_access(nv, procs, monkeypatch):
    """``temporal_inputs`` pops ``video_metadata`` from the captured output and returns it; the
    object refuses anything else (what ``models.generate`` would do with it)."""
    from flora_deploy.nv.capture import CapturedInputsAccessError

    cap = CaptureProcessor(procs["tempseg"])
    seen = []

    def fake_generate(model, proc, inputs, budget):
        seen.append(inputs)
        with pytest.raises(CapturedInputsAccessError):
            dict(inputs)
        with pytest.raises(CapturedInputsAccessError):
            inputs["input_ids"]
        return '{"segments":[]}'

    def fake_decode(video, wanted, canonical=False):
        return {i: synthetic_frame(video.path.stem, i) for i in wanted}

    monkeypatch.setattr(nv.media, "decode_selected", fake_decode)
    monkeypatch.setattr(nv.pipeline, "generate", fake_generate)
    session = CachedSession(lambda: FakeNvSession({"tempseg": cap}))
    nv.pipeline.segment(make_videos(nv, ("main",)), "human", session, "", lambda *a, **k: None)
    assert len(seen) == len(cap.requests) == 2
