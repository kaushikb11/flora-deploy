"""Gate C request builder: drive NVIDIA's own pipeline functions and record each processor call.

The same harness as ``tests/nvlink/test_nv_link_capture.py``, usable outside pytest (on Modal).
Each case runs an NVIDIA pipeline function (``pipeline.segment``, ``caption_three``,
``caption_single``) end to end with two substitutions: ``media.decode_selected`` returns frames
from a ``FrameSource`` (synthetic or a decoded real video), and ``pipeline.generate`` returns a
canned answer instead of running a model. It runs twice per case:

- with a ``SpyProcessor`` around the real processor: the HF reference (NVIDIA's own PIL-input
  call), whose ``input_ids`` / ``video_grid_thw`` / ``pixel_values_videos`` we keep;
- with ``CaptureProcessor``: the ``CapturedRequest`` objects that go to vLLM.

All NVIDIA objects are reached through the ``NvFamily`` handle from ``nv.linker.load_family``;
this module imports no NVIDIA code itself.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from flora_deploy.nv.capture import CapturedRequest, CaptureProcessor, metadata_to_dict
from flora_deploy.nv.session import CachedSession

ROLES3 = ("main", "left_wrist", "right_wrist")
HF_KEYS = ("input_ids", "video_grid_thw", "pixel_values_videos")

# id: (NVIDIA path, pool/profile, views, instruction suffix)
CASES: dict[str, tuple[str, str, int, str]] = {
    "temporal-human": ("segment", "human", 1, ""),
    "temporal-human-instruction": ("segment", "human", 1, "Fold the towel."),
    "temporal-yam-1view": ("segment", "yam", 1, ""),
    "temporal-yam-3view": ("segment", "yam", 3, ""),
    "temporal-dex-3view": ("segment", "dex", 3, ""),
    "temporal-yam32-3view": ("segment", "yam32", 3, ""),
    "caption-three-yam": ("caption_three", "YAM", 3, ""),
    "caption-three-dex": ("caption_three", "Dex", 3, "Fold the towel."),
    "caption-single-human": ("caption_single", "Human", 1, ""),
    "caption-single-human-instruction": ("caption_single", "Human", 1, "Fold the towel."),
    "caption-single-robot": ("caption_single", "YAM", 1, ""),
}


def processor_kind(case_id: str) -> str:
    """``tempseg`` for segmenter cases, ``captioner`` for caption cases."""
    return "tempseg" if CASES[case_id][0] == "segment" else "captioner"


@dataclass(frozen=True)
class FrameSource:
    """Frames for the fake ``decode_selected``: ``frame(role, index) -> PIL RGB image``."""

    name: str
    size: tuple[int, int]  # (width, height)
    fps: int
    duration_s: float
    frame: Callable[[str, int], Any]


def synthetic_frame(role: str, index: int, size: tuple[int, int] = (640, 480)):
    """Deterministic RGB frame: a moving gradient plus seeded noise (sharp detail on purpose)."""
    from PIL import Image

    w, h = size
    seed = (ROLES3.index(role) if role in ROLES3 else 7) * 100_003 + index
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x + 3 * index) % 256, (y + 5 * index) % 256, (x + y) % 256], axis=-1)
    noise = rng.integers(-40, 41, size=(h, w, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


def synthetic_source(size=(640, 480), fps=30, duration_s=40.0) -> FrameSource:
    return FrameSource("synthetic", size, fps, duration_s,
                       lambda role, i: synthetic_frame(role, i, size))


def decoded_source(name: str, frames: np.ndarray, fps: int, duration_s=40.0) -> FrameSource:
    """A real video's decoded frames (uint8 THWC). Index ``i`` maps to frame ``i % T``; the
    three camera roles get the frame as is, mirrored, and flipped, so they differ."""
    from PIL import Image

    t, h, w, _ = frames.shape

    def frame(role: str, i: int):
        arr = frames[i % t]
        r = ROLES3.index(role) if role in ROLES3 else 0
        if r == 1:
            arr = arr[:, ::-1]
        elif r == 2:
            arr = arr[::-1]
        return Image.fromarray(np.ascontiguousarray(arr), "RGB")

    return FrameSource(name, (w, h), fps, duration_s, frame)


def decode_all(path: str | Path) -> tuple[np.ndarray, Fraction]:
    """Decode every frame of a small video to uint8 RGB (THWC) with PyAV."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    return np.stack(frames), rate


def make_videos(nv, roles, source: FrameSource):
    n = int(source.duration_s * source.fps)
    return {
        role: nv.media.Video(
            path=Path(f"{role}.mp4"), sha256=f"{i:02d}" * 32, rate=Fraction(source.fps),
            pts_us=tuple(Fraction(k * 1_000_000, source.fps) for k in range(n)),
            duration_us=int(source.duration_s * 1_000_000), size=source.size, cfr=True,
        )
        for i, role in enumerate(roles)
    }


class SpyProcessor:
    """Records exactly what NVIDIA passes to the real processor, and what it returns."""

    def __init__(self, real):
        self._real = real
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __call__(self, **kwargs):
        out = self._real(**kwargs)
        self.calls.append(dict(
            prompt=kwargs["text"][0],
            frames=[np.stack([np.asarray(im) for im in video]) for video in kwargs["videos"]],
            metadata=copy.deepcopy(list(kwargs["video_metadata"])),
            output={k: out[k].clone() for k in HF_KEYS if k in out},
        ))
        return out


class _FakeNvSession:
    """Stands in for NVIDIA ``models.Session`` (no weights): returns the given processor."""

    def __init__(self, procs):
        self.procs, self.audit = procs, {}

    def load(self, kind):
        return object(), self.procs[kind]

    def close(self):
        pass


class _Patch:
    """Minimal attribute monkeypatch with guaranteed restore (no pytest dependency)."""

    def __init__(self):
        self._saved: list[tuple[Any, str, Any]] = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        while self._saved:
            obj, name, value = self._saved.pop()
            setattr(obj, name, value)


def drive(nv, case: tuple[str, str, int, str], procs: dict, source: FrameSource) -> int:
    """Run one NVIDIA pipeline function with the given processors; return #generate calls."""
    pipeline = nv.pipeline
    generated: list[int] = []

    def fake_decode(video, wanted, canonical=False):
        return {i: source.frame(video.path.stem, i) for i in wanted}

    def fake_generate(model, proc, inputs, budget):
        generated.append(budget)
        return '{"segments":[]}' if budget == 1024 else "pick up the cup"

    patch = _Patch()
    patch.set(nv.media, "decode_selected", fake_decode)
    patch.set(pipeline, "generate", fake_generate)
    try:
        session = CachedSession(lambda: _FakeNvSession(procs))
        progress = lambda *a, **k: None  # noqa: E731
        kind, pool_or_profile, n_views, instruction = case
        videos = make_videos(nv, ROLES3[:n_views], source)
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
    finally:
        patch.undo()
    return len(generated)


@dataclass
class BuiltRequest:
    case_id: str
    index: int  # n-th processor call of the case
    kind: str  # tempseg | captioner
    source: str
    request: CapturedRequest
    hf: dict  # NVIDIA's own processor output: input_ids, video_grid_thw, pixel_values_videos


def build_requests(nv, procs: dict, case_id: str, source: FrameSource,
                   max_requests: int | None = None) -> list[BuiltRequest]:
    """HF reference + captured requests for one case, checked to be the same calls."""
    case = CASES[case_id]
    kind = processor_kind(case_id)
    spy = SpyProcessor(procs[kind])
    n_ref = drive(nv, case, {**procs, kind: spy}, source)
    cap = CaptureProcessor(procs[kind])
    n_cap = drive(nv, case, {**procs, kind: cap}, source)
    if not (n_ref == n_cap == len(spy.calls) == len(cap.requests) >= 1):
        raise RuntimeError(f"{case_id}: {n_ref}/{n_cap} generate calls, {len(spy.calls)} HF "
                           f"calls, {len(cap.requests)} captured")
    out = []
    for i, (ref, req) in enumerate(zip(spy.calls, cap.requests, strict=True)):
        if req.prompt != ref["prompt"] or len(req.videos) != len(ref["frames"]):
            raise RuntimeError(f"{case_id}#{i}: captured prompt/videos differ from NVIDIA's call")
        for (frames, md), ref_frames, ref_md in zip(req.videos, ref["frames"], ref["metadata"],
                                                    strict=True):
            if not np.array_equal(frames, ref_frames) or md != metadata_to_dict(ref_md):
                raise RuntimeError(f"{case_id}#{i}: captured frames/metadata differ")
        out.append(BuiltRequest(case_id, i, kind, source.name, req, ref["output"]))
        if max_requests is not None and len(out) >= max_requests:
            break
    return out
