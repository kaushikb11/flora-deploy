"""Gate C request builder: drive NVIDIA's own pipeline functions and record each processor call.

The same harness as ``tests/nvlink/test_nv_link_capture.py``, usable outside pytest (on Modal).
Each case runs an NVIDIA pipeline function (``pipeline.segment``, ``caption_three``,
``caption_single``) end to end with two substitutions: ``media.decode_selected`` returns frames
from a ``FrameSource`` (synthetic or a decoded real video), and ``pipeline.generate`` returns a
canned answer instead of running a model. It runs twice per case:

- with a ``SpyProcessor`` around the real processor: the HF reference (NVIDIA's own PIL-input
  call), whose ``input_ids`` / ``video_grid_thw`` / ``pixel_values_videos`` we keep;
- with ``CaptureProcessor``: the ``CapturedRequest`` objects that go to vLLM.

Golden episodes (``build_golden``): the same two runs on a real staged episode of the
``flora-golden`` Volume (``golden/v1/``), with NVIDIA's real ``media.probe`` and
``decode_selected`` (memoized so the second run reuses the first run's frames); only
``generate`` is canned, and the pipeline is stopped after the wanted number of calls.

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
    "temporal-yam-3view-instruction": ("segment", "yam", 3, "Fold the towel."),
    "temporal-tianji-3view": ("segment", "tianji", 3, ""),
    "caption-three-yam": ("caption_three", "YAM", 3, ""),
    "caption-three-dex": ("caption_three", "Dex", 3, "Fold the towel."),
    "caption-three-tianji": ("caption_three", "Tianji", 3, "Fold the towel."),
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
    fps: int | Fraction  # exact rate (e.g. 30000/1001); never truncated
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


def decoded_source(name: str, frames: np.ndarray, fps: int | Fraction,
                   duration_s=40.0) -> FrameSource:
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
    rate = Fraction(source.fps)
    n = int(Fraction(source.duration_s) * rate)
    return {
        role: nv.media.Video(
            path=Path(f"{role}.mp4"), sha256=f"{i:02d}" * 32, rate=rate,
            pts_us=tuple(Fraction(k * 1_000_000) / rate for k in range(n)),
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


class _Enough(BaseException):  # noqa: N818 - control flow, not an error
    """Raised from the fake ``generate`` once enough calls are recorded. It stops NVIDIA's
    pipeline early; a ``BaseException`` so no ``except Exception`` in NVIDIA's code swallows it."""


def run_pipeline(nv, path: str, pool_or_profile: str, videos: dict, segments: list[dict],
                 instruction: str, procs: dict, decode: Callable | None = None,
                 max_calls: int | None = None) -> int:
    """Run one NVIDIA pipeline function (``segment``, ``caption_three``, ``caption_single``)
    with the given processors and a canned ``generate``; return the number of generate calls.
    ``decode`` replaces ``media.decode_selected`` (None keeps NVIDIA's real decoder);
    ``max_calls`` stops the pipeline after that many generate calls."""
    pipeline = nv.pipeline
    generated: list[int] = []

    def fake_generate(model, proc, inputs, budget):
        generated.append(budget)
        if max_calls is not None and len(generated) >= max_calls:
            raise _Enough
        return '{"segments":[]}' if budget == 1024 else "pick up the cup"

    patch = _Patch()
    if decode is not None:
        patch.set(nv.media, "decode_selected", decode)
    patch.set(pipeline, "generate", fake_generate)
    try:
        session = CachedSession(lambda: _FakeNvSession(procs))
        progress = lambda *a, **k: None  # noqa: E731
        if path == "segment":
            pipeline.segment(videos, pool_or_profile, session, instruction, progress)
        elif path == "caption_three":
            pipeline.caption_three(videos, segments, session, instruction, progress,
                                   profile=pool_or_profile)
        elif path == "caption_single":
            pipeline.caption_single(videos, segments, pool_or_profile, session, instruction,
                                    progress)
        else:
            raise ValueError(f"unknown pipeline path {path!r}")
    except _Enough:
        pass
    finally:
        patch.undo()
    return len(generated)


def drive(nv, case: tuple[str, str, int, str], procs: dict, source: FrameSource,
          max_calls: int | None = None) -> int:
    """Run one ``CASES`` entry on a ``FrameSource``; return #generate calls."""

    def fake_decode(video, wanted, canonical=False):
        return {i: source.frame(video.path.stem, i) for i in wanted}

    path, pool_or_profile, n_views, instruction = case
    videos = make_videos(nv, ROLES3[:n_views], source)
    segments = [dict(start_s=2.0, end_s=5.0, metadata={}),
                dict(start_s=6.0, end_s=9.5, metadata={})]
    return run_pipeline(nv, path, pool_or_profile, videos, segments, instruction, procs,
                        decode=fake_decode, max_calls=max_calls)


@dataclass
class BuiltRequest:
    case_id: str
    index: int  # n-th processor call of the case (per source)
    kind: str  # tempseg | captioner
    source: str
    request: CapturedRequest
    hf: dict  # NVIDIA's own processor output: input_ids, video_grid_thw, pixel_values_videos


def _spy_and_capture(label: str, kind: str, procs: dict, run: Callable[[dict], int],
                     source: str, max_requests: int | None) -> list[BuiltRequest]:
    """Run NVIDIA's code twice (spy on the real processor, then ``CaptureProcessor``) and pair
    the calls, checking they are the same calls (prompt, frames, metadata)."""
    spy = SpyProcessor(procs[kind])
    n_ref = run({**procs, kind: spy})
    cap = CaptureProcessor(procs[kind])
    n_cap = run({**procs, kind: cap})
    if not (n_ref == n_cap == len(spy.calls) == len(cap.requests) >= 1):
        raise RuntimeError(f"{label}: {n_ref}/{n_cap} generate calls, {len(spy.calls)} HF "
                           f"calls, {len(cap.requests)} captured")
    out = []
    for i, (ref, req) in enumerate(zip(spy.calls, cap.requests, strict=True)):
        if req.prompt != ref["prompt"] or len(req.videos) != len(ref["frames"]):
            raise RuntimeError(f"{label}#{i}: captured prompt/videos differ from NVIDIA's call")
        for (frames, md), ref_frames, ref_md in zip(req.videos, ref["frames"], ref["metadata"],
                                                    strict=True):
            if not np.array_equal(frames, ref_frames) or md != metadata_to_dict(ref_md):
                raise RuntimeError(f"{label}#{i}: captured frames/metadata differ")
        out.append(BuiltRequest(label, i, kind, source, req, ref["output"]))
        if max_requests is not None and len(out) >= max_requests:
            break
    return out


def build_requests(nv, procs: dict, case_id: str, source: FrameSource,
                   max_requests: int | None = None) -> list[BuiltRequest]:
    """HF reference + captured requests for one case, checked to be the same calls."""
    kind = processor_kind(case_id)
    return _spy_and_capture(
        case_id, kind, procs, lambda p: drive(nv, CASES[case_id], p, source, max_requests),
        source.name, max_requests)


# ---------------------------------------------------------------------------------------------
# Golden episodes (Volume ``flora-golden``, ``golden/v1/``; docs/golden-manifest-v1.md)

# golden pool -> (segmenter pool, caption path, caption profile, camera views); the same mapping
# NVIDIA's ``pipeline.annotate`` applies (single-view robot episodes use profile YAM, yam32 is
# YAM with long windows).
GOLDEN_POOLS: dict[str, tuple[str, str, str, int]] = {
    "human": ("human", "caption_single", "Human", 1),
    "single_view_robot": ("yam", "caption_single", "YAM", 1),
    "yam": ("yam", "caption_three", "YAM", 3),
    "yam32": ("yam32", "caption_three", "YAM", 3),
    "dex": ("dex", "caption_three", "Dex", 3),
    "tianji": ("tianji", "caption_three", "Tianji", 3),
}


@dataclass(frozen=True)
class GoldenEpisode:
    id: str  # "<pool>/<dataset>/<slug>"
    pool: str
    dir: Path
    task: str  # dataset task text (used as the instruction on alternate episodes; never logged)


def load_golden_manifest(root: str | Path) -> dict:
    """``golden/v1/manifest.json`` under the Volume root ``root``."""
    import json

    from flora_deploy.validate.golden import GOLDEN_PREFIX, MANIFEST_NAME

    return json.loads((Path(root) / GOLDEN_PREFIX / MANIFEST_NAME).read_text())


def golden_episodes(root: str | Path, manifest: dict) -> dict[str, list[GoldenEpisode]]:
    """Staged, oracle-eligible episodes of every ``GOLDEN_POOLS`` pool, in manifest order.
    ``root`` is the Volume root (manifest paths start with ``golden/v1/``)."""
    out: dict[str, list[GoldenEpisode]] = {p: [] for p in GOLDEN_POOLS}
    for ep in manifest["episodes"]:
        if ep["pool"] in out and ep.get("status") == "staged" and ep.get("oracle_eligible"):
            out[ep["pool"]].append(GoldenEpisode(ep["id"], ep["pool"], Path(root) / ep["path"],
                                                 ep.get("task") or ""))
    return out


MAX_PER_EPISODE = 3  # build_golden's default: requests taken from one golden episode


def golden_capacity(manifest: dict, quota: int, max_per_episode: int = MAX_PER_EPISODE
                    ) -> dict[str, dict]:
    """Per ``GOLDEN_POOLS`` pool, whether ``quota`` requests can come from the golden
    selection (review 5 M-A): ``episodes`` (staged, oracle-eligible), ``max_requests`` (the
    most :func:`build_golden` can take: ``episodes × max_per_episode``) and ``satisfiable``.
    This is an upper bound: an episode can also yield fewer calls than ``max_per_episode``
    (tianji ``episode_000075`` has 2 TempSeg windows), which only the build shows."""
    counts = {p: 0 for p in GOLDEN_POOLS}
    for ep in manifest.get("episodes", []):
        if ep.get("pool") in counts and ep.get("status") == "staged" and ep.get("oracle_eligible"):
            counts[ep["pool"]] += 1
    return {p: {"episodes": n, "max_requests": n * max_per_episode,
                "satisfiable": n * max_per_episode >= quota}
            for p, n in counts.items()}


def golden_segments(duration_s: float) -> list[dict]:
    """Three caption targets spread over the episode (10%, 40%, 70% in; 1-4 s long)."""
    length = max(1.0, min(4.0, 0.2 * duration_s))
    return [dict(start_s=round(f * duration_s, 3),
                 end_s=round(min(f * duration_s + length, 0.98 * duration_s), 3), metadata={})
            for f in (0.1, 0.4, 0.7)]


class MemoDecode:
    """NVIDIA's real ``media.decode_selected``, memoized per (file, canonical, wanted) so the
    capture run reuses the spy run's frames instead of decoding again. Returns copies."""

    def __init__(self, real: Callable):
        self.real, self.cache = real, {}

    def __call__(self, video, wanted, canonical=False):
        key = (str(video.path), bool(canonical), frozenset(wanted))
        if key not in self.cache:
            self.cache[key] = self.real(video, wanted, canonical)
        return {i: im.copy() for i, im in self.cache[key].items()}


def probe_episode(nv, ep: GoldenEpisode) -> dict:
    """NVIDIA ``media.Video`` per role via NVIDIA's own ``media.probe`` (+ sync check)."""
    n_views = GOLDEN_POOLS[ep.pool][3]
    videos = {role: nv.media.probe(ep.dir / f"{role}.mp4") for role in ROLES3[:n_views]}
    if n_views == 3:
        nv.media.check_synchronized(videos)
    return videos


def build_golden_requests(nv, procs: dict, kind: str, ep: GoldenEpisode, max_requests: int,
                          use_instruction: bool, videos: dict | None = None) -> list[BuiltRequest]:
    """HF reference + captured requests for the first ``max_requests`` processor calls NVIDIA's
    pipeline makes on a real golden episode (real probe, real decode; only ``generate`` is
    canned). ``kind`` picks the segmenter (``tempseg``) or captioner path."""
    seg_pool, cap_path, cap_profile, _ = GOLDEN_POOLS[ep.pool]
    videos = videos if videos is not None else probe_episode(nv, ep)
    instruction = ep.task.strip() if use_instruction else ""
    if kind == "tempseg":
        path, pool_or_profile, segments = "segment", seg_pool, []
    else:
        path, pool_or_profile = cap_path, cap_profile
        segments = golden_segments(videos["main"].duration_s)
    decode = MemoDecode(nv.media.decode_selected)
    return _spy_and_capture(
        f"golden-{ep.pool}", kind, procs,
        lambda p: run_pipeline(nv, path, pool_or_profile, videos, segments, instruction, p,
                               decode=decode, max_calls=max_requests),
        ep.id, max_requests)


# ---------------------------------------------------------------------------------------------
# Compact records (what crosses the builder-process boundary on Modal)


@dataclass
class RequestRecord:
    case_id: str
    index: int
    kind: str
    source: str
    request: CapturedRequest
    hf_ids: list[int]
    hf_grids: list[list[int]]
    hf_pixels: list[dict]  # per video: rows + sha256 in fp32 and bf16 (validate.gatec)


def compact(b: BuiltRequest) -> RequestRecord:
    from flora_deploy.validate.gatec import video_pixel_digests

    grids = b.hf["video_grid_thw"].tolist()
    return RequestRecord(b.case_id, b.index, b.kind, b.source, b.request,
                         b.hf["input_ids"][0].tolist(), grids,
                         video_pixel_digests(b.hf["pixel_values_videos"], grids))


def build_cases(nv, procs: dict, kind: str, sources: list[FrameSource],
                max_requests: int) -> list[RequestRecord]:
    out = []
    for source in sources:
        for case_id in CASES:
            if processor_kind(case_id) == kind:
                out += [compact(b) for b in
                        build_requests(nv, procs, case_id, source, max_requests)]
    return out


def build_golden(nv, procs: dict, kind: str, episodes: dict[str, list[GoldenEpisode]],
                 quota: int, max_per_episode: int = MAX_PER_EPISODE
                 ) -> tuple[list[RequestRecord], list[dict]]:
    """Up to ``quota`` requests per pool, spread over the pool's episodes (at most
    ``ceil(quota / #episodes)`` and ``max_per_episode`` each). Every other episode carries its
    dataset task as the instruction. Errors are recorded by class name only."""
    records, errors = [], []
    for pool, eps in episodes.items():
        if not eps:
            continue
        per_ep = min(max_per_episode, max(1, -(-quota // len(eps))))
        remaining = quota
        for n, ep in enumerate(eps):
            if remaining <= 0:
                break
            try:
                built = build_golden_requests(nv, procs, kind, ep, min(per_ep, remaining),
                                              use_instruction=n % 2 == 0)
            except Exception as exc:  # recorded, not raised: the gate reports coverage
                errors.append(dict(pool=pool, episode=ep.id, error=type(exc).__name__))
                continue
            records += [compact(b) for b in built]
            remaining -= len(built)
    return records, errors


# ---------------------------------------------------------------------------------------------
# The whole-run verdict, as ``scripts/gatec.py`` scores it (also used to re-score a recorded run)

KINDS = ("tempseg", "captioner")
# Where build_part (``modal_app/gatec_app.py``) runs NVIDIA's HF reference processors: the
# vLLM image, not production's img_hfref, so C2 needs a cross-check (review 3 C2).
BUILD_REFERENCE_IMAGE = "img_vllm"


def run_parts(golden: bool = True) -> list[str]:
    return ["cases:synthetic", "cases:sample",
            *([f"golden:{pool}" for pool in GOLDEN_POOLS] if golden else [])]


def run_verdict(builds: list[dict], results: dict[str, dict], *, golden: bool, quota: int,
                capacity: dict[str, dict] | None,
                reference: dict | None = None) -> dict:
    """``validate.gatec.gate_verdict`` with everything the gate requires (review 5 M-A): every
    part of the run, every case, the golden ``quota``, the golden selection's ``capacity``
    (``golden_capacity``) and the HF ``reference`` (default: where build_part ran it,
    ``BUILD_REFERENCE_IMAGE``, with no cross-check: a FAIL until an img_hfref cross-check
    exists)."""
    from flora_deploy.validate.gatec import gate_verdict

    parts = run_parts(golden)
    return gate_verdict(
        builds, results, expected_parts=[(k, p) for k in KINDS for p in parts],
        expected_cases=[*CASES, *([f"golden-{pool}" for pool in GOLDEN_POOLS] if golden else [])],
        quota=quota, capacity=capacity,
        reference=reference if reference is not None else {"image": BUILD_REFERENCE_IMAGE})
