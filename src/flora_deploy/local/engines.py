"""Deterministic fake model engines for ``flora run --local --fake-models``.

No NVIDIA code, no weights, no GPU. Each engine plugs into the *real* worker runner at the
runner's own model boundary, so everything around the models (index receipts → inputs, video
resolution and cuts, part sinks, attempt markers, receipts, certification, provenance stamping,
error isolation, NVIDIA's request grouping) is the production code:

* **segcap** → :class:`FakeSegcapEngine` implements ``workers_impl.segcap.SegcapEngine`` (the
  boundary ``FastExactEngine`` fills on Modal with NVIDIA's ``annotate_batch``). The runner hands
  it the resolved inputs and turns its result dicts into records with the real
  ``stages.reference.stage_records``.
* **reward.terminal / reward.dense** → :func:`fake_rewarder_family` + :class:`FakeLLM` replace
  NVIDIA's Rewarder family (``nv.core`` / ``nv.engine`` / ``nv.dense_reward``) and the vLLM
  engine under the real ``stages.reward.run``: request building, NVIDIA-aligned rounds, parsing,
  record and audit assembly are the production code; only collage decoding, prompt rendering,
  generation and NVIDIA's parsers/stitcher are fakes. ``stages.reward.sampling_params`` imports
  ``vllm``; when it is not installed, :func:`vllm_shim` puts a two-class stand-in into
  ``sys.modules`` for the duration of the run (it is removed afterwards).
* **reward.rubric** → the same fake family adds NVIDIA's rubric interface (``core.rubric_*``,
  ``split_thinking`` / ``parse_rubric`` / ``medoid``, a ``rubric_reward`` module with
  ``base_parser`` / ``main`` / ``video_inputs``) under the real ``stages.rubric.run``; the
  fake LLM answers ``n`` samples per request, some of them unparseable.

Outputs are pure functions of the decoded video frames (sha256 over the decoded RGB arrays) plus
the instruction/task, so a re-run, a preempted-and-restarted shard or a bisection child yields
byte-identical records (compose sees duplicates, never conflicts).

Every record says it is fake: ``provenance.models`` = ``{"<role>": "fake"}``,
``provenance.certificate`` = :data:`FAKE_CERTIFICATE`, segcap ``provenance.engine`` =
:data:`FAKE_ENGINE`; and the job's config hash binds ``derived_weights = fake`` (``local.runner``),
so fake outputs can never mix with real ones in one store.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import threading
import types
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from flora_deploy.local.faults import FakeNvidiaInputError, checkpoint, note_episodes, nv_rejects
from flora_deploy.nv.errors import NvidiaValueError

FAKE_ENGINE = "flora-local-fake"
FAKE_CERTIFICATE = "local-fake-models"
FAKE_MODELS = {"segcap": {"tempseg": "fake", "captioner": "fake"},
               "reward": {"rewarder": "fake"}}

_VERBS = ("reach for", "grasp", "lift", "move", "place", "push", "rotate", "release", "align",
          "pour", "open", "close")
_OBJECTS = ("the cup", "the block", "the towel", "the lid", "the bottle", "the drawer",
            "the sponge", "the bowl", "the spoon", "the box")


# ---------------------------------------------------------------------------------------------
# Video decoding (shared, cached)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decoded:
    frames: np.ndarray  # [T, H, W, 3] uint8
    fps: float
    digest: str  # sha256 over the frames

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0


_CACHE: OrderedDict[tuple[str, int, int], Decoded] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 64
# Decoded RGB is large (an 8 s 640x480 view is 220 MB), so the cache is bounded by bytes too:
# by count alone, a real dataset's episodes filled many GB and the OS killed the run.
_CACHE_MAX_BYTES = 512 << 20


def decode(path: str) -> Decoded:
    """Every frame of the first video stream (PyAV), cached by (path, size, mtime)."""
    import os

    from flora_deploy.media.avsafe import open_input

    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    with open_input(path) as c:
        s = c.streams.video[0]
        rate = s.average_rate or s.guessed_rate or 30
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    if not frames:
        raise FakeNvidiaInputError(f"{path}: no decodable frames")
    arr = np.stack(frames)
    h = hashlib.sha256(f"{arr.shape}".encode())
    h.update(arr.tobytes())
    out = Decoded(arr, float(rate), h.hexdigest())
    with _CACHE_LOCK:
        _CACHE[key] = out
        while len(_CACHE) > 1 and (len(_CACHE) > _CACHE_MAX or sum(
                d.frames.nbytes for d in _CACHE.values()) > _CACHE_MAX_BYTES):
            _CACHE.popitem(last=False)
    return out


def _h(*parts: Any) -> int:
    return int(hashlib.sha256("\0".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------------------------
# segcap
# ---------------------------------------------------------------------------------------------


def fake_segments(digest: str, duration_s: float, instruction: str) -> list[dict]:
    """1-4 contiguous segments over the episode, captions from small word lists."""
    duration_s = max(duration_s, 0.1)
    n = 1 + _h(digest, "n") % 4
    cuts = sorted({round(duration_s * (0.15 + 0.7 * (_h(digest, "cut", i) % 1000) / 1000), 3)
                   for i in range(n - 1)})
    bounds = [0.0, *[c for c in cuts if 0.0 < c < duration_s], round(duration_s, 3)]
    out = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False)):
        verb = _VERBS[_h(digest, "verb", i) % len(_VERBS)]
        obj = _OBJECTS[_h(digest, "obj", i) % len(_OBJECTS)]
        caption = f"{verb} {obj}"
        if instruction:
            caption += f" (task: {instruction})"
        out.append({"start_s": a, "end_s": b, "status": "complete", "caption": caption,
                    "segmentation_provenance": "fake"})
    return out


@dataclass
class FakeSegcapEngine:
    """``workers_impl.segcap.SegcapEngine`` without NVIDIA: per episode, decode its videos, hash
    the frames and derive segments and captions from the hash (:func:`fake_segments`). Episodes
    run one at a time in the calling thread (so fault checkpoints and ``on_done`` follow the
    order of ``items``)."""

    concurrency: int = 1
    calls: int = field(default=0, init=False)
    on_episode: Any = None  # called with each episode's uid before it runs (progress)

    def label(self) -> str:
        return FAKE_ENGINE

    def runtime_provenance(self) -> dict:
        return {"engine": FAKE_ENGINE, "fake_models": True}

    def check(self) -> dict | None:
        return None

    def result_for(self, item: dict) -> dict:
        decoded = [decode(p) for _role, p in sorted(item["videos"].items())]
        digest = hashlib.sha256("".join(d.digest for d in decoded).encode()
                                + item.get("instruction", "").encode()
                                + str(item.get("profile")).encode()).hexdigest()
        duration = min(d.duration_s for d in decoded)
        return {"episode": item["episode"], "profile": item.get("profile"),
                "segments": fake_segments(digest, duration, item.get("instruction", "")),
                "flora_reference": {"model_calls": [], "engine": FAKE_ENGINE},
                "fake": {"digest": digest, "duration_s": round(duration, 6),
                         "frames": [d.num_frames for d in decoded]}}

    def annotate(self, items: list[dict], *, on_done) -> dict:
        self.calls += 1
        done = failed = 0
        note_episodes(len(items))
        for item in items:
            uid = item["episode"]
            if self.on_episode is not None:
                self.on_episode(uid)
            checkpoint(uid)
            try:
                if nv_rejects(uid):
                    raise FakeNvidiaInputError(f"fake model rejected episode {uid}")
                result = self.result_for(item)
            except Exception as exc:  # noqa: BLE001 - the runner turns it into EpisodeErrors
                failed += 1
                on_done(item, None, exc)
                continue
            done += 1
            on_done(item, result, None)
        return {"batch": {"engine": FAKE_ENGINE, "episodes": len(items), "done": done,
                          "failed": failed}}


# ---------------------------------------------------------------------------------------------
# Rewarder (fake NVIDIA family + fake vLLM)
# ---------------------------------------------------------------------------------------------


class EpisodeRegistry:
    """video path tuple -> uid, filled by the runner's episode hook (the collage only gets
    paths). Thread-safe; one per run."""

    def __init__(self):
        self._by_paths: dict[tuple[str, ...], str] = {}
        self._lock = threading.Lock()

    def add(self, videos: Sequence[str], uid: str) -> None:
        with self._lock:
            self._by_paths[tuple(videos)] = uid

    def uid(self, videos: Sequence[str]) -> str | None:
        with self._lock:
            return self._by_paths.get(tuple(videos))


class FakeCollage:
    """NVIDIA's ``Collage`` stand-in: the views decoded and placed side by side."""

    def __init__(self, paths: list[str], registry: EpisodeRegistry):
        uid = registry.uid(paths) or "|".join(paths)
        checkpoint(uid)
        if nv_rejects(uid):
            raise FakeNvidiaInputError(f"{paths[0]}: fake decoder rejected the video")
        views = [decode(p) for p in paths]
        n = min(v.num_frames for v in views)
        h = min(v.frames.shape[1] for v in views)
        self._frames = np.concatenate([v.frames[:n, :h] for v in views], axis=2)
        self.paths = list(paths)
        self.fps = views[0].fps
        self.num_frames = n
        self.duration_s = n / self.fps
        self.height, self.width = int(self._frames.shape[1]), int(self._frames.shape[2])

    def frames(self, indices) -> np.ndarray:
        return self._frames[list(indices)]

    def release(self) -> None:
        self._frames = self._frames[:0]


def _digest_frames(frames: Any) -> str:
    arr = np.asarray(frames)
    return hashlib.sha256(f"{arr.shape}".encode() + arr.tobytes()).hexdigest()


def fake_rewarder_family(registry: EpisodeRegistry | None = None) -> SimpleNamespace:
    """``nv`` for ``stages.reward.run``: ``core``, ``engine`` and ``dense_reward`` with NVIDIA's
    call signatures and deterministic, input-derived behaviour. ``registry`` maps an episode's
    video paths to its uid (fault targeting); the runner's episode hook fills it."""
    registry = registry or EpisodeRegistry()
    core = SimpleNamespace()
    core.registry = registry
    core.Collage = lambda paths: FakeCollage(paths, registry)
    core.TERMINAL_TOP_LOGPROBS = 5
    core.DENSE_MAX_TOKENS = 64
    core.DENSE_TAIL_TRIM_S = 0.0
    core.DENSE_FPS = 4.0

    def terminal_frame_indices(n: int, fps: float) -> list[int]:
        k = min(8, n)
        return sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})

    def dense_windows(n: int, fps: float) -> list[SimpleNamespace]:
        step = max(int(round(2.0 * fps)), 1)  # 2 s clips
        out = []
        for i, start in enumerate(range(0, n, step)):
            idx = tuple(range(start, min(start + step, n), max(step // 4, 1)))
            out.append(SimpleNamespace(index=i, start=start, end=min(start + step, n),
                                       frame_indices=idx))
        return out

    core.terminal_frame_indices = terminal_frame_indices
    core.dense_windows = dense_windows
    core.video_metadata = lambda idx, fps, n: {"frames_indices": list(idx), "fps": fps,
                                               "total_num_frames": n}
    core.terminal_messages = lambda task, r: [{"role": "user", "kind": "terminal",
                                               "task": task, "rationale": bool(r)}]
    core.dense_messages = lambda task, w: [{"role": "user", "kind": "dense", "task": task,
                                            "clip": w.index}]
    core.terminal_regex = lambda r: r"Score: [1-5](\nReason: .*)?" if r else r"Score: [1-5]"
    core.terminal_max_tokens = lambda r: 64 if r else 8
    core.dense_regex = lambda: r"Label: (normal|suboptimal|failure)"

    def parse_terminal(text: str, rationale: bool):
        head, _, reason = text.partition("\nReason: ")
        if not head.startswith("Score: ") or not head[-1:].isdigit():
            raise NvidiaValueError(f"unexpected terminal answer: {text!r}")
        return int(head[-1]), (reason or None) if rationale else None

    def score_probabilities(positions, score: int):
        # NVIDIA renormalizes the top-k score tokens; ours come from the fake logprobs
        top = positions[0][2] if positions else []
        raw = {str(k): 1e-6 for k in range(1, 6)}
        for tok, lp in top:
            if tok in raw:
                raw[tok] += float(np.exp(lp))
        z = sum(raw.values())
        probs = {k: v / z for k, v in raw.items()}
        return probs, sum(int(k) * v for k, v in probs.items())

    core.parse_terminal = parse_terminal
    core.score_probabilities = score_probabilities
    core.parse_dense = lambda text: SimpleNamespace(
        label=text.removeprefix("Label: ").strip() if text.startswith("Label: ") else "normal",
        valid=text.startswith("Label: "))

    labels = {"failure": 1, "suboptimal": 2, "normal": 3}

    def summarize(job, collage, windows, answers, texts, trim, frames):
        scores = [3] * collage.num_frames
        segments = []
        for w, a in zip(windows, answers, strict=True):
            s = labels.get(a.label, 3)
            for f in range(w.start, w.end):
                scores[f] = s
            if segments and segments[-1]["label"] == a.label:
                segments[-1]["end_s"] = round(w.end / collage.fps, 3)
            else:
                segments.append({"label": a.label, "start_s": round(w.start / collage.fps, 3),
                                 "end_s": round(w.end / collage.fps, 3)})
        return {"id": job["id"], "fps": collage.fps, "num_frames": collage.num_frames,
                "segments": segments, "frame_scores": scores,
                "invalid_clips": sum(not a.valid for a in answers), "trim": trim}

    engine = SimpleNamespace(
        render=lambda proc, messages, thinking=False, vision_ids=False:
        f"{proc}|{messages}|thinking={thinking}|vision_ids={vision_ids}",
        COLLAGE_VIDEO_SIZE={"shortest_edge": 48, "longest_edge": 192})
    _add_fake_rubric(core)
    return SimpleNamespace(core=core, engine=engine,
                           dense_reward=SimpleNamespace(summarize=summarize),
                           rubric_reward=_fake_rubric_module(registry),
                           pool_initializer=lambda: {})


RUBRIC_ALLOWED = "ALLOWED="  # the fake prompt carries each step's allowed values (JSON)


def _add_fake_rubric(core: SimpleNamespace) -> None:
    """NVIDIA's rubric functions (``core.rubric_header`` ... ``medoid``), same signatures."""

    def steps(rubric):
        out = []
        for item in rubric:
            points = [float(g["points"]) for g in item["grading"]]
            if not points:
                raise NvidiaValueError("every rubric step needs at least one allowed value")
            out.append({"action": str(item["action"]),
                        "max_score": float(item.get("max_score", max(points))),
                        "points": points})
        if not out:
            raise NvidiaValueError("the rubric has no steps")
        return out

    def rubric_header(robot, cameras):
        if not 1 <= len(cameras) <= 4:
            raise NvidiaValueError("1 to 4 camera videos are supported")
        return f"[fake header: {robot}; {', '.join(cameras)}]\n\n"

    def rubric_prompt(task, rubric, header, task_name=None):
        allowed = json.dumps([s["points"] for s in steps(rubric)])
        return f"{header}Task: {task_name or task}\n{RUBRIC_ALLOWED}{allowed}\n"

    def rubric_messages(prompt, video_paths, fps=2.0, max_frames=166):
        content = [{"type": "video", "video": str(p), "fps": fps, "max_frames": max_frames}
                   for p in video_paths]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "kind": "rubric", "content": content}]

    def split_thinking(text):
        before, closed, after = text.partition("</think>")
        if not closed:
            return "", before.replace("<think>", "", 1).strip() or None
        return after.strip(), before.replace("<think>", "", 1).strip() or None

    def parse_rubric(answer, rubric):
        st = steps(rubric)
        m = re.search(r"ANSWER:\s*(\[[^\]]*\])", answer or "")
        values = None
        if m:
            try:
                values = [float(v) for v in json.loads(m.group(1))]
            except (ValueError, TypeError):
                values = None
        ok = values is not None and len(values) == len(st)
        vals = list(values or []) + [0.0] * len(st)
        return [min(max(0.0, vals[i]), st[i]["max_score"]) for i in range(len(st))], ok

    def medoid(vectors):
        if not vectors:
            raise NvidiaValueError("no parsed rubric answers")
        return min(range(len(vectors)), key=lambda i: (sum(
            sum(abs(x - y) for x, y in zip(vectors[i], o, strict=False)) for o in vectors), i))

    core.RUBRIC_FPS, core.RUBRIC_MAX_FRAMES = 2.0, 8
    core.rubric_header, core.rubric_prompt = rubric_header, rubric_prompt
    core.rubric_messages, core.split_thinking = rubric_messages, split_thinking
    core.parse_rubric, core.medoid = parse_rubric, medoid


def _fake_rubric_module(registry: EpisodeRegistry) -> SimpleNamespace:
    """NVIDIA's ``rubric_reward`` module interface: ``base_parser`` / ``main`` (only argument
    parsing: ``stages.rubric.nvidia_recipe`` reads the defaults through it) and
    ``video_inputs(messages)`` (one ``(frames, metadata)`` per camera, fake-decoded)."""
    mod = SimpleNamespace()

    def base_parser(description, *, tensor_parallel_size=1):
        p = argparse.ArgumentParser(description=description)
        for flag in ("--jobs", "--output", "--model"):
            p.add_argument(flag)
        p.add_argument("--tensor-parallel-size", type=int, default=tensor_parallel_size)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
        p.add_argument("--max-num-seqs", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        return p

    def main(argv=None):
        p = mod.base_parser("fake rubric", tensor_parallel_size=2)
        p.add_argument("--num-samples", type=int, default=9)
        p.add_argument("--fps", type=float, default=2.0)
        p.add_argument("--max-frames", type=int, default=8)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--max-new-tokens", type=int, default=64)
        p.add_argument("--max-model-len", type=int, default=4096)
        p.add_argument("--max-num-batched-tokens", type=int, default=1024)
        p.set_defaults(gpu_memory_utilization=0.85)
        p.parse_args(argv)
        return 0

    def video_inputs(messages):
        videos = [c for c in messages[-1]["content"] if c.get("type") == "video"]
        paths = [v["video"] for v in videos]
        uid = registry.uid(paths) or "|".join(paths)
        checkpoint(uid)
        if nv_rejects(uid):
            raise FakeNvidiaInputError(f"{paths[0]}: fake decoder rejected the video")
        out = []
        for v in videos:
            d = decode(v["video"])
            step = max(int(round(d.fps / float(v["fps"]))), 1)
            idx = list(range(0, d.num_frames, step))[: int(v["max_frames"])]
            out.append((d.frames[idx], {"fps": d.fps / step, "frames_indices": idx,
                                        "total_num_frames": d.num_frames,
                                        "duration": d.duration_s, "do_sample_frames": False}))
        return out

    mod.base_parser, mod.main, mod.video_inputs = base_parser, main, video_inputs
    mod.load_llm = lambda *a, **kw: FakeLLM()
    return mod


class _Tokenizer:
    def decode(self, ids) -> str:
        return str(ids[0])


class FakeLLM:
    """vLLM ``LLM`` stand-in: each answer is a function of its prompt and its video frames."""

    def __init__(self):
        self.calls = 0
        self.requests = 0

    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()

    def reset_prefix_cache(self) -> bool:
        return True

    def _answer(self, prompt: str, frames: Any) -> tuple[str, list[int], list[dict]]:
        d = _h(prompt, _digest_frames(frames))
        if "'kind': 'dense'" in prompt:
            label = ("normal", "normal", "suboptimal", "failure")[d % 4]
            return f"Label: {label}", [0], [{0: SimpleNamespace(logprob=-0.05)}]
        score = 1 + d % 5
        text = f"Score: {score}"
        if "'rationale': True" in prompt:
            text += f"\nReason: fake rationale {d % 997}"
        alt = 1 + (score % 5)
        lp = {score: SimpleNamespace(logprob=-0.2 - (d % 7) / 20),
              alt: SimpleNamespace(logprob=-1.5 - (d % 5) / 10)}
        return text, [score], [lp]

    def _rubric(self, prompt: str, videos: list, n: int) -> SimpleNamespace:
        """``n`` samples; every 4th (by content) stops inside its thinking (unparseable)."""
        m = re.search(re.escape(RUBRIC_ALLOWED) + r"(\[\[.*?\]\])", prompt)
        allowed = json.loads(m.group(1)) if m else [[0.0]]
        d = _h(prompt, *(_digest_frames(f) for f, _m in videos))
        outs = []
        for i in range(n):
            if _h(d, i) % 4 == 0:
                text, reason = f"<think>fake trace {i} cut off", "length"
            else:
                pick = [vals[_h(d, i, j) % len(vals)] for j, vals in enumerate(allowed)]
                text, reason = f"<think>fake trace {i}</think>\nANSWER: {json.dumps(pick)}", "stop"
            outs.append(SimpleNamespace(text=text, token_ids=[0] * (8 + i), logprobs=None,
                                        finish_reason=reason))
        return SimpleNamespace(prompt=prompt, prompt_token_ids=[0] * (len(prompt) // 4),
                               outputs=outs)

    def generate(self, requests, params, use_tqdm=True):
        self.calls += 1
        outs = []
        for r in requests:
            self.requests += 1
            if "'kind': 'rubric'" in r["prompt"]:
                outs.append(self._rubric(r["prompt"], r["multi_modal_data"]["video"],
                                         int(getattr(params, "n", 1) or 1)))
                continue
            frames = r["multi_modal_data"]["video"][0]
            text, ids, lps = self._answer(r["prompt"], frames)
            outs.append(SimpleNamespace(
                prompt=r["prompt"], prompt_token_ids=[0] * (len(r["prompt"]) // 4),
                outputs=[SimpleNamespace(text=text, token_ids=ids, logprobs=lps,
                                         finish_reason="stop")]))
        return outs


# ---------------------------------------------------------------------------------------------
# vllm shim
# ---------------------------------------------------------------------------------------------


class _Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"




_SHIM_LOCK = threading.Lock()
_SHIM = {"users": 0, "saved": None}


def _has_real_vllm() -> bool:
    try:
        from vllm import SamplingParams  # noqa: F401
        from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
    except ImportError:  # not installed (or a stray namespace dir named vllm)
        return False
    return True


@contextlib.contextmanager
def vllm_shim() -> Iterator[bool]:
    """Make ``from vllm import SamplingParams`` (``stages.reward.sampling_params``) work without
    vLLM: if a real one is not importable, put a stand-in ``vllm`` / ``vllm.sampling_params``
    (keyword-bag ``SamplingParams`` / ``StructuredOutputsParams``) into ``sys.modules`` while
    any context is open, and restore what was there afterwards. Yields whether the shim is in
    use. A real vLLM is left alone (the fake LLM ignores the params' type either way)."""
    names = ("vllm", "vllm.sampling_params")
    with _SHIM_LOCK:
        if _SHIM["users"] == 0:
            if _has_real_vllm():
                use = False
            else:
                use = True
                _SHIM["saved"] = {n: sys.modules.get(n) for n in names}
                mod = types.ModuleType("vllm")
                mod._flora_shim = True  # type: ignore[attr-defined]
                mod.SamplingParams = type("SamplingParams", (_Params,), {})  # type: ignore
                sp = types.ModuleType("vllm.sampling_params")
                sp.StructuredOutputsParams = type(  # type: ignore[attr-defined]
                    "StructuredOutputsParams", (_Params,), {})
                mod.sampling_params = sp  # type: ignore[attr-defined]
                sys.modules["vllm"], sys.modules["vllm.sampling_params"] = mod, sp
        else:
            use = bool(getattr(sys.modules.get("vllm"), "_flora_shim", False))
        if use:
            _SHIM["users"] += 1
    try:
        yield use
    finally:
        if use:
            with _SHIM_LOCK:
                _SHIM["users"] -= 1
                if _SHIM["users"] == 0:
                    for n, m in (_SHIM["saved"] or {}).items():
                        if m is None:
                            sys.modules.pop(n, None)
                        else:
                            sys.modules[n] = m
                    _SHIM["saved"] = None
