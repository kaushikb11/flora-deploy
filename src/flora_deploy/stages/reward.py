"""Terminal and dense rewards, batched across episodes, on NVIDIA's offline-``LLM`` recipe.

PLAN.md §6.4 step 4 and D14. Every prompt, frame sample, collage, regex, parser and stitcher
comes from NVIDIA's Rewarder ``batch_utils`` (linked at runtime, never copied): this module only
changes *how requests are grouped*. NVIDIA's scripts call ``llm.generate`` once per
``max_num_seqs`` episodes (terminal) or once per ``max_num_seqs`` clips of one episode (dense);
here one ``llm.generate`` call carries every request of a mode for the whole episode list, and
the engine's own scheduler (``max_num_seqs``) does the batching.

The module is Modal-agnostic. ``nv`` is the handle from
``load_family(Family.REWARDER, code_dir)`` (it needs ``.core``, ``.engine`` and
``.dense_reward``), ``processor`` is NVIDIA's ``engine.load_processor(path)`` and ``llm`` is the
engine built by :func:`build_engine` (NVIDIA's ``engine.load_llm``). ``vllm`` is imported lazily,
only for ``SamplingParams``.

Records follow PLAN.md §6.6: ``terminal{score, expected_score, probabilities[5], rationale?}``
and ``dense{segments, per_step}``, plus ``nv``, the record NVIDIA's script would have written
for the episode (kept for parity checks and for users of NVIDIA's format).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "flora_labels_v1"
MODES = ("terminal", "terminal_rationale", "dense")
# ``--max-model-len`` default of both NVIDIA scripts (terminal_reward.py, dense_reward.py).
# The reference run re-checks it: NVIDIA's ``load_llm`` call is compared to ours.
MAX_MODEL_LEN = 16384
ENGINE_LABEL = "vllm-0.24.0-offline"


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown reward mode {mode!r}; expected one of {MODES}")
    return mode


def with_rationale(mode: str) -> bool:
    return check_mode(mode) == "terminal_rationale"


@dataclass(frozen=True)
class Episode:
    """One episode: camera videos in NVIDIA's collage order (2 views L/R, 3 views L/C/R)."""

    id: str
    task: str
    videos: tuple[str, ...]

    @classmethod
    def from_job(cls, job: dict[str, Any]) -> Episode:
        return cls(id=str(job["id"]), task=job["task"], videos=tuple(job["videos"]))

    def as_job(self) -> dict[str, Any]:
        return {"id": self.id, "task": self.task, "videos": list(self.videos)}


def read_jobs(nv, jobs_path: str | Path) -> list[Episode]:
    """Episodes from a jobs JSONL exactly as NVIDIA's ``engine.read_jobs`` reads it."""
    args = argparse.Namespace(video=None, task=None, jobs=Path(jobs_path))
    return [Episode.from_job(job) for job in nv.engine.read_jobs(args)]


# ---------------------------------------------------------------------------
# Engine (NVIDIA's load_llm, built once and shared by every mode)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSpec:
    """What :func:`build_engine` passed to NVIDIA's ``engine.load_llm``."""

    model_path: str
    args: dict[str, Any]  # the base_parser fields load_llm reads
    kwargs: dict[str, Any]  # max_model_len, mm_processor_kwargs, engine kwargs

    def as_json(self) -> dict[str, Any]:
        return json.loads(json.dumps({"model": self.model_path, **self.args, **self.kwargs}))


LOAD_LLM_ARGS = ("tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs", "seed")


def default_args(nv, model_path: str) -> argparse.Namespace:
    """NVIDIA's ``base_parser`` defaults (the CLI flags both scripts start from)."""
    parser = nv.engine.base_parser("flora-deploy rewarder")
    return parser.parse_args(["--video", "unused", "--output", "unused", "--model", model_path])


def engine_spec(nv, model_path: str) -> EngineSpec:
    """The terminal script's ``load_llm`` call; dense makes the same call minus
    ``max_logprobs`` (whose vLLM default is the same 20, checked in :func:`shared_engine`)."""
    args = default_args(nv, model_path)
    kwargs = {
        "max_model_len": MAX_MODEL_LEN,
        "mm_processor_kwargs": {"do_sample_frames": False, "size": nv.engine.COLLAGE_VIDEO_SIZE},
        "max_logprobs": nv.core.TERMINAL_TOP_LOGPROBS,
    }
    return EngineSpec(model_path, {k: getattr(args, k) for k in LOAD_LLM_ARGS}, kwargs)


def build_engine(nv, model_path: str):
    """Build the shared ``vllm.LLM`` through NVIDIA's ``engine.load_llm``. Returns (llm, spec)."""
    spec = engine_spec(nv, model_path)
    args = argparse.Namespace(**spec.args)
    llm = nv.engine.load_llm(spec.model_path, args, **spec.kwargs)
    return llm, spec


class EngineMismatchError(RuntimeError):
    """NVIDIA's script asked for an engine different from the shared one."""


@contextlib.contextmanager
def shared_engine(
    module, llm, spec: EngineSpec, defaults: dict[str, Any] | None = None
) -> Iterator[list[dict[str, Any]]]:
    """Let NVIDIA's ``module.main`` (terminal_reward / dense_reward) reuse ``llm``.

    Replaces ``module.load_llm`` for the duration with a function that checks the requested
    engine equals ``spec`` (a missing kwarg must equal ``defaults[key]``, vLLM's default) and
    returns ``llm``. Yields the list of calls it saw.
    """
    defaults = defaults or {}
    calls: list[dict[str, Any]] = []
    original = module.load_llm

    def load_llm(path, args, *, max_model_len, videos_per_prompt=1, mm_processor_kwargs=None,
                 **engine_kwargs):
        requested = {
            "max_model_len": max_model_len,
            "mm_processor_kwargs": mm_processor_kwargs,
            **engine_kwargs,
        }
        calls.append({"path": str(path), **{k: getattr(args, k) for k in LOAD_LLM_ARGS},
                      "videos_per_prompt": videos_per_prompt, **requested})
        problems = []
        # NVIDIA's resolve_model() resolves symlinks (e.g. a Modal Volume mount point).
        if Path(path).resolve() != Path(spec.model_path).resolve():
            problems.append(f"model {path} != {spec.model_path}")
        if videos_per_prompt != 1:
            problems.append(f"videos_per_prompt {videos_per_prompt} != 1")
        for key in LOAD_LLM_ARGS:
            if getattr(args, key) != spec.args[key]:
                problems.append(f"{key} {getattr(args, key)!r} != {spec.args[key]!r}")
        for key in sorted(set(requested) | set(spec.kwargs)):
            want = spec.kwargs.get(key, defaults.get(key, _MISSING))
            got = requested.get(key, defaults.get(key, _MISSING))
            if want != got:
                problems.append(f"{key} {got!r} != {want!r}")
        if problems:
            raise EngineMismatchError("; ".join(problems))
        return llm

    module.load_llm = load_llm
    try:
        yield calls
    finally:
        module.load_llm = original


_MISSING = object()


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """One vLLM request plus where its answer goes."""

    episode: int  # index into the episode list
    window: int | None  # dense clip index; None for terminal
    prompt: str
    multi_modal_data: dict[str, Any]
    mm_processor_kwargs: dict[str, Any]

    def as_vllm(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "multi_modal_data": self.multi_modal_data,
            "mm_processor_kwargs": self.mm_processor_kwargs,
        }


@dataclass
class Batch:
    mode: str
    episodes: list[Episode]
    collages: list[Any]
    plans: list[Any]  # terminal: frame indices per episode; dense: windows per episode
    requests: list[Request] = field(default_factory=list)


def _request(nv, collage, indices, prompt: str, episode: int, window: int | None) -> Request:
    return Request(
        episode=episode,
        window=window,
        prompt=prompt,
        multi_modal_data={
            "video": (
                collage.frames(indices),
                nv.core.video_metadata(indices, collage.fps, collage.num_frames),
            )
        },
        mm_processor_kwargs={"do_sample_frames": False},
    )


def build_terminal(nv, processor, episodes: Sequence[Episode], *, rationale: bool) -> Batch:
    """One request per episode: NVIDIA's whole-episode sampling, collage and prompt."""
    core, render = nv.core, nv.engine.render
    batch = Batch("terminal_rationale" if rationale else "terminal", list(episodes), [], [])
    for number, episode in enumerate(episodes):
        collage = core.Collage(list(episode.videos))
        indices = core.terminal_frame_indices(collage.num_frames, collage.fps)
        prompt = render(processor, core.terminal_messages(episode.task, rationale), thinking=False)
        batch.collages.append(collage)
        batch.plans.append(indices)
        batch.requests.append(_request(nv, collage, indices, prompt, number, None))
        collage.release()  # the request holds its own stacked copy
    return batch


def build_dense(nv, processor, episodes: Sequence[Episode]) -> Batch:
    """Every 16 s clip of every episode: NVIDIA's windows, collage and prompt."""
    core, render = nv.core, nv.engine.render
    batch = Batch("dense", list(episodes), [], [])
    for number, episode in enumerate(episodes):
        collage = core.Collage(list(episode.videos))
        windows = core.dense_windows(collage.num_frames, collage.fps)
        collage.frames(sorted({i for w in windows for i in w.frame_indices}))  # one decode pass
        batch.collages.append(collage)
        batch.plans.append(windows)
        for window in windows:
            prompt = render(processor, core.dense_messages(episode.task, window), thinking=False)
            batch.requests.append(
                _request(nv, collage, window.frame_indices, prompt, number, window.index)
            )
        collage.release()
    return batch


def sampling_params(nv, mode: str):
    """NVIDIA's ``SamplingParams`` for ``mode`` (greedy, regex-constrained)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    core = nv.core
    if check_mode(mode) == "dense":
        return SamplingParams(
            temperature=0.0,
            max_tokens=core.DENSE_MAX_TOKENS,
            structured_outputs=StructuredOutputsParams(regex=core.dense_regex()),
        )
    rationale = with_rationale(mode)
    return SamplingParams(
        temperature=0.0,
        max_tokens=core.terminal_max_tokens(rationale),
        logprobs=core.TERMINAL_TOP_LOGPROBS,
        structured_outputs=StructuredOutputsParams(regex=core.terminal_regex(rationale)),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def token_positions(tokenizer, completion) -> list[tuple[str, float, list[tuple[str, float]]]]:
    """``(token text, logprob, top candidates)`` per generated token, the input format of
    NVIDIA's ``core.score_probabilities`` (texts via ``tokenizer.decode([id])``)."""
    return [
        (
            tokenizer.decode([token]),
            float(candidates[token].logprob),
            [(tokenizer.decode([i]), float(c.logprob)) for i, c in candidates.items()],
        )
        for token, candidates in zip(completion.token_ids, completion.logprobs, strict=True)
    ]


def _usage(output) -> dict[str, int]:
    completion = output.outputs[0]
    return {
        "prompt_tokens": len(output.prompt_token_ids or ()),
        "output_tokens": len(completion.token_ids),
    }


def terminal_record(nv, episode: Episode, collage, indices, completion, tokenizer,
                    rationale: bool) -> dict[str, Any]:
    core = nv.core
    score, text = core.parse_terminal(completion.text, rationale)
    probabilities, expected = core.score_probabilities(
        token_positions(tokenizer, completion), score
    )
    terminal: dict[str, Any] = {
        "score": score,
        "expected_score": expected,
        "probabilities": [probabilities[str(k)] for k in range(1, 6)],
    }
    if rationale:
        terminal["rationale"] = text
    # The line NVIDIA's terminal_reward.py writes for this episode (same fields and rounding).
    nv_record = {
        "id": episode.id,
        "task": episode.task,
        "score": score,
        "expected_score": round(expected, 4),
        "progress": round((expected - 1.0) / 4.0, 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "rationale": text,
        "duration_s": round(collage.duration_s, 3),
        "sampled_frames": len(indices),
        "output": completion.text,
    }
    return {"terminal": terminal, "nv": nv_record}


def dense_record(nv, episode: Episode, collage, windows, texts: list[str],
                 tail_trim_s: float) -> dict[str, Any]:
    core = nv.core
    answers = [core.parse_dense(text) for text in texts]
    job = episode.as_job()
    nv_record = nv.dense_reward.summarize(job, collage, windows, answers, texts, tail_trim_s, True)
    dense = {
        "segments": nv_record["segments"],
        "per_step": nv_record["frame_scores"],  # one 1-3 score per source frame
        "fps": nv_record["fps"],
        "num_frames": nv_record["num_frames"],
        "invalid_clips": nv_record["invalid_clips"],
    }
    return {"dense": dense, "nv": nv_record}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class RewardRun:
    mode: str
    records: list[dict[str, Any]]
    stats: dict[str, Any]


def throughput(stats: dict[str, Any]) -> dict[str, float]:
    seconds = stats["generate_s"] or math.nan
    return {
        "requests_per_s": stats["num_requests"] / seconds,
        "episodes_per_s": stats["num_episodes"] / seconds,
        "prompt_tokens_per_s": stats["prompt_tokens"] / seconds,
        "output_tokens_per_s": stats["output_tokens"] / seconds,
    }


def run(llm, nv, processor, episodes: Sequence[Episode], mode: str, *,
        tail_trim_s: float | None = None, provenance: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter) -> RewardRun:
    """Score ``episodes`` in ``mode`` with exactly one ``llm.generate`` call."""
    check_mode(mode)
    episodes = list(episodes)
    t0 = clock()
    if mode == "dense":
        batch = build_dense(nv, processor, episodes)
    else:
        batch = build_terminal(nv, processor, episodes, rationale=with_rationale(mode))
    params = sampling_params(nv, mode)
    t1 = clock()
    outputs = llm.generate([r.as_vllm() for r in batch.requests], params, use_tqdm=False)
    t2 = clock()
    if len(outputs) != len(batch.requests):
        raise RuntimeError(f"{len(outputs)} outputs for {len(batch.requests)} requests")
    for request in batch.requests:  # drop frame arrays as soon as the engine is done
        request.multi_modal_data = {}

    pairs = zip(batch.requests, outputs, strict=True)
    usage = [{"window": r.window, **_usage(o)} for r, o in pairs]
    per_episode: list[list[int]] = [[] for _ in episodes]
    for position, request in enumerate(batch.requests):
        per_episode[request.episode].append(position)

    records = []
    tokenizer = llm.get_tokenizer() if mode != "dense" else None
    trim = nv.core.DENSE_TAIL_TRIM_S if tail_trim_s is None else tail_trim_s
    for number, episode in enumerate(episodes):
        collage, plan = batch.collages[number], batch.plans[number]
        positions = per_episode[number]
        if mode == "dense":
            texts = [outputs[p].outputs[0].text for p in positions]
            body = dense_record(nv, episode, collage, plan, texts, trim)
        else:
            (p,) = positions
            body = terminal_record(nv, episode, collage, plan, outputs[p].outputs[0], tokenizer,
                                   with_rationale(mode))
        records.append({
            "schema": SCHEMA,
            "episode_id": episode.id,
            "stage": "reward",
            "mode": mode,
            **body,
            "requests": [usage[p] for p in positions],
            "provenance": {"engine": ENGINE_LABEL, **(provenance or {})},
        })
    t3 = clock()
    stats = {
        "mode": mode,
        "generate_calls": 1,
        "num_episodes": len(episodes),
        "num_requests": len(batch.requests),
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "build_s": t1 - t0,
        "generate_s": t2 - t1,
        "parse_s": t3 - t2,
        "requests": [{"episode_id": episodes[r.episode].id, **u}
                     for r, u in zip(batch.requests, usage, strict=True)],
    }
    stats["throughput"] = throughput(stats)
    return RewardRun(mode, records, stats)


# ---------------------------------------------------------------------------
# Reference (NVIDIA's own scripts) instrumentation and parity
# ---------------------------------------------------------------------------


def reference_argv(mode: str, jobs: str | Path, output: str | Path, model: str) -> list[str]:
    """argv for NVIDIA's ``terminal_reward.main`` / ``dense_reward.main`` (their defaults)."""
    argv = ["--jobs", str(jobs), "--output", str(output), "--model", model, "--overwrite"]
    if with_rationale(mode):
        argv.append("--with-rationale")
    if mode == "dense":
        argv.append("--frame-scores")
    return argv


class RecordingLLM:
    """Delegates to ``llm`` and records every ``generate`` call (time, token counts)."""

    def __init__(self, llm, clock: Callable[[], float] = time.perf_counter):
        self._llm, self._clock = llm, clock
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompts, *args, **kwargs):
        start = self._clock()
        outputs = self._llm.generate(prompts, *args, **kwargs)
        self.calls.append({
            "num_requests": len(prompts),
            "seconds": self._clock() - start,
            "usage": [_usage(o) for o in outputs],
        })
        return outputs

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def stats(self, mode: str, num_episodes: int) -> dict[str, Any]:
        usage = [u for call in self.calls for u in call["usage"]]
        stats = {
            "mode": mode,
            "generate_calls": len(self.calls),
            "num_episodes": num_episodes,
            "num_requests": len(usage),
            "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
            "output_tokens": sum(u["output_tokens"] for u in usage),
            "generate_s": sum(call["seconds"] for call in self.calls),
            "requests": usage,
        }
        stats["throughput"] = throughput(stats)
        return stats


@contextlib.contextmanager
def tap_score_probabilities(core) -> Iterator[list[list[float]]]:
    """Record the unrounded ``P(1..5)`` of every ``core.score_probabilities`` call (NVIDIA's
    script writes them rounded to 4 places). The function itself is called unchanged."""
    seen: list[list[float]] = []
    original = core.score_probabilities

    def recording(positions, score):
        probabilities, expected = original(positions, score)
        seen.append([probabilities[str(k)] for k in range(1, 6)])
        return probabilities, expected

    core.score_probabilities = recording
    try:
        yield seen
    finally:
        core.score_probabilities = original


def compare(mode: str, reference: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]],
            reference_probabilities: Sequence[Sequence[float]] | None = None,
            tolerance: float = 1e-6) -> dict[str, Any]:
    """Batched records vs NVIDIA's reference lines (same episode order)."""
    check_mode(mode)
    ref_by_id = {line["id"]: line for line in reference}
    episodes = []
    for number, record in enumerate(records):
        ref = ref_by_id.get(record["episode_id"])
        if ref is None:
            episodes.append({"episode_id": record["episode_id"], "missing_reference": True,
                             "identical": False})
            continue
        ours = record["nv"]
        diff_keys = sorted(k for k in set(ref) | set(ours) if ref.get(k) != ours.get(k))
        entry: dict[str, Any] = {"episode_id": record["episode_id"], "diff_keys": diff_keys}
        if mode == "dense":
            ref_texts = [c["output"] for c in ref["clips"]]
            our_texts = [c["output"] for c in ours["clips"]]
            entry["clips"] = len(our_texts)
            entry["clip_text_mismatches"] = [i for i, (a, b) in
                                             enumerate(zip(ref_texts, our_texts, strict=False))
                                             if a != b] + (
                [] if len(ref_texts) == len(our_texts) else ["count"])
            entry["frame_scores_equal"] = ref.get("frame_scores") == ours.get("frame_scores")
            entry["identical"] = not diff_keys
        else:
            entry["text_equal"] = ref["output"] == ours["output"]
            entry["score_equal"] = ref["score"] == ours["score"]
            if reference_probabilities is not None:
                ref_p = reference_probabilities[number]
                our_p = record["terminal"]["probabilities"]
                entry["max_abs_prob_diff"] = max(abs(a - b) for a, b in zip(ref_p, our_p,
                                                                             strict=True))
            entry["identical"] = not diff_keys and (
                entry.get("max_abs_prob_diff", 0.0) <= tolerance
            )
        episodes.append(entry)
    extra = sorted(set(ref_by_id) - {r["episode_id"] for r in records})
    return {
        "mode": mode,
        "identical": bool(episodes) and all(e["identical"] for e in episodes) and not extra,
        "episodes": episodes,
        "reference_only": extra,
    }
