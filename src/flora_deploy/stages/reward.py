"""Terminal and dense rewards, batched across episodes, on NVIDIA's offline-``LLM`` recipe.

PLAN-v3.md §6.4 step 4 and D14. Prompts, frame sampling, collage, regexes, parsers, probability
renormalization and dense stitching all come from NVIDIA's Rewarder ``batch_utils``, linked at
runtime. This module changes *how requests are grouped*, isolates failures per episode and
persists results per round:

- NVIDIA's scripts call ``llm.generate`` once per ``max_num_seqs`` episodes (terminal) or once
  per ``max_num_seqs`` clips of one episode (dense). Here the requests of the whole episode list
  are cut into **size-bounded rounds** (:class:`RoundLimits`: estimated frame + processed-tensor
  bytes and a request count), one ``llm.generate`` per round. By default (``align="nvidia"``)
  each round is exactly one of NVIDIA's ``generate`` calls, because on the real engine batch
  composition changes logprobs (H100: terminal ``P(1..5)`` up to 0.03 apart); ``"packed"``
  rounds fill across episodes for throughput, outside ``exact``. Rounds never change the
  requests themselves, and with a content-keyed fake LLM every grouping gives the same records
  (tested); only numerics can tell groupings apart.
- **Stable NVIDIA groups (review 4 M3).** NVIDIA's terminal score depends on its batch-mates.
  The default grouping is a pure function of the *planned* episode order (``run(planned=...)``:
  the **job's** original episode list, as NVIDIA's script would read the whole jobs file; the
  manifest cuts ``reward.terminal`` shards on group boundaries, so a shard holds whole groups,
  review 5 M-B):
  episode at planned position ``p`` belongs to terminal group ``p // max_num_seqs``, and dense
  clip ``j`` of an episode to chunk ``j // max_num_seqs``. Episodes already done, failed or
  skipped on a resume or bisection leave *holes* in their group (a partial group) and never
  shift the other groups' boundaries. Every audit records its ``grouping`` (group id, planned
  members, the batch that actually ran, ``aligned`` and why not: ``partial_group``,
  ``group_outside_planned``, ``budget_split``, ``retry_singleton``, ``packed``).
- Video decoding (NVIDIA's ``Collage``) runs on a thread pool or a spawn process pool
  (``nv.pool_initializer()``), a bounded number of episodes ahead, so round k+1 decodes while
  round k generates.
- **Per-episode isolation.** A failure to read one episode's videos is recorded before
  ``generate``. A failure to parse one output (NVIDIA's ``parse_terminal`` /
  ``score_probabilities`` raising, e.g. a rationale cut off by ``max_tokens``) or to stitch one
  dense episode becomes an :class:`~flora_deploy.jobs.types.EpisodeError` of kind
  ``nv_failure`` carrying NVIDIA's message and the ``finish_reason``. Every other record is kept.
  An input error raised by ``generate`` for a whole round is retried one episode at a time
  (``kind="infra"`` for the episode that still fails).
- **Partial persistence.** After each round its records, audits and errors are handed to a
  :class:`PartSink` (local part file, then a whole-file copy), so a timeout or preemption loses at
  most the round in flight.

Records are ``records.StageRecord`` (stage ``reward.terminal`` / ``reward.dense``). NVIDIA's full
output line for the episode (what its script would have written) goes into a separate *audit*
entry, for parity checks and for users of NVIDIA's format.

What is ours rather than NVIDIA's. No prompt text, regex or parser is copied. About 20 lines are
*re-expressions of NVIDIA's script glue* (its ``batch_utils`` is under NVIDIA's license, so they
are written to its interface, not copied): the ``SamplingParams`` construction
(:func:`sampling_params`), the ``(token, logprob, candidates)`` list given to
``score_probabilities`` (:func:`token_positions`), the field names / rounding of NVIDIA's terminal
output line (:func:`terminal_record`), and :func:`per_step`, which reads NVIDIA's stitched
per-frame scores at the middle of each 0.5 s step (the convention NVIDIA's ``per_second`` uses
per second). ``tests/nvlink/test_reward_parity.py`` runs NVIDIA's real scripts and is the drift
guard for all of them.

Rubric rewards are a different recipe (TP=2, ``max_model_len`` 98304, one video per camera, 9
samples per episode) and live in ``stages.rubric``: ``check_mode("rubric")`` here raises
``NotImplementedError`` naming it.

Prefix caching. NVIDIA's engine has ``enable_prefix_caching=True`` (hybrid "align" mode, where a
hit can change numerics). The shared text before the video is 104-413 tokens, below one KV block
(784), so only *identical* requests can hit. :func:`run` rejects duplicate episodes and calls
``llm.reset_prefix_cache()`` before every round (and checks it returned True), so a re-run or a
retry in a warm container never hits. ``reset_mm_cache`` is not needed: it does not change numerics.

The module is Modal-agnostic. ``nv`` is the handle from ``load_family(Family.REWARDER, code_dir)``
(``.core``, ``.engine``, ``.dense_reward``), ``processor`` is NVIDIA's
``engine.load_processor(path)`` and ``llm`` is the engine from :func:`build_engine` (NVIDIA's
``engine.load_llm``). ``vllm`` is imported lazily, only for ``SamplingParams``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flora_deploy.jobs.types import EpisodeError, episode_uid
from flora_deploy.records import DenseReward, Provenance, StageRecord, TerminalReward

SCHEMA = "flora_labels_v1"
MODES = ("terminal", "terminal_rationale", "dense")
DEFERRED_MODES = ("rubric",)
STAGES = {"terminal": "reward.terminal", "terminal_rationale": "reward.terminal",
          "dense": "reward.dense"}
# ``--max-model-len`` default of both NVIDIA scripts (terminal_reward.py, dense_reward.py).
# The reference run re-checks it: NVIDIA's ``load_llm`` call is compared to ours.
MAX_MODEL_LEN = 16384
ENGINE_LABEL = "vllm-0.24.0-offline"
MiB = 1 << 20
GiB = 1 << 30


def check_mode(mode: str) -> str:
    if mode in DEFERRED_MODES:
        raise NotImplementedError(
            "rubric rewards run in stages.rubric (TP=2, max_model_len 98304, one video per "
            "camera, 9 samples); stages.reward implements terminal, terminal_rationale and "
            "dense only"
        )
    if mode not in MODES:
        raise ValueError(f"unknown reward mode {mode!r}; expected one of {MODES}")
    return mode


def with_rationale(mode: str) -> bool:
    return check_mode(mode) == "terminal_rationale"


TERMINAL_MODE_ENV = "FLORA_REWARD_TERMINAL_MODE"  # deprecated (review 4 M4)


def terminal_mode(terminal_rationale: bool, env: Mapping[str, str] | None = None) -> str:
    """The ``reward.terminal`` mode from the config (``pipeline.terminal_rationale``, hashed in
    the reward.terminal stage hash). ``FLORA_REWARD_TERMINAL_MODE`` in ``env`` is a deprecated
    override: it may only repeat the config's choice; a different value raises ``ValueError``
    (an unhashed side channel must not change outputs)."""
    mode = "terminal_rationale" if terminal_rationale else "terminal"
    legacy = (env or {}).get(TERMINAL_MODE_ENV)
    if legacy:
        if legacy != mode:
            raise ValueError(f"{TERMINAL_MODE_ENV}={legacy!r} disagrees with the config "
                             f"(pipeline.terminal_rationale={terminal_rationale} -> {mode!r}); "
                             "set pipeline.terminal_rationale instead")
        warnings.warn(f"{TERMINAL_MODE_ENV} is deprecated; pipeline.terminal_rationale decides "
                      "the terminal mode", DeprecationWarning, stacklevel=2)
    return mode


@dataclass(frozen=True)
class Episode:
    """One episode: camera videos in NVIDIA's collage order (2 views L/R, 3 views L/C/R).

    ``envelope_ok`` is the EpisodeIndex's verdict on NVIDIA's reference envelope (``None``: no
    verdict, e.g. an ad-hoc jobs file); it decides ``certified`` on the record.

    ``sources``: per video, in ``videos`` order, ``(role, dataset uri, start_s, end_s)`` of the
    episode in the *source* dataset (``None`` bounds: the whole file). When set, record
    provenance names these instead of the local (possibly cut) path NVIDIA read (review 4 L4).
    """

    id: str
    task: str
    videos: tuple[str, ...]
    dataset: str = "local"
    envelope_ok: bool | None = None
    sources: tuple[tuple[str, str, float | None, float | None], ...] = ()

    @property
    def uid(self) -> str:
        return episode_uid(self.dataset, self.id)

    @classmethod
    def from_job(cls, job: dict[str, Any], *, root: str | Path | None = None,
                 dataset: str = "local", envelope_ok: bool | None = None) -> Episode:
        """A jobs-file line. Relative video paths resolve against ``root`` (NVIDIA resolves them
        against the jobs file's directory); without ``root`` they must be absolute."""
        videos = job["videos"]
        if not isinstance(videos, (list, tuple)):
            raise ValueError(f"episode {job.get('id')!r}: 'videos' must be a list")
        if root is not None:
            videos = [str((Path(root) / Path(v).expanduser()).resolve()) for v in videos]
        return cls(id=str(job["id"]), task=job["task"], videos=tuple(str(v) for v in videos),
                   dataset=str(job.get("dataset", dataset)),
                   envelope_ok=job.get("envelope_ok", envelope_ok))

    def as_job(self) -> dict[str, Any]:
        return {"id": self.id, "task": self.task, "videos": list(self.videos)}


def read_jobs(nv, jobs_path: str | Path, *, dataset: str = "local") -> list[Episode]:
    """Episodes from a jobs JSONL exactly as NVIDIA's ``engine.read_jobs`` reads it (absolute
    paths, NVIDIA's id rules)."""
    args = argparse.Namespace(video=None, task=None, jobs=Path(jobs_path))
    return [Episode.from_job(job, dataset=dataset) for job in nv.engine.read_jobs(args)]


def validate_episodes(episodes: Sequence[Episode]) -> list[Episode]:
    """Caller-level input checks (a violation is a bug in the caller, so it raises): unique
    uids, a non-empty task, 1-4 videos, absolute paths. A missing or unreadable video is *not*
    checked here; it becomes that episode's error."""
    episodes = list(episodes)
    seen: set[str] = set()
    for ep in episodes:
        if not isinstance(ep, Episode):
            raise TypeError(f"expected Episode, got {type(ep).__name__}")
        if ep.uid in seen:
            raise ValueError(f"duplicate episode {ep.uid!r}")
        seen.add(ep.uid)
        if not str(ep.task or "").strip():
            raise ValueError(f"episode {ep.uid!r}: empty task")
        if not 1 <= len(ep.videos) <= 4:
            raise ValueError(f"episode {ep.uid!r}: {len(ep.videos)} videos (1-4 supported)")
        relative = [v for v in ep.videos if not os.path.isabs(v)]
        if relative:
            raise ValueError(f"episode {ep.uid!r}: relative video paths {relative}; resolve them "
                             "(Episode.from_job(root=...)) first")
        if ep.sources and len(ep.sources) != len(ep.videos):
            raise ValueError(f"episode {ep.uid!r}: {len(ep.sources)} sources for "
                             f"{len(ep.videos)} videos")
    return episodes


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
# Request building (decode side: runs in a pool worker)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollageFacts:
    """What NVIDIA's record builders read from a ``Collage`` (fps, length, duration), without
    its frames, so a prepared episode can cross a process boundary cheaply."""

    fps: float
    num_frames: int
    duration_s: float
    width: int
    height: int


@dataclass
class Prepared:
    """One episode's decoded inputs: per request ``(frames [T,H,W,3] uint8, metadata)``."""

    facts: CollageFacts | None = None
    plan: Any = None  # terminal: frame indices; dense: NVIDIA DenseWindow list
    videos: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)
    error: dict[str, Any] | None = None  # {"kind", "message", "type"} when decoding failed
    decode_s: float = 0.0


def _failure(exc: BaseException) -> dict[str, Any]:
    # NVIDIA's inputs/parsers signal bad data with ValueError: that is NVIDIA's own failure
    # semantics (its script would stop on it). Anything else is infrastructure.
    kind = "nv_failure" if isinstance(exc, ValueError) else "infra"
    return {"kind": kind, "message": str(exc), "type": type(exc).__name__}


def prepare_episode(core, mode: str, episode: Episode) -> Prepared:
    """NVIDIA's collage, frame plan and decoded frames for every request of ``episode``.
    Never raises: a failure is returned in ``Prepared.error``."""
    t0 = time.perf_counter()
    try:
        collage = core.Collage(list(episode.videos))
        if mode == "dense":
            plan = core.dense_windows(collage.num_frames, collage.fps)
            collage.frames(sorted({i for w in plan for i in w.frame_indices}))  # one decode pass
            index_lists = [w.frame_indices for w in plan]
        else:
            plan = core.terminal_frame_indices(collage.num_frames, collage.fps)
            index_lists = [plan]
        videos = [(collage.frames(indices),
                   core.video_metadata(indices, collage.fps, collage.num_frames))
                  for indices in index_lists]
        facts = CollageFacts(float(collage.fps), int(collage.num_frames),
                             float(collage.duration_s), int(getattr(collage, "width", 0)),
                             int(getattr(collage, "height", 0)))
        collage.release()  # each request holds its own stacked copy
    except Exception as exc:  # noqa: BLE001 - isolated per episode, reported as EpisodeError
        return Prepared(error=_failure(exc), decode_s=time.perf_counter() - t0)
    return Prepared(facts, plan, videos, None, time.perf_counter() - t0)


def _prepare_in_pool_worker(mode: str, episode: Episode) -> Prepared:
    """Process-pool entry point: the worker linked the Rewarder family in its initializer
    (``nv.pool_initializer()``)."""
    from flora_deploy.nv.linker import loaded_family

    nv = loaded_family()
    if nv is None:
        raise RuntimeError("pool worker has no linked NVIDIA family; use nv.pool_initializer()")
    return prepare_episode(nv.core, mode, episode)


@dataclass
class Request:
    """One vLLM request plus where its answer goes."""

    episode: int  # index into the episode list
    window: int | None  # dense clip index; None for terminal
    prompt: str
    multi_modal_data: dict[str, Any]
    mm_processor_kwargs: dict[str, Any]
    est_bytes: int = 0
    # NVIDIA group (``align="nvidia"``): terminal ``planned_position // group``; dense
    # ``(planned_position, clip // group)``. ``None``: count-based boundaries.
    group: Any = None

    def as_vllm(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "multi_modal_data": self.multi_modal_data,
            "mm_processor_kwargs": self.mm_processor_kwargs,
        }


def episode_requests(nv, processor, mode: str, number: int, episode: Episode,
                     prepared: Prepared, limits: RoundLimits) -> list[Request]:
    """Render NVIDIA's prompts (main process) for a prepared episode."""
    core, render = nv.core, nv.engine.render
    if mode == "dense":
        prompts = [(w.index, render(processor, core.dense_messages(episode.task, w),
                                    thinking=False)) for w in prepared.plan]
    else:
        messages = core.terminal_messages(episode.task, with_rationale(mode))
        prompts = [(None, render(processor, messages, thinking=False))]
    return [
        Request(number, window, prompt, {"video": video}, {"do_sample_frames": False},
                limits.estimate(video[0]))
        for (window, prompt), video in zip(prompts, prepared.videos, strict=True)
    ]


def sampling_params(nv, mode: str):
    """NVIDIA's ``SamplingParams`` for ``mode`` (greedy, regex-constrained); re-expressed from
    its scripts."""
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
# Rounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundLimits:
    """Bounds of one ``llm.generate`` round.

    Sizing (WS-H review H2, NVIDIA's ``plan_collage`` on 640x480 views): a 3-view request holds
    ~61 MiB of uint8 frames and the HF video processor makes ~243 MiB of float32
    ``pixel_values_videos`` from it (4x), all resident in the engine until ``generate`` returns.
    ``estimate`` is ``raw * (1 + processed_factor)``, ~304 MiB for 3 views. The defaults cap a
    round at 6 GiB estimated (~20 three-view requests) and 32 requests (4x NVIDIA's
    ``max_num_seqs`` of 8, so the engine stays full); with one prepared round waiting while
    another generates, frame + tensor memory stays under ~12 GiB, next to the 2 x 4 GiB MM
    processor caches, inside the 80 GiB container. A single request above the byte budget still
    runs, alone.

    ``align`` decides how requests are grouped inside those bounds:

    - ``"nvidia"`` (default, the ``exact`` preset): every round is exactly one of the
      ``generate`` calls NVIDIA's script makes on the *planned* episode list: terminal group
      ``planned_position // nvidia_group`` (``max_num_seqs``, 8), dense chunk
      ``clip // nvidia_group`` *of one episode*. Missing members leave a partial group; they
      never move the other groups' boundaries (review 4 M3). Batch
      composition changes numerics on the real engine (measured on H100: greedy text and
      scores equal, terminal ``P(1..5)`` up to 0.03 apart between a 3-request call and calls of
      2 + 1), so only NVIDIA's own grouping reproduces NVIDIA's outputs. 8 four-view requests
      are ~3.2 GiB estimated, inside the default byte budget, so the budget never splits an
      NVIDIA group in practice; if it does, the round is counted in ``stats["budget_splits"]``.
    - ``"packed"``: fill each round up to ``max_requests`` / ``max_bytes`` across episodes, for
      throughput (dense packs clips of several episodes). Outputs then match NVIDIA's only up to
      batch-composition numerics: not ``exact``.
    """

    max_requests: int = 32
    max_bytes: int = 6 * GiB
    processed_factor: float = 4.0
    align: str = "nvidia"
    nvidia_group: int = 8

    def __post_init__(self) -> None:
        if (self.max_requests < 1 or self.max_bytes < 1 or self.processed_factor < 0
                or self.nvidia_group < 1 or self.align not in ("nvidia", "packed")):
            raise ValueError(f"invalid round limits {self}")

    def estimate(self, frames) -> int:
        raw = int(getattr(frames, "nbytes", 0))
        return int(raw * (1.0 + self.processed_factor))

    def fits(self, count: int, size: int, request: Request) -> bool:
        """Size bounds only (count, bytes)."""
        if count == 0:
            return True
        return count + 1 <= self.max_requests and size + request.est_bytes <= self.max_bytes

    def boundary(self, mode: str, batch: Sequence[Request], request: Request) -> bool:
        """NVIDIA's call boundary before ``request`` (``align="nvidia"``): a new group starts,
        or (without group ids) the count-based rule on the list as given."""
        if self.align != "nvidia" or not batch:
            return False
        if request.group is not None:
            return request.group != batch[-1].group
        if mode == "dense":
            return request.episode != batch[0].episode or len(batch) >= self.nvidia_group
        return len(batch) >= self.nvidia_group


UNBOUNDED = RoundLimits(max_requests=1 << 30, max_bytes=1 << 62, align="packed")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def token_positions(tokenizer, completion) -> list[tuple[str, float, list[tuple[str, float]]]]:
    """``(token text, logprob, top candidates)`` per generated token, the input format of
    NVIDIA's ``core.score_probabilities`` (texts via ``tokenizer.decode([id])``); re-expressed
    from NVIDIA's terminal script."""
    return [
        (
            tokenizer.decode([token]),
            float(candidates[token].logprob),
            [(tokenizer.decode([i]), float(c.logprob)) for i, c in candidates.items()],
        )
        for token, candidates in zip(completion.token_ids, completion.logprobs, strict=True)
    ]


def _usage(output) -> dict[str, Any]:
    completion = output.outputs[0]
    return {
        "prompt_tokens": len(output.prompt_token_ids or ()),
        "output_tokens": len(completion.token_ids),
        "finish_reason": getattr(completion, "finish_reason", None),
    }


def terminal_record(nv, episode: Episode, collage, indices, completion, tokenizer,
                    rationale: bool) -> tuple[TerminalReward, dict[str, Any]]:
    """NVIDIA's parse + probabilities. Raises NVIDIA's ``ValueError`` on a bad answer."""
    core = nv.core
    score, text = core.parse_terminal(completion.text, rationale)
    probabilities, expected = core.score_probabilities(
        token_positions(tokenizer, completion), score
    )
    terminal = TerminalReward(score=score, expected_score=expected,
                              probabilities=tuple(probabilities[str(k)] for k in range(1, 6)),
                              rationale=text if rationale else None)
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
    return terminal, nv_record


def per_step(nv, frame_scores: Sequence[int], fps: float, num_frames: int) -> list[int]:
    """One 1-3 score per 0.5 s of the episode, the dense model's group length
    (``2 / core.DENSE_FPS``): NVIDIA's stitched per-frame score read at the middle of each step,
    the convention NVIDIA's ``per_second`` uses per second."""
    step_s = 2.0 / nv.core.DENSE_FPS
    steps = math.ceil(num_frames / fps / step_s - 1e-9)
    return [int(frame_scores[min(num_frames - 1, int((k + 0.5) * step_s * fps))])
            for k in range(steps)]


def dense_record(nv, episode: Episode, collage, windows, texts: list[str],
                 tail_trim_s: float) -> tuple[DenseReward, dict[str, Any]]:
    """NVIDIA's ``parse_dense`` + ``summarize`` (an unparseable clip is NVIDIA's all-normal,
    ``valid=False``). Raises NVIDIA's ``ValueError`` if stitching fails."""
    answers = [nv.core.parse_dense(text) for text in texts]
    nv_record = nv.dense_reward.summarize(episode.as_job(), collage, windows, answers, texts,
                                          tail_trim_s, True)
    dense = DenseReward(segments=tuple(nv_record["segments"]),
                        per_step=tuple(per_step(nv, nv_record["frame_scores"], collage.fps,
                                                collage.num_frames)))
    return dense, nv_record


# ---------------------------------------------------------------------------
# Provenance and certification
# ---------------------------------------------------------------------------


def config_hash(nv, mode: str, *, spec: EngineSpec | None, models: dict[str, str],
                tail_trim_s: float | None, extra: dict[str, Any] | None = None) -> str:
    """sha256 over everything that changes this stage's outputs (D11): engine arguments,
    sampling (max tokens, regex), tail trim, model pins, logic version, ``extra`` (e.g. the
    image fingerprint). The decoder is excluded (it never changes outputs). Round limits are
    excluded too, but they are *not* output-neutral in general: ``align="packed"`` and budget
    splits change batch composition, hence numerics. ``exact`` uses ``align="nvidia"``, whose
    groups depend only on the job's planned order, and every audit records its ``grouping``.
    :func:`run` adds ``{"grouping": "packed"}`` to ``extra`` in packed mode (so the record's
    ``provenance.config_hash`` differs from NVIDIA mode's) and leaves those records
    uncertified (``packed_grouping``)."""
    from flora_deploy import LOGIC_VERSION

    core = nv.core
    if mode == "dense":
        sampling = {"max_tokens": core.DENSE_MAX_TOKENS, "regex": core.dense_regex()}
    else:
        r = with_rationale(mode)
        sampling = {"max_tokens": core.terminal_max_tokens(r), "regex": core.terminal_regex(r),
                    "logprobs": core.TERMINAL_TOP_LOGPROBS}
    blob = {"stage": STAGES[mode], "mode": mode, "engine_label": ENGINE_LABEL,
            "engine": spec.as_json() if spec else None, "sampling": sampling,
            "tail_trim_s": tail_trim_s if mode == "dense" else None, "models": models,
            "logic_version": LOGIC_VERSION, "extra": extra or {}}
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def base_provenance(nv, mode: str, *, spec: EngineSpec | None = None,
                    models: dict[str, str] | None = None, gpu: str | None = None,
                    preset: str = "exact", certificate: str | None = None,
                    tail_trim_s: float | None = None,
                    extra: dict[str, Any] | None = None) -> Provenance:
    from flora_deploy import LOGIC_VERSION, __version__

    models = dict(models or {})
    return Provenance(
        models=models, engine=ENGINE_LABEL, preset=preset,
        config_hash=config_hash(nv, mode, spec=spec, models=models, tail_trim_s=tail_trim_s,
                                extra=extra),
        logic_version=LOGIC_VERSION, framework=f"flora-deploy {__version__}", gpu=gpu,
        certificate=certificate,
    )


def certification(episode: Episode) -> tuple[bool, str | None]:
    """certified only inside NVIDIA's reference envelope, per the EpisodeIndex verdict."""
    if episode.envelope_ok is True:
        return True, None
    if episode.envelope_ok is False:
        return False, "outside_reference_envelope"
    return False, "oracle_ineligible_source"


# ---------------------------------------------------------------------------
# Partial persistence
# ---------------------------------------------------------------------------


def _jsonl(rows: Sequence[Any]) -> bytes:
    lines = []
    for row in rows:
        if isinstance(row, StageRecord):
            lines.append(row.to_json())
        else:
            obj = asdict(row) if dataclasses.is_dataclass(row) else row
            lines.append(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False, default=str))
    return "".join(line + "\n" for line in lines).encode()


@dataclass
class RoundOutput:
    k: int
    config_hash: str
    records: list[StageRecord]
    audits: list[dict[str, Any]]
    errors: list[EpisodeError]


class PartSink(Protocol):
    def write(self, out: RoundOutput) -> list[str]:
        """Persist one round whole; return the published paths."""
        ...


def _write_whole(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class LocalPartSink:
    """Part files per round in ``local_dir`` (``part-<k>.jsonl``, ``.audit.jsonl``,
    ``.errors.jsonl``), each copied whole to ``dest_dir`` (temp name + rename) when given."""

    def __init__(self, local_dir: str | Path, dest_dir: str | Path | None = None,
                 prefix: str = "part"):
        self.local_dir, self.prefix = Path(local_dir), prefix
        self.dest_dir = Path(dest_dir) if dest_dir is not None else None
        self.written: list[str] = []

    def write(self, out: RoundOutput) -> list[str]:
        paths = []
        for suffix, rows in (("", out.records), (".audit", out.audits),
                             (".errors", out.errors)):
            if not rows:
                continue
            name = f"{self.prefix}-{out.k:05d}{suffix}.jsonl"
            local = self.local_dir / name
            _write_whole(local, _jsonl(rows))
            if self.dest_dir is not None:
                target = self.dest_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f".{name}.tmp")
                shutil.copyfile(local, tmp)
                os.replace(tmp, target)
                local = target
            paths.append(str(local))
        self.written += paths
        return paths


class StorePartSink:
    """Local part file per round, then a whole-file ``store.put_bytes`` to the job layout
    (``JobPaths.part`` / ``audit`` / ``errors``)."""

    def __init__(self, store, paths, *, kind: str, shard_id: str, container_id: str,
                 local_dir: str | Path):
        self.store, self.paths, self.kind = store, paths, kind
        self.shard_id, self.container_id = shard_id, container_id
        self.local = LocalPartSink(local_dir, prefix=f"{shard_id}-{container_id}")

    def write(self, out: RoundOutput) -> list[str]:
        local = self.local.write(out)
        targets = {".audit.jsonl": self.paths.audit, ".errors.jsonl": self.paths.errors}
        published = []
        for path in local:
            maker = next((m for s, m in targets.items() if path.endswith(s)), self.paths.part)
            dest = maker(out.config_hash, self.kind, self.shard_id, self.container_id, out.k)
            self.store.put_bytes(dest, Path(path).read_bytes())
            published.append(dest)
        return published


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class RewardRun:
    mode: str
    records: list[StageRecord]
    audits: list[dict[str, Any]]
    errors: list[EpisodeError]
    stats: dict[str, Any]
    parts: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return json.loads(json.dumps({
            "mode": self.mode,
            "records": [json.loads(r.to_json()) for r in self.records],
            "audits": self.audits,
            "errors": [asdict(e) for e in self.errors],
            "stats": self.stats,
            "parts": self.parts,
        }, default=str))


def throughput(stats: dict[str, Any]) -> dict[str, float]:
    seconds = stats["generate_s"] or math.nan
    return {
        "requests_per_s": stats["num_requests"] / seconds,
        "episodes_per_s": stats["num_episodes"] / seconds,
        "prompt_tokens_per_s": stats["prompt_tokens"] / seconds,
        "output_tokens_per_s": stats["output_tokens"] / seconds,
    }


def _reset_prefix_cache(llm) -> None:
    if llm.reset_prefix_cache() is False:
        raise RuntimeError("llm.reset_prefix_cache() returned False (requests still running?)")


@contextlib.contextmanager
def _decoder(nv, kind: str, workers: int) -> Iterator[Executor | None]:
    if kind == "inline":
        yield None
    elif kind == "thread":
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flora-decode") as pool:
            yield pool
    elif kind == "process":
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context(
                "spawn"), **nv.pool_initializer()) as pool:
            yield pool
    else:
        raise ValueError(f"unknown decoder {kind!r}; expected inline, thread or process")


def _prepared_stream(nv, mode: str, episodes: Sequence[Episode], pool: Executor | None,
                     kind: str, lookahead: int) -> Iterator[Prepared]:
    """Prepared episodes in input order, at most ``lookahead`` decoding ahead of the consumer."""
    if pool is None:
        for ep in episodes:
            yield prepare_episode(nv.core, mode, ep)
        return
    if kind == "process":
        def submit(ep):
            return pool.submit(_prepare_in_pool_worker, mode, ep)
    else:
        def submit(ep):
            return pool.submit(prepare_episode, nv.core, mode, ep)
    queue: deque[Future] = deque()
    todo = iter(episodes)
    for ep in todo:
        queue.append(submit(ep))
        if len(queue) >= lookahead:
            break
    while queue:
        future = queue.popleft()
        nxt = next(todo, None)
        if nxt is not None:
            queue.append(submit(nxt))
        try:
            yield future.result()
        except Exception as exc:  # noqa: BLE001 - e.g. a crashed pool worker
            yield Prepared(error={**_failure(exc), "kind": "infra"})


def plan_positions(episodes: Sequence[Episode], planned: Sequence[str] | None, offset: int,
                   group: int, total: int | None = None
                   ) -> tuple[list[int], Callable[[int], tuple[list[str], bool]]]:
    """Planned position of every episode, and ``members(g)``: ``(uids, complete)`` for terminal
    group ``g`` (job positions ``g*group .. (g+1)*group - 1``). ``episodes`` must be a
    subsequence of ``planned``, in planned order (a caller bug otherwise, so it raises).

    ``planned`` covers job positions ``offset .. offset+len(planned)-1`` of a planned order of
    ``total`` episodes (default ``total = offset + len(planned)``: the list runs to the end of
    the job). ``complete`` is False when part of the group lies outside that window (a group
    cut by a shard edge, review 5 M-B): its true members are unknown here, so it is never
    reported as aligned."""
    if offset < 0:
        raise ValueError(f"planned_offset {offset} < 0")
    uids = [e.uid for e in episodes] if planned is None else [str(u) for u in planned]
    end = offset + len(uids)
    total = end if total is None else int(total)
    if total < end:
        raise ValueError(f"planned_total {total} < planned_offset + len(planned) = {end}")
    index: dict[str, int] = {}
    for i, uid in enumerate(uids):
        if uid in index:
            raise ValueError(f"planned episode order lists {uid!r} twice")
        index[uid] = i
    positions = []
    for ep in episodes:
        if ep.uid not in index:
            raise ValueError(f"episode {ep.uid!r} is not in the planned episode order")
        positions.append(offset + index[ep.uid])
    if any(b <= a for a, b in zip(positions, positions[1:], strict=False)):
        raise ValueError("episodes are not in their planned order (NVIDIA groups follow it)")

    def members(g: int) -> tuple[list[str], bool]:
        lo, hi = g * group, min((g + 1) * group, total)
        complete = lo >= offset and hi <= end
        return uids[max(0, lo - offset):max(0, hi - offset)], complete

    return positions, members


def run(llm, nv, processor, episodes: Sequence[Episode], mode: str, *,
        tail_trim_s: float | None = None, spec: EngineSpec | None = None,
        models: dict[str, str] | None = None, gpu: str | None = None, preset: str = "exact",
        certificate: str | None = None, config_extra: dict[str, Any] | None = None,
        limits: RoundLimits | None = None, decoder: str = "thread", workers: int = 4,
        lookahead: int | None = None, sink: PartSink | None = None,
        audit_ref: Callable[[Episode], str] | None = None, shard_id: str = "adhoc",
        container_id: str = "local", clock: Callable[[], float] = time.perf_counter,
        planned: Sequence[str] | None = None, planned_offset: int = 0,
        planned_total: int | None = None,
        ) -> RewardRun:
    """Score ``episodes`` in ``mode`` in size-bounded ``llm.generate`` rounds (module doc).

    ``planned``: the original planned episode uids, of which ``episodes`` is the remaining
    subsequence: the **job's** reward-episode order (the reward worker passes the manifest's
    episode list, review 5 M-B), as NVIDIA's script would read the whole jobs file. A window of
    it is allowed: ``planned_offset`` = its first job position and ``planned_total`` = the
    job's length (default: the window runs to the job's end); a group that the window cuts is
    run with the members it has and flagged ``group_outside_planned`` (never ``aligned``).
    NVIDIA groups are taken from job positions, so a resume or a bisection that drops
    done/failed episodes keeps every other episode's batch-mates, and shard sizing never picks
    them. Without ``planned``, positions are the indices of ``episodes`` (NVIDIA's script run
    on this list)."""
    check_mode(mode)
    episodes = validate_episodes(episodes)
    limits = limits or RoundLimits()
    if spec is not None and limits.align == "nvidia":  # NVIDIA's group is its max_num_seqs
        limits = dataclasses.replace(limits, nvidia_group=int(spec.args["max_num_seqs"]))
    positions, members_of = plan_positions(episodes, planned, planned_offset,
                                           limits.nvidia_group, planned_total)
    if limits.align != "nvidia":  # review 5 L-c: visible in every record, not only the audit
        config_extra = {**(config_extra or {}), "grouping": limits.align}
    anchor = "list" if planned is None else "planned"
    lookahead = lookahead or max(2, 2 * workers)
    stage = STAGES[mode]
    trim = nv.core.DENSE_TAIL_TRIM_S if tail_trim_s is None else tail_trim_s
    base = base_provenance(nv, mode, spec=spec, models=models, gpu=gpu, preset=preset,
                           certificate=certificate, tail_trim_s=trim, extra=config_extra)
    rationale = mode == "terminal_rationale"
    stats: dict[str, Any] = {
        "mode": mode, "generate_calls": 0, "rounds": [], "num_episodes": len(episodes),
        "num_requests": 0, "prompt_tokens": 0, "output_tokens": 0, "decode_s": 0.0,
        "decode_wait_s": 0.0, "generate_s": 0.0, "parse_s": 0.0, "retries": 0,
        "budget_splits": 0,
        "requests": [], "flagged": {}, "limits": asdict(limits), "decoder": decoder,
        "workers": workers if decoder != "inline" else 0, "config_hash": base.config_hash,
        "grouping": {"align": limits.align, "anchor": anchor, "group": limits.nvidia_group,
                     "planned_offset": planned_offset,
                     "planned_total": (planned_offset + len(planned) if planned_total is None
                                       else planned_total) if planned is not None else None,
                     "planned_digest": hashlib.sha256("\n".join(
                         str(u) for u in (planned if planned is not None
                                          else [e.uid for e in episodes])).encode()
                     ).hexdigest()[:16],
                     "unaligned": 0},
    }
    out = RewardRun(mode, [], [], [], stats)
    if not episodes:
        stats["throughput"] = throughput(stats)
        return out
    t_start = clock()
    params = sampling_params(nv, mode)
    tokenizer = llm.get_tokenizer() if mode != "dense" else None

    facts: dict[int, CollageFacts] = {}
    plans: dict[int, Any] = {}
    expected: dict[int, int] = {}
    answers: dict[int, dict[int, Any]] = {}  # episode -> {position: (output | None, usage)}
    failed: dict[int, dict[str, Any]] = {}
    pending = {"records": [], "audits": [], "errors": []}
    batch_of: dict[int, list[str]] = {}  # episode -> uids of the generate call it ran in
    split_groups: set[Any] = set()  # NVIDIA groups a byte-budget split cut in two
    retried: set[int] = set()  # episodes re-run alone after a failed group call
    group_size = limits.nvidia_group

    def grouping(number: int) -> dict[str, Any]:
        pos = positions[number]
        reasons: list[str] = []
        out_g: dict[str, Any] = {"anchor": anchor, "position": pos}
        if mode == "dense":
            out_g["grouping_id"] = f"dense:{pos}"
            out_g["chunks"] = math.ceil(expected[number] / group_size)
            if any(isinstance(g, tuple) and g[0] == pos for g in split_groups):
                reasons.append("budget_split")
        else:
            g = pos // group_size
            members, complete = members_of(g)
            out_g.update(grouping_id=f"terminal:{g}", members=members,
                         batch=batch_of.get(number, []))
            if not complete:
                reasons.append("group_outside_planned")
            if out_g["batch"] != members:
                reasons.append("partial_group")
            if g in split_groups:
                reasons.append("budget_split")
        if number in retried:
            reasons.append("retry_singleton")
        if limits.align != "nvidia":
            reasons = ["packed"]
        out_g["aligned"] = not reasons
        out_g["reasons"] = reasons
        return out_g

    def source_of(ep: Episode, number: int) -> dict[str, dict[str, Any]]:
        duration = round(facts[number].duration_s, 6)
        if not ep.sources:  # ad-hoc episodes: the files NVIDIA read
            return {f"camera_{i}": {"uri": v, "range_s": [0.0, duration]}
                    for i, v in enumerate(ep.videos)}
        out_s = {}
        for role, uri, start, end in ep.sources:
            lo = 0.0 if start is None else float(start)
            hi = lo + duration if end is None else float(end)
            out_s[role] = {"uri": uri, "range_s": [lo, hi]}
        return out_s

    def error(number: int, kind: str, message: str, **diagnostics) -> None:
        ep = episodes[number]
        pending["errors"].append(EpisodeError(
            uid=ep.uid, stage=stage, shard_id=shard_id, container_id=container_id, kind=kind,
            message=message, diagnostics={"episode_id": ep.id, "mode": mode, **diagnostics}))

    def finish(number: int) -> None:
        ep, got = episodes[number], answers.pop(number)
        if number in failed:
            error(number, **failed.pop(number))
            return
        outs = [got[p] for p in sorted(got)]
        usage = [u for _, u in outs]
        finish_reasons = [u.get("finish_reason") for u in usage]
        flags: dict[str, Any] = {}
        try:
            if mode == "dense":
                texts = [o.outputs[0].text for o, _ in outs]
                dense, nv_record = dense_record(nv, ep, facts[number], plans[number], texts, trim)
                truncated = [i for i, r in enumerate(finish_reasons) if r == "length"]
                if truncated:
                    flags["truncated_clips"] = truncated
                if nv_record.get("invalid_clips"):
                    flags["invalid_clips"] = nv_record["invalid_clips"]
                body = {"dense": dense}
            else:
                completion = outs[0][0].outputs[0]
                terminal, nv_record = terminal_record(nv, ep, facts[number], plans[number],
                                                      completion, tokenizer, rationale)
                if finish_reasons[0] == "length":
                    flags["truncated"] = True
                body = {"terminal": terminal}
        except ValueError as exc:  # NVIDIA's parser/stitcher rejected the answer
            error(number, "nv_failure", str(exc), finish_reasons=finish_reasons,
                  outputs=[o.outputs[0].text for o, _ in outs])
            return
        certified, reason = certification(ep)
        if certified and limits.align != "nvidia":  # not NVIDIA's batches: not the reference
            certified, reason = False, "packed_grouping"
        group_info = grouping(number)
        if not group_info["aligned"] and limits.align == "nvidia":
            flags["unaligned"] = group_info["reasons"]
            stats["grouping"]["unaligned"] += 1
        provenance = dataclasses.replace(base, source=source_of(ep, number))
        pending["records"].append(StageRecord(
            uid=ep.uid, dataset=ep.dataset, episode_id=ep.id, stage=stage,
            provenance=provenance, certified=certified, uncertified_reason=reason,
            audit_ref=audit_ref(ep) if audit_ref else None, **body))
        pending["audits"].append({"uid": ep.uid, "episode_id": ep.id, "stage": stage,
                                  "mode": mode, "config_hash": base.config_hash,
                                  "nv": nv_record, "requests": usage, "flags": flags,
                                  "grouping": group_info})
        if flags:
            stats["flagged"][ep.uid] = flags

    def generate(batch: list[Request]) -> list[tuple[Request, Any]]:
        """``llm.generate`` on ``batch``; on an exception, retry per episode (isolation)."""
        _reset_prefix_cache(llm)
        stats["generate_calls"] += 1
        try:
            outputs = llm.generate([r.as_vllm() for r in batch], params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(f"{len(outputs)} outputs for {len(batch)} requests")
            for r, o in zip(batch, outputs, strict=True):
                prompt = getattr(o, "prompt", None)
                if prompt is not None and prompt != r.prompt:
                    raise RuntimeError("vLLM returned outputs out of input order")
            return list(zip(batch, outputs, strict=True))
        except Exception as exc:
            groups: dict[int, list[Request]] = {}
            for r in batch:
                groups.setdefault(r.episode, []).append(r)
            if len(groups) == 1:
                number = batch[0].episode
                failed.setdefault(number, {"kind": "infra", "message": f"generate: {exc!r}"})
                return [(r, None) for r in batch]
            stats["retries"] += 1
            retried.update(groups)  # their batch-mates change: not NVIDIA's call any more
            results, ok = [], 0
            for group in groups.values():
                part = generate(group)
                ok += any(o is not None for _, o in part)
                results += part
            if ok == 0:  # every episode fails alone too: systemic (dead engine), not one input
                raise
            return results

    def flush(k: int, batch: list[Request], wait_s: float) -> None:
        t0 = clock()
        pairs = generate(batch) if batch else []
        t1 = clock()
        uids = list(dict.fromkeys(episodes[r.episode].uid for r in batch))
        for r in batch:
            batch_of[r.episode] = uids
        for r, o in pairs:
            r.multi_modal_data = {}  # drop frame arrays as soon as the engine is done
            usage = {"window": r.window, **(_usage(o) if o is not None else {})}
            answers[r.episode][len(answers[r.episode])] = (o, usage)
            stats["requests"].append({"episode_id": episodes[r.episode].id, **usage})
            if o is not None:
                stats["num_requests"] += 1
                stats["prompt_tokens"] += usage["prompt_tokens"]
                stats["output_tokens"] += usage["output_tokens"]
            if len(answers[r.episode]) == expected[r.episode]:
                finish(r.episode)
        t2 = clock()
        stats["generate_s"] += t1 - t0
        stats["parse_s"] += t2 - t1
        stats["rounds"].append({"k": k, "requests": len(batch),
                                "est_mib": round(sum(r.est_bytes for r in batch) / MiB, 1),
                                "decode_wait_s": wait_s, "generate_s": t1 - t0})
        round_out = RoundOutput(k, base.config_hash, pending["records"], pending["audits"],
                                pending["errors"])
        out.records += round_out.records
        out.audits += round_out.audits
        out.errors += round_out.errors
        if sink is not None and (round_out.records or round_out.errors):
            out.parts += sink.write(round_out)
        pending.update(records=[], audits=[], errors=[])

    k, batch, size, wait = 0, [], 0, 0.0
    with _decoder(nv, decoder, workers) as pool:
        stream = _prepared_stream(nv, mode, episodes, pool, decoder, lookahead)
        for number in range(len(episodes)):
            t0 = clock()
            prepared = next(stream)
            wait += clock() - t0
            stats["decode_s"] += prepared.decode_s
            if prepared.error is not None:
                error(number, prepared.error["kind"], prepared.error["message"],
                      phase="decode", type=prepared.error["type"])
                continue
            try:
                requests = episode_requests(nv, processor, mode, number, episodes[number],
                                            prepared, limits)
            except Exception as exc:  # noqa: BLE001 - e.g. NVIDIA's empty-task ValueError
                f = _failure(exc)
                error(number, f["kind"], f["message"], phase="prompt", type=f["type"])
                continue
            facts[number], plans[number] = prepared.facts, prepared.plan
            expected[number], answers[number] = len(requests), {}
            if limits.align == "nvidia":
                pos = positions[number]
                for j, request in enumerate(requests):
                    request.group = (pos, j // group_size) if mode == "dense" else (
                        pos // group_size)
            for request in requests:
                aligned = limits.boundary(mode, batch, request)
                fits = limits.fits(len(batch), size, request)
                if aligned or not fits:
                    if not fits and not aligned and limits.align == "nvidia":
                        stats["budget_splits"] += 1
                        split_groups.add(request.group)
                    flush(k, batch, wait)
                    k, batch, size, wait = k + 1, [], 0, 0.0
                batch.append(request)
                size += request.est_bytes
        if batch or pending["errors"]:
            flush(k, batch, wait)
    stats["decode_wait_s"] = sum(r["decode_wait_s"] for r in stats["rounds"])
    stats["wall_s"] = clock() - t_start
    stats["throughput"] = throughput(stats)
    return out


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
            "call_sizes": [call["num_requests"] for call in self.calls],
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


def compare(mode: str, reference: Sequence[dict[str, Any]], batched: dict[str, Any],
            reference_probabilities: Sequence[Sequence[float]] | None = None,
            tolerance: float = 1e-6) -> dict[str, Any]:
    """A batched run (``RewardRun.as_json()``) vs NVIDIA's reference lines, by episode id.
    ``reference_probabilities[i]`` are the unrounded P(1..5) of ``reference[i]``."""
    check_mode(mode)
    ref_by_id = {line["id"]: line for line in reference}
    ref_p = ({line["id"]: p for line, p in zip(reference, reference_probabilities, strict=True)}
             if reference_probabilities is not None else None)
    records = {r["episode_id"]: r for r in batched["records"]}
    episodes = []
    for audit in batched["audits"]:
        eid = audit["episode_id"]
        ref = ref_by_id.get(eid)
        if ref is None:
            episodes.append({"episode_id": eid, "missing_reference": True, "identical": False})
            continue
        ours = audit["nv"]
        diff_keys = sorted(k for k in set(ref) | set(ours) if ref.get(k) != ours.get(k))
        entry: dict[str, Any] = {"episode_id": eid, "diff_keys": diff_keys}
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
            if ref_p is not None:
                our_p = records[eid]["terminal"]["probabilities"]
                entry["max_abs_prob_diff"] = max(abs(a - b) for a, b in zip(ref_p[eid], our_p,
                                                                             strict=True))
            entry["identical"] = not diff_keys and (
                entry.get("max_abs_prob_diff", 0.0) <= tolerance
            )
        episodes.append(entry)
    seen = {a["episode_id"] for a in batched["audits"]}
    extra = sorted(set(ref_by_id) - seen)
    errors = [{"episode_id": e["diagnostics"].get("episode_id"), "kind": e["kind"],
               "message": e["message"]} for e in batched.get("errors", [])]
    return {
        "mode": mode,
        "identical": bool(episodes) and all(e["identical"] for e in episodes) and not extra
        and not errors,
        "episodes": episodes,
        "reference_only": extra,
        "errors": errors,
    }


def same_outputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two batched runs (``RewardRun.as_json()``, e.g. rounds vs one unbounded call): records,
    NVIDIA audit lines and errors must be equal, in order. Labels are compared without the
    grouping-dependent provenance (``config_hash``, certification: packed mode marks its records,
    review 5 L-c); ``provenance_equal`` reports whether those match too."""
    def labels(run):
        out = []
        for r in run["records"]:
            r = dict(r)
            r.pop("certified", None)
            r.pop("uncertified_reason", None)
            r["provenance"] = {k: v for k, v in (r.get("provenance") or {}).items()
                               if k != "config_hash"}
            out.append(r)
        return out

    def nv_lines(run):
        return [x["nv"] for x in run["audits"]]

    def errs(run):
        return [(e["uid"], e["kind"], e["message"]) for e in run["errors"]]

    result = {"records_equal": labels(a) == labels(b),
              "provenance_equal": a["records"] == b["records"],
              "nv_equal": nv_lines(a) == nv_lines(b), "errors_equal": errs(a) == errs(b),
              "num_records": [len(a["records"]), len(b["records"])]}
    result["identical"] = all(result[k] for k in ("records_equal", "nv_equal", "errors_equal"))
    return result
