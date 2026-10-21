"""Rubric rewards (``reward.rubric``) on NVIDIA's offline-``LLM`` recipe, one episode per call.

PLAN-v3.md §7 (``RubricWorker``: TP=2, NVIDIA's settings, per-episode files). Everything that
reaches the model comes from NVIDIA's Rewarder ``batch_utils``, linked at runtime and never
modified (D17): the engine arguments are read from NVIDIA's own ``rubric_reward`` argument parser
(:func:`nvidia_recipe`), the prompt from ``core.rubric_header`` / ``rubric_prompt`` /
``rubric_messages`` and ``engine.render(thinking=True, vision_ids=True)``, the videos from
NVIDIA's ``rubric_reward.video_inputs`` (``qwen_vl_utils`` with decord, one video per camera at
``RUBRIC_FPS`` / ``RUBRIC_MAX_FRAMES`` and NVIDIA's pixel budgets), and the answers are read with
``core.split_thinking`` / ``parse_rubric`` / ``medoid``.

NVIDIA's semantics, as implemented:

- one ``llm.generate([request], SamplingParams(n=9, temperature=1.0, top_p=1.0, top_k=0,
  max_tokens=20480))`` per episode, as NVIDIA's script calls it (no seed per request: the
  engine's ``seed=0``), on ``load_llm(..., tensor_parallel_size=2, gpu_memory_utilization=0.85,
  max_model_len=98304, videos_per_prompt=<cameras>, mm_processor_kwargs={"do_resize": False},
  max_num_batched_tokens=24576)``;
- every sample is split at ``</think>`` and parsed against the rubric (missing steps score 0,
  values are clamped to ``[0, max]``; a wrong-length answer is ``parsed: false``);
- the reported scores are the **medoid** of the parsed samples: NVIDIA's ``core.medoid`` (the
  smallest summed L1 distance to all parsed samples, ties to the lowest index), mapped back to
  the sample index (:func:`medoid_sample`); ``total`` is NVIDIA's ``round(sum(scores), 4)``;
- no parsed sample: NVIDIA writes ``scores: null``. Here NVIDIA's own ``core.medoid`` raises
  ("no parsed rubric answers") and the episode becomes an ``EpisodeError(kind="nv_failure")``
  carrying NVIDIA's line: NVIDIA's verdict, terminal and excused.

Sampling at temperature 1 makes outputs a distribution: ``exact`` reproduces NVIDIA's requests
(inputs) identically and its outputs only in distribution (PLAN-v3.md §2, "the same distribution
for rubric").

What is ours (re-expressed to NVIDIA's interface, not copied: ``rubric_reward.py`` is under
NVIDIA's license): the job line built from the config (:meth:`RubricSpec.job`), the ~6 lines of
``prepare`` / ``SamplingParams`` / ``load_llm`` glue of NVIDIA's ``main``, the sample dicts and
NVIDIA's output line (:func:`rubric_record`). ``tests/nvlink/test_rubric_identity.py`` runs
NVIDIA's unmodified ``rubric_reward.main`` with the model replaced and compares every request,
``SamplingParams``, ``load_llm`` argument and output line; it is the drift guard.

Failure isolation per episode uses ``nv.errors.classify``: NVIDIA's verdict (``nv_failure``) is
recorded and excused; ``infra`` / ``python`` become ``kind="infra"`` errors (the worker retries
the shard); a CUDA fault or OOM raises :class:`RubricCrash` (the container is replaced, D15).
Each finished episode is handed to the ``PartSink`` at once (a preemption loses at most the
episode in flight); the next episode's request is built on a thread while one generates.

Prefix caching: NVIDIA's engine has it on; the 9 samples of one request share their prompt
blocks there too. ``llm.reset_prefix_cache()`` runs before every episode, so a retry in a warm
container never hits an earlier episode's blocks.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flora_deploy.jobs.types import EpisodeError
from flora_deploy.records import Provenance, RubricReward, StageRecord
from flora_deploy.stages import reward as R

MODE = "rubric"
STAGE = "reward.rubric"
ENGINE_LABEL = R.ENGINE_LABEL
LOAD_LLM_ARGS = R.LOAD_LLM_ARGS
# NVIDIA's rubric script reads these from its own parser (``rubric_reward.main``).
SAMPLING_ARGS = ("num_samples", "temperature", "max_new_tokens")
FRAME_ARGS = ("fps", "max_frames")
ENGINE_ARGS = ("max_model_len", "max_num_batched_tokens")
# The one ``mm_processor_kwargs`` NVIDIA's rubric ``load_llm`` call passes.
MM_PROCESSOR_KWARGS = {"do_resize": False}
CRASH_KINDS = ("cuda_fatal", "oom")


# ---------------------------------------------------------------------------
# The job (config) and NVIDIA's recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricSpec:
    """``pipeline.rubric`` (``config.schema.rubric_spec``): NVIDIA's rubric job fields, minus the
    per-episode ``id`` / ``task`` / ``videos``. ``roles``: the camera role read for Video 1..N."""

    steps: tuple[dict[str, Any], ...]
    roles: tuple[str, ...]
    robot: str | None = None
    cameras: tuple[str, ...] | None = None  # descriptions, in ``roles`` order
    header: str | None = None
    task_name: str | None = None

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> RubricSpec:
        if not d or not d.get("steps") or not d.get("cameras"):
            raise ValueError("rubric spec needs `steps` and `cameras` (pipeline.rubric)")
        cams = list(d["cameras"])
        header = d.get("header")
        return cls(steps=tuple(json.loads(json.dumps(s)) for s in d["steps"]),
                   roles=tuple(str(c["role"]) for c in cams), robot=d.get("robot"),
                   cameras=None if header else tuple(str(c["description"]) for c in cams),
                   header=header, task_name=d.get("task_name"))

    def as_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def job(self, episode: R.Episode) -> dict[str, Any]:
        """The line NVIDIA's ``--jobs`` file would hold for ``episode`` (videos in role order,
        as :func:`episode_for_roles` puts them)."""
        if len(episode.videos) != len(self.roles):
            raise ValueError(f"episode {episode.uid!r}: {len(episode.videos)} videos for "
                             f"{len(self.roles)} rubric cameras")
        job: dict[str, Any] = {"id": episode.id, "task": episode.task,
                               "videos": list(episode.videos),
                               "rubric": [dict(s) for s in self.steps]}
        if self.header:
            job["header"] = self.header
        else:
            job["robot"], job["cameras"] = self.robot, list(self.cameras or ())
        if self.task_name:
            job["task_name"] = self.task_name
        return job


class RubricCamerasMismatch(RuntimeError):
    """The episode's camera roles are not the rubric's: a configuration error, never excused."""


def episode_for_roles(spec: RubricSpec, roles: Sequence[str]) -> tuple[str, ...]:
    """The episode's roles in rubric video order; refuses a different set of cameras."""
    if set(roles) != set(spec.roles) or len(roles) != len(spec.roles):
        raise RubricCamerasMismatch(
            f"episode cameras {sorted(roles)} != pipeline.rubric.cameras {list(spec.roles)}: "
            "NVIDIA's rubric prompt describes every video it is shown, in order")
    return spec.roles


@dataclass(frozen=True)
class Recipe:
    """NVIDIA's rubric settings: its ``load_llm`` call (``engine``), sampling and frame rate."""

    engine: R.EngineSpec
    num_samples: int
    temperature: float
    max_new_tokens: int
    fps: float
    max_frames: int

    def sampling_json(self) -> dict[str, Any]:
        return {"n": self.num_samples, "temperature": self.temperature, "top_p": 1.0,
                "top_k": 0, "max_tokens": self.max_new_tokens}

    def as_json(self) -> dict[str, Any]:
        return {"engine": self.engine.as_json(), "sampling": self.sampling_json(),
                "frames": {"fps": self.fps, "max_frames": self.max_frames}}


class _Parsed(Exception):  # noqa: N818 - control flow, never escapes nvidia_args
    def __init__(self, args: argparse.Namespace):
        super().__init__("parsed")
        self.namespace = args


def nvidia_args(nv, model_path: str) -> argparse.Namespace:
    """NVIDIA's ``rubric_reward.main`` argument defaults, read from its own parser: ``main`` runs
    until ``parse_args`` returns (NVIDIA's ``base_parser`` is wrapped for the call only), so
    the defaults (``--num-samples 9``, ``--tensor-parallel-size 2``, ...) are NVIDIA's, never
    copies."""
    module = nv.rubric_reward
    original = module.base_parser

    def base_parser(*a, **kw):
        parser = original(*a, **kw)
        parse = parser.parse_args

        def parse_args(argv=None, namespace=None):
            raise _Parsed(parse(argv, namespace))

        parser.parse_args = parse_args
        return parser

    module.base_parser = base_parser
    try:
        module.main(["--jobs", "unused.jsonl", "--output", "unused.jsonl", "--model",
                     str(model_path)])
    except _Parsed as parsed:
        return parsed.namespace
    finally:
        module.base_parser = original
    raise RuntimeError("NVIDIA's rubric_reward.main returned before parsing its arguments")


def nvidia_recipe(nv, model_path: str, videos_per_prompt: int) -> Recipe:
    """NVIDIA's rubric ``load_llm`` call and sampling for jobs of ``videos_per_prompt`` cameras
    (NVIDIA passes ``max(len(job["videos"]))`` over its jobs file; one rubric config has one
    camera count)."""
    a = nvidia_args(nv, model_path)
    engine = R.EngineSpec(str(model_path), {k: getattr(a, k) for k in LOAD_LLM_ARGS}, {
        "max_model_len": a.max_model_len,
        "videos_per_prompt": int(videos_per_prompt),
        "mm_processor_kwargs": dict(MM_PROCESSOR_KWARGS),
        "max_num_batched_tokens": a.max_num_batched_tokens,
    })
    return Recipe(engine, int(a.num_samples), float(a.temperature), int(a.max_new_tokens),
                  float(a.fps), int(a.max_frames))


def build_engine(nv, model_path: str, videos_per_prompt: int):
    """The rubric ``vllm.LLM`` through NVIDIA's ``engine.load_llm``. Returns (llm, recipe)."""
    recipe = nvidia_recipe(nv, model_path, videos_per_prompt)
    args = argparse.Namespace(**recipe.engine.args)
    llm = nv.engine.load_llm(recipe.engine.model_path, args, **recipe.engine.kwargs)
    return llm, recipe


@contextlib.contextmanager
def shared_engine(module, llm, recipe: Recipe) -> Iterator[list[dict[str, Any]]]:
    """Let NVIDIA's ``rubric_reward.main`` reuse ``llm``: its ``load_llm`` is replaced for the
    duration by a check that the requested engine equals ``recipe.engine`` (raises
    ``stages.reward.EngineMismatchError``). Yields the calls it saw."""
    spec, calls = recipe.engine, []
    original = module.load_llm

    def load_llm(path, args, *, max_model_len, videos_per_prompt=1, mm_processor_kwargs=None,
                 **engine_kwargs):
        requested = {"max_model_len": max_model_len, "videos_per_prompt": videos_per_prompt,
                     "mm_processor_kwargs": mm_processor_kwargs, **engine_kwargs}
        calls.append({"path": str(path), **{k: getattr(args, k) for k in LOAD_LLM_ARGS},
                      **requested})
        problems = []
        if Path(path).resolve() != Path(spec.model_path).resolve():
            problems.append(f"model {path} != {spec.model_path}")
        problems += [f"{k} {getattr(args, k)!r} != {spec.args[k]!r}" for k in LOAD_LLM_ARGS
                     if getattr(args, k) != spec.args[k]]
        problems += [f"{k} {requested.get(k)!r} != {spec.kwargs.get(k)!r}"
                     for k in sorted(set(requested) | set(spec.kwargs))
                     if requested.get(k) != spec.kwargs.get(k)]
        if problems:
            raise R.EngineMismatchError("; ".join(problems))
        return llm

    module.load_llm = load_llm
    try:
        yield calls
    finally:
        module.load_llm = original


def sampling_params(recipe: Recipe):
    """NVIDIA's rubric ``SamplingParams`` (re-expressed from its script)."""
    from vllm import SamplingParams

    return SamplingParams(n=recipe.num_samples, temperature=recipe.temperature, top_p=1.0,
                          top_k=0, max_tokens=recipe.max_new_tokens)


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


def episode_messages(nv, spec: RubricSpec, recipe: Recipe, episode: R.Episode) -> list[dict]:
    """NVIDIA's ``prepare``: header, prompt and chat messages for the episode's job line."""
    core, job = nv.core, spec.job(episode)
    header = job.get("header") or core.rubric_header(job["robot"], job["cameras"])
    prompt = core.rubric_prompt(job["task"], job["rubric"], header, job.get("task_name"))
    return core.rubric_messages(prompt, job["videos"], fps=recipe.fps,
                                max_frames=recipe.max_frames)


def episode_request(nv, processor, spec: RubricSpec, recipe: Recipe, episode: R.Episode
                    ) -> dict[str, Any]:
    """The request NVIDIA's script passes to ``llm.generate`` for the episode: the rendered
    prompt (thinking on, vision ids on) and NVIDIA's ``video_inputs(messages)``."""
    messages = episode_messages(nv, spec, recipe, episode)
    text = nv.engine.render(processor, messages, thinking=True, vision_ids=True)
    return {"prompt": text, "multi_modal_data": {"video": nv.rubric_reward.video_inputs(messages)}}


def medoid_sample(core, samples: Sequence[Mapping[str, Any]]) -> int:
    """NVIDIA's choice: the parsed sample that is ``core.medoid`` of the parsed score vectors
    (index into ``samples``). Raises NVIDIA's ``ValueError`` when no sample parsed."""
    parsed = [i for i, s in enumerate(samples) if s["parsed"]]
    return parsed[core.medoid([samples[i]["scores"] for i in parsed])]


def rubric_record(nv, spec: RubricSpec, episode: R.Episode, output
                  ) -> tuple[RubricReward | None, dict[str, Any], BaseException | None]:
    """NVIDIA's per-sample parse, the medoid and NVIDIA's output line. Returns
    ``(reward, nv_line, failure)``: ``failure`` is NVIDIA's exception when no sample parsed
    (``reward`` is then ``None`` and the line has NVIDIA's ``scores: null``)."""
    core = nv.core
    rubric = [dict(s) for s in spec.steps]
    samples = []
    for completion in output.outputs:
        answer, _thinking = core.split_thinking(completion.text)
        scores, ok = core.parse_rubric(answer, rubric)
        samples.append({"scores": scores, "parsed": ok, "answer": answer,
                        "generated_tokens": len(completion.token_ids)})
    parsed = sum(1 for s in samples if s["parsed"])
    failure: BaseException | None = None
    try:
        best: int | None = medoid_sample(core, samples)
    except Exception as exc:  # noqa: BLE001 - NVIDIA's verdict, classified by the caller
        best, failure = None, exc
    scores = samples[best]["scores"] if best is not None else None
    line = {
        "id": episode.id,
        "task": episode.task,
        "steps": [step["action"] for step in rubric],
        "scores": scores,
        "total": round(sum(scores), 4) if scores is not None else None,
        "medoid_sample": best,
        "parsed_samples": parsed,
        "samples": samples,
        "prompt_tokens": len(output.prompt_token_ids or ()),
    }
    if best is None:
        return None, line, failure
    reward = RubricReward(scores=tuple(float(s) for s in scores), total=float(line["total"]),
                          samples=len(samples), medoid_index=int(best), parsed_samples=parsed)
    return reward, line, None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def config_hash(recipe: Recipe | None, spec: RubricSpec, *, models: Mapping[str, str],
                extra: Mapping[str, Any] | None = None) -> str:
    """sha256 over what changes the outputs (D11): NVIDIA's engine call, sampling and frame
    rate, the rubric job fields, model pins, logic version and ``extra``."""
    from flora_deploy import LOGIC_VERSION

    blob = {"stage": STAGE, "mode": MODE, "engine_label": ENGINE_LABEL,
            "recipe": recipe.as_json() if recipe else None, "rubric": spec.as_json(),
            "models": dict(models), "logic_version": LOGIC_VERSION, "extra": dict(extra or {})}
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _source(ep: R.Episode, video: Sequence[Any] | None) -> dict[str, dict[str, Any]]:
    """Per camera: the source uri and the range NVIDIA read (its video metadata's duration)."""
    durations = [float((m or {}).get("duration") or 0.0) for _f, m in (video or ())]
    out: dict[str, dict[str, Any]] = {}
    if not ep.sources:
        for i, v in enumerate(ep.videos):
            d = durations[i] if i < len(durations) else 0.0
            out[f"camera_{i}"] = {"uri": v, "range_s": [0.0, round(d, 6)]}
        return out
    for i, (role, uri, start, end) in enumerate(ep.sources):
        lo = 0.0 if start is None else float(start)
        d = durations[i] if i < len(durations) else 0.0
        out[role] = {"uri": uri, "range_s": [lo, lo + round(d, 6) if end is None else float(end)]}
    return out


class RubricCrash(RuntimeError):
    """A CUDA fault / OOM on an episode: the engine is unusable, the container is replaced."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.crash_kind = kind
        self.crash_message = message


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class _Built:
    request: dict[str, Any] | None = None
    error: BaseException | None = None
    decode_s: float = 0.0
    meta: list[tuple[None, dict[str, Any]]] = field(default_factory=list)  # per camera


def _build(nv, processor, spec, recipe, ep) -> _Built:
    t0 = time.perf_counter()
    try:
        request = episode_request(nv, processor, spec, recipe, ep)
        meta = [(None, dict(m)) for _f, m in request["multi_modal_data"]["video"]]
        return _Built(request, None, time.perf_counter() - t0, meta)
    except Exception as exc:  # noqa: BLE001 - isolated per episode
        return _Built(None, exc, time.perf_counter() - t0)


def run(llm, nv, processor, episodes: Sequence[R.Episode], spec: RubricSpec, *,
        recipe: Recipe | None = None, models: Mapping[str, str] | None = None,
        gpu: str | None = None, preset: str = "exact", certificate: str | None = None,
        config_extra: Mapping[str, Any] | None = None, sink: R.PartSink | None = None,
        shard_id: str = "adhoc", container_id: str = "local", prefetch: bool = True,
        classify: Callable[[BaseException], str] | None = None,
        clock: Callable[[], float] = time.perf_counter) -> R.RewardRun:
    """Score ``episodes`` with NVIDIA's rubric recipe, one ``generate`` per episode (module
    docstring). ``recipe``: NVIDIA's settings (default: :func:`nvidia_recipe` on ``nv`` with
    the rubric's camera count). ``classify``: ``nv.errors.classify`` (injectable)."""
    if classify is None:
        from flora_deploy.nv.errors import classify
    episodes = R.validate_episodes(episodes)
    recipe = recipe or nvidia_recipe(nv, "", len(spec.roles))
    models = dict(models or {})
    from flora_deploy import LOGIC_VERSION, __version__

    base = Provenance(models=models, engine=ENGINE_LABEL, preset=preset,
                      config_hash=config_hash(recipe, spec, models=models, extra=config_extra),
                      logic_version=LOGIC_VERSION, framework=f"flora-deploy {__version__}",
                      gpu=gpu, certificate=certificate)
    stats: dict[str, Any] = {
        "mode": MODE, "generate_calls": 0, "num_episodes": len(episodes), "num_requests": 0,
        "num_samples": 0, "prompt_tokens": 0, "output_tokens": 0, "decode_s": 0.0,
        "decode_wait_s": 0.0, "generate_s": 0.0, "parse_s": 0.0, "requests": [],
        "flagged": {}, "config_hash": base.config_hash, "recipe": recipe.as_json(),
    }
    out = R.RewardRun(MODE, [], [], [], stats)
    if not episodes:
        stats["throughput"] = R.throughput(stats)
        return out
    t_start = clock()
    params = sampling_params(recipe)

    def error(ep: R.Episode, exc: BaseException | None, phase: str, *, kind: str | None = None,
              message: str | None = None, **diagnostics) -> EpisodeError:
        k = kind or classify(exc)  # type: ignore[arg-type]
        msg = message if message is not None else f"{type(exc).__name__}: {exc}"
        if k in CRASH_KINDS:
            raise RubricCrash(k, f"episode {ep.uid}: {msg}") from exc
        return EpisodeError(
            uid=ep.uid, stage=STAGE, shard_id=shard_id, container_id=container_id,
            kind="nv_failure" if k == "nv_failure" else "infra",
            message=str(exc) if k == "nv_failure" and message is None else msg,
            diagnostics={"episode_id": ep.id, "mode": MODE, "phase": phase,
                         "classified": k, **({"type": type(exc).__name__} if exc else {}),
                         **diagnostics})

    def emit(k: int, records=(), audits=(), errors=()) -> None:
        ro = R.RoundOutput(k, base.config_hash, list(records), list(audits), list(errors))
        out.records += ro.records
        out.audits += ro.audits
        out.errors += ro.errors
        if sink is not None and (ro.records or ro.errors):
            out.parts += sink.write(ro)

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flora-rubric") \
        if prefetch else None
    try:
        nxt: Future | None = None
        for k, ep in enumerate(episodes):
            t0 = clock()
            if nxt is not None:
                built = nxt.result()
            else:
                built = _build(nv, processor, spec, recipe, ep)
            stats["decode_wait_s"] += clock() - t0
            stats["decode_s"] += built.decode_s
            nxt = pool.submit(_build, nv, processor, spec, recipe, episodes[k + 1]) \
                if pool is not None and k + 1 < len(episodes) else None
            if built.error is not None:
                emit(k, errors=[error(ep, built.error, "inputs")])
                continue
            request, built.request = built.request, None
            R._reset_prefix_cache(llm)
            stats["generate_calls"] += 1
            t1 = clock()
            try:
                outputs = llm.generate([request], params, use_tqdm=False)
                if len(outputs) != 1 or len(outputs[0].outputs) != recipe.num_samples:
                    raise RuntimeError(f"generate returned {len(outputs)} output(s) / "
                                       f"{[len(o.outputs) for o in outputs]} samples for one "
                                       f"request with n={recipe.num_samples}")
            except Exception as exc:  # noqa: BLE001 - classified: infra, or a crash
                emit(k, errors=[error(ep, exc, "generate", message=f"generate: {exc!r}")])
                continue
            finally:
                stats["generate_s"] += clock() - t1
                request = None  # drop the frames as soon as the engine is done
            output = outputs[0]
            finish = [getattr(c, "finish_reason", None) for c in output.outputs]
            usage = {"prompt_tokens": len(output.prompt_token_ids or ()),
                     "output_tokens": sum(len(c.token_ids) for c in output.outputs),
                     "finish_reasons": finish}
            stats["requests"].append({"episode_id": ep.id, **usage})
            stats["num_requests"] += 1
            stats["num_samples"] += len(output.outputs)
            stats["prompt_tokens"] += usage["prompt_tokens"]
            stats["output_tokens"] += usage["output_tokens"]
            t2 = clock()
            try:
                reward, line, failure = rubric_record(nv, spec, ep, output)
            except Exception as exc:  # noqa: BLE001 - NVIDIA's parser raised: classified
                emit(k, errors=[error(ep, exc, "parse", finish_reasons=finish)])
                continue
            finally:
                stats["parse_s"] += clock() - t2
            if failure is not None or reward is None:
                emit(k, errors=[error(ep, failure, "medoid", nv=line, finish_reasons=finish,
                                      outputs=[c.text for c in output.outputs])])
                continue
            flags: dict[str, Any] = {}
            truncated = [i for i, r in enumerate(finish) if r == "length"]
            if truncated:
                flags["truncated_samples"] = truncated
            unparsed = [i for i, s in enumerate(line["samples"]) if not s["parsed"]]
            if unparsed:
                flags["unparsed_samples"] = unparsed
            certified, reason = R.certification(ep)
            provenance = dataclasses.replace(base, source=_source(ep, built.meta))
            rec = StageRecord(uid=ep.uid, dataset=ep.dataset, episode_id=ep.id, stage=STAGE,
                              provenance=provenance, certified=certified,
                              uncertified_reason=reason, rubric=reward)
            audit = {"uid": ep.uid, "episode_id": ep.id, "stage": STAGE, "mode": MODE,
                     "config_hash": base.config_hash, "nv": line, "requests": [usage],
                     "flags": flags}
            if flags:
                stats["flagged"][ep.uid] = flags
            emit(k, records=[rec], audits=[audit])
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    stats["wall_s"] = clock() - t_start
    stats["throughput"] = R.throughput(stats)
    return out


__all__ = ["MODE", "STAGE", "Recipe", "RubricCamerasMismatch", "RubricCrash", "RubricSpec",
           "build_engine", "config_hash", "episode_for_roles", "episode_messages",
           "episode_request", "medoid_sample", "nvidia_args", "nvidia_recipe", "rubric_record",
           "run", "sampling_params", "shared_engine"]
