"""``balanced`` segment stage: NVIDIA's segmenter on vLLM through the micro-batcher (PLAN.md §6.3).

NVIDIA's code does all of the work except running the model:

- inputs: NVIDIA's ``api.DataEnginePipeline._inputs`` (``media.probe`` per camera, the size,
  profile, wrist and sync checks), then the pool ``pipeline.annotate`` picks;
- the segmenter: built as ``pipeline.segment`` builds it (``nv.segment_glue``: the inline
  3-view construction or ``adapter.production_segmenter``), with the predictor below;
- per window: NVIDIA's own ``pipeline.Predictor.predict`` (window from payload, frame plan,
  ``frames_for``, ``human_transport``, ``temporal_inputs``), unchanged.

``MicroBatchPredictor`` subclasses NVIDIA's ``Predictor`` and adds the ``apredict`` hook that
NVIDIA's ``FineSubtaskSegmenter`` prefers. ``apredict(request)`` runs NVIDIA's ``predict`` in a
worker thread with ``pipeline.generate`` tapped (``GenerateTap``): instead of running HF, the tap
takes the call's ``CapturedRequest`` from the ``CaptureProcessor`` NVIDIA's builders called and
returns a unique sentinel string. NVIDIA's ``predict`` therefore returns its response with one
sentinel per window. The captured requests (Gate C wrap, ``nv.capture.vllm_prompt``) go to the
micro-batcher together, so all windows of all live episodes share ``llm.generate`` rounds. Each
completion is then accepted by NVIDIA's rules, **in NVIDIA's call order**:

1. NVIDIA's 32k guard on vLLM's prompt length (``CapturedRequest.check_context``, which raises
   an ``nv.budget.ContextBudgetError`` that is also NVIDIA's ``InputErrorForModel``); a prompt
   vLLM rejected as longer than ``max_model_len`` is the same failure;
2. EOS reached and ``tokenizer.decode(ids, skip_special_tokens=True).strip()`` non-empty, else
   NVIDIA's ``InputErrorForModel`` (``models.generate`` does exactly this).

The first failing call raises, as NVIDIA's sequential ``predict`` would; otherwise the sentinels
in the response and in ``Predictor.calls`` (NVIDIA's temporal audit) are replaced by the texts
and the response goes back to NVIDIA's segmenter, which parses, stitches, retries (up to two
more ``apredict`` rounds) and repairs exactly as on its own sync path. Any NVIDIA exception
fails the episode, as in ``pipeline.segment``. Engine (infrastructure) failures raise
``EngineRequestError``.

LICENSE REVIEW FLAG (PLAN.md §11): ``pool_for`` restates one line of ``pipeline.annotate``
(``'yam32' if long_windows else PROFILE_POOLS[profile]``), and ``segment_episode`` the shape of
``pipeline.segment`` (source, binding, segmenter, the returned rows). Everything else calls
NVIDIA's functions. ``tests/nvlink/test_balanced_nvlink.py`` checks the result against NVIDIA's
own ``pipeline.segment`` for canned model outputs.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import itertools
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.engines.offline_llm import nvidia_sampling
from flora_deploy.nv.capture import (
    VIDEO_TOKEN,
    VISION_END,
    VISION_START,
    CapturedRequest,
    CaptureError,
    CaptureProcessor,
    vllm_prompt,
)

SENTINEL_PREFIX = "\x00flora-generate:"
_SENTINEL_IDS = itertools.count()


class EngineRequestError(RuntimeError):
    """The engine failed this request (infrastructure; never NVIDIA semantics)."""


# ---------------------------------------------------------------------------------------------
# The generate tap
# ---------------------------------------------------------------------------------------------


@dataclass
class TappedCall:
    captured: CapturedRequest
    budget: int
    sentinel: str


@dataclass
class Collector:
    """The generate calls one NVIDIA function made in the current context, in call order."""

    calls: list[TappedCall] = field(default_factory=list)
    procs: list[Any] = field(default_factory=list)  # CaptureProcessors seen by the tap

    def add(self, captured: CapturedRequest, budget: int) -> str:
        sentinel = f"{SENTINEL_PREFIX}{next(_SENTINEL_IDS)}\x00"
        self.calls.append(TappedCall(captured, int(budget), sentinel))
        return sentinel


_COLLECTOR: contextvars.ContextVar[Collector | None] = contextvars.ContextVar(
    "flora_generate_collector", default=None)


def run_collecting(collector: Collector, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Run ``fn`` with ``pipeline.generate`` calls recorded into ``collector`` (this context).
    Every processor call must have been paired with a ``generate`` (review 6 L2): a leftover
    captured request means FIFO pairing could put the wrong prompt under a caption, so it
    raises ``CaptureError``."""
    token = _COLLECTOR.set(collector)
    try:
        result = fn(*args, **kwargs)
    finally:
        _COLLECTOR.reset(token)
    left = sum(len(p.requests) for p in collector.procs)
    if left:
        raise CaptureError(f"{left} unpaired processor call(s) left after "
                           f"{len(collector.calls)} generate call(s)")
    return result


class GenerateTap:
    """Replaces ``nv.pipeline.generate`` (the name NVIDIA's ``Predictor.predict`` and
    ``caption_*`` call) with a dispatcher: inside ``run_collecting`` it records the captured
    request and returns a sentinel; anywhere else it calls NVIDIA's original. Reference-counted
    per family handle; ``release()`` restores NVIDIA's function when the last user leaves."""

    _lock = threading.Lock()
    _active: dict[int, GenerateTap] = {}

    def __init__(self, nv):
        self.nv = nv
        self.original = nv.pipeline.generate
        self.users = 0

        def generate(model, proc, inputs, budget):
            collector = _COLLECTOR.get()
            if collector is None:
                return self.original(model, proc, inputs, budget)
            if not isinstance(proc, CaptureProcessor):
                raise CaptureError("tapped generate: the processor is not a CaptureProcessor")
            if type(inputs).__name__ != "_CapturedInputs":
                raise CaptureError(f"tapped generate: unexpected inputs {type(inputs)!r}")
            if not any(p is proc for p in collector.procs):
                collector.procs.append(proc)
            return collector.add(proc.pop_request(), budget)

        generate.__flora_tap__ = True  # type: ignore[attr-defined]
        self.replacement = generate

    @classmethod
    def acquire(cls, nv) -> GenerateTap:
        with cls._lock:
            tap = cls._active.get(id(nv))
            if tap is None:
                if getattr(nv.pipeline.generate, "__flora_tap__", False):
                    raise RuntimeError("pipeline.generate is tapped by another handle")
                tap = cls._active[id(nv)] = GenerateTap(nv)
                nv.pipeline.generate = tap.replacement
            tap.users += 1
            return tap

    def release(self) -> None:
        with GenerateTap._lock:
            self.users -= 1
            if self.users > 0:
                return
            GenerateTap._active.pop(id(self.nv), None)
            if self.nv.pipeline.generate is self.replacement:
                self.nv.pipeline.generate = self.original


# ---------------------------------------------------------------------------------------------
# Requests and acceptance
# ---------------------------------------------------------------------------------------------


def unwrap_prompt(prompt: str) -> str:
    """Inverse of the Gate C wrap (``nv.capture.wrap_bare_video_tokens``)."""
    return prompt.replace(VISION_START + VIDEO_TOKEN + VISION_END, VIDEO_TOKEN)


def request_digest(prompt: str, videos: Sequence[tuple[Any, Mapping]]) -> str:
    """Content digest of a model call: bare prompt, every frame byte and the metadata."""
    h = hashlib.sha256(unwrap_prompt(prompt).encode())
    for frames, md in videos:
        h.update(str(frames.shape).encode())
        h.update(np.ascontiguousarray(frames))
        h.update(json.dumps(dict(md), sort_keys=True, default=str).encode())
    return h.hexdigest()


def to_vllm_request(captured: CapturedRequest, n: int) -> VllmRequest:
    d = vllm_prompt(captured)
    videos = tuple((frames, md) for frames, md in d["multi_modal_data"]["video"])
    return VllmRequest(prompt=d["prompt"], videos=videos,
                       key=f"{request_digest(d['prompt'], videos)[:24]}#{n}")


def incomplete_error(nv, why: str) -> Exception:
    """NVIDIA's ``InputErrorForModel`` (what ``models.generate`` raises for an incomplete or
    empty output); a plain ``RuntimeError`` subclass when no family is linked (fakes)."""
    cls = getattr(getattr(nv, "models", None), "InputErrorForModel", None)
    msg = f"The model did not return a complete output for this input ({why})."
    return cls(msg) if isinstance(cls, type) else IncompleteOutput(msg)


class IncompleteOutput(RuntimeError):
    pass


def stopped_on_eos(completion: Completion, eos: int) -> bool:
    ids = completion.token_ids
    if ids and ids[-1] == eos:
        return True
    return completion.finish_reason == "stop" and completion.stop_reason in (None, eos)


def accept(nv, call: TappedCall, completion: Completion, tokenizer, eos: int) -> str:
    """NVIDIA's ``models.generate`` acceptance for one call, on the engine's completion."""
    if completion.error is not None:
        if completion.too_long:
            from flora_deploy.nv.budget import error_class

            raise error_class(nv)(
                f"vLLM rejected the prompt as longer than max_model_len: {completion.error}",
                prompt_tokens=-1, budget=call.budget, limit=-1)
        raise EngineRequestError(completion.error)
    if completion.prompt_tokens is None:
        raise EngineRequestError("engine returned no prompt length")
    call.captured.check_context(completion.prompt_tokens, nv=nv)
    ids = list(completion.token_ids)
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    if not stopped_on_eos(completion, eos):
        raise incomplete_error(nv, f"no EOS within {call.budget} tokens, "
                                   f"finish_reason={completion.finish_reason}")
    if not text:
        raise incomplete_error(nv, "empty output")
    return text


@dataclass
class CallStats:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    rounds: int = 0

    def add(self, completions: Sequence[Completion]) -> None:
        self.rounds += 1
        for c in completions:
            self.calls += 1
            self.prompt_tokens += c.prompt_tokens or 0
            self.output_tokens += len(c.token_ids)

    def as_json(self) -> dict:
        return dict(self.__dict__)


async def run_tapped(nv, batcher, fn: Callable[[], Any], *, tokenizer, eos: int,
                     sampling_for: Callable[[TappedCall], Sampling] | None = None,
                     stats: CallStats | None = None) -> tuple[Any, dict[str, str]]:
    """Run NVIDIA function ``fn`` in a worker thread with generate tapped, send its calls to
    ``batcher`` together, accept the completions in call order. Returns ``(fn's result,
    {sentinel: text})``; raises the first NVIDIA failure."""
    collector = Collector()

    def collect() -> tuple[Any, list[VllmRequest]]:
        out = run_collecting(collector, fn)
        # building the requests hashes every frame (request_digest): off the loop (review 6 L5)
        return out, [to_vllm_request(c.captured, i) for i, c in enumerate(collector.calls)]

    result, requests = await asyncio.to_thread(collect)
    if not collector.calls:
        return result, {}
    if sampling_for is None:
        samplings = [nvidia_sampling(c.budget, eos) for c in collector.calls]
    else:
        samplings = [sampling_for(c) for c in collector.calls]
    completions = await batcher.generate_many(requests, samplings)
    if stats is not None:
        stats.add(completions)
    texts = {}
    for call, comp in zip(collector.calls, completions, strict=True):
        texts[call.sentinel] = accept(nv, call, comp, tokenizer, eos)
    return result, texts


def substitute(value: Any, texts: Mapping[str, str]) -> Any:
    if isinstance(value, str) and value.startswith(SENTINEL_PREFIX):
        if value not in texts:
            raise RuntimeError("unresolved generate sentinel")
        return texts[value]
    return value


# ---------------------------------------------------------------------------------------------
# The predictor
# ---------------------------------------------------------------------------------------------

_PREDICTOR_CLASSES: dict[int, type] = {}


def predictor_class(nv) -> type:
    """``MicroBatchPredictor``: a subclass of NVIDIA's ``pipeline.Predictor`` with ``apredict``."""
    base = nv.pipeline.Predictor
    cls = _PREDICTOR_CLASSES.get(id(base))
    if cls is not None:
        return cls

    class MicroBatchPredictor(base):  # type: ignore[misc, valid-type]
        """NVIDIA's ``Predictor`` (unchanged ``predict``) plus the async hook."""

        _flora: _PredictorRuntime

        async def apredict(self, request):
            rt = self._flora
            start = len(self.calls)
            response, texts = await run_tapped(
                rt.nv, rt.batcher, lambda: self.predict(request), tokenizer=rt.tokenizer,
                eos=rt.eos, stats=rt.stats)
            preds = response.get("predictions") if isinstance(response, dict) else None
            if not isinstance(preds, list) or len(preds) != len(texts):
                raise RuntimeError("NVIDIA's predict returned an unexpected response shape")
            for p in preds:
                p["prediction_json"] = substitute(p["prediction_json"], texts)
            for call in self.calls[start:]:
                call["output"] = substitute(call["output"], texts)
            rt.requests += 1
            return response

    MicroBatchPredictor.__qualname__ = "MicroBatchPredictor"
    _PREDICTOR_CLASSES[id(base)] = MicroBatchPredictor
    return MicroBatchPredictor


@dataclass
class _PredictorRuntime:
    nv: Any
    batcher: Any
    tokenizer: Any
    eos: int
    stats: CallStats
    requests: int = 0  # apredict calls (grid + retries)


# ---------------------------------------------------------------------------------------------
# Episode inputs and the stage
# ---------------------------------------------------------------------------------------------

PROFILES = {"human": "human", "yam": "yam", "dex": "dex", "tianji": "tianji"}


@dataclass(frozen=True)
class EpisodeVideos:
    """NVIDIA's probed inputs for one episode."""

    videos: dict  # role -> NVIDIA media.Video, in NVIDIA's role order
    profile: str  # NVIDIA's canonical profile name ("Human", "YAM", "Dex", "Tianji")
    pool: str  # human | yam | yam32 | dex | tianji
    instruction: str

    @property
    def duration_s(self) -> float:
        return float(self.videos["main"].duration_s)


def pool_for(nv, canonical: str, yam32: bool) -> str:
    """``pipeline.annotate``'s pool choice (restated; license flag in the module docstring)."""
    return "yam32" if yam32 else nv.config.PROFILE_POOLS[canonical]


def prepare_videos(nv, paths: Mapping[str, str], profile: str, *, yam32: bool = False,
                   instruction: str = "") -> EpisodeVideos:
    """NVIDIA's own input validation and probe (``api.DataEnginePipeline._inputs``)."""
    videos, canonical = nv.api.DataEnginePipeline._inputs(
        paths["main"], profile, paths.get("left_wrist"), paths.get("right_wrist"), yam32,
        instruction)
    return EpisodeVideos(videos=videos, profile=canonical, pool=pool_for(nv, canonical, yam32),
                         instruction=instruction)


@dataclass
class SegmentResult:
    rows: list[dict]  # NVIDIA's pipeline.segment rows: start_s, end_s, metadata
    calls: list[dict]  # NVIDIA's Predictor.calls (temporal audit)
    apredict_rounds: int
    stats: CallStats


def _noop_progress(*_a, **_k) -> None:
    return None


class SegmentStage:
    """TempSeg on a micro-batched engine. One per container; ``segment`` per episode."""

    def __init__(self, nv, processor):
        self.nv = nv
        self.processor = processor
        self.tokenizer = processor.tokenizer
        eos = self.tokenizer.eos_token_id
        if type(eos) is not int:
            raise ValueError(f"TempSeg tokenizer EOS must be one id, got {eos!r}")
        self.eos = eos
        self.tap = GenerateTap.acquire(nv)
        self.Predictor = predictor_class(nv)

    def close(self) -> None:
        self.tap.release()

    async def segment(self, batcher, ep: EpisodeVideos) -> SegmentResult:
        """``pipeline.segment(videos, pool, session, instruction, progress)`` with the model
        calls micro-batched. NVIDIA's exceptions propagate unchanged."""
        from flora_deploy.nv import segment_glue as glue

        nv, videos, pool = self.nv, ep.videos, ep.pool
        main = videos["main"]
        source = glue.annotation_source(nv, source_id=glue.source_id_for(main.sha256),
                                        location=str(main.path), duration_s=main.duration_s,
                                        fps=float(main.rate))
        bound = glue.model_binding(nv, pool)
        proc = CaptureProcessor.for_kind(nv, self.processor, "tempseg")
        predictor = self.Predictor(videos, source, pool, bound, None, proc, ep.instruction,
                                   _noop_progress)
        stats = CallStats()
        predictor._flora = _PredictorRuntime(nv, batcher, self.tokenizer, self.eos, stats)
        engine = glue.segmenter_for(nv, predictor, bound, pool, len(videos))
        segments = await engine.segment_async(source)
        rows = [dict(start_s=s.span.start_s, end_s=s.span.end_s, metadata=dict(s.metadata))
                for s in segments]
        return SegmentResult(rows, predictor.calls, predictor._flora.requests, stats)


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

UNCERTIFIED = "uncertified_stack"


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def segment_record(rows: Sequence[Mapping], *, uid: str, dataset: str, episode_id: str,
                   provenance, certificate_ok: bool = False):
    """NVIDIA's segment rows as a ``segment`` ``StageRecord``. Uncertified (``uncertified_stack``)
    until a certificate covers this stack; the full NVIDIA metadata rides in ``nv_metadata``
    (the caption stage needs it back)."""
    from flora_deploy.records import Segment, StageRecord

    segs = []
    for i, row in enumerate(rows):
        md = json_safe(dict(row.get("metadata") or {}))
        segs.append(Segment(
            event_id=f"event_{i:04d}", start_s=float(row["start_s"]), end_s=float(row["end_s"]),
            status="complete", fine_event_id=md.get("fine_event_id"),
            segmentation_provenance=md.get("segmentation_provenance", "fine"),
            nv_metadata=md))
    return StageRecord(uid=uid, dataset=dataset, episode_id=episode_id, stage="segment",
                       provenance=provenance, certified=certificate_ok,
                       uncertified_reason=None if certificate_ok else UNCERTIFIED,
                       segments=tuple(segs))


def rows_from_record(record) -> list[dict]:
    """Back to NVIDIA's ``pipeline.segment`` rows (what ``caption_*`` take)."""
    if record.stage != "segment":
        raise ValueError(f"expected a segment record, got {record.stage!r}")
    return [dict(start_s=s.start_s, end_s=s.end_s, metadata=dict(s.nv_metadata))
            for s in record.segments]


def config_hash(kind: str, *, spec: Mapping[str, Any] | None, models: Mapping[str, str],
                budget: int | None, extra: Mapping[str, Any] | None = None) -> str:
    """sha256 over what changes this stage's outputs (D11): engine spec (no GPU), sampling,
    model pins, logic version and digest, ``extra`` (image fingerprint, accel patches)."""
    from flora_deploy import LOGIC_VERSION
    from flora_deploy.config.logic import logic_digest

    stage = {"tempseg": "segment", "captioner": "caption"}[kind]
    blob = {"stage": stage, "kind": kind, "engine": dict(spec or {}),
            "sampling": {"max_tokens": budget, "temperature": 0.0, "stop": "eos"},
            "models": dict(models), "logic_version": LOGIC_VERSION,
            "logic_digest": logic_digest(stage), "extra": dict(extra or {})}
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def provenance(kind: str, *, engine_label: str, spec: Mapping[str, Any] | None,
               models: Mapping[str, str], gpu: str | None, budget: int | None,
               certificate: str | None = None, extra: Mapping[str, Any] | None = None,
               source: Mapping[str, dict] | None = None):
    from flora_deploy import LOGIC_VERSION, __version__
    from flora_deploy.records import Provenance

    return Provenance(models=dict(models), engine=engine_label, preset="balanced",
                      config_hash=config_hash(kind, spec=spec, models=models, budget=budget,
                                              extra=extra),
                      logic_version=LOGIC_VERSION, framework=f"flora-deploy {__version__}",
                      gpu=gpu, certificate=certificate, source=dict(source or {}))
