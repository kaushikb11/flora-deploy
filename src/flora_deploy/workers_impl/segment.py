"""Balanced SegmentWorker core (P1b-1): ``workers.ShardWorker`` around ``stages.segment``.

``BalancedShardRunner`` is shared by the segment and caption workers. It follows the worker
contract of ``workers.py`` exactly as ``workers_impl.reward.RewardShardRunner`` does (Volume
refresh, step 0 superseded -> ``None``, attempt marker first, skip done, skip upstream-poisoned,
per-episode isolation into ``EpisodeError`` files, flush + ``mark_preempted`` on SIGINT,
``mark_exception`` + re-raise on a python failure, receipt last naming every part) and reuses
its ``ShardPartSink``. What differs is step 4: the shard's episodes run as coroutines on one
event loop, so their model calls share ``llm.generate`` rounds (D14). An episode's records are
added to the sink when it finishes; the sink is copied to the store every ``flush_every``
finished episodes and at the end (and by ``preempt`` / ``@modal.exit``).

Two-phase episodes (WP1a; :class:`EpisodeGates`). A job with ``phased = True`` runs its CPU
preparation (input resolve, NVIDIA's ``prepare_videos`` probe) under ``gates.prep()``, **outside**
``MicroBatcher.episode()``, and its inference under ``gates.infer(batcher)``, inside it. At most
``prefetch_episodes`` episodes are preparing or prepared-and-waiting, at most
``infer_episodes`` inferring (``engines.batch_policy``): decoding overlaps generation, and a
preparing episode (a long probe takes 30-40 s) never holds a round back until ``max_wait_s``.
Both phases run in the one episode coroutine, so wrappers around ``job.run`` (accel's
``active.episode()`` scope, ``workers_impl.balanced``) cover both. A job without ``phased`` runs
whole in the inference phase. The work itself, and so every model input, is the same as on the
one-phase path (``phased=False``; ``tests/unit/test_balanced_phases.py``,
``tests/nvlink/test_balanced_accel.py``). The loop gets an explicit thread pool sized to the
container's cores (``executor_workers``). The shard's audit row (``shard_audit``) records the
batch policy, the batcher and gate statistics and, with ``sample_resources``, the container's
peak RSS, core utilization and GPU memory (``observe.sampler.ContainerSampler``).

The per-stage work is an ``EpisodeJob``: ``SegmentEpisodeJob`` (here) and
``workers_impl.caption.CaptionEpisodeJob``. Errors are classified by ``_classify`` (review 6
M1) per job phase (``phase``): ``inputs`` (index receipt, upstream records) by segcap's
``classify_input_error``, ``nvidia`` (NVIDIA's probe and stage) by the shared
``nv.errors.classify`` (NVIDIA's code identified by where it was loaded from: its classes,
builtins its own frames raise, its builtin ``RuntimeError`` rejections and our
``ContextBudgetError`` copy of its guard are ``nv_failure``), ``records`` (our record
building) never NVIDIA's. Only ``nv_failure`` is terminal; an ``infra`` / ``python`` /
``oom`` episode error makes the shard raise ``ShardInfraError`` (no receipt, retried). A CUDA
fault (``cuda_fatal``) stops the shard at once and calls ``fatal`` (``os._exit``) after the mark.

Records are uncertified (``uncertified_stack``) until certificates exist; the sink stamps the
job's stage key (``JobFiles.key``, which covers the engine spec: ``config.hashing``) as
``provenance.config_hash`` and keeps the stage's own spec hash as ``provenance.engine_hash``;
the engine audit goes to the audit lines. The receipt carries the stage key (review 6 L3).
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flora_deploy.engines.base import BatchEngine
from flora_deploy.engines.batch_policy import BatchPolicy, policy_for
from flora_deploy.engines.microbatch import BatcherClosed, EngineBatchError, MicroBatcher
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.deps import StoreManifestSource
from flora_deploy.jobs.orchestrator import utc
from flora_deploy.jobs.store import Store
from flora_deploy.jobs.types import (
    AttemptMarker,
    EpisodeError,
    EpisodeRef,
    Shard,
    ShardReceipt,
    safe_key,
)
from flora_deploy.nv import errors as nv_errors
from flora_deploy.records import StageRecord
from flora_deploy.workers_impl.reward import (
    ShardInfraError,
    ShardPartSink,
    VideoResolver,
    _container_ks,
    cuda_fatal,
    default_container_id,
    end_failed_shard,
    parse_index_receipt,
)
from flora_deploy.workers_impl.segcap import classify_input_error

# Golden / index pools -> NVIDIA ``api`` profile and the yam32 flag (when a receipt lacks them).
POOL_PROFILES: Mapping[str, tuple[str, bool]] = {
    "human": ("human", False), "yam": ("yam", False), "yam32": ("yam", True),
    "dex": ("dex", False), "tianji": ("tianji", False), "single_view_robot": ("yam", False),
}
ROLES = ("main", "left_wrist", "right_wrist")


# ---------------------------------------------------------------------------------------------
# Episode inputs
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeInputs:
    """What the stages need for one episode: local camera files and NVIDIA's call arguments."""

    uid: str
    dataset: str
    episode_id: str
    paths: dict[str, str]  # role -> local video path (main / left_wrist / right_wrist)
    profile: str  # NVIDIA api profile: human | yam | dex | tianji
    yam32: bool
    instruction: str
    envelope_ok: bool | None = None
    sources: dict[str, dict] = field(default_factory=dict)  # provenance: role -> uri / range


class InputResolver:
    """``EpisodeRef`` -> ``EpisodeInputs`` from its index receipt (``EpisodeRef.index_ref``)."""

    def __init__(self, store: Store, *, media_root: str | None = None,
                 local_dir: str | Path = "/tmp/flora-balanced", default_task: str | None = None,
                 resolver: VideoResolver | None = None):
        self.store = store
        self.default_task = default_task
        self.resolver = resolver or VideoResolver(media_root, Path(local_dir) / "cuts")

    def __call__(self, ref: EpisodeRef) -> EpisodeInputs:
        raw = self.store.get_bytes(ref.index_ref)
        idx = parse_index_receipt(raw)
        d = json.loads(raw)
        profile, yam32 = POOL_PROFILES.get(ref.pool, (None, False))
        profile = d.get("profile") or profile
        yam32 = yam32 if d.get("yam32") is None else bool(d["yam32"])
        if not profile:
            raise ValueError(f"episode {ref.uid!r}: no NVIDIA profile for pool {ref.pool!r}")
        if set(idx.sources) - set(ROLES) or "main" not in idx.sources:
            raise ValueError(f"episode {ref.uid!r}: camera roles {sorted(idx.sources)} are not "
                             "main[/left_wrist/right_wrist]")
        source = d.get("instruction", "dataset_task")
        if source == "none":
            instruction = ""
        elif source == "dataset_task" and idx.task is not None:
            instruction = str(idx.task)
        else:
            instruction = self.default_task or ""
        paths = {role: self.resolver.resolve(ref.uid, role, idx.sources[role])
                 for role in ROLES if role in idx.sources}
        sources = {role: {"uri": s.uri, "range_s": [s.start_s, s.end_s]}
                   for role, s in idx.sources.items()}
        return EpisodeInputs(uid=ref.uid, dataset=ref.dataset, episode_id=ref.episode_id,
                             paths=paths, profile=str(profile), yam32=yam32,
                             instruction=instruction, envelope_ok=ref.envelope_ok,
                             sources=sources)


# ---------------------------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------------------------


PHASES = ("inputs", "nvidia", "records")
_PLUMBING = (EngineBatchError, BatcherClosed, OSError, TimeoutError)


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Tag an exception raised inside the block with the job phase it came from
    (``exc.flora_phase``) for :func:`_classify`. The innermost tag wins."""
    try:
        yield
    except BaseException as exc:
        if getattr(exc, "flora_phase", None) is None:
            try:
                exc.flora_phase = name  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - an exception without a __dict__
                pass
        raise


def _classify(exc: BaseException, nv: Any = None) -> str:
    """The single call site for the balanced failure taxonomy (review 6 M1): one of
    ``nv_failure`` | ``infra`` | ``python`` | ``cuda_fatal`` | ``oom``. The ``nvidia`` phase (the
    default) is the shared classifier's verdict (``nv.errors.classify``: NVIDIA's code by where
    it was loaded from, the same call segcap makes), the ``inputs`` phase is segcap's
    ``classify_input_error`` and the ``records`` phase (our record building) is never an
    NVIDIA verdict. ``nv`` is accepted for callers' symmetry and unused: the linked family is
    found by ``nv.linker.loaded_family`` (and fakes by ``nv.errors.register_nv_root``)."""
    from flora_deploy.engines.offline_llm import engine_dead
    from flora_deploy.stages.segment import EngineRequestError

    msg = f"{type(exc).__name__}: {exc}"
    if cuda_fatal(msg) or engine_dead(msg):  # the engine is unusable (offline_llm.EngineDead)
        return "cuda_fatal"
    if "out of memory" in msg.lower() or type(exc).__name__ == "OutOfMemoryError":
        return "oom"
    where = getattr(exc, "flora_phase", None) or "nvidia"
    if isinstance(exc, _PLUMBING + (EngineRequestError,)):
        return "infra"
    if where == "inputs":
        return classify_input_error(exc)
    if where == "records":
        return "python"
    return nv_errors.classify(exc)


def classify_error(nv, exc: BaseException) -> str:
    """``EpisodeError.kind`` for :func:`_classify`'s category: only ``nv_failure`` is
    terminal; everything else is ``infra`` (retried, never excused)."""
    return "nv_failure" if _classify(exc, nv) == "nv_failure" else "infra"


# ---------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------


_ENGINE_GONE = re.compile(r"engine[ _-]?core|enginedead|engine is dead|engine died", re.I)


def is_crash(message: str) -> bool:
    """A message meaning the container's engine is unusable: a CUDA fault or a dead vLLM
    engine, whatever its wording (``EngineDeadError``, the engine-core process gone)."""
    from flora_deploy.engines.offline_llm import engine_dead

    return cuda_fatal(message) or engine_dead(message) or bool(_ENGINE_GONE.search(message))


def container_fatal(exc: BaseException, engine: Any = None) -> bool:
    """Whether the container must be replaced after ``exc`` (review 8a R4): a CUDA fault, or
    the vLLM engine is dead whatever the message says (``EngineDead`` / ``EngineDeadError``,
    the engine-core process gone, ``OfflineEngine.dead`` set), in the shard error or in any of
    its episodes' infra errors."""
    from flora_deploy.engines.offline_llm import EngineDead

    msgs = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, ShardInfraError):
        msgs += list(exc.infra.values())
    if isinstance(exc, EngineDead) or getattr(engine, "dead", None):
        return True
    return any(map(is_crash, msgs))


@dataclass
class EpisodeOutcome:
    records: list[StageRecord]
    audits: list[dict] = field(default_factory=list)


class EpisodeJob(Protocol):
    stage: str  # record stage ("segment" / "caption")
    nv: Any

    async def run(self, ref: EpisodeRef, batcher: MicroBatcher, ctx: ShardContext
                  ) -> EpisodeOutcome: ...


@dataclass
class ShardContext:
    jf: A.JobFiles
    shard: Shard
    manifests: StoreManifestSource
    container_id: str
    cache: dict = field(default_factory=dict)  # per-shard memo for jobs (e.g. upstream records)
    gates: EpisodeGates | None = None  # the runner's two-phase bounds (None: inline)


# ---------------------------------------------------------------------------------------------
# Two-phase episodes (WP1a)
# ---------------------------------------------------------------------------------------------


class GatesStopped(RuntimeError):
    """The shard is stopping (the engine died under another episode): do not start inference."""


@dataclass(eq=False)
class _Slot:
    """One episode coroutine's place in the gates."""

    prep_held: bool = False
    inferring: bool = False


_SLOT: contextvars.ContextVar[_Slot | None] = contextvars.ContextVar(
    "flora_episode_slot", default=None)


class EpisodeGates:
    """Bounds of the two episode phases (module docstring).

    ``prep()``: wait for one of ``prefetch`` slots; the slot is held until the episode's
    ``infer()`` has its inference slot (or the episode ends), so episodes prepared ahead are
    bounded too. ``infer(batcher)``: wait for one of ``infer`` slots, release the prep slot,
    enter ``batcher.episode()``. Inference never waits for a prep slot, so the two bounds cannot
    deadlock. Outside :meth:`slot`, ``prep()`` is a no-op and ``infer()`` is just
    ``batcher.episode()`` (as with ``ctx.gates`` ``None``: :data:`INLINE`)."""

    def __init__(self, prefetch: int, infer: int):
        self.prefetch, self.infer_limit = max(1, int(prefetch)), max(1, int(infer))
        self._prep = asyncio.Semaphore(self.prefetch)
        self._infer = asyncio.Semaphore(self.infer_limit)
        self.stopped = False
        self.preparing = 0  # holding a prep slot (preparing, or prepared and waiting)
        self.inferring = 0
        self.peak_preparing = 0
        self.peak_inferring = 0
        self.peak_admitted = 0

    @contextmanager
    def slot(self) -> Iterator[_Slot]:
        """One episode coroutine; releases whatever it still holds at exit."""
        slot = _Slot()
        token = _SLOT.set(slot)
        try:
            yield slot
        finally:
            _SLOT.reset(token)
            self._release_prep(slot)

    def _release_prep(self, slot: _Slot) -> None:
        if slot.prep_held:
            slot.prep_held = False
            self.preparing -= 1
            self._prep.release()

    def _peaks(self) -> None:
        self.peak_preparing = max(self.peak_preparing, self.preparing)
        self.peak_inferring = max(self.peak_inferring, self.inferring)
        self.peak_admitted = max(self.peak_admitted, self.preparing + self.inferring)

    @asynccontextmanager
    async def prep(self) -> AsyncIterator[None]:
        slot = _SLOT.get()
        if slot is None or slot.prep_held or slot.inferring:
            yield
            return
        await self._prep.acquire()
        slot.prep_held = True
        self.preparing += 1
        self._peaks()
        try:
            yield
        except BaseException:
            self._release_prep(slot)
            raise

    @asynccontextmanager
    async def infer(self, batcher: Any) -> AsyncIterator[None]:
        slot = _SLOT.get()
        if slot is None or slot.inferring:
            async with batcher.episode():
                yield
            return
        if self.stopped:
            raise GatesStopped("the shard is stopping")
        await self._infer.acquire()
        try:
            if self.stopped:
                raise GatesStopped("the shard is stopping")
            self._release_prep(slot)
            slot.inferring = True
            self.inferring += 1
            self._peaks()
            try:
                async with batcher.episode():
                    yield
            finally:
                slot.inferring = False
                self.inferring -= 1
        finally:
            self._infer.release()

    def as_json(self) -> dict[str, int]:
        return {"prefetch": self.prefetch, "infer": self.infer_limit,
                "peak_preparing": self.peak_preparing, "peak_inferring": self.peak_inferring,
                "peak_admitted": self.peak_admitted}


class _InlineGates:
    """``ctx.gates`` absent (a job run directly, e.g. by the nvlink tests): no bounds."""

    @asynccontextmanager
    async def prep(self) -> AsyncIterator[None]:
        yield

    @asynccontextmanager
    async def infer(self, batcher: Any) -> AsyncIterator[None]:
        async with batcher.episode():
            yield


INLINE = _InlineGates()


def gates_of(ctx: ShardContext | None) -> EpisodeGates | _InlineGates:
    gates = getattr(ctx, "gates", None)
    return INLINE if gates is None else gates


@dataclass(frozen=True)
class StageInfo:
    """Provenance inputs shared by the jobs."""

    engine_label: str
    engine_spec: Mapping[str, Any] | None = None  # EngineSpec.hashed()
    models: Mapping[str, str] = field(default_factory=dict)
    gpu: str | None = None
    certificate: str | None = None
    config_extra: Mapping[str, Any] = field(default_factory=dict)


class SegmentEpisodeJob:
    """One episode: NVIDIA's probe, then ``stages.segment.SegmentStage.segment``."""

    stage = "segment"
    phased = True  # CPU preparation outside the batcher's episode (EpisodeGates)

    def __init__(self, stage, info: StageInfo, inputs: Callable[[EpisodeRef], EpisodeInputs],
                 *, budget: int | None = None):
        self.seg = stage
        self.nv = stage.nv
        self.info = info
        self.inputs = inputs
        if budget is None:
            from flora_deploy.nv.budget import budget_for

            budget = budget_for(self.nv, "tempseg")
        self.budget = budget

    async def run(self, ref: EpisodeRef, batcher: MicroBatcher, ctx: ShardContext
                  ) -> EpisodeOutcome:
        from flora_deploy.stages import segment as S

        gates = gates_of(ctx)
        async with gates.prep():  # phase 1: CPU, outside the batcher's episode
            with phase("inputs"):
                inp = await asyncio.to_thread(self.inputs, ref)
            t0 = time.perf_counter()
            with phase("nvidia"):
                ep = await asyncio.to_thread(S.prepare_videos, self.nv, inp.paths, inp.profile,
                                             yam32=inp.yam32, instruction=inp.instruction)
            t1 = time.perf_counter()
        async with gates.infer(batcher):  # phase 2: inference, inside it
            tq = time.perf_counter()
            with phase("nvidia"):
                res = await self.seg.segment(batcher, ep)
        t2 = time.perf_counter()
        with phase("records"):
            return self._outcome(ref, inp, ep, res, t0, t1, t2, tq)

    def _outcome(self, ref, inp, ep, res, t0, t1, t2, tq=None) -> EpisodeOutcome:
        from flora_deploy.stages import segment as S

        prov = S.provenance("tempseg", engine_label=self.info.engine_label,
                            spec=self.info.engine_spec, models=self.info.models,
                            gpu=self.info.gpu, budget=self.budget,
                            certificate=self.info.certificate, extra=self.info.config_extra,
                            source=inp.sources)
        rec = S.segment_record(res.rows, uid=ref.uid, dataset=ref.dataset,
                               episode_id=ref.episode_id, provenance=prov)
        audit = {"uid": ref.uid, "stage": "segment", "config_hash": prov.config_hash,
                 "pool": ep.pool, "profile": ep.profile, "duration_s": ep.duration_s,
                 "envelope_ok": inp.envelope_ok, "temporal_windows": S.json_safe(res.calls),
                 "apredict_rounds": res.apredict_rounds, "calls": res.stats.as_json(),
                 "timings_s": timings(t0, t1, t2, tq, "segment")}
        return EpisodeOutcome([rec], [audit])


def timings(t0: float, t1: float, t2: float, tq: float | None, name: str) -> dict[str, float]:
    """Episode timings: ``probe``, ``infer_wait`` (prepared, waiting for an inference slot)
    and the stage's own time."""
    tq = t1 if tq is None else tq
    return {"probe": round(t1 - t0, 3), "infer_wait": round(tq - t1, 3),
            name: round(t2 - tq, 3)}


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancedWorkerSettings:
    """Worker plumbing. The batching bounds default to the shard kind's
    ``engines.batch_policy`` (code, hashed in the logic digest); a value set here overrides it
    (tests and the fake workers; a real worker sets none). ``max_inflight_episodes`` is the
    inference bound (``BatchPolicy.infer_episodes``). ``phased=False``: every episode runs
    whole inside ``MicroBatcher.episode()`` (the one-phase path, kept for the identity proofs).
    ``executor_workers``: the event loop's thread pool (default: the container's cores).
    ``sample_resources``: container resource sampling (``observe.sampler.ContainerSampler``,
    at most every 10 s) into the startup facts and the shard audit row."""

    max_inflight_episodes: int | None = None
    prefetch_episodes: int | None = None
    flush_every: int = 8
    max_batch: int | None = None
    max_wait_s: float | None = None
    max_batch_bytes: int | None = None
    max_pending_bytes: int | None = None
    phased: bool = True
    executor_workers: int | None = None
    sample_resources: bool = False
    local_dir: str = "/tmp/flora-balanced"
    gpu: str | None = None
    container_id: str = field(default_factory=default_container_id)

    def policy(self, kind: str) -> BatchPolicy:
        return policy_for(kind, infer_episodes=self.max_inflight_episodes,
                          prefetch_episodes=self.prefetch_episodes, max_batch=self.max_batch,
                          max_wait_s=self.max_wait_s, max_batch_bytes=self.max_batch_bytes,
                          max_pending_bytes=self.max_pending_bytes)


def executor_workers(settings: BalancedWorkerSettings) -> int:
    """Threads of the episode loop's executor: the container's cores (at least 4), or the
    setting."""
    if settings.executor_workers:
        return max(1, int(settings.executor_workers))
    from flora_deploy.observe.sampler import available_cores

    return max(4, available_cores())


@dataclass
class _Active:
    jf: A.JobFiles
    shard: Shard
    marker: AttemptMarker
    sink: ShardPartSink | None = None
    ended: bool = False


class BalancedShardRunner:
    """``ShardWorker`` core for ``segment`` / ``caption`` shards (module docstring)."""

    def __init__(self, store: Store, engine: BatchEngine, job: EpisodeJob,
                 settings: BalancedWorkerSettings, *,
                 call_id: Callable[[], str | None] = lambda: "",
                 clock: Callable[[], float] = time.time,
                 observer: Callable[..., None] | None = None,
                 fatal: Callable[[BaseException], None] | None = None):
        """``fatal``: called after the shard is marked when a CUDA fault left the engine
        unusable (``os._exit`` in the Modal worker, as ``RewardShardRunner``)."""
        self.store, self.engine, self.job, self.settings = store, engine, job, settings
        self.fatal = fatal
        self.call_id, self.clock = call_id, clock
        self.observer = observer
        self.container_id = safe_key(settings.container_id)
        self._lock = threading.RLock()
        self._active: _Active | None = None
        self._next_k: dict[tuple[str, str, str], int] = {}
        self.last_stats: dict[str, Any] = {}
        self.last_gates: dict[str, Any] = {}
        self.policy = settings.policy(job.stage)
        self.batcher_cls: Callable[..., MicroBatcher] = MicroBatcher  # the simulation's seam
        self.sampler: Any = None
        self.compile_cache: Any = None  # engines.compile_cache.WorkerCache (balanced_app)
        if settings.sample_resources:
            from flora_deploy.observe.sampler import ContainerSampler

            self.sampler = ContainerSampler().start()

    def startup_facts(self) -> dict[str, Any]:
        """For the worker's startup facts: the effective batch policy, the loop's executor
        size and the container's resources so far (peak RSS covers the engine build)."""
        out: dict[str, Any] = {"batch_policy": self.policy.as_json(),
                               "phased": self.settings.phased,
                               "executor_workers": executor_workers(self.settings),
                               "cache_blocks": getattr(self.engine, "cache_blocks", None),
                               # the compile cache is balanced_app's startup "compile_cache"
                               "preemptions": self._preemptions()}
        if self.sampler is not None:
            try:
                out["resources"] = self.sampler.snapshot()
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                out["resources"] = {"error": repr(exc)}
        return out

    # -- contract ------------------------------------------------------------------------
    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        refresh = getattr(self.store, "refresh", None)
        if callable(refresh):
            refresh()
        jf = A.JobFiles(self.store, job_id, config_hash)
        manifests = StoreManifestSource(jf)
        shard = manifests.shard(shard_id)
        if A.superseded(jf, shard.kind, shard_id, token):  # step 0
            return None
        marker = A.write_attempt(  # step 1
            jf, shard.kind, shard_id, self.container_id, started_utc=utc(self.clock()),
            orchestrator_attempt=A.token_attempt(token), token=token,
            call_id=self.call_id() or "")
        active = _Active(jf, shard, marker)
        with self._lock:
            self._active = active
        try:
            return self._run(active, manifests)
        except KeyboardInterrupt:
            self._end(active, preempted=True)
            raise
        except Exception as exc:  # crash: crashed=True (D15), then fatal (review 8a R4)
            from flora_deploy.engines.offline_llm import EngineDead

            dead = isinstance(exc, EngineDead) or bool(getattr(self.engine, "dead", None))
            end_failed_shard(exc, lambda **kw: self._end(active, **kw), self.fatal,
                             crash="cuda_fatal" if dead else None, is_crash=is_crash)
            raise
        finally:
            with self._lock:
                if self._active is active:
                    self._active = None

    def preempt(self) -> bool:
        with self._lock:
            active = self._active
        return active is not None and self._end(active, preempted=True)

    def on_exit(self) -> None:
        self.preempt()

    # -- internals -----------------------------------------------------------------------
    def _preemptions(self) -> int | None:
        """The engine's cumulative preemption count (``OfflineEngine.preemptions``); None when
        the engine has no counter or it cannot be read."""
        get = getattr(self.engine, "preemptions", None)
        try:
            n = get() if callable(get) else None
        except Exception:  # noqa: BLE001 - unknown, never 0
            return None
        return n if type(n) is int else None

    def _cache_identity(self) -> str | None:
        ident = getattr(getattr(self.compile_cache, "status", None), "identity", None)
        return ident() if callable(ident) else None

    def _engine_facts(self, records: Sequence[Any], p0: int | None) -> list[Any]:
        """Stamp ``provenance.compile_cache`` and ``engine_preemptions`` (the counter's change
        over this episode's window, so a superset of the episodes whose requests were
        preempted; None when unknown at either end)."""
        p1 = self._preemptions()
        delta = p1 - p0 if p0 is not None and p1 is not None else None
        ident = self._cache_identity()
        return [dataclasses.replace(r, provenance=dataclasses.replace(
            r.provenance, compile_cache=ident, engine_preemptions=delta)) for r in records]

    def _end(self, active: _Active, *, preempted: bool = False, error: str | None = None,
             crashed: bool = False) -> bool:
        with self._lock:
            if active.ended:
                return False
            active.ended = True
        if active.sink is not None:
            try:
                active.sink.flush()
            except Exception:  # noqa: BLE001 - the mark matters more; parts are redone
                pass
        jf, kind, sid = active.jf, active.shard.kind, active.shard.shard_id
        now = utc(self.clock())
        if preempted:
            A.mark_preempted(jf, kind, sid, self.container_id, ended_utc=now,
                             marker=active.marker)
        else:
            A.mark_exception(jf, kind, sid, self.container_id, ended_utc=now,
                             error=error or "error", marker=active.marker, crashed=crashed)
        return True

    def _error(self, uid: str, shard: Shard, exc: BaseException, category: str
               ) -> EpisodeError:
        kind = "nv_failure" if category == "nv_failure" else "infra"
        return EpisodeError(uid=uid, stage=self.job.stage, shard_id=shard.shard_id,  # type: ignore[arg-type]
                            container_id=self.container_id, kind=kind,  # type: ignore[arg-type]
                            message=f"{type(exc).__name__}: {exc}"[:4000],
                            diagnostics={"phase": getattr(exc, "flora_phase", None) or "nvidia",
                                         "category": category, "type": type(exc).__name__,
                                         "mro": [c.__name__ for c in type(exc).__mro__][:6]})

    def _run(self, active: _Active, manifests: StoreManifestSource) -> ShardReceipt:
        t0 = time.perf_counter()
        jf, shard, s = active.jf, active.shard, self.settings
        if shard.kind != self.job.stage:
            raise ValueError(f"{self.job.stage} worker cannot run a {shard.kind!r} shard")
        prior = P.shard_parts(jf, shard)  # step 2
        done = set(prior.done)
        key = (jf.job_id, jf.config_hash, shard.shard_id)
        ks = _container_ks(prior.parts, self.container_id)
        start_k = max([self._next_k.get(key, 0), *(k + 1 for k in ks)])
        sink = ShardPartSink(jf, shard.kind, shard.shard_id, self.container_id, s.local_dir,
                             start_k=start_k)
        active.sink = sink

        poisoned: set[str] = set()  # step 3: excluded upstream (poison pill or nv_failure)
        if shard.depends_on:
            m = manifests.load()
            excluded = {u: why for u, why in A.upstream_excluded(jf, shard, m.shards).items()
                        if u not in done}
            A.write_upstream_failures(jf, shard, excluded)
            poisoned = set(excluded)

        refs = {e.uid: e for e in manifests.base().episodes}
        todo = [u for u in shard.episodes if u not in done and u not in poisoned]
        ctx = ShardContext(jf, shard, manifests, self.container_id)
        t_sample = self.sampler.clock() if self.sampler is not None else 0.0
        self.last_stats, self.last_gates = {}, {}
        p_shard = self._preemptions()
        if todo:
            asyncio.run(self._episodes(todo, refs, sink, ctx, active))  # step 4
        sink.add(audits=[self._shard_audit(shard, len(todo), t_sample, p_shard)])
        sink.flush()
        self._next_k[key] = sink.k

        finished = done | sink.done
        failed = (sink.failed | poisoned) - finished
        infra = {u: m for u, m in sink.infra.items() if u not in finished and u not in failed}
        if infra:  # never terminal: no receipt, the shard is retried (as segcap / reward)
            raise ShardInfraError(shard.shard_id, infra)
        missing = [u for u in shard.episodes if u not in finished and u not in failed]
        if missing:
            raise RuntimeError(f"shard {shard.shard_id!r}: no record or error for {missing}")
        receipt = ShardReceipt(  # step 6: LAST
            shard_id=shard.shard_id, kind=shard.kind, config_hash=jf.key(shard.kind),
            container_id=self.container_id,
            episodes_done=tuple(u for u in shard.episodes if u in finished),
            episodes_failed=tuple(u for u in shard.episodes if u in failed),
            parts=tuple(sorted(set(prior.parts) | set(sink.parts))), gpu=s.gpu,
            wall_seconds=round(time.perf_counter() - t0, 3), finished_utc=utc(self.clock()))
        P.write_receipt(jf, receipt)
        with self._lock:
            active.ended = True
        return receipt

    def _shard_audit(self, shard: Shard, episodes: int, t_sample: float,
                     p_shard: int | None = None) -> dict[str, Any]:
        """The shard's audit row: batch policy, batcher and gate stats, container resources,
        the engine's preemptions during the shard (None: unknown) and the compile cache."""
        p1 = self._preemptions()
        row: dict[str, Any] = {"shard_audit": True, "shard_id": shard.shard_id,
                               "stage": self.job.stage, "episodes_run": episodes,
                               "batch_policy": self.policy.as_json(),
                               "phased": self.settings.phased,
                               "batcher": dict(self.last_stats), "gates": dict(self.last_gates),
                               "preemptions": (p1 - p_shard if p1 is not None
                                               and p_shard is not None else None),
                               "compile_cache": self._cache_identity(),
                               # the build's cache block facts (check_mamba_blocks), persisted
                               # with the shard so future limits rest on measured counts
                               "cache_blocks": getattr(self.engine, "cache_blocks", None)}
        if self.sampler is not None:
            try:
                self.sampler.sample()
                row["resources"] = self.sampler.summary_since(t_sample)
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                row["resources"] = {"error": repr(exc)}
        return row

    async def _episodes(self, todo: list[str], refs: Mapping[str, EpisodeRef],
                        sink: ShardPartSink, ctx: ShardContext,
                        active: _Active | None = None) -> None:
        s, pol = self.settings, self.policy
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(executor_workers(s),
                                                     thread_name_prefix="flora-episode"))
        one_phase = asyncio.Semaphore(pol.infer_episodes)  # phased=False only
        gates = EpisodeGates(pol.prefetch_episodes, pol.infer_episodes)
        if s.phased:
            ctx.gates = gates
        phased_job = s.phased and bool(getattr(self.job, "phased", False))
        finished = 0
        async with self.batcher_cls(self.engine, max_batch=pol.max_batch,
                                    max_wait_s=pol.max_wait_s,
                                    max_batch_bytes=pol.max_batch_bytes,
                                    max_pending_bytes=pol.max_pending_bytes,
                                    observer=self.observer, clock=loop.time) as mb:

            dead: list[BaseException] = []

            async def run_job(ref: EpisodeRef) -> EpisodeOutcome:
                if not s.phased:  # one phase: the whole episode inside the batcher's
                    async with one_phase, mb.episode():
                        return await self.job.run(ref, mb, ctx)
                if phased_job:
                    return await self.job.run(ref, mb, ctx)
                async with gates.prep():  # a job without phases: all of it is inference
                    pass
                async with gates.infer(mb):
                    return await self.job.run(ref, mb, ctx)

            async def one(uid: str) -> None:
                nonlocal finished
                if dead:  # the engine died under an earlier episode: never start
                    return
                with gates.slot():
                    p0 = self._preemptions()
                    try:
                        if uid not in refs:
                            raise KeyError(f"episode {uid!r} is not in the manifest")
                        out = await run_job(refs[uid])
                    except GatesStopped:
                        return
                    except Exception as exc:  # noqa: BLE001 - this episode's EpisodeError
                        category = _classify(exc, getattr(self.job, "nv", None))
                        sink.add(errors=[self._error(uid, ctx.shard, exc, category)])
                        if category == "cuda_fatal":  # the engine is dead: fail fast
                            err = ShardInfraError(ctx.shard.shard_id,
                                                  {uid: f"{type(exc).__name__}: {exc}"})
                            dead.append(err)
                            gates.stopped = True
                            raise err from exc
                    else:
                        digest = pol.as_json()["digest"]
                        sink.add(records=self._engine_facts(out.records, p0),
                                 audits=[{**a, "batch_policy": digest} for a in out.audits])
                finished += 1
                if finished % max(1, s.flush_every) == 0:
                    await asyncio.to_thread(sink.flush)

            tasks = [asyncio.ensure_future(one(u)) for u in todo]
            try:
                await asyncio.gather(*tasks)
            except BaseException as exc:
                if active is not None and isinstance(exc, asyncio.CancelledError
                                                     | KeyboardInterrupt):
                    # SIGINT: asyncio.run cancels this task. Flush + mark now, before the
                    # batcher closes and the loop's teardown waits on any thread (review 6 M4).
                    self._end(active, preempted=True)
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            finally:
                self.last_stats = mb.audit()["batcher"]["stats"]
                self.last_gates = gates.as_json()
