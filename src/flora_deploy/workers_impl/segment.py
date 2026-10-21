"""Balanced SegmentWorker core (P1b-1): ``workers.ShardWorker`` around ``stages.segment``.

``BalancedShardRunner`` is shared by the segment and caption workers. It follows the worker
contract of ``workers.py`` exactly as ``workers_impl.reward.RewardShardRunner`` does (Volume
refresh, step 0 superseded -> ``None``, attempt marker first, skip done, skip upstream-poisoned,
per-episode isolation into ``EpisodeError`` files, flush + ``mark_preempted`` on SIGINT,
``mark_exception`` + re-raise on a python failure, receipt last naming every part) and reuses
its ``ShardPartSink``. What differs is step 4: the shard's episodes run as coroutines on one
event loop, at most ``max_inflight_episodes`` at a time, each inside
``MicroBatcher.episode()``, so their model calls share ``llm.generate`` rounds (D14). An
episode's records are added to the sink when it finishes; the sink is copied to the store every
``flush_every`` finished episodes and at the end (and by ``preempt`` / ``@modal.exit``).

The per-stage work is an ``EpisodeJob``: ``SegmentEpisodeJob`` (here) and
``workers_impl.caption.CaptionEpisodeJob``. Errors are classified by ``_classify`` (review 6
M1), which reuses segcap's classifiers per job phase (``phase``): ``inputs`` (index receipt,
upstream records) by ``classify_input_error``, ``nvidia`` (NVIDIA's probe and stage) by
``classify_nvidia_error`` (NVIDIA's classes, ``ValueError`` input checks, its builtin
``RuntimeError`` rejections and ``ContextBudgetError`` are ``nv_failure``), ``records`` (our
record building) never NVIDIA's. Only ``nv_failure`` is terminal; an ``infra`` / ``python`` /
``oom`` episode error makes the shard raise ``ShardInfraError`` (no receipt, retried). A CUDA
fault (``cuda_fatal``) stops the shard at once and calls ``fatal`` (``os._exit``) after the mark.

Records are uncertified (``uncertified_stack``) until certificates exist; the sink stamps the
job's stage key (``JobFiles.key``, which covers the engine spec: ``config.hashing``) as
``provenance.config_hash`` and keeps the stage's own spec hash as ``provenance.engine_hash``;
the engine audit goes to the audit lines. The receipt carries the stage key (review 6 L3).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flora_deploy.engines.base import BatchEngine
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
from flora_deploy.records import StageRecord
from flora_deploy.workers_impl.reward import (
    ShardInfraError,
    ShardPartSink,
    VideoResolver,
    _container_ks,
    cuda_fatal,
    default_container_id,
    parse_index_receipt,
)
from flora_deploy.workers_impl.segcap import classify_input_error, classify_nvidia_error

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


def _nvidia_class(nv, cls: type) -> bool:
    alias = getattr(nv, "alias", None)
    finder = getattr(nv, "_finder", None)
    mod = getattr(cls, "__module__", "") or ""
    if alias and (mod == alias or mod.startswith(alias + ".")):
        return True
    owns = getattr(finder, "owns", None)
    try:
        return bool(owns and owns(mod))
    except Exception:  # noqa: BLE001
        return False


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
    ``nv_failure`` | ``infra`` | ``python`` | ``cuda_fatal`` | ``oom``. Reuses segcap's
    classifiers (``classify_input_error`` for the ``inputs`` phase, ``classify_nvidia_error``
    for the ``nvidia`` phase, the default), so the two paths cannot drift; the ``records``
    phase (our record building) is never an NVIDIA verdict."""
    from flora_deploy.nv.budget import ContextBudgetError
    from flora_deploy.stages.segment import EngineRequestError

    msg = f"{type(exc).__name__}: {exc}"
    from flora_deploy.engines.offline_llm import engine_dead

    if cuda_fatal(msg) or engine_dead(msg):  # the engine is unusable (offline_llm.EngineDead)
        return "cuda_fatal"
    if "out of memory" in msg.lower() or type(exc).__name__ == "OutOfMemoryError":
        return "oom"
    where = getattr(exc, "flora_phase", None) or "nvidia"
    if isinstance(exc, _PLUMBING + (EngineRequestError,)):
        return "infra"
    if where == "inputs":
        kind = classify_input_error(exc)
    elif where == "records":
        kind = "infra"
    elif isinstance(exc, ContextBudgetError):  # NVIDIA's InputErrorForModel guard
        kind = "nv_failure"
    else:
        kind, _ = classify_nvidia_error(exc)
        if kind == "infra" and nv is not None and \
                any(_nvidia_class(nv, c) for c in type(exc).__mro__):
            kind = "nv_failure"
    if kind == "infra" and not isinstance(exc, OSError):
        return "infra" if where == "inputs" else "python"
    return kind


def classify_error(nv, exc: BaseException) -> str:
    """``EpisodeError.kind`` for :func:`_classify`'s category: only ``nv_failure`` is
    terminal; everything else is ``infra`` (retried, never excused)."""
    return "nv_failure" if _classify(exc, nv) == "nv_failure" else "infra"


# ---------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------


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

        with phase("inputs"):
            inp = await asyncio.to_thread(self.inputs, ref)
        t0 = time.perf_counter()
        with phase("nvidia"):
            ep = await asyncio.to_thread(S.prepare_videos, self.nv, inp.paths, inp.profile,
                                         yam32=inp.yam32, instruction=inp.instruction)
            t1 = time.perf_counter()
            res = await self.seg.segment(batcher, ep)
        t2 = time.perf_counter()
        with phase("records"):
            return self._outcome(ref, inp, ep, res, t0, t1, t2)

    def _outcome(self, ref, inp, ep, res, t0, t1, t2) -> EpisodeOutcome:
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
                 "timings_s": {"probe": round(t1 - t0, 3), "segment": round(t2 - t1, 3)}}
        return EpisodeOutcome([rec], [audit])


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancedWorkerSettings:
    max_inflight_episodes: int = 32
    flush_every: int = 8
    max_batch: int = 128
    max_wait_s: float = 2.0
    max_batch_bytes: int = 3 << 30
    max_pending_bytes: int = 6 << 30
    local_dir: str = "/tmp/flora-balanced"
    gpu: str | None = None
    container_id: str = field(default_factory=default_container_id)


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
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._end(active, error=msg)
            infra = exc.infra.values() if isinstance(exc, ShardInfraError) else ()
            if self.fatal is not None and (cuda_fatal(msg) or any(map(cuda_fatal, infra))):
                self.fatal(exc)  # the engine is unusable: replace the container
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
    def _end(self, active: _Active, *, preempted: bool = False, error: str | None = None
             ) -> bool:
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
                             error=error or "error", marker=active.marker)
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

        poisoned: set[str] = set()  # step 3
        if shard.depends_on:
            m = manifests.load()
            poisoned = A.upstream_poisoned(jf, shard, m.shards) - done
            A.write_upstream_failures(jf, shard, poisoned)

        refs = {e.uid: e for e in manifests.base().episodes}
        todo = [u for u in shard.episodes if u not in done and u not in poisoned]
        ctx = ShardContext(jf, shard, manifests, self.container_id)
        if todo:
            asyncio.run(self._episodes(todo, refs, sink, ctx, active))  # step 4
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

    async def _episodes(self, todo: list[str], refs: Mapping[str, EpisodeRef],
                        sink: ShardPartSink, ctx: ShardContext,
                        active: _Active | None = None) -> None:
        s = self.settings
        sem = asyncio.Semaphore(max(1, s.max_inflight_episodes))
        finished = 0
        async with MicroBatcher(self.engine, max_batch=s.max_batch, max_wait_s=s.max_wait_s,
                                max_batch_bytes=s.max_batch_bytes,
                                max_pending_bytes=s.max_pending_bytes,
                                observer=self.observer) as mb:

            dead: list[BaseException] = []

            async def one(uid: str) -> None:
                nonlocal finished
                async with sem, mb.episode():
                    if dead:  # the engine died under an earlier episode: never start
                        return
                    try:
                        if uid not in refs:
                            raise KeyError(f"episode {uid!r} is not in the manifest")
                        out = await self.job.run(refs[uid], mb, ctx)
                    except Exception as exc:  # noqa: BLE001 - this episode's EpisodeError
                        category = _classify(exc, getattr(self.job, "nv", None))
                        sink.add(errors=[self._error(uid, ctx.shard, exc, category)])
                        if category == "cuda_fatal":  # the engine is dead: fail fast
                            err = ShardInfraError(ctx.shard.shard_id,
                                                  {uid: f"{type(exc).__name__}: {exc}"})
                            dead.append(err)
                            raise err from exc
                    else:
                        sink.add(records=out.records, audits=out.audits)
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
