"""RewardWorker core (P1a-5r): ``workers.ShardWorker`` around ``stages.reward``. Modal-agnostic.

:class:`RewardShardRunner` processes one ``reward.terminal`` / ``reward.dense`` shard per
:meth:`~RewardShardRunner.run_shard` call, following the worker contract in ``workers.py``:

0. ``attempts.superseded`` → return ``None`` having written nothing (the shard is looked up with
   ``jobs.deps.StoreManifestSource.shard``, which also finds bisection children);
1. ``attempts.write_attempt`` with the spawn ``token`` and the Modal call id;
2. episodes already in earlier containers' parts (``progress.shard_parts``) are skipped;
3. episodes whose upstream shard was poison-pilled (``attempts.upstream_poisoned``) are reported
   with ``attempts.write_upstream_failures`` and never run (reward shards have no
   ``depends_on`` today; the step is generic);
4. the remaining episodes go through one ``stages.reward.run`` call. Its rounds mirror NVIDIA's
   ``generate`` grouping by default (``RoundLimits(align="nvidia")``) and it hands each round to
   a :class:`ShardPartSink`, which appends the round to local files and copies them whole to the
   store (``JobPaths.part`` / ``audit`` / ``errors``, one ``k`` per flush). Episode inputs come
   from the index receipt each ``EpisodeRef.index_ref`` names: the task and one video per camera
   role, in NVIDIA's collage order (:func:`collage_order`); an episode range inside a
   concatenated file is stream-copy cut to a local MP4 first (:class:`VideoResolver`). An input
   that cannot be resolved, a video NVIDIA's ``Collage`` rejects and an answer NVIDIA's parser
   rejects all become ``EpisodeError`` files (``nv_failure`` for NVIDIA's ``ValueError``
   semantics, ``infra`` otherwise), never a crash;
5. ``KeyboardInterrupt`` (Python's SIGINT) or :meth:`~RewardShardRunner.preempt` (from
   ``@modal.exit``) flushes what is pending and calls ``attempts.mark_preempted``; any other
   exception flushes, calls ``attempts.mark_exception`` and re-raises;
6. ``ShardReceipt`` is written LAST, with ``parts`` = the earlier parts seen in step 2 plus this
   container's own.

Records carry the *job's* config hash in ``provenance.config_hash`` (the job hash covers every
setting that changes outputs, ``config.hashing``; ``progress.shard_parts`` and compose filter on
it). The stage's own hash stays in each audit line as ``config_hash``, next to
``job_config_hash``. ``StageRecord.audit_ref`` is left unset: the audit path names the container,
so it would make the same record from two containers differ (compose would report a conflict);
audits are found under ``JobPaths.audit`` by ``(uid, stage)``.

Rubric rewards are deferred (``stages.reward.check_mode``): a ``reward.rubric`` shard raises
``NotImplementedError`` after its attempt marker, which is then marked with the error.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    ShardKind,
    ShardReceipt,
    safe_key,
)
from flora_deploy.media.types import SourceRef
from flora_deploy.records import StageRecord
from flora_deploy.stages import reward as R

# NVIDIA's collage order (``stages.reward.Episode``): 3 views left/center/right; fewer views put
# the exterior camera first, then the wrists (the WS-H samples: ``image``, ``hand_image``).
COLLAGE_ORDER_3 = ("left_wrist", "main", "right_wrist")
COLLAGE_ORDER = ("main", "left_wrist", "right_wrist")


def collage_order(roles: Iterable[str]) -> tuple[str, ...]:
    """Camera roles in NVIDIA's collage order; unknown role names follow, sorted."""
    roles = set(roles)
    known = COLLAGE_ORDER_3 if roles >= set(COLLAGE_ORDER_3) else COLLAGE_ORDER
    return tuple(r for r in known if r in roles) + tuple(sorted(roles - set(known)))


def mode_for(kind: ShardKind, terminal_mode: str = "terminal") -> str:
    """``stages.reward`` mode of a shard kind (rubric raises: deferred)."""
    if kind == "reward.terminal":
        if terminal_mode not in ("terminal", "terminal_rationale"):
            raise ValueError(f"terminal_mode {terminal_mode!r}")
        return terminal_mode
    if kind == "reward.dense":
        return "dense"
    if kind == "reward.rubric":
        return R.check_mode("rubric")  # raises NotImplementedError
    raise ValueError(f"RewardWorker cannot run shard kind {kind!r}")


# ---------------------------------------------------------------------------
# Episode inputs (index receipts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexInputs:
    """What the worker reads from an episode's index receipt."""

    episode_id: str
    dataset: str
    task: str | None
    sources: dict[str, SourceRef]  # camera role -> source
    envelope_ok: bool | None = None


def parse_index_receipt(data: bytes | str) -> IndexInputs:
    """An index receipt: ``dataclasses.asdict(media.types.EpisodeIndex)`` as JSON (Fractions as
    strings; only ``episode_id``, ``dataset``, ``task``, ``streams.<role>.source`` and
    ``envelope.ok`` are read). TODO(P1a-6b): switch to the index job's reader once it lands."""
    d = json.loads(data)
    sources: dict[str, SourceRef] = {}
    for role, stream in (d.get("streams") or {}).items():
        src = stream.get("source") if isinstance(stream, Mapping) else None
        if not isinstance(src, Mapping) or not src.get("uri"):
            raise ValueError(f"index receipt: stream {role!r} has no source uri")
        sources[str(role)] = SourceRef(uri=str(src["uri"]), start_s=src.get("start_s"),
                                       end_s=src.get("end_s"))
    env = d.get("envelope")
    return IndexInputs(episode_id=str(d["episode_id"]), dataset=str(d["dataset"]),
                       task=d.get("task"), sources=sources,
                       envelope_ok=env.get("ok") if isinstance(env, Mapping) else None)


Cutter = Callable[[SourceRef, Path], Path]


def _stream_copy_cut(source: SourceRef, dest: Path) -> Path:
    from flora_deploy.media.cuts import stream_copy_cut

    return stream_copy_cut(source, dest)


@dataclass
class VideoResolver:
    """Source uri → absolute local video path for NVIDIA's ``Collage``.

    Relative uris resolve against ``media_root`` (e.g. the dataset Volume mount). A source with an
    episode range (one episode of a concatenated LeRobot v3 MP4) is cut without re-encoding
    (``media.cuts.stream_copy_cut``) to ``cut_dir/<uid>/<role>.mp4``, reused if present. A cut
    that would not decode like the source raises NVIDIA-semantics ``ValueError``
    (``NotKeyframeAligned``)."""

    media_root: str | None = None
    cut_dir: str | Path = "/tmp/flora-reward/cuts"
    cutter: Cutter = _stream_copy_cut

    def resolve(self, uid: str, role: str, source: SourceRef) -> str:
        uri = source.uri
        if not os.path.isabs(uri):
            if self.media_root is None:
                raise FileNotFoundError(f"relative video uri {uri!r} and no media_root")
            uri = str(Path(self.media_root) / uri)
        if source.start_s is None and source.end_s is None:
            return uri
        dest = Path(self.cut_dir) / safe_key(uid) / f"{safe_key(role)}.mp4"
        if not dest.exists():
            self.cutter(SourceRef(uri, source.start_s, source.end_s), dest)
        return str(dest)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def _line(row: Any) -> str:
    if isinstance(row, StageRecord):
        return row.to_json()
    obj = dataclasses.asdict(row) if dataclasses.is_dataclass(row) else row
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str)


class ShardPartSink:
    """``stages.reward.PartSink`` for one shard in one container.

    ``write`` appends a round's records / audits / errors to local files for the current ``k``
    and flushes; :meth:`flush` copies each local file whole to the store
    (``JobPaths.part`` / ``audit`` / ``errors`` with this container id and ``k``) and only then
    advances ``k`` and counts the episodes as done / failed. A flush interrupted half-way leaves
    the pending files in place and the next flush rewrites the same paths whole, so calling it
    again (``finally``, ``@modal.exit``) is always safe. Records are re-stamped with the job's
    config hash."""

    AREAS = ("part", "audit", "errors")

    def __init__(self, jf: A.JobFiles, kind: ShardKind, shard_id: str, container_id: str,
                 local_dir: str | Path, *, start_k: int = 0):
        self.jf, self.kind, self.shard_id, self.container_id = jf, kind, shard_id, container_id
        self.local_dir = Path(local_dir) / safe_key(shard_id)
        self.k = start_k
        self.parts: list[str] = []
        self.published: list[str] = []
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self._pending_done: set[str] = set()
        self._pending_failed: set[str] = set()
        self._lock = threading.RLock()
        for area in self.AREAS:
            p = self._local(area)
            if p.exists():  # leftovers of an earlier call of this container: never re-sent
                p.unlink()

    def _local(self, area: str, k: int | None = None) -> Path:
        k = self.k if k is None else k
        return self.local_dir / area / f"{safe_key(self.container_id)}-{k}.jsonl"

    def _dest(self, area: str, k: int) -> str:
        maker = {"part": self.jf.paths.part, "audit": self.jf.paths.audit,
                 "errors": self.jf.paths.errors}[area]
        return maker(self.jf.config_hash, self.kind, self.shard_id, self.container_id, k)

    def _append(self, area: str, rows: Sequence[Any]) -> None:
        if not rows:
            return
        p = self._local(area)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write("".join(_line(r) + "\n" for r in rows))

    def _stamp(self, rec: StageRecord) -> StageRecord:
        return dataclasses.replace(rec, provenance=dataclasses.replace(
            rec.provenance, config_hash=self.jf.config_hash))

    def add(self, records: Sequence[StageRecord] = (), audits: Sequence[dict] = (),
            errors: Sequence[EpisodeError] = ()) -> None:
        with self._lock:
            self._append("part", [self._stamp(r) for r in records])
            self._append("audit", [{**a, "job_config_hash": self.jf.config_hash}
                                   for a in audits])
            self._append("errors", list(errors))
            self._pending_done |= {r.uid for r in records}
            self._pending_failed |= {e.uid for e in errors}

    def write(self, out: R.RoundOutput) -> list[str]:
        self.add(out.records, out.audits, out.errors)
        return self.flush()

    def flush(self) -> list[str]:
        with self._lock:
            local = [(a, self._local(a)) for a in self.AREAS if self._local(a).exists()]
            if not local:
                return []
            k, paths = self.k, []
            for area, p in local:
                dest = self._dest(area, k)
                self.jf.store.put_bytes(dest, p.read_bytes())
                paths.append(dest)
                if area == "part":
                    self.parts.append(dest)
            for _, p in local:
                p.unlink()
            self.published += paths
            self.done |= self._pending_done
            self.failed |= self._pending_failed - self.done
            self._pending_done, self._pending_failed = set(), set()
            self.k = k + 1
            return paths


def _container_ks(paths: Iterable[str], container_id: str) -> list[int]:
    out = []
    head = f"{container_id}-"
    for p in paths:
        name = p.rsplit("/", 1)[-1]
        if name.startswith(head) and name.endswith(".jsonl"):
            tail = name[len(head):-len(".jsonl")]
            if tail.isdigit():
                out.append(int(tail))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardEngine:
    """The engine built once per container (``stages.reward.build_engine`` + NVIDIA's
    processor). ``spec`` set: NVIDIA's group size is its ``max_num_seqs``."""

    llm: Any
    processor: Any
    spec: R.EngineSpec | None = None


def default_container_id() -> str:
    return os.environ.get("MODAL_TASK_ID") or f"local-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class RewardWorkerSettings:
    terminal_mode: str = "terminal"  # or "terminal_rationale" for reward.terminal shards
    preset: str = "exact"
    models: Mapping[str, str] = field(default_factory=dict)
    gpu: str | None = None
    certificate: str | None = None
    config_extra: Mapping[str, Any] = field(default_factory=dict)
    limits: R.RoundLimits | None = None  # default: RoundLimits() (align="nvidia")
    decoder: str = "thread"
    workers: int = 4
    tail_trim_s: float | None = None  # None: NVIDIA's DENSE_TAIL_TRIM_S
    envelope: str = "warn"  # D13: "ignore" certifies regardless of the envelope verdict
    default_task: str | None = None  # used when an index receipt has no task
    media_root: str | None = None
    local_dir: str = "/tmp/flora-reward"
    container_id: str = field(default_factory=default_container_id)


@dataclass
class _Active:
    jf: A.JobFiles
    shard: Shard
    marker: AttemptMarker
    sink: ShardPartSink | None = None
    ended: bool = False


class RewardShardRunner:
    """``ShardWorker`` core for reward shards (module docstring)."""

    def __init__(self, store: Store, engine: RewardEngine, nv, settings: RewardWorkerSettings,
                 *, call_id: Callable[[], str | None] = lambda: "",
                 clock: Callable[[], float] = time.time, resolver: VideoResolver | None = None):
        self.store, self.engine, self.nv, self.settings = store, engine, nv, settings
        self.call_id, self.clock = call_id, clock
        self.container_id = safe_key(settings.container_id)
        self.resolver = resolver or VideoResolver(
            settings.media_root, Path(settings.local_dir) / "cuts")
        self._lock = threading.RLock()
        self._active: _Active | None = None
        self._next_k: dict[tuple[str, str, str], int] = {}  # same-container re-runs

    # -- contract ------------------------------------------------------------------------

    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        jf = A.JobFiles(self.store, job_id, config_hash)
        manifests = StoreManifestSource(jf)
        shard = manifests.shard(shard_id)
        if A.superseded(jf, shard.kind, shard_id, token):  # step 0: write nothing
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
        except KeyboardInterrupt:  # SIGINT (preemption): flush, mark, let Modal retry
            self._end(active, preempted=True)
            raise
        except Exception as exc:
            self._end(active, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with self._lock:
                if self._active is active:
                    self._active = None

    def preempt(self) -> bool:
        """``@modal.exit`` / SIGINT hook: flush the shard in flight and mark it preempted.
        Idempotent; a no-op when no shard is running. Returns True if it marked one."""
        with self._lock:
            active = self._active
        return active is not None and self._end(active, preempted=True)

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
            except Exception:  # noqa: BLE001 - the mark below matters more; parts are redone
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

    def _episode(self, ref: EpisodeRef) -> R.Episode:
        idx = parse_index_receipt(self.store.get_bytes(ref.index_ref))
        if not idx.sources:
            raise ValueError(f"episode {ref.uid!r}: index receipt has no camera streams")
        task = idx.task or self.settings.default_task
        if not str(task or "").strip():
            raise ValueError(f"episode {ref.uid!r}: no task instruction")
        videos = tuple(self.resolver.resolve(ref.uid, role, idx.sources[role])
                       for role in collage_order(idx.sources))
        envelope_ok = True if self.settings.envelope == "ignore" else ref.envelope_ok
        ep = R.Episode(id=ref.episode_id, task=str(task), videos=videos, dataset=ref.dataset,
                       envelope_ok=envelope_ok)
        if ep.uid != ref.uid:
            raise RuntimeError(f"index receipt is for {ep.uid!r}, manifest says {ref.uid!r}")
        R.validate_episodes([ep])
        return ep

    def _run(self, active: _Active, manifests: StoreManifestSource) -> ShardReceipt:
        t0 = time.perf_counter()
        jf, shard, s = active.jf, active.shard, self.settings
        mode = mode_for(shard.kind, s.terminal_mode)

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
        stage = R.STAGES[mode]
        episodes, errors = [], []
        for uid in todo:  # step 4: inputs (per-episode isolation)
            try:
                if uid not in refs:
                    raise KeyError(f"episode {uid!r} is not in the manifest")
                episodes.append(self._episode(refs[uid]))
            except Exception as exc:  # noqa: BLE001 - becomes this episode's EpisodeError
                kind = "nv_failure" if isinstance(exc, ValueError) else "infra"
                errors.append(EpisodeError(
                    uid=uid, stage=stage, shard_id=shard.shard_id,
                    container_id=self.container_id, kind=kind, message=str(exc),
                    diagnostics={"phase": "inputs", "type": type(exc).__name__,
                                 "mode": mode}))
        if errors:
            sink.add(errors=errors)
            sink.flush()
        if episodes:
            R.run(self.engine.llm, self.nv, self.engine.processor, episodes, mode,
                  tail_trim_s=s.tail_trim_s, spec=self.engine.spec, models=dict(s.models),
                  gpu=s.gpu, preset=s.preset, certificate=s.certificate,
                  config_extra=dict(s.config_extra), limits=s.limits, decoder=s.decoder,
                  workers=s.workers, sink=sink,
                  shard_id=shard.shard_id, container_id=self.container_id)
        sink.flush()
        self._next_k[key] = sink.k

        finished = done | sink.done
        failed = (sink.failed | poisoned) - finished
        missing = [u for u in shard.episodes if u not in finished and u not in failed]
        if missing:  # never write a receipt that loses episodes
            raise RuntimeError(f"shard {shard.shard_id!r}: no record or error for {missing}")
        receipt = ShardReceipt(  # step 6: LAST
            shard_id=shard.shard_id, kind=shard.kind, config_hash=jf.config_hash,
            container_id=self.container_id,
            episodes_done=tuple(u for u in shard.episodes if u in finished),
            episodes_failed=tuple(u for u in shard.episodes if u in failed),
            parts=tuple(sorted(set(prior.parts) | set(sink.parts))), gpu=s.gpu,
            wall_seconds=round(time.perf_counter() - t0, 3), finished_utc=utc(self.clock()))
        P.write_receipt(jf, receipt)
        with self._lock:
            active.ended = True
        return receipt
