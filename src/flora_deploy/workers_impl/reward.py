"""RewardWorker core (P1a-5r): ``workers.ShardWorker`` around ``stages.reward``. Modal-agnostic.

:class:`RewardShardRunner` processes one ``reward.terminal`` / ``reward.dense`` shard per
:meth:`~RewardShardRunner.run_shard` call, following the worker contract in ``workers.py``:

Before step 0 a Volume store is refreshed (``store.refresh()``), so a warm container never reads
a stale ledger or stale parts.

0. ``attempts.superseded`` → return ``None`` having written nothing (the shard is looked up with
   ``jobs.deps.StoreManifestSource.shard``, which also finds bisection children);
1. ``attempts.write_attempt`` with the spawn ``token`` and the Modal call id;
2. episodes already in earlier containers' parts (``progress.shard_parts``) are skipped;
3. episodes whose upstream shard was poison-pilled (``attempts.upstream_poisoned``) are reported
   with ``attempts.write_upstream_failures`` and never run (reward shards have no
   ``depends_on`` today; the step is generic);
4. the remaining episodes go through one ``stages.reward.run`` call. Its rounds mirror NVIDIA's
   ``generate`` grouping by default (``RoundLimits(align="nvidia")``), over the **job's**
   planned episode list (``planned=`` the manifest's episodes, review 5 M-B; the manifest cuts
   terminal shards on NVIDIA group boundaries), and it hands each round to
   a :class:`ShardPartSink`, which appends the round to local files and copies them whole to the
   store (``JobPaths.part`` / ``audit`` / ``errors``, one ``k`` per flush). Episode inputs come
   from the index receipt each ``EpisodeRef.index_ref`` names: the task and one video per camera
   role, in NVIDIA's collage order (:func:`collage_order`); an episode range inside a
   concatenated file is stream-copy cut to a local MP4 first (:class:`VideoResolver`). A video
   NVIDIA's ``Collage`` rejects and an answer NVIDIA's parser rejects become
   ``EpisodeError(kind="nv_failure")`` files: NVIDIA's own semantics, terminal, excused.
   **Infrastructure errors are not terminal**: an input that cannot be read (missing media, a
   transient store or I/O error, a cut that fails) is retried once in-process and then, like a
   ``generate`` failure the stage reports as ``infra`` (CUDA faults), is written to ``errors/``
   for diagnosis, the finished episodes are flushed, and the shard **raises**
   :class:`ShardInfraError` instead of writing a receipt, so the episode is retried by
   ``modal.Retries`` / the orchestrator's bounded respawns and a persistent failure ends
   ``shard_failed`` (the job ``failed``), never ``done`` without the episode. After a CUDA fault
   the container is unusable: ``RewardShardRunner(fatal=...)`` (``os._exit`` in the Modal worker)
   is called after the marker is written. An episode without a task instruction is a
   configuration error (:class:`TaskInstructionMissing`): the shard fails, nothing is excused;
5. ``KeyboardInterrupt`` (Python's SIGINT) or :meth:`~RewardShardRunner.preempt` (from
   ``@modal.exit``) flushes what is pending and calls ``attempts.mark_preempted``; any other
   exception flushes, calls ``attempts.mark_exception`` and re-raises;
6. ``ShardReceipt`` is written LAST, with ``parts`` = the earlier parts seen in step 2 plus this
   container's own.

Records carry the shard kind's **stage hash** in ``provenance.config_hash``
(``JobFiles.key(kind)``: the key the shard's files live under, ``jobs.store.JobPaths``; the job
hash when the manifest has no stage hashes); ``progress.shard_parts`` and compose accept it. The
stage's own hash stays in each audit line as ``config_hash``, next to ``job_config_hash``.

Concurrent inputs (``@modal.concurrent``): every in-flight shard of the container is tracked, and
:meth:`~RewardShardRunner.preempt` (``@modal.exit``) flushes and marks **each** of them preempted
(M4), not only the latest.

``StageRecord.audit_ref`` is left unset: the audit path names the container, so it would make
the same record from two containers differ (compose would report a conflict);
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
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.deps import StoreManifestSource
from flora_deploy.jobs.manifest import NVIDIA_TERMINAL_GROUP
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
    """An episode's index receipt. Two layouts are read:

    * ``flora_index_receipt_v1`` (``jobs.indexing``, ``flora index``): ``status`` must be
      ``indexed`` (a ``source_error`` receipt raises ``ValueError``); ``dataset.name``; stream
      uris are dataset-relative (resolved against ``RewardWorkerSettings.media_root``); the task
      is the resolved ``instruction_text`` when the receipt has one (``IndexJob.instruction_for``:
      the fixed text, the dataset task, or ``""`` for ``none``), else the receipt's ``task`` for
      the ``dataset_task`` source and ``None`` for ``fixed`` / ``none`` (older receipts; the
      worker then uses ``default_task``: the configured fixed text, ``FLORA_INSTRUCTION_TEXT``);
    * plain ``dataclasses.asdict(media.types.EpisodeIndex)`` JSON (tests, ad-hoc jobs).

    Only the episode id, dataset, task, ``streams.<role>.source`` and ``envelope.ok`` are read."""
    d = json.loads(data)
    task = d.get("task")
    dataset = d.get("dataset")
    if d.get("schema") == "flora_index_receipt_v1":
        if d.get("status") != "indexed":
            raise ValueError(f"episode {d.get('uid')!r} was not indexed: {d.get('error')}")
        dataset = dataset["name"]
        if d.get("instruction_text"):
            task = d["instruction_text"]
        elif d.get("instruction", "dataset_task") != "dataset_task":
            task = None
    sources: dict[str, SourceRef] = {}
    for role, stream in (d.get("streams") or {}).items():
        src = stream.get("source") if isinstance(stream, Mapping) else None
        if not isinstance(src, Mapping) or not src.get("uri"):
            raise ValueError(f"index receipt: stream {role!r} has no source uri")
        sources[str(role)] = SourceRef(uri=str(src["uri"]), start_s=src.get("start_s"),
                                       end_s=src.get("end_s"))
    env = d.get("envelope")
    return IndexInputs(episode_id=str(d["episode_id"]), dataset=str(dataset), task=task,
                       sources=sources,
                       envelope_ok=env.get("ok") if isinstance(env, Mapping) else None)


Cutter = Callable[[SourceRef, Path], Path]


def _stream_copy_cut(source: SourceRef, dest: Path) -> Path:
    from flora_deploy.media.cuts import stream_copy_cut

    return stream_copy_cut(source, dest)


HF_SCHEME = "hf://datasets/"


def parse_hf_root(media_root: str) -> tuple[str, str, str]:
    """``hf://datasets/<owner>/<name>@<revision>[/<subdir>]`` -> (repo, revision, subdir)."""
    if not media_root.startswith(HF_SCHEME):
        raise ValueError(f"not a Hub media root: {media_root!r}")
    rest = media_root[len(HF_SCHEME):]
    head, at, tail = rest.partition("@")
    if not at or head.count("/") != 1:
        raise ValueError(f"media root {media_root!r}: use hf://datasets/<owner>/<name>@<rev>[/sub]")
    revision, _, subdir = tail.partition("/")
    if not revision:
        raise ValueError(f"media root {media_root!r}: a pinned revision is required")
    return head, revision, subdir.strip("/")


def hub_fetch(repo: str, revision: str, filename: str, local_dir: str) -> str:
    """One dataset file at a pinned revision (``huggingface_hub``; retried by the caller)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                           repo_type="dataset", local_dir=local_dir)


Fetcher = Callable[[str, str, str, str], str]


@dataclass
class VideoResolver:
    """Source uri → absolute local video path for NVIDIA's ``Collage``.

    Relative uris resolve against ``media_root``: a directory (a Volume mount holding the staged
    dataset, ``FLORA_MEDIA_ROOT``) or ``hf://datasets/<repo>@<revision>[/<subdir>]``, a Hub
    dataset at its pinned revision, fetched file by file into ``hub_dir`` (C1). A relative uri
    without a media root raises ``FileNotFoundError`` (an infra error: never terminal). A source
    with an episode range (one episode of a concatenated LeRobot v3 MP4) is cut without
    re-encoding (``media.cuts.stream_copy_cut``) to ``cut_dir/<uid>/<role>.mp4``, reused if
    present. A cut that would not decode like the source raises NVIDIA-semantics ``ValueError``
    (``NotKeyframeAligned``)."""

    media_root: str | None = None
    cut_dir: str | Path = "/tmp/flora-reward/cuts"
    cutter: Cutter = _stream_copy_cut
    hub_dir: str | Path = "/tmp/flora-reward/hub"
    fetch: Fetcher = hub_fetch

    def local(self, uri: str) -> str:
        if os.path.isabs(uri):
            return uri
        if self.media_root is None:
            raise FileNotFoundError(f"relative video uri {uri!r} and no media_root")
        if self.media_root.startswith(HF_SCHEME):
            repo, rev, sub = parse_hf_root(self.media_root)
            dest = Path(self.hub_dir) / f"{repo.replace('/', '--')}@{rev[:12]}"
            rel = f"{sub}/{uri}" if sub else uri
            if (dest / rel).is_file():
                return str(dest / rel)
            return str(self.fetch(repo, rev, rel, str(dest)))
        return str(Path(self.media_root) / uri)

    def resolve(self, uid: str, role: str, source: SourceRef) -> str:
        uri = self.local(source.uri)
        if source.start_s is None and source.end_s is None:
            return uri
        dest = Path(self.cut_dir) / safe_key(uid) / f"{safe_key(role)}.mp4"
        if not dest.exists():
            self.cutter(SourceRef(uri, source.start_s, source.end_s), dest)
        return str(dest)


class PreflightError(RuntimeError):
    """The container cannot do any work (media root or store unreachable). Raised at worker
    startup, before the GPU engine is built."""


def check_media_root(media_root: str | None, *, hub_check: Callable[[str, str], Any] | None = None
                     ) -> str:
    """Fail fast (C1): the configured media root must exist and be readable (a directory), or be
    a reachable Hub dataset revision (``hf://``). Returns a one-line description."""
    if not media_root:
        raise PreflightError("FLORA_MEDIA_ROOT is not set: index receipts hold dataset-relative "
                             "video paths and this container cannot resolve them (the app "
                             "factory sets it from input.stage_to or the Hub dataset)")
    if media_root.startswith(HF_SCHEME):
        repo, rev, _sub = parse_hf_root(media_root)
        try:
            if hub_check is not None:
                hub_check(repo, rev)
            else:
                from huggingface_hub import HfApi

                HfApi().dataset_info(repo, revision=rev)
        except Exception as exc:  # noqa: BLE001 - reported with the reason
            raise PreflightError(f"Hub dataset {repo}@{rev} is not reachable: {exc!r}") from exc
        return f"hub {repo}@{rev}"
    root = Path(media_root)
    if not root.is_dir():
        raise PreflightError(f"media root {media_root} does not exist or is not a directory "
                             "(is the dataset Volume mounted and staged?)")
    if not os.access(root, os.R_OK | os.X_OK):
        raise PreflightError(f"media root {media_root} is not readable")
    try:
        next(iter(root.iterdir()), None)
    except OSError as exc:
        raise PreflightError(f"media root {media_root} cannot be listed: {exc!r}") from exc
    return f"dir {media_root}"


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
    again (``finally``, ``@modal.exit``) is always safe. Records are re-stamped with the kind's
    stage hash (``JobFiles.key``) and the kind's certification binding (``binding``:
    ``workers_impl.binding``). ``failed`` holds terminal exclusions only; episodes with an
    ``infra`` error are in ``infra`` (uid -> message) and never count as failed."""

    AREAS = ("part", "audit", "errors")

    def __init__(self, jf: A.JobFiles, kind: ShardKind, shard_id: str, container_id: str,
                 local_dir: str | Path, *, start_k: int = 0, binding: Any = None):
        self.jf, self.kind, self.shard_id, self.container_id = jf, kind, shard_id, container_id
        # ``workers_impl.binding.CertBinding`` of this kind: embodiment / cert_key / gpu
        self.binding = binding
        self.local_dir = Path(local_dir) / safe_key(shard_id)
        self.k = start_k
        self.parts: list[str] = []
        self.published: list[str] = []
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.infra: dict[str, str] = {}
        self._pending_done: set[str] = set()
        self._pending_failed: set[str] = set()
        self._pending_infra: dict[str, str] = {}
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
        key, own = self.jf.key(self.kind), rec.provenance.config_hash
        keep = rec.provenance.engine_hash
        if keep is None and own and own not in (key, self.jf.config_hash):
            keep = own  # the stage's own hash is not lost (review 6 M3)
        from flora_deploy.workers_impl.binding import stamp

        return stamp(dataclasses.replace(rec, provenance=dataclasses.replace(
            rec.provenance, config_hash=key, engine_hash=keep)), self.binding)

    def add(self, records: Sequence[StageRecord] = (), audits: Sequence[dict] = (),
            errors: Sequence[EpisodeError] = ()) -> None:
        with self._lock:
            self._append("part", [self._stamp(r) for r in records])
            self._append("audit", [{**a, "job_config_hash": self.jf.config_hash}
                                   for a in audits])
            self._append("errors", list(errors))
            self._pending_done |= {r.uid for r in records}
            self._pending_failed |= {e.uid for e in errors if e.kind != "infra"}
            self._pending_infra.update({e.uid: e.message for e in errors if e.kind == "infra"})

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
            self.infra.update({u: m for u, m in self._pending_infra.items()
                               if u not in self.done})
            self._pending_done, self._pending_failed = set(), set()
            self._pending_infra = {}
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


class ShardInfraError(RuntimeError):
    """Some episodes of the shard hit infrastructure errors: no receipt; the shard is retried."""

    def __init__(self, shard_id: str, infra: Mapping[str, str]):
        self.infra = dict(infra)
        first = next(iter(self.infra.items()))
        super().__init__(f"shard {shard_id!r}: {len(self.infra)} episode(s) hit infrastructure "
                         f"errors and will be retried (e.g. {first[0]}: {first[1][:300]})")


class TaskInstructionMissing(RuntimeError):
    """A reward episode has no task instruction: a configuration error, never excused."""


CUDA_FATAL_MARKERS = ("CUDA error", "cudaError", "CUBLAS_STATUS", "CUDNN_STATUS",
                      "illegal memory access", "device-side assert", "NCCL error")


def cuda_fatal(message: str) -> bool:
    """A CUDA fault leaves the engine unusable: the container must not run another input."""
    return any(m in message for m in CUDA_FATAL_MARKERS)


INPUT_RETRIES = 1  # in-process retries of an episode's input loading (transient I/O)


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
    default_task: str | None = None  # the fixed instruction (``FLORA_INSTRUCTION_TEXT``)
    # shard kind -> ``workers_impl.binding.CertBinding`` stamped into its records
    cert_bindings: Mapping[str, Any] = field(default_factory=dict)
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
                 clock: Callable[[], float] = time.time, resolver: VideoResolver | None = None,
                 episodes: Callable[[EpisodeRef], R.Episode] | None = None,
                 fatal: Callable[[BaseException], None] | None = None):
        """``episodes``: EpisodeRef -> ``stages.reward.Episode`` (default: from the index receipt
        ``EpisodeRef.index_ref`` names, :meth:`episode_from_index`). ``fatal``: called after the
        marker is written when an error leaves the engine unusable (a CUDA fault); the Modal
        worker passes ``os._exit`` so the container is replaced."""
        self.store, self.engine, self.nv, self.settings = store, engine, nv, settings
        self.episode_for = episodes or self.episode_from_index
        self.call_id, self.clock = call_id, clock
        self.container_id = safe_key(settings.container_id)
        self.resolver = resolver or VideoResolver(
            settings.media_root, Path(settings.local_dir) / "cuts")
        self.fatal = fatal
        self._lock = threading.RLock()
        self._active: dict[int, _Active] = {}  # one per in-flight input (M4)
        self._next_k: dict[tuple[str, str, str], int] = {}  # same-container re-runs

    # -- contract ------------------------------------------------------------------------

    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        refresh = getattr(self.store, "refresh", None)
        if callable(refresh):  # Volume stores: see current ledgers / parts in a warm container
            refresh()
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
            self._active[id(active)] = active
        try:
            return self._run(active, manifests)
        except KeyboardInterrupt:  # SIGINT (preemption): flush, mark, let Modal retry
            self._end(active, preempted=True)
            raise
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._end(active, error=msg)
            if self.fatal is not None and cuda_fatal(msg):
                self.fatal(exc)
            raise
        finally:
            with self._lock:
                self._active.pop(id(active), None)

    def preempt(self) -> bool:
        """``@modal.exit`` / SIGINT hook: flush **every** shard in flight (one per concurrent
        input) and mark each preempted. Idempotent; a no-op when no shard is running. Returns
        True if it marked at least one."""
        with self._lock:
            actives = list(self._active.values())
        marked = [self._end(a, preempted=True) for a in actives]
        return any(marked)

    def on_exit(self) -> None:
        """The ``@modal.exit`` hook (runs after SIGINT, and on a normal scale-down)."""
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

    def episode_from_index(self, ref: EpisodeRef) -> R.Episode:
        idx = parse_index_receipt(self.store.get_bytes(ref.index_ref))
        if not idx.sources:
            raise ValueError(f"episode {ref.uid!r}: index receipt has no camera streams")
        task = idx.task or self.settings.default_task
        if not str(task or "").strip():
            raise TaskInstructionMissing(
                f"episode {ref.uid!r}: no task instruction (the index receipt has none and "
                "FLORA_INSTRUCTION_TEXT is unset); NVIDIA's reward prompts need one: use "
                "episode_instruction.source dataset_task with tasks, or fixed + text")
        roles = collage_order(idx.sources)
        videos = tuple(self.resolver.resolve(ref.uid, role, idx.sources[role]) for role in roles)
        sources = tuple((role, idx.sources[role].uri, idx.sources[role].start_s,
                         idx.sources[role].end_s) for role in roles)  # provenance (L4)
        envelope_ok = True if self.settings.envelope == "ignore" else ref.envelope_ok
        ep = R.Episode(id=ref.episode_id, task=str(task), videos=videos, dataset=ref.dataset,
                       envelope_ok=envelope_ok, sources=sources)
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
                             start_k=start_k, binding=s.cert_bindings.get(shard.kind))
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
            if uid not in refs:
                raise KeyError(f"episode {uid!r} is not in the manifest")
            for attempt in range(INPUT_RETRIES + 1):
                try:
                    episodes.append(self.episode_for(refs[uid]))
                    break
                except TaskInstructionMissing:
                    raise  # configuration: fail the shard, never excuse the episode
                except ValueError as exc:  # NVIDIA semantics (parse, Collage, cut alignment)
                    errors.append(EpisodeError(
                        uid=uid, stage=stage, shard_id=shard.shard_id,
                        container_id=self.container_id, kind="nv_failure", message=str(exc),
                        diagnostics={"phase": "inputs", "type": type(exc).__name__,
                                     "mode": mode}))
                    break
                except Exception as exc:  # noqa: BLE001 - infra: retried, then not terminal
                    if attempt < INPUT_RETRIES:
                        continue
                    errors.append(EpisodeError(
                        uid=uid, stage=stage, shard_id=shard.shard_id,
                        container_id=self.container_id, kind="infra",
                        message=f"{type(exc).__name__}: {exc}",
                        diagnostics={"phase": "inputs", "type": type(exc).__name__,
                                     "mode": mode, "attempts": attempt + 1}))
        if errors:
            sink.add(errors=errors)
            sink.flush()
        # NVIDIA groups follow the *job's* planned order (review 4 M3, review 5 M-B): the
        # manifest's episode list is the jobs file NVIDIA's script would read whole. A shard
        # (or a bisection child) gets the batch-mates the job gives it, whatever the shard
        # sizing; a group the shard does not hold whole is flagged, never "aligned".
        planned = [e.uid for e in manifests.base().episodes]
        if episodes:
            group = int((self.engine.spec.args if self.engine.spec else {}).get(
                "max_num_seqs", NVIDIA_TERMINAL_GROUP))
            if shard.kind == "reward.terminal" and group != NVIDIA_TERMINAL_GROUP:
                warnings.warn(f"engine max_num_seqs {group} != the manifest's terminal group "
                              f"{NVIDIA_TERMINAL_GROUP}: shards cut NVIDIA groups (flagged "
                              "unaligned)", RuntimeWarning, stacklevel=2)
            R.run(self.engine.llm, self.nv, self.engine.processor, episodes, mode,
                  tail_trim_s=s.tail_trim_s, spec=self.engine.spec, models=dict(s.models),
                  gpu=s.gpu, preset=s.preset, certificate=s.certificate,
                  config_extra=dict(s.config_extra), limits=s.limits, decoder=s.decoder,
                  workers=s.workers, sink=sink,
                  shard_id=shard.shard_id, container_id=self.container_id,
                  planned=planned)
        sink.flush()
        self._next_k[key] = sink.k

        finished = done | sink.done
        failed = (sink.failed | poisoned) - finished
        infra = {u: m for u, m in sink.infra.items() if u not in finished and u not in failed}
        if infra:  # never terminal: no receipt, the shard is retried (module docstring)
            raise ShardInfraError(shard.shard_id, infra)
        missing = [u for u in shard.episodes if u not in finished and u not in failed]
        if missing:  # never write a receipt that loses episodes
            raise RuntimeError(f"shard {shard.shard_id!r}: no record or error for {missing}")
        receipt = ShardReceipt(  # step 6: LAST
            shard_id=shard.shard_id, kind=shard.kind,
            container_id=self.container_id,
            episodes_done=tuple(u for u in shard.episodes if u in finished),
            episodes_failed=tuple(u for u in shard.episodes if u in failed),
            config_hash=jf.key(shard.kind),
            parts=tuple(sorted(set(prior.parts) | set(sink.parts))), gpu=s.gpu,
            wall_seconds=round(time.perf_counter() - t0, 3), finished_utc=utc(self.clock()))
        P.write_receipt(jf, receipt)
        with self._lock:
            active.ended = True
        return receipt
