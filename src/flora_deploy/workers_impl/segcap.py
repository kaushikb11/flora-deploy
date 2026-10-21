"""SegcapWorker core (P1a-5s): ``workers.ShardWorker`` around the fast ``exact`` pipeline.

:class:`SegcapShardRunner` processes one ``segcap`` shard per :meth:`~SegcapShardRunner.run_shard`
call, following the worker contract in ``workers.py`` exactly as its reference implementation
``workers_impl.reward.RewardShardRunner`` does (whose part sink, index-receipt reader, video
resolver, infra error and CUDA-fault detection it reuses):

Before step 0 a Volume store is refreshed (``store.refresh()``). Every file goes through
``jobs.attempts.JobFiles(store, job_id, config_hash).paths``: ``config_hash`` is the **job** hash
and the paths are re-keyed by the ``segcap`` stage hash from the config's manifest.

0. ``attempts.superseded`` → return ``None`` having written nothing;
1. ``attempts.write_attempt`` with the spawn ``token`` and the Modal call id;
2. episodes already in earlier containers' parts (``progress.shard_parts``: both ``segment`` and
   ``caption`` records present) are skipped;
3. episodes whose upstream shard was poison-pilled are reported (``write_upstream_failures``)
   and never run (``segcap`` shards have no ``depends_on`` today; the step is generic);
4. the remaining episodes' inputs come from their index receipts (``EpisodeRef.index_ref``):
   camera roles → local files (an episode range inside a concatenated file is stream-copy cut,
   ``reward.VideoResolver``), NVIDIA's profile and ``yam32`` flag, and NVIDIA's ``instruction``
   (:func:`instruction_for`: the receipt's resolved ``instruction_text``, index receipts v2,
   else the configured fixed text ``FLORA_INSTRUCTION_TEXT``). They go through one
   :class:`FastExactEngine` batch: NVIDIA's own ``api.DataEnginePipeline().annotate`` per episode
   on the resident models (``stages.reference.annotate_batch``). As **each** episode finishes (in
   its own thread) its ``segment`` and ``caption`` ``StageRecord``s and one audit line (NVIDIA's
   full result with our measurements and the runtime provenance) are appended and flushed whole
   to the store. An episode NVIDIA rejects (``InputError``) and an input NVIDIA's semantics
   reject (``ValueError``: a source-error receipt, no cameras, a cut that would not decode like
   the source) become ``EpisodeError(kind="nv_failure")`` files, one per stage: terminal,
   excused. **Infrastructure errors are not terminal**: an input that cannot be read (missing
   index receipt or media, a transient store or I/O error) is retried once in-process; a
   persistent one, and an infra failure inside NVIDIA's run (e.g. a CUDA fault), is written to
   ``errors/`` as ``kind="infra"`` for diagnosis, the finished episodes are flushed, and the
   shard **raises** :class:`~flora_deploy.workers_impl.reward.ShardInfraError` instead of writing
   a receipt (``modal.Retries`` / the orchestrator's bounded respawns retry it). After a CUDA
   fault the container is unusable: ``SegcapShardRunner(fatal=...)`` (``os._exit`` in the Modal
   worker) is called after the marker is written. A ``fixed`` instruction with no text anywhere
   is a configuration error (:class:`InstructionMissing`): the shard fails, nothing is excused.
   The runner refuses (``DeterminismError``) to run when the engine's D-1 determinism mode is not
   the configured one (``SegcapWorkerSettings.determinism``: ``RunSettings.determinism`` via
   ``FLORA_DETERMINISM``);
5. ``KeyboardInterrupt`` (SIGINT) or :meth:`~SegcapShardRunner.preempt` (``@modal.exit``) flushes
   and calls ``attempts.mark_preempted`` for **every** shard in flight (``@modal.concurrent``
   inputs); any other exception flushes, ``mark_exception``, re-raises;
6. ``ShardReceipt`` LAST, ``parts`` = earlier parts + this container's own, ``config_hash`` = the
   stage hash (``JobFiles.key``), ``episodes_failed`` = terminal exclusions only.

Records: ``provenance.config_hash`` is the stage hash (stamped by the part sink);
``certified`` per the episode's envelope verdict (``EpisodeRef.envelope_ok``, D13;
``envelope="ignore"`` certifies regardless); ``provenance.engine`` is the engine label
(``nvidia-hf``, ``+deterministic`` when the D-1 mode is on, since that is a different
reference); ``provenance.source`` the camera sources. Everything else that describes how the
output was produced (exact profile, active ``accel`` patches, ``gpu_thread``, determinism record,
BF16 Captioner receipt, image fingerprint, ``NvFamily.audit_report``) is in the episode's audit
line (``JobPaths.audit``), together with ``identity_calls`` for ``accel.identity.compare_runs``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.deps import StoreManifestSource
from flora_deploy.jobs.manifest import stages_of
from flora_deploy.jobs.orchestrator import utc
from flora_deploy.jobs.store import Store
from flora_deploy.jobs.types import EpisodeError, EpisodeRef, ShardReceipt, safe_key
from flora_deploy.media.types import SourceRef
from flora_deploy.records import Provenance
from flora_deploy.workers_impl.reward import (
    INPUT_RETRIES,
    ShardInfraError,
    ShardPartSink,
    VideoResolver,
    _Active,
    _container_ks,
    cuda_fatal,
    default_container_id,
    parse_index_receipt,
)

ENGINE_LABEL = "nvidia-hf"
AUDIT_SCHEMA = "flora_segcap_audit_v1"
# G3 scheduling (docs/exact-cost.md). Neither changes any model input (only which episode starts
# when), so neither is hashed: see ``config.schema.ExactScheduling``.
EPISODE_ORDERS = ("plan", "longest_first")
MAX_OVERLAP_SHARDS = 4
SCHEDULING_ENV = "FLORA_SEGCAP_SCHEDULING"  # JSON {"episode_order", "max_shards"}
# EpisodeRef.pool -> (NVIDIA profile, yam32), as ``jobs.indexing`` maps it (receipts carry both).
POOL_PROFILES = {"human": ("human", False), "yam": ("yam", False), "yam32": ("yam", True),
                 "dex": ("dex", False), "tianji": ("tianji", False)}


# ---------------------------------------------------------------------------
# Episode inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegcapInputs:
    """One episode, ready for ``stages.reference.annotate_batch``."""

    uid: str
    profile: str
    long_windows: bool
    instruction: str
    videos: dict[str, str]  # role -> local path
    sources: dict[str, SourceRef]  # role -> source (provenance)
    duration_s: float = 0.0

    def item(self) -> dict:
        return {"episode": self.uid, "videos": dict(self.videos), "profile": self.profile,
                "long_windows": self.long_windows, "instruction": self.instruction,
                "duration_s": self.duration_s}


class InstructionMissing(RuntimeError):
    """A ``fixed`` instruction with no text (neither the index receipt's ``instruction_text``
    nor ``FLORA_INSTRUCTION_TEXT``): a configuration error, never an excused episode."""


def instruction_for(receipt: Mapping[str, Any], task: str | None,
                    fixed_text: str | None) -> str:
    """NVIDIA's ``instruction`` for an episode (``jobs.indexing.IndexJob.instruction_for``).

    Index receipts v2 carry the resolved ``instruction_text`` (the fixed text, the dataset task,
    or ``""`` for ``none``), which wins. A ``fixed`` receipt without it (v1) uses the configured
    fixed text (``FLORA_INSTRUCTION_TEXT``); neither raises :class:`InstructionMissing`."""
    source = receipt.get("instruction", "dataset_task")
    if source == "none":
        return ""
    text = receipt.get("instruction_text")
    if source == "fixed":
        if isinstance(text, str) and text.strip():
            return text
        if fixed_text and fixed_text.strip():
            return fixed_text
        raise InstructionMissing(
            "index receipt asks for a fixed instruction but neither it (instruction_text) nor "
            "the worker (FLORA_INSTRUCTION_TEXT) has the text: set "
            "pipeline.episode_instruction.text")
    if isinstance(text, str):
        return text
    return task or ""


def source_provenance(sources: Mapping[str, SourceRef]) -> dict[str, dict[str, Any]]:
    return {role: {"uri": s.uri, "range_s": None if s.start_s is None and s.end_s is None
                   else [s.start_s, s.end_s]} for role, s in sorted(sources.items())}


# ---------------------------------------------------------------------------
# Failure taxonomy (review 5 H2)
# ---------------------------------------------------------------------------


class InputRejected(ValueError):
    """An episode's index receipt says NVIDIA's input checks cannot pass (``status`` is not
    ``indexed``: ``flora index`` recorded a source error). NVIDIA semantics: terminal."""


# NVIDIA's ``generate`` runs with ``max_time=300`` (``batch_utils/models.py``); an incomplete call
# whose GPU-lock hold reached this is a time stop, not an input-determined rejection.
NVIDIA_MAX_TIME_S = 300.0
TIME_STOP_S = NVIDIA_MAX_TIME_S - 10.0
# The linked family's modules are imported under ``flora_nv_<family>_<rev8>``
# (``nv.linker.ALIAS_PREFIX``): an exception class defined there is NVIDIA's own.
NV_MODULE_PREFIX = "flora_nv_"
# NVIDIA rejections raised as builtin RuntimeError (no class of their own): episode semantics.
NVIDIA_EPISODE_REJECTIONS = (
    "The pipeline returned an invalid caption interval.",  # api.py
    "No common caption input geometry fits",  # vendor/robot_caption_geometry.py
)
# Never an episode verdict: the GPU, the runtime or I/O failed (retry; CUDA faults exit).
INFRA_TYPE_NAMES = frozenset({"OutOfMemoryError", "AcceleratorError", "NotReady",
                              "CudaError", "CUDAError"})
INCOMPLETE_OUTPUT = "did not return a complete output"


def _split_type(type_name: str) -> tuple[str, str]:
    module, _, name = type_name.rpartition(".")
    return module, name


def _builtin_oserror(module: str, name: str) -> bool:
    import builtins

    cls = getattr(builtins, name, None) if module in ("builtins", "") else None
    return isinstance(cls, type) and issubclass(cls, OSError)


def classify_input_error(exc: BaseException) -> str:
    """Inputs phase. ``nv_failure`` (terminal) only for NVIDIA-semantics rejections: a receipt
    that records a source error (:class:`InputRejected`), a media check (``media.index.
    MediaError``: e.g. ``NotKeyframeAligned``, a cut that would not decode like the source) and
    NVIDIA's own ``InputError``. Everything else (a truncated / undecodable index receipt, a
    missing file, a store error, a receipt without streams) is ``infra``: retried, never
    excused."""
    from flora_deploy.media.index import MediaError
    from flora_deploy.stages.reference import NvidiaInputError

    if isinstance(exc, InputRejected | MediaError | NvidiaInputError):
        return "nv_failure"
    return "infra"


def classify_nvidia_error(err: BaseException, timing: Mapping[str, Any] | None = None
                          ) -> tuple[str, dict[str, Any]]:
    """NVIDIA phase: ``(kind, diagnostics)`` for the error an episode raised inside NVIDIA's
    ``annotate`` (``stages.reference.transport_error``: the original qualified type name and
    message). ``timing``: the episode's GPU-call evidence (:meth:`FastExactEngine.
    episode_timing`). In order:

    1. CUDA / torch / OOM / ``NotReady`` / ``OSError`` → ``infra``;
    2. ``InputErrorForModel`` "did not return a complete output" whose longest call reached
       NVIDIA's ``max_time`` → ``infra`` (``time_stop``: timing-, not input-determined);
    3. NVIDIA's input rejections (``InputError``, ``InputErrorForModel``, ``ValueError``) →
       ``nv_failure``;
    4. a class defined in the linked NVIDIA family (``flora_nv_*`` modules, e.g.
       ``FineSegmentationOutputError``, ``FineWorkerError``, ``U2WorkerError``) → ``nv_failure``;
    5. NVIDIA's known builtin-``RuntimeError`` episode rejections → ``nv_failure``;
    6. anything else (our code, an unknown builtin error) → ``infra``."""
    from flora_deploy.stages.reference import NvidiaInputError

    type_name = getattr(err, "type_name", None) or \
        f"{type(err).__module__}.{type(err).__qualname__}"
    message = str(getattr(err, "message", None) or err)
    module, name = _split_type(type_name)
    diag: dict[str, Any] = {}
    if timing:
        diag.update(timing)
    root = module.split(".", 1)[0]
    if (cuda_fatal(message) or "out of memory" in message.lower() or name in INFRA_TYPE_NAMES
            or root == "torch" or _builtin_oserror(module, name)):
        return "infra", diag
    if name == "InputErrorForModel" and INCOMPLETE_OUTPUT in message and timing \
            and float(timing.get("max_call_s") or 0.0) >= TIME_STOP_S:
        diag["time_stop"] = True
        return "infra", diag
    if isinstance(err, NvidiaInputError | ValueError):
        return "nv_failure", diag
    if root.startswith(NV_MODULE_PREFIX):
        return "nv_failure", diag
    if module in ("builtins", "") and name == "RuntimeError" and \
            message.strip() in NVIDIA_EPISODE_REJECTIONS:
        return "nv_failure", diag
    return "infra", diag


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


OnDone = Callable[[dict, "dict | None", "Exception | None"], None]


class SegcapEngine(Protocol):
    concurrency: int

    def annotate(self, items: list[dict], *, on_done: OnDone) -> dict:
        """``stages.reference.annotate_batch`` over ``items``; ``on_done`` per episode."""
        ...

    def label(self) -> str: ...

    def runtime_provenance(self) -> dict: ...

    def check(self) -> dict | None:
        """Before an episode's records are written: ``nv.audit()`` + determinism verify."""
        ...


class BatchCancelled(RuntimeError):
    """A queued episode was not started: the batch was interrupted (preemption)."""


def _install_call_timer(lock: Any) -> dict[int, list[float]]:
    """Time every GPU-lock hold (one NVIDIA ``generate`` call, taken in the episode's thread)
    per thread, so an episode that fails still has timing evidence (review 5 M4). The lock's
    behaviour is unchanged; installed once per lock."""
    calls = getattr(lock, "_flora_call_times", None)
    if calls is not None:
        return calls
    from contextlib import contextmanager

    calls = {}
    orig = lock.held

    @contextmanager
    def held(label=None):
        with orig(label):
            t = time.perf_counter()
            try:
                yield
            finally:
                calls.setdefault(threading.get_ident(), []).append(time.perf_counter() - t)

    lock.held = held
    lock._flora_call_times = calls
    return calls


@dataclass
class FastExactEngine:
    """The resident fast ``exact`` composite the Modal ``ReferenceWorker`` builds once per
    container (``@modal.enter``): NVIDIA's family ``nv`` with ``install_nvidia_entry`` /
    ``install_gpu_lock`` (and ``accel.gpu_thread``) active, its ``CachedSession``, the GPU lock,
    the processor-clone pool and the ``accel.ActivePatches``.

    ``nv.audit()`` runs before every batch, before every episode starts and after it finished
    (both NVIDIA stages ran in between), so a family module imported or changed mid-run fails the
    episode's records closed. With a ``determinism`` record, ``determinism.verify`` runs before
    each episode's records are written (D-1: in the ``deterministic`` mode a drift raises; in
    ``nvidia`` mode it is reported in the audit line).

    :meth:`annotate` runs the batch in a helper thread and waits interruptibly: on SIGINT
    (``KeyboardInterrupt`` in the caller) it returns at once (queued episodes never start, the
    in-flight ones finish and flush in the background), so the runner writes the preempt mark
    within Modal's 30 s grace instead of after minutes-long episodes (review 5 H3)."""

    nv: Any
    session: Any
    gpu_lock: Any = None
    clones: Any = None
    accel: Any = None
    gpu_thread: Any = None
    concurrency: int = 4
    profile: Mapping[str, Any] = field(default_factory=dict)
    determinism: Mapping[str, Any] | None = None
    image_fingerprint: str | None = None
    captioner_bf16: Any = None
    progress: Callable[[str, float, str], None] | None = None
    sampler_factory: Callable[[], Any] | None = None
    # G3 shard overlap (docs/exact-cost.md): concurrent ``annotate`` calls (one per shard in
    # flight, ``@modal.concurrent``) share ``concurrency`` episode slots; off: a second
    # concurrent call is refused.
    overlap: bool = False
    _audits: int = 0
    _drift: str | None = None
    _calls: dict = field(default_factory=dict, init=False, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _state: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _batches: int = field(default=0, init=False, repr=False)
    _seq: int = field(default=0, init=False, repr=False)
    _slots: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.gpu_lock is not None:
            self._calls = _install_call_timer(self.gpu_lock)
        if self.overlap:
            from flora_deploy.stages.reference import EpisodeSlots

            if self.concurrency < 2:
                raise ValueError("FastExactEngine(overlap=True) needs concurrency >= 2 (the GPU "
                                 "lock and processor clones of the concurrent composite)")
            self._slots = EpisodeSlots(self.concurrency)

    @property
    def max_batches(self) -> int:
        """How many ``annotate`` calls may run at once (``SegcapShardRunner.max_shards``)."""
        return MAX_OVERLAP_SHARDS if self.overlap else 1

    def _audit(self) -> dict | None:
        report = getattr(self.nv, "audit_report", None)
        if callable(report):
            self._audits += 1
            return report()
        audit = getattr(self.nv, "audit", None)
        if callable(audit):
            self._audits += 1
            audit()
        return None

    def check(self) -> dict | None:
        report = self._audit()
        if self.determinism is not None:
            from flora_deploy.accel import determinism as det

            try:
                det.verify(self.determinism)
                self._drift = None
            except det.DeterminismError as e:
                if self.determinism.get("mode") != det.NVIDIA:
                    raise
                self._drift = str(e)  # NVIDIA's default sets nothing: report, never drop
        return report

    def mode(self) -> str:
        return str((self.determinism or {}).get("mode", "nvidia"))

    def label(self) -> str:
        mode = self.mode()
        return ENGINE_LABEL if mode == "nvidia" else f"{ENGINE_LABEL}+{mode}"

    def runtime_provenance(self) -> dict:
        from flora_deploy.accel import gpu_thread as gt

        det = None
        if self.determinism is not None:
            det = {k: self.determinism.get(k) for k in ("schema", "mode", "in_effect")}
            if self._drift:
                det["flags_changed_since_enter"] = self._drift
        bf16 = self.captioner_bf16
        return {
            "exact_profile": dict(self.profile),
            "accel": self.accel.provenance() if self.accel is not None else None,
            "gpu_thread": (self.gpu_thread.provenance() if self.gpu_thread is not None
                           else gt.disabled_provenance()),
            "determinism": det,
            "image_fingerprint": self.image_fingerprint,
            "captioner_bf16": bf16.provenance() if bf16 is not None else None,
            "concurrency": self.concurrency,
            "overlap": self.overlap,
        }

    def episode_timing(self) -> dict[str, Any]:
        """The calling episode thread's GPU calls so far (call from ``on_done``)."""
        from flora_deploy.stages.reference import TIMING_FLAG_S

        times = list(self._calls.get(threading.get_ident(), ()))
        longest = max(times, default=0.0)
        return {"gpu_calls": len(times), "max_call_s": round(longest, 3),
                "timing_dependent": longest > TIMING_FLAG_S}

    def cancel(self) -> None:
        """Queued episodes of the running batch never start (``@modal.exit``)."""
        self._cancel.set()

    def _on_start(self, ep: dict) -> None:
        if self._cancel.is_set():
            raise BatchCancelled(f"episode {ep.get('episode')!r} not started: interrupted")
        self._calls[threading.get_ident()] = []
        self._audit()

    def annotate(self, items: list[dict], *, on_done: OnDone) -> dict:
        from flora_deploy.accel import gpu_thread as gt

        if self.gpu_thread is not None and gt._find_mark(self.nv.pipeline.generate) \
                is not self.gpu_thread:
            raise RuntimeError("gpu_thread is on but NVIDIA generate() does not run on it")
        with self._state:
            if self._batches and not self.overlap:
                raise RuntimeError("FastExactEngine.annotate called while another batch runs; "
                                   "concurrent shards need FastExactEngine(overlap=True)")
            first = self._batches == 0
            if first:  # a running batch's cancel (preemption) is never undone by a newcomer
                self._cancel.clear()
            self._batches += 1
            self._seq += 1
            seq = self._seq
        try:
            return self._annotate(items, on_done, first=first, seq=seq)
        finally:
            with self._state:
                self._batches -= 1

    def _annotate(self, items: list[dict], on_done: OnDone, *, first: bool, seq: int) -> dict:
        from flora_deploy.stages.reference import annotate_batch

        self._audit()
        box: dict[str, Any] = {}
        finished = threading.Event()
        shared = ({"slots": self._slots, "slot_key": seq, "cancelled": self._cancel.is_set,
                   "reset_lock": first} if self.overlap else {})

        def run() -> None:
            try:
                box["out"] = annotate_batch(
                    self.nv, self.session, items,
                    concurrency=min(self.concurrency, max(len(items), 1)),
                    gpu_lock=self.gpu_lock, clones=self.clones, accel=self.accel,
                    sampler=self.sampler_factory() if self.sampler_factory else None,
                    progress=self.progress, on_start=self._on_start, on_done=on_done,
                    **shared)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                box["exc"] = exc
            finally:
                finished.set()

        threading.Thread(target=run, name="flora-batch", daemon=True).start()
        try:
            while not finished.wait(0.2):  # interruptible: SIGINT lands here
                pass
        except BaseException:
            self._cancel.set()  # queued episodes never start; in-flight ones flush on their own
            raise
        if "exc" in box:
            raise box["exc"]
        return box["out"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegcapWorkerSettings:
    preset: str = "exact"
    models: Mapping[str, str] = field(default_factory=dict)
    gpu: str | None = None
    certificate: str | None = None  # None for exact (reference by construction)
    envelope: str = "warn"  # D13: "ignore" certifies regardless of the envelope verdict
    instruction_text: str | None = None  # NVIDIA's instruction for a ``fixed`` source
    media_root: str | None = None
    # D-1 mode the job was configured (and hashed) with: ``RunSettings.determinism``, the
    # factory's ``FLORA_DETERMINISM``. The engine must run in exactly this mode.
    determinism: str = "nvidia"
    local_dir: str = "/tmp/flora-segcap"
    container_id: str = field(default_factory=default_container_id)
    # G3 scheduling (docs/exact-cost.md; ``config.schema.ExactScheduling``, not hashed):
    # ``episode_order``: "plan" (manifest order) or "longest_first" (LPT by the manifest's
    # ``duration_s``, ties in plan order); ``max_shards``: shards this container runs at once
    # (``@modal.concurrent`` inputs; > 1 needs an engine with ``max_batches`` >= it).
    episode_order: str = "plan"
    max_shards: int = 1


def scheduling_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """``SegcapWorkerSettings`` scheduling fields from ``FLORA_SEGCAP_SCHEDULING`` (JSON written
    by the app factory from ``config.schema.exact_scheduling``); defaults when unset."""
    import os

    env = os.environ if env is None else env
    raw = env.get(SCHEDULING_ENV)
    if not raw:
        return {"episode_order": "plan", "max_shards": 1}
    doc = json.loads(raw)
    out = {"episode_order": str(doc.get("episode_order", "plan")),
           "max_shards": int(doc.get("max_shards", 1))}
    check_scheduling(**out)
    return out


def check_scheduling(episode_order: str, max_shards: int) -> None:
    if episode_order not in EPISODE_ORDERS:
        raise ValueError(f"episode_order {episode_order!r}: use one of {EPISODE_ORDERS}")
    if not 1 <= max_shards <= MAX_OVERLAP_SHARDS:
        raise ValueError(f"max_shards {max_shards}: must be 1..{MAX_OVERLAP_SHARDS}")


def order_episodes(uids: list[str], refs: Mapping[str, EpisodeRef], order: str) -> list[str]:
    """The order a shard's episodes start in. ``longest_first`` is LPT list scheduling: the
    longest episode (manifest ``duration_s``) first, so the shard's makespan is not set by a
    long episode that started last; ties keep plan order. Only start order changes: each
    episode is still one independent NVIDIA ``annotate``."""
    if order == "plan":
        return list(uids)
    if order == "longest_first":
        pos = {u: i for i, u in enumerate(uids)}
        return sorted(uids, key=lambda u: (-float(refs[u].duration_s or 0.0), pos[u]))
    raise ValueError(f"episode_order {order!r}: use one of {EPISODE_ORDERS}")


def certification(ref: EpisodeRef, envelope: str = "warn") -> tuple[bool, str | None]:
    """certified only inside NVIDIA's reference envelope (``stages.reward.certification``)."""
    if envelope == "ignore" or ref.envelope_ok is True:
        return True, None
    if ref.envelope_ok is False:
        return False, "outside_reference_envelope"
    return False, "oracle_ineligible_source"


def engine_mode(engine: Any) -> str:
    """The D-1 determinism mode an engine runs in (``nvidia`` when it carries no record)."""
    from flora_deploy.accel import determinism as det

    return str((getattr(engine, "determinism", None) or {}).get("mode", det.NVIDIA))


class ConcurrentShardRefused(RuntimeError):
    """More shards arrived at once than ``SegcapWorkerSettings.max_shards`` allows (default 1:
    one shard per container). Use ``resources.segcap.max_inputs: 1``, or turn on
    ``pipeline.exact_scheduling.shard_overlap`` (G3), whose engine shares its episode slots
    between the shards in flight."""


class SegcapShardRunner:
    """``ShardWorker`` core for ``segcap`` shards (module docstring)."""

    def __init__(self, store: Store, engine: SegcapEngine, settings: SegcapWorkerSettings, *,
                 call_id: Callable[[], str | None] = lambda: "",
                 clock: Callable[[], float] = time.time, resolver: VideoResolver | None = None,
                 inputs: Callable[[EpisodeRef], SegcapInputs] | None = None,
                 fatal: Callable[[BaseException], None] | None = None):
        """``inputs``: EpisodeRef -> :class:`SegcapInputs` (default: from the index receipt,
        :meth:`inputs_from_index`). ``fatal``: called after the marker is written when an error
        leaves the engine unusable (a CUDA fault); the Modal worker passes ``os._exit`` so the
        container is replaced."""
        from flora_deploy.accel import determinism as det

        det.check_mode(settings.determinism)
        check_scheduling(settings.episode_order, settings.max_shards)
        if settings.max_shards > int(getattr(engine, "max_batches", 1)):
            raise ValueError(
                f"max_shards={settings.max_shards} needs an engine that runs that many batches "
                f"at once (this one: {getattr(engine, 'max_batches', 1)}; "
                "FastExactEngine(overlap=True))")
        self.store, self.engine, self.settings = store, engine, settings
        self.inputs_for = inputs or self.inputs_from_index
        self.call_id, self.clock = call_id, clock
        self.container_id = safe_key(settings.container_id)
        self.resolver = resolver or VideoResolver(
            settings.media_root, Path(settings.local_dir) / "cuts")
        self.fatal = fatal
        self._lock = threading.RLock()
        self._active: dict[int, _Active] = {}  # every in-flight input (marked by preempt)
        self._running = 0  # shards in flight (<= settings.max_shards: ConcurrentShardRefused)
        self._next_k: dict[tuple[str, str, str], int] = {}
        self.last_batch: dict | None = None  # the last annotate batch summary (diagnostics)
        self.skipped_refreshes = 0

    # -- contract ------------------------------------------------------------------------

    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        with self._lock:  # review 5 M1: refuse, before writing anything, an extra input
            if self._running >= self.settings.max_shards:
                raise ConcurrentShardRefused(
                    f"shard {shard_id!r} arrived while {self._running} segcap shard(s) run in "
                    f"this container (max_shards={self.settings.max_shards}): set "
                    "resources.segcap.max_inputs: 1, or pipeline.exact_scheduling.shard_overlap")
            self._running += 1
            others = self._running > 1
        try:
            return self._run_shard(job_id, shard_id, config_hash, token, others=others)
        finally:
            with self._lock:
                self._running -= 1

    def _media_on_store(self) -> bool:
        """The media root lies inside the store's Volume mount (``factory.media_for`` stages
        inputs there when ``stage_to`` names the store Volume): NVIDIA then has Volume files
        open while an episode runs."""
        mount, media = getattr(self.store, "mount", None), self.settings.media_root
        if mount is None or not media or "://" in str(media):
            return False
        try:
            Path(media).resolve().relative_to(Path(mount).resolve())
        except (ValueError, OSError):
            return False
        return True

    def _refresh(self, others: bool) -> None:
        """Volume stores: see current ledgers / parts in a warm container. A Volume reload
        fails (or blanks the mount) while any Volume file is open; ``VolumeStore.refresh``
        guards its own calls, but not NVIDIA's open media files. So while another shard of this
        container runs (G3 overlap) and the media live on the store Volume, the reload is
        skipped: this shard sees the container's view as of the last reload, which the
        contract tolerates (eventually consistent listings; step 0/2 races exist anyway)."""
        refresh = getattr(self.store, "refresh", None)
        if not callable(refresh):
            return
        if others and self._media_on_store():
            with self._lock:
                self.skipped_refreshes += 1
            return
        refresh()

    def _run_shard(self, job_id: str, shard_id: str, config_hash: str,
                   token: str, *, others: bool = False) -> ShardReceipt | None:
        self._refresh(others)
        jf = A.JobFiles(self.store, job_id, config_hash)  # paths keyed by the stage hash
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
        except KeyboardInterrupt:  # SIGINT (preemption): mark at once, let Modal retry
            self._end(active, preempted=True)  # in-flight episodes keep flushing (engine)
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
                self._active.pop(id(active), None)

    def preempt(self) -> bool:
        """``@modal.exit`` / SIGINT hook: stop queued episodes from starting, flush **every**
        shard in flight and mark each preempted at once (never waits for running episodes, so
        the mark lands within Modal's 30 s grace). Idempotent; a no-op when no shard is running.
        Returns True if it marked at least one."""
        with self._lock:
            actives = list(self._active.values())
        cancel = getattr(self.engine, "cancel", None)
        if actives and callable(cancel):
            cancel()
        marked = [self._end(a, preempted=True) for a in actives]
        return any(marked)

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

    def inputs_from_index(self, ref: EpisodeRef) -> SegcapInputs:
        raw = json.loads(self.store.get_bytes(ref.index_ref))
        if raw.get("schema") == "flora_index_receipt_v1" and raw.get("status") != "indexed":
            raise InputRejected(f"episode {ref.uid!r} was not indexed: {raw.get('error')}")
        idx = parse_index_receipt(json.dumps(raw))  # dataset / sources
        if (idx.dataset, idx.episode_id) != (ref.dataset, ref.episode_id):
            raise RuntimeError(f"index receipt is for {idx.dataset}/{idx.episode_id}, "
                               f"manifest says {ref.uid!r}")
        if not idx.sources:
            raise ValueError(f"episode {ref.uid!r}: index receipt has no camera streams")
        profile, yam32 = POOL_PROFILES.get(ref.pool, (ref.pool, False))
        profile = str(raw.get("profile") or profile)
        long_windows = bool(raw.get("yam32", yam32))
        task = raw.get("task") if isinstance(raw.get("task"), str) else None
        instruction = instruction_for(raw, task, self.settings.instruction_text)
        videos = {role: self.resolver.resolve(ref.uid, role, src)
                  for role, src in sorted(idx.sources.items())}
        return SegcapInputs(uid=ref.uid, profile=profile, long_windows=long_windows,
                            instruction=instruction, videos=videos, sources=dict(idx.sources),
                            duration_s=float(ref.duration_s or 0.0))

    def _provenance(self, inputs: SegcapInputs | None) -> Provenance:
        from flora_deploy import LOGIC_VERSION, __version__

        s = self.settings
        return Provenance(
            models=dict(s.models), engine=self.engine.label(), preset=s.preset,
            config_hash="", logic_version=LOGIC_VERSION,  # stage hash stamped by the sink
            framework=f"flora-deploy {__version__}", gpu=s.gpu, certificate=s.certificate,
            source=source_provenance(inputs.sources) if inputs else {})

    def _errors(self, shard, uid: str, exc: BaseException, phase: str, kind: str,
                **diagnostics: Any) -> list[EpisodeError]:
        type_name = getattr(exc, "type_name", None) or type(exc).__name__
        message = str(getattr(exc, "message", None) or exc)
        if kind == "infra":
            message = f"{type_name}: {message}"
        return [EpisodeError(uid=uid, stage=st, shard_id=shard.shard_id,
                             container_id=self.container_id, kind=kind,  # type: ignore[arg-type]
                             message=message,
                             diagnostics={"phase": phase, "type": type_name, **diagnostics})
                for st in stages_of(shard.kind)]

    def _load_inputs(self, shard, ref: EpisodeRef) -> SegcapInputs | list[EpisodeError]:
        """One episode's inputs, or its error files: NVIDIA semantics at once (terminal), infra
        after ``INPUT_RETRIES`` in-process retries (not terminal). A configuration error
        (:class:`InstructionMissing`) propagates and fails the shard."""
        attempt = 0
        while True:
            try:
                return self.inputs_for(ref)
            except InstructionMissing:
                raise
            except Exception as exc:  # noqa: BLE001 - becomes this episode's EpisodeError
                kind = classify_input_error(exc)
                if kind == "nv_failure":
                    return self._errors(shard, ref.uid, exc, "inputs", kind)
                if attempt >= INPUT_RETRIES:
                    return self._errors(shard, ref.uid, exc, "inputs", kind,
                                        attempts=attempt + 1)
                attempt += 1

    def _run(self, active: _Active, manifests: StoreManifestSource) -> ShardReceipt:
        from flora_deploy.accel import determinism as det
        from flora_deploy.stages.reference import stage_records

        t0 = time.perf_counter()
        jf, shard, s = active.jf, active.shard, self.settings
        if shard.kind != "segcap":
            raise ValueError(f"SegcapWorker cannot run shard kind {shard.kind!r}")
        if engine_mode(self.engine) != s.determinism:  # never label outputs with another mode
            raise det.DeterminismError(
                f"the engine runs determinism mode {engine_mode(self.engine)!r} but the job "
                f"was configured (and hashed) with {s.determinism!r} (FLORA_DETERMINISM)")

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
        missing_refs = [u for u in todo if u not in refs]
        if missing_refs:
            raise KeyError(f"episode {missing_refs[0]!r} is not in the manifest")
        todo = order_episodes(todo, refs, s.episode_order)  # G3: start order only
        inputs: dict[str, SegcapInputs] = {}
        errors: list[EpisodeError] = []
        for uid in todo:  # step 4: inputs (per-episode isolation)
            got = self._load_inputs(shard, refs[uid])
            if isinstance(got, SegcapInputs):
                inputs[uid] = got
            else:
                errors += got
        if errors:
            sink.add(errors=errors)
            sink.flush()

        def on_done(item: dict, result: dict | None, err: Exception | None) -> None:
            uid = item["episode"]
            timing_of = getattr(self.engine, "episode_timing", None)
            gpu_timing = timing_of() if callable(timing_of) else None
            if err is not None:  # nv_failure: terminal; infra (CUDA, I/O, time stop): raise
                kind, diag = classify_nvidia_error(err, gpu_timing)
                sink.add(errors=self._errors(shard, uid, err, "nvidia", kind, **diag))
                sink.flush()
                return
            assert result is not None
            # review 5 M3: the mode travels with every result, so mode checks on production
            # audit lines (stages.reference.check_same_mode) never default to "nvidia"
            fr = result.setdefault("flora_reference", {})
            fr.setdefault("determinism", {
                "schema": (getattr(self.engine, "determinism", None) or {}).get("schema"),
                "mode": engine_mode(self.engine)})
            audit_report = self.engine.check()  # nv.audit + determinism, before records
            ref, inp = refs[uid], inputs[uid]
            certified, reason = certification(ref, s.envelope)
            records = stage_records(result, uid=uid, dataset=ref.dataset,
                                    episode_id=ref.episode_id, provenance=self._provenance(inp),
                                    certified=certified, uncertified_reason=reason)
            # review 4 M4: calls near NVIDIA's max_time=300 are timing-, not input-determined
            timing = [i for i, c in enumerate(result.get("flora_reference", {})
                                              .get("model_calls", []))
                      if c.get("timing_dependent")]
            if timing:
                print(f"FLORA_WARN {json.dumps({'uid': uid, 'timing_dependent_calls': timing})}",
                      flush=True)
            audit = {"schema": AUDIT_SCHEMA, "uid": uid, "stage": "segcap",
                     "timing_dependent_calls": timing, "gpu_timing": gpu_timing,
                     "config_hash": jf.key(shard.kind), "job_config_hash": jf.config_hash,
                     "shard_id": shard.shard_id, "container_id": self.container_id,
                     "runtime": {**self.engine.runtime_provenance(), "nv_audit": audit_report,
                                 "scheduling": {"episode_order": s.episode_order,
                                                "max_shards": s.max_shards}},
                     "result": result}
            sink.add(records=records, audits=[audit])
            sink.flush()  # every finished episode is on the store before the next one lands

        if inputs:
            batch = self.engine.annotate([inputs[u].item() for u in todo if u in inputs],
                                         on_done=on_done)
            self.last_batch = batch.get("batch")
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
