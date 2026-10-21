"""SegcapWorker core (P1a-5s): ``workers.ShardWorker`` around the fast ``exact`` pipeline.

:class:`SegcapShardRunner` processes one ``segcap`` shard per :meth:`~SegcapShardRunner.run_shard`
call, following the worker contract in ``workers.py`` (the same structure as
``workers_impl.reward.RewardShardRunner``, whose part sink, index-receipt reader and video
resolver it reuses):

Before step 0 a Volume store is refreshed (``store.refresh()``).

0. ``attempts.superseded`` → return ``None`` having written nothing;
1. ``attempts.write_attempt`` with the spawn ``token`` and the Modal call id;
2. episodes already in earlier containers' parts (``progress.shard_parts``: both ``segment`` and
   ``caption`` records present) are skipped;
3. episodes whose upstream shard was poison-pilled are reported (``write_upstream_failures``)
   and never run (``segcap`` shards have no ``depends_on`` today; the step is generic);
4. the remaining episodes' inputs come from their index receipts (``EpisodeRef.index_ref``):
   camera roles → local files (an episode range inside a concatenated file is stream-copy cut,
   ``reward.VideoResolver``), NVIDIA's profile and ``yam32`` flag, and NVIDIA's ``instruction``
   (the receipt's instruction source). They go through one :class:`FastExactEngine` batch:
   NVIDIA's own ``api.DataEnginePipeline().annotate`` per episode on the resident models
   (``stages.reference.annotate_batch``). As **each** episode finishes (in its own thread) its
   ``segment`` and ``caption`` ``StageRecord``s and one audit line (NVIDIA's full result with our
   measurements and the runtime provenance) are appended and flushed whole to the store. An input
   that cannot be resolved and an episode NVIDIA rejects become ``EpisodeError`` files
   (``nv_failure`` for NVIDIA's ``InputError`` / ``ValueError`` semantics, ``infra`` otherwise),
   one per stage, never a crash;
5. ``KeyboardInterrupt`` (SIGINT) or :meth:`~SegcapShardRunner.preempt` (``@modal.exit``) flushes
   and calls ``attempts.mark_preempted``; any other exception flushes, ``mark_exception``,
   re-raises;
6. ``ShardReceipt`` LAST, ``parts`` = earlier parts + this container's own.

Records: ``certified`` per the episode's envelope verdict (``EpisodeRef.envelope_ok``, D13;
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
    ShardPartSink,
    VideoResolver,
    _Active,
    _container_ks,
    default_container_id,
    parse_index_receipt,
)

ENGINE_LABEL = "nvidia-hf"
AUDIT_SCHEMA = "flora_segcap_audit_v1"
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


def instruction_for(receipt: Mapping[str, Any], task: str | None,
                    fixed_text: str | None) -> str:
    """NVIDIA's ``instruction`` for an episode (``jobs.indexing.IndexJob.instruction_for``)."""
    source = receipt.get("instruction", "dataset_task")
    if source == "none":
        return ""
    if source == "fixed":
        if fixed_text is None:
            raise RuntimeError("index receipt asks for a fixed instruction but the worker has "
                               "no FLORA_INSTRUCTION_TEXT")
        return fixed_text
    return task or ""


def source_provenance(sources: Mapping[str, SourceRef]) -> dict[str, dict[str, Any]]:
    return {role: {"uri": s.uri, "range_s": None if s.start_s is None and s.end_s is None
                   else [s.start_s, s.end_s]} for role, s in sorted(sources.items())}


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
    ``nvidia`` mode it is reported in the audit line)."""

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
    _audits: int = 0
    _drift: str | None = None

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

    def label(self) -> str:
        mode = (self.determinism or {}).get("mode", "nvidia")
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
        }

    def annotate(self, items: list[dict], *, on_done: OnDone) -> dict:
        from flora_deploy.stages.reference import annotate_batch

        self._audit()
        return annotate_batch(
            self.nv, self.session, items, concurrency=min(self.concurrency, max(len(items), 1)),
            gpu_lock=self.gpu_lock, clones=self.clones, accel=self.accel,
            sampler=self.sampler_factory() if self.sampler_factory else None,
            progress=self.progress, on_start=lambda ep: self._audit(), on_done=on_done)


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
    local_dir: str = "/tmp/flora-segcap"
    container_id: str = field(default_factory=default_container_id)


def certification(ref: EpisodeRef, envelope: str = "warn") -> tuple[bool, str | None]:
    """certified only inside NVIDIA's reference envelope (``stages.reward.certification``)."""
    if envelope == "ignore" or ref.envelope_ok is True:
        return True, None
    if ref.envelope_ok is False:
        return False, "outside_reference_envelope"
    return False, "oracle_ineligible_source"


def _error_kind(exc: BaseException) -> str:
    from flora_deploy.stages.reference import NvidiaInputError

    return "nv_failure" if isinstance(exc, NvidiaInputError | ValueError) else "infra"


class SegcapShardRunner:
    """``ShardWorker`` core for ``segcap`` shards (module docstring)."""

    def __init__(self, store: Store, engine: SegcapEngine, settings: SegcapWorkerSettings, *,
                 call_id: Callable[[], str | None] = lambda: "",
                 clock: Callable[[], float] = time.time, resolver: VideoResolver | None = None,
                 inputs: Callable[[EpisodeRef], SegcapInputs] | None = None):
        """``inputs``: EpisodeRef -> :class:`SegcapInputs` (default: from the index receipt,
        :meth:`inputs_from_index`)."""
        self.store, self.engine, self.settings = store, engine, settings
        self.inputs_for = inputs or self.inputs_from_index
        self.call_id, self.clock = call_id, clock
        self.container_id = safe_key(settings.container_id)
        self.resolver = resolver or VideoResolver(
            settings.media_root, Path(settings.local_dir) / "cuts")
        self._lock = threading.RLock()
        self._active: _Active | None = None
        self._next_k: dict[tuple[str, str, str], int] = {}
        self.last_batch: dict | None = None  # the last annotate batch summary (diagnostics)

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
        idx = parse_index_receipt(json.dumps(raw))  # status / dataset / sources / task
        if (idx.dataset, idx.episode_id) != (ref.dataset, ref.episode_id):
            raise RuntimeError(f"index receipt is for {idx.dataset}/{idx.episode_id}, "
                               f"manifest says {ref.uid!r}")
        if not idx.sources:
            raise ValueError(f"episode {ref.uid!r}: index receipt has no camera streams")
        profile, yam32 = POOL_PROFILES.get(ref.pool, (ref.pool, False))
        profile = str(raw.get("profile") or profile)
        long_windows = bool(raw.get("yam32", yam32))
        task = raw.get("task") if isinstance(raw.get("task"), str) else idx.task
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
            config_hash="", logic_version=LOGIC_VERSION,  # job hash stamped by the sink
            framework=f"flora-deploy {__version__}", gpu=s.gpu, certificate=s.certificate,
            source=source_provenance(inputs.sources) if inputs else {})

    def _errors(self, shard, uid: str, exc: BaseException, phase: str) -> list[EpisodeError]:
        kind = _error_kind(exc)
        type_name = getattr(exc, "type_name", None) or type(exc).__name__
        message = getattr(exc, "message", None) or str(exc)
        return [EpisodeError(uid=uid, stage=st, shard_id=shard.shard_id,
                             container_id=self.container_id, kind=kind,  # type: ignore[arg-type]
                             message=str(message),
                             diagnostics={"phase": phase, "type": type_name})
                for st in stages_of(shard.kind)]

    def _run(self, active: _Active, manifests: StoreManifestSource) -> ShardReceipt:
        from flora_deploy.stages.reference import stage_records

        t0 = time.perf_counter()
        jf, shard, s = active.jf, active.shard, self.settings
        if shard.kind != "segcap":
            raise ValueError(f"SegcapWorker cannot run shard kind {shard.kind!r}")

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
        inputs: dict[str, SegcapInputs] = {}
        errors: list[EpisodeError] = []
        for uid in todo:  # step 4: inputs (per-episode isolation)
            try:
                if uid not in refs:
                    raise KeyError(f"episode {uid!r} is not in the manifest")
                inputs[uid] = self.inputs_for(refs[uid])
            except Exception as exc:  # noqa: BLE001 - becomes this episode's EpisodeError
                errors += self._errors(shard, uid, exc, "inputs")
        if errors:
            sink.add(errors=errors)
            sink.flush()

        def on_done(item: dict, result: dict | None, err: Exception | None) -> None:
            uid = item["episode"]
            if err is not None:
                sink.add(errors=self._errors(shard, uid, err, "nvidia"))
                sink.flush()
                return
            assert result is not None
            audit_report = self.engine.check()  # nv.audit + determinism, before records
            ref, inp = refs[uid], inputs[uid]
            certified, reason = certification(ref, s.envelope)
            records = stage_records(result, uid=uid, dataset=ref.dataset,
                                    episode_id=ref.episode_id, provenance=self._provenance(inp),
                                    certified=certified, uncertified_reason=reason)
            audit = {"schema": AUDIT_SCHEMA, "uid": uid, "stage": "segcap",
                     "shard_id": shard.shard_id, "container_id": self.container_id,
                     "runtime": {**self.engine.runtime_provenance(), "nv_audit": audit_report},
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
