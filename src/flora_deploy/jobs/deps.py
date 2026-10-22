"""Real adapters: the P1a-2 file modules (``manifest``, ``attempts``, ``progress``) behind the
orchestrator's Protocols (``ManifestSource``, ``AttemptLedger``, ``ProgressDeriver``), and
:func:`build_deps`, which a deployment factory (``orchestrator_app`` ``factory=``) returns.

Consistency rules the adapters keep (see ``jobs.store`` for what each backend guarantees):

* Bisection children are read from durable replan files with ``exists``/``get`` (never ``list``),
  so :meth:`StoreManifestSource.load` reflects :meth:`~StoreManifestSource.replan` immediately and
  a quarantined parent can never look satisfied while its children are invisible (C2).
* Records an adapter wrote itself are returned at once even before a lagging ``list`` shows them.
* Every record is filtered to the job's config hash.

Cost: the first ``load()`` of a process (and one every ``rescan_s``) checks the replan file of each
multi-episode shard (one ``exists`` each); in between only this process writes replans (the
orchestrator lease makes it the single writer), so its own writes are cached. Workers use
:meth:`StoreManifestSource.shard`, which reads at most one replan file.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.manifest import load_manifest
from flora_deploy.jobs.orchestrator import BudgetGate, OrchestratorDeps, utc

# The orchestrator's own container (``jobs.pools``: 0.125 cores, 256 MiB request,
# preemptible): charged by the default budget gate.
from flora_deploy.jobs.pools import (
    ORCHESTRATOR_CORES,
    ORCHESTRATOR_MEMORY_GIB,
    ORCHESTRATOR_PREEMPTIBLE,
)
from flora_deploy.jobs.settings import RunSettings
from flora_deploy.jobs.store import Store
from flora_deploy.jobs.types import ControlRecord, Manifest, Shard

Clock = Callable[[], float]


class StoreManifestSource:
    """``ManifestSource`` over ``manifest.json`` + per-shard replan files."""

    def __init__(self, jf: A.JobFiles, *, clock: Clock = time.time, rescan_s: float = 600.0):
        self.jf, self.clock, self.rescan_s = jf, clock, rescan_s
        self._base: Manifest | None = None
        self._replans: dict[str, A.Replan] = {}
        self._scanned_at = -float("inf")

    def base(self) -> Manifest:
        if self._base is None:
            m = load_manifest(self.jf.store, self.jf.job_id, self.jf.config_hash)
            if m.config_hash != self.jf.config_hash:
                raise ValueError(f"manifest is for config {m.config_hash!r}, "
                                 f"not {self.jf.config_hash!r}")
            self._base = m
        return self._base

    @property
    def replans(self) -> Mapping[str, A.Replan]:
        return dict(self._replans)

    def _scan(self) -> None:
        now = self.clock()
        if now - self._scanned_at < self.rescan_s:
            return
        for s in self.base().shards:
            if len(s.episodes) > 1 and s.shard_id not in self._replans:
                rp = A.read_replan(self.jf, s)
                if rp is not None:
                    self._replans[s.shard_id] = rp
        self._scanned_at = now

    def load(self) -> Manifest:
        m = self.base()
        self._scan()
        shards = {s.shard_id: s for s in m.shards}
        todo = list(shards)
        while todo:
            rp = self._replans.get(todo.pop())
            for c in rp.children if rp else ():
                if c.shard_id not in shards:
                    shards[c.shard_id] = c
                    todo.append(c.shard_id)
        return dataclasses.replace(m, shards=tuple(shards.values()))

    def replan(self, parent: Shard, children: Sequence[Shard]) -> None:
        if len(parent.episodes) <= 1:
            return  # single-episode shards are poison-pilled, not bisected
        sp = P.shard_parts(self.jf, parent)
        self._replans[parent.shard_id] = A.write_replan(
            self.jf, parent, children, created_utc=utc(self.clock()), done=parent.episodes,
            done_parts=sp.parts)

    def shard(self, shard_id: str) -> Shard:
        """Worker-side lookup (one manifest read, at most one replan read): a manifest shard,
        or a bisection child ``<kind>-ep-<uid>`` found through its parent's replan file."""
        m = self.base()
        for s in m.shards:
            if s.shard_id == shard_id:
                return s
        for s in m.shards:
            if len(s.episodes) > 1 and any(A.child_shard_id(s.kind, u) == shard_id
                                           for u in s.episodes):
                rp = self._replans.get(s.shard_id) or A.read_replan(self.jf, s)
                if rp is not None:
                    self._replans[s.shard_id] = rp
                    for c in rp.children:
                        if c.shard_id == shard_id:
                            return c
        raise KeyError(f"unknown shard {shard_id!r} (not in the manifest or any replan)")


class StoreAttemptLedger:
    """``AttemptLedger`` over attempt markers, quarantine / shard_failed records and errors."""

    def __init__(self, jf: A.JobFiles, *, limit: int, clock: Clock = time.time,
                 manifests: StoreManifestSource | None = None, app: str = ""):
        self.jf, self.limit, self.clock, self.manifests = jf, limit, clock, manifests
        self.app = app  # the deployed worker app (startup-failure markers); "" = none
        self._own_q: dict[str, ControlRecord] = {}
        self._own_f: dict[str, ControlRecord] = {}

    def starts(self, shard: Shard) -> int:
        return A.counted_starts(self.jf, shard.kind, shard.shard_id, limit=self.limit)

    def quarantined(self) -> Mapping[str, ControlRecord]:
        # Only real records: a replan whose record is missing (crash in between) is finished by
        # the orchestrator's normal D15 path, which reuses the persisted children.
        out = {k: v[-1] for k, v in P.quarantine_records(self.jf).items()}
        out.update({k: v for k, v in self._own_q.items() if k not in out})
        return out

    def quarantine(self, shard: Shard, reason: str, detail: Mapping[str, Any]) -> None:
        sp = P.shard_parts(self.jf, shard)
        q = A.quarantine(self.jf, shard, created_utc=utc(self.clock()), done=sp.done,
                         done_parts=sp.parts, starts=detail.get("starts"), reason=reason,
                         diagnostics=dict(detail))
        self._own_q[shard.shard_id] = q.record

    def fail(self, shard: Shard, reason: str, detail: Mapping[str, Any]) -> None:
        now = utc(self.clock())
        A.mark_failed(self.jf, shard, created_utc=now, reason=reason, detail=detail)
        self._own_f[shard.shard_id] = ControlRecord(
            "shard_failed", shard.shard_id, reason, now,
            {**dict(detail), "config_hash": self.jf.config_hash})

    def failed(self) -> Mapping[str, ControlRecord]:
        out = {k: v[-1] for k, v in P.failed_records(self.jf).items()}
        out.update({k: v for k, v in self._own_f.items() if k not in out})
        return out

    def bisect(self, shard: Shard, done: set[str]) -> tuple[Shard, ...]:
        return A.bisect(shard, done=done)

    def record_poison(self, shard: Shard, diagnostics: Mapping[str, Any]) -> None:
        # ``quarantine`` already wrote it for an unfinished single episode; never overwrite, and
        # never poison an episode whose records already exist (it is done, not poisoned)
        if self.jf.store.exists(A.poison_error_path(self.jf, shard)):
            return
        if P.shard_parts(self.jf, shard).done:
            return
        A.write_poison(self.jf, shard, message="poison pill (D15)", diagnostics=diagnostics)

    def call_for_token(self, shard: Shard, token: str) -> str | None:
        return A.call_for_token(self.jf, shard.kind, shard.shard_id, token)

    def call_started(self, shard: Shard, token: str, since_utc: str) -> bool:
        """A container started this spawn of ``shard`` (``attempts.call_started``)."""
        return A.call_started(self.jf, shard.kind, shard.shard_id, token=token,
                              since_utc=since_utc)

    def startup_failures(self, kind: str, since_utc: str) -> list[A.StartupFailure]:
        """Containers of the worker app serving ``kind`` whose startup failed at or after
        ``since_utc`` (``attempts.read_startup_failures``)."""
        if not self.app:
            return []
        return A.read_startup_failures(self.jf.store, self.app, kind=kind, since_utc=since_utc)

    def is_poisoned(self, shard: Shard) -> bool:
        """Poison file of a single-episode shard exists (e.g. isolated under another config that
        shares this kind's stage hash): the orchestrator then never spawns it again."""
        return A.is_poisoned(self.jf, shard)


class StoreProgress:
    """``ProgressDeriver``: episodes already in this shard's visible part files."""

    def __init__(self, jf: A.JobFiles):
        self.jf = jf

    def episodes_done(self, shard: Shard) -> set[str]:
        return set(P.shard_parts(self.jf, shard).done)


def default_budget_gate(store: Store, settings: RunSettings, manifests: StoreManifestSource,
                        attempts: StoreAttemptLedger, *, clock: Clock = time.time) -> BudgetGate:
    """The production gate (review-4 M2/M6): :class:`~flora_deploy.jobs.budget.
    ProjectionBudgetGate` with ``planner/rates.json`` estimates over the manifest's episode
    durations (no placeholder-rate unit guessing), pinned prices times the region multiplier of
    ``settings.region``, the deployed ``scaledown_window`` as idle tail and
    ``settings.min_containers`` warm containers counted as spend."""
    from flora_deploy.jobs.budget import (
        ProjectionBudgetGate,
        RateTableEstimator,
        pinned_rates,
        region_multiplier_for,
    )

    def durations() -> dict[str, float]:  # lazily: the manifest may be saved after build_deps
        return {e.uid: e.duration_s for e in manifests.base().episodes}

    def packing_rates() -> dict[str, float]:
        return dict(manifests.base().rates)

    return ProjectionBudgetGate(
        settings=settings, store=store, shards=lambda: manifests.load().shards,
        rates=pinned_rates(region_multiplier=region_multiplier_for(settings.region)),
        scaledown_s=float(settings.scaledown_s), clock=clock,
        orchestrator_cores=ORCHESTRATOR_CORES, orchestrator_memory_gib=ORCHESTRATOR_MEMORY_GIB,
        orchestrator_preemptible=ORCHESTRATOR_PREEMPTIBLE,
        quarantined=lambda: attempts.quarantined().keys(),
        estimator=RateTableEstimator.default(settings.preset, durations=durations,
                                             packing_rates=packing_rates),
        stage_hashes=lambda: manifests.base().stage_hashes)


def deployment_settings_kw(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """``config.schema.to_run_settings`` keyword arguments a deps factory running inside the
    deployed orchestrator should pass, from the env ``modal_app.factory`` sets there
    (``FLORA_MIN_CONTAINERS``, ``FLORA_SCALEDOWN_S``), so the budget gate charges the deployed
    warm pool and idle tail."""
    import os

    from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S

    env = os.environ if env is None else env
    return {"min_containers": int(env.get("FLORA_MIN_CONTAINERS", "0") or 0),
            "scaledown_s": float(env.get("FLORA_SCALEDOWN_S", WORKER_SCALEDOWN_S))}


def build_deps(store: Store, settings: RunSettings, *, budget: BudgetGate | None = None,
               worker_app: str = "", worker_classes: Mapping[str, str] | None = None,
               clock: Clock = time.time, rescan_s: float = 600.0) -> OrchestratorDeps:
    """Everything the orchestrator / reconciler needs for one job + config, over ``store``.

    ``budget`` defaults to :func:`default_budget_gate` (the real projection gate). Pass
    ``AllowAll()`` explicitly to run without budget enforcement (tests only)."""
    jf = A.JobFiles(store, settings.job_id, settings.config_hash)
    manifests = StoreManifestSource(jf, clock=clock, rescan_s=rescan_s)
    attempts = StoreAttemptLedger(jf, limit=settings.max_starts_before_quarantine, clock=clock,
                                  manifests=manifests, app=worker_app)
    if budget is None:
        budget = default_budget_gate(store, settings, manifests, attempts, clock=clock)
    return OrchestratorDeps(
        settings=settings, store=store, manifests=manifests, attempts=attempts,
        progress=StoreProgress(jf), budget=budget, worker_app=worker_app,
        worker_classes=dict(worker_classes or {}))
