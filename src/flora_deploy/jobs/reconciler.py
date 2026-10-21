"""Stateless reconciler (PLAN.md §8): rebuild a job's status from files alone.

Inputs are the same files the orchestrator uses (manifest, per-shard ``call.json`` ledgers,
receipts, attempt markers, quarantine/control records, ``errors/``). Nothing is cached between
calls, so ``flora status`` can run it any time, from anywhere, while an orchestrator runs or after
one died. :func:`revive_if_needed` restarts a dead orchestrator (lease expired, work remaining).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from flora_deploy.jobs.orchestrator import (
    RESPAWNS_EXHAUSTED,
    AttemptLedger,
    Lease,
    LeaseInfo,
    ManifestSource,
    ShardLedger,
    call_path,
    control_view,
    deps_ready,
    read_controls,
    resolve,
)
from flora_deploy.jobs.settings import MAX_EPISODE_ERROR_RATE
from flora_deploy.jobs.spawner import Spawner
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import MAX_STARTS_BEFORE_QUARANTINE, JobState

STUCK_AFTER_S = 6 * 3600.0

# Per-shard derived status, as counted in ``JobStatus.counts``.
SHARD_STATUSES = ("done", "running", "pending", "waiting", "quarantined", "poisoned", "failed",
                  "cancelled")


@dataclass(frozen=True)
class StuckCall:
    shard_id: str
    kind: str
    call_id: str | None
    reason: str
    starts: int


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    config_hash: str
    state: JobState
    counts: dict[str, dict[str, int]]  # kind -> shard status -> n
    active_calls: int
    stuck_calls: tuple[StuckCall, ...]
    quarantined: tuple[str, ...]  # bisected parents
    poisoned: tuple[str, ...]  # single-episode shards isolated to errors/
    failed: tuple[str, ...]
    errors: tuple[str, ...]  # store paths under errors/
    orchestrator: LeaseInfo | None
    notes: tuple[str, ...] = field(default=())

    @property
    def orchestrator_alive(self) -> bool:
        return self.orchestrator is not None


def reconcile(*, job_id: str, config_hash: str, store: Store, manifests: ManifestSource,
              attempts: AttemptLedger, spawner: Spawner | None = None, lease: Lease | None = None,
              clock: Callable[[], float] = time.time, stuck_after_s: float = STUCK_AFTER_S,
              max_starts: int = MAX_STARTS_BEFORE_QUARANTINE,
              compose_done: Callable[[], bool] | None = None,
              max_error_rate: float = MAX_EPISODE_ERROR_RATE) -> JobStatus:
    """Derive status. ``spawner`` (optional) lets it poll live calls to spot ones whose outcome
    the orchestrator never processed; ``lease`` tells whether an orchestrator is alive.

    A resolved job is ``done`` only under the orchestrator's error-rate guard
    (``progress.error_rate_problems`` over receipts' ``episodes_failed`` and poison-pilled
    episodes); otherwise ``completed_with_errors`` (``notes`` say why)."""
    manifest = manifests.load()
    paths = JobPaths(job_id, config_hash=config_hash,
                     stage_hashes=dict(getattr(manifest, "stage_hashes", {}) or {}))
    shards = list(manifest.shards)
    quarantine_recs = attempts.quarantined()
    failed_fn = getattr(attempts, "failed", None)
    failed_recs = failed_fn() if failed_fn is not None else {}
    controls = control_view(read_controls(store, paths), job_id, config_hash)
    notes: list[str] = []

    status: dict[str, str] = {}
    ledgers: dict[str, ShardLedger] = {}
    for s in shards:
        p = call_path(paths, config_hash, s.kind, s.shard_id)
        led = ShardLedger.from_bytes(store.get_bytes(p)) if store.exists(p) else None
        if led is not None:
            ledgers[s.shard_id] = led
        if store.exists(paths.receipt(config_hash, s.kind, s.shard_id)):
            status[s.shard_id] = "done"
            if led is not None and led.status in ("spawning", "active"):
                notes.append(f"{s.shard_id}: receipt exists but ledger says {led.status}")
        elif led is not None and led.status in ("quarantined", "poisoned", "failed", "cancelled",
                                                "done"):
            status[s.shard_id] = led.status
        elif s.shard_id in failed_recs or (s.shard_id in quarantine_recs and
                                           quarantine_recs[s.shard_id].reason ==
                                           RESPAWNS_EXHAUSTED):
            status[s.shard_id] = "failed"  # terminal; never bisected (H1)
        elif s.shard_id in quarantine_recs:
            # quarantine written, ledger not yet: the orchestrator will finish it on its next tick
            status[s.shard_id] = "quarantined"
        elif led is not None and led.status in ("spawning", "active", "retry"):
            status[s.shard_id] = "running" if led.status != "retry" else "pending"
        else:
            status[s.shard_id] = "new"

    # a quarantined parent resolves only through the children its ledger lists (C2); the
    # quarantine record's children are a fallback when the ledger write was lost
    children = {sid: led.children for sid, led in ledgers.items() if led.status == "quarantined"}
    for sid, rec in quarantine_recs.items():
        if sid not in children and status.get(sid) == "quarantined":
            children[sid] = tuple(c["shard_id"] for c in rec.detail.get("children", ()))
    satisfied, dead = resolve(shards, lambda sid: status.get(sid, "new"), children)
    by_id = {s.shard_id: s for s in shards}
    for s in shards:
        if status[s.shard_id] == "new":
            deps_ok = deps_ready(s, by_id, satisfied, children)
            status[s.shard_id] = "pending" if deps_ok else "waiting"

    now = clock()
    stuck: list[StuckCall] = []
    for s in shards:
        led = ledgers.get(s.shard_id)
        if status[s.shard_id] != "running" or led is None:
            continue
        starts = attempts.starts(s)
        reason = None
        if starts >= max_starts:
            reason = f"{starts} starts without a receipt (poison pill not yet handled)"
        elif led.status == "spawning":
            reason = "spawn intent without a recorded call id"
        elif spawner is not None and led.call_id:
            st = spawner.poll(led.call_id)
            if st.state != "pending":
                reason = f"call is {st.state} but the orchestrator has not processed it"
        if reason is None and led.updated_utc:
            age = now - _parse_utc(led.updated_utc)
            if age > stuck_after_s:
                reason = f"pending for {age / 3600:.1f} h"
        if reason:
            stuck.append(StuckCall(s.shard_id, s.kind, led.call_id, reason, starts))

    counts: dict[str, dict[str, int]] = {}
    for s in shards:
        c = counts.setdefault(s.kind, dict.fromkeys(SHARD_STATUSES, 0))
        c[status[s.shard_id]] += 1
    active = sum(1 for v in status.values() if v == "running")
    errors = tuple(p for kind in sorted({s.kind for s in shards})
                   for p in store.list(paths.area_prefix(config_hash, "errors", kind)))

    state: JobState
    resolved = bool(shards) and active == 0 and \
        all(s.shard_id in satisfied or s.shard_id in dead for s in shards)
    # a cancel that arrives after every shard was satisfied changes nothing (review 6 M1): the
    # job's outcome stands; a cancel that stopped work (cancelled = dead shards) wins
    if controls.cancelled is not None and not (resolved and not dead):
        state = "cancelled"
    elif resolved:
        problems = [] if dead else _error_rate_problems(store, paths, config_hash, shards,
                                                         status, max_error_rate,
                                                         getattr(manifest, "excluded", None))
        notes += problems
        if dead:
            state = "failed"
        elif problems:
            state = "completed_with_errors"
        elif compose_done is None or compose_done():
            state = "done"
        else:
            state = "composing"
    elif controls.budget_stopped is not None:
        state = "budget_stopped"  # draining while active_calls > 0
    elif not ledgers and not any(v == "done" for v in status.values()):
        state = "planned"
    else:
        state = "running"

    return JobStatus(
        job_id=job_id, config_hash=config_hash, state=state, counts=counts, active_calls=active,
        stuck_calls=tuple(stuck),
        quarantined=tuple(sorted(k for k, v in status.items() if v == "quarantined")),
        poisoned=tuple(sorted(k for k, v in status.items() if v == "poisoned")),
        failed=tuple(sorted(k for k, v in status.items() if v == "failed")),
        errors=errors, orchestrator=lease.current() if lease is not None else None,
        notes=tuple(notes),
    )


def _error_rate_problems(store: Store, paths: JobPaths, config_hash: str, shards, status,
                         max_error_rate: float, unplanned=None) -> list[str]:
    from flora_deploy.jobs.manifest import from_json
    from flora_deploy.jobs.progress import (
        error_rate_problems,
        tally_exclusions,
        with_unplanned,
    )
    from flora_deploy.jobs.types import ShardReceipt

    planned: dict[str, set[str]] = {}
    excluded: dict[str, set[str]] = {}
    for s in shards:
        if not s.bisected_from:
            planned.setdefault(s.kind, set()).update(s.episodes)
        st = status.get(s.shard_id)
        if st == "poisoned":
            excluded.setdefault(s.kind, set()).update(s.episodes)
        elif st == "done":
            p = paths.receipt(config_hash, s.kind, s.shard_id)
            if store.exists(p):
                r = from_json(ShardReceipt, store.get_bytes(p))
                excluded.setdefault(s.kind, set()).update(r.episodes_failed)
    tallies = tally_exclusions(planned, excluded)
    return error_rate_problems(with_unplanned(tallies, unplanned), max_error_rate)


def needs_orchestrator(status: JobStatus) -> bool:
    """Work remains and nobody holds the lease."""
    return status.state in ("planned", "running") and not status.orchestrator_alive


def revive_if_needed(status: JobStatus, launch: Callable[[], object]) -> bool:
    """Call ``launch()`` (e.g. ``orchestrator_app.run.spawn``) if the orchestrator is dead.

    Safe to race: a duplicate launch loses the lease and exits without spawning anything.
    """
    if needs_orchestrator(status):
        launch()
        return True
    return False


def _parse_utc(ts: str) -> float:
    from datetime import UTC, datetime

    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC).timestamp()
