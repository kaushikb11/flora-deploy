"""Shard orchestrator (PLAN.md §8, D6, D15): a deterministic, restartable scheduling loop.

Everything the loop decides is driven by store files plus a :class:`~flora_deploy.jobs.spawner.
Spawner`; nothing lives only in memory. A restarted orchestrator rebuilds its view from:

- the manifest (``ManifestSource``), including shards re-planned by bisection;
- one ledger file per shard, ``.../calls/<kind>/<ab>/<cd>/<shard>/call.json`` (:func:`call_path`),
  which records the spawned call id so a successor *re-attaches* via ``poll`` instead of
  re-spawning;
- receipts (``JobPaths.receipt``), the only proof a shard is done;
- attempt markers (``AttemptLedger.starts``) for poison-pill detection (Modal retries crashes and
  OOMs forever inside one call, so the call itself never reports them);
- ``ControlRecord`` files for cancel / budget_stop / resume.

One instance per job is enforced by a :class:`Lease` (Modal: :class:`DictLease` over a
``modal.Dict`` using ``put(skip_if_exists=True)``). The loop stops before ``deadline_s`` and returns
``"continue"`` so the Modal wrapper can spawn its successor.

The planning pieces (manifest, attempts/quarantine/bisection, progress) belong to P1a-2 and the
budget gate to P1a-4; they are consumed here only through the small ``Protocol``s below.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from flora_deploy.jobs.settings import RunSettings
from flora_deploy.jobs.spawner import Spawner
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import ControlRecord, Manifest, Shard

DEADLINE_S = 23 * 3600  # Modal caps one function attempt at 24 h; leave an hour of margin
LEASE_TTL_S = 300.0
POLL_INTERVAL_S = 15.0
MAX_RESPAWNS = 3  # respawns after failed / expired / externally cancelled calls, per shard

Event = dict[str, Any]
EventSink = Callable[[Event], None]

TickResult = Literal["running", "done", "failed", "cancelled", "budget_stopped", "continue",
                     "lost_lease"]
RunResult = Literal["done", "failed", "cancelled", "budget_stopped", "continue", "lost_lease",
                    "lease_held", "max_ticks"]

# Ledger statuses (``call.json``). ``retry`` = respawn wanted; ``spawning`` = intent written but the
# call id was never recorded (crash between spawn and write).
LedgerStatus = Literal["new", "spawning", "active", "retry", "done", "quarantined", "poisoned",
                       "failed", "cancelled"]
_LIVE = ("spawning", "active")


# ----------------------------------------------------------------------------------------------
# Protocols other workstreams implement (P1a-2: manifest/attempts/progress; P1a-4: budget)
# ----------------------------------------------------------------------------------------------


class ManifestSource(Protocol):
    def load(self) -> Manifest:
        """The effective manifest: planned shards plus every shard added by :meth:`replan`."""
        ...

    def replan(self, parent: Shard, children: Sequence[Shard]) -> None:
        """Persist ``children`` (``bisected_from == parent.shard_id``). Idempotent by shard id."""
        ...


class AttemptLedger(Protocol):
    def starts(self, shard: Shard) -> int:
        """Number of distinct ``AttemptMarker`` files for ``shard`` (container starts)."""
        ...

    def quarantined(self) -> Mapping[str, ControlRecord]:
        """shard_id -> its ``quarantine`` ControlRecord."""
        ...

    def quarantine(self, shard: Shard, reason: str, detail: Mapping[str, Any]) -> None:
        """Write a ``ControlRecord(kind="quarantine", target=shard_id)``. Idempotent."""
        ...

    def bisect(self, shard: Shard, done: set[str]) -> tuple[Shard, ...]:
        """Pure planning: child shards covering ``shard.episodes - done`` (D15: single-episode
        shards ``<kind>-ep-<uid>``). Deterministic, so a crash-and-redo yields the same ids."""
        ...

    def record_poison(self, shard: Shard, diagnostics: Mapping[str, Any]) -> None:
        """Write ``EpisodeError(kind="poison_pill")`` for a single-episode shard to ``errors/``."""
        ...


class ProgressDeriver(Protocol):
    def episodes_done(self, shard: Shard) -> set[str]:
        """uids that already have records in this shard's part files (any attempt)."""
        ...


@dataclass(frozen=True)
class BudgetDecision:
    allow: bool
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)


class BudgetGate(Protocol):
    def admit(self, shard: Shard, in_flight: Mapping[str, int]) -> BudgetDecision:
        """Called before every spawn. ``in_flight`` is active calls per shard kind.
        A denial makes the orchestrator write ``budget_stop`` and drain."""
        ...


class AllowAll:
    """Budget gate that never stops (tests, or jobs without a budget)."""

    def admit(self, shard: Shard, in_flight: Mapping[str, int]) -> BudgetDecision:
        return BudgetDecision(True)


# ----------------------------------------------------------------------------------------------
# Lease
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaseInfo:
    holder: str
    expires_at: float
    generation: int


class Lease(Protocol):
    def acquire(self, holder: str, ttl_s: float) -> bool: ...
    def renew(self, holder: str, ttl_s: float) -> bool: ...
    def release(self, holder: str) -> None: ...
    def current(self) -> LeaseInfo | None:
        """The live (unexpired) lease, or ``None`` if nobody holds it."""
        ...


class KV(Protocol):
    """The subset of ``modal.Dict`` the lease needs."""

    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool: ...
    def get(self, key: Any, default: Any | None = None) -> Any: ...
    def contains(self, key: Any) -> bool: ...


class DictLease:
    """Single-holder lease over an atomic create-if-absent KV (``modal.Dict``).

    ``modal.Dict`` has no compare-and-swap, so takeover of an expired lease cannot overwrite the
    old entry safely. Instead each lease is a *generation* key ``<name>/gen/<g>``: taking over an
    expired generation ``g`` means creating ``g+1`` with ``put(skip_if_exists=True)``, which exactly
    one contender wins. The holder renews by rewriting its own generation key, after checking that
    ``g+1`` doesn't exist; a holder that finds ``g+1`` has lost the lease and must stop.
    """

    def __init__(self, kv: KV, name: str, clock: Callable[[], float] = time.time):
        self.kv, self.name, self.clock = kv, name, clock
        self._gen: int | None = None

    def _key(self, g: int) -> str:
        return f"{self.name}/gen/{g}"

    def _head(self) -> tuple[int, dict | None]:
        g = int(self.kv.get(f"{self.name}/hint", 0) or 0)
        if g and not self.kv.contains(self._key(g)):
            g = 0
        while self.kv.contains(self._key(g + 1)):
            g += 1
        return g, (self.kv.get(self._key(g)) if g else None)

    def _expired(self, v: dict | None) -> bool:
        return v is None or float(v["expires_at"]) <= self.clock()

    def acquire(self, holder: str, ttl_s: float) -> bool:
        g, v = self._head()
        if v is not None and v["holder"] == holder and not self._expired(v):
            self._gen = g
            return self.renew(holder, ttl_s)
        if not self._expired(v):
            return False
        new = g + 1
        val = {"holder": holder, "expires_at": self.clock() + ttl_s}
        if not self.kv.put(self._key(new), val, skip_if_exists=True):
            return False
        self.kv.put(f"{self.name}/hint", new)
        self._gen = new
        return True

    def renew(self, holder: str, ttl_s: float) -> bool:
        g = self._gen
        if g is None or self.kv.contains(self._key(g + 1)):
            self._gen = None
            return False
        v = self.kv.get(self._key(g))
        if v is None or v["holder"] != holder:
            self._gen = None
            return False
        self.kv.put(self._key(g), {"holder": holder, "expires_at": self.clock() + ttl_s})
        return True

    def release(self, holder: str) -> None:
        g = self._gen
        if g is not None and not self.kv.contains(self._key(g + 1)):
            v = self.kv.get(self._key(g))
            if v is not None and v["holder"] == holder:
                self.kv.put(self._key(g), {"holder": holder, "expires_at": 0.0})
        self._gen = None

    def current(self) -> LeaseInfo | None:
        g, v = self._head()
        if self._expired(v):
            return None
        assert v is not None
        return LeaseInfo(v["holder"], float(v["expires_at"]), g)


# ----------------------------------------------------------------------------------------------
# Files: per-shard ledger and control records
# ----------------------------------------------------------------------------------------------


def call_path(paths: JobPaths, config_hash: str, kind: str, shard_id: str) -> str:
    """Per-shard orchestrator ledger. Lives under ``<cfg_hash>/`` like attempts and receipts.

    (Candidate for promotion to ``JobPaths.call`` when the contracts are next revised.)
    """
    return f"{paths._cfg(config_hash, 'calls', kind, shard_id)}/call.json"  # noqa: SLF001


@dataclass
class ShardLedger:
    shard_id: str
    kind: str
    status: LedgerStatus = "new"
    call_id: str | None = None
    spawn_n: int = 0
    respawns: int = 0
    reason: str | None = None
    children: tuple[str, ...] = ()
    calls: list[dict[str, Any]] = field(default_factory=list)
    updated_utc: str = ""

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> ShardLedger:
        d = json.loads(data)
        d["children"] = tuple(d.get("children", ()))
        return cls(**d)


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compact(ts: str) -> str:
    return "".join(c for c in ts if c.isalnum() or c == ".")


def write_control(store: Store, paths: JobPaths, record: ControlRecord) -> str:
    """Write one ControlRecord as its own whole file; returns its path (used by CLI too).

    ``created_utc`` must use :func:`utc`'s format: cancel/budget_stop/resume are ordered by it."""
    path = paths.control(record.kind, record.target, _compact(record.created_utc))
    data = json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode()
    store.put_bytes(path, data)
    return path


def read_controls(store: Store, paths: JobPaths,
                  kinds: Iterable[str] = ("cancel", "budget_stop", "resume"),
                  ) -> list[ControlRecord]:
    out: list[ControlRecord] = []
    for kind in kinds:
        for p in store.list(f"{paths.root}/control/{kind}/"):
            try:
                out.append(ControlRecord(**json.loads(store.get_bytes(p))))
            except (KeyError, TypeError, ValueError):
                continue  # half-visible / malformed file; never fatal
    return out


@dataclass(frozen=True)
class ControlView:
    cancelled: ControlRecord | None
    budget_stopped: ControlRecord | None  # latest budget_stop not superseded by a later resume


def control_view(records: Iterable[ControlRecord], job_id: str) -> ControlView:
    mine = sorted((r for r in records if r.target == job_id), key=lambda r: r.created_utc)
    cancel = next((r for r in mine if r.kind == "cancel"), None)
    stop: ControlRecord | None = None
    for r in mine:
        if r.kind == "budget_stop":
            stop = r
        elif r.kind == "resume":
            stop = None
    return ControlView(cancel, stop)


# ----------------------------------------------------------------------------------------------
# Resolution (shared with the reconciler)
# ----------------------------------------------------------------------------------------------


def resolve(shards: Sequence[Shard], status: Callable[[str], str]) -> tuple[set[str], set[str]]:
    """Return (satisfied, dead) shard ids.

    satisfied: receipt (``done``), or the shard is an isolated poison episode (``poisoned``), or it
    was quarantined and bisected and all of its children are satisfied.
    dead: ``failed``/``cancelled``, a dependency is dead, or a bisection child is dead.
    """
    by_id = {s.shard_id: s for s in shards}
    children: dict[str, list[str]] = {}
    for s in shards:
        if s.bisected_from:
            children.setdefault(s.bisected_from, []).append(s.shard_id)
    sat: dict[str, bool] = {}
    dead: dict[str, bool] = {}

    def is_sat(sid: str, depth: int = 0) -> bool:
        if sid in sat:
            return sat[sid]
        if depth > len(by_id) + 1:
            raise ValueError(f"cycle in shard graph at {sid!r}")
        st = status(sid)
        if st in ("done", "poisoned"):
            r = True
        elif st == "quarantined":
            r = all(is_sat(c, depth + 1) for c in children.get(sid, ()))
        else:
            r = False
        sat[sid] = r
        return r

    def is_dead(sid: str, depth: int = 0) -> bool:
        if sid in dead:
            return dead[sid]
        if depth > len(by_id) + 1:
            raise ValueError(f"cycle in shard graph at {sid!r}")
        s = by_id.get(sid)
        st = status(sid)
        if s is None:
            r = False  # unknown dependency: wait (it may appear after a replan)
        elif st in ("failed", "cancelled"):
            r = True
        elif is_sat(sid):
            r = False
        elif st == "quarantined":
            r = any(is_dead(c, depth + 1) for c in children.get(sid, ()))
        else:
            r = any(is_dead(d, depth + 1) for d in s.depends_on)
        dead[sid] = r
        return r

    return ({s.shard_id for s in shards if is_sat(s.shard_id)},
            {s.shard_id for s in shards if is_dead(s.shard_id)})


# ----------------------------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    result: RunResult
    ticks: int
    spawned: int


@dataclass
class OrchestratorDeps:
    """What a deployment wires together (the Modal wrapper builds this from a factory)."""

    settings: RunSettings
    store: Store
    manifests: ManifestSource
    attempts: AttemptLedger
    progress: ProgressDeriver
    budget: BudgetGate = field(default_factory=AllowAll)
    worker_app: str = ""
    worker_classes: Mapping[str, str] = field(default_factory=dict)  # shard kind -> Modal Cls


class Orchestrator:
    def __init__(self, *, settings: RunSettings, store: Store, spawner: Spawner, lease: Lease,
                 manifests: ManifestSource, attempts: AttemptLedger, progress: ProgressDeriver,
                 budget: BudgetGate | None = None, events: EventSink | None = None,
                 clock: Callable[[], float] = time.time, deadline_s: float = DEADLINE_S,
                 lease_ttl_s: float = LEASE_TTL_S, max_respawns: int = MAX_RESPAWNS,
                 holder: str | None = None):
        self.settings = settings
        self.job_id = settings.job_id
        self.cfg = settings.config_hash
        self.paths = JobPaths(settings.job_id)
        self.store, self.spawner, self.lease = store, spawner, lease
        self.manifests, self.attempts, self.progress = manifests, attempts, progress
        self.budget = budget or AllowAll()
        self.events = events or (lambda e: None)
        self.clock = clock
        self.deadline_s, self.lease_ttl_s, self.max_respawns = deadline_s, lease_ttl_s, max_respawns
        self.holder = holder or f"orch-{uuid.uuid4().hex[:12]}"
        self.k = settings.max_starts_before_quarantine
        self.started_at = clock()
        self.spawned = 0
        self._held = False
        self._ledgers: dict[str, ShardLedger] = {}
        self._own_controls: list[ControlRecord] = []  # visible before a lagging list() shows them

    # events / files -------------------------------------------------------------------------
    def _emit(self, event: str, **kw: Any) -> None:
        self.events({"event": event, "job_id": self.job_id, "t": self.clock(),
                     "holder": self.holder, **kw})

    def _ledger(self, shard: Shard) -> ShardLedger:
        led = self._ledgers.get(shard.shard_id)
        if led is None:
            p = call_path(self.paths, self.cfg, shard.kind, shard.shard_id)
            if self.store.exists(p):
                led = ShardLedger.from_bytes(self.store.get_bytes(p))
                if led.status in _LIVE:
                    self._emit("reattached", shard_id=shard.shard_id, call_id=led.call_id,
                               status=led.status)
            else:
                led = ShardLedger(shard.shard_id, shard.kind)
            self._ledgers[shard.shard_id] = led
        return led

    def _save(self, led: ShardLedger, **changes: Any) -> None:
        for k, v in changes.items():
            setattr(led, k, v)
        led.updated_utc = utc(self.clock())
        self.store.put_bytes(call_path(self.paths, self.cfg, led.kind, led.shard_id),
                             led.to_bytes())

    def _receipt_exists(self, shard: Shard) -> bool:
        return self.store.exists(self.paths.receipt(self.cfg, shard.kind, shard.shard_id))

    def _control(self, kind: str, reason: str, **detail: Any) -> ControlRecord:
        rec = ControlRecord(kind=kind, target=self.job_id, reason=reason,  # type: ignore[arg-type]
                            created_utc=utc(self.clock()), detail=dict(detail))
        write_control(self.store, self.paths, rec)
        self._own_controls.append(rec)
        return rec

    def _controls(self) -> ControlView:
        seen = {(r.kind, r.created_utc) for r in self._own_controls}
        listed = [r for r in read_controls(self.store, self.paths)
                  if (r.kind, r.created_utc) not in seen]
        return control_view([*listed, *self._own_controls], self.job_id)

    # lifecycle ------------------------------------------------------------------------------
    def acquire(self) -> bool:
        self._held = self.lease.acquire(self.holder, self.lease_ttl_s)
        if self._held:
            self._emit("lease_acquired")
        else:
            cur = self.lease.current()
            self._emit("lease_held", by=cur.holder if cur else None)
        return self._held

    def release(self) -> None:
        if self._held:
            self.lease.release(self.holder)
            self._held = False
            self._emit("lease_released")

    def run_until_idle(self, *, sleep: Callable[[float], None] = time.sleep,
                       poll_interval_s: float = POLL_INTERVAL_S,
                       max_ticks: int | None = None) -> RunOutcome:
        if not self._held and not self.acquire():
            return RunOutcome("lease_held", 0, 0)
        ticks = 0
        try:
            while True:
                result = self.tick()
                ticks += 1
                if result != "running":
                    return RunOutcome(result, ticks, self.spawned)
                if max_ticks is not None and ticks >= max_ticks:
                    return RunOutcome("max_ticks", ticks, self.spawned)
                sleep(poll_interval_s)
        finally:
            self.release()

    # the loop body --------------------------------------------------------------------------
    def tick(self) -> TickResult:
        if not self.lease.renew(self.holder, self.lease_ttl_s):
            self._held = False
            self._emit("lease_lost")
            return "lost_lease"
        manifest = self.manifests.load()
        shards = list(manifest.shards)
        for s in shards:
            if s.kind not in self.settings.pools:
                raise ValueError(f"shard {s.shard_id!r}: no pool configured for kind {s.kind!r}")
        by_id = {s.shard_id: s for s in shards}
        controls = self._controls()

        if controls.cancelled is not None:
            for s in shards:
                led = self._ledger(s)
                if led.status in _LIVE:
                    if led.call_id:
                        self.spawner.cancel(led.call_id)
                    self._save(led, status="cancelled", reason="job cancelled")
            self._emit("cancelled", reason=controls.cancelled.reason)
            return "cancelled"

        quarantined = self.attempts.quarantined()
        for s in shards:
            led = self._ledger(s)
            if led.status == "spawning":
                # crash between "intent" and "call id" writes: the call may or may not exist.
                self._emit("orphan_spawn_possible", shard_id=s.shard_id)
                self._save(led, status="retry")
            elif led.status in ("new", "retry", "active") and s.shard_id in quarantined:
                # quarantine record written but ledger not updated before a crash: finish it
                rec = quarantined[s.shard_id]
                if rec.reason == "respawns_exhausted":
                    self._save(led, status="failed", call_id=None, reason=rec.reason)
                else:
                    self._quarantine(s, led, reason=rec.reason, redo=True)
            elif led.status == "active":
                self._watch(s, led)

        # re-load if bisection added shards this tick
        if any(led.status == "quarantined" and led.children and
               not set(led.children) <= set(by_id) for led in self._ledgers.values()):
            shards = list(self.manifests.load().shards)
            by_id = {s.shard_id: s for s in shards}

        def status(sid: str) -> str:
            s = by_id.get(sid)
            return self._ledger(s).status if s else "new"

        satisfied, dead = resolve(shards, status)

        if self.clock() - self.started_at >= self.deadline_s:
            self._emit("continue", active=self._active_count())
            self.release()
            return "continue"

        stopped = controls.budget_stopped is not None
        if not stopped:
            stopped = not self._spawn_eligible(shards, satisfied, dead, status)

        active = self._active_count()
        if all(sid in satisfied or sid in dead for sid in by_id) and active == 0:
            result: TickResult = "failed" if dead else "done"
            self._emit(result, shards=len(by_id), dead=sorted(dead))
            return result
        if stopped and active == 0:
            self._emit("budget_stopped")
            return "budget_stopped"
        return "running"

    def _active_count(self) -> int:
        return sum(1 for led in self._ledgers.values() if led.status in _LIVE)

    def _watch(self, s: Shard, led: ShardLedger) -> None:
        assert led.call_id
        if self._receipt_exists(s):
            self._finish(s, led, "receipt")
            return
        starts = self.attempts.starts(s)
        if starts >= self.k:
            self._emit("poison_detected", shard_id=s.shard_id, call_id=led.call_id, starts=starts)
            self._quarantine(s, led, reason="poison_pill", starts=starts)
            return
        st = self.spawner.poll(led.call_id)
        if st.state == "pending":
            return
        if st.state == "success":
            self._finish(s, led, "success")
            return
        # failed / expired / cancelled by someone else: bounded respawn
        self._emit("call_" + st.state, shard_id=s.shard_id, call_id=led.call_id, error=st.error)
        led.calls.append({"call_id": led.call_id, "outcome": st.state, "error": st.error,
                          "utc": utc(self.clock())})
        if led.respawns >= self.max_respawns:
            self.attempts.quarantine(s, "respawns_exhausted",
                                     {"last_state": st.state, "error": st.error,
                                      "respawns": led.respawns})
            self._save(led, status="failed", call_id=None,
                       reason=f"{st.state} after {led.respawns} respawns: {st.error}")
            self._emit("shard_failed", shard_id=s.shard_id, reason=led.reason)
        else:
            self._save(led, status="retry", call_id=None, respawns=led.respawns + 1)

    def _finish(self, s: Shard, led: ShardLedger, how: str) -> None:
        led.calls.append({"call_id": led.call_id, "outcome": "success", "utc": utc(self.clock())})
        self._save(led, status="done")
        self._emit("shard_done", shard_id=s.shard_id, call_id=led.call_id, via=how)

    def _quarantine(self, s: Shard, led: ShardLedger, *, reason: str, starts: int | None = None,
                    redo: bool = False) -> None:
        """D15: cancel the looping call, quarantine, then bisect (or isolate a single episode)."""
        if led.call_id:
            self.spawner.cancel(led.call_id)
            led.calls.append({"call_id": led.call_id, "outcome": "cancelled_poison",
                              "utc": utc(self.clock())})
        detail = {"starts": starts, "call_id": led.call_id}
        if not redo:
            self.attempts.quarantine(s, reason, detail)
        self._emit("quarantined", shard_id=s.shard_id, reason=reason, starts=starts)
        if len(s.episodes) <= 1:
            self.attempts.record_poison(s, {**detail, "reason": reason})
            self._save(led, status="poisoned", call_id=None, reason=reason)
            self._emit("poisoned", shard_id=s.shard_id, episodes=list(s.episodes))
            return
        done = self.progress.episodes_done(s)
        children = self.attempts.bisect(s, done)
        self.manifests.replan(s, children)
        self._save(led, status="quarantined", call_id=None, reason=reason,
                   children=tuple(c.shard_id for c in children))
        self._emit("bisected", shard_id=s.shard_id, children=[c.shard_id for c in children],
                   done=sorted(done))

    # spawning -------------------------------------------------------------------------------
    def _in_flight(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for led in self._ledgers.values():
            if led.status in _LIVE:
                out[led.kind] = out.get(led.kind, 0) + 1
        return out

    def _gpus(self, in_flight: Mapping[str, int]) -> int:
        """GPUs the in-flight calls can occupy: ceil(calls / max_inputs) containers per kind."""
        total = 0
        for kind, n in in_flight.items():
            pool = self.settings.pools[kind]  # type: ignore[index]
            total += math.ceil(n / max(pool.max_inputs, 1)) * pool.gpu_count
        return total

    def _spawn_eligible(self, shards: Sequence[Shard], satisfied: set[str], dead: set[str],
                        status: Callable[[str], str]) -> bool:
        """Spawn what fits. Returns False if the budget gate stopped the job."""
        in_flight = self._in_flight()
        for s in shards:
            led = self._ledger(s)
            if led.status not in ("new", "retry") or s.shard_id in dead:
                continue
            if not all(d in satisfied for d in s.depends_on):
                continue
            pool = self.settings.pools[s.kind]
            if in_flight.get(s.kind, 0) >= pool.max_containers * pool.max_inputs:
                continue
            trial = {**in_flight, s.kind: in_flight.get(s.kind, 0) + 1}
            if self._gpus(trial) > self.settings.max_gpus:
                continue
            if self._receipt_exists(s):  # e.g. a previous call finished but its ledger write lost
                self._finish(s, led, "receipt")
                continue
            decision = self.budget.admit(s, dict(in_flight))
            if not decision.allow:
                self._control("budget_stop", decision.reason or "budget gate",
                              **dict(decision.detail))
                self._emit("budget_stop", reason=decision.reason, active=self._active_count())
                return False
            self._save(led, status="spawning", spawn_n=led.spawn_n + 1)
            try:
                call_id = self.spawner.spawn(s.kind, self.job_id, s.shard_id, self.cfg)
            except Exception as exc:  # Modal API hiccup: try again next tick
                self._save(led, status="retry")
                self._emit("spawn_error", shard_id=s.shard_id, error=repr(exc))
                return True
            led.calls.append({"call_id": call_id, "outcome": "spawned", "spawn_n": led.spawn_n,
                              "utc": utc(self.clock())})
            self._save(led, status="active", call_id=call_id)
            self.spawned += 1
            in_flight = trial
            self._emit("spawned", shard_id=s.shard_id, kind=s.kind, call_id=call_id,
                       spawn_n=led.spawn_n, gpus=self._gpus(in_flight))
        return True
