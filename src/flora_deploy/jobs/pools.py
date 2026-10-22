"""End a worker pool's idle tail when its work is over, and the orchestrator's own footprint.

**Why.** Every deployed worker class keeps an idle container for its ``scaledown_window``
(``jobs.settings.WORKER_SCALEDOWN_S``, 300 s) after its last input, so the next shard lands on a
warm container. After a pool's *last* shard that wait buys nothing: on the first real balanced
job (6 episodes) the segment L40S idled its full 300 s after its last shard ($0.23 of its $0.50).

**What modal 1.6.2 can do** (read from the installed client): ``Cls.from_name(app, cls)()
.update_autoscaler(scaledown_window=..., min_containers=..., max_containers=...,
buffer_containers=...)`` overrides one class's autoscaler until the next deploy
(``FunctionUpdateSchedulingParams``; ``Function.autoscaler.update`` is the newer spelling, and
``update_autoscaler`` warns as deprecated since 2026-10-16). ``scaledown_window`` must be in
[2, 3600]; ``max_containers`` cannot be 0 through the new API, and there is no per-class "stop
idle containers" call (``modal container stop`` takes one container id and cannot tell an idle
container from one serving another job; ``modal app stop`` stops every class). So the lever is
per class: drop ``scaledown_window`` to :data:`DRAINED_SCALEDOWN_S` when the class has nothing
left to run, and put it back before it gets work again. Whether Modal re-applies a lowered window
to a container that is *already* idle (rather than only to containers going idle afterwards) is
not documented; one cheap live check settles it (``docs/cold-start.md``, "Ending idle").

**Rules** (:class:`PoolScaler`, driven by ``Orchestrator.tick``):

- *Demand.* A shard kind has demand while any of its shards is neither satisfied nor dead
  (``orchestrator.resolve``): queued, retrying, live, waiting for a dependency, or a quarantined
  parent whose children are not all known. A ``caption`` shard waiting on ``segment`` is
  demand, so caption is never drained while segment still produces. Kinds that share a worker
  class (``reward.terminal`` / ``reward.dense``) share its demand.
- *Several jobs on one app.* One app serves every job of its config. Each orchestrator records
  its demand per class in the app's pool Dict (``<watch dict>-pools``, key ``<class>/<job>``);
  a class is drained only when no other job's entry says busy (entries older than
  :data:`STALE_S` are a dead orchestrator's and are ignored; a revived one re-marks itself busy
  and restores the window before it spawns).
- *Restore before work.* A busy job re-reads the class state every tick and restores the
  deployed window if anyone drained it, before ``_spawn_eligible`` runs. Modal ``Dict`` has no
  compare-and-swap, so a drain racing another job's restore can win; the busy job puts it back
  on its next tick (one poll interval of short window: at worst an extra cold start, never a
  lost shard).
- *Preemption.* A preempted worker input stays the same live call, so its kind keeps demand.
  The drain never touches ``max_containers``: a re-spawned or retried shard always gets a
  container (a cold one if the warm one already left).
- *Job end* (``done``, ``completed_with_errors``, ``failed``, ``cancelled``,
  ``budget_stopped``): the job withdraws its entries and drains whatever no other job needs.
  ``min_containers`` warm containers (``flora deploy --min-containers``) are left alone.

**Pre-warming a dependent pool** (:meth:`PoolScaler.prewarm`). Caption shards are spawned only
once their segment dependency is met (``orchestrator.deps_ready``), so on a small job caption's
cold start (~325 s, ``planner/rates.json`` startup row) runs *after* segment's, serially. When a
kind has dependencies, no call of its own yet and no warm pool (``min_containers`` 0), the
orchestrator estimates when its first shard becomes runnable (:func:`first_ready_at`: each live
dependency call's spawn time + the dependency's startup + its work estimate, from the budget
gate's estimator) and, at ``ready - own startup``, raises the class's ``min_containers`` to 1 so
a container is booting while segment finishes. Once the kind's first call is live (it lands on
that container), ``min_containers`` goes back to the deployed value. Decided once per kind and
recorded as a ``pool_prewarm_plan`` event (``prewarm`` or ``skip`` with the reason and the
expected saving) and in the job's pool file (``prewarm_plans``, :func:`read_prewarm_plans`: the
decision time ``t`` and the latest ``start_at`` / ``ready_at``, so ``flora report`` and the H0
rehearsal can compare the planned start with the container's real one; a successor generation
reloads it and does not decide again); ``pool_prewarm_started`` / ``pool_prewarm_released``
follow. Cost: the error of the ready estimate in idle H100 time (about $0.09 per minute early);
saving: up to one caption startup of wall-clock. The warm-up container is charged to the budget
from its start (``BudgetState.prewarms``, ``jobs.budget.calls_cost``). Two jobs on one app
share the class's ``min_containers`` override: one job's release can end another's pre-warm
early (a cold start, never a lost shard).

**Budget.** Drains are recorded per kind as ``(start, end | None)`` intervals
(:meth:`PoolScaler.drains`, persisted to :func:`drains_path` so a successor generation and
``flora report`` see them) and passed to ``jobs.budget.calls_cost``, which charges a container
going idle only until ``drain start + DRAINED_SCALEDOWN_S`` instead of the full window
(:func:`idle_tail_s`).

**The orchestrator itself** (constants below; ``modal_app.factory`` and
``modal_app.orchestrator_app`` deploy it with them, ``jobs.deps`` and ``planner.options`` price
it). It sleeps ``POLL_INTERVAL_S`` between ticks and a tick is store listings and small JSON
files: importing everything it uses peaks at about 60 MiB RSS and 0.3 CPU-s locally. It ran at
0.25 cores / 1 GiB, non-preemptible (3x CPU and memory price), with Modal's default 60 s idle
after every call. Now: 0.125 cores (Modal's minimum request; usage above it is billed as used,
not throttled), memory request 256 MiB with the old 1 GiB as the hard limit (billed at
max(request, use), so no new OOM risk), **preemptible**, and a 2 s ``scaledown_window`` on the
orchestrator and reconciler functions. Preemption is survivable by design (PLAN.md G4, §2): the
lease is a ``modal.Dict`` entry held by the function *call* id, which a preempted-and-restarted
input keeps, so it re-takes its own lease at once; in-flight shards are re-attached from their
``call.json`` ledgers; a crash between spawn intent and call-id write is adopted by spawn token
or respawned after ``SPAWN_GRACE_S``. The cost of a preemption is one CPU cold start (seconds),
well inside the workers' 300 s window, so no worker idles out because of it.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

from flora_deploy.jobs.types import safe_key

DRAINED_SCALEDOWN_S = 2  # modal 1.6.2: scaledown_window must be in [2, 3600]
STALE_S = 15 * 60.0  # another job's pool entry older than this is a dead orchestrator's
HEARTBEAT_S = 60.0  # re-write an unchanged pool entry this often (keeps it fresh)
POOL_DICT_SUFFIX = "-pools"  # the app's pool Dict: ``<watch dict>-pools`` (modal_app.factory)
POOL_DICT_ENV = "FLORA_POOL_DICT"

# The orchestrator's container (module docstring).
ORCHESTRATOR_CPU = 0.125
ORCHESTRATOR_MEMORY_MIB = 256  # request
ORCHESTRATOR_MEMORY_LIMIT_MIB = 1024  # the old request, kept as the OOM limit
ORCHESTRATOR_PREEMPTIBLE = True
ORCHESTRATOR_SCALEDOWN_S = 2  # idle after a generation / reconcile call returns
ORCHESTRATOR_CORES = ORCHESTRATOR_CPU  # billed as requested (usage above it is extra)
ORCHESTRATOR_MEMORY_GIB = ORCHESTRATOR_MEMORY_MIB / 1024

Interval = tuple[float, float | None]
Drains = Mapping[str, tuple[Interval, ...]]


class Autoscaler(Protocol):
    def set_scaledown(self, cls_name: str, seconds: int) -> None:
        """Override worker class ``cls_name``'s ``scaledown_window`` (until the next deploy)."""
        ...

    def set_min_containers(self, cls_name: str, n: int) -> None:
        """Override worker class ``cls_name``'s ``min_containers`` (until the next deploy)."""
        ...


class PoolKV(Protocol):
    """The subset of ``modal.Dict`` the pool registry needs."""

    def put(self, key: Any, value: Any) -> Any: ...
    def get(self, key: Any, default: Any | None = None) -> Any: ...
    def pop(self, key: Any, default: Any | None = None) -> Any: ...
    def items(self) -> Iterable[tuple[Any, Any]]: ...


def pool_dict_name(watch_dict: str) -> str:
    return f"{watch_dict}{POOL_DICT_SUFFIX}"


def drains_path(root: str, config_hash: str) -> str:
    """Where a job's drain intervals live (``JobPaths(job).root``, one file per config)."""
    return f"{root}/{safe_key(config_hash)}/pools/drains.json"


def read_drains(store: Any, root: str, config_hash: str
                ) -> tuple[dict[str, tuple[Interval, ...]], dict[str, Interval]]:
    """``(drains, prewarms)`` a job's orchestrator recorded (empty when none or unreadable)."""
    try:
        p = drains_path(root, config_hash)
        if not store.exists(p):
            return {}, {}
        raw = json.loads(store.get_bytes(p))
        drains = {str(k): tuple((float(a), None if b is None else float(b)) for a, b in v)
                  for k, v in dict(raw.get("drains", {})).items()}
        prewarms = {str(k): (float(a), None if b is None else float(b))
                    for k, (a, b) in dict(raw.get("prewarms", {})).items()}
        return drains, prewarms
    except Exception:  # the report over-counts (full windows), never under
        return {}, {}


def read_prewarm_plans(store: Any, root: str, config_hash: str) -> dict[str, dict[str, Any]]:
    """Kind -> the pre-warm decision a job's orchestrator recorded (``decision``, ``t``, and
    ``start_at`` / ``ready_at`` / ``saves_s`` or ``reason``); empty when none or unreadable."""
    try:
        p = drains_path(root, config_hash)
        if not store.exists(p):
            return {}
        raw = json.loads(store.get_bytes(p))
        return {str(k): dict(v) for k, v in dict(raw.get("prewarm_plans", {})).items()}
    except Exception:
        return {}


def idle_tail_s(t: float, scaledown_s: float, drains: Iterable[Interval] = (),
                drained_s: float = DRAINED_SCALEDOWN_S) -> float:
    """Idle seconds billed for a container that went idle at ``t``: the deployed window, cut
    short by a drain in force at ``t`` (``drained_s``) or one starting inside the window
    (until ``drain start + drained_s``)."""
    end = t + scaledown_s
    for d0, d1 in drains:
        if d0 <= t and (d1 is None or t < d1):
            end = min(end, t + drained_s)
        elif t < d0 < end:
            end = min(end, d0 + drained_s)
    return max(min(end - t, scaledown_s), 0.0)


def kind_demand(shards: Iterable[Any], satisfied: set[str], dead: set[str]) -> dict[str, bool]:
    """Kind -> whether any of its shards still needs a worker (module docstring)."""
    out: dict[str, bool] = {}
    for s in shards:
        busy = s.shard_id not in satisfied and s.shard_id not in dead
        out[s.kind] = out.get(s.kind, False) or busy
    return out


def first_ready_at(kind: str, shards: Iterable[Any], *, satisfied: set[str],
                   live_spawned: Mapping[str, float], now: float,
                   startup_s: Callable[[str], float],
                   work_s: Callable[[Any], float]) -> float | None:
    """Earliest expected time a ``kind`` shard with dependencies becomes runnable, or None when
    no such shard has every dependency either satisfied or live (nothing to time it from).

    A satisfied dependency is ready now; a live one at ``spawned + startup(dep kind) +
    work(dep)`` (never before ``now``)."""
    by_id = {s.shard_id: s for s in shards}
    best: float | None = None
    for s in by_id.values():
        if s.kind != kind or not s.depends_on or s.shard_id in satisfied:
            continue
        ready: float | None = now
        for d in s.depends_on:
            if d in satisfied:
                continue
            dep = by_id.get(d)
            if dep is None or d not in live_spawned:
                ready = None
                break
            assert ready is not None
            ready = max(ready, live_spawned[d] + startup_s(dep.kind) + work_s(dep))
        if ready is not None and (best is None or ready < best):
            best = ready
    return best


class PoolScaler:
    """Per job: drain worker classes with no remaining work, restore them before work.

    ``classes``: shard kind -> worker class name (``OrchestratorDeps.worker_classes``).
    ``static_s``: the deployed ``scaledown_window`` (restored). Errors from the autoscaler or the
    Dict never stop a tick: they are reported (``pool_scale_error``) and retried next tick."""

    def __init__(self, *, job_id: str, classes: Mapping[str, str], static_s: float,
                 autoscaler: Autoscaler, kv: PoolKV, clock: Callable[[], float] = time.time,
                 events: Callable[[dict[str, Any]], None] | None = None,
                 drained_s: int = DRAINED_SCALEDOWN_S, stale_s: float = STALE_S,
                 heartbeat_s: float = HEARTBEAT_S, store: Any = None, path: str | None = None,
                 startup_s: Callable[[str], float] | None = None,
                 work_s: Callable[[Any], float] | None = None):
        # pre-warm estimates (the budget gate's estimator); None: no pre-warm
        self.startup_s, self.work_s = startup_s, work_s
        self.job_id = job_id
        self.classes = dict(classes)
        self.static_s = int(math.ceil(static_s))
        self.autoscaler, self.kv, self.clock = autoscaler, kv, clock
        self.events = events or (lambda e: None)
        self.drained_s, self.stale_s, self.heartbeat_s = drained_s, stale_s, heartbeat_s
        self.store, self.path = store, path
        self.enabled = self.static_s > drained_s and bool(self.classes)
        self._written: dict[str, tuple[bool, float]] = {}  # class -> (busy, at) last put
        self._drains: dict[str, list[list[float | None]]] = {}
        self.base_min = 0  # the deployed min_containers (restored after a pre-warm)
        self._prewarm_plan: dict[str, dict[str, Any]] = {}  # kind -> decision (once)
        self._plan_meta: dict[str, dict[str, Any]] = {}  # kind -> decided at, own startup
        self._prewarms: dict[str, list[float | None]] = {}  # kind -> [start, end | None]
        self._dep_kinds: set[str] = set()  # kinds with dependencies (released at job end)
        if store is not None and path is not None:
            self._load()

    # persistence ------------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            if not self.store.exists(self.path):
                return
            raw = json.loads(self.store.get_bytes(self.path))
        except Exception:  # unreadable: the budget over-counts (full windows), never under
            return
        for kind, ivs in dict(raw.get("drains", {})).items():
            self._drains[str(kind)] = [[float(a), None if b is None else float(b)]
                                       for a, b in ivs]
        for kind, (a, b) in dict(raw.get("prewarms", {})).items():
            self._prewarms[str(kind)] = [float(a), None if b is None else float(b)]
            self._dep_kinds.add(str(kind))
        for kind, plan in dict(raw.get("prewarm_plans", {})).items():
            self._prewarm_plan[str(kind)] = {k: v for k, v in dict(plan).items()
                                             if k not in ("t", "startup_s")}
            self._plan_meta[str(kind)] = {k: plan[k] for k in ("t", "startup_s") if k in plan}

    def _save(self) -> None:
        if self.store is None or self.path is None:
            return
        plans = {k: {**p, **self._plan_meta.get(k, {})} for k, p in self._prewarm_plan.items()}
        data = {"schema": "flora_pool_drains_v1", "drained_s": self.drained_s,
                "drains": self._drains, "prewarms": self._prewarms, "prewarm_plans": plans}
        try:
            self.store.put_bytes(self.path, json.dumps(data, sort_keys=True).encode())
        except Exception as exc:
            self._emit("pool_scale_error", op="save", error=repr(exc))

    # views ------------------------------------------------------------------------------------
    def drains(self) -> dict[str, tuple[Interval, ...]]:
        """Kind -> drain intervals ``(start, end | None)`` (``end``: restored)."""
        return {k: tuple((float(a), b) for a, b in ivs)  # type: ignore[arg-type]
                for k, ivs in self._drains.items()}

    def prewarms(self) -> dict[str, tuple[float, float | None]]:
        """Kind -> ``(start, end | None)`` of its pre-warm container's wait for a call
        (``end``: its first call was spawned, or the pre-warm was released)."""
        return {k: (float(v[0]), v[1]) for k, v in self._prewarms.items()}  # type: ignore[arg-type]

    # pre-warm ---------------------------------------------------------------------------------
    def tick_prewarm(self, shards: Iterable[Any], *, satisfied: set[str],
                     live_spawned: Mapping[str, float], called_kinds: set[str],
                     min_containers: int = 0) -> None:
        """Every tick: :meth:`prewarm` each kind that has dependencies (``called_kinds``: kinds
        with any call in this job, live or ended)."""
        if not self.enabled or self.startup_s is None or self.work_s is None:
            return
        shards = list(shards)
        startup, work = self.startup_s, self.work_s
        for kind in sorted({s.kind for s in shards if s.depends_on}):
            self._dep_kinds.add(kind)
            try:
                ready = first_ready_at(kind, shards, satisfied=satisfied,
                                       live_spawned=live_spawned, now=self.clock(),
                                       startup_s=startup, work_s=work)
                own = float(startup(kind))
            except Exception as exc:  # an estimator gap never stops a tick
                self._emit("pool_scale_error", op="prewarm_estimate", kind=kind, error=repr(exc))
                continue
            self.prewarm(kind, ready_at=ready, startup_s=own, has_call=kind in called_kinds,
                         min_containers=min_containers)

    def prewarm(self, kind: str, *, ready_at: float | None, startup_s: float, has_call: bool,
                min_containers: int = 0) -> None:
        """Pre-warm ``kind``'s class so it is ready about when its first shard is runnable
        (module docstring). Called every tick for each kind with dependencies."""
        cls = self.classes.get(kind)
        if cls is None:
            return
        self.base_min = int(min_containers)
        try:
            self._prewarm_step(kind, cls, ready_at, float(startup_s), has_call, self.clock())
        except Exception as exc:  # Modal API hiccup: try again next tick
            self._emit("pool_scale_error", worker_class=cls, op="prewarm", error=repr(exc))

    def _prewarm_step(self, kind: str, cls: str, ready_at: float | None, startup: float,
                      has_call: bool, now: float) -> None:
        cur = self._prewarms.get(kind)
        if cur is not None:
            if cur[1] is None and has_call:  # its first call is live: back to the deployed min
                self.autoscaler.set_min_containers(cls, self.base_min)
                cur[1] = now
                self._save()
                self._emit("pool_prewarm_released", worker_class=cls, kind=kind,
                           idle_s=round(now - float(cur[0]), 1))
            return
        plan = self._prewarm_plan.get(kind)
        if plan is None:
            if has_call:
                plan = {"decision": "skip", "reason": "the pool already has a call"}
            elif self.base_min > 0:
                plan = {"decision": "skip", "reason": "min_containers already keeps it warm"}
            elif startup <= 0:
                plan = {"decision": "skip", "reason": "no startup to hide"}
            elif ready_at is None:
                return  # nothing to time it from yet (dependencies not spawned): decide later
            else:
                saves = min(startup, max(ready_at - now, 0.0))
                plan = ({"decision": "prewarm", "start_at": ready_at - startup,
                         "ready_at": ready_at, "saves_s": round(saves, 1)} if saves > 0 else
                        {"decision": "skip", "reason": "its first shard is runnable now"})
            self._prewarm_plan[kind] = plan
            self._plan_meta[kind] = {"t": now, "startup_s": startup}
            self._save()
            self._emit("pool_prewarm_plan", worker_class=cls, kind=kind, startup_s=startup,
                       **plan)
        if plan["decision"] != "prewarm" or has_call:
            return
        if ready_at is not None:  # follow the estimate as dependencies progress
            plan["start_at"], plan["ready_at"] = ready_at - startup, ready_at
        if now >= float(plan["start_at"]):
            self.autoscaler.set_min_containers(cls, max(self.base_min, 1))
            self._prewarms[kind] = [now, None]
            self._save()
            self._emit("pool_prewarm_started", worker_class=cls, kind=kind,
                       ready_at=plan.get("ready_at"))

    def _release_prewarms(self) -> None:
        """Job end: put ``min_containers`` back on every class a pre-warm may have raised, also
        one an earlier generation raised (never leave a warm GPU behind)."""
        if not self.enabled or self.startup_s is None:
            return
        now = self.clock()
        for kind in sorted(self._dep_kinds | set(self._prewarms)):
            cls = self.classes.get(kind)
            if cls is None:
                continue
            try:
                self.autoscaler.set_min_containers(cls, self.base_min)
            except Exception as exc:
                self._emit("pool_scale_error", op="prewarm_release", error=repr(exc))
                continue
            cur = self._prewarms.get(kind)
            if cur is not None and cur[1] is None:
                cur[1] = now
                self._emit("pool_prewarm_released", worker_class=cls, kind=kind,
                           idle_s=round(now - float(cur[0]), 1))
        self._save()

    def _emit(self, event: str, **kw: Any) -> None:
        self.events({"event": event, "job_id": self.job_id, "t": self.clock(), **kw})

    def _kinds_of(self, cls: str) -> list[str]:
        return sorted(k for k, c in self.classes.items() if c == cls)

    def _state_key(self, cls: str) -> str:
        return f"{cls}/state"

    def _job_key(self, cls: str) -> str:
        return f"{cls}/job/{self.job_id}"

    def _open(self, kind: str) -> bool:
        ivs = self._drains.get(kind)
        return bool(ivs) and ivs[-1][1] is None

    def _mark(self, cls: str, drained: bool, now: float) -> None:
        changed = False
        for kind in self._kinds_of(cls):
            if drained and not self._open(kind):
                self._drains.setdefault(kind, []).append([now, None])
                changed = True
            elif not drained and self._open(kind):
                self._drains[kind][-1][1] = now
                changed = True
        if changed:
            self._save()

    # the per-tick step ------------------------------------------------------------------------
    def update(self, demand: Mapping[str, bool], *, final: bool = False) -> None:
        """Apply this tick's demand. ``final``: the job ended; withdraw and drain."""
        if not self.enabled:
            return
        now = self.clock()
        for cls in sorted(set(self.classes.values())):
            busy = (not final) and any(demand.get(k, False) for k in self._kinds_of(cls))
            try:
                self._step(cls, busy, final, now)
            except Exception as exc:  # Modal API / Dict hiccup: try again next tick
                self._emit("pool_scale_error", worker_class=cls, error=repr(exc))

    def final(self) -> None:
        self._release_prewarms()
        self.update({}, final=True)

    def _step(self, cls: str, busy: bool, final: bool, now: float) -> None:
        last = self._written.get(cls)
        if final:
            self.kv.pop(self._job_key(cls), None)
            self._written.pop(cls, None)
        elif last is None or last[0] != busy or now - last[1] >= self.heartbeat_s:
            self.kv.put(self._job_key(cls), {"busy": busy, "t": now})
            self._written[cls] = (busy, now)
        state = self.kv.get(self._state_key(cls)) or {}
        window = int(state.get("window", self.static_s))
        if busy:
            if window != self.static_s:  # drained by us or another job: put it back first
                self.autoscaler.set_scaledown(cls, self.static_s)
                self.kv.put(self._state_key(cls), {"window": self.static_s, "by": self.job_id,
                                                   "t": now})
                self._emit("pool_restored", worker_class=cls, scaledown_s=self.static_s,
                           kinds=self._kinds_of(cls))
            self._mark(cls, False, now)
            return
        if window == self.drained_s:
            self._mark(cls, True, now)  # someone else drained it (still drained for our bill)
            return
        others = self._busy_others(cls, now)
        if others:
            return
        self.autoscaler.set_scaledown(cls, self.drained_s)
        self.kv.put(self._state_key(cls), {"window": self.drained_s, "by": self.job_id, "t": now})
        self._mark(cls, True, now)
        self._emit("pool_drained", worker_class=cls, scaledown_s=self.drained_s,
                   kinds=self._kinds_of(cls), final=final)

    def _busy_others(self, cls: str, now: float) -> list[str]:
        prefix = f"{cls}/job/"
        out: list[str] = []
        for key, val in list(self.kv.items()):
            k = str(key)
            if not k.startswith(prefix) or k == self._job_key(cls):
                continue
            try:
                if val.get("busy") and now - float(val.get("t", 0.0)) < self.stale_s:
                    out.append(k[len(prefix):])
            except (AttributeError, TypeError, ValueError):
                continue  # malformed entry: ignore
        return out
