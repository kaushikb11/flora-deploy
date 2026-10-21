"""End-to-end: real stores + real P1a-2 manifest/attempts/progress (through ``jobs.deps``) + the
real orchestrator and reconciler, driven by a fake Modal (``TokenSpawner``) and a fake worker that
follows the ``workers.py`` contract step by step (superseded check, marker with token + call id,
skip visible parts, skip upstream-poisoned episodes, whole parts, preempt / exception marks,
receipt LAST naming every part).

Each scenario checks the orchestrator's verdict *and* the file-derived truth: ``progress.derive``
(complete), ``compose_input`` (nothing missing, nothing duplicated wrongly) and the stateless
reconciler (same job state)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.deps import StoreManifestSource, build_deps
from flora_deploy.jobs.manifest import load_manifest, save_manifest, stages_of
from flora_deploy.jobs.orchestrator import (
    DictLease,
    Orchestrator,
    RunOutcome,
    control_view,
    resolve,
    utc,
    write_control,
)
from flora_deploy.jobs.reconciler import needs_orchestrator, reconcile
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import ControlRecord, Manifest, Shard, ShardReceipt
from flora_deploy.records import Provenance, Segment, StageRecord
from tests.fakes.jobs import FakeBudget, FakeClock, FakeDict, TokenSpawner
from tests.fakes.modal import FakeStore, Preempted

JOB, CFG = "job1", "cfg1"
SEGCAP = {"segcap": PoolSettings(gpu="H200", max_containers=2)}


class TickingLocalStore(LocalStore):
    """LocalStore (strongly consistent) with the ``tick`` FakeSpawner calls."""

    def tick(self) -> None:
        pass


def record(uid: str, stage: str, cfg: str) -> StageRecord:
    return StageRecord(uid=uid, dataset="d", episode_id=uid, stage=stage,  # type: ignore[arg-type]
                       provenance=Provenance(models={}, engine="fake", preset="exact",
                                             config_hash=cfg, logic_version="1",
                                             framework="test"),
                       certified=True, segments=(Segment(f"{uid}:{stage}", 0.0, 1.0),))


@dataclass
class Behavior:
    oom: set[str] = field(default_factory=set)  # crash (no mark) every time the uid is reached
    oom_kinds: set[str] | None = None  # restrict ``oom`` to these shard kinds
    preempt: dict[str, int] = field(default_factory=dict)  # shard -> preemptions left
    preempt_after: int = 1  # ...after this many new episodes in the start
    preempt_after_by: dict[str, int] = field(default_factory=dict)  # per-shard override
    pyfail: dict[str, int] = field(default_factory=dict)  # shard -> python exceptions left
    receipt_parts: str = "all"  # "all" (contract) | "own" (a sloppy worker)


class World:
    def __init__(self, spec: dict[str, list[str]], *, lag: int = 0, local=None,
                 kinds: dict[str, str] | None = None, deps: dict[str, list[str]] | None = None,
                 pools: dict[str, PoolSettings] | None = None, behavior: Behavior | None = None,
                 python_retries: int = 0, budget=None, cfg: str = CFG, store=None,
                 kv: FakeDict | None = None, clock: FakeClock | None = None):
        self.cfg = cfg
        self.store = store or (TickingLocalStore(local) if local is not None else FakeStore(lag))
        self.clock = clock or FakeClock()
        self.pools = dict(pools or SEGCAP)
        self.settings = RunSettings(job_id=JOB, preset="exact", config_hash=cfg, budget_usd=10.0,
                                    max_gpus=8, pools=self.pools)  # type: ignore[arg-type]
        shards = tuple(Shard(sid, (kinds or {}).get(sid, "segcap"), tuple(eps),  # type: ignore
                             60.0 * len(eps), depends_on=tuple((deps or {}).get(sid, ())))
                       for sid, eps in spec.items())
        self.manifest = Manifest(JOB, cfg, (), shards, "t")
        save_manifest(self.store, self.manifest)
        self.jf = A.JobFiles(self.store, JOB, cfg)
        self.b = behavior or Behavior()
        self.budget = budget
        self.deps = build_deps(self.store, self.settings, budget=budget, clock=self.clock)
        self.processed: list[tuple[str, str]] = []
        self.superseded: list[str] = []
        self.spawner = TokenSpawner(self.store, {k: self._worker(k) for k in self.pools},
                                    pools=self.pools, python_retries=python_retries,
                                    max_restarts=50)
        self.kv = kv or FakeDict()
        self.events: list[dict] = []

    # the contract worker ----------------------------------------------------------------------
    def _worker(self, kind: str):
        b = self.b

        def run(job_id, shard_id, cfg, container, store, *, call_id, token):
            jf = A.JobFiles(store, job_id, cfg)
            if A.superseded(jf, kind, shard_id, token):  # step 0
                self.superseded.append(call_id)
                return None
            src = StoreManifestSource(jf)  # a fresh container: nothing cached
            shard = src.shard(shard_id)
            marker = A.write_attempt(jf, kind, shard_id, container,  # step 1
                                     started_utc=utc(self.clock()),
                                     orchestrator_attempt=A.token_attempt(token), token=token,
                                     call_id=call_id)
            try:
                if b.pyfail.get(shard_id, 0) > 0:
                    b.pyfail[shard_id] -= 1
                    raise RuntimeError(f"bug in {shard_id}")
                prev = P.shard_parts(jf, shard)  # step 2
                done = set(prev.done)
                skip: set[str] = set()
                if shard.depends_on:  # step 3
                    ups = [s for s in src.load().shards if s.shard_id in shard.depends_on]
                    skip = A.upstream_poisoned(jf, shard, ups) - done
                    A.write_upstream_failures(jf, shard, skip)
                own: list[str] = []
                new = 0
                for k, uid in enumerate(shard.episodes):  # step 4
                    if uid in done or uid in skip:
                        continue
                    if uid in b.oom and (b.oom_kinds is None or kind in b.oom_kinds):
                        raise Preempted("oom")  # the container dies: no mark
                    if b.preempt.get(shard_id, 0) > 0 and \
                            new >= b.preempt_after_by.get(shard_id, b.preempt_after):
                        b.preempt[shard_id] -= 1
                        A.mark_preempted(jf, kind, shard_id, container,  # step 5
                                         ended_utc=utc(self.clock()), marker=marker)
                        raise Preempted("preempt")
                    own.append(P.write_part(jf, kind, shard_id, container, k,
                                            [record(uid, st, cfg) for st in stages_of(kind)]))
                    self.processed.append((shard_id, uid))
                    new += 1
            except Preempted:
                raise
            except Exception as exc:
                A.mark_exception(jf, kind, shard_id, container, ended_utc=utc(self.clock()),
                                 error=repr(exc), marker=marker)
                raise
            parts = tuple(sorted(set(prev.parts) | set(own))) if b.receipt_parts == "all" \
                else tuple(own)
            rec = ShardReceipt(shard_id, kind, cfg, container,  # type: ignore[arg-type]
                               tuple(u for u in shard.episodes if u not in skip),
                               tuple(u for u in shard.episodes if u in skip), parts, "H200", 1.0,
                               utc(self.clock()))
            P.write_receipt(jf, rec)  # step 6: LAST
            return rec

        return run

    # driving ----------------------------------------------------------------------------------
    def lease(self) -> DictLease:
        return DictLease(self.kv, f"orchestrator/{JOB}", clock=self.clock)

    def orchestrator(self, fresh: bool = True, **kw) -> Orchestrator:
        d = build_deps(self.store, self.settings, budget=self.budget, clock=self.clock) \
            if fresh else self.deps  # a restarted process rebuilds its adapters
        args = dict(settings=self.settings, store=self.store, spawner=self.spawner,
                    lease=self.lease(), manifests=d.manifests, attempts=d.attempts,
                    progress=d.progress, budget=d.budget, events=self.events.append,
                    clock=self.clock)
        args.update(kw)
        return Orchestrator(**args)

    def sleep(self, dt: float) -> None:
        self.clock.advance(dt)
        self.spawner.tick()

    def run(self, orch: Orchestrator | None = None, max_ticks: int = 400) -> RunOutcome:
        orch = orch or self.orchestrator()
        return orch.run_until_idle(sleep=self.sleep, poll_interval_s=10.0, max_ticks=max_ticks)

    # truth from files -------------------------------------------------------------------------
    def flush(self, n: int = 20) -> None:
        for _ in range(n):
            self.store.tick()

    def derive(self):
        self.flush()
        prog = P.derive(self.jf, load_manifest(self.store, JOB),
                        limit=self.settings.max_starts_before_quarantine)
        return prog, P.compose_input(self.jf, prog)

    def status(self):
        d = build_deps(self.store, self.settings, clock=self.clock)
        return reconcile(job_id=JOB, config_hash=self.cfg, store=self.store,
                         manifests=d.manifests, attempts=d.attempts, lease=self.lease(),
                         clock=self.clock, max_starts=self.settings.max_starts_before_quarantine)

    def names(self) -> list[str]:
        return [e["event"] for e in self.events]

    def spawns(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for sid in self.spawner.spawns:
            out[sid] = out.get(sid, 0) + 1
        return out

    def receipt(self, shard_id: str, kind: str = "segcap") -> bool:
        return self.store.exists(JobPaths(JOB).receipt(self.cfg, kind, shard_id))

    def assert_complete(self, *, excused: set[tuple[str, str]] = frozenset()) -> None:
        prog, ci = self.derive()
        assert prog.complete, {k: v.state for k, v in prog.shards.items()}
        assert not ci.missing and not ci.conflicts and ci.foreign == 0
        assert ci.excused == set(excused)
        st = self.status()
        assert st.state == ("completed_with_errors" if excused else "done")
        assert not needs_orchestrator(st), st


def _spec(n: int, eps: int = 3) -> dict[str, list[str]]:
    return {f"segcap-{i:05d}": [f"d--e{i}-{j}" for j in range(eps)] for i in range(n)}


def _stores():
    return [pytest.param({"lag": 0}, id="fake-lag0"), pytest.param({"lag": 2}, id="fake-lag2"),
            pytest.param({"local": True}, id="local")]


def _world(tmp_path, store_kw, spec, **kw) -> World:
    if store_kw.get("local"):
        return World(spec, local=tmp_path / "store", **kw)
    return World(spec, lag=store_kw["lag"], **kw)


# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("store_kw", _stores())
def test_happy_path(tmp_path, store_kw):
    w = _world(tmp_path, store_kw, _spec(5))
    assert w.run().result == "done"
    assert w.spawns() == dict.fromkeys(_spec(5), 1)
    assert sorted(w.processed) == sorted((s, u) for s, eps in _spec(5).items() for u in eps)
    w.assert_complete()


@pytest.mark.parametrize("store_kw", _stores())
def test_preemptions_never_quarantine_or_poison(tmp_path, store_kw):
    """H2: a 6-episode shard preempted 3x (progressing each time) and a single long episode
    preempted 4x both finish normally: preempted starts are not counted."""
    spec = {"segcap-00000": [f"d--a{j}" for j in range(6)], "segcap-00001": ["d--long"]}
    w = _world(tmp_path, store_kw, spec,
               behavior=Behavior(preempt={"segcap-00000": 3}, preempt_after=1))
    w.b.preempt["segcap-00001"] = 4
    w.b.preempt_after_by["segcap-00001"] = 0  # preempted before its only episode finishes
    out = w.run()
    assert out.result == "done"
    assert "quarantined" not in w.names() and "poisoned" not in w.names()
    assert w.spawns() == {"segcap-00000": 1, "segcap-00001": 1}
    call = next(c for c in w.spawner.calls.values() if c.shard_id == "segcap-00001")
    assert call.starts == 5  # four preempted starts + the one that finished
    w.assert_complete()


@pytest.mark.parametrize("store_kw", _stores())
def test_oom_poison_pill_is_bisected_and_isolated(tmp_path, store_kw):
    """D15 with the real modules; C2: under listing lag the parent is resolved only after every
    child's receipt exists or the child is a recorded poison pill."""
    spec = {"segcap-00000": ["d--e0", "d--e1", "d--e2"], "segcap-00001": ["d--x0"]}
    w = _world(tmp_path, store_kw, spec, behavior=Behavior(oom={"d--e1"}))
    out = w.run()
    assert out.result == "completed_with_errors"
    kids = ["segcap-ep-d--e1", "segcap-ep-d--e2"]
    bis = next(e for e in w.events if e["event"] == "bisected")
    assert bis["children"] == kids and bis["done"] == ["d--e0"]
    # C2: when the orchestrator said done, every child was resolved by files
    assert w.receipt("segcap-ep-d--e2")
    assert w.store.exists(A.poison_error_path(
        w.jf, Shard("segcap-ep-d--e1", "segcap", ("d--e1",), 0.0)))
    assert ("segcap-ep-d--e2", "d--e2") in w.processed
    assert w.spawns() == {"segcap-00000": 1, "segcap-00001": 1, **dict.fromkeys(kids, 1)}
    # the looping calls were cancelled, nothing is left running
    assert not w.spawner.live()
    w.assert_complete(excused={("d--e1", "segment"), ("d--e1", "caption")})
    assert w.status().poisoned == ("segcap-ep-d--e1",)


class LaggingReplans:
    """A ManifestSource whose ``load`` does not show replanned children for ``hide`` calls (the
    review's native-mode lag). The orchestrator must wait, never report done early (C2)."""

    def __init__(self, inner, hide: int):
        self.inner, self.hide = inner, hide

    def load(self):
        m = self.inner.load()
        if self.hide > 0 and any(s.bisected_from for s in m.shards):
            self.hide -= 1
            return Manifest(m.job_id, m.config_hash, m.episodes,
                            tuple(s for s in m.shards if not s.bisected_from), m.created_utc)
        return m

    def replan(self, parent, children):
        self.inner.replan(parent, children)


def test_lagging_children_never_resolve_a_quarantined_parent():
    w = World({"segcap-00000": ["d--e0", "d--e1", "d--e2"]}, lag=2,
              behavior=Behavior(oom={"d--e1"}))
    d = build_deps(w.store, w.settings, clock=w.clock)
    orch = w.orchestrator(manifests=LaggingReplans(d.manifests, hide=6), attempts=d.attempts,
                          progress=d.progress)
    assert w.run(orch).result == "completed_with_errors"
    assert w.receipt("segcap-ep-d--e2")
    w.assert_complete(excused={("d--e1", "segment"), ("d--e1", "caption")})


def test_resolve_treats_unknown_children_as_unsatisfied():
    parent = Shard("p", "segcap", ("a", "b"), 1.0)
    st = {"p": "quarantined"}
    sat, dead = resolve([parent], lambda s: st.get(s, "new"), {"p": ("p-ep-a", "p-ep-b")})
    assert "p" not in sat and "p" not in dead
    kids = [Shard("p-ep-a", "segcap", ("a",), 1.0, bisected_from="p"),
            Shard("p-ep-b", "segcap", ("b",), 1.0, bisected_from="p")]
    st.update({"p-ep-a": "done", "p-ep-b": "poisoned"})
    sat, dead = resolve([parent, *kids], lambda s: st.get(s, "new"), {"p": ("p-ep-a", "p-ep-b")})
    assert "p" in sat and not dead


@pytest.mark.parametrize("store_kw", _stores())
def test_python_failure_is_respawned(tmp_path, store_kw):
    w = _world(tmp_path, store_kw, _spec(2), behavior=Behavior(pyfail={"segcap-00001": 1}))
    assert w.run().result == "done"
    assert w.spawns() == {"segcap-00000": 1, "segcap-00001": 2}
    assert "call_failed" in w.names() and "quarantined" not in w.names()
    w.assert_complete()


@pytest.mark.parametrize("store_kw", _stores())
def test_exhausted_respawns_fail_without_bisection(tmp_path, store_kw):
    """H1: python failures exhaust the respawns -> shard_failed, terminal. Not bisected, no
    children ever run, reconciler agrees (failed, no revive), a restart changes nothing."""
    w = _world(tmp_path, store_kw, {"segcap-00000": ["d--a", "d--b"], "segcap-00001": ["d--c"]},
               behavior=Behavior(pyfail={"segcap-00000": 99}))
    assert w.run().result == "failed"
    assert "bisected" not in w.names() and "quarantined" not in w.names()
    assert w.spawns() == {"segcap-00000": 4, "segcap-00001": 1}  # 1 + MAX_RESPAWNS
    assert not any(s.bisected_from for s in w.deps.manifests.load().shards)
    assert A.counted_starts(w.jf, "segcap", "segcap-00000") == 0  # exceptions are not D15
    st = w.status()
    assert st.state == "failed" and st.failed == ("segcap-00000",)
    assert not needs_orchestrator(st)
    prog, _ = w.derive()
    assert prog.shards["segcap-00000"].state == "failed" and not prog.complete
    w.clock.advance(1000)
    assert w.run().result == "failed"
    assert w.spawns() == {"segcap-00000": 4, "segcap-00001": 1}


@pytest.mark.parametrize("store_kw", _stores())
def test_expired_call_is_recovered_from_its_receipt(tmp_path, store_kw):
    w = _world(tmp_path, store_kw, _spec(2))
    w.spawner.inject("segcap-00000", "expire")
    assert w.run().result == "done"
    assert w.spawns() == dict.fromkeys(_spec(2), 1)
    w.assert_complete()


@pytest.mark.parametrize("store_kw", _stores())
def test_orchestrator_restart_reattaches(tmp_path, store_kw):
    w = _world(tmp_path, store_kw, _spec(6), behavior=Behavior(preempt={"segcap-00000": 1}))
    a = w.orchestrator(holder="A")
    assert a.acquire() and a.tick() == "running"
    w.sleep(10)
    assert a.tick() == "running"
    del a  # killed: no release
    w.clock.advance(400)
    st = w.status()
    assert st.state == "running" and needs_orchestrator(st)
    assert w.run(w.orchestrator(holder="B")).result == "done"
    assert w.spawns() == dict.fromkeys(_spec(6), 1)
    assert "reattached" in w.names()
    w.assert_complete()


def test_budget_stop_then_resume():
    budget = FakeBudget(limit=2)
    w = World(_spec(5), lag=1, budget=budget)
    assert w.run().result == "budget_stopped"
    assert sum(w.spawns().values()) == 2
    w.store.tick()  # the reconciler reads the stop record through a (lagging) list
    st = w.status()
    assert st.state == "budget_stopped" and not needs_orchestrator(st)
    budget.limit = None
    w.clock.advance(5)
    write_control(w.store, JobPaths(JOB), ControlRecord("resume", JOB, "operator",
                                                        utc(w.clock())))
    w.clock.advance(5)
    w.store.tick()
    assert needs_orchestrator(w.status())
    assert w.run().result == "done"
    assert w.spawns() == dict.fromkeys(_spec(5), 1)
    w.assert_complete()


def test_cancel_stops_everything():
    w = World(_spec(3), lag=0, behavior=Behavior(oom={"d--e0-1", "d--e1-1"}))
    orch = w.orchestrator()
    assert orch.acquire() and orch.tick() == "running"
    write_control(w.store, JobPaths(JOB), ControlRecord("cancel", JOB, "operator",
                                                        utc(w.clock())))
    assert orch.tick() == "cancelled"
    assert not w.spawner.live()
    st = w.status()
    assert st.state == "cancelled" and not needs_orchestrator(st)


@pytest.mark.parametrize("receipt_parts", ["all", "own"])
@pytest.mark.parametrize("lag", [0, 2])
def test_receipt_parts_contract_loses_nothing(receipt_parts, lag):
    """H3: whether the receipt names every part (contract) or only its own container's,
    compose reads every record of the shard."""
    w = World(_spec(2, eps=4), lag=lag, behavior=Behavior(
        preempt={"segcap-00000": 2, "segcap-00001": 1}, preempt_after=1,
        receipt_parts=receipt_parts))
    assert w.run().result == "done"
    w.assert_complete()
    _, ci = w.derive()
    assert len(ci.records) == 2 * 4 * 2
    if receipt_parts == "all":
        r = P.read_receipt(w.jf, "segcap", "segcap-00000")
        assert r is not None and len(r.parts) == 4  # parts of all three containers


def test_spawn_then_crash_window_adopts_the_call():
    """M1: the orchestrator dies after spawn() but before recording the call id. The call runs
    and its marker carries the spawn token + call id, so the successor adopts it."""
    w = World(_spec(1), lag=1, behavior=Behavior(preempt={"segcap-00000": 1}))
    a = w.orchestrator(holder="A")
    assert a.acquire()
    w.spawner.crash_after_spawn = True
    with pytest.raises(SystemExit):
        a.tick()
    led = a._ledgers["segcap-00000"]  # noqa: SLF001
    assert led.status == "spawning" and led.token
    w.sleep(10)  # the orphaned call starts and writes its marker
    w.clock.advance(400)  # A's lease expires
    out = w.run(w.orchestrator(holder="B"))
    assert out.result == "done"
    assert "spawn_adopted" in w.names()
    assert w.spawns() == {"segcap-00000": 1}
    w.assert_complete()


def test_spawn_then_crash_window_duplicate_is_inert():
    """M1: the orphaned call has not started when the successor gives up waiting; it respawns
    with a new token and the late orphan exits without doing (or counting) anything."""
    w = World(_spec(1), lag=0)
    a = w.orchestrator(holder="A")
    assert a.acquire()
    w.spawner.crash_after_spawn = True
    with pytest.raises(SystemExit):
        a.tick()
    w.clock.advance(400)  # no spawner tick: the orphan is still queued
    b = w.orchestrator(holder="B")
    assert b.acquire() and b.tick() == "running"
    assert "orphan_spawn_possible" in w.names()
    assert w.run(b).result == "done"
    assert w.spawns() == {"segcap-00000": 2}
    orphan = next(iter(w.spawner.calls))
    assert w.superseded == [orphan]
    assert sorted(w.processed) == [("segcap-00000", f"d--e0-{j}") for j in range(3)]
    assert A.count_starts(w.jf, "segcap", "segcap-00000") == 1  # the orphan wrote no marker
    w.assert_complete()


def test_spawn_intent_waits_for_a_queued_call_within_the_grace_window():
    w = World(_spec(1), lag=0, behavior=Behavior(preempt={"segcap-00000": 1}))
    a = w.orchestrator(holder="A")
    assert a.acquire()
    w.spawner.crash_after_spawn = True
    with pytest.raises(SystemExit):
        a.tick()
    w.kv.data.clear()  # A's lease is gone at once
    b = w.orchestrator(holder="B")
    assert b.acquire() and b.tick() == "running"
    assert "orphan_spawn_possible" not in w.names()  # still inside spawn_grace_s
    assert w.run(b).result == "done"
    assert "spawn_adopted" in w.names() and w.spawns() == {"segcap-00000": 1}


class RacyKV(FakeDict):
    """Runs ``hook`` once, right after the first ``contains`` of ``key`` (between a holder's
    check and its write)."""

    def __init__(self):
        super().__init__()
        self.key, self.hook = None, None

    def contains(self, key) -> bool:
        r = super().contains(key)
        if key == self.key and self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return r


def test_lease_renew_race_is_detected():
    """M2: B takes over between A's ``contains(g+1)`` check and A's write: renew-then-verify
    reports the loss instead of returning True."""
    kv, clock = RacyKV(), FakeClock()
    a = DictLease(kv, "L", clock=clock)
    b = DictLease(kv, "L", clock=clock)
    assert a.acquire("A", 60)
    clock.advance(61)  # A stalled past its TTL
    kv.key, kv.hook = "L/gen/2", lambda: b.acquire("B", 60)
    assert not a.renew("A", 60)
    assert b.current().holder == "B"


class StealingBudget(FakeBudget):
    """Admits, but on the first admit another orchestrator steals the (expired) lease: the
    orchestrator is mid-tick, past its renew, about to spawn."""

    def __init__(self, steal):
        super().__init__()
        self.steal = steal

    def admit(self, shard, in_flight):
        if self.steal is not None:
            steal, self.steal = self.steal, None
            steal()
        return super().admit(shard, in_flight)


def test_orchestrator_that_loses_its_lease_mid_tick_does_not_spawn():
    w = World(_spec(3), lag=0)
    a = w.orchestrator(holder="A")
    b = w.orchestrator(holder="B")

    def steal():
        w.clock.advance(400)
        assert b.acquire()

    a.budget = StealingBudget(steal)
    assert a.acquire()
    assert a.tick() == "lost_lease"
    assert w.spawns() == {}
    assert w.run(b).result == "done"
    assert w.spawns() == dict.fromkeys(_spec(3), 1)


def test_poisoned_segment_episode_does_not_flow_into_caption():
    """M4: segment poison-pills "b"; the dependent caption shard skips it (reports it failed)
    instead of running it, and compose excuses both stages."""
    pools = {"segment": PoolSettings(gpu="L40S", max_containers=2),
             "caption": PoolSettings(gpu="H100!", max_containers=2)}
    w = World({"segment-00000": ["a", "b", "c"], "caption-00000": ["a", "b", "c"]}, lag=1,
              pools=pools, kinds={"segment-00000": "segment", "caption-00000": "caption"},
              deps={"caption-00000": ["segment-00000"]},
              behavior=Behavior(oom={"b"}, oom_kinds={"segment"}))
    assert w.run().result == "completed_with_errors"
    assert ("caption-00000", "b") not in w.processed
    assert {u for s, u in w.processed if s == "caption-00000"} == {"a", "c"}
    r = P.read_receipt(w.jf, "caption", "caption-00000")
    assert r is not None and r.episodes_failed == ("b",)
    w.assert_complete(excused={("b", "segment"), ("b", "caption")})


def test_control_records_are_scoped_per_config():
    """M6: config A's budget_stop does not stop a re-run of the same job under config B; an
    unscoped operator cancel applies to every config; timestamps are validated."""
    store, kv, clock = FakeStore(0), FakeDict(), FakeClock()
    a = World(_spec(3), store=store, kv=kv, clock=clock, budget=FakeBudget(limit=1), cfg="cfgA")
    assert a.run().result == "budget_stopped"
    recs = [ControlRecord("budget_stop", JOB, "x", utc(clock()), {"config_hash": "cfgA"})]
    assert control_view(recs, JOB, "cfgB").budget_stopped is None
    assert control_view(recs, JOB, "cfgA").budget_stopped is not None
    clock.advance(5)
    b = World(_spec(3), store=store, kv=kv, clock=clock, cfg="cfgB")  # re-planned under B
    assert b.run().result == "done"
    assert b.spawns() == dict.fromkeys(_spec(3), 1)
    with pytest.raises(ValueError):
        write_control(store, JobPaths(JOB), ControlRecord("resume", JOB, "x",
                                                          "2026-10-21T00:00:00+00:00"))
    unscoped = [ControlRecord("cancel", JOB, "op", utc(clock()))]
    assert control_view(unscoped, JOB, "cfgB").cancelled is not None


def test_crash_after_replan_before_record_reuses_children():
    """C2 write order with the real adapters: the replan file is durable before the record; a
    successor finishes the quarantine with the same children."""
    w = World({"segcap-00000": ["d--e0", "d--e1", "d--e2"]}, lag=1,
              behavior=Behavior(oom={"d--e1"}))
    orch = w.orchestrator(holder="A")
    real = orch.attempts.quarantine

    def crash(shard, reason, detail):
        raise SystemExit("killed between replan and record")

    orch.attempts.quarantine = crash  # type: ignore[method-assign]
    assert orch.acquire()
    with pytest.raises(SystemExit):
        while orch.tick() == "running":
            w.sleep(10)
    del real
    rp = A.read_replan(w.jf, w.manifest.shards[0])
    assert rp is not None and [c.shard_id for c in rp.children] == ["segcap-ep-d--e1",
                                                                     "segcap-ep-d--e2"]
    assert P.quarantine_records(w.jf) == {}
    w.clock.advance(400)
    assert w.run(w.orchestrator(holder="B")).result == "completed_with_errors"
    assert w.spawns()["segcap-00000"] == 1
    w.assert_complete(excused={("d--e1", "segment"), ("d--e1", "caption")})


class CapGate(FakeBudget):
    """A hard-cap gate: while ``trip`` is set, ``overspend`` names every live call."""

    def __init__(self):
        super().__init__()
        self.trip = False
        self.states: list = []

    def overspend(self, state):
        self.states.append(state)
        return [c.call_id for c in state.live()] if self.trip else []


def test_budget_hard_cap_cancels_without_consuming_respawns():
    gate = CapGate()
    w = World(_spec(3, eps=4), lag=0, budget=gate,
              behavior=Behavior(preempt={"segcap-00000": 2, "segcap-00001": 2}))
    orch = w.orchestrator(fresh=False)
    assert orch.acquire() and orch.tick() == "running"
    live = {c.call_id for c in orch.budget_state().live()}
    assert len(live) == 2
    w.sleep(10)
    gate.trip = True
    assert orch.tick() == "budget_stopped"
    assert {c.call_id for c in gate.states[-1].live()} == live
    assert all(w.spawner.calls[c].state == "cancelled" for c in live)
    for sid in ("segcap-00000", "segcap-00001"):
        led = orch._ledgers[sid]  # noqa: SLF001
        assert led.status == "retry" and led.respawns == 0
        assert led.calls[-1]["outcome"] == "cancelled_budget"
    assert "call_cancelled" not in w.names() and "budget_overspend_cancel" in w.names()
    ended = [c for c in orch.budget_state().calls if c.call_id in live]
    assert all(c.outcome == "cancelled_budget" and not c.live for c in ended)
    orch.release()

    gate.trip = False
    w.clock.advance(5)
    write_control(w.store, JobPaths(JOB), ControlRecord("resume", JOB, "operator",
                                                        utc(w.clock())))
    w.clock.advance(5)
    assert w.run().result == "done"
    w.assert_complete()


def test_call_cancelled_by_a_driver_during_a_budget_stop_is_requeued():
    """A budget cancel made outside ``enforce_budget`` (e.g. by a driver) is recognized from the
    active budget_stop and does not consume a respawn either."""
    w = World(_spec(1, eps=4), lag=0, behavior=Behavior(preempt={"segcap-00000": 3}))
    orch = w.orchestrator()
    assert orch.acquire() and orch.tick() == "running"
    cid = orch._ledgers["segcap-00000"].call_id  # noqa: SLF001
    orch._control("budget_stop", "driver hard cap")  # noqa: SLF001
    w.spawner.cancel(cid)
    assert orch.tick() == "budget_stopped"
    led = orch._ledgers["segcap-00000"]  # noqa: SLF001
    assert led.status == "retry" and led.respawns == 0 and "call_cancelled" not in w.names()
