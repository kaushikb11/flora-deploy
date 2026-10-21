"""Local dry run of the canary's plumbing (no Modal): scenarios, manifest, settings, the
budget gate wiring, ``summarize`` and ``evaluate``, with the real orchestrator + adapters driving
``ContractWorker`` through the fake spawner."""

from __future__ import annotations

import pytest

from flora_deploy.jobs.deps import build_deps
from flora_deploy.jobs.manifest import save_manifest
from flora_deploy.jobs.orchestrator import DictLease, Orchestrator
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.modal_app import canary_app as C
from flora_deploy.modal_app.orchestrator_app import TickDriver
from tests.conformance.store_suite import DictStore
from tests.fakes.jobs import FakeClock, FakeDict, TokenSpawner
from tests.fakes.modal import Preempted


class _Exit137(BaseException):
    pass


class TickingLocalStore(LocalStore):
    def tick(self) -> None:
        pass


class _Live(DictStore):
    """What a worker container sees: the Volume's committed state, read-after-write."""

    def __init__(self, data: dict[str, bytes]):
        super().__init__()
        self.data = data

    def tick(self) -> None:
        pass


class _StaleView(DictStore):
    """The orchestrator container's Volume mount: its own writes at once, other containers'
    commits only via ``refresh()`` (``VolumeStore.refresh``), and each refresh shows the state
    as of the *previous* refresh, so a worker that commits between the orchestrator's reload
    and its ``poll`` is a success whose receipt this container cannot see yet."""

    def __init__(self, server: dict[str, bytes]):
        super().__init__()
        self.server, self.own, self.snap = server, {}, {}
        self.data = {}

    def put_bytes(self, path: str, data: bytes) -> None:
        super().put_bytes(path, data)
        self.own[path] = self.server[path] = bytes(data)

    def refresh(self) -> bool:
        self.data = {**self.snap, **self.own}
        self.snap = dict(self.server)
        return True


def _run(sc: C.Scenario, tmp_path, *, stale_view: bool = False):
    job = f"canary-local-{sc.name}"
    if stale_view:
        server: dict[str, bytes] = {}
        worker_store, store = _Live(server), _StaleView(server)
    else:
        worker_store = store = TickingLocalStore(tmp_path)
    save_manifest(store, C.build_manifest(job, sc))
    settings = C.settings_for(job, sc)
    clock = FakeClock()
    deps = build_deps(store, settings, clock=clock)

    def worker(job_id, shard_id, cfg, container, st, *, call_id, token):
        def before(shard, uid):
            if uid in sc.behavior.poison:
                raise _Exit137  # like os._exit: no python handler runs

        try:
            return C.ContractWorker(st, container, call_id, clock=clock, before_episode=before,
                                    flush_every=sc.behavior.flush_every).run_shard(
                job_id, shard_id, cfg, token)
        except _Exit137:
            raise Preempted("container died") from None

    spawner = TokenSpawner(worker_store, {C.KIND: worker}, pools=settings.pools,
                           python_retries=0, max_restarts=50)
    events: list[dict] = []
    orch = Orchestrator(settings=settings, store=store, spawner=spawner,
                        lease=DictLease(FakeDict(), f"orchestrator/{job}", clock=clock),
                        manifests=deps.manifests, attempts=deps.attempts, progress=deps.progress,
                        budget=deps.budget, clock=clock, events=events.append)

    def sleep(dt):
        clock.advance(dt)
        spawner.tick()

    out = TickDriver(orch).run_until_idle(sleep=sleep, poll_interval_s=3.0, max_ticks=200)
    run = {"outcome": {"result": out.result}, "events": {"reattached": 1},
           "crashed_holder": "x", "poll_after_terminate": {}, "event_log": events}
    if stale_view:
        store.refresh()
        store.refresh()
    return run, C.summarize(store, job, sc)


def test_scenario_round_trip():
    for sc in C.scenarios().values():
        assert C.Scenario.from_dict(sc.to_dict()) == sc
    assert 6 <= len(C.scenarios()["happy"].shards) <= 10


@pytest.mark.parametrize("name", ["happy", "poison", "restart"])
def test_local_run_passes_evaluate(name, tmp_path):
    sc = C.scenarios()[name]
    run, summary = _run(sc, tmp_path)
    assert C.evaluate(sc, run, summary) == [], summary


def test_evaluate_flags_duplicates_and_missing(tmp_path):
    sc = C.scenarios()["happy"]
    run, summary = _run(sc, tmp_path)
    bad = dict(summary, raw_duplicates=[["u", "segment", 2]], missing=[["u", "caption"]])
    problems = C.evaluate(sc, run, bad)
    assert any("duplicate" in p for p in problems) and any("missing" in p for p in problems)


def test_budget_settings_are_tiny_and_cpu_priced():
    sc = C.scenarios()["budget"]
    s = C.settings_for("j", sc)
    rate = C.canary_rates().pool_per_second(s.pools[C.KIND])
    assert 0 < rate < 1e-5 and s.budget_usd == pytest.approx(rate * sc.budget_container_s)


def test_success_whose_receipt_this_container_cannot_see_yet_is_not_a_failure():
    """Canary finding (happy path on real Modal, Volume v2): a worker commits its receipt after
    the orchestrator's per-tick reload, the orchestrator then polls the call -> ``success``, but
    ``exists(receipt)`` is still False in its (stale) Volume view. The orchestrator must not
    treat that as "call succeeded without a receipt" (a failure that burns a respawn and spawns
    a duplicate call): the call returned the receipt, written and committed LAST."""
    sc = C.scenarios()["happy"]
    run, summary = _run(sc, None, stale_view=True)
    failed = [e for e in run["event_log"] if e["event"] == "call_failed"]
    assert not failed, failed
    assert C.evaluate(sc, run, summary) == [], summary
