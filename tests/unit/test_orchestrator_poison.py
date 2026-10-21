"""D15 poison pills: OOM loop -> attempt markers -> cancel -> quarantine -> bisect -> the single bad
episode is isolated to errors/ and every other episode still gets labeled."""

from __future__ import annotations

import json

from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import EpisodeError
from tests.fakes.jobs import CFG, JOB, Harness, WorkerBehavior

EPS = [f"ds--e{j}" for j in range(5)]


def test_oom_episode_is_isolated_by_bisection():
    h = Harness({"segcap-000": EPS, "segcap-001": ["ds--x0", "ds--x1"]}, lag=1,
                behavior=WorkerBehavior(oom={"ds--e2"}))
    out = h.run()
    assert out.result == "done"  # a poisoned episode is an episode error, not a job failure

    # 1. the looping call was cancelled and the parent quarantined + bisected
    first_call = next(c for c in h.spawner.calls.values() if c.shard_id == "segcap-000")
    assert first_call.state == "cancelled" and first_call.starts >= 3
    assert h.attempts.quarantined()["segcap-000"].reason == "poison_pill"
    bis = next(e for e in h.events if e["event"] == "bisected")
    # episodes processed before the crash point are not re-planned
    assert bis["done"] == ["ds--e0", "ds--e1"]
    assert bis["children"] == ["segcap-ep-ds--e2", "segcap-ep-ds--e3", "segcap-ep-ds--e4"]

    # 2. the bad episode's own shard also looped, and was isolated (not bisected again)
    assert h.attempts.poison_records == ["ds--e2"]
    err_path = JobPaths(JOB).errors(CFG, "segcap", "segcap-ep-ds--e2", "orchestrator", 0)
    err = EpisodeError(**json.loads(h.store.get_bytes(err_path)))
    assert err.uid == "ds--e2" and err.kind == "poison_pill"
    bad_calls = [c for c in h.spawner.calls.values() if c.shard_id == "segcap-ep-ds--e2"]
    assert len(bad_calls) == 1 and bad_calls[0].state == "cancelled"

    # 3. every good episode is labeled exactly once; bisection children got receipts
    assert h.done_uids() == set(EPS + ["ds--x0", "ds--x1"]) - {"ds--e2"}
    assert len(h.processed) == len(h.done_uids())
    assert h.receipt("segcap-ep-ds--e3") and h.receipt("segcap-ep-ds--e4")
    assert h.spawn_counts()["segcap-000"] == 1  # never respawned, only bisected


def test_poisoned_dependency_does_not_block_dependents():
    from flora_deploy.jobs.settings import PoolSettings

    pools = {"segment": PoolSettings(gpu="L40S", max_containers=2),
             "caption": PoolSettings(gpu="H100!", max_containers=2)}
    spec = {"segment-0": ["a", "b"], "caption-0": ["a", "b"]}
    h = Harness(spec, pools=pools, kinds={"segment-0": "segment", "caption-0": "caption"},
                deps={"caption-0": ["segment-0"]}, behavior=WorkerBehavior(oom={"b"}))
    # the caption worker would OOM on "b" too; make only the segment stage poisonous
    h.spawner.workers["caption"] = _skip_oom(h)
    assert h.run().result == "done"
    assert "caption-0" in h.spawn_counts()
    spawned = {e["shard_id"]: e["t"] for e in h.events if e["event"] == "spawned"}
    poisoned = next(e["t"] for e in h.events if e["event"] == "poisoned")
    assert spawned["caption-0"] >= poisoned  # waited until segment-0's children all resolved


def _skip_oom(h):
    from tests.fakes.jobs import make_worker

    return make_worker(h.manifests, WorkerBehavior(), "caption", h.processed)


def test_crash_between_quarantine_and_bisect_is_redone_on_restart():
    h = Harness({"segcap-000": EPS}, lag=1, behavior=WorkerBehavior(oom={"ds--e2"}))
    orch = h.orchestrator()
    assert orch.acquire()

    real_bisect = h.attempts.bisect
    calls = {"n": 0}

    def crashing_bisect(shard, done):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SystemExit("orchestrator killed mid-quarantine")
        return real_bisect(shard, done)

    h.attempts.bisect = crashing_bisect  # type: ignore[method-assign]
    try:
        while orch.tick() == "running":
            h.sleep(10)
    except SystemExit:
        pass
    assert "segcap-000" in h.attempts.quarantined()

    h.clock.advance(1000)  # the dead orchestrator's lease expires
    out = h.run()
    assert out.result == "done"
    assert h.done_uids() == set(EPS) - {"ds--e2"}
    assert h.attempts.poison_records == ["ds--e2"]
    assert h.spawn_counts()["segcap-000"] == 1
