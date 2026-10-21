"""Orchestrator scheduling over FakeStore(lag>0) + FakeSpawner: happy path, chaining, caps,
preemptions, python failures, expired results."""

from __future__ import annotations

from flora_deploy.jobs.settings import PoolSettings
from tests.fakes.jobs import Harness, WorkerBehavior


def _spec(n_shards: int, eps: int = 3) -> dict[str, list[str]]:
    return {f"segcap-{i:03d}": [f"ds--e{i}-{j}" for j in range(eps)] for i in range(n_shards)}


def test_happy_path_spawns_each_shard_once_and_finishes():
    h = Harness(_spec(5), lag=2)
    out = h.run()
    assert out.result == "done"
    assert h.spawn_counts() == {sid: 1 for sid in _spec(5)}
    assert len(h.processed) == 15 and len(h.done_uids()) == 15
    assert all(h.receipt(sid) for sid in _spec(5))
    assert h.spawner.peak_by_kind["segcap"] <= 2  # max_containers(2) x max_inputs(1)
    assert "lease_acquired" in h.names() and h.names()[-2:] == ["done", "lease_released"]


def _tap(fn, order):
    def spawn(kind, job_id, shard_id, cfg):
        order.append(shard_id)
        return fn(kind, job_id, shard_id, cfg)
    return spawn


def test_chaining_waits_for_dependency_receipts():
    pools = {"segment": PoolSettings(gpu="L40S", max_containers=2),
             "caption": PoolSettings(gpu="H100!", max_containers=2)}
    spec = {"segment-0": ["a", "b"], "segment-1": ["c"], "caption-0": ["a", "b"],
            "caption-1": ["c"]}
    kinds = {k: k.split("-")[0] for k in spec}
    deps = {"caption-0": ["segment-0"], "caption-1": ["segment-1"]}
    h = Harness(spec, pools=pools, kinds=kinds, deps=deps, lag=1)
    assert h.run().result == "done"
    spawned_at = {e["shard_id"]: e["t"] for e in h.events if e["event"] == "spawned"}
    done_at = {e["shard_id"]: e["t"] for e in h.events if e["event"] == "shard_done"}
    assert spawned_at["caption-0"] >= done_at["segment-0"] > spawned_at["segment-0"]
    assert spawned_at["caption-1"] >= done_at["segment-1"]
    assert h.spawn_counts() == dict.fromkeys(spec, 1)


def test_gpu_cap_never_exceeded_with_concurrent_inputs():
    # segcap: 2 GPUs/container, 2 containers, 2 inputs each -> 4 calls on 4 GPUs.
    # reward: 1 GPU, 3 containers, 1 input -> 3 calls on 3 GPUs. max_gpus = 7.
    pools = {"segcap": PoolSettings(gpu="H200:2", gpu_count=2, max_containers=2, max_inputs=2),
             "reward.terminal": PoolSettings(gpu="H100!", max_containers=3)}
    spec = {**_spec(9, eps=1), **{f"reward-{i}": [f"r{i}"] for i in range(9)}}
    kinds = {k: ("reward.terminal" if k.startswith("reward") else "segcap") for k in spec}
    h = Harness(spec, pools=pools, kinds=kinds, max_gpus=7, lag=1)
    assert h.run().result == "done"
    assert h.spawner.peak_gpus == 7  # saturated, never above
    assert h.spawner.peak_by_kind == {"segcap": 4, "reward.terminal": 3}
    assert max(e["gpus"] for e in h.events if e["event"] == "spawned") <= 7
    assert sum(h.spawn_counts().values()) == 18


def test_gpu_cap_binds_below_pool_limits():
    # pools could use 4 GPUs, but the orchestrator is told max_gpus=4 and one pool takes 2/call
    pools = {"segcap": PoolSettings(gpu="H200:2", gpu_count=2, max_containers=2)}
    h = Harness(_spec(6, eps=1), pools=pools, max_gpus=4, lag=1)
    assert h.run().result == "done"
    assert h.spawner.peak_gpus <= 4 and h.spawner.peak_by_kind["segcap"] <= 2


def test_preemptions_are_invisible_and_cause_no_respawn():
    h = Harness(_spec(3), lag=1)
    for sid in _spec(3):
        h.spawner.inject(sid, "preempt", times=4)  # Modal restarts the same call
    h.behavior.crash_after_marker["segcap-001"] = 1  # one preemption that left a marker
    assert h.run().result == "done"
    assert h.spawn_counts() == dict.fromkeys(_spec(3), 1)
    assert len(h.processed) == 9  # nothing processed twice
    assert "quarantined" not in h.names()


def test_python_failures_respawn_with_bound_then_fail_the_shard():
    h = Harness(_spec(3), lag=1,
                behavior=WorkerBehavior(init_fail={"segcap-000": 2, "segcap-001": 99}))
    out = h.run(h.orchestrator(max_respawns=3))
    counts = h.spawn_counts()
    assert counts["segcap-000"] == 3  # 2 failed calls, then success
    assert counts["segcap-001"] == 4  # 1 + max_respawns, then given up
    assert counts["segcap-002"] == 1
    assert out.result == "failed"
    assert h.receipt("segcap-000") and not h.receipt("segcap-001")
    failed = [e for e in h.events if e["event"] == "shard_failed"]
    assert [e["shard_id"] for e in failed] == ["segcap-001"]
    # H1: a terminal shard_failed record, not a quarantine (never bisected)
    assert h.attempts.failed()["segcap-001"].reason == "respawns_exhausted"
    assert "segcap-001" not in h.attempts.quarantined()
    assert "bisected" not in [e["event"] for e in h.events]


def test_failed_shard_blocks_its_dependents_and_job_ends_failed():
    pools = {"segment": PoolSettings(gpu="L40S"), "caption": PoolSettings(gpu="H100!")}
    spec = {"segment-0": ["a"], "caption-0": ["a"]}
    h = Harness(spec, pools=pools, kinds={"segment-0": "segment", "caption-0": "caption"},
                deps={"caption-0": ["segment-0"]},
                behavior=WorkerBehavior(init_fail={"segment-0": 99}))
    assert h.run(h.orchestrator(max_respawns=1)).result == "failed"
    assert "caption-0" not in h.spawn_counts()


def test_expired_result_with_receipt_counts_as_done():
    h = Harness(_spec(2), lag=1)
    h.spawner.inject("segcap-000", "expire")
    assert h.run().result == "done"
    assert h.spawn_counts() == {"segcap-000": 1, "segcap-001": 1}


def test_expired_result_without_receipt_is_respawned():
    h = Harness(_spec(1), lag=1)
    h.spawner.inject("segcap-000", "expire")
    orch = h.orchestrator()
    assert orch.acquire()
    assert orch.tick() == "running"
    h.sleep(10)
    # the worker's receipt write was lost: only its parts survived
    h.store._data.pop(next(p for p in h.store._data if "/receipts/" in p))
    h.sleep(10)
    while (r := orch.tick()) == "running":
        h.sleep(10)
    assert r == "done" and h.spawn_counts() == {"segcap-000": 2}
    assert "call_expired" in h.names()
    assert len(h.processed) == 3  # the second call re-processed nothing
