"""Review 8a R4: a deterministic CUDA fault in the reward runner ends isolated (marker
``crashed=True`` -> D15 quarantine, bisection, poison pill), not in an endless crash loop. The
shared path is ``workers_impl.reward.end_failed_shard``."""

from __future__ import annotations

import shutil

from flora_deploy.workers_impl.reward import ShardInfraError, end_failed_shard

TEMPLATE = __import__("pathlib").Path(__file__).resolve().parents[2] / \
    "tests/fixtures/local_demo_exact.yaml"
CUDA = "CUDA error: an illegal memory access was encountered"


def test_end_failed_shard_marks_a_crash_before_the_container_exit():
    order = []

    def end(**kw):
        order.append(("end", kw))
        return True

    kind = end_failed_shard(RuntimeError(CUDA), end, lambda exc: order.append(("fatal", exc)))
    assert kind == "cuda_fatal"
    assert [o[0] for o in order] == ["end", "fatal"]
    assert order[0][1]["crashed"] is True and order[0][1]["error"].startswith("cuda_fatal: ")
    # an episode's CUDA fault inside a ShardInfraError is a crash too
    order.clear()
    end_failed_shard(ShardInfraError("s", {"u": f"RuntimeError: {CUDA}"}), end, order.append)
    assert order[0][1]["crashed"] is True and len(order) == 2
    # a python error: marked, not D15-counted, the container lives on
    order.clear()
    assert end_failed_shard(ValueError("bug"), end, lambda e: order.append(("fatal", e))) is None
    assert order == [("end", {"error": "ValueError: bug", "crashed": False})]
    # a crash the runner already saw (segcap's per-episode OOM), and a custom predicate
    order.clear()
    end_failed_shard(ValueError("x"), end, None, crash="oom", crash_message="boom")
    assert order == [("end", {"error": "oom: boom", "crashed": True})]
    order.clear()
    end_failed_shard(RuntimeError("EngineDead"), end, None, is_crash=lambda m: "EngineDead" in m)
    assert order[0][1]["crashed"] is True


def test_reward_cuda_fault_ends_isolated(tmp_path, monkeypatch):
    """p5r: reward.terminal faults on every generate. It used to restart 50x, D15 0/50."""
    from flora_deploy.jobs import attempts as A
    from flora_deploy.local import LocalRunner, sample_dataset
    from flora_deploy.local import engines as LE

    runs = {"n": 0}
    orig = LE.FakeLLM.generate

    def generate(self, requests, params, use_tqdm=True):
        if any("'kind': 'terminal'" in r["prompt"] for r in requests):
            runs["n"] += 1
            raise RuntimeError(CUDA)
        return orig(self, requests, params, use_tqdm)

    monkeypatch.setattr(LE.FakeLLM, "generate", generate)
    shutil.copy(TEMPLATE, tmp_path / "flora.yaml")
    sample_dataset(tmp_path / "videos")
    lr = LocalRunner(tmp_path / "flora.yaml", store=tmp_path / "store")
    try:
        lr.index()
        lr.plan()
        outcome = lr.orchestrate(max_ticks=400)
        events: dict[str, int] = {}
        for e in lr.events:
            events[e["event"]] = events.get(e["event"], 0) + 1
        jf = A.JobFiles(lr.store, lr.job_id, lr.settings.config_hash)
        terminal = [s for s in lr.manifest.shards if s.kind == "reward.terminal"]
        counted = sum(A.counted_starts(jf, s.kind, s.shard_id) for s in terminal)
        stats = lr.spawner.stats()
    finally:
        lr.close()
    assert outcome != "max_ticks", (outcome, runs, stats, events)
    assert events.get("quarantined", 0) >= 1 and events.get("poisoned", 0) >= 1, events
    assert counted >= 1
    assert stats["crashes"] < 50, stats
