import pytest

from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.types import ShardReceipt
from tests.fakes.modal import FakeSpawner, FakeStore


def _receipt(shard_id, container):
    return ShardReceipt(shard_id=shard_id, kind="segcap", config_hash="c", container_id=container,
                        episodes_done=("u1",), episodes_failed=(), parts=(), gpu=None,
                        wall_seconds=1.0, finished_utc="t")


def worker(job_id, shard_id, cfg, container, store):
    store.put_bytes(f"att/{shard_id}/{container}", b"1")
    return _receipt(shard_id, container)


def test_store_listing_lags_but_reads_are_immediate():
    s = FakeStore(lag=2)
    s.put_bytes("a/x", b"1")
    assert s.exists("a/x") and list(s.list("a/")) == []
    s.tick()
    s.tick()
    assert list(s.list("a/")) == ["a/x"]
    with pytest.raises(TypeError):
        s.put_bytes("a/y", "text")  # type: ignore[arg-type]


def test_preemption_restarts_same_call_in_new_container():
    store = FakeStore()
    sp = FakeSpawner(store, {"segcap": worker}, eager=True)
    sp.inject("s1", "preempt", times=2)
    cid = sp.spawn("segcap", "job", "s1", "c")
    st = sp.poll(cid)
    assert st.state == "success" and len(sp.calls[cid].containers) == 3


def test_poison_pill_stays_pending_forever():
    sp = FakeSpawner(FakeStore(), {"segcap": worker}, eager=True, max_restarts=5)
    sp.inject("s1", "oom", times=100)
    cid = sp.spawn("segcap", "job", "s1", "c")
    assert sp.poll(cid).state == "pending" and sp.calls[cid].starts == 5


def test_python_exceptions_exhaust_retries_then_fail():
    def bad(*a):
        raise RuntimeError("boom")

    sp = FakeSpawner(FakeStore(), {"segcap": bad}, eager=True, python_retries=2)
    cid = sp.spawn("segcap", "job", "s1", "c")
    st = sp.poll(cid)
    assert st.state == "failed" and "boom" in st.error


def test_run_settings_enforce_gpu_cap():
    pools = {"segcap": PoolSettings(gpu="H200", max_containers=6),
             "reward.terminal": PoolSettings(gpu="H100!", max_containers=5)}
    with pytest.raises(ValueError, match="max_gpus"):
        RunSettings(job_id="j", preset="exact", config_hash="c", budget_usd=10, max_gpus=10,
                    pools=pools)
    RunSettings(job_id="j", preset="exact", config_hash="c", budget_usd=10, max_gpus=11,
                pools=pools)
