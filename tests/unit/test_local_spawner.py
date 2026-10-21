"""``local.spawner.LocalSpawner`` (Modal call semantics in-process), ``local.faults`` parsing and
checkpoints, and ``FileKV`` under the real ``DictLease``."""

from __future__ import annotations

import threading
import time

import pytest

from flora_deploy.jobs.orchestrator import DictLease
from flora_deploy.local.faults import (
    FOREVER,
    ContainerCrash,
    FaultPlan,
    InjectedFailure,
    checkpoint,
    maybe_fail,
    parse_faults,
)
from flora_deploy.local.spawner import FileKV, LocalSpawner, LocalSpawnError


def wait_state(sp: LocalSpawner, cid: str, want: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = sp.poll(cid).state
        if st == want:
            return st
        time.sleep(0.005)
    return sp.poll(cid).state


class Script:
    """A runner whose ``run_shard`` follows a per-container script of outcomes."""

    def __init__(self, outcomes, log):
        self.outcomes, self.log = outcomes, log

    def run_shard(self, job_id, shard_id, config_hash, token=""):
        what = self.outcomes.pop(0) if self.outcomes else "ok"
        self.log.append((shard_id, what, token))
        if what == "preempt":
            raise KeyboardInterrupt
        if what == "crash":
            raise ContainerCrash()
        if what == "error":
            raise RuntimeError("boom")
        return f"receipt-{shard_id}"


def make(outcomes, **kw):
    log, built = [], []
    shared = list(outcomes)

    def runners(kind, container_id, call_id):
        built.append(container_id)
        return Script(shared, log)

    return LocalSpawner(runners, **kw), log, built


def test_success_and_token_reach_the_runner():
    sp, log, built = make(["ok"])
    cid = sp.spawn("segcap", "j", "s1", "cfg", token="tok-1")
    assert wait_state(sp, cid, "success") == "success"
    assert sp.poll(cid).receipt == "receipt-s1"
    assert log == [("s1", "ok", "tok-1")] and len(built) == 1
    sp.close()


def test_preempt_and_crash_restart_the_same_call_in_new_containers():
    sp, log, built = make(["preempt", "crash", "ok"])
    cid = sp.spawn("segcap", "j", "s1", "cfg")
    assert wait_state(sp, cid, "success") == "success"
    call = sp.calls[cid]
    assert (call.preemptions, call.crashes) == (1, 1)
    assert len(set(call.containers)) == 3 == len(built)  # dead containers are never reused
    sp.close()


def test_python_errors_count_against_retries_then_fail():
    sp, log, built = make(["error"] * 3, python_retries=2)
    cid = sp.spawn("segcap", "j", "s1", "cfg")
    st = wait_state(sp, cid, "failed")
    assert st == "failed" and "boom" in sp.poll(cid).error
    assert len(log) == 3 and len(built) == 1  # the container survived each failed input
    sp.close()


def test_warm_container_is_reused_by_the_next_call():
    sp, log, built = make(["ok", "ok"])
    a = sp.spawn("segcap", "j", "s1", "cfg")
    wait_state(sp, a, "success")
    b = sp.spawn("segcap", "j", "s2", "cfg")
    wait_state(sp, b, "success")
    assert sp.calls[a].containers == sp.calls[b].containers and len(built) == 1
    sp.close()


def test_crash_loop_stays_pending_until_cancelled():
    sp, log, built = make(["crash"] * 100, max_restarts=5, restart_delay_s=0.0)
    cid = sp.spawn("segcap", "j", "s1", "cfg")
    sp.calls[cid].finished.wait(5)
    assert sp.poll(cid).state == "pending" and sp.calls[cid].crashes == 5
    sp.cancel(cid)
    assert sp.poll(cid).state == "cancelled"
    sp.close()


def test_cancel_interrupts_a_running_call_at_its_checkpoint():
    gate = threading.Event()

    class Slow:
        def run_shard(self, job_id, shard_id, config_hash, token=""):
            gate.wait(5)
            checkpoint("u")
            return "never"

    sp = LocalSpawner(lambda k, c, f: Slow())
    cid = sp.spawn("segcap", "j", "s1", "cfg")
    sp.cancel(cid)
    assert sp.poll(cid).state == "cancelled"
    gate.set()
    assert sp.calls[cid].finished.wait(5)
    assert sp.poll(cid).state == "cancelled" and sp.poll(cid).receipt is None
    sp.close()


def test_unknown_call_ids_poll_expired_and_infra_spawnerr_faults():
    sp, log, built = make(["ok", "ok"], faults=FaultPlan.parse("infra:s1,spawnerr:s2"))
    assert sp.poll("lfc-from-another-process").state == "expired"
    cid = sp.spawn("segcap", "j", "s1", "cfg")
    assert sp.poll(cid).state == "expired" and not log  # never ran
    with pytest.raises(LocalSpawnError):
        sp.spawn("segcap", "j", "s2", "cfg")
    cid2 = sp.spawn("segcap", "j", "s2", "cfg")  # the next try works
    assert wait_state(sp, cid2, "success") == "success"
    assert sp.faults.fired("infra") == 1 and sp.faults.fired("spawnerr") == 1
    sp.close()


def test_parse_faults():
    fs = parse_faults(["preempt:segcap-*,oom:ep003", "fail:reward.terminal-00000x3",
                       "nverr:ep001@reward.dense"])
    assert [(f.kind, f.target, f.times) for f in fs] == [
        ("preempt", "segcap-*", 1), ("oom", "ep003", FOREVER),
        ("fail", "reward.terminal-00000", 3), ("nverr", "ep001@reward.dense", FOREVER)]
    assert fs[0].matches_shard("segcap-00001") and not fs[0].matches_shard("reward.dense-0")
    assert fs[1].matches_episode("videos--ep003", "segcap")
    assert fs[3].matches_episode("x--ep001", "reward.dense")
    assert not fs[3].matches_episode("x--ep001", "segcap")
    for bad in ("nope:x", "preempt", "preempt:a@segcap", "oom:ep1x0"):
        with pytest.raises(ValueError):
            parse_faults(bad)


def test_checkpoints_are_inert_outside_a_container():
    checkpoint("u")
    maybe_fail("part")


def test_injected_failure_is_a_plain_exception():
    assert issubclass(InjectedFailure, Exception)
    assert not issubclass(ContainerCrash, Exception)  # passes every except Exception


def test_filekv_backs_the_real_dict_lease(tmp_path):
    t = [1000.0]
    kv = FileKV(tmp_path / "kv")
    a = DictLease(kv, "orch/job", clock=lambda: t[0])
    b = DictLease(FileKV(tmp_path / "kv"), "orch/job", clock=lambda: t[0])  # another process
    assert a.acquire("A", 10.0)
    assert not b.acquire("B", 10.0)
    assert b.current().holder == "A"
    t[0] += 11.0  # A died; its lease expires
    assert b.acquire("B", 10.0)
    assert not a.renew("A", 10.0)  # A lost it for good
    assert kv.put("k", 1, skip_if_exists=True) and not kv.put("k", 2, skip_if_exists=True)
    assert kv.get("k") == 1 and kv.contains("k") and kv.get("missing", 7) == 7
