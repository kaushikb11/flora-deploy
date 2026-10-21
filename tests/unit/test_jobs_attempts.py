from __future__ import annotations

import pytest

from flora_deploy.jobs.attempts import (
    JobFiles,
    bisect,
    child_shard_id,
    count_starts,
    quarantine,
    read_control,
    should_quarantine,
    write_attempt,
)
from flora_deploy.jobs.manifest import from_json
from flora_deploy.jobs.progress import read_errors, write_receipt
from flora_deploy.jobs.types import AttemptMarker, Shard, ShardReceipt
from tests.fakes.modal import FakeStore


def _jf(lag=0):
    return JobFiles(FakeStore(lag=lag), "job", "cfg")


SHARD = Shard("segcap-00000", "segcap", ("d--e1", "d--e2", "d--e3"), 30.0)


def test_markers_one_per_container_and_lagging_count_is_lower_bound():
    jf = _jf(lag=2)
    for i in range(3):
        write_attempt(jf, "segcap", SHARD.shard_id, f"ta-{i}", started_utc="t",
                      orchestrator_attempt=1)
    write_attempt(jf, "segcap", SHARD.shard_id, "ta-0", started_utc="t2", orchestrator_attempt=2)
    assert count_starts(jf, "segcap", SHARD.shard_id) == 0  # not listed yet
    assert count_starts(jf, "segcap", SHARD.shard_id, known=["ta-9"]) == 1
    assert not should_quarantine(jf, SHARD, limit=3)
    jf.store.tick()
    jf.store.tick()
    assert count_starts(jf, "segcap", SHARD.shard_id) == 3  # same container id counted once
    assert should_quarantine(jf, SHARD, limit=3)
    p = jf.paths.attempt("cfg", "segcap", SHARD.shard_id, "ta-0")
    assert from_json(AttemptMarker, jf.store.get_bytes(p)).orchestrator_attempt == 2


def test_receipt_prevents_quarantine_even_when_listing_lags():
    jf = _jf(lag=0)
    for i in range(5):
        write_attempt(jf, "segcap", SHARD.shard_id, f"ta-{i}", started_utc="t",
                      orchestrator_attempt=1)
    jf.store.lag = 10  # receipt invisible to list, but exists() is read-after-write
    write_receipt(jf, ShardReceipt(SHARD.shard_id, "segcap", "cfg", "ta-4", SHARD.episodes, (),
                                   (), None, 1.0, "t"))
    assert not should_quarantine(jf, SHARD, limit=3)
    with pytest.raises(ValueError):
        should_quarantine(jf, SHARD, limit=0)


def test_bisect_unfinished_episodes():
    kids = bisect(SHARD, done=["d--e2"], durations={"d--e1": 10, "d--e2": 10, "d--e3": 40})
    assert [k.shard_id for k in kids] == ["segcap-ep-d--e1", "segcap-ep-d--e3"]
    assert [k.est_gpu_seconds for k in kids] == [5.0, 20.0]
    assert all(k.bisected_from == SHARD.shard_id and len(k.episodes) == 1 for k in kids)
    assert bisect(kids[0]) == ()
    long_uid = "d--" + "e" * 195
    assert len(child_shard_id("reward.terminal", long_uid)) <= 200


def test_quarantine_bisects_then_poison_pills_idempotently():
    jf = _jf()
    q = quarantine(jf, SHARD, created_utc="2026-10-21T00:00:00Z", done=["d--e1"],
                   done_parts=["p1"], starts=3)
    assert [c.episodes for c in q.children] == [("d--e2",), ("d--e3",)] and not q.poisoned
    rec = read_control(jf.store, q.control_path)
    assert rec.kind == "quarantine" and rec.detail["done_parts"] == ["p1"]
    child = q.children[0]
    q1 = quarantine(jf, child, created_utc="2026-10-21T00:01:00Z", starts=3, diagnostics={"x": 1})
    q2 = quarantine(jf, child, created_utc="2026-10-21T00:02:00Z", starts=4)
    assert q1.poisoned == q2.poisoned == ("d--e2",) and q1.error_path == q2.error_path
    errs = read_errors(jf, q2.error_path)
    assert [(e.uid, e.stage, e.kind) for e in errs] == [("d--e2", "segment", "poison_pill"),
                                                        ("d--e2", "caption", "poison_pill")]
    # a single-episode shard whose episode is already done is not poisoned
    q3 = quarantine(jf, child, created_utc="t3", done=["d--e2"])
    assert q3.poisoned == () and q3.error_path is None


def test_preempted_and_exception_starts_do_not_count():
    from flora_deploy.jobs.attempts import counted_starts, mark_exception, mark_preempted

    jf = _jf()
    for i in range(4):
        m = write_attempt(jf, "segcap", SHARD.shard_id, f"ta-{i}", started_utc="t",
                          orchestrator_attempt=1)
        if i < 2:
            mark_preempted(jf, "segcap", SHARD.shard_id, f"ta-{i}", ended_utc="t2", marker=m)
    mark_exception(jf, "segcap", SHARD.shard_id, "ta-2", ended_utc="t3", error="boom")
    assert count_starts(jf, "segcap", SHARD.shard_id) == 4
    assert counted_starts(jf, "segcap", SHARD.shard_id) == 1
    assert counted_starts(jf, "segcap", SHARD.shard_id, limit=5) == 4  # below limit: no reads
    assert not should_quarantine(jf, SHARD, limit=2)
    p = jf.paths.attempt("cfg", "segcap", SHARD.shard_id, "ta-0")
    got = from_json(AttemptMarker, jf.store.get_bytes(p))
    assert got.preempted and got.ended_utc == "t2"


def test_spawn_tokens_and_superseded_workers():
    import json

    from flora_deploy.jobs.attempts import (
        call_for_token,
        spawn_token,
        superseded,
        token_attempt,
    )

    jf = _jf()
    t1, t2 = (spawn_token("job", "cfg", SHARD.shard_id, n) for n in (1, 2))
    assert t1 == spawn_token("job", "cfg", SHARD.shard_id, 1) != t2 and token_attempt(t2) == 2
    assert not superseded(jf, "segcap", SHARD.shard_id, t1)  # no ledger: a manual run
    led = jf.paths.call("cfg", "segcap", SHARD.shard_id)
    jf.store.put_bytes(led, json.dumps({"token": t2, "status": "spawning"}).encode())
    assert superseded(jf, "segcap", SHARD.shard_id, t1)
    assert not superseded(jf, "segcap", SHARD.shard_id, t2)
    jf.store.put_bytes(led, json.dumps({"token": t2, "status": "quarantined"}).encode())
    assert superseded(jf, "segcap", SHARD.shard_id, t2)
    write_attempt(jf, "segcap", SHARD.shard_id, "ta-1", started_utc="t", orchestrator_attempt=2,
                  token=t2, call_id="fc-9")
    assert call_for_token(jf, "segcap", SHARD.shard_id, t2) == "fc-9"
    assert call_for_token(jf, "segcap", SHARD.shard_id, t1) is None


def test_replan_is_durable_and_first_writer_wins():
    from flora_deploy.jobs.attempts import read_replan, write_replan

    jf = _jf(lag=5)  # listing lag must not matter: replans are read with exists/get
    first = bisect(SHARD, done=["d--e1"])
    write_replan(jf, SHARD, first, created_utc="t1", done=SHARD.episodes, done_parts=["p1"])
    rp = read_replan(jf, SHARD)
    assert rp is not None and rp.children == first and rp.done == ("d--e1",)
    write_replan(jf, SHARD, bisect(SHARD), created_utc="t2")  # a redo with another done set
    assert read_replan(jf, SHARD).children == first
    q = quarantine(jf, SHARD, created_utc="2026-10-21T00:00:00Z", done=[], starts=3)
    assert q.children == first and q.record.detail["done"] == ["d--e1"]


def test_upstream_poisoned_episodes_are_found_by_exists():
    from flora_deploy.jobs.attempts import upstream_poisoned

    jf = _jf(lag=5)
    seg = Shard("segment-00000", "segment", ("a", "b", "c"), 3.0)
    single = Shard("segment-00001", "segment", ("d",), 1.0)
    cap = Shard("caption-00000", "caption", ("a", "b", "c", "d"), 4.0,
                depends_on=("segment-00000", "segment-00001"))
    child_b = next(c for c in bisect(seg) if c.episodes == ("b",))
    quarantine(jf, child_b, created_utc="t", starts=3)
    quarantine(jf, single, created_utc="t", starts=3)
    assert upstream_poisoned(jf, cap, [seg, single]) == {"b", "d"}
