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
