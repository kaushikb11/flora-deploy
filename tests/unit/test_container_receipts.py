"""What lands under ``<job root>/containers/`` must never break ``flora report`` or the budget
gate. H0 on Modal (2026-10-22): the fake workers' startup facts (``write_startup_facts``) were
written there, carried ``kind`` / ``schema`` / ``startup``, and ``ContainerReceipt(**json)``
raised, so the report had no per-container rows. Every shape is built from its writer."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import pytest

from flora_deploy.jobs.budget import parse_container_receipt, read_container_receipts
from flora_deploy.jobs.containers import (
    read_startup_facts,
    startup_facts_path,
    write_startup_facts,
)
from flora_deploy.jobs.orchestrator import utc
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import ContainerReceipt
from tests.fakes.modal import FakeStore
from tests.unit.test_budget_overspend import JOB, T0, gate


def _receipt(cid="c1", stopped=None) -> ContainerReceipt:
    return ContainerReceipt(cid, "w", "H200", 1, 0.0, None, 0.0, None, None, True,
                            utc(T0 - 100), utc(T0), stopped)


def _put(st, r: ContainerReceipt) -> None:
    st.put_bytes(JobPaths(JOB).container(r.container_id), json.dumps(asdict(r)).encode())


def test_container_receipts_round_trip():
    st = FakeStore()
    rs = [_receipt("c1"), _receipt("c2", stopped=utc(T0))]
    for r in rs:
        _put(st, r)
    assert sorted(read_container_receipts(st, JOB), key=lambda r: r.container_id) == rs


def test_an_old_receipt_without_optional_fields_loads():
    d = asdict(_receipt())
    del d["stopped_utc"]
    assert parse_container_receipt(d) == _receipt()


def test_unknown_keys_are_ignored_with_a_debug_log(caplog):
    d = {**asdict(_receipt()), "kind": "segment", "future_field": [1, 2]}
    with caplog.at_level(logging.DEBUG, logger="flora_deploy.jobs.budget"):
        assert parse_container_receipt(d, "p.json") == _receipt()
    assert "future_field" in caplog.text and "kind" in caplog.text


@pytest.mark.parametrize("doc", [[], "x", {"container_id": "c1"},
                                 {**asdict(_receipt()), "schema": "something_else_v1"}])
def test_non_receipts_are_skipped(doc):
    assert parse_container_receipt(doc) is None


def test_startup_facts_live_outside_the_receipts_prefix():
    assert "/containers/" not in startup_facts_path(JOB, "ta-1")
    st = FakeStore()
    write_startup_facts(st, JOB, "ta-1", "segment", {"phased": True}, now=T0)
    assert list(st.list(JobPaths(JOB).root + "/containers/")) == []
    assert read_startup_facts(st, JOB)["ta-1"]["kind"] == "segment"


def test_legacy_startup_facts_under_containers_break_nothing():
    """What the H0 rehearsal's workers wrote: facts in ``containers/``, next to a receipt."""
    src = FakeStore()
    write_startup_facts(src, JOB, "ta-1", "caption", {"phased": True}, now=T0)
    st = FakeStore()
    st.put_bytes(f"{JobPaths(JOB).root}/containers/ta-1.json",
                 src.get_bytes(startup_facts_path(JOB, "ta-1")))
    st.put_bytes(f"{JobPaths(JOB).root}/containers/junk.json", b"{not json")
    _put(st, _receipt())
    assert read_container_receipts(st, JOB) == [_receipt()]
    assert read_startup_facts(st, JOB)["ta-1"]["kind"] == "caption"
    assert gate(store=st, refresh_s=0).spend_report().receipts_usd == pytest.approx(100.0)
