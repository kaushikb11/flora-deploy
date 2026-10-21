"""planner/rates.json loading and lookup."""

from __future__ import annotations

import json

import pytest

from flora_deploy.planner.rates import RatesFileError, RateTable


@pytest.fixture(scope="module")
def table() -> RateTable:
    return RateTable.load()


def test_shipped_rates_have_provenance(table):
    assert table.rates and table.startup
    for e in [*table.rates, *table.startup]:
        assert e.provenance and e.date and e.status in ("measured", "projected", "placeholder")
        assert e.uncertainty >= 0


def test_w3a_numbers(table):
    # docs/exact-concurrency.md: 0.245 video-h per container-h at concurrency 4
    assert table.rate("exact", "segcap", 1, 4).container_s_per_video_s == pytest.approx(
        1 / 0.245, rel=0.01)
    assert table.rate("exact", "segcap", 1, 1).container_s_per_video_s == pytest.approx(
        1 / 0.116, rel=0.01)


def test_lookup_rules(table):
    assert table.rate("exact", "segcap", 1, 6).concurrency == 4  # largest <= request
    assert table.rate("exact", "segcap", 1, 100).concurrency == 8
    assert table.rate("exact", "segcap", 3, 4).concurrency == 2
    assert table.rate("exact", "segcap", 2, 1).cameras == 3  # nearest; tie -> larger
    assert table.rate("exact", "reward.dense", 1).preset == "*"
    assert table.concurrencies("exact", "segcap", 1) == [1, 4, 8]
    with pytest.raises(KeyError):
        table.rate("exact", "segment", 1)


def test_startup_lookup(table):
    assert table.startup_for("exact", "segcap").startup_s == 213
    assert table.startup_for("exact", "reward.dense").startup_s == 224
    assert table.startup_for("balanced", "segment").startup_s == 130
    assert table.startup_for("balanced", "reward.terminal").startup_s == 224  # kind beats preset


def test_bad_files(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{")
    with pytest.raises(RatesFileError, match="not valid JSON"):
        RateTable.load(p)
    p.write_text(json.dumps({"schema": 2}))
    with pytest.raises(RatesFileError, match="schema 1"):
        RateTable.load(p)
    bad = {"schema": 1, "rates": [{"preset": "exact", "kind": "segcap", "cameras": 1,
                                   "concurrency": 1, "container_s_per_video_s": -1,
                                   "uncertainty": 0, "status": "measured", "provenance": "x",
                                   "date": "d"}]}
    p.write_text(json.dumps(bad))
    with pytest.raises(RatesFileError, match="positive"):
        RateTable.load(p)
    p.write_text(json.dumps({"schema": 1, "rates": [{"kind": "segcap"}]}))
    with pytest.raises(RatesFileError, match=r"rates\[0\]"):
        RateTable.load(p)
