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
        assert e.provenance and e.date and e.status in (
            "measured", "projected", "placeholder", "stale")
        assert e.uncertainty >= 0


def test_cu128_numbers(table):
    """Review 5 M-C: the deployed exact worker runs gpu_thread (P1a-5s): 6.4 measured at c=4,
    the default lookup. S1's 7.0 (gpu_thread off) is used only with the feature off. Perf-1's
    4.35 projection of the same configuration is superseded (stale)."""
    e = table.rate("exact", "segcap", 1, 4)  # features=None: the deployed configuration
    assert e.container_s_per_video_s == 6.4 and e.status == "measured"
    assert "cu128" in e.image and e.requires == ("gpu_thread",) and e.cameras == "*"
    off = table.rate("exact", "segcap", 1, 4, features=())
    assert off.container_s_per_video_s == pytest.approx(1 / 0.143, rel=0.01)
    assert off.status == "measured" and off.excludes == ("gpu_thread",)
    assert table.startup_for("exact", "segcap").startup_s == 103  # S1 @modal.enter


def test_measured_beats_projected_at_equal_concurrency():
    """Review 5 M-C: a projection that needs an enabled feature must not displace a measured
    row of the same configuration."""
    row = {"preset": "exact", "kind": "segcap", "cameras": 1, "concurrency": 4,
           "uncertainty": 0.35, "provenance": "x", "date": "d"}
    t = RateTable.from_dict({"schema": 1, "rates": [
        {**row, "container_s_per_video_s": 7.0, "status": "measured"},
        {**row, "container_s_per_video_s": 4.35, "status": "projected", "requires": ["f"]},
    ]}, source="t")
    assert t.rate("exact", "segcap", 1, 4, features=("f",)).container_s_per_video_s == 7.0
    t2 = RateTable.from_dict({"schema": 1, "rates": [
        {**row, "container_s_per_video_s": 7.0, "status": "measured", "excludes": ["f"]},
        {**row, "container_s_per_video_s": 6.4, "status": "measured", "requires": ["f"]},
        {**row, "container_s_per_video_s": 4.35, "status": "projected", "requires": ["f"]},
    ]}, source="t")
    assert t2.rate("exact", "segcap", 1, 4, features=("f",)).container_s_per_video_s == 6.4
    assert t2.rate("exact", "segcap", 1, 4, features=()).container_s_per_video_s == 7.0


def test_stale_rows_are_never_used(table):
    stale = [e for e in table.rates if e.status == "stale"]
    assert stale and all("cu130" in e.image or "Superseded" in e.provenance for e in stale)
    for cams in (1, 2, 3):
        for c in (1, 2, 4, 8, 100):
            for feats in (None, (), ("gpu_thread",), ("gpu_thread", "long_lived_container")):
                assert table.rate("exact", "segcap", cams, c, feats).status != "stale"


def test_lookup_rules(table):
    assert table.rate("exact", "segcap", 1, 6).concurrency == 4  # largest <= request
    assert table.rate("exact", "segcap", 1, 100).concurrency == 4  # x8 is not deployed
    full = ("gpu_thread", "long_lived_container")
    assert table.rate("exact", "segcap", 1, 100, features=full).concurrency == 8
    assert table.rate("exact", "segcap", 1, 100, features=full).status == "projected"
    assert table.rate("exact", "segcap", 1, 1).concurrency == 4  # no usable c=1 row on cu128
    assert table.rate("exact", "segcap", 3, 4).cameras == "*"  # measured on a 1-/3-view mix
    assert table.rate("exact", "reward.dense", 1).preset == "*"
    assert table.concurrencies("exact", "segcap", 1) == [4]
    assert table.concurrencies("exact", "segcap", 1, full) == [4, 8]
    assert table.features() == {"gpu_thread", "long_lived_container"}
    with pytest.raises(KeyError):
        table.rate("exact", "segment", 1)


def test_startup_lookup(table):
    assert table.startup_for("exact", "segcap").startup_s == 103
    assert table.startup_for("exact", "reward.dense").startup_s == 229
    assert table.startup_for("balanced", "segment").startup_s == 130
    assert table.startup_for("balanced", "reward.terminal").startup_s == 229  # kind beats preset


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
