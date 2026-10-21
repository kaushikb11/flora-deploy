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
    """Review 4 M5: S1 on the production cu128 img_hfref measured 0.143 video-h per GPU-h at
    c=4 (7.0); gpu_thread's 4.35 is a projection, used only when the feature is enabled."""
    e = table.rate("exact", "segcap", 1, 4)
    assert e.container_s_per_video_s == pytest.approx(1 / 0.143, rel=0.01)
    assert e.status == "measured" and "cu128" in e.image and e.requires == ()
    g = table.rate("exact", "segcap", 1, 4, features={"gpu_thread"})
    assert g.container_s_per_video_s == 4.35 and g.status == "projected"
    assert g.requires == ("gpu_thread",)
    assert table.startup_for("exact", "segcap").startup_s == 103  # S1 @modal.enter


def test_stale_rows_are_never_used(table):
    stale = [e for e in table.rates if e.status == "stale"]
    assert stale and all("cu130" in e.image for e in stale)
    for cams in (1, 2, 3):
        for c in (1, 2, 4, 8, 100):
            assert table.rate("exact", "segcap", cams, c).status != "stale"
            assert table.rate("exact", "segcap", cams, c, features={"gpu_thread"}
                              ).status != "stale"


def test_lookup_rules(table):
    assert table.rate("exact", "segcap", 1, 6).concurrency == 4  # largest <= request
    assert table.rate("exact", "segcap", 1, 100).concurrency == 4  # x8 needs gpu_thread
    assert table.rate("exact", "segcap", 1, 100, features=("gpu_thread",)).concurrency == 8
    assert table.rate("exact", "segcap", 1, 1).concurrency == 4  # no usable c=1 row on cu128
    assert table.rate("exact", "segcap", 3, 4).cameras == 1  # 3-view rows are stale
    assert table.rate("exact", "reward.dense", 1).preset == "*"
    assert table.concurrencies("exact", "segcap", 1) == [4]
    assert table.concurrencies("exact", "segcap", 1, ("gpu_thread",)) == [4, 8]
    assert table.features() == {"gpu_thread"}
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
