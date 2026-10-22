"""The tables of ``docs/costs.md`` are generated from ``planner/rates.json`` and the planner
(``scripts/cost_table.py``), so the cost guide cannot drift from the rates ``flora plan`` uses."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from flora_deploy.planner.rates import RateTable
from scripts import cost_table as C


def test_costs_md_is_up_to_date():
    assert C.DOC.read_text(encoding="utf-8") == C.render(), (
        "docs/costs.md is stale: run `uv run python scripts/cost_table.py --write`")


def test_the_table_is_the_planners_price():
    """Each row is ``flora plan``'s all-in estimate, and its parts add up to it."""
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.planner.options import PlanRequest, WorkloadSummary, build_option

    for v in C.SIZES:
        p = C.price(v, C.SEGCAP)
        req = PlanRequest(workload=WorkloadSummary.from_hours(v), stages=C.SEGCAP,
                          presets=("balanced",), max_gpus=2)
        o = build_option(req, RateTable.load(), pinned_rates(), "balanced", 1)
        assert p.cost == o.cost_usd
        assert p.containers == {"segment": 1, "caption": 1}
        assert p.busy + p.startup + p.idle + p.orchestrator == pytest.approx(p.cost, abs=0.006)
        assert p.label.startswith("measured")
    small, big = C.price(0.05, C.SEGCAP), C.price(100.0, C.SEGCAP)
    assert small.per_h() > big.per_h() > C.busy_per_video_h(C.SEGCAP)


def test_a_rate_change_changes_the_doc():
    """A new measurement (what scripts/wave_h_finalize.py writes) moves the generated numbers."""
    data = json.loads(
        (C.DOC.parents[1] / "src/flora_deploy/planner/rates.json").read_text(encoding="utf-8"))
    for r in data["rates"]:
        if r["preset"] == "balanced" and r["kind"] == "caption":
            r["container_s_per_video_s"] = 0.45
    table = replace(RateTable.from_dict(data, source="x"), source="planner/rates.json")
    old = C.render()
    new = C.render(table=table)
    assert new != old
    assert C.busy_per_video_h(C.SEGCAP, table) < C.busy_per_video_h(C.SEGCAP)
    # the batch-fill estimate is shown only while it is below the measured rate
    assert C.batch_fill_table(RateTable.load()) is not None
    for r in data["rates"]:
        if r["preset"] == "balanced" and r["kind"] in C.BATCH_FILL_ESTIMATE:
            r["container_s_per_video_s"] = C.BATCH_FILL_ESTIMATE[r["kind"]]
    measured = RateTable.from_dict(data, source="x")
    assert C.batch_fill_table(measured) is None
    assert "Full batches" not in C.block_levers(measured)


def test_every_number_is_labelled():
    text = C.DOC.read_text(encoding="utf-8")
    short = text.split("<!-- cost-table:short -->")[1].split("<!-- /cost-table:short -->")[0]
    assert "measured (n=1, ±50%)" in short
    levers = text.split("<!-- cost-table:levers -->")[1].split("<!-- /cost-table:levers -->")[0]
    assert "estimate" in levers and "not measured" in levers


def test_missing_or_unknown_blocks_are_refused():
    with pytest.raises(ValueError, match="misses"):
        C.render("no blocks here\n")
    extra = "<!-- cost-table:nope -->\n<!-- /cost-table:nope -->\n"
    text = C.DOC.read_text(encoding="utf-8") + extra
    with pytest.raises(ValueError, match="unknown"):
        C.render(text)
