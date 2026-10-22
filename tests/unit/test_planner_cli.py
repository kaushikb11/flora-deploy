"""`flora plan` / `flora estimate` CLI."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import template_text
from flora_deploy.cli.plan import parse_duration
from flora_deploy.config.schema import load_config

runner = CliRunner()
ARGS = ["--video-hours", "100", "--budget", "6000", "--deadline", "96h", "--max-gpus", "10"]


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "flora.yaml"
    p.write_text(template_text("exact_mp4_dir"))
    return p


def test_parse_duration():
    assert parse_duration("48h") == 48 * 3600
    assert parse_duration("90m") == 5400
    assert parse_duration("2d") == 2 * 86400
    assert parse_duration("12") == 12 * 3600
    with pytest.raises(typer.BadParameter):
        parse_duration("soon")


def test_plan_table(cfg):
    r = runner.invoke(app, ["plan", "-c", str(cfg), *ARGS], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert "exact x4" in r.output and "balanced x1" in r.output
    assert "reference" in r.output and "needs certificates" in r.output
    assert r.output.index("balanced x1") < r.output.index("exact x4")  # balanced first
    assert "Recommended: 2 (exact x4)" in r.output
    assert "exact x8 +long_lived_container" in r.output and "projection only" in r.output
    assert "warning: includes projected" in r.output
    assert "pinned 2026-10-21" in r.output


def test_plan_json(cfg):
    r = runner.invoke(app, ["plan", "-c", str(cfg), *ARGS, "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["workload"]["video_hours"] == 100 and d["recommended"] == 2
    assert d["options"][0]["preset"] == "balanced"
    assert 2 <= len(d["options"]) <= 4
    assert d["options"][0]["peak_gpus"] <= 10


def test_plan_infeasible_message(cfg):
    r = runner.invoke(app, ["plan", "-c", str(cfg), "--video-hours", "100", "--budget", "50",
                            "--preset", "exact"], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert "infeasible" in r.output and "No option fits" in r.output


def test_plan_write(cfg):
    r = runner.invoke(app, ["plan", "-c", str(cfg), *ARGS, "--preset", "exact", "--write", "1"])
    assert r.exit_code == 0, r.output
    c = load_config(cfg)
    seg = c.pools()["segcap"]  # review 6 H2: 4 episodes inside one input per container
    assert (seg.max_inputs, seg.episode_concurrency) == (1, 4) and c.modal.budget_usd == 6000
    assert (cfg.parent / "flora.yaml.bak").read_text() == template_text("exact_mp4_dir")


def test_plan_refuses_infeasible_write(cfg):
    r = runner.invoke(app, ["plan", "-c", str(cfg), "--video-hours", "100", "--budget", "50",
                            "--preset", "exact", "--write", "1"])
    assert r.exit_code == 1 and "infeasible" in r.output
    assert cfg.read_text() == template_text("exact_mp4_dir")


def test_plan_needs_workload_and_config(cfg, tmp_path):
    r = runner.invoke(app, ["plan", "-c", str(cfg)])
    assert r.exit_code == 2 and "--video-hours" in r.output
    r = runner.invoke(app, ["plan", "-c", str(tmp_path / "nope.yaml"), "--video-hours", "1"])
    assert r.exit_code == 2 and "flora init" in r.output


def test_plan_index_file(cfg, tmp_path):
    idx = tmp_path / "summary.json"
    idx.write_text(json.dumps({"groups": [{"pool": "human", "cameras": 3, "video_s": 7200,
                                           "episodes": 100}]}))
    r = runner.invoke(app, ["plan", "-c", str(cfg), "--index", str(idx), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["workload"]["episodes"] == 100


def test_estimate_placeholder_and_rates(cfg):
    r = runner.invoke(app, ["estimate", "-c", str(cfg), "--sample", "5"])
    assert r.exit_code == 2 and "deployed app" in r.output
    from flora_deploy.planner import options

    rates = __import__("pathlib").Path(options.__file__).with_name("rates.json")
    r = runner.invoke(app, ["estimate", "-c", str(cfg), "--rates", str(rates)],
                      env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert "segcap" in r.output and "measured" in r.output


def _rates_file(tmp_path, status: str):
    """The shipped rates.json with the balanced slot rows set here (placeholder, or measured
    n=1 ±50%), so the labels are tested whatever the shipped slot holds today."""
    from importlib import resources

    from flora_deploy.planner.rates import MEASURED_SLOTS

    d = json.loads(resources.files("flora_deploy.planner").joinpath("rates.json").read_text())
    for r in d["rates"]:
        if (r["preset"], r["kind"]) in MEASURED_SLOTS:
            r.update(status=status, uncertainty=1.0, n=None, source=None)
            if status == "measured":
                r.update(uncertainty=0.5, n=1, source="test run")
    p = tmp_path / f"rates-{status}.json"
    p.write_text(json.dumps(d))
    return p


def test_plan_labels_unmeasured_rates_estimate(cfg, tmp_path):
    rates = _rates_file(tmp_path, "placeholder")
    r = runner.invoke(app, ["plan", "-c", str(cfg), *ARGS, "--rates", str(rates)],
                      env={"COLUMNS": "250"})
    assert r.exit_code == 0, r.output
    row = next(ln for ln in r.output.splitlines() if "balanced x1" in ln)
    assert "estimate" in row and "measured" not in row
    d = json.loads(runner.invoke(app, ["plan", "-c", str(cfg), *ARGS, "--json", "--rates",
                                       str(rates)]).output)
    labels = {o["preset"]: o["rates"] for o in d["options"] if not o["requires"]}
    assert labels == {"balanced": "estimate (placeholder)", "exact": "measured"}
    # once measured (n=1), the row says so, with its sample count and band, not "estimate"
    rates = _rates_file(tmp_path, "measured")
    r = runner.invoke(app, ["plan", "-c", str(cfg), *ARGS, "--rates", str(rates)],
                      env={"COLUMNS": "250"})
    assert r.exit_code == 0, r.output
    row = next(ln for ln in r.output.splitlines() if "balanced x1" in ln)
    assert "measured (n=1, ±50%)" in row and "estimate" not in row
    d = json.loads(runner.invoke(app, ["plan", "-c", str(cfg), *ARGS, "--json", "--rates",
                                       str(rates)]).output)
    labels = {o["preset"]: o["rates"] for o in d["options"] if not o["requires"]}
    assert labels == {"balanced": "measured (n=1, ±50%)", "exact": "measured"}
