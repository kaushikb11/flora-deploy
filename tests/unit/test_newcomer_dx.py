"""The README walked cold (newcomer report, 2026-10-22): each friction point it hit, as a check.

The README path is `flora init` (balanced, `allow_uncertified: true`), `flora doctor`, `flora
plan --video-hours 1` against the template's $25 budget, and the local demo (`flora init -t
local_demo`, `flora run --local --fake-models --sample-data`, `flora results`, `status`,
`report`)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import template_text
from flora_deploy.config.schema import load_config

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
runner = CliRunner()

# Internal vocabulary that must not reach a user: decision ids, review/gate names, research
# notes, phase ids.
JARGON = re.compile(r"\bD\d{1,2}\b|\bGate [A-D]?\b|\bGate certificate|poison pill|"
                    r"modal_verification|§\d|\bP\d[a-z]?-\d|\bWave H\b|\bH-lite\b|review \d")


def assert_no_jargon(text: str, where: str) -> None:
    hits = sorted({m.group(0) for m in JARGON.finditer(text)})
    assert not hits, f"{where}: internal jargon {hits}"


# ---------------------------------------------------------------------------------------------
# 1. `flora plan` honours `allow_uncertified: true`


@pytest.fixture
def balanced_cfg(tmp_path):
    p = tmp_path / "flora.yaml"
    p.write_text(template_text("balanced_lerobot"))
    return p


def _plan(cfg, *args, columns="200"):
    return runner.invoke(app, ["plan", "-c", str(cfg), "--video-hours", "1", *args],
                         env={"COLUMNS": columns})


def test_plan_honours_allow_uncertified_from_the_config(balanced_cfg):
    """With `allow_uncertified: true` in flora.yaml, balanced is runnable, labelled uncertified
    and, as the cheapest runnable option, recommended. Nothing claims a flag is still needed,
    and "No option fits" never appears while an option is under budget."""
    assert load_config(balanced_cfg).allow_uncertified
    r = _plan(balanced_cfg)
    assert r.exit_code == 0, r.output
    out = r.output
    assert "Recommended: 1 (balanced x1)" in out
    assert "uncertified" in out and "allowed by `allow_uncertified: true`" in out
    assert "No option fits" not in out and "No runnable option" not in out
    assert "--allow-uncertified" not in out and "unavailable" not in out
    d = json.loads(_plan(balanced_cfg, "--json").output)
    bal = d["options"][d["recommended"] - 1]
    assert bal["preset"] == "balanced" and bal["cost_usd"] < 25
    assert bal["uncertified_allowed"] and not bal["unavailable"]
    assert bal["certification"] == "needs certificates"  # allowed is not certified
    r = _plan(balanced_cfg, "--write", "1")  # and writing the recommendation keeps balanced
    assert r.exit_code == 0, r.output
    assert load_config(balanced_cfg).preset == "balanced"


def test_plan_recommends_the_cheapest_runnable_option(balanced_cfg):
    d = json.loads(_plan(balanced_cfg, "--budget", "100", "--json").output)
    runnable = [o for o in d["options"] if o["status"] != "infeasible" and not o["requires"]
                and not o["unavailable"]]
    best = d["options"][d["recommended"] - 1]
    assert len(runnable) >= 2  # exact fits a $100 budget too: the cheaper one still wins
    assert best["cost_usd"] == min(o["cost_usd"] for o in runnable)
    assert best["preset"] == "balanced"


def test_plan_without_opt_in_says_why_not_no_option_fits(tmp_path):
    """An exact config (allow_uncertified: false) asking about balanced: it fits the budget but
    cannot run. The message names the reason and the fix instead of telling the user to raise
    a budget that is not the problem."""
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(template_text("exact_lerobot"))
    assert not load_config(cfg).allow_uncertified
    r = _plan(cfg, "--preset", "balanced", "--budget", "100", "--max-gpus", "4")
    assert r.exit_code == 0, r.output
    assert "unavailable" in r.output and "allow_uncertified: true" in r.output
    assert "No option fits" not in r.output
    assert "No runnable option" in r.output and "uncertified" in r.output


def test_plan_readable_at_80_columns(balanced_cfg):
    """No truncated cells (rich's "…") at 80 columns: the options are stacked instead."""
    r = _plan(balanced_cfg, columns="80")
    assert r.exit_code == 0, r.output
    assert "…" not in r.output and "┃" not in r.output
    assert "1. balanced x1 (recommended): ok, uncertified" in r.output
    assert "cost $" in r.output and "wall-clock" in r.output
    wide = _plan(balanced_cfg, columns="200").output
    assert "…" not in wide and "┃" in wide


def test_plan_cost_says_what_it_includes(balanced_cfg):
    out = _plan(balanced_cfg).output
    assert "all-in estimate" in out and "startup" in out
    assert "H100! means exactly an H100" in out
    assert_no_jargon(out, "flora plan")


# ---------------------------------------------------------------------------------------------
# 2 and 5. The local demo: `flora results` shows the labels; status / report show no spend


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """The quickstart, once: `flora init -t local_demo` then the local fake-models run."""
    import os

    from flora_deploy import sdk
    from tests.fakes.flora_backend import FakeBackend

    tmp = tmp_path_factory.mktemp("demo")
    old, env = Path.cwd(), os.environ.pop("FLORA_STORE", None)
    sdk.set_backend(FakeBackend(tmp / "remote-stores"))
    os.chdir(tmp)
    try:
        assert runner.invoke(app, ["init", "--template", "local_demo"]).exit_code == 0
        out = runner.invoke(app, ["run", "--local", "--fake-models", "--sample-data"])
        assert out.exit_code == 0, out.output
        yield tmp, out.output
    finally:
        os.chdir(old)
        sdk.set_backend(None)
        if env is not None:
            os.environ["FLORA_STORE"] = env


def _in(demo, argv, columns="80"):
    import os

    old = Path.cwd()
    os.chdir(demo[0])
    try:
        r = runner.invoke(app, argv, env={"COLUMNS": columns})
    finally:
        os.chdir(old)
    assert r.exit_code == 0, r.output
    return r.output


def test_results_caption_shows_spans_and_captions(demo):
    out = _in(demo, ["results", "caption"])
    assert re.search(r"\d+\.\d–\d+\.\d s", out), out  # a segment's start–end
    assert "(task: " in out  # the stand-in captioner's text, not a segment count
    assert "segments" not in out.split("\n", 2)[-1]
    assert all(len(line) <= 80 for line in out.splitlines() if "│" in line), out
    assert "certified: no (fake_models) for every record" in out


def test_results_segment_terminal_dense_show_their_labels(demo):
    seg = _in(demo, ["results", "segment"])
    assert re.search(r"\d+\.\d–\d+\.\d s", seg)
    term = _in(demo, ["results", "reward.terminal"])
    assert "score (1-5)" in term and "expected" in term
    assert re.search(r"│ [1-5] +│ \d\.\d\d", term), term
    dense = _in(demo, ["results", "reward.dense"])
    assert "summary" in dense and re.search(r"[123]×\d+.*: [123]( [123])*", dense), dense


def test_dense_summary_fits_its_width():
    from flora_deploy.cli.results import dense_summary

    assert dense_summary([3, 3, 3, 3, 1, 1]) == "3×4 1×2: 3 3 3 3 1 1"
    long = dense_summary([3] * 100 + [1], width=12)
    assert long.startswith("3×100 1×1: ") and long.endswith(" …")
    assert len(long.split(": ", 1)[1]) <= 12


def test_local_demo_status_and_report_show_no_spend(demo):
    for argv in (["status"], ["report"]):
        out = _in(demo, argv, columns="200")
        assert "n/a: local run, fake models; nothing charged" in out and "$0" in out, out
        assert not re.search(r"\$[1-9]|\$0\.[0-9]*[1-9]", out), out  # no dollar amounts
        assert "(running)" not in out and "per video-hour" not in out
        assert_no_jargon(out, f"flora {argv[0]}")
    assert "all finished" in _in(demo, ["report"], columns="200")
    out = _in(demo, ["report", "--json"])
    d = json.loads(out[out.index("{"):])  # after the one stand-in note line
    assert d["spend"]["simulated"] and d["usd_per_video_hour"] is None
    assert all(c["stopped"] for c in d["spend"]["containers"])


def test_local_demo_run_output_has_no_jargon(demo):
    assert_no_jargon(demo[1], "flora run --local")


# ---------------------------------------------------------------------------------------------
# 4. Setup truth: the default template, the Volume, Hugging Face


def test_init_default_template_writes_to_a_volume(tmp_path):
    from flora_deploy.cli.init import DEFAULT_TEMPLATE, TEMPLATES

    cfg = load_config(_write(tmp_path, template_text(DEFAULT_TEMPLATE)))
    assert cfg.output.sink == f"volume://flora-results/{cfg.project}"
    for name in TEMPLATES:  # no template ships a placeholder bucket that fails doctor
        assert "my-bucket" not in template_text(name), name
        assert_no_jargon(template_text(name), f"template {name}")
        assert "exact_lerobot.yaml" not in template_text(name), name


def _write(tmp_path, text):
    p = tmp_path / "flora.yaml"
    p.write_text(text)
    return p


def test_doctor_does_not_fail_on_a_volume_deploy_creates(tmp_path):
    from flora_deploy.cli import doctor as d
    from tests.unit.test_doctor import FakeProbes, by_name, write_cfg

    p = write_cfg(tmp_path, **{"s3://my-bucket/flora/lerobot-small":
                               "volume://flora-results/lerobot-small"})
    c = by_name(d.run_checks(p, FakeProbes(volume="missing")))
    assert c["store.volume"].status == "pass" and "`flora deploy` creates it" in \
        c["store.volume"].message


def test_deploy_creates_the_sink_volume():
    """The README says a Volume needs no setup: the app factory mounts the sink Volume with
    create_if_missing, so `flora deploy` makes it."""
    from flora_deploy.config.schema import loads_config
    from flora_deploy.modal_app.factory import plan_app

    cfg = loads_config(template_text("exact_mp4_dir"), source="t")
    assert cfg.output.sink.startswith("volume://flora-results/")
    plan = plan_app(cfg, derived_receipts={"captioner_bf16": "0" * 64})
    mounts = [m for o in plan.objects for m in o.volumes.values()]
    assert any(m.name == "flora-results" and m.create for m in mounts)


def test_doctor_says_each_certificate_problem_once(tmp_path):
    from flora_deploy.cli import doctor as d
    from tests.unit.test_doctor import FakeProbes, by_name

    p = _write(tmp_path, template_text("balanced_lerobot"))
    c = by_name(d.check_certificates(FakeProbes(), load_config(p)))
    fixes = [x.fix for x in c.values()]
    assert len(c) == 1 and len(fixes) == len(set(fixes)), c
    assert_no_jargon(" ".join(x.message + x.fix for x in c.values()), "doctor certificates")


# ---------------------------------------------------------------------------------------------
# 3, 4, 7, 8. The README: definitions, prerequisites, links, no jargon


def _section(heading: str) -> str:
    return README.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


def test_readme_you_need_matches_what_doctor_checks():
    need = _section("Label your own data").split("```", 1)[0]
    assert "nothing to set up" not in need
    assert "`flora deploy` creates the Volume" in need
    assert "Hugging Face" in need and "nvidia/Flora-*" in need and "not gated" in need
    assert "hf auth login" in need


def test_readme_cost_definitions_match_plan_and_report(balanced_cfg):
    cost = _section("Cost")
    assert "of GPU work" in cost and "`flora plan` shows the all-in estimate" in cost
    assert "startup" in cost and "idle" in cost and "dominate small jobs" in cost
    m = re.search(r"\*\*\$(\d+\.\d\d) per video-hour\*\*", cost)
    assert m, cost
    # the plan's all-in estimate includes startup and rewards, so it is above the busy figure
    d = json.loads(_plan(balanced_cfg, "--json").output)
    bal = next(o for o in d["options"] if o["preset"] == "balanced")
    assert bal["cost_usd"] > float(m.group(1))
    # index cost, and that doctor / plan are free
    assert "`flora index`" in cost and "--local" in cost and "$0.06" in cost


def test_readme_has_no_internal_jargon_and_links_examples():
    assert_no_jargon(README, "README.md")
    assert "](examples/)" in README
    ex = (ROOT / "examples/lerobot_small/README.md").read_text(encoding="utf-8")
    assert "Run on Modal" not in ex and "placeholders until measured" not in ex
    heading = re.search(r"\"([^\"]+)\" steps in the \[top-level README\]", ex).group(1)
    assert f"## {heading}\n" in README


def test_index_help_says_what_it_costs():
    out = runner.invoke(app, ["index", "--help"], env={"COLUMNS": "200"}).output
    assert "Modal CPU" in out and "$0.06" in out and "free" in out


# ---------------------------------------------------------------------------------------------
# 3. The run report: (d) is done when its deployed job was, whatever its inline lab did


def test_report_shows_d_done_when_only_its_inline_lab_failed():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import wave_h as wh
    import wave_h_finalize as fin

    st = wh.State.from_json({"run_id": "r", "created": 0.0, "billing_since": 0.0,
                             "steps": {"balanced": {
        "status": "failed", "attempts": [],
        "result": {"kinds": {"tempseg": {"error": "x"}},
                   "jobs": {"human": {"state": "done"}}}}}})
    assert fin.inline_lab(st)
    assert fin.step_status(st, "balanced").startswith("done")
    st.steps["balanced"].result["jobs"]["human"]["state"] = "failed"
    assert fin.step_status(st, "balanced") == "failed"


def test_cost_block_is_plain_and_busy_only():
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "scripts"))
    import wave_h_finalize as fin

    f = SimpleNamespace(date="2026-10-22", cost={
        "balanced": {"usd_per_video_h": 5.61, "uncertainty": 0.5}, "exact": None})
    block = fin.cost_block(f)
    assert "$5.61 per video-hour** of GPU work" in block
    assert "`flora plan` shows the all-in estimate" in block
    assert 2 <= len(block.splitlines()) <= 3
    assert_no_jargon(block, "README cost block")
