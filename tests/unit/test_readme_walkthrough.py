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
    assert bal["certification"] == "partly certified"  # allowed is not certified
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
    assert "1. balanced x1 (recommended): ok, partly certified" in r.output  # rewards: reference
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
    assert "certified: no (fake models) for every record" in out


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
    """Nothing is charged; the one dollar amount is the as-if-on-Modal price the local budget
    gate counts, labelled as such, against the config's budget (newcomer 2, item 7)."""
    for argv in (["status"], ["report"]):
        out = _in(demo, argv, columns="200")
        assert "n/a: local run, fake models; nothing charged" in out and "$0" in out, out
        m = re.search(r"priced as if on Modal: \$(\d+\.\d\d) of the \$5\.00 budget \(a "
                      r"projected Modal-equivalent cost", out)
        assert m and float(m.group(1)) > 0, out
        rest = out.replace(m.group(0), "").replace("$5.00", "")
        assert not re.search(r"\$[1-9]|\$0\.[0-9]*[1-9]", rest), out  # no other amounts
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
    fixes = [x.fix for x in c.values() if x.fix]
    problems = [x for x in c.values() if "reference" not in x.message]  # rewards: reference
    assert len(problems) == 1 and len(fixes) == len(set(fixes)), c
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


# ---------------------------------------------------------------------------------------------
# Second cold walk (2026-10-22): the README's own-data path before anything exists on Modal


def test_index_local_creates_a_missing_volume_before_downloading(tmp_path, monkeypatch):
    """`flora index --local` on the default (volume://) sink, before `flora deploy`: the store is
    opened with ``create=True`` in the config's environment, before anything is downloaded, so
    the Volume exists for the receipts (as `flora deploy` would make it). The doctor, README and
    this path agree: nothing to create by hand."""
    import flora_deploy.export.results as res
    import flora_deploy.jobs.indexing as indexing

    seen: list[tuple[str, dict]] = []

    class Stop(Exception):
        pass

    def opened(uri, **kw):
        seen.append((uri, kw))
        return object()

    def download(*a, **k):
        assert seen, "downloaded before the store (and its Volume) was opened"
        raise Stop

    monkeypatch.setattr(res, "store_from_uri", opened)
    monkeypatch.setattr(indexing, "open_dataset", download)
    cfg = _write(tmp_path, template_text("local_demo"))
    (tmp_path / "videos").mkdir()
    r = runner.invoke(app, ["index", "--local", "-c", str(cfg)], env={"COLUMNS": "200"})
    assert isinstance(r.exception, Stop), r.output
    assert seen == [("modal-volume://flora-results/local-demo",
                     {"environment": "main", "create": True})], seen
    readme = " ".join((ROOT / "README.md").read_text().split())
    assert "create the Volume first" not in readme and "is created" in readme


def test_results_without_a_run_says_so(tmp_path):
    """A job that never ran: the error says what that means, not just a store path."""
    from flora_deploy.export.results import ResultsNotFound, resolve_config_hash
    from flora_deploy.jobs.stores import LocalStore

    with pytest.raises(ResultsNotFound, match="has not been run here"):
        resolve_config_hash(LocalStore(str(tmp_path)), "my-labels")
    store = tmp_path / "store"
    store.mkdir()
    r = runner.invoke(app, ["results", "caption", "--job", "my-labels", "--store", str(store)],
                      env={"COLUMNS": "200"})
    assert r.exit_code == 2, r.output
    assert "has not been run here" in r.output, r.output


def test_store_from_uri_names_the_volume_spelling():
    from flora_deploy.export.results import store_from_uri

    with pytest.raises(ValueError, match="modal-volume://flora-results/my-labels"):
        store_from_uri("volume://flora-results/my-labels")


def test_plan_without_a_summary_offers_the_free_index(tmp_path):
    cfg = _write(tmp_path, template_text("local_demo"))
    r = runner.invoke(app, ["plan", "-c", str(cfg)], env={"COLUMNS": "200"})
    assert r.exit_code == 2
    assert "`flora run --local --fake-models --sample-data --config " in " ".join(r.output.split())


def test_results_more_rows_is_singular_for_one(demo):
    out = _in(demo, ["results", "reward.dense", "--limit", "5"], columns="200")
    assert "... 1 more row (" in out, out


def test_doctor_fixes_name_commands_not_repo_paths(tmp_path):
    from flora_deploy.cli import doctor as d
    from tests.unit.test_doctor import FakeProbes, by_name, write_cfg

    bad = _write(tmp_path, template_text("local_demo") + "\nnot_a_key: 1\n")
    c = by_name(d.run_checks(bad, FakeProbes()))
    assert "src/" not in c["config"].fix and "flora init --list" in c["config"].fix
    p = write_cfg(tmp_path, **{"s3://my-bucket/flora/lerobot-small":
                               "volume://flora-results/lerobot-small"})
    c = by_name(d.run_checks(p, FakeProbes(volume="missing")))
    vol = c["store.volume"]
    assert "with or without --local" in vol.message and "modal volume create" in vol.fix


def test_cli_docs_keep_code_spans_literal():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import cli_docs

    cell = cli_docs._cell("default: <dir> (Modal: use `flora resume <job>`)")
    assert cell == "default: &lt;dir&gt; (Modal: use `flora resume <job>`)"


# ---------------------------------------------------------------------------------------------
# Third walk (2026-10-22): leftovers. The Modal Environment, a local-only doctor, readable
# counts, help panels.


class NotFoundError(Exception):  # modal.exception.NotFoundError, matched by name
    pass


def test_index_local_uses_the_configured_modal_environment(tmp_path, monkeypatch):
    """`flora index --local` opens (and, when missing, creates) the sink Volume in
    `modal.environment`, not Modal's default Environment."""
    import flora_deploy.export.results as res
    import flora_deploy.jobs.indexing as indexing

    seen = {}

    class Stop(Exception):
        pass

    def store_from_uri(uri, **kw):
        seen.update(kw, uri=uri)
        return object()

    def download(*a, **k):
        raise Stop

    monkeypatch.setattr(res, "store_from_uri", store_from_uri)
    monkeypatch.setattr(indexing, "open_dataset", download)
    text = template_text("local_demo").replace("environment: main", "environment: staging")
    cfg = _write(tmp_path, text)
    (tmp_path / "videos").mkdir()
    r = runner.invoke(app, ["index", "--local", "-c", str(cfg)], env={"COLUMNS": "200"})
    assert isinstance(r.exception, Stop), r.output
    assert seen["environment"] == "staging" and seen["create"] is True, seen


def test_index_on_modal_runs_in_the_configured_environment(tmp_path, monkeypatch):
    """The Modal path runs the index app in `modal.environment` too (mocked: no Modal call)."""
    import contextlib
    import importlib
    from types import SimpleNamespace

    import modal

    runs = []

    class FakeApp:
        def run(self, **kw):
            runs.append(kw)
            return contextlib.nullcontext()

    summary = {"counts": dict.fromkeys(
        ("episodes", "indexed", "source_error", "failed", "resumed", "envelope_ok",
         "out_of_envelope", "oracle_eligible", "planned"), 0),
        "dataset": {"name": "x", "revision": "r"}, "envelope_policy": "warn", "video_s": 0.0,
        "durations": {}, "envelope_reasons": {}, "ineligibility": {}, "source_errors": {},
        "codecs": {}, "failed": []}
    fake = SimpleNamespace(app=FakeApp(), run_index_remote=SimpleNamespace(
        remote=lambda *a: summary))
    monkeypatch.setattr(importlib, "reload", lambda m: fake)
    monkeypatch.setattr(modal, "enable_output", contextlib.nullcontext)
    monkeypatch.delenv("FLORA_INDEX_VOLUME", raising=False)
    text = template_text("balanced_lerobot")
    assert "environment: main" in text
    cfg = _write(tmp_path, text.replace("environment: main", "environment: staging"))
    r = runner.invoke(app, ["index", "-c", str(cfg)], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert runs == [{"environment_name": "staging"}]


class _NoModal:
    """Doctor probes that fail the test on any Modal or Hub call."""

    def __init__(self, files=None):
        self.files = files

    def __getattr__(self, name):
        if name.startswith(("modal_", "volume_", "hf_", "store_", "certificate")):
            return lambda *a, **k: pytest.fail(f"local-only doctor called {name}")
        raise AttributeError(name)

    def local_files(self, path, pattern):
        return self.files


def test_doctor_on_the_local_demo_skips_modal(tmp_path, monkeypatch):
    """The local demo needs no Modal account: doctor makes no Modal / Hub call, reports no
    warning or failure, and says in one line how to move the config to Modal."""
    from flora_deploy.cli import doctor as d

    cfg = _write(tmp_path, template_text("local_demo"))
    assert d.local_only(load_config(cfg))
    checks = d.run_checks(cfg, _NoModal(files=None))  # ./videos not written yet
    assert [c for c in checks if c.status in ("warn", "fail")] == [], checks
    c = {x.name: x for x in checks}
    assert c["modal"].status == "info" and "to run on Modal" in c["modal"].fix
    assert "flora prepare" not in " ".join(x.message + x.fix for x in checks)
    assert c["input"].status == "info" and "--sample-data" in c["input"].fix
    assert {x.name: x for x in d.run_checks(cfg, _NoModal(files=3))}["input"].status == "pass"
    monkeypatch.setattr(d, "_probes_factory", lambda: _NoModal(files=None))
    r = runner.invoke(app, ["doctor", "-c", str(cfg)], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert "INFO modal" in r.output and "0 warnings, 0 failed, 3 notes" in r.output
    assert_no_jargon(r.output, "flora doctor (local demo)")


def test_doctor_still_checks_modal_for_a_staged_local_input():
    """A local input.path staged to a Volume (input.stage_to) runs on Modal: full checks."""
    from flora_deploy.cli import doctor as d

    cfg = load_config(ROOT / "src/flora_deploy/config/templates/balanced_mp4_dir.yaml")
    assert cfg.input.repo is None and not d.local_only(cfg)


def test_counts_and_hours_render_as_text():
    from flora_deploy.cli._app import counts_text, hours_text, per_stage_text
    from flora_deploy.local.runner import event_text

    assert counts_text({"success": 5}) == "5 success"
    assert counts_text({"failure": 1, "success": 4, "preempted": 0}) == "4 success, 1 failure"
    assert counts_text({}) == "none" and counts_text(None, "0") == "0"
    assert per_stage_text({"caption": 0, "segment": 0}) == "0"
    assert per_stage_text({"segment": 2, "caption": 1, "reward.dense": 0}) == \
        "caption 1, segment 2"
    assert [hours_text(h) for h in (0, 0.00001, 0.0072, 0.11, 1.0, 2.5, 12.34, 1234.5)] == \
        ["0", "<0.0001", "0.0072", "0.11", "1", "2.5", "12.3", "1,234"]
    assert event_text({"dead": [], "problems": [], "shards": 5}) == "shards=5"
    assert event_text({"dead": ["a", "b"], "by": {"x": 1}, "n": None}) == "by=x:1 dead=a,b"


def test_run_and_report_print_no_python_reprs(demo):
    run_out = demo[1]
    report = _in(demo, ["report"], columns="200")
    for where, out in (("flora run --local", run_out), ("flora report", report)):
        assert not re.search(r"\{'|\{\}|=\[\]", out), f"{where}:\n{out}"
    # the plain-words summary (Fix-DX3): counts read as text, zero counts left out
    assert re.search(r"workers: \d+ shard runs \(\d+ succeeded", run_out), run_out
    assert "errors: none" in report and "excused" not in report, report


def test_plan_shows_small_workloads_with_two_digits(tmp_path):
    cfg = _write(tmp_path, template_text("local_demo"))
    r = runner.invoke(app, ["plan", "-c", str(cfg), "--video-hours", "0.11", "--episodes", "5"],
                      env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    assert "0.11 video-hours" in r.output, r.output


def test_help_groups_commands_into_panels_in_workflow_order():
    from flora_deploy.cli._app import HELP_PANELS, command_name

    names = [command_name(i) for i in app.registered_commands]
    listed = [n for ns in HELP_PANELS.values() for n in ns]
    assert sorted(names) == sorted(listed), "every command belongs to exactly one panel"
    assert names == listed  # `--help` lists them in panel order
    assert all(i.rich_help_panel in HELP_PANELS for i in app.registered_commands)
    out = runner.invoke(app, ["--help"], env={"COLUMNS": "120"}).output
    pos = [out.index(f"─ {p} ") for p in HELP_PANELS]
    assert pos == sorted(pos), out
    assert out.index("─ Get started ") < out.index(" init ") < out.index("─ Run ")


def test_readme_test_command_matches_ci():
    """The README's contributor command is the one CI runs (`-n auto --dist loadfile`), with a
    timing that is not the old, unreachable ~30 s."""
    line = next(ln for ln in README.splitlines() if ln.startswith("uv run pytest"))
    assert "-n auto --dist loadfile" in line and "~30 s" not in line, line


# ---------------------------------------------------------------------------------------------
# Newcomer evaluation 2 (2026-10-23): the demo's plan, labels and notation


def test_plan_after_the_local_demo_finds_its_index(demo):
    """`flora plan` right after `flora run --local` reads the index the run wrote in the local
    store (no `flora index`, which would open the sink's Modal Volume)."""
    out = _in(demo, ["plan"], columns="200")
    assert "flora-local-store" in out and "summary.json" in out, out
    assert "no workload" not in out and "Recommended:" in out


def test_plan_without_a_workload_suggests_the_free_local_command(tmp_path):
    import os

    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert runner.invoke(app, ["init", "--template", "local_demo"]).exit_code == 0
        r = runner.invoke(app, ["plan"], env={"COLUMNS": "200"})
    finally:
        os.chdir(old)
    assert r.exit_code == 2
    msg = " ".join(r.output.split())
    assert "flora run --local --fake-models --sample-data" in msg, msg
    assert "flora index" not in msg  # it would create the sink's Modal Volume


def test_plan_labels_one_option_one_way(balanced_cfg):
    """Table, flags and JSON use the same fidelity words (item 4)."""
    out = _plan(balanced_cfg).output
    assert "partly certified" in out
    assert "balanced: uncertified" not in out and "uncertified (not every" not in out
    line = next(ln for ln in out.splitlines() if "balanced: partly certified (" in ln)
    assert "reference" in line and "not certified" in line, line
    d = json.loads(_plan(balanced_cfg, "--json").output)
    bal = next(o for o in d["options"] if o["preset"] == "balanced")
    assert bal["certification"] == "partly certified"
    assert not any(f.startswith("balanced: uncertified") for f in bal["flags"])


def test_plan_pools_say_containers_and_gpus(balanced_cfg):
    """`2 containers × 1 H200, 4 episodes each`, never `2xH200 x4 ep` (item 9)."""
    out = _plan(balanced_cfg, "--budget", "100").output
    pools = [ln for ln in out.splitlines() if ". pools: " in ln]
    assert pools and all("×" in ln and " each" in ln for ln in pools), pools
    assert not re.search(r"\d+x(H200|H100|L40S)", out), out
    assert re.search(r"segcap \d+ containers? × 1 H200, \d+ episodes? each", out), out


def test_sdk_plan_matches_the_cli(balanced_cfg):
    """Same options in the same order as `flora plan --json`, the same certificates applied,
    and the CLI's recommendation (item 10)."""
    from flora_deploy import sdk
    from flora_deploy.planner.options import recommended

    opts = sdk.plan(str(balanced_cfg), video_hours=1)
    d = json.loads(_plan(balanced_cfg, "--json").output)
    assert [o.name for o in opts] == [o["name"] for o in d["options"]]
    assert [o.cost_usd for o in opts] == [o["cost_usd"] for o in d["options"]]
    best = recommended(opts, certified={o.name for o in opts if o.certification == "certified"})
    assert opts.index(best) + 1 == d["recommended"]


def test_results_caption_counts_episodes_and_subtasks(demo):
    out = _in(demo, ["results", "caption"], columns="200")
    title = next(ln for ln in out.splitlines() if " caption: " in ln)
    assert re.search(r"caption: 6 episodes, \d+ subtasks", title), title
    assert "records" not in title


def test_balanced_template_idle_tail_matches_the_planner():
    from flora_deploy.planner.options import POOL_END_TAIL_S

    text = template_text("balanced_lerobot")
    assert f"about {POOL_END_TAIL_S:g} s after it runs out of work" in text
    assert "about 2 s after" not in text


def test_deploy_dry_run_explains_index_and_fits_80_columns(tmp_path):
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(template_text("balanced_lerobot"))
    args = ["deploy", "--dry-run", "-c", str(cfg), "--allow-uncertified",
            "--receipt", "captioner_bf16=" + "c" * 64, "--receipt", "tempseg_merged=" + "b" * 64]
    r = runner.invoke(app, args, env={"COLUMNS": "80"})
    assert r.exit_code == 0, r.output
    assert "index_run_shard" not in r.output and "defines no" not in r.output
    assert "flora-index" in " ".join(r.output.split())
    table = [ln for ln in r.output.splitlines() if "│" in ln]
    assert table and "…" not in "".join(table), r.output
    assert any("SegmentWorker" in ln and "49152/61440" in ln for ln in table), r.output
