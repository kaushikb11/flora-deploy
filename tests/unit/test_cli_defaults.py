"""Job-scoped commands default the job and store from ``flora.yaml`` (``cli._jobref``).

Covers: config present or absent; ``--store`` / ``FLORA_STORE`` overriding the sink; real vs
fake-models runs (the fake one is used only when there is no real one, with one stand-in line;
both present: the real one, mentioning the fake); ``flora results <stage>`` (positional, hidden
``--stage`` alias); ``flora plan`` reading the summary ``flora index`` wrote; ``flora init``'s
next steps; and ``flora deploy && flora run`` with zero flags on a prepared config.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy import sdk
from flora_deploy.cli import app
from flora_deploy.jobs.stores import LocalStore
from tests.fakes.flora_backend import JOB, RECEIPTS, TEMPLATE, DeployedApp, FakeBackend

FAKE = f"{JOB}.fake"
RECEIPT = [f"--receipt=captioner_bf16={RECEIPTS['captioner_bf16']}"]
cli = CliRunner()


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    a = DeployedApp(tmp_path / "proj")
    sdk.set_backend(a.backend)
    monkeypatch.chdir(a.tmp)  # flora.yaml in the current directory, as a user has it
    monkeypatch.delenv("FLORA_STORE", raising=False)
    yield a
    sdk.set_backend(None)


def _real_run(a: DeployedApp) -> None:
    out = cli.invoke(app, ["run", *RECEIPT, "--poll", "0"])
    assert out.exit_code == 0, out.output


def _fake_pointer(store_dir: Path, job_id: str = FAKE) -> None:
    """A fake-models run's pointer (what `flora run --local` writes), without running one."""
    sdk.write_run_spec(LocalStore(store_dir), {
        "schema": sdk.RUN_SPEC_SCHEMA, "job_id": job_id, "config_hash": "f" * 16,
        "app_name": "local", "fake_models": True, "local": {"fake_models": True}})


@pytest.fixture(autouse=True)
def offline(tmp_path):
    """Never the Modal client: remote stores (the template's volume:// sink) are empty fakes."""
    sdk.set_backend(FakeBackend(tmp_path / "remote-stores"))
    yield
    sdk.set_backend(None)


def _local_demo(tmp: Path, monkeypatch) -> Path:
    """A real `flora run --local --fake-models --sample-data` in ``tmp`` (the quickstart)."""
    tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp)
    monkeypatch.delenv("FLORA_STORE", raising=False)
    assert cli.invoke(app, ["init", "--template", "local_demo"]).exit_code == 0
    out = cli.invoke(app, ["run", "--local", "--fake-models", "--sample-data"])
    assert out.exit_code == 0, out.output
    assert "read results with `flora results caption`" in out.output
    return tmp / "flora-local-store"


# -- config present: job id = project, store = output.sink ------------------------------------


def test_job_commands_default_from_the_config(deployed):
    _real_run(deployed)
    for argv in (["status"], ["report"], ["logs", "--kind", "orchestrator"]):
        out = cli.invoke(app, argv)
        assert out.exit_code == 0, (argv, out.output)
        assert "stand-in" not in out.output
    assert JOB in cli.invoke(app, ["status"]).output
    js = json.loads(cli.invoke(app, ["status", "--json"]).output)
    assert js["job_id"] == JOB and js["state"] == "done"
    res = cli.invoke(app, ["results", "caption", "--format", "jsonl"])
    assert res.exit_code == 0, res.output
    assert {json.loads(x)["stage"] for x in res.output.splitlines()} == {"caption"}
    # a finished job: cancel / resume refuse (they found it: not "no run spec")
    for argv in (["cancel", "--yes"], ["resume"]):
        out = cli.invoke(app, argv)
        assert out.exit_code == 2 and "no run spec" not in out.output, out.output


def test_explicit_job_and_config_still_work(deployed):
    _real_run(deployed)
    other = deployed.tmp / "elsewhere"
    other.mkdir()
    cfg = str(deployed.config_path)
    assert cli.invoke(app, ["status", JOB, "--config", cfg]).exit_code == 0
    out = cli.invoke(app, ["results", "--job", JOB, "--stage", "caption", "--config", cfg])
    assert out.exit_code == 0, out.output  # the hidden --stage alias


# -- config absent ----------------------------------------------------------------------------


def test_no_config_needs_a_job_id_and_a_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLORA_STORE", raising=False)
    for argv in (["status"], ["report"], ["results", "caption"]):
        out = cli.invoke(app, argv)
        assert out.exit_code == 2 and "no job id" in out.output, (argv, out.output)
    out = cli.invoke(app, ["status", JOB])
    assert out.exit_code == 2 and "--store" in out.output


def test_no_config_with_store_and_job(tmp_path, monkeypatch):
    store = _local_demo(tmp_path / "demo", monkeypatch)
    monkeypatch.chdir(tmp_path)  # no flora.yaml here
    out = cli.invoke(app, ["results", "caption", "--job", FAKE, "--store", str(store)])
    assert out.exit_code == 0, out.output
    assert "stand-in" not in out.output  # asked for by name: no note
    monkeypatch.setenv("FLORA_STORE", str(store))
    assert cli.invoke(app, ["status", FAKE]).exit_code == 0


# -- --store / FLORA_STORE override the sink --------------------------------------------------


def test_store_option_overrides_the_sink(deployed, tmp_path):
    _real_run(deployed)
    elsewhere = tmp_path / "other-store"
    _fake_pointer(elsewhere)
    # only that store is searched: the real run in the sink is not seen, the fake one is
    out = cli.invoke(app, ["status", "--store", str(elsewhere), "--json"])
    assert FAKE in out.output and "stand-in" in out.output
    assert '"job_id": "local-demo"' not in out.output
    empty = tmp_path / "empty-store"
    empty.mkdir()
    out = cli.invoke(app, ["status", "--store", str(empty)])
    assert out.exit_code == 2 and "no run spec" in out.output


def test_flora_store_env_overrides_the_sink(deployed, tmp_path, monkeypatch):
    empty = tmp_path / "empty-store"
    empty.mkdir()
    monkeypatch.setenv("FLORA_STORE", str(empty))
    out = cli.invoke(app, ["status"])
    assert out.exit_code == 2 and "no run spec" in out.output
    monkeypatch.delenv("FLORA_STORE")
    _real_run(deployed)
    assert cli.invoke(app, ["status"]).exit_code == 0


# -- real vs fake-models runs -----------------------------------------------------------------


def test_quickstart_reads_the_local_fake_run(tmp_path, monkeypatch):
    """`flora run --local` then `flora results caption`: no real run anywhere (the sink is a
    Modal Volume this test cannot reach), so the local fake-models run, with one stand-in
    line; status and report the same."""
    _local_demo(tmp_path, monkeypatch)
    for argv in (["results", "caption"], ["status"], ["report"]):
        out = cli.invoke(app, argv)
        assert out.exit_code == 0, (argv, out.output)
        lines = [ln for ln in out.output.splitlines() if "stand-in models" in ln]
        assert len(lines) == 1 and FAKE in lines[0], (argv, out.output)
    out = cli.invoke(app, ["results", "caption", "--format", "jsonl"])
    recs = [json.loads(x) for x in out.output.splitlines() if x.startswith("{")]
    assert recs and all(r["provenance"]["certificate"] == "local-fake-models" for r in recs)


def test_explicit_real_job_with_only_a_fake_run_is_refused(tmp_path, monkeypatch):
    _local_demo(tmp_path, monkeypatch)
    out = cli.invoke(app, ["results", "caption", "--job", JOB, "--store", "flora-local-store"])
    assert out.exit_code == 2 and FAKE in out.output


def test_real_run_wins_over_a_fake_one(deployed):
    _real_run(deployed)
    _fake_pointer(deployed.tmp / "flora-local-store")
    out = cli.invoke(app, ["status"])
    assert out.exit_code == 0, out.output
    assert "stand-in" not in out.output
    assert f"a fake-models run of this job also exists ({FAKE}" in out.output
    js = cli.invoke(app, ["report", "--json"])
    assert json.loads(js.stdout)["state"] == "done"  # the note goes to stderr
    res = cli.invoke(app, ["results", "caption", "--format", "jsonl"])
    recs = [json.loads(x) for x in res.stdout.splitlines()]
    assert recs and all(r["audit_ref"].startswith(f"flora/{JOB}/") for r in recs)


def test_fake_run_in_the_sink_is_found_too(deployed):
    """A Modal fake-models run (`flora run --fake-models`) lives in the sink itself."""
    _fake_pointer(deployed.store.root)
    out = cli.invoke(app, ["status", "--json"])
    assert "stand-in models" in out.output and FAKE in out.output


# -- flora results <stage> --------------------------------------------------------------------


def test_results_stage_argument(deployed):
    _real_run(deployed)
    assert cli.invoke(app, ["results", "--stage", "caption"]).exit_code == 0
    assert cli.invoke(app, ["results", "caption", "--stage", "caption"]).exit_code == 0
    both = cli.invoke(app, ["results", "caption", "--stage", "segment"])
    assert both.exit_code == 2 and "two stages" in both.output
    none = cli.invoke(app, ["results"])
    assert none.exit_code == 2 and "which stage" in none.output
    bad = cli.invoke(app, ["results", "nope"])
    assert bad.exit_code == 2 and "unknown stage" in bad.output
    helptext = cli.invoke(app, ["results", "--help"]).output
    assert "RESULTS [OPTIONS] [STAGE]" in helptext.upper() and "--stage" not in helptext


# -- flora plan reads what flora index wrote --------------------------------------------------


def test_plan_reads_the_index_summary_by_default(tmp_path, monkeypatch):
    from flora_deploy.cli.index import default_index_summary
    from flora_deploy.local import sample_dataset

    monkeypatch.chdir(tmp_path)
    shutil.copy(TEMPLATE, tmp_path / "flora.yaml")
    sample_dataset(tmp_path / "videos")
    none = cli.invoke(app, ["plan"])
    # a local dataset: the free local run, not `flora index` (it opens the sink's Volume)
    assert none.exit_code == 2 and "`flora run --local --fake-models`" in " ".join(
        none.output.split())
    idx = cli.invoke(app, ["index", "--local", "--store", str(tmp_path / "store")])
    assert idx.exit_code == 0, idx.output
    assert default_index_summary(Path("flora.yaml")) == Path("summary.json")
    assert (tmp_path / "summary.json").is_file() and "next: flora plan\n" in idx.output
    out = cli.invoke(app, ["plan", "--json"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["options"]


def test_index_and_plan_agree_next_to_another_config(tmp_path, monkeypatch):
    from flora_deploy.cli.index import default_index_summary

    sub = tmp_path / "cfg"
    sub.mkdir()
    assert default_index_summary(sub / "flora.yaml") == sub / "summary.json"


# -- flora init -------------------------------------------------------------------------------


def test_init_default_template_names_the_keys_to_edit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = cli.invoke(app, ["init"])
    assert out.exit_code == 0, out.output
    assert "template balanced_lerobot" in out.output
    for key in ("input.repo", "input.revision", "pipeline.embodiment", "output.sink"):
        assert key in out.output
    assert "flora doctor" in out.output


def test_every_template_has_next_steps_naming_real_keys():
    import yaml

    from flora_deploy.cli.init import NEXT_STEPS, TEMPLATES, template_text

    assert set(NEXT_STEPS) == set(TEMPLATES)
    for name, (keys, _) in NEXT_STEPS.items():
        assert len(keys) <= 4
        data = yaml.safe_load(template_text(name))
        for key in keys:
            node = data
            for part in key.split("."):
                assert isinstance(node, dict) and part in node, f"{name}: no key {key}"
                node = node[part]


# -- flora deploy && flora run, zero flags ----------------------------------------------------


def test_deploy_then_run_with_zero_flags_on_a_prepared_config(tmp_path, monkeypatch):
    """`flora prepare` records the derived-weight receipts in flora.yaml; after that `flora
    deploy && flora run` needs no flags (Modal faked: the deploy seam and the record writer)."""
    from flora_deploy.cli import deploy as deploy_mod
    from flora_deploy.modal_app.factory import deployment_path

    a = DeployedApp(tmp_path / "proj", deploy=False)
    with a.config_path.open("a") as f:  # what `flora prepare` records
        f.write("weights:\n  derived:\n" + "".join(f"    {k}: '{v}'\n"
                                                     for k, v in RECEIPTS.items()))
    sdk.set_backend(a.backend)
    monkeypatch.chdir(a.tmp)
    monkeypatch.delenv("FLORA_STORE", raising=False)
    monkeypatch.setattr(deploy_mod, "_deploy", lambda modal_app, plan: "ap-fake")
    monkeypatch.setattr(deploy_mod, "_ensure_volume", lambda name, env: None)  # no Modal

    def _record(plan, rec):
        rel = deployment_path(plan.app_name)
        a.backend.store(plan.store.uri).put_bytes(rel, json.dumps(rec).encode())
        return rel

    monkeypatch.setattr(deploy_mod, "_record", _record)
    import flora_deploy.modal_app.factory as factory

    monkeypatch.setattr(factory, "materialize", lambda plan: object())
    try:
        dep = cli.invoke(app, ["deploy"])
        assert dep.exit_code == 0, dep.output
        run = cli.invoke(app, ["run"])
        assert run.exit_code == 0, run.output
        assert "done" in run.output
        assert cli.invoke(app, ["status"]).exit_code == 0
        assert cli.invoke(app, ["results", "caption"]).exit_code == 0
    finally:
        sdk.set_backend(None)
