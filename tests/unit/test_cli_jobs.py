"""`flora run` (Modal path) / `status` / `resume` / `cancel` / `report` / `logs` through the CLI,
with the fake backend installed as the SDK default (``sdk.set_backend``). Exit codes follow
``flora run --local`` / ``flora compose``: 0 done, 1 failed, 3 completed_with_errors,
4 budget_stopped, 5 cancelled, 2 refusals."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from flora_deploy import sdk
from flora_deploy.cli import app
from tests.fakes.flora_backend import JOB, RECEIPTS, DeployedApp

cli = CliRunner()
RECEIPT = [f"--receipt=captioner_bf16={RECEIPTS['captioner_bf16']}"]


@pytest.fixture
def deployed(tmp_path, request):
    kw = getattr(request, "param", {})
    a = DeployedApp(tmp_path, **kw)
    sdk.set_backend(a.backend)
    yield a
    sdk.set_backend(None)


def run(a: DeployedApp, *args: str):
    return cli.invoke(app, ["run", "--config", str(a.config_path), *RECEIPT, "--poll", "0",
                            *args])


def job_cmd(a: DeployedApp, cmd: str, *args: str):
    return cli.invoke(app, [cmd, JOB, "--config", str(a.config_path), *args])


def test_run_streams_to_done_then_status_report_logs(deployed):
    out = run(deployed)
    assert out.exit_code == 0, out.output
    assert "done" in out.output and "composed: complete" in out.output
    st = job_cmd(deployed, "status")
    assert st.exit_code == 0 and "done" in st.output and "segcap: 3/3 shards done" in st.output
    js = json.loads(job_cmd(deployed, "status", "--json").output)
    assert js["state"] == "done" and js["composed"]
    rep = job_cmd(deployed, "report")
    assert rep.exit_code == 0, rep.output
    assert "per video-hour" in rep.output and "caption: caption.jsonl" in rep.output
    assert "certification: none (fake models)" in rep.output
    rj = json.loads(job_cmd(deployed, "report", "--json").output)
    assert rj["usd_per_video_hour"] > 0
    lg = job_cmd(deployed, "logs", "--kind", "orchestrator")
    assert lg.exit_code == 0 and "line 1" in lg.output
    assert job_cmd(deployed, "logs", "--kind", "caption").exit_code == 2
    # a finished job: resume and cancel refuse, a resubmit refuses
    assert job_cmd(deployed, "resume").exit_code == 2
    assert job_cmd(deployed, "cancel", "--yes").exit_code == 2
    again = run(deployed)
    assert again.exit_code == 2 and "already submitted" in again.output


@pytest.mark.parametrize("deployed", [{"run": False}], indirect=True)
def test_detach_status_cancel(deployed):
    out = run(deployed, "--detach")
    assert out.exit_code == 0, out.output
    assert "detached" in out.output and "planned" in out.output
    st = job_cmd(deployed, "status")
    assert st.exit_code == 0 and "no orchestrator holds the lease" in st.output
    assert job_cmd(deployed, "cancel").exit_code == 2  # declined (no input): a refusal
    c = job_cmd(deployed, "cancel", "--yes")
    assert c.exit_code == 0, c.output
    assert job_cmd(deployed, "status").exit_code == 5
    assert job_cmd(deployed, "resume").exit_code == 2


@pytest.mark.parametrize("deployed", [{"faults": "oom:ep001@segcap"}], indirect=True)
def test_completed_with_errors_exit_code(deployed):
    out = run(deployed)
    assert out.exit_code == 3, out.output
    assert job_cmd(deployed, "status").exit_code == 3
    rep = job_cmd(deployed, "report")
    assert rep.exit_code == 3 and "episodes set aside 1" in rep.output


def test_budget_stop_then_resume(deployed):
    out = run(deployed, "--budget", "0.01")
    assert out.exit_code == 4, out.output
    assert "flora resume" in out.output
    assert job_cmd(deployed, "resume").exit_code == 2  # still over budget
    r = job_cmd(deployed, "resume", "--budget-usd", "50")
    assert r.exit_code == 0, r.output
    assert "with budget $50.00" in r.output
    assert job_cmd(deployed, "status").exit_code == 0


@pytest.mark.parametrize("deployed", [{"deploy": False}], indirect=True)
def test_run_refuses_an_undeployed_config(deployed):
    out = run(deployed)
    assert out.exit_code == 2 and "not deployed" in out.output
    assert deployed.backend.spawned == []


def test_run_refusals(deployed):
    no_receipt = cli.invoke(app, ["run", "--config", str(deployed.config_path)])
    assert no_receipt.exit_code == 2 and "captioner_bf16" in no_receipt.output
    for flag in ("--fake-models", "--sample-data"):
        o = run(deployed, flag)
        assert o.exit_code == 2 and "--local" in o.output
    assert run(deployed, "--resume").exit_code == 2
    assert run(deployed, "--receipt", "bad").exit_code == 2
    missing = cli.invoke(app, ["run", "--config", str(deployed.tmp / "nope.yaml")])
    assert missing.exit_code == 2
    assert deployed.backend.spawned == []


def test_job_commands_refuse_unknown_jobs(deployed):
    for cmd in ("status", "report", "resume", "logs"):
        o = cli.invoke(app, [cmd, "nobody", "--config", str(deployed.config_path)])
        assert o.exit_code == 2 and "no run spec" in o.output
    o = cli.invoke(app, ["status", JOB, "--config", str(deployed.tmp / "missing.yaml")])
    assert o.exit_code == 2 and "--store" in o.output


def test_store_option_finds_the_job(deployed):
    assert run(deployed, "--detach").exit_code == 0
    o = cli.invoke(app, ["status", JOB, "--store", deployed.store_uri, "--config",
                         str(deployed.tmp / "absent.yaml")])
    assert o.exit_code == 0, o.output


def test_local_mode_still_dispatches_to_run_local(tmp_path):
    from pathlib import Path

    tpl = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
    (tmp_path / "flora.yaml").write_text(tpl.read_text())
    o = cli.invoke(app, ["run", "--local", "--fake-models", "--sample-data",
                         "--config", str(tmp_path / "flora.yaml")])
    assert o.exit_code == 0 and "DONE" in o.output, o.output
    o = cli.invoke(app, ["run", "--local", "--fake-models", "--json",
                         "--config", str(tmp_path / "flora.yaml")])
    assert o.exit_code == 0 and json.loads(o.output)["state"] == "done"
    assert cli.invoke(app, ["run", "--local", "--config",
                            str(tmp_path / "flora.yaml")]).exit_code == 2
