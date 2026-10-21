"""``flora deploy`` (dry run, --emit, a stubbed deploy) and the emitted script, offline."""

from __future__ import annotations

import ast
import json
import sys
import types
from importlib import resources
from pathlib import Path

import modal
import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy.cli import app as cli
from flora_deploy.cli import deploy as deploy_cli
from flora_deploy.config.schema import parse_config
from flora_deploy.modal_app import factory as f
from flora_deploy.modal_app.emit import render_script, write_script

runner = CliRunner()
RECEIPT = "captioner_bf16=" + "0" * 64


class RewardWorker:
    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        return None  # pragma: no cover


@pytest.fixture
def fake_workers_app(monkeypatch):
    mod = types.ModuleType("flora_deploy.modal_app.workers_app")
    mod.RewardWorker = RewardWorker
    monkeypatch.setitem(sys.modules, "flora_deploy.modal_app.workers_app", mod)
    return mod


def template(name: str) -> dict:
    return yaml.safe_load((resources.files("flora_deploy.config") / "templates" / name)
                          .read_text())


def write_cfg(tmp_path: Path, name: str, **changes) -> Path:
    d = template(name)
    d.update(changes)
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def exact_plan(**kw):
    return f.plan_app(parse_config(template("exact_mp4_dir.yaml")),
                      derived_receipts={"captioner_bf16": "0" * 64}, **kw)


def _exec(src: str):
    ns: dict = {"__name__": "flora_emitted"}
    exec(compile(src, "emitted.py", "exec"), ns)  # noqa: S102 - our own generated code
    return ns


def _decorator_calls(src: str) -> dict[str, dict]:
    """Literal keyword arguments of every ``app.cls(...)`` / ``app.function(...)`` call, keyed
    by the object they decorate."""
    out: dict[str, dict] = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)):
            continue
        deco = node.func
        if not (isinstance(deco.func, ast.Attribute) and deco.func.attr in ("cls", "function")):
            continue
        target = node.args[0]
        while isinstance(target, ast.Call):
            target = target.args[0]
        kws = {}
        for k in deco.keywords:
            try:
                kws[k.arg] = ast.literal_eval(k.value)
            except ValueError:
                kws[k.arg] = ast.unparse(k.value)
        key = kws.get("name", target.id)
        out[key] = kws
    return out


def test_emitted_script_compiles_and_matches_the_factory():
    plan = exact_plan(reconcile_cron="on", store="s3://bkt/jobs")
    src = render_script(plan)
    compile(src, "emitted.py", "exec")
    ns = _exec(src)
    assert ns["PLAN"] == plan.to_json()
    emitted, built = ns["app"], f.build_app(plan)
    assert emitted.name == built.name == plan.app_name
    efns, ecls = f.registered(emitted)
    bfns, bcls = f.registered(built)
    assert set(efns) == set(bfns) and set(ecls) == set(bcls)
    for tag, fn in bfns.items():
        a, b = fn._spec, efns[tag]._spec
        assert (a.gpus, a.cpu, a.memory, str(a.scheduler_placement), sorted(a.volumes),
                len(a.secrets)) == (b.gpus, b.cpu, b.memory, str(b.scheduler_placement),
                                    sorted(b.volumes), len(b.secrets)), tag
    calls = _decorator_calls(src)
    assert set(calls) == {o.name for o in plan.objects}
    for o in plan.objects:
        kw = calls[o.name]
        for key in ("gpu", "min_containers", "max_containers", "scaledown_window",
                    "startup_timeout"):
            assert kw.get(key) == getattr(o, key), (o.name, key)
        assert kw["cpu"] == o.cpu and kw["timeout"] == o.timeout and kw["env"] == o.env
        assert kw["memory"] == o.memory
        assert ("nonpreemptible" in kw) == o.nonpreemptible
    assert "schedule=modal.Cron('*/10 * * * *')" in src


def test_emitted_script_inlines_no_secrets():
    src = render_script(exact_plan(store="s3://bkt/jobs", bucket_secret="my-creds"))
    assert "modal.Secret.from_name('my-creds')" in src
    assert "from_dict" not in src and "AWS_SECRET" not in src


def test_emitted_concurrency(fake_workers_app):
    d = template("rewards_only.yaml")
    d["resources"] = {"reward.terminal": {"gpu": "H100!", "max_inputs": 2},
                      "reward.dense": {"gpu": "H100!", "max_inputs": 2}}
    plan = f.plan_app(parse_config(d))
    src = render_script(plan)
    assert "modal.concurrent(max_inputs=2)(RewardWorker)" in src
    ns = _exec(src)
    assert set(f.registered(ns["app"])[1]) == {"RewardWorker"}


def test_write_script(tmp_path):
    out = tmp_path / "app.py"
    src = write_script(exact_plan(), out)
    assert out.read_text() == src


# ---------------------------------------------------------------------------------------------
# CLI


def test_cli_dry_run_prints_the_plan(tmp_path):
    cfg = write_cfg(tmp_path, "exact_mp4_dir.yaml")
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--receipt", RECEIPT, "--dry-run",
                              "--json"])
    assert res.exit_code == 0, res.output
    doc = json.loads(res.output[: res.output.rindex("}") + 1])
    assert doc["app_name"] == exact_plan().app_name
    assert doc["worker_classes"] == {"segcap": "ReferenceWorker"}
    assert "nothing deployed" in res.output


def test_cli_emit_writes_without_deploying(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_cli, "_deploy", lambda *a: pytest.fail("deployed"))
    cfg = write_cfg(tmp_path, "exact_mp4_dir.yaml")
    out = tmp_path / "emitted.py"
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--receipt", RECEIPT, "--emit", str(out)])
    assert res.exit_code == 0, res.output
    compile(out.read_text(), str(out), "exec")


def test_cli_deploy_records_the_deployment(tmp_path, monkeypatch):
    seen = {}

    def fake_deploy(app, plan):
        seen["app"] = app.name
        return "ap-123"

    def fake_record(plan, record):
        seen["record"] = record
        return "modal-volume://flora-results/x"

    monkeypatch.setattr(deploy_cli, "_deploy", fake_deploy)
    monkeypatch.setattr(deploy_cli, "_record", fake_record)
    cfg = write_cfg(tmp_path, "exact_mp4_dir.yaml")
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--receipt", RECEIPT])
    assert res.exit_code == 0, res.output
    plan = exact_plan()
    rec = seen["record"]
    assert seen["app"] == plan.app_name == rec["app_name"]
    assert rec["app_id"] == "ap-123" and rec["stage_hashes"] == plan.stage_hashes
    assert rec["image_fingerprints"] == plan.image_fingerprints
    assert rec["worker_classes"] == {"segcap": "ReferenceWorker"}
    assert rec["functions"] == ["orchestrator_run", "reconcile"]
    assert f.deployment_path(plan.app_name) == f"flora/deployments/{plan.app_name}.json"


def test_cli_record_failure_is_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_cli, "_deploy", lambda app, plan: "ap-1")

    def boom(plan, record):
        raise OSError("no bucket")

    monkeypatch.setattr(deploy_cli, "_record", boom)
    cfg = write_cfg(tmp_path, "exact_mp4_dir.yaml")
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--receipt", RECEIPT])
    assert res.exit_code == 1 and "could not record" in res.output


@pytest.mark.parametrize("args, msg", [
    ([], "captioner_bf16"),  # missing derived-weight receipt
    (["--receipt", "oops"], "NAME=HASH"),
    (["--receipt", RECEIPT, "--store", "file:///tmp/x"], "local store"),
    (["--receipt", RECEIPT, "--determinism", "fast"], "determinism"),
])
def test_cli_refusals(tmp_path, args, msg):
    cfg = write_cfg(tmp_path, "exact_mp4_dir.yaml")
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--dry-run", *args])
    assert res.exit_code == 2, res.output
    assert msg in res.output.replace("\n", "")


def test_cli_missing_worker_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "flora_deploy.modal_app.workers_app", None)
    cfg = write_cfg(tmp_path, "rewards_only.yaml")
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--dry-run"])
    assert res.exit_code == 2 and "reward.terminal" in res.output
