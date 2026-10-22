"""A fresh Modal workspace: the workers mount ``flora-cache`` read-only, which cannot create it,
so ``flora deploy`` creates it client-side (version 2, the config's environment) first, and
``flora doctor`` checks it. No Modal calls: every seam is faked."""

from __future__ import annotations

import sys
import types
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy.cli import app as cli
from flora_deploy.cli import deploy as deploy_cli
from flora_deploy.cli import doctor as d
from flora_deploy.config.schema import parse_config
from flora_deploy.layout import CACHE_MOUNT, CACHE_VOLUME
from flora_deploy.modal_app import factory as f

runner = CliRunner()
BAL = ["--receipt", "captioner_bf16=" + "0" * 64, "--receipt", "tempseg_merged=" + "1" * 64]


def balanced_dict(**changes) -> dict:
    text = (resources.files("flora_deploy.config") / "templates" / "exact_mp4_dir.yaml").read_text()
    d_ = yaml.safe_load(text)
    d_.update(preset="balanced", allow_uncertified=True, resources={})
    d_["modal"]["max_gpus"] = 4
    d_["modal"]["environment"] = "fresh-ws"
    d_.update(changes)
    return d_


def write(tmp_path: Path, d_: dict) -> Path:
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(d_))
    return p


@pytest.fixture
def seams(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(deploy_cli, "_ensure_volume",
                        lambda name, env: calls.append(("ensure", name, env)))
    monkeypatch.setattr(f, "materialize", lambda plan: SimpleNamespace(name=plan.app_name))
    monkeypatch.setattr(deploy_cli, "_deploy",
                        lambda app, plan: calls.append(("deploy", plan.app_name)) or "ap-1")
    monkeypatch.setattr(deploy_cli, "_record", lambda plan, rec: "store://x")
    return calls


def test_balanced_plan_mounts_the_cache_read_only_without_creating_it(tempseg_pinned):
    """The precondition this fix is for (and the plan stays exactly as it was)."""
    plan = f.plan_app(parse_config(balanced_dict()),
                      derived_receipts={"captioner_bf16": "0" * 64, "tempseg_merged": "1" * 64})
    m = plan.object("SegmentWorker").volumes[str(CACHE_MOUNT)]
    assert m.name == CACHE_VOLUME and m.read_only and not m.create
    assert deploy_cli._cache_volumes(plan) == [CACHE_VOLUME]


def test_deploy_creates_the_cache_volume_first(tmp_path, seams, tempseg_pinned):
    res = runner.invoke(cli, ["deploy", "-c", str(write(tmp_path, balanced_dict())), *BAL])
    assert res.exit_code == 0, res.output
    assert seams[0] == ("ensure", CACHE_VOLUME, "fresh-ws")  # before the deploy
    assert seams[1][0] == "deploy" and len(seams) == 2


def test_dry_run_and_emit_create_nothing(tmp_path, seams, tempseg_pinned):
    cfg = str(write(tmp_path, balanced_dict()))
    assert runner.invoke(cli, ["deploy", "-c", cfg, *BAL, "--dry-run"]).exit_code == 0
    assert runner.invoke(cli, ["deploy", "-c", cfg, *BAL, "--emit",
                               str(tmp_path / "out.py")]).exit_code == 0
    assert seams == []


def test_a_failed_create_is_a_refusal_with_the_fix(tmp_path, monkeypatch, seams,
                                                   tempseg_pinned):
    def boom(name, env):
        raise RuntimeError("permission denied")

    monkeypatch.setattr(deploy_cli, "_ensure_volume", boom)
    res = runner.invoke(cli, ["deploy", "-c", str(write(tmp_path, balanced_dict())), *BAL])
    out = " ".join(res.output.split())
    assert res.exit_code == 2, res.output
    assert "modal volume create <name> --version 2 --env fresh-ws" in out and CACHE_VOLUME in out
    assert seams == []  # nothing deployed


def test_exact_mounts_no_cache_and_creates_nothing():
    text = (resources.files("flora_deploy.config") / "templates" / "exact_mp4_dir.yaml").read_text()
    plan = f.plan_app(parse_config(yaml.safe_load(text)),
                      derived_receipts={"captioner_bf16": "0" * 64})
    assert deploy_cli._cache_volumes(plan) == []


def test_real_ensure_volume_creates_version_2_in_the_environment(monkeypatch):
    seen = {}

    class Vol:
        def hydrate(self):
            seen["hydrated"] = True
            return self

    def from_name(name, **kw):
        seen.update(name=name, **kw)
        return Vol()

    fake = types.SimpleNamespace(Volume=types.SimpleNamespace(from_name=from_name))
    monkeypatch.setitem(sys.modules, "modal", fake)
    deploy_cli._ensure_volume(CACHE_VOLUME, "fresh-ws")
    assert seen == {"name": CACHE_VOLUME, "version": 2, "create_if_missing": True,
                    "environment_name": "fresh-ws", "hydrated": True}


# ---------------------------------------------------------------------------------------------
# doctor


class Probes:
    def __init__(self, volume: str):
        self.volume = volume
        self.asked: list[tuple[str, str]] = []

    def modal_volume(self, name, environment):
        self.asked.append((name, environment))
        return self.volume


@pytest.mark.parametrize("outcome, status", [("ok", "pass"), ("missing", "warn"),
                                             ("error:boom", "warn")])
def test_doctor_checks_the_cache_volume(outcome, status):
    cfg = parse_config(balanced_dict())
    p = Probes(outcome)
    (c,) = d.check_cache(p, cfg)
    assert c.name == "cache.volume" and c.status == status
    assert p.asked == [(CACHE_VOLUME, "fresh-ws")]
    if outcome != "ok":
        assert f"modal volume create {CACHE_VOLUME} --version 2 --env fresh-ws" in c.fix
    if outcome == "missing":
        assert "flora deploy" in c.message


def test_doctor_skips_the_cache_check_when_no_worker_mounts_it():
    text = (resources.files("flora_deploy.config") / "templates" / "exact_mp4_dir.yaml").read_text()
    assert d.check_cache(Probes("missing"), parse_config(yaml.safe_load(text))) == []
