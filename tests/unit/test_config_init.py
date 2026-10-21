"""`flora init` and the packaged templates (``flora_deploy/config/templates``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import TEMPLATES, template_text, templates_dir
from flora_deploy.config.schema import loads_config, to_run_settings

REPO = Path(__file__).resolve().parents[2]

runner = CliRunner()
RECEIPTS = {"captioner_bf16": "c" * 64, "tempseg_merged": "t" * 64}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_validate_and_map(name):
    cfg = loads_config(template_text(name))
    rs = to_run_settings(cfg, "job", derived_receipts=RECEIPTS)
    assert rs.pools and len(rs.config_hash) == 64


def test_every_packaged_yaml_is_a_template():
    names = {p.name for p in templates_dir().iterdir() if p.name.endswith(".yaml")}
    assert names == set(TEMPLATES.values())


def test_templates_ship_in_the_wheel(tmp_path):
    """`flora init` from a wheel (no source checkout): the templates are package data."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    out = subprocess.run([uv, "build", "--wheel", "--out-dir", str(tmp_path), str(REPO)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        assert {f"flora_deploy/config/templates/{t}" for t in TEMPLATES.values()} <= names
        site = tmp_path / "site"
        zf.extractall(site)
    code = ("import sys; from flora_deploy.cli.init import template_text, templates_dir; "
            "assert str(templates_dir()).startswith(sys.argv[1]), templates_dir(); "
            "print(len(template_text('exact_lerobot')))")
    env = {**os.environ, "PYTHONPATH": str(site)}
    run = subprocess.run([sys.executable, "-c", code, str(site)], capture_output=True, text=True,
                         env=env, cwd=tmp_path, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert int(run.stdout) == len(template_text("exact_lerobot"))


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_init_writes_valid_file(tmp_path, name):
    out = tmp_path / "sub" / "flora.yaml"
    r = runner.invoke(app, ["init", str(out), "--template", name])
    assert r.exit_code == 0, r.output
    assert out.read_text() == template_text(name)
    assert "license_ack: research-only" in out.read_text()
    loads_config(out.read_text())


def test_init_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "flora.yaml"
    out.write_text("mine")
    r = runner.invoke(app, ["init", str(out)])
    assert r.exit_code == 1 and "--force" in r.output
    assert out.read_text() == "mine"
    r = runner.invoke(app, ["init", str(out), "--force"])
    assert r.exit_code == 0, r.output
    loads_config(out.read_text())


def test_init_unknown_template(tmp_path):
    r = runner.invoke(app, ["init", str(tmp_path / "f.yaml"), "-t", "nope"])
    assert r.exit_code == 2 and "unknown template" in r.output
    assert not (tmp_path / "f.yaml").exists()
