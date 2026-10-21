"""`flora init` and the examples/ templates."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import TEMPLATES, examples_dir, template_text
from flora_deploy.config.schema import load_config, to_run_settings

runner = CliRunner()
RECEIPTS = {"captioner_bf16": "c" * 64, "tempseg_merged": "t" * 64}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_validate_and_map(name):
    cfg = load_config(examples_dir() / TEMPLATES[name])
    rs = to_run_settings(cfg, "job", derived_receipts=RECEIPTS)
    assert rs.pools and len(rs.config_hash) == 64


def test_every_example_is_a_template():
    assert {p.name for p in examples_dir().glob("*.yaml")} == set(TEMPLATES.values())


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_init_writes_valid_file(tmp_path, name):
    out = tmp_path / "sub" / "flora.yaml"
    r = runner.invoke(app, ["init", str(out), "--template", name])
    assert r.exit_code == 0, r.output
    assert out.read_text() == template_text(name)
    assert "license_ack: research-only" in out.read_text()
    load_config(out)


def test_init_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "flora.yaml"
    out.write_text("mine")
    r = runner.invoke(app, ["init", str(out)])
    assert r.exit_code == 1 and "--force" in r.output
    assert out.read_text() == "mine"
    r = runner.invoke(app, ["init", str(out), "--force"])
    assert r.exit_code == 0, r.output
    load_config(out)


def test_init_unknown_template(tmp_path):
    r = runner.invoke(app, ["init", str(tmp_path / "f.yaml"), "-t", "nope"])
    assert r.exit_code == 2 and "unknown template" in r.output
    assert not (tmp_path / "f.yaml").exists()
