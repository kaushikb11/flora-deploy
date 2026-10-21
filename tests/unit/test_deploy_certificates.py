"""``flora deploy`` shows each stage's Gate certificate status and why (``cli._certs``)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from typer.testing import CliRunner

from flora_deploy.cli import _certs
from flora_deploy.cli import app as cli
from flora_deploy.config.schema import parse_config
from flora_deploy.validate import certkey

runner = CliRunner()
RECEIPT = "captioner_bf16=" + "0" * 64


def template(name: str) -> dict:
    return yaml.safe_load((resources.files("flora_deploy.config") / "templates" / name)
                          .read_text())


def write_cfg(tmp_path: Path, name: str) -> Path:
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(template(name)))
    return p


def test_exact_dry_run_prints_reference(tmp_path):
    res = runner.invoke(cli, ["deploy", "-c", str(write_cfg(tmp_path, "exact_mp4_dir.yaml")),
                              "--receipt", RECEIPT, "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "certificate segcap on H200" in res.output and "reference" in res.output
    assert "reference stack" in res.output


def test_report_statuses_and_reasons(monkeypatch):
    cfg = parse_config(template("balanced_mp4_dir.yaml"))
    certs = [
        {"id": "cert_p", "preset": "balanced", "kind": "caption", "gpu": "L40S",
         "verdict": "provisional", "reasons": ["no noise run"], "covered_pools": ["yam"],
         "generated_at": "2026-10-20T00:00:00Z"},
        {"id": "cert_old", "preset": "balanced", "kind": "reward.dense", "gpu": "L40S",
         "verdict": "certified", "covered_pools": [cfg.pipeline.embodiment]},
    ]
    seen = {}

    def status(preset, pools, **kw):
        seen.update(kw, preset=preset)
        return {"segment": {"gpu": "L40S", "status": "unverified_weights", "certificate": None},
                "caption": {"gpu": "L40S", "status": "provisional", "certificate": "cert_p"},
                "reward.dense": {"gpu": "L40S", "status": "none", "certificate": None},
                "reward.terminal": {"gpu": "H100", "status": "none", "certificate": None}}

    monkeypatch.setattr(certkey, "status", status)
    monkeypatch.setattr(certkey, "load_certificates", lambda cert_dir=None: certs)
    rep = _certs.certificate_report(cfg, {"segment": "L40S", "caption": "L40S",
                                          "reward.dense": "L40S", "reward.terminal": "H100"},
                                    derived_content={"captioner_bf16": "c" * 64})
    assert seen["derived_content"] == {"captioner_bf16": "c" * 64}
    assert seen["embodiment"] == cfg.pipeline.embodiment and seen["preset"] == "balanced"
    assert rep["segment"]["status"] == "unverified_weights"
    assert "flora prepare" in rep["segment"]["why"]
    assert rep["caption"]["why"].startswith("cert_p") and "no noise run" in rep["caption"]["why"]
    assert "does not unlock" in rep["caption"]["why"]
    assert "none for this stack" in rep["reward.dense"]["why"]
    assert rep["reward.terminal"]["why"] == "no certificate for balanced reward.terminal on H100"


def test_deploy_json_carries_certificates(tmp_path):
    import json

    res = runner.invoke(cli, ["deploy", "-c", str(write_cfg(tmp_path, "exact_mp4_dir.yaml")),
                              "--receipt", RECEIPT, "--dry-run", "--json"])
    assert res.exit_code == 0, res.output
    doc = json.loads(res.output[: res.output.rindex("}") + 1])
    assert doc["certificates"]["segcap"]["status"] == "reference"


def test_derived_content_from_pins_fails_closed(monkeypatch, tmp_path):
    from dataclasses import replace

    from flora_deploy.config import receipts as R
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

    d = template("balanced_mp4_dir.yaml")
    d["weights"] = {"derived": {"captioner_bf16": "a" * 64}}
    cfg = parse_config(d, allow_uncertified=True)
    pin = DERIVED_PINS_BY_KEY["captioner-bf16"]
    got = R.derived_content(cfg)
    assert got.get("captioner_bf16") == pin.content_sha256  # recorded and pinned
    assert "tempseg_merged" not in got  # no receipt: left out, so it matches nothing
    unpinned = replace(DERIVED_PINS_BY_KEY["tempseg-merged"], content_sha256=None)
    monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", unpinned)
    assert "tempseg_merged" not in R.derived_content(cfg, {"tempseg_merged": "b" * 64})
    pinned = replace(unpinned, content_sha256="c" * 64)
    monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", pinned)
    assert R.derived_content(cfg, {"tempseg_merged": "b" * 64})["tempseg_merged"] == "c" * 64
    bare = parse_config(template("balanced_mp4_dir.yaml"), allow_uncertified=True)
    assert R.derived_content(bare) == {}

    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(d))
    loaded, content = R.load_certified(p, allow_uncertified=True,
                                       overrides={"tempseg_merged": "b" * 64})
    assert loaded.project == cfg.project
    assert content == {"captioner_bf16": pin.content_sha256, "tempseg_merged": "c" * 64}
