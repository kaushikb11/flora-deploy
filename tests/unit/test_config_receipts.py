"""``config/receipts.py``: receipt resolution (flora.yaml ``weights.derived`` + overrides), the
pin check, the comment-preserving ``flora.yaml`` edit, and the deploy / run / sdk wiring."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy import sdk
from flora_deploy.cli import app as cli
from flora_deploy.config import receipts as rc
from flora_deploy.config.hashing import config_hash
from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
from flora_deploy.config.schema import ConfigError, loads_yaml, parse_config

PINNED = "0e91e467c7ffdb636573e779672d11b203571f7869ca94317f1005eb700d9c65"  # a receipt file sha
TEMPSEG = "a" * 64
runner = CliRunner()

@pytest.fixture(autouse=True)
def _tempseg_unpinned(monkeypatch):
    """These tests build synthetic TempSeg receipts: run them in the not-yet-pinned state the
    tests describe, whatever the real pin is (pinned 2026-10-22)."""
    from dataclasses import replace as _replace

    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY as _PINS

    monkeypatch.setitem(_PINS, "tempseg-merged",
                        _replace(_PINS["tempseg-merged"], content_sha256=None))



def template_text(name: str) -> str:
    return (resources.files("flora_deploy.config") / "templates" / name).read_text()


def exact(**weights) -> dict:
    d = yaml.safe_load(template_text("exact_mp4_dir.yaml"))
    if weights:
        d["weights"] = {"derived": weights}
    return d


def balanced(**weights) -> dict:
    d = exact(**weights)
    d.update(preset="balanced", allow_uncertified=True)
    d.pop("resources", None)
    return d


# --- schema --------------------------------------------------------------------------------


def test_schema_accepts_and_validates_weights_derived():
    assert parse_config(exact()).weights.derived == {}
    cfg = parse_config(balanced(captioner_bf16=PINNED, tempseg_merged=TEMPSEG))
    assert cfg.weights.derived == {"captioner_bf16": PINNED, "tempseg_merged": TEMPSEG}
    with pytest.raises(ConfigError, match="unknown derived weight"):
        parse_config(exact(captioner_fp8=TEMPSEG))
    with pytest.raises(ConfigError, match="64-hex sha256"):
        parse_config(exact(captioner_bf16="abc"))


# --- resolution precedence -----------------------------------------------------------------


def test_needed_follows_preset_and_stages():
    assert rc.needed(parse_config(exact())) == ["captioner_bf16"]
    assert rc.needed(parse_config(balanced())) == ["captioner_bf16", "tempseg_merged"]
    rewards = exact()
    rewards["pipeline"]["stages"] = ["reward.terminal"]
    rewards.pop("resources")
    assert rc.needed(parse_config(rewards)) == []


def test_resolve_reads_config_and_overrides_win():
    cfg = parse_config(balanced(captioner_bf16=PINNED, tempseg_merged=TEMPSEG))
    assert rc.resolve(cfg) == {"captioner_bf16": PINNED, "tempseg_merged": TEMPSEG}
    got = rc.resolve(cfg, {"tempseg_merged": "b" * 64})
    assert got == {"captioner_bf16": PINNED, "tempseg_merged": "b" * 64}
    # an override alone is enough (and is not pin-checked: the explicit escape hatch)
    assert rc.resolve(parse_config(exact()), {"captioner_bf16": "0" * 64}) == {
        "captioner_bf16": "0" * 64}
    # an empty override does not erase a recorded receipt
    assert rc.resolve(cfg, {"tempseg_merged": ""})["tempseg_merged"] == TEMPSEG


def test_resolved_receipts_hash_like_explicit_ones():
    cfg_yaml = parse_config(exact(captioner_bf16=PINNED))
    cfg_cli = parse_config(exact())
    kw = {"image_fingerprint": lambda name: "fp"}
    assert (config_hash(cfg_yaml, derived_receipts=rc.resolve(cfg_yaml), **kw).hash
            == config_hash(cfg_cli, derived_receipts={"captioner_bf16": PINNED}, **kw).hash)


def test_missing_receipt_says_run_flora_prepare():
    cfg = parse_config(balanced(captioner_bf16=PINNED))
    with pytest.raises(rc.MissingReceiptError) as ei:
        rc.resolve(cfg)
    msg = str(ei.value)
    assert "tempseg_merged" in msg and "captioner_bf16" not in msg.split("needs")[1]
    assert "Run `flora prepare`" in msg and "--receipt NAME=<sha256>" in msg


# --- pins ----------------------------------------------------------------------------------


def _receipt(**changes) -> dict:
    from tests.fakes.derived import captioner_receipt

    return {**captioner_receipt(), **changes}


def test_pin_mismatch_is_a_hard_error(monkeypatch):
    from dataclasses import replace

    from flora_deploy.prep.convert_captioner import content_digest

    good = _receipt()
    pins = dict(DERIVED_PINS_BY_KEY)
    pins["captioner-bf16"] = replace(pins["captioner-bf16"], content_sha256=content_digest(good))
    monkeypatch.setattr(rc, "DERIVED_PINS_BY_KEY", pins)
    assert rc.check_pinned("captioner_bf16", good) == content_digest(good)
    other = _receipt(casts={"F32->BF16:loaded": 1})
    with pytest.raises(rc.ReceiptPinMismatch) as ei:
        rc.check_pinned("captioner_bf16", other, where="the Volume receipt")
    msg = str(ei.value)
    assert f"the Volume receipt has content digest {content_digest(other)}" in msg
    assert "pins.DERIVED_PINS['captioner-bf16'] pins " + content_digest(good) in msg
    # a stale stored digest is a mismatch too; unpinned names pass
    with pytest.raises(rc.ReceiptPinMismatch, match="stored content_sha256"):
        rc.check_pinned("captioner_bf16", {**good, "content_sha256": "0" * 64})
    rc.check_pinned("tempseg_merged", _receipt(kind="tempseg-merged"))


# --- flora.yaml edit -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["exact_lerobot.yaml", "exact_mp4_dir.yaml",
                                  "local_demo.yaml", "rewards_only.yaml"])
def test_yaml_round_trip_keeps_shipped_templates_intact(name):
    text = template_text(name)
    out = rc.set_derived_text(text, {"captioner_bf16": PINNED})
    assert out.startswith(text if text.endswith("\n") else text + "\n")  # every byte kept
    assert "# Derived weights built in your Modal Volume" in out
    data = loads_yaml(out)
    assert data == {**loads_yaml(text), "weights": {"derived": {"captioner_bf16": PINNED}}}
    # a second write merges and replaces only our block; the result stays a valid config
    again = rc.set_derived_text(out, {"tempseg_merged": TEMPSEG})
    assert again.startswith(text.rstrip("\n"))
    assert loads_yaml(again)["weights"] == {"derived": {"captioner_bf16": PINNED,
                                                       "tempseg_merged": TEMPSEG}}
    assert rc.set_derived_text(again, {"tempseg_merged": TEMPSEG}) == again  # idempotent
    parse_config(loads_yaml(again), allow_uncertified=True)


def test_yaml_edit_replaces_an_existing_block_in_place():
    text = ("project: p  # name\n"
            "# weights recorded by hand\n"
            "weights: {derived: {tempseg_merged: " + "c" * 64 + "}}  # old\n"
            "\n"
            "# output section\n"
            "output:\n"
            "  sink: s3://b/x   # keep me\n")
    out = rc.set_derived_text(text, {"captioner_bf16": PINNED})
    assert out.startswith("project: p  # name\n# weights recorded by hand\n")
    assert out.endswith("\n\n# output section\noutput:\n  sink: s3://b/x   # keep me\n")
    assert loads_yaml(out)["weights"]["derived"] == {"tempseg_merged": "c" * 64,
                                                     "captioner_bf16": PINNED}
    assert loads_yaml(out)["output"] == {"sink": "s3://b/x"}


def test_yaml_edit_refuses_what_it_cannot_edit_safely():
    with pytest.raises(ConfigError, match="only `derived"):
        rc.set_derived_text("weights: {other: 1}\n", {"captioner_bf16": PINNED})
    # a flow-style top level cannot take an appended block: refused, with the block to paste
    with pytest.raises(ConfigError, match="add this top-level block by hand"):
        rc.set_derived_text("{project: p}\n", {"captioner_bf16": PINNED})


def test_record_writes_atomically(tmp_path):
    p = tmp_path / "flora.yaml"
    p.write_text(template_text("exact_mp4_dir.yaml"))
    assert rc.record(p, {"captioner_bf16": PINNED}) is True
    assert rc.record(p, {"captioner_bf16": PINNED}) is False
    assert loads_yaml(p.read_text())["weights"]["derived"]["captioner_bf16"] == PINNED
    assert not list(tmp_path.glob("*.tmp"))


# --- deploy / run / sdk read the receipts from the config ----------------------------------


def write_cfg(tmp_path: Path, d: dict, name: str = "flora.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def _plan_json(output: str) -> dict:
    return json.JSONDecoder().raw_decode(output)[0]


def test_deploy_reads_receipts_from_config(tmp_path):
    cfg = write_cfg(tmp_path, exact(captioner_bf16=PINNED))
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--dry-run", "--json"])
    assert res.exit_code == 0, res.output
    via_cli = write_cfg(tmp_path, exact(), "cli.yaml")
    res2 = runner.invoke(cli, ["deploy", "-c", str(via_cli), "--dry-run", "--json",
                               "--receipt", f"captioner_bf16={PINNED}"])
    assert res2.exit_code == 0, res2.output
    assert _plan_json(res.output)["config_hash"] == _plan_json(res2.output)["config_hash"]


def test_deploy_missing_receipt_message(tmp_path):
    cfg = write_cfg(tmp_path, exact())
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--dry-run"])
    assert res.exit_code == 2
    out = " ".join(res.output.split())
    assert "captioner_bf16" in out and "Run `flora prepare`" in out


def test_run_missing_receipt_message(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk, "_backend", object())  # never reached
    cfg = write_cfg(tmp_path, balanced(captioner_bf16=PINNED))
    res = runner.invoke(cli, ["run", "-c", str(cfg), "--detach"])
    assert res.exit_code == 2, res.output
    out = " ".join(res.output.split())
    assert "tempseg_merged" in out and "Run `flora prepare`" in out


def test_sdk_submit_resolves_from_config(tmp_path):
    with pytest.raises(sdk.FloraError, match="flora prepare"):
        sdk.submit(parse_config(exact()), backend=object())
    seen = {}

    def planner(cfg, **kw):
        seen.update(kw["derived_receipts"])
        raise ValueError("stop here")

    with pytest.raises(sdk.FloraError, match="stop here"):
        sdk.submit(parse_config(exact(captioner_bf16=PINNED)), backend=object(),
                   planner=planner)
    assert seen == {"captioner_bf16": PINNED}


def test_fake_models_needs_no_real_receipts():
    """H0 on real Modal (2026-10-22): `flora deploy --fake-models` refused a config without
    `weights.derived`, though fake workers load no weights."""
    cfg = parse_config(balanced())
    with pytest.raises(rc.MissingReceiptError):
        rc.resolve(cfg)  # real deployments still fail closed
    assert rc.resolve(cfg, fake_models=True) == {}  # as plan_app(fake_models=True) plans it
    assert rc.resolve(cfg, {"captioner_bf16": PINNED}, fake_models=True)["captioner_bf16"] == PINNED


def test_fake_deploy_hash_matches_the_in_process_fake_plan():
    """H0: `deployed app == the client's plan` failed when the CLI invented fake receipts."""
    from flora_deploy.modal_app.factory import plan_app

    cfg = parse_config(balanced())
    via_cli = plan_app(cfg, fake_models=True,
                       derived_receipts=rc.resolve(cfg, fake_models=True))
    assert via_cli.app_name == plan_app(cfg, fake_models=True).app_name


def test_deploy_fake_models_dry_run_without_receipts(tmp_path):
    cfg = write_cfg(tmp_path, exact())
    res = runner.invoke(cli, ["deploy", "-c", str(cfg), "--dry-run", "--fake-models"])
    assert res.exit_code == 0, res.output
    assert "flora prepare" not in " ".join(res.output.split())
