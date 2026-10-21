"""Review 8b lows: the TempSeg receipt's LoRA rule describes merger v2, ``--receipt`` values are
validated, and the lerobot_small example's camera comment matches the dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.config import receipts
from flora_deploy.config.schema import load_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "lerobot_small" / "flora.yaml"
SHA = "a" * 64


def test_tempseg_receipt_rule_is_the_v2_rule():
    from flora_deploy.prep.merge_tempseg import MERGER_VERSION, LoraSpec

    assert MERGER_VERSION == 2
    rule = LoraSpec(32, 64.0, frozenset({"q_proj"})).record()["rule"]
    assert "fp64" in rule and "@" not in rule  # a fixed-order FP64 rank sum, not a matmul
    assert rule != "bf16(fp32(W) + scale * (fp32(B) @ fp32(A)))"  # v1's


@pytest.mark.parametrize("value", ["not-a-sha", "A" * 64, "a" * 63, "g" * 64])
def test_receipt_overrides_must_be_64_hex(value):
    cfg = load_config(EXAMPLE)
    with pytest.raises(ValueError, match="64-hex"):
        receipts.resolve(cfg, {"captioner_bf16": value, "tempseg_merged": SHA})


def test_receipt_overrides_must_name_a_known_receipt():
    cfg = load_config(EXAMPLE)
    with pytest.raises(ValueError, match="unknown"):
        receipts.resolve(cfg, {"captioner_bf16": SHA, "tempseg_merged": SHA, "captioner": SHA})
    assert receipts.resolve(cfg, {"captioner_bf16": SHA, "tempseg_merged": SHA}) == {
        "captioner_bf16": SHA, "tempseg_merged": SHA}


def test_cli_refuses_a_bad_receipt():
    out = CliRunner().invoke(app, ["deploy", "--dry-run", "-c", str(EXAMPLE), "--receipt",
                                   "captioner_bf16=not-a-sha", "--receipt",
                                   f"tempseg_merged={SHA}"])
    assert out.exit_code == 2 and "64-hex" in out.output


def test_lerobot_small_camera_comment():
    text = EXAMPLE.read_text()
    cams = yaml.safe_load(text)["input"]["cameras"]
    assert len(cams) == 3 and "observation.images.cam_low" not in cams
    head = text.split("project:")[0]
    assert "4 cameras" in head and "3 cameras" not in head
    assert "cam_low" in head  # which camera is unused, and why, up front
