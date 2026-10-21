"""Every ``examples/*/flora.yaml`` validates; ``lerobot_small`` is the balanced example."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from flora_deploy.config.schema import load_config

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*/flora.yaml")), ids=lambda p: p.parent.name)
def test_example_validates(path):
    load_config(path)


def test_lerobot_small_is_balanced_and_pinned():
    cfg = load_config(EXAMPLES / "lerobot_small" / "flora.yaml")
    assert cfg.preset == "balanced"
    assert cfg.input.repo == "lerobot/aloha_static_cups_open"
    assert re.fullmatch(r"[0-9a-f]{40}", cfg.input.revision)
    readme = (EXAMPLES / "lerobot_small" / "README.md").read_text()
    assert "estimate" in readme.lower() and "$" in readme


def test_lerobot_small_cameras_match_the_golden_mapping():
    """The dataset has 4 cameras (``meta/info.json``: cam_high, cam_low, cam_left_wrist,
    cam_right_wrist); the example maps 3 of them exactly as the golden set does, on the same
    embodiment pool, and its README names the embodiment."""
    import json

    cfg = load_config(EXAMPLES / "lerobot_small" / "flora.yaml")
    sel = json.loads((Path(__file__).resolve().parents[2] / "src/flora_deploy/validate/"
                      "golden_selection_v1.json").read_text())
    ds = next(k for k, v in sel["datasets"].items() if v["repo"] == cfg.input.repo)
    assert sel["datasets"][ds]["revision"] == cfg.input.revision
    case = next(c for c in sel["episodes"] if c["dataset"] == ds)
    assert {role: key for key, role in cfg.input.cameras.items()} == case["cameras"]
    assert cfg.pipeline.embodiment == case["pool"]
    readme = (EXAMPLES / "lerobot_small" / "README.md").read_text()
    assert f"Embodiment: `{cfg.pipeline.embodiment}`" in readme
