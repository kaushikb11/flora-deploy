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
