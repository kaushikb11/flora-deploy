"""Real-Modal CPU canary (``scripts/canary.py``). Costs cents; excluded by default
(``-m modal`` to run)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "canary.py"


def _canary():
    spec = importlib.util.spec_from_file_location("flora_canary_script", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.modal
@pytest.mark.parametrize("scenario", ["happy", "preempt", "poison", "restart", "budget"])
def test_canary_scenario(scenario):
    report = _canary().run([scenario])
    r = report["scenarios"][scenario]
    assert r["pass"], r["problems"]
