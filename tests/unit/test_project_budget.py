"""The project budget is one setting (``FLORA_PROJECT_BUDGET_USD``, default $50), read by
``scripts/spend.py``, ``scripts/cert_preflight.py`` and the ``wave_h.py --global-cap`` ceiling.
No Modal: billing and commands are faked."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load(name: str, alias: str):
    """A fresh copy of ``scripts/<name>.py`` (module constants read the environment now)."""
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod  # dataclasses look their module up while it loads
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(alias, None)
    return mod


spend = load("spend", "spend_budget_test")


def test_default_is_50_and_the_env_var_sets_it():
    assert spend.project_budget_usd({}) == 50.0 == spend.DEFAULT_PROJECT_BUDGET_USD
    assert spend.project_budget_usd({"FLORA_PROJECT_BUDGET_USD": ""}) == 50.0
    assert spend.project_budget_usd({"FLORA_PROJECT_BUDGET_USD": "120"}) == 120.0
    assert spend.project_budget_usd({"FLORA_PROJECT_BUDGET_USD": " $75.5 "}) == 75.5


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "inf", "nan"])
def test_a_bad_value_is_refused_with_the_name(bad):
    with pytest.raises(SystemExit, match="FLORA_PROJECT_BUDGET_USD"):
        spend.project_budget_usd({"FLORA_PROJECT_BUDGET_USD": bad})


def _billing(monkeypatch):
    monkeypatch.setattr(spend, "report", lambda since: [{"description": "flora-x", "cost": 10}])


def test_spend_reads_the_env_var(monkeypatch, capsys):
    _billing(monkeypatch)
    monkeypatch.setenv("FLORA_PROJECT_BUDGET_USD", "80")
    monkeypatch.setattr(sys, "argv", ["spend.py"])
    spend.main()
    assert "of $80.00  (left $70.00)" in capsys.readouterr().out


def test_spend_flag_overrides_the_env_var(monkeypatch, capsys):
    _billing(monkeypatch)
    monkeypatch.setenv("FLORA_PROJECT_BUDGET_USD", "80")
    monkeypatch.setattr(sys, "argv", ["spend.py", "--budget", "30"])
    spend.main()
    assert "of $30.00  (left $20.00)" in capsys.readouterr().out


def test_spend_default_unchanged(monkeypatch, capsys):
    _billing(monkeypatch)
    monkeypatch.delenv("FLORA_PROJECT_BUDGET_USD", raising=False)
    monkeypatch.setattr(sys, "argv", ["spend.py"])
    spend.main()
    assert "of $50.00  (left $40.00)" in capsys.readouterr().out


def test_cert_preflight_uses_the_setting_and_passes_it_to_spend(monkeypatch):
    monkeypatch.setenv("FLORA_PROJECT_BUDGET_USD", "64")
    cp = load("cert_preflight", "cert_preflight_budget_test")
    assert cp.PROJECT_BUDGET_USD == 64.0
    calls = []

    class Env:
        def py(self, *args, timeout=0.0, env=None):
            calls.append(args)
            return cp.Result(0, "TOTAL                    $  20.00  of $64.00\n", "")

    assert cp.billed_total(Env()) == (20.0, "")
    assert calls[-1][calls[-1].index("--budget") + 1] == "64"
    assert cp.billed_total(Env(), 70.0) == (20.0, "")
    assert calls[-1][calls[-1].index("--budget") + 1] == "70"
    plan = type("P", (), {"upper": 5.0})()
    c = cp.check_budget(Env(), plan, budget=26.0, margin=2.0)  # 26 - 20 - 5 = 1 < 2
    assert c.status == cp.FAIL and "FLORA_PROJECT_BUDGET_USD" in c.fix
    assert calls[-1][calls[-1].index("--budget") + 1] == "26"


def test_cert_preflight_default_unchanged(monkeypatch):
    monkeypatch.delenv("FLORA_PROJECT_BUDGET_USD", raising=False)
    assert load("cert_preflight", "cert_preflight_budget_test2").PROJECT_BUDGET_USD == 50.0


def test_wave_h_global_cap_ceiling_is_the_project_budget(monkeypatch):
    monkeypatch.setenv("FLORA_PROJECT_BUDGET_USD", "60")
    wh = load("wave_h", "wave_h_budget_test")
    assert wh.MAX_GLOBAL_CAP_USD == 60.0
    assert wh.check_global_cap(55.0, spent=0.0, derived=5.4) is None
    assert "ceiling" in (wh.check_global_cap(61.0, spent=0.0, derived=5.4) or "")
