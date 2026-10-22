"""Each runbook's Modal ``flora index`` gets its own app name, so billing (reported per app
name) attributes it to that run: ``flora index --app-name`` / ``$FLORA_INDEX_APP``, and
``scale_test.py`` passes ``flora-st-<run>-index-a<n>``. No Modal: the app is faked."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
SHA = "d793c969cf716001dcca18a0842c3d7e9de9e41b"

CONFIG = f"""\
project: idx-app
license_ack: research-only
modal: {{max_gpus: 2, budget_usd: 5}}
preset: exact
input: {{format: lerobot_v21, repo: org/ds, revision: {SHA}}}
pipeline:
  stages: [segment, caption]
  embodiment: yam
output: {{sink: "volume://flora-canary/idx"}}
"""


class Stop(Exception):
    pass


@pytest.fixture
def fake_index_app(monkeypatch):
    """``importlib.reload(index_app)`` returns a stand-in whose ``app.run`` records its kwargs
    and whose remote call stops the command (nothing reaches Modal)."""
    seen: dict = {}

    @contextlib.contextmanager
    def run(**kw):
        seen["run"] = kw
        yield

    def remote(*a, **k):
        raise Stop

    fake = types.SimpleNamespace(app=types.SimpleNamespace(run=run),
                                 run_index_remote=types.SimpleNamespace(remote=remote))
    real_reload = importlib.reload
    monkeypatch.setattr(importlib, "reload", lambda m: fake if m.__name__.endswith(
        "index_app") else real_reload(m))
    monkeypatch.delenv("FLORA_INDEX_APP", raising=False)
    return seen


def invoke(tmp_path, *args, env=None):
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(CONFIG)
    return CliRunner().invoke(app, ["index", "-c", str(cfg), *args], env=env)


def test_default_app_name_is_unchanged(tmp_path, fake_index_app):
    r = invoke(tmp_path)
    assert isinstance(r.exception, Stop), r.output
    assert fake_index_app["run"] == {}  # modal.App("flora-index")'s own name


def test_app_name_flag_and_env(tmp_path, fake_index_app):
    r = invoke(tmp_path, "--app-name", "flora-st-r1-index-a1")
    assert isinstance(r.exception, Stop), r.output
    assert fake_index_app["run"] == {"name": "flora-st-r1-index-a1"}
    r = invoke(tmp_path, env={"FLORA_INDEX_APP": "flora-x-index"})
    assert isinstance(r.exception, Stop), r.output
    assert fake_index_app["run"] == {"name": "flora-x-index"}


@pytest.mark.parametrize("bad", ["", "has space", "x" * 65, "-lead"])
def test_bad_app_name_is_refused(tmp_path, fake_index_app, bad):
    r = invoke(tmp_path, "--app-name", bad)
    assert r.exit_code == 2 and "--app-name" in r.output and "run" not in fake_index_app


# ---------------------------------------------------------------------------------------------
# scale_test.py


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scale_test_real_index_uses_its_own_app_name(monkeypatch, tmp_path):
    sc = sys.modules.get("scale_test") or _load("scale_test")
    wh = sys.modules["wave_h"]
    st = wh.State("1022130531d5", 0.0, "2026-10-22")
    ctx = sc.Ctx("real", st, tmp_path, tmp_path / "state.json", 0.0, step="index")
    st.step("index").attempts += [wh.Attempt(started=0.0, status="failed"),
                                  wh.Attempt(started=1.0)]
    calls: list[list[str]] = []

    def sh(args, *, timeout=1800, check=True):
        calls.append(list(args))
        (tmp_path / "summary.json").write_text(json.dumps({"counts": {"indexed": 1}}))

    ctx.sh = sh  # type: ignore[method-assign]
    monkeypatch.setattr(wh, "resolve_receipts", lambda *a, **k: {})
    monkeypatch.setattr(sc, "_write_config", lambda ctx, derived=None: None)
    monkeypatch.setattr(sc, "_flora", lambda: "flora")
    monkeypatch.setattr(sc, "subset_problems", lambda c, v: [])
    sc.run_index(ctx)
    name = "flora-st-1022130531d5-index-a2"
    (args,) = calls
    assert args[args.index("--app-name") + 1] == name
    assert st.step("index").current.apps == [name]  # noted before it can spend
    assert "flora-index" not in json.dumps(calls) and not name.startswith("flora-wh-")
