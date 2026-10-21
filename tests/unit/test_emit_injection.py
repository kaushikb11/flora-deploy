"""``flora deploy --emit`` never turns a config value into code (security review 11b F2)."""

from __future__ import annotations

import ast
import dataclasses
from importlib import resources

import pytest
import yaml

from flora_deploy.config.schema import ConfigError, parse_config
from flora_deploy.modal_app import factory as f
from flora_deploy.modal_app.emit import render_script

MARK = "INJECTED_MARKER"
HOSTILE = [
    'x"""; INJECTED_MARKER = 1; """',  # the review's probe
    "x'''; INJECTED_MARKER = 1; '''",
    "x\nINJECTED_MARKER = 1\n",
    "x\\",
    "x`; INJECTED_MARKER=1 #",
    "en-x",  # Modal reserves en- for environment ids
    "-x",
    "a" * 65,
    "",
]


def template() -> dict:
    return yaml.safe_load((resources.files("flora_deploy.config") / "templates"
                           / "exact_mp4_dir.yaml").read_text())


def plan(**kw):
    return f.plan_app(parse_config(template()), derived_receipts={"captioner_bf16": "0" * 64},
                      **kw)


@pytest.mark.parametrize("env", HOSTILE)
def test_config_refuses_hostile_environment(env):
    d = template()
    d.setdefault("modal", {})["environment"] = env
    with pytest.raises(ConfigError, match="environment"):
        parse_config(d)


@pytest.mark.parametrize("field,value", [("region", 'us"""; X = 1; """'),
                                         ("tags", {"k": 'v"""; X = 1; """'})])
def test_config_refuses_hostile_region_and_tag_values(field, value):
    d = template()
    d.setdefault("modal", {})[field] = value
    with pytest.raises(ConfigError):
        parse_config(d)


@pytest.mark.parametrize("env", ["main", "dev", "my_env-2", "legacy.env", "A" * 64])
def test_config_accepts_modal_environment_names(env):
    d = template()
    d.setdefault("modal", {})["environment"] = env
    assert parse_config(d).modal.environment == env


def _top_level_kinds(src: str) -> list[str]:
    return [type(n).__name__ for n in ast.parse(src).body]


@pytest.mark.parametrize("value", HOSTILE[:5])
def test_render_script_has_no_extra_statements_for_hostile_plan(value):
    """Defence in depth: a plan that bypassed the schema still renders inert source."""
    good = plan()
    expected = _top_level_kinds(render_script(good))
    bad = dataclasses.replace(good, environment=value, project=value, preset=value,
                              config_hash=value)
    src = render_script(bad)
    assert _top_level_kinds(src) == expected
    tree = ast.parse(src)
    assert ast.get_docstring(tree) is not None and MARK not in ast.get_docstring(tree)
    names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert MARK not in names
    ns = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
          and n.targets[0].id in ("MODAL_ENVIRONMENT", "PROJECT", "PRESET", "CONFIG_HASH")}
    assert ns == {"MODAL_ENVIRONMENT": value, "PROJECT": value, "PRESET": value,
                  "CONFIG_HASH": value}
