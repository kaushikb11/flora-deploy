"""README-as-test: what README.md tells a user to type must exist.

Every ``flora`` command in a fenced bash block (and in an inline code span) is a real command
whose flags it accepts and whose required options it passes; every template it names ships;
every repo path it mentions exists; the config block uses only real keys; the SDK snippet's
calls exist. CI also runs the "Try it locally" block verbatim (``scripts/readme_blocks.py``).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import typer

from flora_deploy.cli import app
from flora_deploy.cli.init import TEMPLATES
from scripts.readme_blocks import blocks, commands, logical_lines, try_it_script

REPO = Path(__file__).resolve().parents[2]
TEXT = (REPO / "README.md").read_text(encoding="utf-8")
BLOCKS = blocks(TEXT)
BASH = [b for b in BLOCKS if b.lang == "bash"]
GROUP = typer.main.get_command(app)  # typer vendors click: duck-type it below

# `flora ...` inside inline code spans (the "Managing jobs" table), optional parts unbracketed.
INLINE = [re.sub(r"[\[\]]", "", s) for s in re.findall(r"`(flora [^`]+)`", TEXT)]


def _flora_argvs() -> list[tuple[str, list[str]]]:
    out = []
    for b in BASH:
        for argv in commands(b.code):
            if argv and argv[0] == "flora":
                out.append((" ".join(argv), argv[1:]))
    for s in INLINE:
        argv = s.split()
        out.append((s, argv[1:]))
    return out


FLORA = _flora_argvs()


def test_readme_has_the_blocks_this_test_reads():
    assert BASH and FLORA
    assert any(b.heading.startswith("Try it locally") for b in BASH)


def _command(args: list[str]):
    cmd = GROUP
    while hasattr(cmd, "commands") and args and not args[0].startswith("-"):
        sub = cmd.commands.get(args[0])
        assert sub is not None, f"`flora {args[0]}` is not a command"
        cmd, args = sub, args[1:]
    return cmd, args


def _flags(args: list[str]) -> list[str]:
    return [a.split("=", 1)[0] for a in args if a.startswith("-") and a not in ("-", "--")]


@pytest.mark.parametrize(("line", "args"), FLORA, ids=[f[0] for f in FLORA])
def test_flora_commands_and_flags_exist(line, args):
    cmd, rest = _command(list(args))
    assert cmd is not GROUP, f"{line}: no subcommand"
    known = {}
    for p in cmd.params:
        for o in (*p.opts, *p.secondary_opts):
            if o.startswith("-"):
                known[o] = p
    flags = _flags(rest)
    unknown = [f for f in flags if f not in known]
    assert not unknown, f"{line}: `flora {cmd.name}` has no option(s) {unknown}"
    given = {known[f].name for f in flags}
    for p in cmd.params:
        if p.param_type_name == "option" and p.required and p.name not in given:
            pytest.fail(f"{line}: `flora {cmd.name}` requires {p.opts[0]}")


def test_named_templates_exist():
    named = set()
    for _, args in FLORA:
        for i, a in enumerate(args):
            if a in ("--template", "-t") and i + 1 < len(args):
                named.add(args[i + 1])
            elif a.startswith("--template="):
                named.add(a.split("=", 1)[1])
    named |= {m for m in re.findall(r"\b(\w+)\.yaml\b", TEXT) if m != "flora"}
    assert named, "the README names no template"
    missing = sorted(named - set(TEMPLATES))
    assert not missing, f"README names templates that do not ship: {missing}"


_PATH = re.compile(r"(?<![\w/.:-])((?:src|docs|scripts|tests|\.github)/[\w./-]*)")


def _paths() -> set[str]:
    found = {p.rstrip(".") for p in _PATH.findall(TEXT)}
    for target in re.findall(r"\]\(([^)]+)\)", TEXT):
        if not re.match(r"^[a-z]+://|^#|^mailto:", target):
            found.add(target.split("#", 1)[0])
    for b in BASH:  # `file.py::function` -> the file
        for argv in commands(b.code):
            found |= {a.split("::", 1)[0] for a in argv if _PATH.fullmatch(a.split("::")[0])}
    return {p for p in found if p}


def test_referenced_paths_exist():
    paths = _paths()
    assert "src/flora_deploy/config/templates/" in paths or any(
        p.startswith("src/") for p in paths)
    missing = sorted(p for p in paths if not (REPO / p).exists())
    assert not missing, f"README references paths that do not exist: {missing}"


def test_modal_entrypoints_exist():
    """`modal run file.py::fn` names a function defined in that file."""
    refs = [a for b in BASH for argv in commands(b.code) for a in argv
            if re.fullmatch(r"[\w./-]+\.py::\w+", a)]
    for ref in refs:
        path, fn = ref.split("::", 1)
        src = (REPO / path).read_text(encoding="utf-8")
        assert re.search(rf"^def {fn}\(", src, re.M), f"{ref}: no function {fn}"


def _model_keys(model, data, where=""):
    """Unknown keys in ``data`` against the pydantic ``model`` (dict-valued fields: values)."""
    from typing import get_args, get_origin

    from pydantic import BaseModel

    bad = []
    for k, v in data.items():
        field = model.model_fields.get(k)
        if field is None:
            bad.append(f"{where}{k}")
            continue
        ann = field.annotation
        if not isinstance(v, dict):
            continue
        if get_origin(ann) is dict:  # e.g. resources: {<shard kind>: ResourceOverride}
            key_t, val_t = get_args(ann)
            allowed = get_args(key_t)  # a Literal of the allowed keys, if any
            for kk, vv in v.items():
                if allowed and kk not in allowed:
                    bad.append(f"{where}{k}.{kk}")
                elif isinstance(vv, dict) and isinstance(val_t, type) and issubclass(
                        val_t, BaseModel):
                    bad += _model_keys(val_t, vv, f"{where}{k}.{kk}.")
        else:
            sub = [a for a in (ann, *get_args(ann)) if isinstance(a, type)
                   and issubclass(a, BaseModel)]
            if sub:
                bad += _model_keys(sub[0], v, f"{where}{k}.")
    return bad


def test_config_block_uses_real_keys():
    import yaml

    from flora_deploy.config.presets import gpu_problem
    from flora_deploy.config.schema import FloraConfig

    (cfg_block,) = [b for b in BLOCKS if b.lang == "yaml" and b.heading == "Config"]
    data = yaml.safe_load(cfg_block.code)
    assert not _model_keys(FloraConfig, data)
    # A preset that needs certificates (none exist yet) must opt in, or the config is refused.
    if data.get("preset", "exact") != "exact":
        assert data.get("allow_uncertified") is True
    for kind, r in (data.get("resources") or {}).items():
        if "gpu" in r:
            assert gpu_problem(data["preset"], kind, r["gpu"], 1) is None


def test_config_block_validates_with_its_placeholders_filled():
    """The README's config, with `<...>` placeholders filled, is a valid flora.yaml (GPU cap
    vs. resources, certification opt-in, ...)."""
    from flora_deploy.config.schema import loads_config

    (cfg_block,) = [b for b in BLOCKS if b.lang == "yaml" and b.heading == "Config"]
    text = re.sub(r"<full commit sha>", "a1" * 20, cfg_block.code)
    assert "<" not in text, "unknown placeholder in the README config block"
    loads_config(text, source="README.md config block")


def test_sdk_snippet_calls_exist():
    from flora_deploy import sdk

    (py,) = [b for b in BLOCKS if b.lang == "python"]
    for kw in re.findall(r"sdk\.submit\([^)]*?(\w+)=", py.code):
        assert kw in inspect.signature(sdk.submit).parameters
    for meth in re.findall(r"job\.(\w+)\(", py.code):
        assert callable(getattr(sdk.Job, meth, None)), f"sdk.Job has no {meth}()"


def test_try_it_script_is_the_readme_block():
    script = try_it_script(TEXT)
    (block,) = [b for b in BASH if b.heading.startswith("Try it locally")]
    body = script.splitlines()[1:]
    assert body and all(ln in logical_lines(block.code) for ln in body)
    assert body[0].startswith("flora init")
