"""README-as-test: what README.md tells a user to type must exist.

Every ``flora`` command in a fenced bash block (in an inline code span, and in a ``# flora ...``
comment of a Python block) is a real command that parses with the real typer signature: its
flags exist, its positional arguments fit the command's arguments (count and values), and it
passes every required option. Every template it names ships; every repo path it mentions
exists; the config block uses only real keys and validates; the SDK snippet's calls exist; the
Modal flow deploys. CI runs the quickstart ("Try it in a few minutes") block verbatim from the
built wheel (``scripts/readme_blocks.py``).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import typer

from flora_deploy.cli import app
from flora_deploy.cli.init import TEMPLATES
from scripts.readme_blocks import (
    TRY_IT_HEADING,
    block_under,
    blocks,
    commands,
    is_install,
    logical_lines,
    try_it_script,
)

REPO = Path(__file__).resolve().parents[2]
TEXT = (REPO / "README.md").read_text(encoding="utf-8")
BLOCKS = blocks(TEXT)
BASH = [b for b in BLOCKS if b.lang == "bash"]
GROUP = typer.main.get_command(app)  # typer vendors click: duck-type it below

# The blocks the tests below read, by the heading that contains them (a LookupError names the
# headings present when one is missing).
QUICKSTART = TRY_IT_HEADING  # "Try it in a few minutes": the block CI runs
MODAL_FLOW = "Label your own data"
CONFIG = "Configuration"

# `flora ...` inside inline code spans, optional parts unbracketed.
INLINE = [re.sub(r"[\[\]]", "", s) for s in re.findall(r"`(flora [^`]+)`", TEXT)]
# `# flora ...` comments in Python blocks (e.g. how to get caption.parquet).
PY_COMMENTS = [m for b in BLOCKS if b.lang == "python"
               for m in re.findall(r"#\s*(flora [^#\n]+)", b.code)]


def _flora_argvs() -> list[tuple[str, list[str]]]:
    out = []
    for b in BASH:
        for argv in commands(b.code):
            if argv and argv[0] == "flora":
                out.append((" ".join(argv), argv[1:]))
    for s in [*INLINE, *PY_COMMENTS]:
        argv = s.split()
        out.append((s.strip(), argv[1:]))
    return out


FLORA = _flora_argvs()


def test_readme_has_the_blocks_this_test_reads():
    assert BASH and FLORA
    for heading in (QUICKSTART, MODAL_FLOW):
        block_under(TEXT, heading)  # raises, naming the headings present
    block_under(TEXT, CONFIG, "yaml")
    assert PY_COMMENTS, "the Python blocks no longer say how to get the parquet files"


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
    # positional arguments: as many as the command takes, every required one given
    pos = _positionals(rest, known)
    args = [p for p in cmd.params if p.param_type_name == "argument"]
    if not any(p.nargs == -1 for p in args):
        assert len(pos) <= sum(p.nargs for p in args), (
            f"{line}: `flora {cmd.name}` takes {len(args)} positional argument(s) "
            f"({[p.name for p in args]}), the README passes {pos}")
    required = [p for p in args if p.required]
    assert len(pos) >= len(required), f"{line}: `flora {cmd.name}` requires {required[0].name}"
    # and the real parser accepts the whole line (types, choices, values of options)
    try:
        cmd.make_context(f"flora {cmd.name}", list(rest), resilient_parsing=False)
    except Exception as e:  # click.UsageError / BadParameter (typer vendors click)
        pytest.fail(f"{line}: `flora {cmd.name}` does not parse it: {e}")


def _positionals(args: list[str], known: dict) -> list[str]:
    """The positional tokens of ``args``: not a flag, and not the value of an option before it."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a.startswith("-") and a not in ("-", "--"):
            p = known.get(a.split("=", 1)[0])
            takes_value = p is not None and not getattr(p, "is_flag", False) and \
                not getattr(p, "count", False)
            i += 2 if takes_value and "=" not in a else 1
            continue
        out.append(a)
        i += 1
    return out


def test_positional_values_are_real():
    """``flora results <stage>``: the stages the README names exist; ``flora init PATH``-style
    positionals are checked by the parser above."""
    from flora_deploy.export.compose import STAGES

    stages = []
    for _line, args in FLORA:
        cmd, rest = _command(list(args))
        if cmd.name == "results":
            known = {o: p for p in cmd.params for o in (*p.opts, *p.secondary_opts)
                     if o.startswith("-")}
            stages += _positionals(rest, known)[:1]
    assert stages, "the README never reads results back with `flora results <stage>`"
    assert set(stages) <= set(STAGES), f"unknown stage(s) {set(stages) - set(STAGES)}"


def test_positional_check_sees_option_values():
    cmd, rest = _command(["results", "caption", "--format", "parquet", "--out", "x.parquet"])
    known = {o: p for p in cmd.params for o in (*p.opts, *p.secondary_opts) if o.startswith("-")}
    assert _positionals(rest, known) == ["caption"]


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
    assert paths, "the README references no repo path (the docs links at least)"
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

    cfg_block = block_under(TEXT, CONFIG, "yaml")
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

    cfg_block = block_under(TEXT, CONFIG, "yaml")
    text = re.sub(r"<full commit sha>", "a1" * 20, cfg_block.code)
    assert "<" not in text, "unknown placeholder in the README config block"
    loads_config(text, source="README.md config block")


def _sdk_calls(code: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """``([(sdk function, [every keyword]), ...], [job method, ...])`` of the snippet (ast)."""
    import ast

    funcs, methods = [], []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name):
            if node.func.value.id == "sdk":
                funcs.append((node.func.attr, [k.arg for k in node.keywords]))
            elif node.func.value.id == "job":
                methods.append(node.func.attr)
    return funcs, methods


def test_sdk_snippet_calls_exist():
    from flora_deploy import sdk

    (py,) = [b for b in BLOCKS if b.lang == "python" and "sdk." in b.code]
    funcs, methods = _sdk_calls(py.code)
    assert funcs, "the SDK snippet calls no sdk function"
    for name, kws in funcs:
        fn = getattr(sdk, name, None)
        assert callable(fn), f"sdk has no {name}()"
        params = inspect.signature(fn).parameters
        bad = [k for k in kws if k is None or k not in params]  # None: a **kwargs splat
        assert not bad, f"sdk.{name}() has no keyword(s) {bad}"
    for meth in methods:
        assert callable(getattr(sdk.Job, meth, None)), f"sdk.Job has no {meth}()"


def test_sdk_snippet_check_sees_every_keyword():
    """Review 7 T2: the regex saw only the first keyword of a call."""
    funcs, _ = _sdk_calls('job = sdk.submit("f", job_id="x", bogus_kw=1)\n')
    assert funcs == [("submit", ["job_id", "bogus_kw"])]


def _tempseg_unpinned() -> bool:
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

    return DERIVED_PINS_BY_KEY["tempseg-merged"].content_sha256 is None


@pytest.mark.xfail(_tempseg_unpinned(), strict=True,
                   reason="balanced segment is refused until the TempSeg v2 content digest is "
                          "pinned (wave_h step 0); flips to a hard pass once it is")
def test_readme_modal_flow_is_deployable():
    """The "Label your own data" flow (`flora init [--template <t>]` then `flora deploy`) plans
    an app that registers a worker class for every shard kind the template's preset runs
    (strict: the balanced SegmentWorker / CaptionWorker are wired, P1b-1). Without
    ``--template`` the template is ``flora init``'s real default."""
    from flora_deploy.cli.init import template_text
    from flora_deploy.config.schema import loads_config
    from flora_deploy.modal_app.factory import PENDING_WORKERS, plan_app

    block = block_under(TEXT, MODAL_FLOW)
    argvs = [a for a in commands(block.code) if a[:2] == ["flora", "init"]]
    assert argvs, f"the {MODAL_FLOW!r} block runs no `flora init`"
    argv = argvs[0]
    if "--template" in argv:
        template = argv[argv.index("--template") + 1]
    else:
        init = GROUP.commands["init"]
        (tpl,) = [p for p in init.params if p.name == "template"]
        template = tpl.default
        assert template == "balanced_lerobot"
    assert any(a[:2] == ["flora", "deploy"] for a in commands(block.code))
    cfg = loads_config(template_text(template), source=template)
    assert not sorted(k for k in cfg.pools() if k in PENDING_WORKERS)
    receipts = {"tempseg_merged": "a" * 64, "captioner_bf16": "b" * 64}
    plan = plan_app(cfg, store="s3://bucket/labels", derived_receipts=receipts)
    assert set(plan.worker_classes) == set(cfg.pools())
    assert {plan.object(n).name for n in plan.worker_classes.values()}


def test_try_it_script_is_the_readme_block():
    """What CI runs is the quickstart block minus only its install line(s): every other line,
    in order, including the `flora results` read-back."""
    script = try_it_script(TEXT)
    block = block_under(TEXT, QUICKSTART)
    lines = logical_lines(block.code)
    body = script.splitlines()[1:]
    assert body == [ln for ln in lines if not is_install(ln)]
    assert any(is_install(ln) for ln in lines), "the quickstart no longer says how to install"
    flora = [ln.split("#")[0].split() for ln in body if ln.lstrip().startswith("flora ")]
    assert flora[0][:2] == ["flora", "init"] and flora[-1][:2] == ["flora", "results"]
    assert ["flora", "run", "--local", "--fake-models", "--sample-data"] in flora


def test_block_lookup_fails_loudly():
    with pytest.raises(LookupError, match="Try it"):
        block_under("## Other\n\n```bash\nflora init\n```\n", "Try it")
