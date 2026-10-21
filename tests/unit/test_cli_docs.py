"""``docs/cli.md`` is generated from the typer app (``scripts/cli_docs.py``) and covers every
registered command and option, so it cannot drift from the CLI."""

from __future__ import annotations

from scripts.cli_docs import DOC, commands, render


def test_cli_md_is_up_to_date():
    assert DOC.read_text(encoding="utf-8") == render(), (
        "docs/cli.md is stale: run `uv run python scripts/cli_docs.py --write`")


def test_cli_md_covers_every_command_and_option():
    text = DOC.read_text(encoding="utf-8")
    cmds = commands()
    assert {"status", "logs", "cancel", "resume", "report", "results", "run", "init"} <= set(cmds)
    for name, cmd in cmds.items():
        if cmd.hidden:
            continue
        assert f"### `flora {name}`" in text, f"docs/cli.md misses `flora {name}`"
        section = text.split(f"### `flora {name}`", 1)[1].split("\n### ", 1)[0]
        for p in cmd.params:
            if p.hidden:
                continue
            label = p.name.upper() if p.param_type_name == "argument" else p.opts[0]
            assert f"`{label}`" in section, f"docs/cli.md: `flora {name}` misses {label}"


def test_cli_md_has_the_exit_code_table():
    text = DOC.read_text(encoding="utf-8")
    for code in range(6):
        assert f"\n| {code} | " in text
