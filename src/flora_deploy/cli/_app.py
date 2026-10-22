"""Shared Typer app, console and text helpers. Each command lives in its own module (one owner
per module)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run NVIDIA Flora on Modal.")
console = Console()

# `flora --help` panels, in workflow order (``arrange_help``; docs/cli.md follows the same order).
# A command missing here lands in the last panel, so none is ever hidden.
HELP_PANELS: dict[str, tuple[str, ...]] = {
    "Get started": ("init", "doctor", "prepare"),
    "Run": ("index", "plan", "deploy", "run", "resume", "cancel"),
    "Inspect results": ("status", "report", "results", "logs", "export"),
    "Certification": ("certify",),
    "Advanced": ("estimate", "compose", "version"),
}


def command_name(info: Any) -> str:
    """A registered typer ``CommandInfo``'s command name (as typer derives it)."""
    return info.name or info.callback.__name__.lower().replace("_", "-")


def arrange_help(target: typer.Typer) -> None:
    """Group ``target``'s commands into :data:`HELP_PANELS` (``rich_help_panel``) and list them
    in that order. Typer shows panels in the order their first command is registered."""
    rank = {name: (p, i) for p, (_, names) in enumerate(HELP_PANELS.items())
            for i, name in enumerate(names)}
    last = list(HELP_PANELS)[-1]
    for info in target.registered_commands:
        name = command_name(info)
        info.rich_help_panel = list(HELP_PANELS)[rank[name][0]] if name in rank else last
    target.registered_commands.sort(
        key=lambda info: rank.get(command_name(info), (len(HELP_PANELS), 0)))


def _print_version(value: bool) -> None:
    if value:
        from flora_deploy import __version__

        typer.echo(f"flora-deploy {__version__}")
        raise typer.Exit()


@app.callback()
def _root(version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True,
        help="Print the version and exit.")) -> None:
    """Run NVIDIA Flora on Modal."""


_HASH = re.compile(r"\b([0-9a-f]{12})[0-9a-f]{4,}\b")


def short_hashes(text: str) -> str:
    """``text`` with every hex id longer than 12 characters cut to 12 and an ellipsis (store
    paths hold 64-character config hashes; ``--json`` keeps the full values)."""
    return _HASH.sub(lambda m: m.group(1) + "…", text)


def term_width() -> int:
    """The terminal width now: ``$COLUMNS`` when set (rich reads it only once, at import), else
    what the console detects."""
    import os

    cols = os.environ.get("COLUMNS", "")
    return int(cols) if cols.isdigit() and int(cols) > 0 else console.width


# ---------------------------------------------------------------------------------------------
# Plain-text rendering of counts and amounts (never a Python repr in front of a user)


def counts_text(counts: Mapping[str, Any] | None, empty: str = "none") -> str:
    """``{'success': 4, 'failure': 1}`` -> ``4 success, 1 failure`` (largest first, zeros left
    out); nothing to count -> ``empty``."""
    items = [(str(k), v) for k, v in (counts or {}).items() if v]
    if not items:
        return empty
    if not all(isinstance(v, int) for _, v in items):
        return ", ".join(f"{k} {v}" for k, v in items)
    return ", ".join(f"{v:,} {k}" for k, v in sorted(items, key=lambda kv: (-kv[1], kv[0])))


def per_stage_text(counts: Mapping[str, Any] | None) -> str:
    """Stage -> count: ``0`` when every stage has none, else ``segment 2, caption 1`` (the
    stages with any, in name order)."""
    nz = {str(k): v for k, v in (counts or {}).items() if v}
    if not nz:
        return "0"
    return ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
                     for k, v in sorted(nz.items()))


def hours_text(h: float) -> str:
    """Hours with sensible precision: ``0.11``, ``0.0072``, ``2.5``, ``12.3``, ``1,234``."""
    if h <= 0:
        return "0"
    if h < 0.0001:
        return "<0.0001"
    if h < 10:
        return f"{h:.3g}"
    if h < 1000:
        return f"{h:,.1f}"
    return f"{h:,.0f}"
