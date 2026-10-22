"""Shared Typer app and console. Each command lives in its own module (one owner per module)."""

from __future__ import annotations

import re

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run NVIDIA Flora on Modal.")
console = Console()


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
