"""Shared Typer app and console. Each command lives in its own module (one owner per module)."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run NVIDIA Flora on Modal.")
console = Console()


def term_width() -> int:
    """The terminal width now: ``$COLUMNS`` when set (rich reads it only once, at import), else
    what the console detects."""
    import os

    cols = os.environ.get("COLUMNS", "")
    return int(cols) if cols.isdigit() and int(cols) > 0 else console.width
