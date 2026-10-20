"""`flora` command-line interface."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from flora_deploy import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run NVIDIA Flora on Modal.")
console = Console()


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi()


def _gib(n: int) -> str:
    return f"{n / 2**30:,.2f} GiB"


@app.command()
def version() -> None:
    """Print the framework version."""
    console.print(__version__)


@app.command()
def prepare(
    dry_run: bool = typer.Option(False, "--dry-run", help="List pinned files and sizes only."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output (with --dry-run)."),
) -> None:
    """Download pinned Flora snapshots into your Modal Volume (dry run available)."""
    if not dry_run:
        console.print("[red]Downloading to Modal is not implemented yet; use --dry-run.[/red]")
        raise typer.Exit(code=2)

    from flora_deploy.prep.download import plan_all

    plans = plan_all(_hf_api())
    if as_json:
        payload = [
            {
                "key": p.pin.key,
                "repo": p.pin.repo,
                "revision": p.pin.revision,
                "role": p.pin.role,
                "files": len(p.files),
                "bytes": p.total_bytes,
            }
            for p in plans
        ]
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Pinned Hugging Face snapshots (nothing downloaded)")
    for col in ("key", "repo", "revision", "role", "files", "size"):
        table.add_column(col)
    for p in plans:
        table.add_row(
            p.pin.key,
            p.pin.repo,
            p.pin.revision[:12],
            p.pin.role,
            str(len(p.files)),
            _gib(p.total_bytes),
        )
    console.print(table)
    console.print(f"Total: {_gib(sum(p.total_bytes for p in plans))}")


if __name__ == "__main__":
    app()
