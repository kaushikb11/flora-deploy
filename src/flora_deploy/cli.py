"""`flora` command-line interface."""

from __future__ import annotations

import json
from typing import Annotated

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


def _run_prepare_remote(keys: list[str]) -> list:
    """Run ``prepare_pin`` for each key on Modal in an ephemeral ``flora-prep`` app."""
    import modal

    from flora_deploy.modal_app import prep_app

    with modal.enable_output(), prep_app.app.run():
        return list(
            prep_app.prepare_pin.map(keys, return_exceptions=True, wrap_returned_exceptions=False)
        )


def _print_receipts(keys: list[str], results: list) -> bool:
    table = Table(title="flora-weights snapshots")
    for col in ("key", "status", "files", "size", "download", "hash", "receipt / error"):
        table.add_column(col)
    all_ok = True
    for key, r in zip(keys, results, strict=True):
        if isinstance(r, BaseException):
            all_ok = False
            table.add_row(key, "[red]error[/red]", "", "", "", "", f"{type(r).__name__}: {r}")
            continue
        all_ok &= bool(r["ok"])
        t = r.get("timings_s", {})
        status = r["status"] if r["ok"] else f"[red]{r['status']}[/red]"
        table.add_row(
            key,
            status,
            str(r["files"]),
            _gib(r["bytes"]),
            f"{t['download']:.0f} s" if "download" in t else "-",
            f"{t['hash']:.0f} s" if "hash" in t else "-",
            r["receipt"],
        )
    console.print(table)
    for key, r in zip(keys, results, strict=True):
        if isinstance(r, dict):
            for c in r["failures"]:
                console.print(f"[red]{key}: {c['name']} failed: {c['detail']}[/red]")
    return all_ok


@app.command()
def prepare(
    dry_run: bool = typer.Option(False, "--dry-run", help="List pinned files and sizes only."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Pin key to prepare (repeatable).")
    ] = None,
) -> None:
    """Download pinned Flora snapshots into your Modal Volume and verify them (or --dry-run)."""
    from flora_deploy.config.pins import PINS, PINS_BY_KEY

    unknown = [k for k in only or [] if k not in PINS_BY_KEY]
    if unknown:
        console.print(f"[red]Unknown pin key(s): {', '.join(unknown)}. "
                      f"Known: {', '.join(PINS_BY_KEY)}[/red]")
        raise typer.Exit(code=2)
    pins = [p for p in PINS if not only or p.key in only]

    if not dry_run:
        keys = [p.key for p in pins]
        if not as_json:
            console.print(f"Preparing {len(keys)} snapshot(s) on Modal: {', '.join(keys)}")
        results = _run_prepare_remote(keys)
        if as_json:
            typer.echo(json.dumps(
                [r if isinstance(r, dict) else {"key": k, "ok": False, "error": repr(r)}
                 for k, r in zip(keys, results, strict=True)],
                indent=2,
            ))
            ok = all(isinstance(r, dict) and r["ok"] for r in results)
        else:
            ok = _print_receipts(keys, results)
        if not ok:
            raise typer.Exit(code=1)
        return

    from flora_deploy.prep.download import plan_all

    plans = plan_all(_hf_api(), pins)
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
