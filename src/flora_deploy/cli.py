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


def _git_provenance() -> dict:
    """Commit of the local checkout whose source Modal mounts (None outside a git checkout)."""
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True,
                                 timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no", "--", str(here)) if commit \
        else None
    return {"git_commit": commit or None, "git_dirty": None if status is None else bool(status)}


def _run_prepare_remote(keys: list[str], *, reverify: bool = False, quiet: bool = False) -> list:
    """Run ``prepare_pin`` for each key on Modal in an ephemeral ``flora-prep`` app.

    ``quiet`` (``--json``): no Modal output, and anything still printed to stdout during the run
    goes to stderr, so stdout carries only the JSON document."""
    import contextlib
    import sys

    import modal

    from flora_deploy.modal_app import prep_app

    kwargs = {"reverify": reverify, "provenance": _git_provenance()}
    with contextlib.ExitStack() as stack:
        if quiet:
            stack.enter_context(contextlib.redirect_stdout(sys.stderr))
        else:
            stack.enter_context(modal.enable_output())
        stack.enter_context(prep_app.app.run())
        return list(prep_app.prepare_pin.map(keys, kwargs=kwargs, return_exceptions=True))


def _error_text(r: BaseException) -> str:
    return f"{type(r).__name__}: {r}"


def _print_receipts(keys: list[str], results: list) -> bool:
    from rich.markup import escape

    table = Table(title="flora-weights snapshots")
    for col in ("key", "status", "files", "size", "download", "hash", "receipt / error"):
        table.add_column(col)
    all_ok = True
    for key, r in zip(keys, results, strict=True):
        if isinstance(r, BaseException):
            all_ok = False
            table.add_row(escape(key), "[red]error[/red]", "", "", "", "",
                          escape(_error_text(r)))
            continue
        all_ok &= bool(r["ok"])
        t = r.get("timings_s", {})
        status = escape(r["status"]) if r["ok"] else f"[red]{escape(r['status'])}[/red]"
        table.add_row(
            escape(key),
            status,
            str(r["files"]),
            _gib(r["bytes"]),
            f"{t['download']:.0f} s" if "download" in t else "-",
            f"{t['hash']:.0f} s" if "hash" in t else "-",
            escape(str(r["receipt"])),
        )
    console.print(table)
    for key, r in zip(keys, results, strict=True):
        if isinstance(r, BaseException):
            console.print(f"[red]{escape(key)}: {escape(_error_text(r))}[/red]")
            continue
        if r.get("note"):
            console.print(f"[dim]{escape(key)}: {escape(r['note'])}[/dim]")
        for c in r["failures"]:
            console.print(f"[red]{escape(key)}: {escape(c['name'])} failed: "
                          f"{escape(str(c['detail']))}[/red]")
    return all_ok


@app.command()
def prepare(
    dry_run: bool = typer.Option(False, "--dry-run", help="List pinned files and sizes only."),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable output (only JSON on stdout)."
    ),
    reverify: bool = typer.Option(
        False, "--reverify",
        help="Ignore existing receipts and re-hash every file (catches silent storage corruption).",
    ),
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Pin key to prepare (repeatable).")
    ] = None,
) -> None:
    """Download pinned Flora snapshots into your Modal Volume and verify them (or --dry-run)."""
    from rich.markup import escape

    from flora_deploy.config.pins import PINS, PINS_BY_KEY

    unknown = [k for k in only or [] if k not in PINS_BY_KEY]
    if unknown:
        console.print(f"[red]Unknown pin key(s): {escape(', '.join(unknown))}. "
                      f"Known: {', '.join(PINS_BY_KEY)}[/red]")
        raise typer.Exit(code=2)
    pins = [p for p in PINS if not only or p.key in only]

    if not dry_run:
        keys = [p.key for p in pins]
        if not as_json:
            console.print(f"Preparing {len(keys)} snapshot(s) on Modal: {', '.join(keys)}"
                          + (" (re-verifying)" if reverify else ""))
        results = _run_prepare_remote(keys, reverify=reverify, quiet=as_json)
        if as_json:
            typer.echo(json.dumps(
                [r if isinstance(r, dict) else
                 {"key": k, "ok": False, "status": "error", "error": _error_text(r)}
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
