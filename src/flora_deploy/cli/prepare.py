"""`flora version` and `flora prepare`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from flora_deploy import __version__
from flora_deploy.cli._app import app, console  # noqa: E402,F401  (re-exported for tests)


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


# --- derived weights (OPT-2: modal_app/convert_app.py) ----------------------------------------


def _read_weights_file(path: str) -> bytes | None:
    """One file of the ``flora-weights`` Volume, read through the Modal client (no container);
    ``None`` when it does not exist."""
    import modal

    from flora_deploy.layout import WEIGHTS_VOLUME

    vol = modal.Volume.from_name(WEIGHTS_VOLUME, version=2)
    try:
        return b"".join(vol.read_file(path))
    except FileNotFoundError:
        return None


def _run_derived_remote(names: list[str], *, force: bool = False, quiet: bool = False
                        ) -> dict[str, dict | BaseException]:
    """Build ``names`` on Modal in an ephemeral ``flora-convert`` app, concurrently. Each
    conversion reuses its own passing receipt unless ``force``."""
    import contextlib
    import sys

    import modal

    from flora_deploy.modal_app import convert_app
    from flora_deploy.prep.derived import DERIVED_BUILDS

    kwargs = {"force": force, "provenance": _git_provenance()}
    out: dict[str, dict | BaseException] = {}
    with contextlib.ExitStack() as stack:
        if quiet:
            stack.enter_context(contextlib.redirect_stdout(sys.stderr))
        else:
            stack.enter_context(modal.enable_output())
        stack.enter_context(convert_app.app.run())
        calls = {n: getattr(convert_app, DERIVED_BUILDS[n].function).spawn(**kwargs)
                 for n in names}
        for n, call in calls.items():
            try:
                out[n] = call.get()
            except Exception as e:  # noqa: BLE001 - reported per weight
                out[n] = e
    return out


def _derived_config(config: Path, *, required: bool):
    """The config whose preset decides the derived weights; ``None`` when there is no config
    and none is required."""
    from rich.markup import escape

    from flora_deploy.config.schema import ConfigError, load_config

    if not config.exists():
        if required:
            console.print(f"[red]{escape(str(config))} not found: the derived weights to build "
                          "come from its preset (pass --config, or `flora init`)[/red]")
            raise typer.Exit(code=2)
        return None
    try:
        return load_config(config, allow_uncertified=True)
    except ConfigError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(code=2) from None


def _print_derived_plan(steps: list) -> None:
    from rich.markup import escape

    from flora_deploy.prep.derived import summarize

    table = Table(title="Derived weights (nothing built)")
    for col in ("name", "action", "reason", "est. cost"):
        table.add_column(col)
    for s in steps:
        table.add_row(s.build.name, s.action, escape(s.reason),
                      f"${s.est_usd:.2f}" if s.action != "reuse" else "-")
    console.print(table)
    summary = summarize(steps)
    console.print(f"Would build: {', '.join(summary['build']) or 'nothing'}; estimated cost "
                  f"${summary['est_usd']:.2f} (CPU containers, pinned Modal prices). Receipts "
                  "are recorded in flora.yaml under weights.derived.")


def _prepare_derived(config: Path, names: list[str], *, force: bool, quiet: bool
                     ) -> tuple[bool, list[dict]]:
    """Reuse or build each derived weight, check its receipt against the pins and record the
    receipts in ``config``. Returns (all ok, one JSON row per weight)."""
    from rich.markup import escape

    from flora_deploy.config import receipts as rc
    from flora_deploy.config.schema import ConfigError
    from flora_deploy.prep.derived import plan_derived, receipt_sha256, summarize

    steps = plan_derived(names, _read_weights_file, force=force)
    todo = [s.build.name for s in steps if s.action != "reuse"]
    if not quiet:
        reused = summarize(steps)["reuse"]
        console.print(f"Derived weights: building {', '.join(todo) or 'nothing'}"
                      + (f" (est. ${summarize(steps)['est_usd']:.2f})" if todo else "")
                      + (f"; reusing {', '.join(reused)}" if reused else ""))
    built = _run_derived_remote(todo, force=force, quiet=quiet) if todo else {}

    rows: list[dict] = []
    found: dict[str, str] = {}
    for s in steps:
        name, path = s.build.name, s.build.receipt_path()
        row = {**s.as_dict(), "ok": False, "status": "error"}
        rows.append(row)
        sha = s.receipt_sha256
        if s.action != "reuse":
            r = built.get(name)
            if isinstance(r, BaseException):
                row["error"] = _error_text(r)
                continue
            if not isinstance(r, dict) or r.get("ok") is not True:
                row["error"] = f"{s.build.function} returned no passing receipt"
                continue
            try:
                raw = _read_weights_file(path)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"could not read {path}: {_error_text(e)}"
                continue
            if raw is None:
                row["error"] = f"no receipt at {path} after the build"
                continue
            sha = receipt_sha256(raw)
        row["receipt_sha256"] = sha
        try:
            rc.check_pinned(name, sha, where=f"the {name} receipt on the Volume ({path})")
        except rc.ReceiptPinMismatch as e:
            row["error"] = str(e)
            continue
        row.update(ok=True, status="reused" if s.action == "reuse" else "built")
        found[name] = sha

    written = False
    if found:
        try:
            written = rc.record(config, found)
        except (ConfigError, OSError) as e:
            for row in rows:
                if row["ok"]:
                    row.update(ok=False, status="error", error=f"not recorded: {e}")
    if not quiet:
        table = Table(title="Derived weights")
        for col in ("name", "status", "receipt sha256 / error"):
            table.add_column(col)
        for row in rows:
            status = row["status"] if row["ok"] else f"[red]{row['status']}[/red]"
            table.add_row(row["key"], status,
                          escape(row["receipt_sha256"] if row["ok"] else row.get("error", "")))
        console.print(table)
        for row in rows:
            if not row["ok"]:
                console.print(f"[red]{escape(row['key'])}: {escape(row.get('error', ''))}[/red]",
                              soft_wrap=True)
        if found and all(r["ok"] for r in rows if r["key"] in found):
            console.print(f"{'recorded' if written else 'already recorded'} in "
                          f"{escape(str(config))} under weights.derived: "
                          f"{', '.join(sorted(found))}")
    return all(r["ok"] for r in rows), rows


@app.command()
def prepare(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="List pinned files and sizes, and the derived weights that would be built, with "
             "their estimated cost. Builds nothing."),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable output (only JSON on stdout)."
    ),
    reverify: bool = typer.Option(
        False, "--reverify",
        help="Ignore existing receipts and re-hash every file (catches silent storage corruption).",
    ),
    only: Annotated[
        list[str] | None, typer.Option(
            "--only", help="Pin key to prepare (repeatable). Skips the derived weights.")
    ] = None,
    config: Annotated[Path, typer.Option(
        "--config", "-c",
        help="flora.yaml: its preset decides the derived weights, and their receipts are "
             "recorded in it (weights.derived).")] = Path("flora.yaml"),
    skip_derived: bool = typer.Option(
        False, "--skip-derived", help="Prepare the snapshots only."),
    only_derived: bool = typer.Option(
        False, "--only-derived", help="Build the derived weights only (snapshots prepared)."),
    force_derived: bool = typer.Option(
        False, "--force-derived", help="Rebuild the derived weights even if their receipts pass."),
) -> None:
    """Download and verify the pinned Flora snapshots in your Modal Volume, then build the
    derived weights the config's preset needs and record their receipts in flora.yaml."""
    from rich.markup import escape

    from flora_deploy.config.pins import PINS, PINS_BY_KEY

    unknown = [k for k in only or [] if k not in PINS_BY_KEY]
    if unknown:
        console.print(f"[red]Unknown pin key(s): {escape(', '.join(unknown))}. "
                      f"Known: {', '.join(PINS_BY_KEY)}[/red]")
        raise typer.Exit(code=2)
    if skip_derived and (only_derived or force_derived):
        console.print("[red]--skip-derived cannot be combined with --only-derived or "
                      "--force-derived[/red]")
        raise typer.Exit(code=2)
    if only and (only_derived or force_derived):
        console.print("[red]--only selects snapshots; it cannot be combined with "
                      "--only-derived or --force-derived[/red]")
        raise typer.Exit(code=2)
    pins = [p for p in PINS if not only or p.key in only]

    cfg, names = None, []
    if not skip_derived and not only:
        from flora_deploy.config.receipts import needed

        cfg = _derived_config(config, required=only_derived or force_derived)
        names = needed(cfg) if cfg is not None else []
        if not as_json:
            if cfg is None:
                console.print(f"[dim]{escape(str(config))} not found: preparing snapshots only "
                              "(derived weights come from a config's preset)[/dim]")
            elif not names:
                console.print(f"[dim]preset {cfg.preset!r} with these stages needs no derived "
                              "weights[/dim]")

    if dry_run:
        rows: list[dict] = []
        if not only_derived:
            rows += _dry_run_snapshots(pins, as_json=as_json)
        if names:
            from flora_deploy.prep.derived import plan_derived

            steps = plan_derived(names, _read_weights_file, force=force_derived)
            rows += [s.as_dict() for s in steps]
            if not as_json:
                _print_derived_plan(steps)
        if as_json:
            typer.echo(json.dumps(rows, indent=2))
        return

    rows, ok = [], True
    if not only_derived:
        keys = [p.key for p in pins]
        if not as_json:
            console.print(f"Preparing {len(keys)} snapshot(s) on Modal: {', '.join(keys)}"
                          + (" (re-verifying)" if reverify else ""))
        results = _run_prepare_remote(keys, reverify=reverify, quiet=as_json)
        rows = [r if isinstance(r, dict) else
                {"key": k, "ok": False, "status": "error", "error": _error_text(r)}
                for k, r in zip(keys, results, strict=True)]
        if as_json:
            ok = all(isinstance(r, dict) and r["ok"] for r in results)
        else:
            ok = _print_receipts(keys, results)
        if not ok and names and not as_json:
            console.print("[red]not building the derived weights: a snapshot failed[/red]")
    if ok and names:
        ok, derived_rows = _prepare_derived(config, names, force=force_derived, quiet=as_json)
        rows += derived_rows
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
    if not ok:
        raise typer.Exit(code=1)


def _dry_run_snapshots(pins: list, *, as_json: bool) -> list[dict]:
    from flora_deploy.prep.download import plan_all

    plans = plan_all(_hf_api(), pins)
    if as_json:
        return [
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
    return []


