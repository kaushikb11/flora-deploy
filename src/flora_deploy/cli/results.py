"""`flora compose` (parts → deduped outputs + summary) and `flora results` (read them back)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console

StoreOpt = Annotated[str, typer.Option(
    "--store", envvar="FLORA_STORE",
    help="State-of-record store: file:///dir (or a path), s3://bucket\\[/prefix], "
         "modal-volume://name.")]
JobOpt = Annotated[str, typer.Option("--job", help="Job id.")]
CfgOpt = Annotated[str | None, typer.Option(
    "--config-hash", help="Config hash (default: the job manifest's).")]


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


def _store(uri: str):
    from flora_deploy.export.results import store_from_uri

    try:
        return store_from_uri(uri)
    except (ValueError, ImportError) as e:
        raise _fail(str(e), 2) from None


@app.command()
def compose(
    job: JobOpt,
    store: StoreOpt,
    config_hash: CfgOpt = None,
    partial: Annotated[bool, typer.Option(
        "--partial", help="Compose what exists even if the job is not complete.")] = False,
    max_error_rate: Annotated[float, typer.Option(
        help="Excluded share of a stage above which the job is completed_with_errors.")] = 0.1,
) -> None:
    """Compose a job's parts into per-stage JSONL + Parquet and a summary.json."""
    from flora_deploy.export.compose import ComposeError
    from flora_deploy.export.compose import compose as do_compose

    st = _store(store)
    try:
        res = do_compose(st, job, config_hash=config_hash, partial=partial,
                         max_error_rate=max_error_rate)
    except FileNotFoundError:
        raise _fail(f"job {job!r} has no manifest in {store}") from None
    except ComposeError as e:
        raise _fail(str(e)) from None
    s = res.summary
    for stage, c in s["stages"].items():
        share = "-" if c["certified_share"] is None else f"{c['certified_share']:.1%}"
        console.print(f"{stage}: {c['records']} records ({c['rows']} rows), certified {share}, "
                      f"missing {c['missing']}, excused {c['excused']}")
    console.print(f"duplicates resolved {s['duplicates_resolved']}, conflicts "
                  f"{len(s['conflicts'])}, poison pills {len(s['poison_pills'])}, "
                  f"errors {s['errors']['by_kind']}")
    _print_status(s)
    console.print(f"summary: {res.paths['summary']}")
    if s["status"] == "completed_with_errors":
        raise typer.Exit(code=3)


def _print_status(s: dict) -> None:
    from rich.markup import escape

    color = {"complete": "green", "completed_with_errors": "yellow"}.get(s.get("status"), "red")
    console.print(f"[{color}]status {escape(str(s.get('status')))}[/{color}]")
    for p in s.get("problems", ())[:20]:
        console.print(f"  [{color}]{escape(p)}[/{color}]")


@app.command()
def results(
    job: JobOpt,
    stage: Annotated[str, typer.Option("--stage", help="segment | caption | reward.terminal | "
                                                       "reward.dense | reward.rubric")],
    store: StoreOpt,
    fmt: Annotated[str, typer.Option("--format", help="jsonl | parquet | table")] = "table",
    out: Annotated[Path | None, typer.Option("--out", help="Write to this file.")] = None,
    config_hash: CfgOpt = None,
    limit: Annotated[int, typer.Option(help="Rows shown by --format table.")] = 50,
) -> None:
    """Read a composed stage's records."""
    from flora_deploy.export.compose import STAGES
    from flora_deploy.export.results import (
        ResultsNotFound,
        iter_records,
        read_jsonl,
        read_parquet_bytes,
    )

    if stage not in STAGES:
        raise _fail(f"unknown stage {stage!r}; choose one of {', '.join(STAGES)}", 2)
    if fmt not in ("jsonl", "parquet", "table"):
        raise _fail(f"unknown format {fmt!r}; use jsonl, parquet or table", 2)
    if fmt == "parquet" and out is None:
        raise _fail("--format parquet needs --out <file>", 2)
    st = _store(store)
    from flora_deploy.export.results import load_summary

    try:
        summary = load_summary(st, job, config_hash=config_hash)
    except (ResultsNotFound, FileNotFoundError) as e:
        raise _fail(str(e)) from None
    expected = summary.get("stages", {}).get(stage, {}).get("expected", 0)
    records = summary.get("stages", {}).get(stage, {}).get("records", 0)
    if summary.get("status", "complete") != "complete":
        err = typer.echo
        err(f"warning: job {job} is {summary.get('status')}: "
            + "; ".join(summary.get("problems", ()) or ["see summary.json"]), err=True)
    if expected and not records:
        raise _fail(f"{stage}: 0 records for {expected} planned episodes (status "
                    f"{summary.get('status')}); see `flora compose` and the job's errors/")
    try:
        if fmt == "jsonl":
            data = read_jsonl(st, job, stage, config_hash=config_hash)
        elif fmt == "parquet":
            if out is None:
                raise _fail("--format parquet needs --out <file>", 2)
            data = read_parquet_bytes(st, job, stage, config_hash=config_hash)
        else:
            recs = list(iter_records(st, job, stage, config_hash=config_hash))
    except (ResultsNotFound, FileNotFoundError) as e:
        raise _fail(str(e)) from None

    if fmt != "table":
        if out is None:
            sys.stdout.write(data.decode())
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            console.print(f"Wrote {out}")
        return

    rows = [{"uid": r.uid, "certified": r.certified, "reason": r.uncertified_reason,
             "segments": len(r.segments),
             "score": (r.terminal.score if r.terminal else
                       r.rubric.total if r.rubric else None)} for r in recs]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        console.print(f"Wrote {out}")
        return
    from rich.table import Table

    t = Table(title=f"{job} {stage}: {len(recs)} records")
    for col in ("uid", "certified", "reason", "segments", "score"):
        t.add_column(col)
    for x in rows[:limit]:
        t.add_row(*("" if x[c] is None else str(x[c])
                    for c in ("uid", "certified", "reason", "segments", "score")))
    console.print(t)
    if len(rows) > limit:
        console.print(f"... {len(rows) - limit} more (raise --limit or use --format jsonl)")
