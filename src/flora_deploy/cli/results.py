"""`flora compose` (parts → deduped outputs + summary) and `flora results` (read them back)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console
from flora_deploy.cli.run import ConfigOpt, EnvOpt

StoreOpt = Annotated[str, typer.Option(
    "--store", envvar="FLORA_STORE",
    help="State-of-record store: file:///dir (or a path), s3://bucket\\[/prefix], "
         "modal-volume://name.")]
JobOpt = Annotated[str, typer.Option("--job", help="Job id.")]
JobStoreOpt = Annotated[str | None, typer.Option(
    "--store", envvar="FLORA_STORE",
    help="Job store: file:///dir (or a path), s3://bucket\\[/prefix], modal-volume://name "
         "(default: the config's output.sink, then <config dir>/flora-local-store).")]
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


def _check_namespace(st, job: str, config_hash: str | None) -> None:
    """Refuse to read a fake-models run as a real job or the reverse (review 8b M5)."""
    from flora_deploy import sdk

    try:
        sdk.check_results_namespace(st, job, config_hash)
    except sdk.FloraError as e:
        raise _fail(str(e)) from None


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
    _check_namespace(st, job, config_hash)
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
                  f"{len(s['conflicts'])}, episodes set aside {len(s['poison_pills'])}, "
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
    stage: Annotated[str | None, typer.Argument(
        help="segment | caption | reward.terminal | reward.dense | reward.rubric",
        show_default=False)] = None,
    job: Annotated[str | None, typer.Option(
        "--job", help="Job id (default: the config's project; its fake-models run "
                      "<project>.fake when there is no real one).", show_default=False)] = None,
    store: JobStoreOpt = None,
    fmt: Annotated[str, typer.Option("--format", help="jsonl | parquet | table")] = "table",
    out: Annotated[Path | None, typer.Option("--out", help="Write to this file.")] = None,
    config_hash: CfgOpt = None,
    limit: Annotated[int, typer.Option(help="Rows shown by --format table.")] = 50,
    config: ConfigOpt = Path("flora.yaml"),
    env: EnvOpt = None,
    stage_opt: Annotated[str | None, typer.Option(
        "--stage", hidden=True, help="Same as the STAGE argument (older spelling).")] = None,
) -> None:
    """Read a composed stage's records (job and store default from flora.yaml)."""
    from flora_deploy.export.compose import STAGES
    from flora_deploy.export.results import (
        ResultsNotFound,
        iter_records,
        read_jsonl,
        read_parquet_bytes,
    )

    if stage is not None and stage_opt is not None and stage != stage_opt:
        raise _fail(f"two stages given ({stage} and --stage {stage_opt}); give one", 2)
    stage = stage or stage_opt
    if stage is None:
        raise _fail(f"which stage? `flora results <stage>`, one of {', '.join(STAGES)}", 2)
    if stage not in STAGES:
        raise _fail(f"unknown stage {stage!r}; choose one of {', '.join(STAGES)}", 2)
    if fmt not in ("jsonl", "parquet", "table"):
        raise _fail(f"unknown format {fmt!r}; use jsonl, parquet or table", 2)
    if fmt == "parquet" and out is None:
        raise _fail("--format parquet needs --out <file>", 2)
    from flora_deploy.cli._jobref import ResolveError, print_notes, resolve
    from flora_deploy.config.schema import ConfigError

    try:
        ref = resolve(job, store, config, env)
    except (ResolveError, ConfigError) as e:
        raise _fail(str(e)) from None
    print_notes(ref)
    st, job = ref.store, ref.job_id
    _check_namespace(st, job, config_hash)
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
             "certificate": r.provenance.certificate, "segments": len(r.segments),
             "score": (r.terminal.score if r.terminal else
                       r.rubric.total if r.rubric else None)} for r in recs]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        console.print(f"Wrote {out}")
        return
    from rich.markup import escape
    from rich.table import Table

    from flora_deploy.cli._app import term_width

    cols, label_rows = label_table(stage, recs)
    # one record per episode; a caption / segment table has one row per subtask / segment
    n_ep = f"{len(recs)} episode{'s' if len(recs) != 1 else ''}"
    noun = {"caption": "subtasks", "segment": "segments"}.get(stage, "rows")
    title = (n_ep if len(label_rows) == len(recs)
             else f"{n_ep}, {len(label_rows)} {noun}")
    t = Table()  # the header is its own line: a table title wraps to a narrow table's width
    shown = label_rows[:limit]
    same = {r[-1] for r in label_rows}
    if len(same) == 1:  # every record has the same certification: say it once, below
        cols, shown = cols[:-1], [r[:-1] for r in shown]
    width = term_width()
    fits = [min(max([len(col), *(len(r[j]) for r in shown)]), 40) for j, col in enumerate(cols)]
    # the text column (caption / summary) gets what the others leave, and is truncated to it
    room = width - (3 * len(cols) + 1) - sum(f for f, c in zip(fits, cols, strict=True)
                                             if c not in WIDE_COLUMNS)
    for col, fit in zip(cols, fits, strict=True):
        if col in WIDE_COLUMNS:
            w = max(12, room)
            t.add_column(col, no_wrap=True, overflow="ellipsis", max_width=w)
        else:
            t.add_column(col, no_wrap=True, overflow="ellipsis", min_width=fit)
    for row in shown:
        t.add_row(*(escape(c) for c in row))
    console.print(f"{escape(job)} {stage}: {title}", soft_wrap=True, highlight=False)
    console.print(t, width=width)
    if len(same) == 1:
        console.print(f"certified: {escape(same.pop())} for every record", soft_wrap=True)
    ids = sorted({r.provenance.certificate for r in recs
                  if r.certified and r.provenance.certificate})
    if ids:
        console.print("certificate: " + ", ".join(ids), soft_wrap=True)
    if len(label_rows) > limit:
        more = len(label_rows) - limit
        console.print(f"... {more} more row{'s' if more != 1 else ''} (raise --limit or use "
                      "--format jsonl)", soft_wrap=True)


# Columns of `flora results` tables that take the remaining width and are truncated to it.
WIDE_COLUMNS = ("caption", "summary", "scores")


def _span(s) -> str:
    return f"{s.start_s:.1f}–{s.end_s:.1f} s"


# ``uncertified_reason`` in plain words (the JSONL / Parquet keep the code).
_REASON_WORDS = {
    "uncertified_stack": "no certificate for this stage",
    "outside_reference_envelope": "outside NVIDIA's tested conditions",
    "oracle_ineligible_source": "video cannot be compared with the reference",
    "fake_models": "fake models",
}


def _certified(r) -> str:
    """``yes (certificate cert_1a2b3c4d)`` (the id shortened; ``flora results`` prints it in
    full below the table), ``yes (reference)`` or ``no (<why, in plain words>)``."""
    p = r.provenance
    if r.certified:
        return f"yes (certificate {p.certificate[:13]})" if p.certificate else "yes (reference)"
    reason = r.uncertified_reason
    if p.certificate == "local-fake-models":  # whatever else is true, these are stand-ins
        reason = "fake_models"
    if reason == "uncovered_pool":
        why = f"not certified for {p.embodiment}" if p.embodiment else "embodiment not covered"
    else:
        why = _REASON_WORDS.get(str(reason), str(reason or "-"))
    return f"no ({why})"


def dense_summary(per_step, width: int = 24) -> str:
    """``3×4 1×2: 3 3 3 3 1 1``: counts per step label (1 failure, 2 suboptimal, 3 normal),
    then the first values, as many as fit in ``width`` characters."""
    from collections import Counter

    counts = Counter(per_step)
    head = " ".join(f"{k}×{counts[k]}" for k in sorted(counts, reverse=True))
    vals, shown = [str(v) for v in per_step], ""
    for i in range(len(vals), 0, -1):
        shown = " ".join(vals[:i]) + (" …" if i < len(vals) else "")
        if len(shown) <= width:
            break
    return f"{head}: {shown}" if vals else "-"


def label_table(stage: str, recs) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """The labels themselves, per stage: one row per segment (span, and caption for
    ``caption``); the score and expected score for ``reward.terminal``; a per-step summary for
    ``reward.dense``; the total and per-step scores for ``reward.rubric``. Each row also says
    whether the record is certified (and why not)."""
    rows: list[tuple[str, ...]] = []
    if stage in ("segment", "caption"):
        cols = ("uid", "#", "span", *(("caption",) if stage == "caption" else ()), "certified")
        for r in recs:
            if not r.segments:
                rows.append((r.uid, "-", "-", *(("-",) if stage == "caption" else ()),
                             _certified(r)))
            for i, s in enumerate(r.segments):
                text = ((s.caption or "-").replace("\n", " "),) if stage == "caption" else ()
                rows.append((r.uid, str(i), _span(s), *text, _certified(r)))
        return cols, rows
    if stage == "reward.terminal":
        for r in recs:
            t = r.terminal
            rows.append((r.uid, str(t.score) if t else "-",
                         f"{t.expected_score:.2f}" if t else "-", _certified(r)))
        return ("uid", "score (1-5)", "expected", "certified"), rows
    if stage == "reward.dense":
        for r in recs:
            d = r.dense
            rows.append((r.uid, str(len(d.per_step)) if d else "-",
                         dense_summary(d.per_step) if d else "-", _certified(r)))
        return ("uid", "steps", "summary", "certified"), rows
    for r in recs:  # reward.rubric
        u = r.rubric
        rows.append((r.uid, f"{u.total:g}" if u else "-",
                     " ".join(f"{x:g}" for x in u.scores) if u else "-", _certified(r)))
    return ("uid", "total", "scores", "certified"), rows
