"""`flora report <job>`: what a job cost and produced.

Spend (total and its source, budget, projection; per shard kind from the orchestrator's call
history; per container from container receipts), video-hours planned and done, $/video-hour,
an error summary (error files, excused and missing episodes, poison pills, problems from the
composed summary), certification per stage (certified share, uncertified reasons, and whether
the preset needs Gate certificates) and the store paths of the outputs.

Spend is an estimate at pinned Modal list prices (``jobs.budget``); Modal's billing report
(``modal billing report``) is the authority. Exit codes as ``flora status``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from flora_deploy.cli._app import app, console
from flora_deploy.cli.run import (
    ConfigOpt,
    EnvOpt,
    JobArg,
    JsonOpt,
    StoreOpt,
    fail,
    open_job_cli,
)


def _p(s: str) -> None:
    console.print(s, soft_wrap=True, highlight=False)


@app.command()
def report(job: JobArg = None, store: StoreOpt = None,
           config: ConfigOpt = Path("flora.yaml"), env: EnvOpt = None,
           as_json: JsonOpt = False) -> None:
    """Spend by kind/container, $/video-hour, errors, certification and output locations."""
    from rich.markup import escape

    from flora_deploy import sdk

    j = open_job_cli(job, store, config, env)

    job = j.job_id  # resolved (default: the config's project)
    try:
        r = j.report()
    except (sdk.FloraError, FileNotFoundError, ValueError) as e:
        raise fail(str(e)) from None
    if as_json:
        typer.echo(json.dumps(r, indent=2, sort_keys=True, default=str))
        raise typer.Exit(code=sdk.exit_code(r["state"]))
    sp = r["spend"]
    _p(f"[bold]{escape(job)}[/bold] {r['state']}  config {r['config_hash'][:12]}  app "
       f"{escape(r['app_name'])}")
    vh, per = r["video_hours"], r["usd_per_video_hour"]
    if sp.get("simulated"):
        from flora_deploy.cli.run import SIMULATED_SPEND

        _p(SIMULATED_SPEND)
        if sp["containers"]:
            done = all(c["stopped"] for c in sp["containers"])
            _p(f"  containers: {len(sp['containers'])} local stand-ins, "
               + ("all finished" if done else "some still running"))
        _p(f"video-hours: {vh['done']:.3f} done of {vh['planned']:.3f} planned")
    else:
        _spend(sp, vh, per)
    _rest(r, job)


def _spend(sp: dict, vh: dict, per: float | None) -> None:
    from rich.markup import escape

    _p(f"spend ${sp['total_usd']:,.2f} (source {sp['source']}) of budget "
       f"${sp['budget_usd']:,.2f}; projected ${sp['projected_usd']:,.2f}; prices "
       f"{escape(sp['prices'])}")
    for kind, k in sp["by_kind"].items():
        _p(f"  {escape(kind)}: {k['calls']} calls, ${k['usd']:,.2f} (call-history estimate)")
    if sp["containers"]:
        total = sum(c["usd"] for c in sp["containers"])
        _p(f"  containers: {len(sp['containers'])}, ${total:,.2f} from container receipts")
        for c in sp["containers"][:20]:
            _p(f"    {escape(c['container_id'])} {c['gpu'] or 'cpu'}x{c['gpu_count']} "
               f"${c['usd']:,.2f}{'' if c['stopped'] else ' (running)'}")
    _p(f"video-hours: {vh['done']:.3f} done of {vh['planned']:.3f} planned; "
       + (f"${per:,.2f} per video-hour all-in" if per is not None else "$/video-hour: n/a yet"))
    if per is not None:
        _p("  all-in: GPU work plus each container's startup and idle time (`flora plan` "
           "estimates work and startup, not idle); on a small job startup and idle dominate")


def _rest(r: dict, job: str) -> None:
    from rich.markup import escape

    from flora_deploy import sdk

    e = r["errors"]
    _p(f"errors: {e['files']} error file(s)" + (
        f", by kind {e['by_kind']}, episodes set aside {len(e['poison_pills'])}, excused "
        f"{e['excused']}, missing {e['missing']}" if "by_kind" in e else ""))
    for prob in [*e.get("problems", []), *e["notes"]][:10]:
        _p(f"  [yellow]{escape(prob)}[/yellow]")
    c = r["certification"]
    _p(f"certification: preset {c['preset']} " + (
        "(uncertified until its accuracy certificates exist)" if c["needs_certificate"]
        else "(reference by construction)"))
    for stage, s in c["stages"].items():
        share = "-" if s["certified_share"] is None else f"{s['certified_share']:.1%}"
        reasons = ", ".join(f"{k}={v}" for k, v in s["uncertified_reasons"].items())
        _p(f"  {stage}: certified {share}" + (f" (uncertified: {escape(reasons)})" if reasons
                                              else ""))
    if r["composed"]:
        _p(f"outputs ({escape(r['store'])}):")
        for stage, fmts in r["outputs"].items():
            for fmt, path in fmts.items():
                _p(f"  {stage} {fmt}: {escape(path)}")
        _p(f"  summary: {escape(r['summary'])}")
    else:
        _p(f"not composed yet: `flora compose --job {escape(job)} --store {escape(r['store'])}`")
    raise typer.Exit(code=sdk.exit_code(r["state"]))
