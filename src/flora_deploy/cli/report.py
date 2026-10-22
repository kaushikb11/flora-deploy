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

from flora_deploy.cli._app import app, console, hours_text
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
        _p(f"video-hours: {hours_text(vh['done'])} done of {hours_text(vh['planned'])} planned")
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
    _p(f"video-hours: {hours_text(vh['done'])} done of {hours_text(vh['planned'])} planned; "
       + (f"${per:,.2f} per video-hour all-in" if per is not None else "$/video-hour: n/a yet"))
    if per is not None:
        _p("  all-in: GPU work plus each container's startup and idle time (`flora plan` "
           "estimates work and startup, not idle); on a small job startup and idle dominate")


def _rest(r: dict, job: str) -> None:
    from rich.markup import escape

    from flora_deploy import sdk

    e = r["errors"]
    parts = [f"{e['files']} error file(s)"] if e["files"] else []
    if "by_kind" in e:
        parts += [f"{k} {v}" for k, v in e["by_kind"].items() if v]
        if e["poison_pills"]:
            parts.append(f"episodes set aside {len(e['poison_pills'])}")
        for what, key in (("skipped", "excused"), ("missing", "missing")):
            nz = {k: v for k, v in (e.get(key) or {}).items() if v}
            if nz:
                parts.append(f"{what} " + ", ".join(f"{k} {v}" for k, v in nz.items()))
    _p("errors: " + ("; ".join(parts) if parts else "none"))
    for prob in [*e.get("problems", []), *e["notes"]][:10]:
        _p(f"  [yellow]{escape(prob)}[/yellow]")
    c = r["certification"]
    fake = c["stages"] and all(set(s["uncertified_reasons"]) == {"fake_models"}
                               for s in c["stages"].values())
    if fake:
        _p("certification: none (fake models)")
    else:
        _p(f"certification: preset {c['preset']} " + (
            "(segment and caption need accuracy certificates; each stage below)"
            if c["needs_certificate"]
            else "(NVIDIA's own code: certified unless an episode says otherwise)"))
    for stage, s in ({} if fake else c["stages"]).items():
        share = "-" if s["certified_share"] is None else f"{s['certified_share']:.1%}"
        reasons = ", ".join(f"{k}={v}" for k, v in s["uncertified_reasons"].items())
        _p(f"  {stage}: certified {share}" + (f" (uncertified: {escape(reasons)})" if reasons
                                              else ""))
    if r["composed"]:
        import posixpath

        from flora_deploy.cli._app import short_hashes

        paths = [p for fmts in r["outputs"].values() for p in fmts.values()]
        top = posixpath.dirname(posixpath.commonprefix(paths) + "x") if paths else ""
        _p(f"outputs in {escape(short_hashes(top))}/ (full paths: `flora report --json`):")
        for stage, fmts in r["outputs"].items():
            _p(f"  {stage}: " + ", ".join(escape(short_hashes(p[len(top) + 1:] if top else p))
                                         for p in fmts.values()))
        _p(f"  summary: {escape(short_hashes(r['summary']))}")
    else:
        _p(f"not composed yet: `flora compose --job {escape(job)} --store {escape(r['store'])}`")
    raise typer.Exit(code=sdk.exit_code(r["state"]))
