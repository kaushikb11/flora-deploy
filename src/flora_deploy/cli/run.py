"""`flora run`: start a job on the deployed app (default) or run it locally (``--local``).

* **Modal** (default): ``sdk.submit`` plans the config's app offline, refuses when it is not
  deployed or was deployed for another config hash (pass the same ``--receipt`` /
  ``--determinism`` as ``flora deploy``), plans the job from ``flora index``'s receipts, writes
  the run spec and spawns the deployed orchestrator. Then it streams status until the job stops
  and composes the outputs, unless ``--detach``.
* **Local** (``--local --fake-models``): ``cli/run_local.py`` (G2): the whole job system in this
  process with fake models; same flags and exit codes as before.

Exit codes (both paths, and ``flora status`` / ``flora report``): 0 ``done`` (or still running
with ``--detach``), 1 ``failed``, 3 ``completed_with_errors``, 4 ``budget_stopped``, 5 any
other stop (``cancelled``); 2 usage errors and refusals (not deployed, config mismatch, nothing
indexed).

This module also holds the options and helpers the job commands (status, resume, cancel,
report, logs) share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from flora_deploy.cli._app import app, console

JobArg = Annotated[str, typer.Argument(help="Job id (default at submit: the config's project).")]
StoreOpt = Annotated[str | None, typer.Option(
    "--store", envvar="FLORA_STORE",
    help="Job store: modal-volume://name\\[/path], s3://bucket\\[/prefix], file:///dir "
         "(default: the config's output.sink).")]
ConfigOpt = Annotated[Path, typer.Option(
    "--config", "-c", help="flora.yaml (finds the store and Modal environment).")]
EnvOpt = Annotated[str | None, typer.Option(
    "--env", envvar="MODAL_ENVIRONMENT", help="Modal environment of a modal-volume store.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON.")]


def fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]", soft_wrap=True)
    return typer.Exit(code=code)


def open_job_cli(job: str, store: str | None, config: Path, env: str | None):
    """``sdk.open_job`` with CLI errors (exit 2)."""
    from flora_deploy import sdk
    from flora_deploy.config.schema import ConfigError

    cfg = None
    if store is None or env is None:
        if config.exists():
            try:
                from flora_deploy.config.schema import load_config

                cfg = load_config(config, allow_uncertified=True)
            except ConfigError as e:
                raise fail(str(e)) from None
        elif store is None:
            raise fail(f"{config} not found: pass --store (or FLORA_STORE) or --config")
    try:
        return sdk.open_job(job, store=store, config=cfg, environment=env)
    except (sdk.FloraError, ValueError, ImportError) as e:
        raise fail(str(e)) from None


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _dur(s: float | None) -> str:
    if s is None:
        return "-"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def print_view(v: Any, *, brief: bool = False) -> None:
    """One job status (``sdk.JobView``) for humans."""
    from rich.markup import escape

    colour = {"done": "green", "completed_with_errors": "yellow", "running": "cyan",
              "composing": "cyan", "planned": "cyan"}.get(v.state, "red")
    orch = "alive" if v.orchestrator_alive else "not running"
    console.print(f"[bold]{escape(v.job_id)}[/bold] [{colour}]{v.state}[/{colour}]  "
                  f"spend {_money(v.spend_usd)} of {_money(v.budget_usd)} (projected "
                  f"{_money(v.projected_usd)})  ETA {_dur(v.eta_s)}  active calls "
                  f"{v.active_calls}  orchestrator {orch}", soft_wrap=True)
    for kind, c in sorted(v.counts.items()):
        total = sum(c.values())
        parts = ", ".join(f"{k} {n}" for k, n in c.items() if n)
        console.print(f"  {escape(kind)}: {c.get('done', 0)}/{total} shards done ({parts})",
                      soft_wrap=True)
    if brief:
        return
    console.print(f"  config {v.config_hash[:12]}  app {escape(v.app_name)}  episodes "
                  f"{v.episodes} ({v.video_s / 3600:.3f} video-h)  spend source "
                  f"{v.spend_source}", soft_wrap=True)
    for s in v.stuck:
        console.print(f"  [yellow]stuck {escape(s['shard_id'])}: {escape(s['reason'])}[/yellow]",
                      soft_wrap=True)
    for n in v.notes[:10]:
        console.print(f"  [yellow]{escape(n)}[/yellow]", soft_wrap=True)
    if v.errors:
        console.print(f"  error files: {len(v.errors)} (see `flora report {v.job_id}`)")
    if v.composed:
        console.print(f"  composed: {v.summary_status} (read with `flora results --job "
                      f"{v.job_id} --stage <stage>`)", soft_wrap=True)
    elif v.state in ("composing", "completed_with_errors"):
        console.print(f"  not composed yet: `flora compose --job {v.job_id}` or `flora run` / "
                      "`Job.wait()` compose it", soft_wrap=True)
    if v.state == "budget_stopped":
        console.print(f"  continue with `flora resume {v.job_id} [--budget-usd X]`")
    elif v.state in ("planned", "running") and not v.orchestrator_alive:
        console.print(f"  no orchestrator holds the lease: `flora resume {v.job_id}` restarts "
                      "it (it can take a minute to start after a submit)", soft_wrap=True)


def _receipts(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise fail(f"--receipt {item!r}: use NAME=HASH (e.g. captioner_bf16=<sha256>)")
        out[name] = value
    return out


@app.command("run")
def run(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    detach: Annotated[bool, typer.Option(
        "--detach", help="Modal: start the job and return (watch with `flora status`).")
    ] = False,
    receipt: Annotated[list[str] | None, typer.Option(
        help="Modal: derived-weight receipt NAME=HASH, as given to `flora deploy` (repeatable).")
    ] = None,
    determinism: Annotated[str | None, typer.Option(
        help="Modal: the --determinism the app was deployed with (nvidia | deterministic).")
    ] = None,
    allow_uncertified: Annotated[bool, typer.Option(
        "--allow-uncertified", help="Modal: as deployed with --allow-uncertified.")] = False,
    bucket_secret: Annotated[str, typer.Option(
        help="Modal: Secret with the bucket credentials (bucket stores).")] = "flora-bucket",
    poll: Annotated[float, typer.Option(help="Modal: seconds between status polls.")] = 15.0,
    local: Annotated[bool, typer.Option(
        "--local", help="Run in this process (needs --fake-models).")] = False,
    fake_models: Annotated[bool, typer.Option(
        "--fake-models", help="Local: deterministic fake engines instead of NVIDIA's models.")
    ] = False,
    inject: Annotated[list[str] | None, typer.Option(
        "--inject", help="Local: faults, KIND:TARGET[xN] (see flora_deploy.local.faults).")
    ] = None,
    store: Annotated[str | None, typer.Option(
        "--store", help="Modal: job store URI (default: output.sink). Local: store directory "
                        "(default: <config dir>/flora-local-store).")] = None,
    job: Annotated[str | None, typer.Option("--job", help="Job id (default: project).")] = None,
    resume: Annotated[bool, typer.Option(
        "--resume", help="Local: lift a budget stop (Modal: use `flora resume <job>`).")] = False,
    budget: Annotated[float | None, typer.Option(
        "--budget", help="USD budget override (default: modal.budget_usd).")] = None,
    sample_data: Annotated[bool, typer.Option(
        "--sample-data", help="Local: write a synthetic mp4_dir dataset to input.path first.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Local: every event.")
                       ] = False,
) -> None:
    """Run a job: on the deployed Modal app (default) or locally with --local --fake-models."""
    if local:
        from flora_deploy.cli import run_local

        run_local.run(config=config, local=True, fake_models=fake_models, inject=inject,
                      store=Path(store) if store else None, job=job, resume=resume,
                      budget=budget, sample_data=sample_data, as_json=as_json, verbose=verbose)
        return
    if fake_models or inject or sample_data:
        raise fail("--fake-models / --inject / --sample-data are local-mode flags: add --local")
    if resume:
        raise fail("on Modal, continue a stopped job with `flora resume <job> [--budget-usd X]`")
    from flora_deploy import sdk
    from flora_deploy.config.schema import ConfigError, load_config

    try:
        cfg = load_config(config, allow_uncertified=allow_uncertified)
    except FileNotFoundError:
        raise fail(f"{config} not found; run `flora init` first") from None
    except ConfigError as e:
        raise fail(str(e)) from None

    def log(msg: str) -> None:
        if not as_json:
            console.print(msg, highlight=False, markup=False, soft_wrap=True)

    try:
        j = sdk.submit(cfg, job_id=job, store=store, receipts=_receipts(receipt),
                       determinism=determinism, allow_uncertified=allow_uncertified,
                       budget_usd=budget, bucket_secret=bucket_secret, log=log)
    except sdk.FloraError as e:
        raise fail(str(e)) from None
    if detach:
        v = j.status()
        if as_json:
            typer.echo(json.dumps(v.as_dict(), indent=2, sort_keys=True, default=str))
        else:
            print_view(v, brief=True)
            console.print(f"detached: watch with `flora status {j.job_id}`, stop with `flora "
                          f"cancel {j.job_id}`")
        raise typer.Exit(code=0 if v.state not in sdk.STOPPED else sdk.exit_code(v.state))

    last: list[str] = []

    def show(v) -> None:
        if as_json:
            return
        line = (v.state, json.dumps(v.counts, sort_keys=True), round(v.spend_usd, 2))
        if last != [str(line)]:
            print_view(v, brief=True)
            last[:] = [str(line)]

    try:
        v = j.wait(poll_s=poll, on_status=show)
    except sdk.FloraError as e:
        raise fail(str(e), 5) from None
    except KeyboardInterrupt:
        console.print(f"\nstopped watching; the job keeps running. `flora status {j.job_id}` / "
                      f"`flora cancel {j.job_id}`")
        raise typer.Exit(code=130) from None
    if as_json:
        typer.echo(json.dumps(v.as_dict(), indent=2, sort_keys=True, default=str))
    else:
        print_view(v)
    raise typer.Exit(code=sdk.exit_code(v.state))
