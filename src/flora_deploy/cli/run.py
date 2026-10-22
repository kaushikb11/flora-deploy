"""`flora run`: start a job on the deployed app (default) or run it locally (``--local``).

* **Modal** (default): ``sdk.submit`` plans the config's app offline, refuses when it is not
  deployed or was deployed for another config hash (pass the same ``--receipt`` /
  ``--determinism`` as ``flora deploy``), plans the job from ``flora index``'s receipts, writes
  the run spec and spawns the deployed orchestrator. Then it streams status until the job stops
  and composes the outputs, unless ``--detach``.
* **Local** (``--local --fake-models``): ``cli/run_local.py`` (G2): the whole job system in this
  process with fake models; same flags and exit codes as before.
* **Modal, fake models** (``--fake-models`` without ``--local``): submit to the app
  ``flora deploy --fake-models`` deployed (the H0 CPU rehearsal: the real deployed path with
  G2's fake engines; its own config hash, so it never touches real outputs).

Fake-models jobs (both modes) run under the job id ``<job>.fake`` (``jobs.store.fake_job_id``),
so they never share a manifest pointer, run spec, index or lease with the real job of the same
id; ``flora status`` / ``flora results`` refuse to read one as the other (review 8b M5).

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

JobArg = Annotated[str | None, typer.Argument(
    help="Job id (default: the config's project; its fake-models run <project>.fake when "
         "there is no real one).", show_default=False)]
StoreOpt = Annotated[str | None, typer.Option(
    "--store", envvar="FLORA_STORE",
    help="Job store: modal-volume://name\\[/path], s3://bucket\\[/prefix], file:///dir or a "
         "directory (default: the config's output.sink, then <config dir>/flora-local-store).")]
ConfigOpt = Annotated[Path, typer.Option(
    "--config", "-c", help="flora.yaml (finds the store and Modal environment).")]
EnvOpt = Annotated[str | None, typer.Option(
    "--env", envvar="MODAL_ENVIRONMENT", help="Modal environment of a modal-volume store.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON.")]


def fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]", soft_wrap=True)
    return typer.Exit(code=code)


def open_job_cli(job: str | None, store: str | None, config: Path, env: str | None):
    """The job a job-scoped command acts on (``cli._jobref.resolve``: the job id and store
    default from ``flora.yaml``), opened with ``sdk.open_job_in``; CLI errors exit 2. Prints the
    resolution's notes (fake-models run, ...) on stderr."""
    from flora_deploy import sdk
    from flora_deploy.cli._jobref import ResolveError, print_notes, resolve
    from flora_deploy.config.schema import ConfigError

    try:
        r = resolve(job, store, config, env)
    except ConfigError as e:
        raise fail(str(e)) from None
    except ResolveError as e:
        raise fail(str(e)) from None
    print_notes(r)
    try:
        return sdk.open_job_in(r.store, r.job_id, store_uri=r.store_uri)
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


# What `flora status` / `flora report` say instead of dollars for a `flora run --local` job.
SIMULATED_SPEND = "spend $0 (n/a: local run, fake models; nothing charged)"


def print_view(v: Any, *, brief: bool = False) -> None:
    """One job status (``sdk.JobView``) for humans."""
    from rich.markup import escape

    colour = {"done": "green", "completed_with_errors": "yellow", "running": "cyan",
              "composing": "cyan", "planned": "cyan"}.get(v.state, "red")
    orch = "alive" if v.orchestrator_alive else "not running"
    spend = (SIMULATED_SPEND if getattr(v, "simulated", False) else
             f"spend {_money(v.spend_usd)} of {_money(v.budget_usd)} (projected "
             f"{_money(v.projected_usd)})")
    console.print(f"[bold]{escape(v.job_id)}[/bold] [{colour}]{v.state}[/{colour}]  "
                  f"{spend}  ETA {_dur(v.eta_s)}  active calls "
                  f"{v.active_calls}  orchestrator {orch}", soft_wrap=True)
    for kind, c in sorted(v.counts.items()):
        total = sum(c.values())
        parts = ", ".join(f"{k} {n}" for k, n in c.items() if n)
        console.print(f"  {escape(kind)}: {c.get('done', 0)}/{total} shards done ({parts})",
                      soft_wrap=True)
    if brief:
        return
    console.print(f"  config {v.config_hash[:12]}  app {escape(v.app_name)}  episodes "
                  f"{v.episodes} ({v.video_s / 3600:.3f} video-h)" + (
                      "" if getattr(v, "simulated", False)
                      else f"  spend source {v.spend_source}"), soft_wrap=True)
    for s in v.stuck:
        console.print(f"  [yellow]stuck {escape(s['shard_id'])}: {escape(s['reason'])}[/yellow]",
                      soft_wrap=True)
    for n in v.notes[:10]:
        console.print(f"  [yellow]{escape(n)}[/yellow]", soft_wrap=True)
    if v.errors:
        console.print(f"  error files: {len(v.errors)} (see `flora report {v.job_id}`)")
    if v.composed:
        console.print(f"  composed: {v.summary_status} (read with `flora results <stage> "
                      f"--job {v.job_id}`)", soft_wrap=True)
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
        help="Modal: derived-weight receipt NAME=HASH (repeatable); overrides weights.derived "
             "in flora.yaml, as for `flora deploy`.")
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
        "--fake-models", help="Deterministic fake engines instead of NVIDIA's models (local, or "
                              "on Modal: the app `flora deploy --fake-models` deployed).")
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
        "--sample-data", help="Local: write a synthetic dataset (as input.format says) to "
                              "input.path first.")
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
    if inject or sample_data:
        raise fail("--inject / --sample-data are local-mode flags: add --local")
    if resume:
        raise fail("on Modal, continue a stopped job with `flora resume <job> [--budget-usd X]`")
    from flora_deploy import sdk
    from flora_deploy.config.receipts import load_certified
    from flora_deploy.config.schema import ConfigError

    try:
        cfg = load_certified(config, allow_uncertified=allow_uncertified,
                             overrides=_receipts(receipt))[0]
    except FileNotFoundError:
        raise fail(f"{config} not found; run `flora init` first") from None
    except ConfigError as e:
        raise fail(str(e)) from None

    def log(msg: str) -> None:
        if not as_json:
            console.print(msg, highlight=False, markup=False, soft_wrap=True)

    planner = None
    if fake_models:
        import functools

        from flora_deploy.modal_app.factory import plan_app

        planner = functools.partial(plan_app, fake_models=True)
    try:
        j = sdk.submit(cfg, job_id=job, store=store, receipts=_receipts(receipt),
                       determinism=determinism, allow_uncertified=allow_uncertified,
                       budget_usd=budget, bucket_secret=bucket_secret, log=log,
                       planner=planner, fake_models=fake_models)
    except sdk.NotDeployed as e:
        if fake_models:
            raise fail(f"{e} (here: `flora deploy --fake-models`); or add --local to run the "
                       "fake models in this process") from None
        raise fail(str(e)) from None
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
