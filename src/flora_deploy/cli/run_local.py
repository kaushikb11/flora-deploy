"""`flora run --local --fake-models`: a whole job on this machine (``flora_deploy.local``).

The ``flora run`` command itself is registered by ``cli/run.py``, which dispatches ``--local``
here (one command: Modal by default, local with ``--local``).

index → plan → orchestrate → compose with the real orchestrator, stores, worker runners and
compose, deterministic fake models, no Modal and no GPU. Results go to a ``LocalStore`` directory
(``--store``, default ``<config dir>/flora-local-store``) and are read with ``flora results``.

Exit codes: 0 ``done``; 1 ``failed``; 3 ``completed_with_errors`` (records for some episodes are
excused: see the errors in the summary); 4 ``budget_stopped``; 5 any other stop; 2 usage errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import console

EXIT = {"done": 0, "failed": 1, "completed_with_errors": 3, "budget_stopped": 4}


def _fail(msg: str, code: int = 2) -> typer.Exit:
    from rich.markup import escape

    console.print(f"[red]{escape(msg)}[/red]")
    return typer.Exit(code=code)


def run(
    config: Annotated[Path, typer.Option("--config", "-c", help="flora.yaml")] = Path("flora.yaml"),
    local: Annotated[bool, typer.Option(
        "--local", help="Run in this process (the only mode this command has so far).")] = False,
    fake_models: Annotated[bool, typer.Option(
        "--fake-models", help="Deterministic fake engines instead of NVIDIA's models "
                              "(required with --local: no GPU).")] = False,
    inject: Annotated[list[str] | None, typer.Option(
        "--inject", help="Faults, KIND:TARGET[xN], comma-separated or repeated: "
                         "preempt|fail|infra|spawnerr:SHARD, oom|nverr:EPISODE[@kind] "
                         "(see flora_deploy.local.faults).")] = None,
    store: Annotated[Path | None, typer.Option(
        "--store", help="Local store directory (default: <config dir>/flora-local-store).")
    ] = None,
    job: Annotated[str | None, typer.Option("--job", help="Job id (default: project).")] = None,
    resume: Annotated[bool, typer.Option(
        "--resume", help="Lift a budget stop (with --budget to raise the budget).")] = False,
    budget: Annotated[float | None, typer.Option(
        "--budget", help="USD budget override (default: modal.budget_usd).")] = None,
    sample_data: Annotated[bool, typer.Option(
        "--sample-data", help="Write a synthetic mp4_dir dataset to input.path first, if it "
                              "does not exist.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print every event.")] = False,
) -> None:
    """Run a whole job locally: index, plan, orchestrate, compose (no Modal, no GPU)."""
    from rich.markup import escape

    from flora_deploy.config.schema import ConfigError, load_config
    from flora_deploy.local import LocalRunError, parse_faults, run_local, sample_dataset
    from flora_deploy.local.runner import summary_lines

    if not local:
        raise _fail("`flora run` on Modal is not wired yet; use `flora run --local "
                    "--fake-models` for a local run")
    if not fake_models:
        raise _fail("--local needs --fake-models: NVIDIA's models need GPUs (deploy to Modal "
                    "for real outputs)")
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        raise _fail(f"{config} not found; run `flora init --template local_demo` first") from None
    except ConfigError as e:
        raise _fail(str(e)) from None
    try:
        faults = parse_faults(inject)
    except ValueError as e:
        raise _fail(str(e)) from None
    if sample_data:
        if cfg.input.repo is not None or cfg.input.format != "mp4_dir":
            raise _fail("--sample-data writes an mp4_dir dataset; set input.format: mp4_dir "
                        "and a local input.path")
        p = Path(str(cfg.input.path)).expanduser()
        root = p if p.is_absolute() else config.parent / p
        if not root.exists():
            sample_dataset(root)
            console.print(f"sample data: {escape(str(root))}")

    def log(msg: str) -> None:
        if not as_json:
            console.print(msg, highlight=False, markup=False, soft_wrap=True)

    from flora_deploy.local.faults import FaultPlan

    try:
        res = run_local(cfg, config_path=config, store=store, job_id=job,
                        faults=FaultPlan(faults), resume=resume, budget_usd=budget, log=log,
                        verbose=verbose)
    except LocalRunError as e:
        raise _fail(str(e)) from None
    if as_json:
        typer.echo(json.dumps(res.as_dict(), indent=2, sort_keys=True, default=str))
    else:
        colour = {"done": "green", "completed_with_errors": "yellow"}.get(res.state, "red")
        lines = summary_lines(res)
        console.print(f"[{colour}]{escape(lines[0])}[/{colour}]", soft_wrap=True)
        for line in lines[1:]:
            console.print(line, highlight=False, markup=False, soft_wrap=True)
    raise typer.Exit(code=EXIT.get(res.state, 5))
