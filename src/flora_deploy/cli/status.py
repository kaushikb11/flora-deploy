"""`flora status <job>`: the job's state rebuilt from the store's files (``jobs.reconciler``).

State (``done``, ``completed_with_errors``, ``failed``, ``budget_stopped``, ``cancelled``,
``running``, ``composing``, ``planned``), per-kind shard progress, stuck calls, error files,
spend so far vs the budget (the real budget gate: container receipts, the call history, pinned
prices), the projected total and an ETA from ``planner/rates.json`` calibrated by finished
calls. Read-only: it never restarts anything (``flora resume`` does).

Exit codes as ``flora run``: 0 while running or ``done``, 1 ``failed``, 3
``completed_with_errors``, 4 ``budget_stopped``, 5 ``cancelled``, 2 job not found.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from flora_deploy.cli._app import app
from flora_deploy.cli.run import (
    ConfigOpt,
    EnvOpt,
    JobArg,
    JsonOpt,
    StoreOpt,
    fail,
    open_job_cli,
    print_view,
)


@app.command()
def status(job: JobArg = None, store: StoreOpt = None,
           config: ConfigOpt = Path("flora.yaml"), env: EnvOpt = None,
           as_json: JsonOpt = False) -> None:
    """Show a job's state, progress, errors, spend vs budget and ETA."""
    from flora_deploy import sdk

    j = open_job_cli(job, store, config, env)
    try:
        v = j.status()
    except (sdk.FloraError, FileNotFoundError, ValueError) as e:
        raise fail(str(e)) from None
    if as_json:
        typer.echo(json.dumps(v.as_dict(), indent=2, sort_keys=True, default=str))
    else:
        print_view(v)
    raise typer.Exit(code=v.exit_code)
