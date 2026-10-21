"""`flora resume <job> [--budget-usd X]`: continue a job on the deployed app.

Lifts a ``budget_stop`` (writes a ``resume`` record; ``--budget-usd`` raises the budget in the
job's run spec, which the deployed orchestrator reads) and spawns the orchestrator again when
none holds the lease (a dead orchestrator). With a live orchestrator it only writes the records.
Refuses ``done``, ``completed_with_errors``, ``failed`` and ``cancelled`` jobs, and a budget
stop whose projection is still over the budget without ``--budget-usd``.

A local job (``flora run --local``) is resumed in this process: its orchestrator runs here until
the job stops, then this prints its status and exits with its code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console
from flora_deploy.cli.run import ConfigOpt, EnvOpt, JobArg, StoreOpt, fail, open_job_cli


@app.command()
def resume(
    job: JobArg,
    budget_usd: Annotated[float | None, typer.Option(
        "--budget-usd", help="New job budget in USD (not below the spend so far).")] = None,
    store: StoreOpt = None, config: ConfigOpt = Path("flora.yaml"), env: EnvOpt = None,
) -> None:
    """Resume a budget-stopped job or restart a dead orchestrator."""
    from flora_deploy import sdk

    j = open_job_cli(job, store, config, env)
    try:
        call_id = j.resume(budget_usd=budget_usd)
    except sdk.FloraError as e:
        raise fail(str(e)) from None
    budget = f" with budget ${j.settings.budget_usd:,.2f}" if budget_usd is not None else ""
    if j.is_local and call_id is not None:
        from flora_deploy.cli.run import print_view

        v = j.status()
        console.print(f"resumed {job}{budget} locally")
        print_view(v)
        raise typer.Exit(code=v.exit_code)
    if call_id is None:
        console.print(f"resumed {job}{budget}; its orchestrator is still running")
    else:
        console.print(f"resumed {job}{budget}: orchestrator {call_id} on {j.app_name}. Watch "
                      f"with `flora status {job}`", soft_wrap=True)
