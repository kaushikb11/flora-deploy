"""`flora cancel <job>`: stop a job for good.

Writes a ``cancel`` control record. The running orchestrator cancels its in-flight calls on its
next tick and ends ``cancelled``; with no live orchestrator the recorded live calls are cancelled
here (``terminate_containers=True``). A cancelled job cannot be resumed; parts already written
stay in the store (``flora compose --partial`` reads them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app, console
from flora_deploy.cli.run import ConfigOpt, EnvOpt, JobArg, StoreOpt, fail, open_job_cli


@app.command()
def cancel(
    job: JobArg,
    reason: Annotated[str, typer.Option(help="Recorded with the cancel.")] = "operator cancel",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")
                   ] = False,
    store: StoreOpt = None, config: ConfigOpt = Path("flora.yaml"), env: EnvOpt = None,
) -> None:
    """Cancel a job (its in-flight calls are cancelled; it cannot be resumed)."""
    from flora_deploy import sdk

    j = open_job_cli(job, store, config, env)
    try:
        state = j.status().state
    except sdk.FloraError as e:
        raise fail(str(e)) from None
    if state in sdk.TERMINAL:
        raise fail(f"job {job} is already {state}; nothing to cancel")
    try:
        ok = yes or typer.confirm(f"Cancel job {job} ({state})? It cannot be resumed.")
    except typer.Abort:  # no answer (no terminal): declined
        ok = False
    if not ok:
        raise typer.Exit(code=2)  # declined: a refusal, not a failed job
    try:
        cancelled = j.cancel(reason)
    except sdk.FloraError as e:
        raise fail(str(e)) from None
    if j.is_local:
        console.print(f"cancel recorded for {job} (a local job: a running `flora run --local` "
                      "stops on its next tick). Check with `flora status`", soft_wrap=True)
    elif cancelled:
        console.print(f"cancelled {job}: no orchestrator was running; cancelled "
                      f"{len(cancelled)} live call(s)")
    else:
        console.print(f"cancel recorded for {job}; the orchestrator stops its calls on its next "
                      "tick. Check with `flora status`", soft_wrap=True)
