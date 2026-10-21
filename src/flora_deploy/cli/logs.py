"""`flora logs <job> [--kind K] [--follow]`: the deployed app's logs (``modal app logs``).

Without ``--kind``: the whole app (shared by every job of the same config). ``--kind
orchestrator``: the job's orchestrator launches. ``--kind <shard kind>`` (e.g. ``segcap``): the
calls the job's ledgers recorded for that kind, live ones first (the 5 most recent; ``--follow``
streams the newest).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flora_deploy.cli._app import app
from flora_deploy.cli.run import ConfigOpt, EnvOpt, JobArg, StoreOpt, fail, open_job_cli


@app.command()
def logs(
    job: JobArg,
    kind: Annotated[str | None, typer.Option(
        "--kind", help="orchestrator, or a shard kind (segcap, reward.terminal, ...).")] = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Stream new lines.")] = False,
    tail: Annotated[int, typer.Option("--tail", "-n", help="Lines per call (not with -f).")
                    ] = 100,
    store: StoreOpt = None, config: ConfigOpt = Path("flora.yaml"), env: EnvOpt = None,
) -> None:
    """Show a job's Modal logs."""
    from flora_deploy import sdk

    j = open_job_cli(job, store, config, env)
    try:
        for line in j.logs(kind=kind, follow=follow, tail=tail):
            typer.echo(line)
    except sdk.FloraError as e:
        raise fail(str(e)) from None
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
