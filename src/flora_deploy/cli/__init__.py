"""`flora` command-line interface.

One module per command group, each registering on the shared ``app`` from ``_app``:
``prepare`` (version, prepare), ``init``, ``index``, ``plan``, ``deploy``, ``run`` (with
``run_local``), ``status``, ``resume``, ``cancel``, ``report``, ``logs``, ``results`` (compose,
results), ``certify``. New commands go in their own modules and are imported here.
"""

from flora_deploy.cli import (  # noqa: F401  (registers commands)
    cancel,
    certify,
    deploy,
    index,
    init,
    logs,
    plan,
    prepare,
    report,
    results,
    resume,
    run,  # `flora run` (Modal; --local dispatches to run_local)
    run_local,
    status,
)
from flora_deploy.cli._app import app

__all__ = ["app"]

if __name__ == "__main__":
    app()
