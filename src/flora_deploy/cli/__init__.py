"""`flora` command-line interface.

One module per command group, each registering on the shared ``app`` from ``_app``:
``prepare`` (version, prepare). New commands (init, index, plan, deploy, run, status, resume,
results, report, cancel) go in their own modules and are imported here.
"""

from flora_deploy.cli import (  # noqa: F401  (registers commands)
    deploy,
    index,
    init,
    plan,
    prepare,
    results,
)
from flora_deploy.cli._app import app

__all__ = ["app"]

if __name__ == "__main__":
    app()
