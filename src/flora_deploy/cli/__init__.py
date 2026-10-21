"""`flora` command-line interface.

One module per command group, each registering on the shared ``app`` from ``_app``:
``prepare`` (version, prepare), ``init``, ``index``, ``plan``, ``deploy``, ``run`` (with
``run_local``), ``status``, ``resume``, ``cancel``, ``report``, ``logs``, ``results`` (compose,
results), ``certify``. New commands go in their own modules and are imported here.

The ``flora`` entry point is :func:`main`: an uncaught exception (a bug, or a refusal some code
path raised instead of reporting) prints its traceback and exits **2**, never 1, which means
``failed`` (a job state); Ctrl-C exits 130.

Exit codes of every command: 0 ``done`` (or running), 1 ``failed``, 2 usage errors, refusals and
internal errors, 3 ``completed_with_errors``, 4 ``budget_stopped``, 5 any other stop
(``cancelled``; local runs also ``interrupted`` and ``lease_held``).
"""

import sys
import traceback

from flora_deploy.cli import (  # noqa: F401  (registers commands)
    cancel,
    certify,
    deploy,
    doctor,
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

__all__ = ["app", "main"]


def main(argv: list[str] | None = None) -> None:
    """``flora``: :data:`app` with exit code 2 for uncaught exceptions (module docstring)."""
    try:
        app(args=argv, prog_name="flora")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - mapped to the refusal exit code
        traceback.print_exc()
        print(f"flora: internal error: {type(exc).__name__}: {exc} (exit 2)", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
