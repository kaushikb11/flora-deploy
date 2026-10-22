"""`flora` command-line interface.

One module per command group, each registering on the shared ``app`` from ``_app``:
``prepare`` (version, prepare), ``init``, ``index``, ``plan``, ``deploy``, ``run`` (with
``run_local``), ``status``, ``resume``, ``cancel``, ``report``, ``logs``, ``results`` (compose,
results), ``certify``. New commands go in their own modules and are imported here.

The ``flora`` entry point is :func:`main`: an uncaught exception exits **2**, never 1, which
means ``failed`` (a job state); Ctrl-C exits 130. An expected condition (no Modal credentials,
Modal unreachable, a missing Volume or app: :func:`expected_error`) prints one line with the
fix; anything else is a bug and prints its traceback. ``FLORA_DEBUG=1`` always shows it.

Exit codes of every command: 0 ``done`` (or running), 1 ``failed``, 2 usage errors, refusals and
internal errors, 3 ``completed_with_errors``, 4 ``budget_stopped``, 5 any other stop
(``cancelled``; local runs also ``interrupted`` and ``lease_held``).
"""

import os
import sys
import traceback

from flora_deploy.cli import (  # noqa: F401  (registers commands)
    cancel,
    certify,
    deploy,
    doctor,
    export,
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
from flora_deploy.cli._app import app, arrange_help

arrange_help(app)  # `flora --help` panels (Get started / Run / Inspect results / ...)

__all__ = ["app", "main"]


def expected_error(exc: BaseException) -> str | None:
    """One line, with the fix, for an exception that is a condition of the user's setup rather
    than a bug; ``None`` for anything else."""
    import re

    name, mod, msg = type(exc).__name__, type(exc).__module__, str(exc).strip()
    if mod.startswith("modal"):
        if name == "AuthError":
            return f"Modal rejected the credentials ({msg}); run `modal setup`"
        if name == "NotFoundError":
            vol = re.search(r"Volume '([^']+)' not found", msg)
            if vol:
                return (f"{msg}. Create it with `modal volume create {vol.group(1)} --version "
                        "2` (`flora deploy` also creates it), or pass --store file:///some/dir")
            if re.search(r"\b(App|Function|Cls)\b", msg):
                return f"{msg}. Run `flora deploy` first"
            return msg
        if name in ("ConnectionError", "TimeoutError"):
            return f"could not reach Modal ({msg}); check your network and retry"
        if name == "PermissionDeniedError":
            return f"Modal refused: {msg}"
    if name in ("NoCredentialsError", "PartialCredentialsError"):
        return f"{msg}: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or use a volume:// sink)"
    if isinstance(exc, PermissionError):
        return f"permission denied: {exc.filename or msg}"
    if isinstance(exc, ConnectionError | TimeoutError):
        return f"network error ({name}: {msg}); check your connection and retry"
    return None


def main(argv: list[str] | None = None) -> None:
    """``flora``: :data:`app` with exit code 2 for uncaught exceptions (module docstring)."""
    try:
        app(args=argv, prog_name="flora")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - mapped to the refusal exit code
        line = expected_error(exc)
        if line is None or os.environ.get("FLORA_DEBUG"):
            traceback.print_exc()
        print(f"flora: {line} (exit 2)" if line else
              f"flora: internal error: {type(exc).__name__}: {exc} (exit 2)", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
