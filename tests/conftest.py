import os

# CLI output must not depend on where the tests run. On GitHub Actions (GITHUB_ACTIONS) or with
# FORCE_COLOR / PY_COLORS set, typer and rich force a colour terminal and wrap at the runner's
# width, so plain-text assertions on --help and tables fail only in CI. Set before anything
# imports typer/rich (they read these at import time); xdist workers load this file too.
for _var in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE"):
    os.environ.pop(_var, None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "200"
os.environ["TERMINAL_WIDTH"] = "200"

import re  # noqa: E402  (after the environment above, on purpose)
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from flora_deploy.config.pins import PINS_BY_KEY  # noqa: E402
from flora_deploy.layout import (  # noqa: E402
    NV_CODE_CACHE_DEFAULT,
    NV_CODE_CACHE_ENV,
    snapshot_dirname,
)


def _nv_root() -> Path:
    return Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()


@pytest.fixture(scope="session")  # stateless; session scope lets module fixtures use it
def nv_code():
    """Return the local dir of a fetched NVIDIA code pin, or skip.

    Usage: ``nv_code("tempseg_code")`` / ``nv_code("captioner_code")`` / ``nv_code("rewarder")``.
    Fetch with ``uv run python scripts/fetch_nv_code.py [--with-processor]``.
    """

    def get(key: str) -> Path:
        path = _nv_root() / snapshot_dirname(PINS_BY_KEY[key])
        if not (path / "batch_utils").is_dir():
            pytest.skip(f"NVIDIA code pin {key!r} not fetched (scripts/fetch_nv_code.py)")
        return path

    return get


# Failures of tests that run NVIDIA's code are redacted (tests/nvtext.py, review 11b F3): CI logs
# are public, and a traceback through batch_utils or an assertion on a rendered prompt would print
# NVIDIA's licensed text. FLORA_SHOW_NV_OUTPUT=1 shows the full report locally.
REDACT_MARKERS = ("nvlink", "torch")
_NVLINK_DIR = Path(__file__).parent / "nvlink"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    from tests import nvtext

    outcome = yield
    rep = outcome.get_result()
    if (rep.failed and not nvtext.show()
            and any(item.get_closest_marker(m) for m in REDACT_MARKERS)):
        if call.excinfo is not None:
            rep.longrepr = nvtext.redact_excinfo(call.excinfo)
        rep.sections = [(name, "[captured output hidden]") for name, _ in rep.sections]


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    from tests import nvtext

    outcome = yield
    rep = outcome.get_result()
    path = Path(str(getattr(collector, "path", "")))
    if rep.failed and not nvtext.show() and path.is_relative_to(_NVLINK_DIR):
        rep.longrepr = (f"collection of {path.name} failed; report redacted (tests/nvtext.py, "
                        f"set {nvtext.SHOW_ENV}=1 locally)")
        rep.sections = []


ORDER_ENV = "FLORA_TEST_ORDER"


def pytest_collection_modifyitems(config, items):
    """``FLORA_TEST_ORDER=reverse`` runs the collected tests in reverse order (CI's order gate,
    ``docs/ci.md``): a test that only passes after another one has run is a leak, not a pass."""
    order = os.environ.get(ORDER_ENV, "")
    if order == "reverse":
        items.reverse()
    elif order:
        raise pytest.UsageError(f"{ORDER_ENV}={order!r}; expected 'reverse' or unset")


FAIL_ON_SKIP_ENV = "FLORA_FAIL_ON_SKIP"
# Skip reasons that mean "the thing under test was not there", not "this case does not apply".
# CI's nvlink job sets FLORA_FAIL_ON_SKIP=1, so a missing dependency or NVIDIA file fails the job
# instead of going green with the tests silently skipped (review 7 G4-1/G4-2, docs/ci.md).
MISSING_SKIP = re.compile(r"could not import|No module named|not fetched", re.IGNORECASE)
_missing_skips: list[str] = []


def _note_skip(report) -> None:
    if report.skipped and isinstance(report.longrepr, tuple):
        reason = str(report.longrepr[2])
        if MISSING_SKIP.search(reason):
            _missing_skips.append(f"{report.nodeid or report.longrepr[0]}: {reason}")


def pytest_runtest_logreport(report):
    _note_skip(report)


def pytest_collectreport(report):  # a module-level importorskip skips at collection
    _note_skip(report)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):  # xdist controller: merge what each worker saw
    _missing_skips.extend(getattr(node, "workeroutput", {}).get("flora_missing_skips", []))


def pytest_sessionfinish(session, exitstatus):
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:  # an xdist worker: hand the list to the controller
        workeroutput["flora_missing_skips"] = list(dict.fromkeys(_missing_skips))
        return
    if os.environ.get(FAIL_ON_SKIP_ENV) != "1":
        return
    missing = list(dict.fromkeys(_missing_skips))
    if missing:
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            tr.write_sep("=", f"{FAIL_ON_SKIP_ENV}: {len(missing)} test(s) skipped for a "
                         "missing dependency or NVIDIA file", red=True)
            for line in missing:
                tr.write_line(line)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def tempseg_pinned(monkeypatch):
    """A stand-in ``DERIVED_PINS["tempseg-merged"]`` content digest. Until merger v2 is built
    and pinned, ``plan_app`` refuses a real balanced ``segment`` plan (review 10 M1); tests of
    the planned app itself use this."""
    from dataclasses import replace

    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

    pin = DERIVED_PINS_BY_KEY["tempseg-merged"]
    if pin.content_sha256 is None:
        pin = replace(pin, content_sha256="c" * 64)
        monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", pin)
    return pin
