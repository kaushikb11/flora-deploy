import os
import re
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname


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
