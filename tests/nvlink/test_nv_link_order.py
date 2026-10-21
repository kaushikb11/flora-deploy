"""The nv test suites pass in either order (no state leaks between unit fakes and real links).

The unit linker tests link fake families in-process and the nvlink tests link NVIDIA's real
SEGCAP family in-process; each must leave ``sys.modules``, ``sys.path``, ``sys.meta_path`` and the
linker state as it found them. The CI-style check, run here on the fast subset in fresh
interpreters (no pytest cache, so no reordering by earlier failures)::

    uv run pytest -q -p no:cacheprovider tests/unit/test_nv_* tests/nvlink/test_nv_link_*
    uv run pytest -q -p no:cacheprovider tests/nvlink/test_nv_link_* tests/unit/test_nv_*
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.nvlink

ROOT = Path(__file__).resolve().parents[2]
UNIT = ["tests/unit/test_nv_linker.py", "tests/unit/test_nv_capture.py",
        "tests/unit/test_nv_budget.py", "tests/unit/test_nv_session.py"]
NVLINK = ["tests/nvlink/test_nv_link_bootstrap.py", "tests/nvlink/test_nv_link_segmenter.py",
          "tests/nvlink/test_nv_link_pool.py"]


@pytest.mark.parametrize("order", ["unit-then-nvlink", "nvlink-then-unit"])
def test_suites_pass_in_either_order(nv_code, order):
    nv_code("tempseg_code")  # skip when NVIDIA's code is not fetched
    files = UNIT + NVLINK if order == "unit-then-nvlink" else NVLINK + UNIT
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *files],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    tail = "\n".join(out.stdout.splitlines()[-15:])
    assert out.returncode == 0, tail + out.stderr[-2000:]
    assert " passed" in tail and "error" not in tail.lower(), tail
