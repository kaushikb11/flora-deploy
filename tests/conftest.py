import os
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname


def _nv_root() -> Path:
    return Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()


@pytest.fixture
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
