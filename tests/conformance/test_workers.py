"""Runs every conformance check against every registered worker, and proves every check fails
on the broken variants built to violate it."""

from __future__ import annotations

import pytest

from tests.conformance.reference_worker import BROKEN, fake_spec
from tests.conformance.registry import WORKERS
from tests.conformance.worker_suite import CHECKS


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize("spec", WORKERS, ids=str)
def test_worker_conforms(spec, check, tmp_path):
    CHECKS[check](spec, tmp_path / "store")


@pytest.mark.parametrize(("check", "variant", "broken"),
                         [(c, n, b) for c, vs in BROKEN.items() for n, b in vs],
                         ids=lambda v: v if isinstance(v, str) else "")
def test_check_catches_broken_worker(check, variant, broken, tmp_path):
    with pytest.raises(AssertionError):
        CHECKS[check](fake_spec("segcap", broken, name=variant), tmp_path / "store")


def test_every_check_has_a_broken_variant():
    assert set(BROKEN) == set(CHECKS)
