"""Every Store implementation passes every store check; every check catches its broken store."""

from __future__ import annotations

import pytest

from flora_deploy.jobs.stores import VolumeStore
from tests.conformance.store_suite import BROKEN_STORES, CHECKS, STORES, DictStore


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize("name", sorted(STORES))
def test_store_conforms(name, check, tmp_path):
    CHECKS[check](STORES[name](tmp_path))


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_reference_dict_store_conforms(check):
    CHECKS[check](DictStore())


@pytest.mark.parametrize(("check", "variant", "make"),
                         [(c, n, m) for c, vs in BROKEN_STORES.items() for n, m in vs],
                         ids=lambda v: v if isinstance(v, str) else "")
def test_check_catches_broken_store(check, variant, make):
    with pytest.raises(AssertionError):
        CHECKS[check](make())


def test_every_store_check_has_a_broken_variant():
    assert set(BROKEN_STORES) == set(CHECKS)


def test_volume_store_commits_every_write(tmp_path):
    """Volume v2: other containers see a write only after ``commit`` (default: every write)."""
    s = STORES["volume"](tmp_path)
    assert isinstance(s, VolumeStore)
    for check in CHECKS.values():
        check(s)
    assert s.volume.commits >= 1 and s.pending_writes == 0
