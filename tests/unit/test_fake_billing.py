"""A ``--fake-models`` job is priced at the CPU containers it runs on, not the pool GPUs
(``RunSettings.fake_models``, ``jobs.settings.billed_pool``, ``jobs.budget.pool_rate``)."""

from __future__ import annotations

from flora_deploy.jobs import budget as B
from flora_deploy.jobs.settings import (
    FAKE_WORKER_CPU,
    FAKE_WORKER_MEMORY_GIB,
    PoolSettings,
    RunSettings,
    billed_pool,
)
from flora_deploy.modal_app import factory as F


def settings(**kw) -> RunSettings:
    return RunSettings(job_id="j", preset="exact", config_hash="c", budget_usd=10.0, max_gpus=8,
                       pools={"segcap": PoolSettings("H200", max_containers=2)}, **kw)


def test_constants_match_the_factory():
    assert (FAKE_WORKER_CPU, FAKE_WORKER_MEMORY_GIB) == (F.FAKE_WORKER_CPU,
                                                         F.FAKE_WORKER_MEMORY_GIB)


def test_fake_jobs_pay_cpu_not_gpu():
    rates = B.pinned_rates()
    real, fake = settings(), settings(fake_models=True)
    assert billed_pool(real, "segcap") == real.pools["segcap"]
    cpu = rates.per_second(None, 0, FAKE_WORKER_CPU, FAKE_WORKER_MEMORY_GIB)
    assert B.pool_rate(rates, fake, "segcap") == cpu
    assert B.pool_rate(rates, real, "segcap") == rates.pool_per_second(real.pools["segcap"])
    assert cpu < B.pool_rate(rates, real, "segcap") / 10
    calls = [B.CallRecord("c1", "s1", "segcap", spawned_at=0.0, ended_at=100.0)]
    assert B.calls_cost(calls, fake, rates, 100.0, scaledown_s=0.0) == 100.0 * cpu


def test_settings_round_trip_keeps_the_flag():
    from flora_deploy import sdk

    s = sdk.settings_from_dict(sdk.settings_to_dict(settings(fake_models=True)))
    assert s.fake_models
    assert not sdk.settings_from_dict({k: v for k, v in sdk.settings_to_dict(settings()).items()
                                       if k != "fake_models"}).fake_models


def test_report_containers_of_a_fake_job_are_priced_as_cpu(monkeypatch):
    """``flora report``'s per-container rows (``jobs.containers``) price a fake job's workers at
    the CPU rate ``pool_rate`` charges, not the pool's GPU list price."""
    from flora_deploy.jobs import containers as C
    from tests.fakes.jobs import make_manifest
    from tests.fakes.modal import FakeStore

    seen = {}

    def capture(markers, receipts, spawned, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(C, "container_rows", capture)
    rates = B.pinned_rates()
    fake = settings(fake_models=True)
    C.job_container_rows(FakeStore(), "j", "c", make_manifest({}), pools=fake.pools,
                         scaledown_s=2.0, rates=rates, settings=fake)
    assert seen["usd_per_s"]("segcap") == B.pool_rate(rates, fake, "segcap")
    assert seen["pool_gpus"]["segcap"] in ("", None)  # no GPU on a fake worker
    real = settings()
    C.job_container_rows(FakeStore(), "j", "c", make_manifest({}), pools=real.pools,
                         scaledown_s=2.0, rates=rates, settings=real)
    assert seen["usd_per_s"]("segcap") == rates.pool_per_second(real.pools["segcap"])
    assert seen["pool_gpus"]["segcap"] == "H200"
