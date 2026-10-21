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
