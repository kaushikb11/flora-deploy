"""Spend math (jobs/budget.py) against hand calculations at Modal list prices."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from flora_deploy.jobs.budget import (
    PINNED_PRICES_SOURCE,
    Rates,
    billed_seconds,
    container_cost,
    gpu_key,
    load_rates,
    pinned_rates,
    rates_from_mapping,
    read_container_receipts,
    region_multiplier_for,
    spend,
)
from flora_deploy.jobs.orchestrator import utc
from flora_deploy.jobs.settings import PoolSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import ContainerReceipt

T0 = 1_800_000_000.0
H200_CONTAINER_HOUR = 4.54 + 8 * 0.0472 + 96 * 0.008  # = 5.6856, docs/exact-concurrency.md


def receipt(cid="c1", *, start=T0, hb=None, stop=None, gpu="H200", n=1, cpu=8.0, mem=96.0,
            preemptible=True) -> ContainerReceipt:
    return ContainerReceipt(
        container_id=cid, app="flora-workers", gpu=gpu, gpu_count=n, cpu_request=cpu,
        cpu_limit=None, memory_request_gib=mem, memory_limit_gib=None, region=None,
        preemptible=preemptible, started_utc=utc(start), heartbeat_utc=utc(hb or start),
        stopped_utc=utc(stop) if stop is not None else None)


def test_pinned_container_rate_matches_doc():
    r = pinned_rates()
    assert r.pool_per_second(PoolSettings(gpu="H200", cpu_cores=8, memory_gib=96)) * 3600 == \
        pytest.approx(H200_CONTAINER_HOUR)
    assert H200_CONTAINER_HOUR == pytest.approx(5.6856)
    assert r.source == PINNED_PRICES_SOURCE


def test_stopped_container_one_hour():
    r = receipt(stop=T0 + 3600)
    assert billed_seconds(r, now=T0 + 99_999) == pytest.approx(3600)
    assert container_cost(r, pinned_rates(), now=T0 + 99_999) == pytest.approx(5.6856)


def test_multi_gpu_and_nonpreemptible_and_region():
    rates = pinned_rates()
    tp2 = receipt(gpu="H100!", n=2, cpu=8, mem=96, stop=T0 + 1800)
    assert container_cost(tp2, rates, T0 + 1800) == pytest.approx(
        0.5 * (2 * 3.95 + 8 * 0.0472 + 96 * 0.008))
    orch = receipt(gpu=None, n=0, cpu=1, mem=2, preemptible=False, stop=T0 + 3600)
    assert container_cost(orch, rates, T0 + 3600) == pytest.approx(3 * (0.0472 + 2 * 0.008))
    regional = rates.with_region(1.15)
    assert container_cost(receipt(stop=T0 + 3600), regional, 0) == pytest.approx(
        1.15 * 5.6856)
    assert region_multiplier_for(None) == 1.0 and region_multiplier_for("us-east") == 1.15


def test_running_container_fresh_and_stale_heartbeat():
    rates = pinned_rates()
    fresh = receipt(hb=T0 + 3500)
    # heartbeat 100 s old, within stale_after_s: billed up to now
    assert billed_seconds(fresh, now=T0 + 3600, stale_after_s=300) == pytest.approx(3600)
    stale = receipt(hb=T0 + 1000)
    # heartbeat 2600 s old: presumed dead at heartbeat + stale_after_s
    assert billed_seconds(stale, now=T0 + 3600, stale_after_s=300) == pytest.approx(1300)
    s = spend([fresh, stale, receipt("c3", stop=T0 + 600)], rates, now=T0 + 3600)
    assert s.containers == 3 and s.running == 2
    per_s = H200_CONTAINER_HOUR / 3600
    assert s.total_usd == pytest.approx(per_s * (3600 + 1300 + 600))
    assert s.running_usd == pytest.approx(per_s * (3600 + 1300))


def test_clock_skew_never_negative():
    assert billed_seconds(receipt(start=T0, stop=T0 - 5), now=T0) == 0.0


def test_receipts_from_store(tmp_path):
    store = LocalStore(tmp_path)
    paths = JobPaths("job1")
    rs = [receipt("ca", stop=T0 + 60), receipt("cb", stop=T0 + 120)]
    for r in rs:
        store.put_bytes(paths.container(r.container_id), json.dumps(asdict(r)).encode())
    got = read_container_receipts(store, "job1")
    assert sorted(r.container_id for r in got) == ["ca", "cb"]
    assert spend(got, pinned_rates(), T0).total_usd == pytest.approx(
        180 * H200_CONTAINER_HOUR / 3600)


def test_gpu_key_and_unknown_gpu():
    assert gpu_key("H100!") == "H100" and gpu_key("h200:2") == "H200"
    with pytest.raises(KeyError, match="no price"):
        pinned_rates().gpu("Z9000")


def test_rates_from_mapping_per_second_and_hourly():
    raw = {"gpu.h200": "0.0013", "gpu.l40s": 2.0, "cpu_core": "0.00002", "memory_gib": 0.009,
           "unparseable": "n/a"}
    r = rates_from_mapping(raw)
    assert r.gpu("H200") == pytest.approx(0.0013)
    assert r.gpu("L40S") == pytest.approx(2.0 / 3600)  # > $0.05/s means $/hour
    assert r.core_per_s == pytest.approx(0.00002)
    assert r.gib_per_s == pytest.approx(0.009 / 3600)
    assert r.gpu("H100") == pytest.approx(3.95 / 3600)  # missing -> pinned
    assert r.source == "Workspace.billing.rates()"


def test_load_rates_provider_and_fallback():
    assert load_rates(None).source == PINNED_PRICES_SOURCE
    live = load_rates(lambda: {"H200": 0.002}, region_multiplier=1.15)
    assert live.gpu("H200") == 0.002 and live.region_multiplier == 1.15

    def broken():
        raise PermissionError("Team plan required")

    fb = load_rates(broken)
    assert fb.gpu("H200") == pytest.approx(4.54 / 3600)
    assert "live rates failed" in fb.source and "Team plan" in fb.source


def test_rates_is_plain_dataclass():
    r = Rates(gpu_per_s={"H200": 1.0}, core_per_s=0.0, gib_per_s=0.0)
    assert r.per_second("H200", 2, 8, 96) == 2.0
