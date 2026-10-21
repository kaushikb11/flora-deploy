"""Contract: ``sdk.Job.report()`` -> ``scripts/wave_h.py``'s job record -> ``wave_h_finalize``'s
rate loader, on a real local fake balanced job (review 11a H1: the report emitted no per-container
``kind`` / ``startup_s`` / ``busy_s`` / ``seconds``, so finalize could never measure a rate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from flora_deploy import sdk
from flora_deploy.jobs.containers import container_rows, parse_utc
from flora_deploy.local import LocalRunner
from flora_deploy.local import sample_dataset as make_sample
from tests.unit.test_balanced_deps import balanced_yaml
from tests.unit.test_wave_h_runbook import _load

wh = _load("wave_h")
fin = _load("wave_h_finalize")
TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("contract")
    make_sample(d / "videos")
    (d / "flora.yaml").write_text(balanced_yaml(TEMPLATE.read_text()))
    r = LocalRunner(d / "flora.yaml", store=d / "store")
    try:
        res = r.run()
    finally:
        r.close()
    assert res.state == "done", res.as_dict()
    return sdk.open_job(r.job_id, store=str(d / "store")).report()


def test_report_emits_the_per_container_fields(report):
    cs = report["spend"]["containers"]
    assert cs, "flora report lists no containers"
    kinds = {c["kind"] for c in cs}
    assert {"segment", "caption"} <= kinds
    for c in cs:
        for f in ("kind", "gpu", "startup_s", "busy_s", "seconds", "usd", "container_id",
                  "gpu_count", "stopped"):
            assert f in c, (f, c)
        assert c["startup_s"] is not None and c["startup_s"] >= 0
        assert c["seconds"] is not None and c["seconds"] >= c["busy_s"] >= 0


def test_finalize_measures_from_a_real_report(report):
    rec = wh.job_record(report)
    st = wh.State("r1", 0.0, "2026-10-21")
    st.step("balanced").result = {"jobs": {"human": {**rec, "app": "a", "job_id": "j"},
                                           "yam": {**rec, "app": "a", "job_id": "j2"}}}
    st.step("balanced").status = "done"
    ms = fin.measure(st)
    for kind in ("segment", "caption"):
        m = ms[("balanced", kind)]
        assert m.problems == [], m.problems  # every field present, on the pool's GPU
        assert m.n == 2 and m.startups and m.container_s > 0


def test_container_rows_math():
    t = "2026-10-22T10:00:{:02d}Z".format
    markers = [{"shard_id": "s1", "container_id": "c1", "kind": "segment",
                "started_utc": t(10), "ended_utc": None, "call_id": "f1"},
               {"shard_id": "s2", "container_id": "c1", "kind": "segment",
                "started_utc": t(30), "ended_utc": None, "call_id": "f2"}]
    receipts = [{"shard_id": "s1", "container_id": "c1", "finished_utc": t(20),
                 "gpu": "NVIDIA L40S"},
                {"shard_id": "s2", "container_id": "c1", "finished_utc": t(35), "gpu": None}]
    spawned = {"f1": parse_utc(t(0)), "f2": parse_utc(t(29))}
    (c,) = container_rows(markers, receipts, spawned, pool_gpus={"segment": "L40S"},
                          scaledown_s=300.0, usd_per_s=lambda k: 0.01)
    assert (c["kind"], c["gpu"], c["gpu_seen"]) == ("segment", "L40S", "NVIDIA L40S")
    assert c["startup_s"] == 10.0 and c["busy_s"] == 15.0
    assert c["seconds"] == 35.0 + 300.0 and c["usd"] == pytest.approx(3.35)
    (c,) = container_rows(markers, receipts, spawned, pool_gpus={"segment": "L40S"},
                          scaledown_s=300.0, stopped_at=parse_utc(t(50)))
    assert c["seconds"] == 50.0  # cut at the app stop
    clipped = wh.clip_containers([{**c, "seconds": 335.0, "usd": 3.35}], parse_utc(t(50)))
    assert clipped[0]["seconds"] == 50.0 and clipped[0]["usd"] == pytest.approx(0.5)
