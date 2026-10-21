"""G3 end to end on the G2 local runner (``flora run --local --fake-models``).

1. A whole job (segcap + rewards) with ``pipeline.exact_scheduling.episode_order:
   longest_first`` and a preemption fault: done, same config hash, byte-identical records to a
   clean plan-order run (start order never changes an output).
2. Shard overlap over G2's real index receipts and manifest: every planned ``segcap`` shard run
   at once in one "container" (``max_shards`` = shard count, shared episode slots) by the
   production ``FastExactEngine`` composite over the fake NVIDIA family, against the same
   shards run one after the other: identical records and model-call input digests."""

from __future__ import annotations

import dataclasses
import json
import shutil
import threading
from pathlib import Path

import pytest
import yaml

from flora_deploy.accel.identity import compare_runs
from flora_deploy.export.results import iter_records
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunner, run_local
from flora_deploy.local import sample_dataset as make_sample
from flora_deploy.workers_impl.segcap import SegcapShardRunner, SegcapWorkerSettings
from tests.conformance.registry import segcap_fake_engine
from tests.conformance.worker_suite import Script

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("g3demo")
    shutil.copy(TEMPLATE, d / "flora.yaml")
    make_sample(d / "videos")
    return d / "flora.yaml"


def records(res) -> dict:
    st = LocalStore(res.store_root)
    return {(r.uid, r.stage): r.to_json() for stage in STAGES
            for r in iter_records(st, res.job_id, stage)}


def with_order(demo: Path, order: str) -> Path:
    doc = yaml.safe_load(demo.read_text())
    doc["pipeline"]["exact_scheduling"] = {"episode_order": order}
    out = demo.with_name(f"flora-{order}.yaml")
    out.write_text(yaml.safe_dump(doc))
    return out


def test_longest_first_job_is_done_and_byte_identical(demo, tmp_path, monkeypatch):
    from flora_deploy.config.schema import exact_scheduling

    clean = run_local(demo, store=tmp_path / "clean")
    assert clean.state == "done", clean.as_dict()

    orig = LocalRunner._runner
    seen: list[str] = []

    def runner(self, kind, container_id, call_id):  # what the app factory's env would carry
        r = orig(self, kind, container_id, call_id)
        if kind == "segcap":
            r.settings = dataclasses.replace(r.settings, **exact_scheduling(self.cfg))
            seen.append(r.settings.episode_order)
        return r

    monkeypatch.setattr(LocalRunner, "_runner", runner)
    lpt_cfg = with_order(demo, "longest_first")
    lpt = run_local(lpt_cfg, store=tmp_path / "lpt",
                    faults="preempt:" + _first_segcap(clean))
    assert lpt.state == "done", lpt.as_dict()
    assert seen and set(seen) == {"longest_first"}
    assert lpt.config_hash == clean.config_hash  # not hashed
    assert records(lpt) == records(clean)


def _first_segcap(res) -> str:
    from flora_deploy.jobs.manifest import load_manifest

    m = load_manifest(LocalStore(res.store_root), res.job_id)
    return next(s.shard_id for s in m.shards if s.kind == "segcap")


def _planned(demo: Path, store: Path):
    lr = LocalRunner(demo, store=store)
    lr.index()
    m = lr.plan()
    return lr, m


def _run_segcap(lr, m, *, overlap: bool, tmp: Path) -> dict:
    eng = segcap_fake_engine(Script(), concurrency=2, sleeps={"ep000": 0.3})
    shards = [s for s in m.shards if s.kind == "segcap"]
    if overlap:
        eng.overlap = True
        eng.__post_init__()
    s = SegcapWorkerSettings(container_id="tg3-1", gpu="H200", models={"tempseg": "fake"},
                             media_root=str(lr.media_root), local_dir=str(tmp / "local"),
                             episode_order="longest_first" if overlap else "plan",
                             max_shards=len(shards) if overlap else 1)
    r = SegcapShardRunner(lr.store, eng, s, call_id=lambda: "fc")
    out = {}

    def go(sid):
        out[sid] = r.run_shard(m.job_id, sid, m.config_hash)

    if overlap:
        ts = [threading.Thread(target=go, args=(sh.shard_id,)) for sh in shards]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    else:
        for sh in shards:
            go(sh.shard_id)
    assert all(out.get(sh.shard_id) is not None for sh in shards), out
    from flora_deploy.jobs import progress as P
    from flora_deploy.jobs.attempts import JobFiles

    jf = JobFiles(lr.store, m.job_id, m.config_hash)
    key = jf.key("segcap")
    st, calls = {}, {}
    for sh in shards:
        for p in P.shard_parts(jf, sh).parts:
            for rec in P.read_part(jf, p):
                st[(rec.uid, rec.stage)] = rec.to_json()
        prefix = jf.paths.audit(key, "segcap", sh.shard_id, "x", 0)[:-len("x-0.jsonl")]
        for p in lr.store.list(prefix):
            for line in lr.store.get_bytes(p).decode().splitlines():
                if line.strip():
                    a = json.loads(line)
                    calls[a["uid"]] = a["result"]["flora_reference"]["model_calls"]
    return {"records": st, "calls": calls, "shards": len(shards),
            "max_running": eng._slots.stats()["max_running"] if overlap else None}


def test_overlapped_shards_on_g2_plan_keep_records_and_inputs(demo, tmp_path):
    lr_a, m_a = _planned(demo, tmp_path / "seq")
    lr_b, m_b = _planned(demo, tmp_path / "ov")
    try:
        assert [s.shard_id for s in m_a.shards] == [s.shard_id for s in m_b.shards]
        seq = _run_segcap(lr_a, m_a, overlap=False, tmp=tmp_path / "a")
        ov = _run_segcap(lr_b, m_b, overlap=True, tmp=tmp_path / "b")
    finally:
        lr_a.close()
        lr_b.close()
    assert seq["shards"] >= 2 and ov["max_running"] == 2
    assert seq["records"] and seq["records"] == ov["records"]
    assert set(seq["calls"]) == set(ov["calls"]) and seq["calls"]
    ident = compare_runs(seq["calls"], ov["calls"])
    assert ident["ok"] and ident["coverage"] == 1.0, ident
