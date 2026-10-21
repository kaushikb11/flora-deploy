"""G3 end to end on the G2 local runner (``flora run --local --fake-models``).

A whole job (segcap + rewards) with ``pipeline.exact_scheduling.episode_order: longest_first``
and a preemption fault: done, same config hash, byte-identical records to a clean plan-order
run (start order never changes an output). G3's shard overlap was removed (review 7b)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest
import yaml

from flora_deploy.export.results import iter_records
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunner, run_local
from flora_deploy.local import sample_dataset as make_sample

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
