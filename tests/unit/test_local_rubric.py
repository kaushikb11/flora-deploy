"""``flora run --local --fake-models`` with ``reward.rubric``: the real index, manifest,
orchestrator, ``RewardShardRunner`` + ``stages.rubric.run`` and compose, with the fake Rewarder
family (rubric interface) and the fake LLM's ``n`` samples per request."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.export.results import iter_records
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import run_local
from flora_deploy.local import sample_dataset as make_sample

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
RUBRIC = """  rubric:
    robot: two-finger gripper on a tabletop arm
    cameras:
      - {role: main, description: overhead view of the workbench}
    steps:
      - action: Pull the drawer open
        grading:
          - {label: Drawer untouched, points: 0.0}
          - {label: Opened partway, points: 0.15}
          - {label: Opened fully, points: 0.3}
      - action: Drop the wooden block into the drawer
        grading:
          - {label: Block not moved, points: 0.0}
          - {label: Block lands outside, points: 0.35}
          - {label: Block inside the drawer, points: 0.7}
"""


def rubric_config(d: Path) -> Path:
    text = TEMPLATE.read_text().replace(
        "stages: [segment, caption, reward.terminal, reward.dense]", "stages: [reward.rubric]")
    text = text.replace("  shard_target_gpu_minutes: 0.5     # GPU-minutes per shard (default "
                        "10); small shards, so the\n"
                        "                                    # demo has several of each kind\n",
                        "  shard_target_gpu_minutes: 0.5\n" + RUBRIC)
    text = text.split("resources:")[0] + "resources:\n  reward.rubric: {max_containers: 1}\n\n" \
        + "output:" + text.split("output:")[1]
    text = text.replace("budget_usd: 5 ", "budget_usd: 100 ")  # 2 x H100 per container
    (d / "flora.yaml").write_text(text)
    return d / "flora.yaml"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("rubric")
    cfg = rubric_config(d)
    make_sample(d / "videos")
    a = run_local(cfg, store=d / "store-a")
    b = run_local(cfg, store=d / "store-b")
    return a, b


def records(res):
    return [r for r in iter_records(LocalStore(res.store_root), res.job_id, "reward.rubric")]


def test_rubric_job_is_done_with_records_and_audits(runs):
    a, _ = runs
    assert a.state == "done", a.as_dict()
    recs = records(a)
    assert recs, "no reward.rubric records"
    for r in recs:
        assert r.stage == "reward.rubric" and r.rubric is not None
        rb = r.rubric
        assert rb.samples == 9 and 1 <= rb.parsed_samples <= 9
        assert 0 <= rb.medoid_index < rb.samples
        assert len(rb.scores) == 2 and rb.scores[0] in (0.0, 0.15, 0.3)
        assert rb.scores[1] in (0.0, 0.35, 0.7)
        assert rb.total == round(sum(rb.scores), 4)
        assert r.provenance.certificate == "local-fake-models"
        assert r.provenance.embodiment == "human"  # stamped (certstamp) by the rubric runner
        assert not r.certified
    audits = [json.loads(line) for p in Path(a.store_root).rglob("*.jsonl")
              if "/audit/" in str(p) and "reward.rubric" in str(p)
              for line in p.read_text().splitlines()]
    assert len(audits) == len(recs)
    for au in audits:
        nv = au["nv"]
        assert set(nv) == {"id", "task", "steps", "scores", "total", "medoid_sample",
                           "parsed_samples", "samples", "prompt_tokens"}
        assert nv["steps"] == ["Pull the drawer open", "Drop the wooden block into the drawer"]
        assert nv["samples"][nv["medoid_sample"]]["parsed"]


def test_rubric_records_are_deterministic_and_exported(runs, tmp_path):
    a, b = runs
    assert [r.to_json() for r in records(a)] == [r.to_json() for r in records(b)]
    out = CliRunner().invoke(app, ["results", "--job", a.job_id, "--stage", "reward.rubric",
                                   "--store", str(a.store_root)])
    assert out.exit_code == 0, out.output
    import pyarrow.parquet as pq

    (table,) = [pq.read_table(p) for p in Path(a.store_root).rglob("reward.rubric.parquet")]
    rows = table.to_pylist()
    assert len(rows) == len(records(a))
    by_uid = {r.uid: r.rubric for r in records(a)}
    for row in rows:
        rb = by_uid[row["uid"]]
        assert row["rubric_scores"] == list(rb.scores) and row["rubric_total"] == rb.total
        assert row["rubric_medoid_index"] == rb.medoid_index
        assert row["rubric_parsed_samples"] == rb.parsed_samples


def test_rubric_camera_mismatch_is_a_config_error(tmp_path):
    d = tmp_path
    cfg = rubric_config(d)
    text = cfg.read_text().replace("role: main", "role: left_wrist")
    cfg.write_text(text)
    make_sample(d / "videos")
    res = run_local(cfg, store=d / "store")
    assert res.state == "failed"  # never excused: no rubric record, no nv_failure exclusion
    errors = [json.loads(p.read_text()).get("error") or "" for p in (d / "store").rglob("*.json")
              if "/attempts/reward.rubric/" in str(p)]
    assert errors and all("RubricCamerasMismatch" in e and "pipeline.rubric.cameras" in e
                          for e in errors)
    assert not records(res)
