import json

import pytest

from flora_deploy.jobs.store import JobPaths
from flora_deploy.records import Provenance, Segment, StageRecord, TerminalReward, record_key

PROV = Provenance(models={"tempseg": "114a"}, engine="hf-ref-5.8.1", preset="exact",
                  config_hash="c" * 16, logic_version="0.0.1", framework="flora-deploy 0.0.1")


def test_record_roundtrips_to_json():
    rec = StageRecord(
        episode_id="ep1", stage="caption", provenance=PROV, certified=True,
        segments=(Segment(event_id="event_0000", start_s=0.0, end_s=1.5, caption="x"),),
    )
    data = json.loads(rec.to_json())
    assert data["schema"] == "flora_labels_v1"
    assert data["segments"][0]["caption"] == "x"
    assert record_key(rec) == ("ep1", "caption")


def test_certified_reason_invariant():
    with pytest.raises(ValueError):
        StageRecord(episode_id="e", stage="segment", provenance=PROV, certified=True,
                    uncertified_reason="uncovered_pool")
    with pytest.raises(ValueError):
        StageRecord(episode_id="e", stage="segment", provenance=PROV, certified=False)
    StageRecord(episode_id="e", stage="reward.terminal", provenance=PROV, certified=False,
                uncertified_reason="outside_reference_envelope",
                terminal=TerminalReward(score=3, expected_score=3.1,
                                        probabilities=(0.0, 0.1, 0.7, 0.2, 0.0)))


def test_job_paths_are_hash_fanned_and_stable():
    p = JobPaths("job1")
    a = p.receipt("caption", "caption-0001")
    assert a == p.receipt("caption", "caption-0001")
    parts = a.split("/")
    assert parts[:4] == ["flora", "job1", "receipts", "caption"]
    assert len(parts[4]) == 2 and len(parts[5]) == 2 and parts[6] == "caption-0001.json"
    assert p.receipt("caption", "caption-0002") != a
