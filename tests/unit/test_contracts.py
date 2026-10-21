import json
import math

import pytest

from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import episode_uid, safe_key
from flora_deploy.records import (
    DenseReward,
    Provenance,
    Segment,
    StageRecord,
    TerminalReward,
    record_key,
)

PROV = Provenance(models={"tempseg": "114a"}, engine="hf-ref-5.8.1", preset="exact",
                  config_hash="c" * 16, logic_version="0.0.1", framework="flora-deploy 0.0.1")


def rec(**kw):
    base = dict(uid="ds--ep1", dataset="ds", episode_id="ep1", stage="caption", provenance=PROV,
                certified=True)
    base.update(kw)
    return StageRecord(**base)


def test_record_roundtrips_to_json_and_back():
    r = rec(segments=(Segment(event_id="event_0000", start_s=0.0, end_s=1.5, caption="x"),),
            terminal=TerminalReward(score=3, expected_score=3.1,
                                    probabilities=(0.0, 0.1, 0.7, 0.2, 0.0)),
            dense=DenseReward(segments=({"label": 2},), per_step=(3, 3, 2)))
    data = json.loads(r.to_json())
    assert data["schema"] == "flora_labels_v1"
    assert StageRecord.from_json(r.to_json()) == r
    assert record_key(r) == ("ds--ep1", "caption")


def test_nan_is_rejected():
    r = rec(terminal=TerminalReward(score=3, expected_score=math.nan,
                                    probabilities=(0.0, 0.1, 0.7, 0.2, 0.0)))
    with pytest.raises(ValueError):
        r.to_json()


def test_certified_reason_invariant():
    with pytest.raises(ValueError):
        rec(uncertified_reason="uncovered_pool")
    with pytest.raises(ValueError):
        rec(certified=False)
    rec(certified=False, uncertified_reason="outside_reference_envelope")


def test_episode_uid_disambiguates_datasets_and_is_path_safe():
    a = episode_uid("lerobot/aloha_static_coffee", "episode_000001")
    b = episode_uid("lerobot/aloha_mobile_wipe", "episode_000001")
    assert a != b and "/" not in a
    assert safe_key(a) == a
    for bad in ("a/b", "../x", "", "x" * 300):
        with pytest.raises(ValueError):
            safe_key(bad)


def test_job_paths_fanned_per_config_and_per_container():
    p = JobPaths("job1")
    r = p.receipt("cfgA", "segcap", "segcap-0001")
    assert r == p.receipt("cfgA", "segcap", "segcap-0001")
    assert r.startswith("flora/job1/cfgA/receipts/segcap/")
    assert p.receipt("cfgB", "segcap", "segcap-0001") != r
    a1 = p.attempt("cfgA", "segcap", "segcap-0001", "ta-111")
    a2 = p.attempt("cfgA", "segcap", "segcap-0001", "ta-222")
    assert a1 != a2  # concurrent starts never collide
    assert p.part("cfgA", "segcap", "segcap-0001", "ta-111", 0) != p.part(
        "cfgA", "segcap", "segcap-0001", "ta-222", 0)
    with pytest.raises(ValueError):
        p.receipt("cfgA", "segcap", "../escape")
