"""``validate/certify.py``: the cold witness's count flips are judged over the union with the
candidate's episodes; its other means keep their tight floors on their own
(``tests/fakes/witness_cases.py``, the table the independent checker is tested on too)."""

from __future__ import annotations

import copy

import pytest

from flora_deploy.validate import certify as C
from flora_deploy.validate import certkey
from tests.fakes.certrecords import BOUNDS, L0_PASS, key, lab_for, rec, segs
from tests.fakes.witness_cases import CASE_IDS, CASES, UNION, WITNESS_N, Case

STAMP = "2026-10-22T00:00:00Z"
SPLIT = [(0.0, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 9.0)]  # one subtask split in two
CAPS = ["reach the mug", "lift the mug", "set the mug down", "let go"]


def uids(n):
    return [f"u{i}" for i in range(n)]


def stage_recs(stage, preset, ids, *, split=(), shift=0.0, captions=None, **kw):
    out = []
    for u in ids:
        b = SPLIT if u in split else [(s + shift, e + shift) for s, e in BOUNDS]
        caps = (captions or CAPS)[:len(b)] if stage == "caption" else None
        out.append(rec(u, stage, preset, segments=segs(b, caps), **kw))
    return out


def certify_case(case: Case, stage: str):
    gpu = "L40S" if stage == "segment" else "H100!"
    main = uids(case.main)
    wit_ids = main[:WITNESS_N]
    main_split = set(main[len(main) - case.main_flips:]) if case.main_flips else set()
    wit_split = set(wit_ids[:case.witness_flips])
    shift = 0.2 if case.drift == "boundary" else 0.0
    wcaps = ["x " + c for c in CAPS] if case.drift == "caption" else None
    return C.certify_stage(
        stage, exact=stage_recs(stage, "exact", main),
        balanced=stage_recs(stage, "balanced", main, split=main_split),
        noise=stage_recs(stage, "exact", wit_ids),
        witness=stage_recs(stage, "balanced", wit_ids, split=wit_split, shift=shift,
                           captions=wcaps, compile_cache="cold:forced"),
        engine_spec=key("balanced", stage, gpu), gpu=gpu,
        l0=L0_PASS if stage == "segment" else lab_for(stage), generated_at=STAMP)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_witness_union_cases(case):
    c = certify_case(case, case.stage)
    assert c["verdict"] == case.verdict, c["reasons"]
    failing = {(ch["class"], ch["metric"]) for ch in c["checks"] if ch["pass"] is False}
    assert failing == set(case.failing)
    flips, n, allowed = UNION[case.id]
    cf = c["metrics"]["count_flips"]
    assert (cf["union"]["flips"], cf["union"]["n"], cf["union"]["allowed"]) == (flips, n, allowed)
    assert cf["union"]["rate"] == pytest.approx(flips / n, abs=1e-6)
    assert cf["groups"] == {"yam": {"n": case.main, "flips": case.main_flips},
                            "cold-witness": {"n": WITNESS_N, "flips": case.witness_flips}}
    union = [ch for ch in c["checks"] if ch["metric"] == "count_mismatch"]
    assert [(ch["class"], ch["flips"], ch["n"], ch["allowed"]) for ch in union] == \
        [("union", flips, n, allowed)]
    if case.verdict == "certified":
        assert c["reasons"] == [] and certkey.policy_problems(c) == []
    else:
        assert "divergence_above_noise_floor" in c["reasons"]


def test_two_in_eighteen_is_the_rounding_rule():
    """0.10 x 18 = 1.8 rounds to 2 (half up): 2 of 18 (11%) pass, 3 fail; 2 of 14 fail."""
    assert certkey.count_flips_allowed(0.1, 18) == 2
    assert certkey.count_flips_allowed(0.1, 14) == 1
    assert certkey.count_flips_allowed(0.1, 15) == 2
    assert [certkey.count_flips_allowed(0.1, n) for n in (6, 10, 12)] == [1, 1, 1]


def test_without_a_witness_the_union_is_the_candidate():
    main = uids(12)
    c = C.certify_stage("segment", exact=stage_recs("segment", "exact", main),
                        balanced=stage_recs("segment", "balanced", main, split={"u11"}),
                        noise=stage_recs("segment", "exact", main[:6]),
                        engine_spec=key("balanced", "segment", "L40S"), gpu="L40S",
                        l0=L0_PASS, generated_at=STAMP)
    assert c["verdict"] == "certified", c["reasons"]
    assert c["metrics"]["count_flips"] == {
        "groups": {"yam": {"n": 12, "flips": 1}},
        "union": {"n": 12, "flips": 1, "rate": round(1 / 12, 6), "threshold": 0.1,
                  "allowed": 1}}


def test_tampered_count_flips_are_caught():
    c = certify_case(CASES[1], "segment")
    assert certkey.policy_problems(c) == []
    for edit in (lambda b: b["metrics"]["count_flips"]["groups"]["cold-witness"].update(flips=0),
                 lambda b: b["metrics"]["count_flips"]["union"].update(flips=1),
                 lambda b: b["metrics"].pop("count_flips")):
        bad = copy.deepcopy(c)
        edit(bad)
        bad["id"] = certkey.certificate_id(bad)
        assert certkey.policy_problems(bad), edit
    over = certify_case(CASES[2], "segment")
    forged = copy.deepcopy(over)
    forged["verdict"] = "certified"
    assert any("union count_mismatch" in p for p in certkey.policy_problems(forged))
