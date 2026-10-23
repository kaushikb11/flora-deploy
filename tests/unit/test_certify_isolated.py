"""``validate/certify.py`` v4: stage-isolated caption certification (``docs/certification.md``,
"Metrics: caption"; the caption pre-registration of 2026-10-23).

The candidate captions the exact oracle's own segments; every pair is (row i, row i) and every
Captioner call must have had the oracle's input digest. Refusals (not verdicts):
``segments_not_oracle``, ``oracle_digests_missing``, ``inputs_not_isolated``. The isolated noise
is scored only on input-identical pairs. End-to-end metrics are reported only."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from flora_deploy.validate import certify as C
from flora_deploy.validate import certkey
from tests.fakes.certrecords import (
    BOUNDS,
    caption_set,
    cold_witness,
    digests_of,
    isolated,
    key,
    lab_for,
    rec,
    segs,
)

STAMP = "2026-10-23T00:00:00Z"
GPU = "H100!"


def run(exact, balanced, noise=None, *, witness="cold", tmp_path=None, edit=None, **kw):
    """certify_stage("caption") with isolated evidence; ``edit(bal, wit, extra)`` tampers."""
    exact, balanced = list(exact), list(balanced)
    wit = cold_witness(balanced) if witness == "cold" else witness
    bal, wit, extra = isolated(exact, balanced, wit, noise, directory=tmp_path)
    if edit is not None:
        bal, wit, extra = edit(bal, wit, extra)
    return C.certify_stage("caption", exact=exact, balanced=bal, noise=noise, witness=wit,
                           engine_spec=key("balanced", "caption", GPU), gpu=GPU,
                           l0=lab_for("caption"), generated_at=STAMP, **{**extra, **kw})


def with_caption(r, i, text):
    segments = list(r.segments)
    segments[i] = dataclasses.replace(segments[i], caption=text)
    return dataclasses.replace(r, segments=tuple(segments))


def refused(reason, *a, **kw):
    with pytest.raises(C.CertifyRefused) as e:
        run(*a, **kw)
    assert e.value.reason == reason, e.value
    return e.value


# ---------------------------------------------------------------------------------------------
# metrics


def test_identical_captions_score_zero_and_certify(tmp_path):
    c = run(caption_set("exact", 12), caption_set("balanced", 12), caption_set("exact", 6),
            tmp_path=tmp_path)
    assert c["verdict"] == "certified", c["reasons"]
    assert c["generator"]["version"] == "4" and c["scoring"] == {"caption": "stage_isolated"}
    assert c["metrics"]["candidate"]["yam"]["means"] == {"caption_mismatch": 0.0,
                                                         "caption_edit_distance": 0.0}
    assert c["metrics"]["candidate"]["yam"]["slots"] == 36
    assert c["input_identity"]["candidate"]["status"] == "pass"
    assert c["input_identity"]["candidate"]["pairs"] == 36
    assert c["input_identity"]["witness"]["status"] == "pass"
    assert {ch["metric"] for ch in c["checks"]} == {"caption_mismatch", "caption_edit_distance"}
    assert c["metrics"]["count_flips"] is None
    assert certkey.policy_problems(c) == []


def test_one_differing_caption_in_an_n_row_episode_is_one_over_n():
    ex = caption_set("exact", 12)
    m = C.caption_isolated_metrics(ex[0], with_caption(caption_set("balanced", 1)[0], 1, "zz"),
                                   digests_of(ex)["u0"])
    assert m["caption_mismatch"] == pytest.approx(1 / 3)
    none_vs_empty = C.caption_isolated_metrics(
        rec("u", "caption", "exact", segments=segs(BOUNDS[:1], [None])),
        rec("u", "caption", "balanced", segments=segs(BOUNDS[:1], [""])), ["d" * 64])
    assert none_vs_empty["caption_mismatch"] == 0.0


def test_rows_without_an_oracle_call_are_not_slots():
    a = rec("u", "caption", "exact", segments=segs(BOUNDS, ["a", "b", "c"]))
    b = rec("u", "caption", "balanced", segments=segs(BOUNDS, ["a", "b", "z"]))
    assert C.caption_isolated_metrics(a, b, ["1" * 64, "2" * 64, None])["caption_mismatch"] == 0
    assert C.caption_isolated_metrics(a, b, [None, None, None]) is None


@pytest.mark.parametrize("noise_level, threshold", [(0.03, 0.075), (0.06, 0.09)])
def test_threshold_is_the_unchanged_rule(noise_level, threshold):
    assert certkey.metric_threshold("caption_mismatch", noise_level, 1.5, 0.0) == \
        pytest.approx(threshold)


def test_noise_counts_only_input_identical_pairs():
    ex = caption_set("exact", 1)[0]
    moved = rec("u0", "caption", "exact",
                segments=segs([(0.0, 2.1), (2.1, 5.0), (5.0, 9.0)], ["x", "y", "c"]))
    d_ex, d_nz = digests_of([ex])["u0"], digests_of([moved])["u0"]
    # rows 0 and 1 moved (different inputs, different captions): only row 2 is a pair
    assert C.noise_pairs(ex, moved, d_ex, d_nz) == [(2, 2)]
    assert C.noise_isolated_metrics(ex, moved, d_ex, d_nz)["caption_mismatch"] == 0.0
    allmoved = rec("u0", "caption", "exact", segments=segs([(0.1, 9.0)], ["q"]))
    assert C.noise_isolated_metrics(ex, allmoved, d_ex, digests_of([allmoved])["u0"]) is None


def test_zero_pair_episode_leaves_noise_n_and_count_flip_episode_counts():
    ex = caption_set("exact", 12)
    nz = caption_set("exact", 6)
    nz[0] = rec("u0", "caption", "exact", segments=segs([(0.3, 9.0)], ["q"]))  # no pair
    nz[1] = rec("u1", "caption", "exact",  # count flip, two rows still input-identical
                segments=segs([(0.0, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 9.0)],
                              ["a", "b1", "b2", "c"]))
    c = run(ex, caption_set("balanced", 12), nz)
    assert c["metrics"]["noise"]["n"] == 5
    assert c["metrics"]["noise"]["pairs"] == 4 * 3 + 2


def test_witness_is_held_to_the_same_rule():
    bal = caption_set("balanced", 12)
    wit = [dataclasses.replace(w, segments=tuple(dataclasses.replace(s, caption="x " + s.caption)
                                                 for s in w.segments))
           for w in cold_witness(bal)]
    c = run(caption_set("exact", 12), bal, caption_set("exact", 6), witness=wit)
    assert c["verdict"] == "fail"
    assert {(ch["class"], ch["metric"]) for ch in c["checks"] if ch["pass"] is False} == {
        ("cold-witness", "caption_mismatch"), ("cold-witness", "caption_edit_distance")}


def test_too_few_isolated_noise_episodes_is_provisional():
    c = run(caption_set("exact", 12), caption_set("balanced", 12), caption_set("exact", 4))
    assert c["verdict"] == "provisional" and "too_few_noise_samples" in c["reasons"]


# ---------------------------------------------------------------------------------------------
# refusals


def test_bounds_off_by_more_than_eps_are_refused():
    bal = caption_set("balanced", 12)
    r = bal[3]
    bal[3] = dataclasses.replace(r, segments=(
        dataclasses.replace(r.segments[0], end_s=r.segments[0].end_s + 2e-6),
        *r.segments[1:]))
    refused("segments_not_oracle", caption_set("exact", 12), bal, caption_set("exact", 6))
    ok = caption_set("balanced", 12)
    ok[3] = dataclasses.replace(r, segments=(
        dataclasses.replace(r.segments[0], end_s=r.segments[0].end_s + 5e-7),
        *r.segments[1:]))
    # within eps: the digests (bounds-derived here) still differ, so it is the input check
    refused("inputs_not_isolated", caption_set("exact", 12), ok, caption_set("exact", 6))


def test_count_mismatch_is_refused():
    bal = caption_set("balanced", 12)
    bal[0] = rec("u0", "caption", "balanced", segments=segs(BOUNDS[:2], ["a", "b"]))
    refused("segments_not_oracle", caption_set("exact", 12), bal)


def test_segments_from_mismatch_is_refused():
    def edit(bal, wit, extra):
        bad = dataclasses.replace(bal[0].provenance, segments_from={
            "sha256": "0" * 64, "oracle_stage_hash": bal[0].provenance.segments_from[
                "oracle_stage_hash"]})
        return [dataclasses.replace(bal[0], provenance=bad), *bal[1:]], wit, extra

    refused("segments_not_oracle", caption_set("exact", 12), caption_set("balanced", 12),
            edit=edit)


def test_altered_oracle_segment_file_is_refused(tmp_path):
    def edit(bal, wit, extra):
        p = extra["oracle_segments"]
        p.write_bytes(p.read_bytes() + b"\n")
        return bal, wit, extra

    refused("segments_not_oracle", caption_set("exact", 12), caption_set("balanced", 12),
            tmp_path=tmp_path, edit=edit)


def test_missing_oracle_segments_or_digests_are_refused():
    def no_file(bal, wit, extra):
        return bal, wit, {**extra, "oracle_segments": None}

    def no_exact(bal, wit, extra):
        return bal, wit, {**extra, "digests": {k: v for k, v in extra["digests"].items()
                                               if k != "exact"}}

    def one_missing(bal, wit, extra):
        d = copy.deepcopy(extra["digests"])
        d["exact"]["u2"][1] = None
        return bal, wit, {**extra, "digests": d}

    ex, bal = caption_set("exact", 12), caption_set("balanced", 12)
    refused("segments_not_oracle", ex, bal, edit=no_file)
    refused("oracle_digests_missing", ex, bal, edit=no_exact)
    refused("oracle_digests_missing", ex, bal, edit=one_missing)


def test_digest_mismatch_and_none_vs_digest_are_refused():
    def mismatch(bal, wit, extra):
        d = copy.deepcopy(extra["digests"])
        d["balanced"]["u5"][0] = "e" * 64
        return bal, wit, {**extra, "digests": d}

    def none_vs_digest(bal, wit, extra):
        d = copy.deepcopy(extra["digests"])
        d["balanced"]["u5"][2] = None
        return bal, wit, {**extra, "digests": d}

    def witness_mismatch(bal, wit, extra):
        d = copy.deepcopy(extra["digests"])
        d["witness"]["u1"][0] = "e" * 64
        return bal, wit, {**extra, "digests": d}

    ex, bal = caption_set("exact", 12), caption_set("balanced", 12)
    for edit in (mismatch, none_vs_digest, witness_mismatch):
        e = refused("inputs_not_isolated", ex, bal, edit=edit)
        assert "harness" in str(e)


def test_noise_without_digests_is_refused():
    def edit(bal, wit, extra):
        return bal, wit, {**extra, "digests": {k: v for k, v in extra["digests"].items()
                                               if k != "noise"}}

    refused("oracle_digests_missing", caption_set("exact", 12), caption_set("balanced", 12),
            caption_set("exact", 6), edit=edit)


# ---------------------------------------------------------------------------------------------
# reported-only metrics, policy, content address


def _e2e_set():
    """J1: the stack on its own segments: one episode's boundary moved, captions differ."""
    e2e = caption_set("balanced", 12)
    e2e[4] = rec("u4", "caption", "balanced",
                 segments=segs([(0.0, 2.1), (2.1, 5.0), (5.0, 9.0)], ["q", "r", "c"]))
    return e2e


def test_reported_block_never_changes_the_verdict_or_the_checks():
    ex, bal, nz = caption_set("exact", 12), caption_set("balanced", 12), caption_set("exact", 6)
    plain = run(ex, bal, nz)
    rep = run(ex, bal, nz, reported_e2e=_e2e_set(),
              families={f"u{i}": ("n-egodex" if i < 6 else "x-wgo") for i in range(12)})
    assert rep["verdict"] == plain["verdict"] == "certified"
    assert rep["checks"] == plain["checks"]
    r = rep["metrics"]["reported"]
    assert r["reported_only"] is True and r["e2e"]["reported_only"] is True
    assert r["e2e"]["means"]["caption_mismatch"] > 0
    assert set(r["families"]) == {"n-egodex", "x-wgo"}
    assert r["set_valued"]["rate"] == 1.0
    assert r["micro"]["candidate"] == {"slots": 36, "differing": 0, "rate": 0.0}
    assert certkey.policy_problems(rep) == []
    with pytest.raises(C.CertifyError, match="family labels"):
        run(ex, bal, nz, families={"u0": "Some Dataset Name"})


def test_policy_refuses_reported_or_count_gates_and_missing_scoring():
    c = run(caption_set("exact", 12), caption_set("balanced", 12), caption_set("exact", 6),
            reported_e2e=_e2e_set())
    for edit in (lambda b: b.pop("scoring"),
                 lambda b: b["input_identity"]["candidate"].update(status="fail"),
                 lambda b: b["input_identity"].update(witness=None),
                 lambda b: b["checks"].append({"class": "union", "metric": "count_mismatch",
                                               "value": 0.0, "threshold": 0.1, "pass": True}),
                 lambda b: b["checks"].append({"class": "yam", "metric": "e2e",
                                               "value": 0.0, "threshold": 0.1, "pass": True})):
        bad = copy.deepcopy(c)
        edit(bad)
        bad["id"] = certkey.certificate_id(bad)
        assert certkey.policy_problems(bad), edit


def test_isolated_certificate_is_content_addressed(tmp_path):
    ex, bal, nz = caption_set("exact", 12), caption_set("balanced", 12), caption_set("exact", 6)
    a = run(ex, bal, nz, tmp_path=tmp_path)
    b = run(ex, bal, nz, tmp_path=tmp_path)
    assert a["id"] == b["id"] == certkey.certificate_id(a)
    p = C.write_certificate(a, tmp_path / "certs")
    assert p.name == f"{a['id']}.json"


def test_segment_certification_is_unchanged_under_v4():
    assert C.STAGE_METRICS["segment"] == ("iou_divergence", "boundary_error_s",
                                          "count_mismatch", "label_disagreement")
    assert "count_mismatch" not in C.STAGE_METRICS["caption"]
    assert certkey.GENERATOR_VERSIONS == ("4",)
    assert "3" in certkey.RETIRED_GENERATOR_VERSIONS



def test_oracle_and_noise_from_the_pre_v4_code_still_pair():
    """Order A: (b) and the noise-12 run come from the code before this change (another segcap
    stage hash and exact key than this checkout computes). Only noise == oracle is required;
    the candidate is bound to this checkout's key on its own."""
    def old(recs):
        out = []
        for r in recs:
            doc = {**r.provenance.cert_key, "logic_digest": "0" * 64}
            out.append(dataclasses.replace(r, provenance=dataclasses.replace(
                r.provenance, cert_key=doc, config_hash="9" * 64)))
        return out

    c = run(old(caption_set("exact", 12)), caption_set("balanced", 12),
            old(caption_set("exact", 6)))
    assert c["verdict"] == "certified", c["reasons"]
    assert c["oracle"]["stage_hash"] == "9" * 64
    assert c["segments_from"]["oracle_stage_hash"] == "9" * 64
    with pytest.raises(C.CertifyError, match="repeat the oracle run"):  # noise != oracle
        run(old(caption_set("exact", 12)), caption_set("balanced", 12), caption_set("exact", 6))
