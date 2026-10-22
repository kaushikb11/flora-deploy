"""Review 13 repros for ``validate/certify.py`` and ``validate/certkey.py``.

- The floor policy (user decision on S3-2): up to about 1 episode in 10 may differ in subtask
  count, while the means of the other metrics, over the episodes whose counts agree, keep their
  tight floors.
- Spares (S3-3): the certificate evaluates every episode both sides labelled; one missing
  balanced, noise or oracle episode does not cost the certificate while >= 10 matched and >= 5
  noise episodes remain.
- Stale lab evidence (S3-1): L0 counts only when the lab ran the current engine settings.
- Preemptions (B1) and the compile cache (W1): recorded per record; preempted or unknown
  records cannot be certified; the certificate states the warm/cold mix and the cold witness.
- A matching ``fail`` certificate overrides reference-by-identity (S3-7)."""

from __future__ import annotations

import copy
import json

import pytest

from flora_deploy.validate import certify as C
from flora_deploy.validate import certkey
from tests.fakes.certrecords import (
    BOUNDS,
    L0_PASS,
    caption_set,
    key,
    lab_for,
    rec,
    segs,
)

STAMP = "2026-10-22T00:00:00Z"
SPLIT = [(0.0, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 9.0)]  # one subtask split in two


def seg_recs(preset, uids, *, split=(), shift=0.0, **kw):
    return [rec(u, "segment", preset,
                segments=segs(SPLIT if u in split else [(s + shift, e + shift)
                                                        for s, e in BOUNDS]), **kw)
            for u in uids]


def uids(n, start=0):
    return [f"u{i}" for i in range(start, start + n)]


def certify(exact, balanced, noise, *, stage="segment", l0=L0_PASS, gpu="L40S", **kw):
    return C.certify_stage(stage, exact=exact, balanced=balanced, noise=noise,
                           engine_spec=key("balanced", stage, gpu), gpu=gpu, l0=l0,
                           generated_at=STAMP, **kw)


# ---------------------------------------------------------------------------------------------
# the floor policy


def test_one_differing_episode_in_twelve_is_certified():
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12), split={"u3"}),
                seg_recs("exact", uids(5)))
    assert c["verdict"] == "certified", c["reasons"]
    means = c["metrics"]["candidate"]["yam"]["means"]
    assert means["count_mismatch"] == pytest.approx(1 / 12, abs=1e-6)
    # the other metrics are taken over the 11 episodes whose counts agree: identical there
    assert means["iou_divergence"] == 0.0 and means["boundary_error_s"] == 0.0
    pm = c["thresholds"]["per_metric"]
    assert pm["count_mismatch"]["absolute_floor"] == 0.1
    assert pm["count_mismatch"]["threshold"] == 0.1
    assert pm["count_mismatch"]["over"] == "all"
    assert pm["boundary_error_s"]["over"] == "count_matched"
    assert c["metrics"]["candidate"]["yam"]["count_matched"] == 11
    assert certkey.policy_problems(c) == []


def test_two_differing_episodes_in_twelve_fail():
    c = certify(seg_recs("exact", uids(12)),
                seg_recs("balanced", uids(12), split={"u3", "u7"}), seg_recs("exact", uids(5)))
    assert c["verdict"] == "fail" and "divergence_above_noise_floor" in c["reasons"]
    bad = [ch for ch in c["checks"] if ch["pass"] is False]
    assert [ch["metric"] for ch in bad] == ["count_mismatch"]


def test_mean_boundary_error_above_the_floor_fails_with_every_count_matching():
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12), shift=0.2),
                seg_recs("exact", uids(5)))
    assert c["metrics"]["candidate"]["yam"]["means"]["count_mismatch"] == 0.0
    assert c["verdict"] == "fail"
    assert "boundary_error_s" in {ch["metric"] for ch in c["checks"] if ch["pass"] is False}


def test_caption_counts_are_bounded_and_caption_means_stay_tight():
    split = segs(SPLIT, ["a", "b", "b2", "c"])
    ex = caption_set("exact", 12)
    one = [*caption_set("balanced", 12)[:11],
           rec("u11", "caption", "balanced", segments=split)]
    c = certify(ex, one, caption_set("exact", 5), stage="caption", l0=lab_for("caption"),
                gpu="H100!")
    assert c["verdict"] == "certified", c["reasons"]
    assert "count_mismatch" in c["metrics"]["candidate"]["yam"]["means"]
    # one differing caption on a count-matched episode: 1/(12*3) of the slots, under 0.075
    off = caption_set("balanced", 12)
    off[0] = rec("u0", "caption", "balanced", segments=segs(BOUNDS, ["a", "b", "z"]))
    assert certify(ex, off, caption_set("exact", 5), stage="caption", l0=lab_for("caption"),
                   gpu="H100!")["verdict"] == "certified"
    off[1] = rec("u1", "caption", "balanced", segments=segs(BOUNDS, ["q", "r", "z"]))
    off[2] = rec("u2", "caption", "balanced", segments=segs(BOUNDS, ["q", "r", "z"]))
    assert certify(ex, off, caption_set("exact", 5), stage="caption", l0=lab_for("caption"),
                   gpu="H100!")["verdict"] == "fail"


def test_floors_are_recorded_and_a_loosened_one_never_matches():
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12)),
                seg_recs("exact", uids(5)))
    assert c["thresholds"]["noise_floor"] == {m: certkey.NOISE_FLOOR[m]
                                              for m in C.STAGE_METRICS["segment"]}
    assert c["thresholds"]["absolute_floor"] == {"count_mismatch": 0.1}
    bad = copy.deepcopy(c)
    bad["thresholds"]["absolute_floor"]["count_mismatch"] = 0.5
    bad["id"] = certkey.certificate_id(bad)
    assert any("floor" in p for p in certkey.policy_problems(bad))
    bad = copy.deepcopy(c)
    bad["thresholds"]["per_metric"]["iou_divergence"]["floor"] = 0.3
    bad["id"] = certkey.certificate_id(bad)
    assert any("floor" in p for p in certkey.policy_problems(bad))
    assert certkey.metric_threshold("count_mismatch", 0.0, 1.5, 0.0) == 0.1
    assert certkey.metric_threshold("count_mismatch", 0.2, 1.5, 0.0) == pytest.approx(0.3)


# ---------------------------------------------------------------------------------------------
# spares: one missing episode at each stage


def test_one_missing_balanced_episode_is_filled_by_a_spare():
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(11)),
                seg_recs("exact", uids(5)))
    assert c["verdict"] == "certified", c["reasons"]
    assert c["metrics"]["candidate"]["yam"]["n"] == 11 and c["metrics"]["missing"] == 1
    # more missing than the spares: provisional, never certified
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(9)),
                seg_recs("exact", uids(5)))
    assert c["verdict"] == "provisional" and "too_few_samples" in c["reasons"]
    c = certify(seg_recs("exact", uids(14)), seg_recs("balanced", uids(11)),
                seg_recs("exact", uids(5)))
    assert c["verdict"] == "provisional" and "incomplete_coverage" in c["reasons"]


def test_one_missing_noise_episode_is_filled_by_the_spare():
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12)),
                seg_recs("exact", uids(5, start=1)))  # 6 planned, u0 missing
    assert c["verdict"] == "certified" and c["metrics"]["noise"]["n"] == 5


def test_oracle_missing_a_noise_episode_uses_the_spare_not_an_error():
    ex = seg_recs("exact", [u for u in uids(12) if u != "u2"])
    c = certify(ex, seg_recs("balanced", uids(12)), seg_recs("exact", uids(6)))
    assert c["verdict"] == "certified", c["reasons"]
    assert c["metrics"]["noise"]["n"] == 5 and c["metrics"]["noise"]["unpaired"] == 1
    # no spare left: too few noise samples (provisional), still a certificate
    c = certify(ex, seg_recs("balanced", uids(12)), seg_recs("exact", uids(5)))
    assert c["verdict"] == "provisional" and "too_few_noise_samples" in c["reasons"]


def test_coverage_knob_is_bounded():
    with pytest.raises(C.CertifyError, match="max_missing"):
        C.Thresholds(max_missing=3)
    assert C.Thresholds(max_missing=0).max_missing == 0


# ---------------------------------------------------------------------------------------------
# stale lab evidence


def test_l0_from_other_engine_settings_is_stale():
    ok = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12)),
                 seg_recs("exact", uids(5)))
    assert ok["l0"]["status"] == "pass"
    assert ok["l0"]["engine_hash"] == L0_PASS["engine_hashes"]["baseline"]
    for lab in ({k: v for k, v in L0_PASS.items() if k != "engine_hashes"},
                {**L0_PASS, "engine_hashes": {"baseline": "0" * 64}}):
        c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12)),
                    seg_recs("exact", uids(5)), l0=lab)
        assert c["l0"]["status"] == "stale" and c["verdict"] == "provisional"
        assert "stale_l0" in c["reasons"]


def test_engine_settings_hash_ignores_logprob_returns_only():
    from flora_deploy.engines.offline_llm import EngineSpec

    worker = EngineSpec.for_gpu("tempseg", "/m", "L40S").hashed()
    lab = EngineSpec.for_gpu("tempseg", "/m", "L40S", logprobs_mode="raw_logits",
                             max_logprobs=20).hashed()
    assert certkey.engine_settings_hash(lab) == certkey.engine_settings_hash(worker)
    assert certkey.engine_settings_hash(worker) == certkey.sha256_hex(worker)
    other = copy.deepcopy(worker)
    other["engine_kwargs"]["max_num_seqs"] = 64
    assert certkey.engine_settings_hash(other) != certkey.engine_settings_hash(worker)


# ---------------------------------------------------------------------------------------------
# preemptions and the compile cache


@pytest.mark.parametrize(("value", "reason"), [(2, "engine_preempted"),
                                               (None, "preemptions_unknown")])
def test_preempted_or_unknown_records_are_not_certified(value, reason):
    bal = seg_recs("balanced", uids(11)) + seg_recs("balanced", ["u11"],
                                                    engine_preemptions=value)
    c = certify(seg_recs("exact", uids(12)), bal, seg_recs("exact", uids(5)))
    assert c["verdict"] == "provisional" and reason in c["reasons"]
    assert c["preemptions"] == {"preempted": int(value is not None),
                                "unknown": int(value is None)}


def test_certificate_states_the_compile_caches_its_evidence_covers():
    warm = "warm:seg-key:" + "f" * 16
    bal = (seg_recs("balanced", uids(6), compile_cache="cold")
           + seg_recs("balanced", uids(6, start=6), compile_cache=warm))
    wit = seg_recs("balanced", uids(6), compile_cache="cold:forced")
    c = certify(seg_recs("exact", uids(12)), bal, seg_recs("exact", uids(6)), witness=wit)
    assert c["verdict"] == "certified", c["reasons"]
    assert c["compile_cache"] == {"candidate": {"cold": 6, warm: 6},
                                  "witness": {"cold:forced": 6},
                                  "covers": sorted(["cold", "cold:forced", warm])}
    assert c["metrics"]["witness"]["n"] == 6
    assert {ch["class"] for ch in c["checks"]} == {"yam", "cold-witness", "union"}
    assert certkey.policy_problems(c) == []


def test_a_diverging_or_unforced_cold_witness_does_not_certify():
    ex, bal, nz = seg_recs("exact", uids(12)), seg_recs("balanced", uids(12)), \
        seg_recs("exact", uids(5))
    far = seg_recs("balanced", uids(6), shift=0.5, compile_cache="cold:forced")
    c = certify(ex, bal, nz, witness=far)
    assert c["verdict"] == "fail"
    assert {ch["class"] for ch in c["checks"] if ch["pass"] is False} == {"cold-witness"}
    warmish = seg_recs("balanced", uids(6), compile_cache="warm:k:" + "f" * 16)
    c = certify(ex, bal, nz, witness=warmish)
    assert c["verdict"] == "provisional" and "witness_not_forced_cold" in c["reasons"]
    c = certify(ex, bal, nz, witness=seg_recs("balanced", uids(4), compile_cache="cold:forced"))
    assert c["verdict"] == "provisional" and "too_few_witness_samples" in c["reasons"]
    with pytest.raises(C.CertifyError, match="witness"):
        certify(ex, bal, nz, witness=seg_recs("balanced", uids(6), compile_cache="cold:forced",
                                              config_hash="c" * 64))


def test_a_tampered_witness_check_is_caught():
    wit = seg_recs("balanced", uids(6), compile_cache="cold:forced")
    c = certify(seg_recs("exact", uids(12)), seg_recs("balanced", uids(12)),
                seg_recs("exact", uids(5)), witness=wit)
    bad = copy.deepcopy(c)
    bad["metrics"]["witness"]["means"]["boundary_error_s"] = 9.0
    bad["id"] = certkey.certificate_id(bad)
    assert any("witness" in p for p in certkey.policy_problems(bad))


def test_cli_certify_takes_the_cold_witness(tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli import app
    from tests.unit.test_certify import _balanced_yaml, _content_args, _jsonl

    def recs(preset, ids, **kw):
        return seg_recs(preset, ids, fp=None, **kw)

    cfg = _balanced_yaml(tmp_path)
    ex = _jsonl(tmp_path / "exact.jsonl", recs("exact", uids(12)))
    bal = _jsonl(tmp_path / "bal.jsonl", recs("balanced", uids(12)))
    nz = _jsonl(tmp_path / "noise.jsonl", recs("exact", uids(6)))
    wit = _jsonl(tmp_path / "wit.jsonl", recs("balanced", uids(6), compile_cache="cold:forced"))
    lab = tmp_path / "lab.json"
    lab.write_text(json.dumps(L0_PASS))
    r = CliRunner().invoke(app, ["certify", "--exact", str(ex), "-c", str(cfg), "--balanced",
                                 str(bal), "--noise", str(nz), "--witness", str(wit), "--l0",
                                 str(lab), "--out-dir", str(tmp_path / "c"), "--json",
                                 *_content_args()])
    assert r.exit_code == 0, r.output
    (cert,) = json.loads(r.output)
    assert cert["verdict"] == "certified" and cert["metrics"]["witness"]["n"] == 6
    assert cert["compile_cache"]["covers"] == ["cold", "cold:forced"]
