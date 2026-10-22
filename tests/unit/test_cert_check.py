"""``scripts/cert_check.py``: the independent certificate check (``docs/certificate-check.md``).

Two halves:

- the reimplemented metric math, against hand-computed values from ``docs/certification.md``
  and, on random inputs, against ``validate/certify.py``'s own per-episode functions (the
  checker never imports them; only this test does, to show the two agree);
- the checks, on a fake certified run (``tests/fakes/certcheck_run.py``) whose certificates the
  real generator writes: a good run passes every row, and each check has a fixture that breaks
  exactly what it guards.

The current rule is generator version 3 (the count-flip tolerance, spares, cold witness,
preemption and cache evidence). Under a version-2 generator the good run fails exactly the rows
that rule adds (:data:`GAPS`), which is what the checker must say about such a certificate.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from flora_deploy.records import DenseReward, Provenance, Segment, StageRecord, TerminalReward
from flora_deploy.validate import certify as CY
from tests.fakes import certcheck_run as R
from tests.fakes import witness_cases as WC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import cert_check as cc  # noqa: E402

V3 = R.VERSION == "3"
STAGES = ("segment", "caption")
GAPS = set() if V3 else {
    *(f"2 pass rule | {s} count-flip tolerance" for s in STAGES),
    *(f"4 evidence | {s} certificate states the caches" for s in STAGES)}
# The good run has one count flip in 12 episodes (rule 3 passes it); a version-2 generator
# fails it, so its good run has none.
BASE = R.Options(flips=1 if V3 else 0)
needs_v3 = pytest.mark.skipif(not V3, reason="the current code's generator predates rule 3")


# ---------------------------------------------------------------------------------------------
# Metric math: hand-computed anchors


def seg(*bounds, captions=None, status="complete"):
    return {"segments": [{"start_s": s, "end_s": e, "status": status,
                          "segmentation_provenance": "fine",
                          "caption": captions[i] if captions else None}
                         for i, (s, e) in enumerate(bounds)]}


def test_iou_and_alignment():
    assert cc.iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert cc.iou((1, 1), (1, 1)) == 1.0 and cc.iou((1, 1), (2, 2)) == 0.0
    assert cc.iou((0, 1), (2, 3)) == 0.0
    a = [(0.0, 2.0), (2.0, 5.0), (5.0, 9.0)]
    b = [(0.0, 2.0), (2.0, 5.0), (5.0, 7.0), (7.0, 9.0)]
    # (5,9) ties between (5,7) and (7,9) at IoU 0.5: resolved from the end, as the doc says
    assert [(i, j) for i, j, _ in cc.alignment(a, b)] == [(0, 0), (1, 1), (2, 3)]
    assert cc.alignment(a, []) == [] and cc.alignment([(0, 1)], [(2, 3)]) == []


def test_segment_episode_anchors():
    one_flip = cc.seg_episode(seg((0, 2), (2, 5), (5, 9)), seg((0, 2), (2, 5), (5, 7), (7, 9)))
    assert one_flip == pytest.approx({"iou_divergence": 1 - 2.5 / 4,
                                      "boundary_error_s": (0 + 0 + 1.0) / 3,
                                      "count_mismatch": 1.0, "label_disagreement": 0.0})
    shifted = cc.seg_episode(seg((0, 2), (2, 5)), seg((0.1, 2.1), (2.1, 5.1)))
    assert shifted["boundary_error_s"] == pytest.approx(0.1)
    assert shifted["count_mismatch"] == 0.0
    assert cc.seg_episode(seg(), seg()) == dict.fromkeys(cc.METRICS_V2["segment"], 0.0)
    lone = cc.seg_episode(seg(), seg((0, 1)))
    assert lone["iou_divergence"] == 1.0 and lone["boundary_error_s"] is None
    assert cc.seg_episode(seg((0, 1)), seg((0, 1), status="partial"))[
        "label_disagreement"] == 1.0


def test_caption_episode_anchors():
    caps = ["reach for it", "lift the blue mug slowly", "put it down"]
    typo = [caps[0], "lift the blue mug slowlv", caps[2]]
    m = cc.cap_episode(seg((0, 1), (1, 2), (2, 3), captions=caps),
                       seg((0, 1), (1, 2), (2, 3), captions=typo))
    assert m["caption_mismatch"] == pytest.approx(1 / 3)
    assert m["caption_edit_distance"] == pytest.approx((1 / 24) / 3)
    assert m["count_mismatch"] == 0.0
    extra = cc.cap_episode(seg((0, 1), captions=["a"]), seg((0, 1), (1, 2), captions=["a", "b"]))
    assert extra == {"caption_mismatch": 0.5, "caption_edit_distance": 0.5,
                     "count_mismatch": 1.0}
    # strip(), and None is ""
    assert cc.cap_episode(seg((0, 1), captions=[" x "]), seg((0, 1), captions=["x"]))[
        "caption_mismatch"] == 0.0
    assert cc.cap_episode(seg((0, 1), captions=[None]), seg((0, 1), captions=[""]))[
        "caption_edit_distance"] == 0.0
    assert cc.lev("kitten", "sitting") == 3 and cc.lev("", "abc") == 3


def test_reward_anchors():
    def dense(*xs):
        return {"dense": {"per_step": list(xs)}}

    # docs: one 2 in ten 3s scores 0.1, two equal constant series 0
    assert cc.dense_episode(dense(*[3] * 10), dense(*([3] * 9 + [2])))[
        "rank_divergence"] == pytest.approx(0.1)
    assert cc.dense_episode(dense(3, 3), dense(3, 3))["rank_divergence"] == 0.0
    assert cc.dense_episode(dense(1, 2, 3), dense(3, 2, 1))["rank_divergence"] == \
        pytest.approx(2.0)
    assert cc.dense_episode(dense(1, 2, 3), dense(1, 2))["length_mismatch"] == 1.0
    assert cc.spearman_rho([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    t = cc.terminal_episode({"terminal": {"score": 4, "expected_score": 4.2}},
                            {"terminal": {"score": 3, "expected_score": 4.0}})
    assert t == pytest.approx({"expected_score_abs_delta": 0.2, "score_mismatch": 1.0})
    assert cc.terminal_episode({"terminal": None}, {"terminal": {"score": 1}}) is None


def test_thresholds_and_count_basis():
    assert cc.threshold("count_mismatch", 0.0, 1.5, 0.0, "3") == 0.1  # 1 in 10 passes
    assert cc.threshold("count_mismatch", 0.0, 1.5, 0.0, "2") == 0.075  # the old rule
    assert cc.threshold("count_mismatch", 0.1, 1.5, 0.0, "3") == 0.15
    assert cc.threshold("boundary_error_s", 0.0, 1.5, 0.0, "3") == 0.15
    assert cc.threshold("expected_score_abs_delta", 0.5, 1.5, 0.0, "3") == 0.05  # the cap
    eps = {"a": {"count_mismatch": 1.0, "iou_divergence": 0.5, "boundary_error_s": 1.0,
                 "label_disagreement": 0.0},
           "b": {"count_mismatch": 0.0, "iou_divergence": 0.0, "boundary_error_s": None,
                 "label_disagreement": 0.0}}
    v3 = cc.means("segment", "3", eps)
    assert v3["count_mismatch"] == 0.5 and v3["iou_divergence"] == 0.0
    assert v3["boundary_error_s"] is None  # undefined values are left out of the mean
    assert cc.means("segment", "2", eps)["iou_divergence"] == 0.25


# ---------------------------------------------------------------------------------------------
# Metric math: agrees with the generator's own functions on random inputs (test-only import)


def _rand_record(rng, stage):
    n = rng.randint(0, 5)
    t, segs = 0.0, []
    for i in range(n):
        s = round(t + rng.choice((0.0, 0.25, 0.5)), 2)
        e = round(s + rng.choice((0.25, 0.5, 1.0, 2.0, 3.5)), 2)
        t = e if rng.random() < 0.8 else s  # sometimes overlapping
        segs.append({"event_id": f"e{i}", "start_s": s, "end_s": e,
                     "status": rng.choice(("complete", "complete", "partial")),
                     "caption": rng.choice(("pick up", "pick up the cup", "place", None, " a ")),
                     "segmentation_provenance": rng.choice(("fine", "coarse"))})
    steps = [rng.randint(1, 3) for _ in range(rng.randint(0, 8))]
    es = round(rng.uniform(1, 5), 3)
    return {"segments": segs, "terminal": {"score": rng.randint(1, 5), "expected_score": es,
                                           "probabilities": [0.2] * 5},
            "dense": {"segments": [], "per_step": steps}}


def _as_stage_record(d, stage):
    prov = Provenance(models={}, engine="x", preset="balanced", config_hash="0" * 64,
                      logic_version="0", framework="t")
    return StageRecord(
        uid="u", dataset="d", episode_id="u", stage=stage, provenance=prov, certified=True,
        segments=tuple(Segment(**s) for s in d["segments"]),
        terminal=TerminalReward(score=d["terminal"]["score"],
                                expected_score=d["terminal"]["expected_score"],
                                probabilities=tuple(d["terminal"]["probabilities"])),
        dense=DenseReward(segments=(), per_step=tuple(d["dense"]["per_step"])))


@pytest.mark.parametrize("stage", ["segment", "caption", "reward.terminal", "reward.dense"])
def test_agrees_with_generator_per_episode(stage):
    rng = random.Random(f"cert-check-{stage}")
    for _ in range(400):
        a, b = _rand_record(rng, stage), _rand_record(rng, stage)
        if rng.random() < 0.2:
            b = json.loads(json.dumps(a))
        mine = cc.EPISODE[stage](a, b)
        theirs = CY.episode_metrics(stage, _as_stage_record(a, stage), _as_stage_record(b, stage))
        assert mine is not None and theirs is not None
        for m, v in theirs.items():
            assert (mine[m] is None) == (v is None), (m, a, b)
            if v is not None:
                assert mine[m] == pytest.approx(v, abs=1e-12), (m, a, b)


# ---------------------------------------------------------------------------------------------
# The checks on a fake certified run


def ok_runner(root):
    return 0, "17 passed"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return R.build(tmp_path_factory.mktemp("cert-run"), BASE, drift_runner=ok_runner)


@pytest.fixture
def fake(base, tmp_path):
    return base.clone(tmp_path / "run")


def build(tmp_path, **opts):
    return R.build(tmp_path / "run", R.Options(**opts), drift_runner=ok_runner)


def rows_of(f, **kw):
    return R.by_item(f.run(**kw))


def assert_fail(rows, *items):
    for item in items:
        assert item in rows, (item, sorted(rows))
        assert rows[item].status == cc.FAIL, (item, rows[item])


def test_good_run_passes_every_row(base):
    rows = base.run()
    assert R.failing(rows) == GAPS, [(r.item, r.detail) for r in rows if r.status == cc.FAIL]
    checks = {r.check for r in rows}
    assert {"1 metrics", "2 pass rule", "3 coverage", "4 evidence", "5 freeze lock",
            "6 leakage", "7 spend"} <= checks
    by = R.by_item(rows)
    assert by["5 freeze lock | tests/unit/test_certificate_drift.py"].status == cc.PASS
    if V3:
        assert by["2 pass rule | segment count-flip tolerance"].detail.startswith(
            "1 of 18 episodes differ in count (1 of 12 human, 0 of 6 cold-witness); "
            "threshold 0.1 allows 2")
        assert by["1 metrics | segment oracle vs cold witness"].status == cc.PASS


def test_cli_table_json_and_exit_code(base, capsys):
    args = ["--run-root", str(base.root), "--cert-dir", str(base.cert_dir), "--lab-dir",
            str(base.inp.lab_dir), "--report", str(base.inp.report), "--nv-root",
            str(base.inp.nv_root)]
    code = cc.main([*args, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == (0 if V3 else 1) and out["ok"] is V3
    assert {r["status"] for r in out["rows"]} <= {cc.PASS, cc.FAIL}
    assert cc.main(args) == code
    text = capsys.readouterr().out
    assert text.startswith("| check | item | result | detail |")
    assert ("Overall: PASS" if V3 else "Overall: FAIL") in text


def test_missing_inputs_fail(tmp_path, fake):
    fake.records("balanced", "caption").unlink()
    for p in fake.cert_dir.glob("*.json"):
        if p.name.startswith("cert_") and json.loads(p.read_text())["stage"] == "segment":
            p.unlink()
    rows = rows_of(fake)
    assert_fail(rows, "0 inputs | balanced caption records", "0 inputs | segment certificate")
    empty = cc.check_all(cc.Inputs(run_root=tmp_path / "nothing", cert_dir=tmp_path / "none",
                                   lab_dir=tmp_path / "none", report=tmp_path / "none.md"))
    inputs = {r.item: r.status for r in empty if r.check == "0 inputs"}
    assert inputs and set(inputs.values()) == {cc.FAIL}
    assert "state" in inputs and "segment certificate" in inputs


# 1. metrics


def test_metric_mismatch(fake):
    def bump(c):
        c["metrics"]["candidate"]["human"]["means"]["boundary_error_s"] += 0.01

    fake.edit_cert("segment", bump)
    fake.edit_cert("caption", lambda c: c["metrics"]["noise"]["means"].update(
        caption_edit_distance=0.5))
    rows = rows_of(fake)
    assert_fail(rows, "1 metrics | segment oracle vs balanced",
                "1 metrics | caption oracle vs noise")
    assert "boundary_error_s" in rows["1 metrics | segment oracle vs balanced"].detail
    assert rows["1 metrics | caption oracle vs balanced"].status == cc.PASS


def test_metric_counts_and_coverage_mismatch(fake):
    fake.edit_cert("segment", lambda c: c["metrics"].update(coverage=0.5))
    fake.edit_cert("caption", lambda c: c["metrics"]["candidate"]["human"].update(n=11))
    rows = rows_of(fake)
    assert_fail(rows, "1 metrics | segment coverage", "1 metrics | caption oracle vs balanced")


def test_records_changed_after_certifying(fake):
    """A record edited after flora certify ran: the recomputed metrics no longer agree."""
    def shift(rows):
        rows[1]["segments"][0]["end_s"] += 0.5

    fake.edit_records("balanced", "segment", shift)
    assert_fail(rows_of(fake), "1 metrics | segment oracle vs balanced")


@needs_v3
def test_witness_metrics_mismatch(fake):
    fake.edit_cert("segment", lambda c: c["metrics"]["witness"].update(unpaired=3))
    assert_fail(rows_of(fake), "1 metrics | segment oracle vs cold witness")


def test_similarity_hook_cannot_be_recomputed(fake):
    fake.edit_cert("caption", lambda c: c["metrics"].update(similarity_hook="mod:fn"))
    assert_fail(rows_of(fake), "1 metrics | caption similarity")


# 2. pass rule


def test_knobs_outside_policy(fake):
    fake.edit_cert("segment", lambda c: c["thresholds"].update(tolerance=2.5, min_samples=8))
    rows = rows_of(fake)
    assert_fail(rows, "2 pass rule | segment knobs within policy")
    assert "tolerance 2.5" in rows["2 pass rule | segment knobs within policy"].detail


def test_loosened_floor_and_threshold(fake):
    def loosen(c):
        pm = c["thresholds"]["per_metric"]["boundary_error_s"]
        pm["floor"], pm["threshold"] = 0.5, 0.75

    fake.edit_cert("segment", loosen)
    assert_fail(rows_of(fake), "2 pass rule | segment floors and caps",
                "2 pass rule | segment thresholds")


def test_check_and_verdict_tampered(fake):
    def flip(c):
        c["checks"][0]["pass"] = not c["checks"][0]["pass"]

    fake.edit_cert("segment", flip)
    fake.edit_cert("caption", lambda c: c.update(verdict="certified", reasons=[], unlocks=True)
                   if c["verdict"] != "certified" else c.update(reasons=["no_l0"]))
    rows = rows_of(fake)
    assert_fail(rows, "2 pass rule | segment checks", "2 pass rule | caption verdict",
                "2 pass rule | caption publishable")


@needs_v3
def test_one_count_flip_in_ten_passes_two_fail(tmp_path):
    """Over the union: 2 in 12 + 6 witness episodes pass (rounded half up), 3 in 18 fail."""
    two = R.by_item(build(tmp_path, flips=2).run())
    assert "certified" in two["2 pass rule | segment verdict"].detail
    three = R.by_item(build(tmp_path / "3", flips=3).run())
    assert three["2 pass rule | segment count-flip tolerance"].status == cc.PASS  # the rule
    assert three["2 pass rule | segment verdict"].status == cc.PASS  # agrees: fail
    assert "fail" in three["2 pass rule | segment verdict"].detail
    assert "union count_mismatch" in three["2 pass rule | segment checks"].detail
    assert_fail(three, "2 pass rule | segment publishable")


@needs_v3
@pytest.mark.parametrize("case", WC.CASES, ids=WC.CASE_IDS)
def test_witness_union_cases(tmp_path, case):
    """The table the generator is tested on (tests/unit/test_certify_witness_union.py): count
    flips over the union of the candidate's and the witness's episodes, witness means alone."""
    opts = {"episodes": R.EPISODES[:case.main], "flips": case.main_flips,
            "witness_flips": case.witness_flips,
            "witness_drift": case.drift if case.drift in ("boundary", "caption", "nan")
            else None,
            "witness": case.drift != "absent"}
    if case.drift == "warm":
        opts["witness_cache"] = R.WARM
    f = build(tmp_path, **opts)
    rows = R.by_item(f.run())
    s = case.stage
    # the checker agrees with the certificate, whatever the verdict
    for item in ("checks", "verdict", "count-flip tolerance"):
        row = rows[f"2 pass rule | {s} {item}"]
        assert row.status == cc.PASS, row
    if case.drift != "absent":
        assert rows[f"1 metrics | {s} oracle vs cold witness"].status == cc.PASS
    verdict = rows[f"2 pass rule | {s} verdict"].detail
    assert verdict.split(" ")[0] == case.verdict, verdict
    for reason in case.reasons:
        assert reason in verdict, verdict
    flips, n, allowed = WC.UNION[case.id]
    assert rows[f"2 pass rule | {s} count-flip tolerance"].detail.startswith(
        f"{flips} of {n} episodes differ in count")
    assert f"allows {allowed}" in rows[f"2 pass rule | {s} count-flip tolerance"].detail
    failing = rows[f"2 pass rule | {s} checks"].detail
    want = {f"{c} {m}" for c, m in case.failing}
    got = set(failing.split("failing: ")[1].split(", ")) if "failing: " in failing else set()
    assert got == want, failing
    if case.verdict == "certified":
        assert rows[f"2 pass rule | {s} publishable"].status == cc.PASS
    else:
        assert_fail(rows, f"2 pass rule | {s} publishable")


@needs_v3
def test_union_count_flips_misstated(fake):
    def hide_witness_flip(c):
        c["metrics"]["count_flips"]["groups"]["cold-witness"]["flips"] = 1
        for ch in c["checks"]:
            if ch["class"] == "union":
                ch["flips"] = 2

    fake.edit_cert("segment", hide_witness_flip)
    assert_fail(rows_of(fake), "2 pass rule | segment count-flip tolerance",
                "2 pass rule | segment checks")


@needs_v3
def test_count_flip_rule_loosened_or_tightened(fake):
    def old_rule(c):
        pm = c["thresholds"]["per_metric"]["count_mismatch"]
        pm["absolute_floor"], pm["threshold"] = 0.0, 0.075
        c["thresholds"]["absolute_floor"] = {}
        for ch in c["checks"]:
            if ch["metric"] == "count_mismatch":
                ch["threshold"] = 0.075

    fake.edit_cert("segment", old_rule)
    rows = rows_of(fake)
    assert_fail(rows, "2 pass rule | segment count-flip tolerance",
                "2 pass rule | segment floors and caps", "2 pass rule | segment thresholds")


@needs_v3
def test_too_few_samples_is_provisional(tmp_path):
    f = build(tmp_path, episodes=R.EPISODES[:8], noise=R.NOISE_IDS[:5], flips=0)
    rows = R.by_item(f.run())
    assert rows["2 pass rule | segment verdict"].status == cc.PASS
    assert "too_few_samples" in rows["2 pass rule | segment verdict"].detail
    assert_fail(rows, "2 pass rule | segment publishable")
    ok = R.by_item(f.run(allow_provisional=True))
    assert ok["2 pass rule | segment publishable"].status == cc.PASS


# 3. coverage


def test_episode_ids_differ(fake):
    def swap(c):
        c["episodes"]["uid_sha256"][0] = "0" * 64

    fake.edit_cert("segment", swap)
    assert_fail(rows_of(fake), "3 coverage | segment episode ids")


def test_noise_episode_not_the_oracles(fake):
    def add(rows):
        extra = json.loads(json.dumps(rows[0]))
        extra["uid"], extra["episode_id"] = "cert-human--z-other", "z-other"
        rows.append(extra)

    fake.edit_records("noise", "segment", add)
    rows = rows_of(fake)
    assert_fail(rows, "3 coverage | segment noise episodes within the oracle's")


def test_balanced_stage_hash_and_key(fake):
    def other_hash(rows):
        rows[2]["provenance"]["config_hash"] = "c" * 64

    def other_key(rows):
        for r in rows:
            r["provenance"]["cert_key"]["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 7

    fake.edit_records("balanced", "segment", other_hash)
    fake.edit_records("balanced", "caption", other_key)
    fake.edit_records("witness", "caption", other_key)
    rows = rows_of(fake)
    assert_fail(rows, "3 coverage | segment one balanced stage hash",
                "3 coverage | caption cert_key is the current code's")
    assert "vllm_engine" in rows["3 coverage | caption cert_key is the current code's"].detail


def test_oracle_binding(fake):
    def other_gpu(rows):
        for r in rows:
            r["provenance"]["gpu"] = "NVIDIA H100 80GB HBM3"

    fake.edit_records("noise", "segment", other_gpu)
    fake.edit_cert("caption", lambda c: c["oracle"].update(determinism="deterministic"))
    rows = rows_of(fake)
    assert_fail(rows, "3 coverage | segment oracle binding", "3 coverage | caption oracle binding")


def test_embodiment_and_gpu(fake):
    def yam(rows):
        rows[0]["provenance"]["embodiment"] = "yam"

    fake.edit_records("exact", "segment", yam)
    fake.edit_cert("caption", lambda c: c.update(gpu="H100"))
    rows = rows_of(fake)
    assert_fail(rows, "3 coverage | segment embodiment", "3 coverage | caption GPU")


# 4. evidence


def test_cache_provenance_missing(fake):
    def drop(rows):
        rows[0]["provenance"]["compile_cache"] = None

    fake.edit_records("balanced", "segment", drop)
    assert_fail(rows_of(fake), "4 evidence | segment cache provenance per record")


def test_no_forced_cold_container(tmp_path):
    rows = R.by_item(build(tmp_path, witness_cache="cold").run())
    assert_fail(rows, "4 evidence | segment forced-cold container")
    nowit = R.by_item(build(tmp_path / "b", witness=False).run())
    assert_fail(nowit, "4 evidence | caption forced-cold container")


@needs_v3
def test_cache_counts_misstated(fake):
    fake.edit_cert("segment", lambda c: c["compile_cache"].update(covers=[R.WARM]))
    assert_fail(rows_of(fake), "4 evidence | segment certificate states the caches")


def test_preemption_after_certifying_fails(fake):
    def preempt(rows):
        rows[4]["provenance"]["engine_preemptions"] = 2

    fake.edit_records("balanced", "segment", preempt)
    assert_fail(rows_of(fake), "4 evidence | segment preemptions")


@needs_v3
def test_preempted_run_is_provisional_not_publishable(tmp_path):
    rows = R.by_item(build(tmp_path, preemptions=3).run())
    assert rows["4 evidence | segment preemptions"].status == cc.PASS  # stated: provisional
    assert "engine_preempted" in rows["2 pass rule | segment verdict"].detail
    assert_fail(rows, "2 pass rule | segment publishable")
    unknown = R.by_item(build(tmp_path / "u", preemptions=None).run())
    assert "preemptions_unknown" in unknown["2 pass rule | caption verdict"].detail


def test_lab_engine_hash_stale(tmp_path):
    rows = R.by_item(build(tmp_path, lab_hash="0" * 64).run())
    assert_fail(rows, "4 evidence | segment lab engine-settings hash",
                "4 evidence | caption lab engine-settings hash")
    if V3:
        assert "stale_l0" in rows["2 pass rule | segment verdict"].detail


def test_lab_missing_or_without_gate_a(fake):
    (fake.inp.lab_dir / "balanced-tempseg.json").unlink()
    p = fake.inp.lab_dir / "balanced-captioner.json"
    lab = json.loads(p.read_text())
    lab["gate_a"]["requests"] = 0
    p.write_text(json.dumps(lab))
    rows = rows_of(fake)
    assert_fail(rows, "4 evidence | segment lab tempseg",
                "4 evidence | caption L0 and Gate A present")


def test_certificate_l0_misstated(fake):
    fake.edit_cert("caption", lambda c: c["l0"].update(requests=7))
    assert_fail(rows_of(fake), "4 evidence | caption certificate L0 / Gate A")


# 5. freeze lock


def test_lock_missing(fake):
    fake.lock_path("segment").unlink()
    rows = rows_of(fake)
    assert_fail(rows, "5 freeze lock | segment lock",
                "5 freeze lock | drift guard (certfreeze.check)")


def test_lock_key_or_modules_differ(fake):
    p = fake.lock_path("segment")
    lock = json.loads(p.read_text())
    lock["key_document"]["logic_digest"] = "0" * 64
    p.write_text(json.dumps(lock))
    q = fake.lock_path("caption")
    lock = json.loads(q.read_text())
    first = sorted(lock["modules"])[0]
    lock["modules"][first] = "0" * 64
    q.write_text(json.dumps(lock))
    rows = rows_of(fake)
    assert_fail(rows, "5 freeze lock | segment lock matches the certificate",
                "5 freeze lock | caption lock is the current code")
    assert first in rows["5 freeze lock | caption lock is the current code"].detail


def test_drift_test_failure(fake):
    rows = rows_of(fake, drift_runner=lambda root: (1, "1 failed, 16 passed"))
    assert_fail(rows, "5 freeze lock | tests/unit/test_certificate_drift.py")


def test_drift_guard_sees_a_drifted_certificate(fake):
    """A certificate issued for a key the current code no longer computes (another logic
    digest; its body, id and lock consistent with it) fails the drift guard, the lock and the
    key rows."""
    lock = json.loads(fake.lock_path("segment").read_text())
    doc = lock["key_document"]
    doc["logic_digest"] = "0" * 64

    def drift(c):
        c["logic"]["digest"] = doc["logic_digest"]
        c["engine_spec_hash"] = cc.sha(doc)

    fake.edit_cert("segment", drift)
    lock["key_document"], lock["engine_spec_hash"] = doc, cc.sha(doc)
    lock["certificate"] = fake.certs["segment"]["id"]
    fake.lock_path("segment").write_text(json.dumps(lock))
    rows = rows_of(fake)
    assert rows["5 freeze lock | segment lock matches the certificate"].status == cc.PASS
    assert_fail(rows, "5 freeze lock | drift guard (certfreeze.check)",
                "5 freeze lock | segment lock is the current code",
                "3 coverage | segment cert_key is the current code's")


# 6. leakage


def test_caption_or_episode_id_in_report(fake):
    with fake.inp.report.open("a", encoding="utf-8") as fh:
        fh.write(f"\nNote: {R.CAPTIONS[1]}\n")
    rows = rows_of(fake)
    assert_fail(rows, "6 leakage | report: record strings")
    fake.inp.report.write_text(fake.inp.report.read_text().replace(R.CAPTIONS[1], R.uid(
        R.EPISODES[4])))
    assert_fail(rows_of(fake), "6 leakage | report: record strings")


def test_free_text_in_certificate(fake):
    fake.edit_cert("segment", lambda c: c.update(note="the operator reviewed this by hand"))
    fake.edit_cert("caption", lambda c: c["metrics"].update(example=R.CAPTIONS[0]))
    rows = rows_of(fake)
    assert_fail(rows, "6 leakage | segment certificate string fields",
                "6 leakage | caption certificate string fields",
                "6 leakage | caption certificate: record strings")


def test_nvidia_text_in_report(fake):
    with fake.inp.report.open("a", encoding="utf-8") as fh:
        fh.write("\n" + R.FAKE_NV + "\n")
    assert_fail(rows_of(fake), "6 leakage | NVIDIA text (shingles)")


def test_nvidia_code_not_fetched(fake, tmp_path):
    rows = rows_of(fake, nv_root=tmp_path / "empty")
    assert_fail(rows, "6 leakage | NVIDIA text (shingles)")


# 7. spend


def test_report_totals_differ_from_state(fake):
    text = fake.inp.report.read_text()
    fake.inp.report.write_text(text.replace("Total: **$4.62**", "Total: **$4.12**"))
    assert_fail(rows_of(fake), "7 spend | report totals")


def test_report_step_row_differs(fake):
    text = fake.inp.report.read_text()
    fake.inp.report.write_text(text.replace("| $1.85 |", "| $1.35 |", 1))
    assert_fail(rows_of(fake), "7 spend | report per-step rows")


def test_state_changed_after_the_report(fake):
    st = json.loads(fake.inp.state.read_text())
    st["billed_total"] = 6.0
    st["gate_cap"] = 5.0
    fake.inp.state.write_text(json.dumps(st))
    rows = rows_of(fake)
    assert_fail(rows, "7 spend | report totals", "7 spend | state is consistent")
    assert "over the gated cap" in rows["7 spend | state is consistent"].detail
