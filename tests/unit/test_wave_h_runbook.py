"""``scripts/wave_h.py``: projections against the Wave H-lite caps, stop rules, resumable state,
the offline certification metrics and the numbers-only report. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wh = _load("wave_h")
h0 = sys.modules["h0_rehearsal"]


def _manifest() -> dict:
    """The golden durations the runbook prices (golden manifest v1, numbers only)."""
    dur = {"human/egodex_add_remove_lid/episode_000014": 14.6,
           "human/egodex_braid_unbraid/episode_000001": 16.1,
           "yam/aloha_static_cups_open/episode_000000": 8.0,
           "yam/aloha_static_towel/episode_000000": 10.0,
           "single_view_robot/wgo/robointer_droid_000010": 15.2,
           "single_view_robot/wgo/galaxea_045": 12.6}
    return {"episodes": [{"id": k, "duration_s": v} for k, v in dur.items()]}


def test_same_six_episodes_as_h0_and_subsets():
    ids = [i for v in h0.EPISODES.values() for i in v]
    assert len(ids) == 6 and len(set(ids)) == 6
    pools = {i.split("/")[0] for i in ids}
    assert pools == {"human", "yam", "single_view_robot"}
    assert set(wh.REWARD_EPISODES) <= set(ids) and len(wh.REWARD_EPISODES) == 3
    assert set(wh.NOISE_EPISODES) <= set(ids) and len(wh.NOISE_EPISODES) == 2


def test_projections_are_honest_about_the_caps():
    """H2: upper projections (upper quantiles, every cold start, scale-down tail and the client
    slack per app) are what the stop rule gates on; the dry run names the minimum honest cap."""
    m = _manifest()
    ps = {step: wh.project(step, m) for step in wh.STEPS}
    for p in ps.values():
        assert p.usd_upper >= p.usd
    for it in ps["exact_e2e"].items + ps["noise"].items:  # every app carries the client slack
        assert it["upper_seconds"] - it["seconds"] >= wh.CLIENT_SLACK_S
    # exact on the preset's H200 pool (two apps: two cold starts); balanced on its L40S + H100
    # pools (two apps, one cold start per pool each) plus the two lab containers
    assert [i["gpu"] for i in ps["exact_e2e"].items] == ["H200", "H200"]
    assert [i["gpu"] for i in ps["balanced"].items] == ["L40S", "H100!", "L40S", "H100!",
                                                        "L40S", "H100"]
    assert wh.CAPS_USD["exact_e2e"] == 1.80 and wh.GLOBAL_CAP_USD == 3.20
    assert ps["exact_e2e"].usd <= wh.CAPS_USD["exact_e2e"]  # nominal: two cold starts fit
    honest = wh.honest_caps(ps)
    for step in ("rewards", "balanced", "noise", "preflight"):
        assert ps[step].usd_upper <= wh.CAPS_USD[step] <= honest[step]
    upper = sum(p.usd_upper for p in ps.values())
    assert upper > wh.GLOBAL_CAP_USD  # reported by the dry run, not hidden
    assert honest["global"] >= upper


def test_noise_repeats_the_oracle_human_job():
    assert wh.NOISE_EPISODES == h0.EPISODES[wh.NOISE_JOB] == h0.EPISODES["human"]
    assert wh.ORACLE_DETERMINISM == "nvidia"  # Fix-Cert: oracle and noise in default mode


def test_container_prices_are_list_prices():
    from flora_deploy.jobs.budget import PINNED_PRICES

    c = wh.Container("H200", 8.0, 96.0)
    assert c.usd_per_h() == pytest.approx(PINNED_PRICES["H200"] + 8 * 0.0472 + 96 * 0.008)
    assert wh.Container(None, 2.0, 4.0).usd_per_h() == pytest.approx(2 * 0.0472 + 4 * 0.008)
    assert wh.Container("H100!", 1, 1).usd_per_s() == wh.Container("H100", 1, 1).usd_per_s()


def test_stop_rules_before_after_and_deadline():
    st = wh.State("r", 0.0, "2026-10-21")
    p = wh.Projection("exact_e2e")
    p.add("x", wh.Container("H200", 8, 96), 600, upper_s=1300)  # nominal fits, upper not
    assert p.usd < wh.CAPS_USD["exact_e2e"] < p.usd_upper
    with pytest.raises(wh.StopRule, match="upper projection"):
        wh.check_before(st, "exact_e2e", p)  # H2: the upper figure is what is gated
    st.step("balanced").attempts.append(wh.Attempt(0.0, "done", tracked_usd=2.5))
    small = wh.Projection("exact_e2e")
    small.add("x", wh.Container("H200", 8, 96), 400, upper_s=600)  # fits the step only
    with pytest.raises(wh.StopRule, match="global"):
        wh.check_before(st, "exact_e2e", small)
    st.step("rewards").billed_usd = 0.95  # billed beats tracked when larger
    with pytest.raises(wh.StopRule, match="rewards"):
        wh.check_after(st, "rewards")
    st2 = wh.State("r", 0.0, "2026-10-21")
    c = wh.Container("H200", 8, 96)
    r = c.usd_per_s()
    # the deadline keeps the client slack: deadline + slack never spends more than the cap
    d = wh.gpu_deadline_s(st2, "noise", c)
    assert d == pytest.approx((wh.CAPS_USD["noise"] - wh.CLIENT_SLACK_S * r) / r)
    assert (d + wh.CLIENT_SLACK_S) * r == pytest.approx(wh.CAPS_USD["noise"])
    assert wh.gpu_deadline_s(st2, "noise", c, spent_in_step=1.2) == 0.0


def test_state_round_trip_and_resume(tmp_path):
    st = wh.State("abc", 1.0, "2026-10-21", {"flora-x": 1.0})
    st.step("preflight").status = "done"
    st.step("preflight").attempts.append(wh.Attempt(1.0, "done", tracked_usd=0.01))
    path = tmp_path / "state.json"
    wh.save_state(st, path)
    back = wh.load_state(path)
    assert back.steps["preflight"].status == "done" and back.spent == pytest.approx(0.01)
    assert back.billing_baseline == {"flora-x": 1.0}


def test_segment_metrics():
    ref = [{"start_s": 0, "end_s": 4, "caption": "grasp the cup"},
           {"start_s": 4, "end_s": 10, "caption": "place the cup"}]
    same = wh.compare_segments(ref, ref)
    assert same == {"count_delta": 0.0, "boundary_mae_s": 0.0, "segment_iou": 1.0,
                    "caption_jaccard": 1.0}
    shifted = wh.compare_segments(ref, [{"start_s": 0, "end_s": 5, "caption": "grasp the cup"},
                                        {"start_s": 5, "end_s": 10, "caption": "lift a box"}])
    assert shifted["boundary_mae_s"] == pytest.approx(1 / 3, abs=1e-3)
    assert 0 < shifted["segment_iou"] < 1 and 0 < shifted["caption_jaccard"] < 1
    assert wh.compare_segments(ref, [])["segment_iou"] == 0.0
    agg = wh.aggregate({"a": same, "b": shifted})
    assert agg["episodes"] == 2 and agg["segment_iou"] < 1


def test_certify_needs_l0_gate_a_and_metrics():
    good = {"boundary_mae_s": 0.2, "segment_iou": 0.9, "caption_jaccard": 0.8}
    l0 = [{"ids_equal": True, "grid_equal": True}]
    gate = {"max_abs_dlogit": 0.01}
    assert wh.certify(good, l0=l0, gate_a=gate)["provisional"]
    assert not wh.certify(good, l0=[{"ids_equal": False, "grid_equal": True}],
                          gate_a=gate)["provisional"]
    assert not wh.certify(good, l0=l0, gate_a=None)["provisional"]
    # against a noise floor: exact-vs-exact IoU 0.97 allows down to 0.94 with factor 2
    v = wh.certify(good, l0=l0, gate_a=gate,
                   noise={"boundary_mae_s": 0.05, "segment_iou": 0.97, "caption_jaccard": 0.95})
    assert v["against"] == "noise floor" and not v["checks"]["segment_iou"]


def test_report_is_numbers_only():
    st = wh.State("r", 0.0, "2026-10-21")
    s = st.step("certify")
    s.status = "done"
    s.result = {"verdict": {"provisional": False}, "rows": [{"caption": "PRIVATE"}],
                "per": {"x": {"task": "PRIVATE", "segment_iou": 0.5}}}
    text = wh.render_report(st, {k: wh.project(k, _manifest()) for k in wh.STEPS})
    assert "PRIVATE" not in text and "segment_iou" in text
    assert "| e. offline provisional certification | done |" in text


def test_dry_run_spends_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(h0, "golden_manifest", _manifest)

    def boom(*a, **k):
        raise AssertionError("dry run must not run a step")

    monkeypatch.setattr(wh, "RUNNERS", {k: boom for k in wh.STEPS})
    monkeypatch.setattr(wh, "billed_by_app", boom)
    state = tmp_path / "state.json"
    assert wh.main(["--dry-run", "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert "total: nominal" in out and "dry run" in out and not state.exists()
    assert "nominal $" in out and "upper $" in out
    assert "NOT HONEST AT $3.20" in out and "at least $" in out  # H2: reported, not hidden
    assert json.dumps(wh.CAPS_USD)  # caps are plain data


# --- H1: receipts resolved as `flora deploy` resolves them -------------------------------------


def _receipt(tag: str) -> bytes:
    return json.dumps({"kind": tag, "ok": True}).encode()


def test_receipts_come_from_the_volume_and_are_pin_checked(monkeypatch):
    from flora_deploy.config import receipts
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.prep.derived import DERIVED_BUILDS, receipt_sha256

    seen = []
    monkeypatch.setattr(receipts, "check_pinned", lambda n, r, where=None: seen.append(n))
    files = {DERIVED_BUILDS["captioner_bf16"].receipt_path(): _receipt("c")}
    got = wh.resolve_receipts(["captioner_bf16"], read=files.get)
    assert got == {"captioner_bf16": receipt_sha256(_receipt("c"))}
    assert seen == ["captioner_bf16"]
    with pytest.raises(RuntimeError, match="flora prepare"):
        wh.resolve_receipts(["tempseg_merged"], read=files.get)
    files[DERIVED_BUILDS["tempseg_merged"].receipt_path()] = _receipt("t")
    assert set(wh.resolve_receipts(["tempseg_merged"], read=files.get)) == {"tempseg_merged"}
    if DERIVED_PINS_BY_KEY["tempseg-merged"].content_sha256 is None:  # balanced fails closed
        with pytest.raises(RuntimeError, match="pinned"):
            wh.resolve_receipts(["tempseg_merged"], read=files.get, require_pin=True)


def test_receipt_pin_mismatch_is_refused(monkeypatch):
    from flora_deploy.config import receipts
    from flora_deploy.prep.derived import DERIVED_BUILDS

    def bad(name, receipt, where=None):
        raise receipts.ReceiptPinMismatch(f"{where}: not the pinned content")

    monkeypatch.setattr(receipts, "check_pinned", bad)
    files = {DERIVED_BUILDS["captioner_bf16"].receipt_path(): _receipt("c")}
    with pytest.raises(receipts.ReceiptPinMismatch):
        wh.resolve_receipts(["captioner_bf16"], read=files.get)


def test_no_pin_attribute_lookups_remain():
    """H1 regression: ``DerivedPin`` has no ``receipt_sha256``."""
    src = (SCRIPTS / "wave_h.py").read_text()
    assert ".receipt_sha256" not in src and "_receipt()" not in src


def test_generated_configs_carry_weights_derived_and_resolve_like_deploy(tmp_path, monkeypatch):
    from flora_deploy.config.receipts import resolve
    from flora_deploy.config.schema import loads_config
    from flora_deploy.modal_app.factory import plan_app

    (tmp_path / "d").mkdir()
    monkeypatch.chdir(tmp_path)
    derived = {"captioner_bf16": "a" * 64, "tempseg_merged": "b" * 64}
    text = h0.config_text(project="wave-h-x-balanced-human", data_dir="d", embodiment="human",
                          stages=("segment", "caption"), store_volume=wh.STORE_VOLUME,
                          media_volume=wh.MEDIA_VOLUME, run_id="x", preset="balanced",
                          derived=derived)
    cfg = loads_config(text, source=str(tmp_path / "flora.yaml"))
    assert cfg.allow_uncertified and resolve(cfg) == derived
    plan = plan_app(cfg, determinism="nvidia", derived_receipts=resolve(cfg))
    assert plan.worker_classes == {"segment": "SegmentWorker", "caption": "CaptionWorker"}


# --- H3: cumulative, crash-safe spend ----------------------------------------------------------


def test_retries_add_instead_of_overwriting():
    st = wh.State("r", 0.0, "x")
    s = st.step("exact_e2e")
    s.attempts.append(wh.Attempt(0.0, "failed", finished=10.0, tracked_usd=1.00))
    s.attempts.append(wh.Attempt(20.0, "done", finished=30.0, tracked_usd=0.90))
    assert s.spent == pytest.approx(1.90) and st.spent == pytest.approx(1.90)
    with pytest.raises(wh.StopRule, match="exact_e2e"):
        wh.check_after(st, "exact_e2e")  # $1.90 > $1.80


def test_legacy_state_files_keep_their_spend(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"run_id": "r", "created": 0, "billing_since": "x",
                                "steps": {"exact_e2e": {"status": "failed", "tracked_usd": 1.0,
                                                        "billed_usd": None, "started": 1.0,
                                                        "finished": 2.0}}}))
    assert wh.load_state(path).spent == pytest.approx(1.0)


def test_killed_step_counts_its_heartbeat_and_budget_until_billing_reconciles(tmp_path):
    path = tmp_path / "state.json"
    st = wh.State("r", 0.0, "x")
    s = st.step("noise")
    s.status = "running"
    a = wh.Attempt(started=100.0, rate_usd_per_s=0.002, budget_usd=0.80, apps=["flora-app"])
    s.attempts.append(a)
    clock = iter([160.0, 400.0])
    hb = wh.Heartbeat(st, a, path, every=3600, clock=lambda: next(clock))
    hb.beat()
    assert a.tracked_usd == pytest.approx(0.12)  # 60 s x $0.002/s, saved to disk
    hb.beat()
    assert a.tracked_usd == pytest.approx(0.60)
    # the process is killed here; a new one loads the state file
    back = wh.load_state(path)
    assert back.steps["noise"].attempts[0].tracked_usd == pytest.approx(0.60)
    assert back.mark_killed() == ["flora-app"] and back.steps["noise"].status == "failed"
    assert back.spent == pytest.approx(0.80)  # its budget: billing has not caught up
    back.billing_baseline = {"flora-app": 0.0}
    assert wh.refresh_billing(back, now={"flora-app": 0.55}, clock=lambda: 400.0 + 3600)
    assert back.steps["noise"].billed_usd == pytest.approx(0.55)
    assert back.spent == pytest.approx(0.60)  # reconciled: max(heartbeat tracked, billed)


def test_failed_report_is_charged_its_budget():
    st = wh.State("r", 0.0, "x")
    st.step("rewards").attempts.append(wh.Attempt(0.0, "failed", finished=1.0,
                                                  budget_usd=0.9, report_error=True))
    assert st.spent == pytest.approx(0.9)


def test_billing_is_reread_for_every_app_and_shared_apps_count_once():
    st = wh.State("r", 0.0, "x", {"lab": 1.0, "exact-human": 0.0})
    st.step("preflight").attempts.append(wh.Attempt(0.0, "done", tracked_usd=0.01,
                                                    apps=["lab"]))
    st.step("balanced").attempts.append(wh.Attempt(0.0, "done", tracked_usd=0.5,
                                                   apps=["lab", "bal-app"]))
    st.step("exact_e2e").attempts.append(wh.Attempt(0.0, "done", tracked_usd=1.0,
                                                    apps=["exact-human"]))
    st.step("noise").attempts.append(wh.Attempt(0.0, "done", tracked_usd=0.3,
                                                apps=["exact-human"]))  # (b)'s app again
    wh.refresh_billing(st, now={"lab": 1.2, "bal-app": 2.0, "exact-human": 1.1})
    assert st.billed_total == pytest.approx(0.2 + 2.0 + 1.1)
    assert st.step("balanced").billed_usd == pytest.approx(2.0)  # only its own app
    assert st.step("exact_e2e").billed_usd is None  # shared with noise: global only
    assert st.spent == pytest.approx(0.01 + 2.0 + 1.0 + 0.3)
    wh.refresh_billing(st, now={"lab": 1.2, "bal-app": 2.0, "exact-human": 5.0})
    assert st.spent == pytest.approx(0.2 + 2.0 + 5.0)  # the billed total wins when larger


def _fake_run(monkeypatch, tmp_path, runner):
    monkeypatch.setattr(h0, "golden_manifest", _manifest)
    monkeypatch.setattr(wh, "billed_by_app", lambda since: {})
    monkeypatch.setattr(wh, "RUNNERS", {**{k: (lambda ctx: {"ok": True, "tracked_usd": 0.0})
                                           for k in wh.STEPS}, "preflight": runner})
    monkeypatch.setattr(wh, "REPORT", tmp_path / "report.md")
    monkeypatch.chdir(tmp_path)  # main() chdirs into --work; monkeypatch restores the cwd
    state = tmp_path / "state.json"
    return state, ["--yes-spend", "--only", "preflight", "--state", str(state), "--work",
                   str(tmp_path / "work")]


def test_main_retry_accumulates_and_resume_skips_done(monkeypatch, tmp_path):
    calls = []

    def flaky(ctx):
        calls.append(1)
        ok = len(calls) > 1
        return {"ok": ok, "tracked_usd": 0.01 if ok else 0.02, "apps": ["flora-balanced-lab"]}

    state, argv = _fake_run(monkeypatch, tmp_path, flaky)
    assert wh.main(argv) == 1  # failed attempt
    assert wh.main(argv) == 0  # retried
    s = wh.load_state(state).steps["preflight"]
    assert [a.status for a in s.attempts] == ["failed", "done"]
    assert s.spent == pytest.approx(0.03)  # both attempts count
    assert wh.main(argv) == 0 and len(calls) == 2  # done: never re-run


def test_main_charges_a_crashed_step(monkeypatch, tmp_path):
    def crash(ctx):
        ctx.note_app("flora-balanced-lab")
        raise RuntimeError("boom")

    state, argv = _fake_run(monkeypatch, tmp_path, crash)
    assert wh.main(argv) == 1
    s = wh.load_state(state).steps["preflight"]
    (a,) = s.attempts
    assert a.status == "failed" and a.report_error and a.apps == ["flora-balanced-lab"]
    assert a.budget_usd == wh.CAPS_USD["preflight"] and s.spent == pytest.approx(a.budget_usd)


def test_main_resumes_after_a_kill(monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr(wh.subprocess, "run", lambda args, **k: stopped.append(args))
    state, argv = _fake_run(monkeypatch, tmp_path,
                            lambda ctx: {"ok": True, "tracked_usd": 0.01, "apps": []})
    st = wh.State("r", 0.0, "x", {"x": 0.0})
    s = st.step("preflight")
    s.status = "running"  # a killed process left it here
    s.attempts.append(wh.Attempt(0.0, heartbeat=10.0, tracked_usd=0.004, budget_usd=0.05,
                                 apps=["flora-balanced-lab"]))
    wh.save_state(st, state)
    assert wh.main(argv) == 0
    s = wh.load_state(state).steps["preflight"]
    assert [a.status for a in s.attempts] == ["killed", "done"]
    assert s.spent == pytest.approx(0.05 + 0.01)  # the killed attempt at its budget
    assert ["modal", "app", "stop", "--yes", "flora-balanced-lab"] in stopped
