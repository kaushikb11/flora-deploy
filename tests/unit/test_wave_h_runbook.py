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
    """The golden durations the runbook prices: the selection's expected durations plus the
    caps' headroom (golden manifest v1, numbers only)."""
    return wh.offline_manifest(wh.DURATION_HEADROOM)


def test_six_staged_human_episodes_and_noise_is_the_oracle_set():
    from importlib import resources

    sel = json.loads((resources.files("flora_deploy.validate")
                      / "golden_selection_v1.json").read_text())
    human = {f"{e['pool']}/{e['dataset']}/{e['slug']}" for e in sel["episodes"]
             if e["pool"] == "human"}
    assert len(wh.EPISODES) == 6 == len(set(wh.EPISODES)) and set(wh.EPISODES) <= human
    assert {i.split("/")[0] for i in wh.EPISODES} == {"human"} and wh.EMBODIMENT == "human"
    assert set(wh.NOISE_EPISODES) <= set(wh.EPISODES)  # Fix-Cert H2: a subset (here: all)
    m = _manifest()
    assert wh.episodes_in(m) == wh.EPISODES and wh.missing_episodes(m) == ()
    short = {"episodes": [e for e in m["episodes"] if e["id"] != wh.EPISODES[-1]]}
    assert wh.missing_episodes(short) == (wh.EPISODES[-1],)
    assert len(wh.episodes_in(short)) == 5  # fewer staged: the runbook uses what exists


def test_projections_are_honest_about_the_caps():
    """H2: upper projections (upper quantiles, every cold start, scale-down tail and the client
    slack per app) are what the stop rule gates on; the dry run names the minimum honest cap."""
    m = _manifest()
    ps = {step: wh.project(step, m) for step in wh.STEPS}
    for p in ps.values():
        assert p.usd_upper >= p.usd
    for it in ps["exact_e2e"].items + ps["noise"].items:  # every app carries the client slack
        assert it["upper_seconds"] - it["seconds"] >= wh.CLIENT_SLACK_S
    # one app per stage: exact on one H200, balanced on one L40S + one H100 plus the two lab
    # containers, step 0 a CPU merge and one L40S check
    assert [i["gpu"] for i in ps["exact_e2e"].items] == ["H200"]
    assert [i["gpu"] for i in ps["noise"].items] == ["H200"]
    assert [i["gpu"] for i in ps["balanced"].items] == ["L40S", "H100!", "L40S", "H100"]
    assert [i["gpu"] for i in ps["tempseg"].items] == ["cpu", "L40S"]
    honest = wh.honest_caps(ps)
    for step in wh.STEPS:  # every step's upper projection fits its own cap
        assert ps[step].usd_upper <= wh.CAPS_USD[step] + 1e-9, step
        assert wh.CAPS_USD[step] == honest[step], step
    assert wh.GLOBAL_CAP_USD == 6.50  # the approved upper
    upper = sum(p.usd_upper for p in ps.values())
    assert upper > wh.GLOBAL_CAP_USD  # reported by the dry run, not hidden
    assert honest["global"] >= upper


def test_noise_repeats_the_oracle_human_job():
    assert wh.NOISE_EPISODES == wh.EPISODES and wh.NOISE_JOB == wh.JOB == "human"
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
    p.add("x", wh.Container("H200", 8, 96), 1000, upper_s=2500)  # nominal fits, upper not
    assert p.usd < wh.CAPS_USD["exact_e2e"] < p.usd_upper
    with pytest.raises(wh.StopRule, match="upper projection"):
        wh.check_before(st, "exact_e2e", p)  # H2: the upper figure is what is gated
    st.step("balanced").attempts.append(wh.Attempt(0.0, "done", tracked_usd=6.0))
    small = wh.Projection("exact_e2e")
    small.add("x", wh.Container("H200", 8, 96), 400, upper_s=600)  # fits the step only
    with pytest.raises(wh.StopRule, match="global"):
        wh.check_before(st, "exact_e2e", small)
    st.step("preflight").billed_usd = 0.10  # billed beats tracked when larger
    with pytest.raises(wh.StopRule, match="preflight"):
        wh.check_after(st, "preflight")
    st2 = wh.State("r", 0.0, "2026-10-21")
    c = wh.Container("H200", 8, 96)
    r = c.usd_per_s()
    # the deadline keeps the client slack: deadline + slack never spends more than the cap
    d = wh.gpu_deadline_s(st2, "noise", c)
    assert d == pytest.approx((wh.CAPS_USD["noise"] - wh.CLIENT_SLACK_S * r) / r)
    assert (d + wh.CLIENT_SLACK_S) * r == pytest.approx(wh.CAPS_USD["noise"])
    assert wh.gpu_deadline_s(st2, "noise", c, spent_in_step=wh.CAPS_USD["noise"]) == 0.0


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
    assert "plan balanced-only" in out and "NOT HONEST" not in out  # the default plan fits
    assert out.count(wh.DEFERRED_NOTE) == 3
    assert "episodes: 6 human" in out
    assert json.dumps(wh.CAPS_USD)  # caps are plain data
    assert wh.main(["--dry-run", "--plan", "full", "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert "NOT HONEST AT $6.50" in out and "at least $" in out  # H2: reported, not hidden
    assert wh.DEFERRED_NOTE not in out and not state.exists()


# --- plans: balanced-only (the default) vs full -------------------------------------------------


def test_balanced_only_runs_0_a_d_and_defers_b_e_f():
    assert wh.DEFAULT_PLAN == "balanced-only"
    assert wh.PLANS["balanced-only"] == ("tempseg", "preflight", "balanced")
    assert wh.deferred_steps("balanced-only") == ("exact_e2e", "certify", "noise")
    assert wh.PLANS["full"] == wh.STEPS and wh.deferred_steps("full") == ()


def test_balanced_only_global_cap_is_its_upper_sum_rounded_up_to_ten_cents():
    m = _manifest()
    ps = {s: wh.project(s, m) for s in wh.STEPS}
    up = sum(ps[s].usd_upper for s in wh.PLANS["balanced-only"])
    cap = wh.plan_global_cap("balanced-only", ps)
    assert up <= cap < up + 0.1 + 1e-9 and round(cap * 10, 6) == int(round(cap * 10))
    assert wh.plan_global_cap("full", ps) == wh.GLOBAL_CAP_USD
    st = wh.State("r", 0.0, "x")
    for s in wh.PLANS["balanced-only"]:  # every step fits its cap and the plan's global cap
        wh.check_before(st, s, ps[s], global_cap=cap)


def test_balanced_prices_a_cold_compile_startup_for_both_pools():
    from flora_deploy.planner.rates import RateTable

    assert wh.COLD_COMPILE_FACTOR == pytest.approx(318 / 229)
    t = RateTable.load()
    cold = wh.balanced_job_seconds(100.0)
    warm = wh.balanced_job_seconds(100.0, compile_factor=1.0)
    for kind in ("segment", "caption"):
        st = t.startup_for("balanced", kind)
        extra = st.startup_s * (wh.COLD_COMPILE_FACTOR - 1)
        assert cold[kind][0] - warm[kind][0] >= extra - 1e-6  # segment also idles longer
        assert cold[kind][1] - warm[kind][1] >= extra  # the upper scales it further
    items = wh.project("balanced", _manifest()).items
    assert sum("cold compile" in i["what"] for i in items) == 2


def test_main_never_runs_a_deferred_step(monkeypatch, tmp_path):
    ran = []
    state, argv = _fake_run(monkeypatch, tmp_path, None)
    monkeypatch.setattr(wh, "RUNNERS", {k: (lambda ctx, k=k: ran.append(k) or {
        "ok": True, "tracked_usd": 0.0}) for k in wh.STEPS})
    argv = argv[:1] + argv[3:]  # --yes-spend without --only
    assert wh.main(argv) == 0
    assert ran == ["tempseg", "preflight", "balanced"]
    st = wh.load_state(state)
    assert not {"exact_e2e", "certify", "noise"} & set(st.steps)
    with pytest.raises(SystemExit):
        wh.main([*argv, "--only", "exact_e2e"])  # deferred: refused by the parser
    text = (tmp_path / "report.md").read_text()
    assert text.count(wh.DEFERRED_NOTE) == 3


def test_balanced_step_does_not_read_the_oracle_state(monkeypatch, tmp_path):
    """(d) takes its 6 episodes from the golden selection, in order, not from (b)."""
    from flora_deploy.config import receipts, schema

    seen = {}
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **k: {})
    monkeypatch.setattr(wh, "_stage_job", lambda ctx, job, ids, **k: seen.setdefault(
        "ids", tuple(ids)) and (tmp_path / "c.yaml", "job"))
    monkeypatch.setattr(schema, "load_config", lambda p: object())
    monkeypatch.setattr(receipts, "derived_content", lambda c: {})
    monkeypatch.setattr(wh, "_run_job", lambda *a, **k: {"app": "a", "state": "failed",
                                                         "spend": 0.0, "job_id": "job"})

    class NoOracle(wh.State):
        def step(self, name):
            assert name not in ("exact_e2e", "noise", "certify"), name
            return super().step(name)

    wh.run_balanced(_Ctx(NoOracle("r", 0.0, "x"), tmp_path, []))
    assert seen["ids"] == wh.EPISODES
    short = {"episodes": [e for e in _manifest()["episodes"] if e["id"] != wh.EPISODES[0]]}
    with pytest.raises(RuntimeError, match="not staged"):
        wh.balanced_episodes(short)


def test_pin_instructions_name_the_exact_one_line_edit():
    pins = SCRIPTS.parent / wh.PINS_FILE
    src = pins.read_text()
    old, new = wh.pin_edit("e" * 64)
    assert src.count(old + "\n") == 1  # the one unpinned line: tempseg-merged's
    block = src[src.index('key="tempseg-merged"'):]
    assert block.index(old) < block.index("),")
    text = wh.pin_instructions({"content_sha256": "e" * 64, "receipt_sha256": "f" * 64})
    assert f"-{old}" in text and f"+{new}" in text and 'DERIVED_PINS["tempseg-merged"]' in text


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
        with pytest.raises(RuntimeError, match="not pinned"):
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


def test_generated_configs_carry_weights_derived_and_resolve_like_deploy(tmp_path, monkeypatch,
                                                                       tempseg_pinned):
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
    s.attempts.append(wh.Attempt(0.0, "failed", finished=10.0, tracked_usd=2.00))
    s.attempts.append(wh.Attempt(20.0, "done", finished=30.0, tracked_usd=1.90))
    assert s.spent == pytest.approx(3.90) and st.spent == pytest.approx(3.90)
    with pytest.raises(wh.StopRule, match="exact_e2e"):
        wh.check_after(st, "exact_e2e")  # $3.90 > $3.30


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
    st.step("noise").attempts.append(wh.Attempt(0.0, "failed", finished=1.0,
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


# --- review 10 L1 / L2 and the GPU-session prep -------------------------------------------------


def test_l1_a_step_that_fails_before_any_app_is_charged_nothing(monkeypatch, tmp_path):
    def refuse(ctx):
        raise RuntimeError("refused on CPU before any app")  # e.g. require_pin

    state, argv = _fake_run(monkeypatch, tmp_path, refuse)
    assert wh.main(argv) == 1
    s = wh.load_state(state).steps["preflight"]
    (a,) = s.attempts
    assert a.status == "failed" and not a.report_error and a.apps == []
    assert s.spent == 0.0 and wh.load_state(state).spent == 0.0


def test_report_never_writes_the_finalize_report():
    assert wh.REPORT != SCRIPTS.parent / "docs" / "wave-h-report.md"
    assert wh.REPORT.parent == SCRIPTS.parent / "reports" / "wave-h"
    src = (SCRIPTS / "wave_h.py").read_text()
    assert 'ROOT / "docs" / "wave-h-report.md"' not in src


EQUIV_OK = {"verify_merged": {"ok": True}, "b_gdn": {"vs_merged": "PASS", "vs_base": "PASS"},
            "windows": [{"inputs_identical": True, "ids_identical": True, "json_equal": True}],
            "total_s": 142.0}


class _Ctx(wh.Ctx):
    """A Ctx whose shell, receipt listing and Modal calls are recorded, not run."""

    def __init__(self, st, tmp_path, rows, equivalence=EQUIV_OK):
        super().__init__(st, _manifest(), tmp_path, state_path=tmp_path / "state.json")
        self.rows, self.cmds, self.equivalence = rows, [], equivalence

    def receipt_rows(self):
        return [dict(r) for r in self.rows]

    def sh(self, args, *, timeout=1800, check=True):
        self.cmds.append(args)
        if args[:2] == ["modal", "run"]:
            step = args[args.index("--step") + 1]
            out = Path(args[args.index("--out") + 1])
            out.write_text(json.dumps(self.equivalence if step == "equivalence"
                                      else {"ok": True}))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def _row(**kw):
    return {"key": "tempseg_merged", "content_sha256": "e" * 64, "receipt_sha256": "f" * 64,
            "version": 2, "content_sha256_pinned": None, "matches_pin": None, **kw}


def _tempseg_ctx(tmp_path, rows, **kw):
    st = wh.State("r", 0.0, "x")
    st.step("tempseg").attempts.append(wh.Attempt(0.0))  # the running attempt
    ctx = _Ctx(st, tmp_path, rows, **kw)
    ctx.step = "tempseg"
    return st, ctx


def test_step0_builds_checks_reports_the_digest_and_stops_for_the_pin(tmp_path):
    st, ctx = _tempseg_ctx(tmp_path, [_row()])
    res = wh.run_tempseg(ctx)
    assert res["needs_pin"] and not res["ok"] and res["content_sha256"] == "e" * 64
    runs = [c for c in ctx.cmds if c[:2] == ["modal", "run"]]
    assert [c[c.index("--step") + 1] for c in runs] == ["merge", "equivalence"]
    assert runs[0][2].endswith("convert_app.py::tempseg")
    assert ["modal", "app", "stop", "--yes", wh.TEMPSEG_APP] in ctx.cmds
    assert st.step("tempseg").current.apps == [wh.TEMPSEG_APP]  # noted before it spent
    assert "e" * 64 in wh.pin_instructions(res) and "DERIVED_PINS" in wh.pin_instructions(res)
    # re-run before pinning: never rebuilds what it already reported
    st.step("tempseg").result = res
    ctx.cmds.clear()
    again = wh.run_tempseg(ctx)
    assert again["needs_pin"] and again["tracked_usd"] == 0.0 and ctx.cmds == []
    # pinned and matching: done, nothing run
    ctx.rows = [_row(content_sha256_pinned="e" * 64, matches_pin=True)]
    done = wh.run_tempseg(ctx)
    assert done["ok"] and ctx.cmds == []


def test_step0_refuses_to_offer_a_pin_when_the_equivalence_check_fails(tmp_path):
    bad = {**EQUIV_OK, "windows": [{"inputs_identical": True, "ids_identical": False}]}
    _st, ctx = _tempseg_ctx(tmp_path, [_row()], equivalence=bad)
    res = wh.run_tempseg(ctx)
    assert not res["ok"] and not res.get("needs_pin") and "do not pin" in res["error"]


def test_steps_needing_tempseg_refuse_until_the_pin_matches(tmp_path):
    ctx = _Ctx(wh.State("r", 0.0, "x"), tmp_path, [_row()])
    with pytest.raises(RuntimeError, match="not pinned"):
        wh.require_tempseg_pin(ctx)
    ctx.rows = [_row(content_sha256_pinned="a" * 64, matches_pin=False)]
    with pytest.raises(RuntimeError, match="re-run step 0"):
        wh.require_tempseg_pin(ctx)
    ctx.rows = [_row(content_sha256_pinned="e" * 64, matches_pin=True)]
    assert wh.require_tempseg_pin(ctx)["matches_pin"] is True


def test_needs_pin_exits_4_and_prints_the_pin(monkeypatch, tmp_path, capsys):
    state, argv = _fake_run(monkeypatch, tmp_path, None)
    monkeypatch.setattr(wh, "RUNNERS", {**wh.RUNNERS, "tempseg": lambda ctx: {
        "ok": False, "needs_pin": True, "content_sha256": "e" * 64, "receipt_sha256": "f" * 64,
        "tracked_usd": 0.2, "apps": [wh.TEMPSEG_APP]}})
    argv[argv.index("preflight")] = "tempseg"
    assert wh.main(argv) == wh.NEEDS_PIN
    assert "e" * 64 in capsys.readouterr().out
    assert wh.load_state(state).steps["tempseg"].status == "needs_pin"


def test_l2_lab_is_skipped_when_the_deployed_balanced_job_failed(monkeypatch, tmp_path):
    from flora_deploy.config import receipts, schema

    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **k: {})
    monkeypatch.setattr(wh, "_stage_job", lambda ctx, *a, **k: (tmp_path / "c.yaml", "job"))
    monkeypatch.setattr(schema, "load_config", lambda p: object())
    monkeypatch.setattr(receipts, "derived_content", lambda c: {"tempseg_merged": "e" * 64})
    monkeypatch.setattr(wh, "_run_job", lambda *a, **k: {
        "app": "flora-bal", "state": "failed", "spend": 0.4, "job_id": "job"})
    res = wh.run_balanced(_Ctx(wh.State("r", 0.0, "x"), tmp_path, []))
    assert not res["ok"] and res["lab"].startswith("skipped: deployed balanced job ended")
    assert res["kinds"] == {} and "flora-balanced-lab" not in res["apps"]
    assert res["tracked_usd"] == pytest.approx(0.4)
    assert res["derived_content"] == {"tempseg_merged": "e" * 64}  # what finalize certifies


def test_job_record_carries_what_finalize_reads():
    from flora_deploy.jobs.containers import parse_utc

    rep = {"state": "done", "spend": {"total_usd": 0.5, "containers": [
        {"container_id": "c", "kind": "segment", "gpu": "L40S", "startup_s": 100.0,
         "busy_s": 50.0, "seconds": 500.0, "usd": 0.4, "started_utc": "2026-10-22T10:01:40Z"}]},
        "video_hours": {"done": 0.05, "planned": 0.05}, "usd_per_video_hour": 10.0}
    r = wh.job_record(rep, stopped_at=parse_utc("2026-10-22T10:06:00Z"))
    assert r["video_s"] == 180.0 and r["spend"] == 0.5 and r["state"] == "done"
    (c,) = r["containers"]
    assert c["seconds"] == 360.0 and c["kind"] == "segment" and c["busy_s"] == 50.0
    assert set(wh.record_paths("balanced")) == {"segment", "caption"}
    assert wh.record_paths("noise")["caption"].endswith("records/noise/human/caption.jsonl")
