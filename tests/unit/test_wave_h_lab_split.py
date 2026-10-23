"""The split Captioner lab (2026-10-23): recertify's d2' passed its 1131 s client deadline with
both Captioner engines in one call on the v4 engine (the lab compiles every engine cold: its
caches are its own, never the deployed workers'), so the certifying plans run the Captioner lab
as two calls, a baseline call and a plugin call on the Gate A inputs the baseline staged, each
its own container and deadline. Also: recertify runs d2' first (then d'), J1 publishes its
compile cache and J1w (forced cold) does not, and a retry reuses a saved baseline call."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.unit import test_wave_h_lab as TL

wh, lab, h0 = TL.wh, TL.lab, TL.h0
EPS = list(wh.EPISODES)
GATE_REL = "balanced-lab/wave-h-r/captioner-gate"
SHA = "5" * 64


@pytest.fixture()
def split_ctx(monkeypatch, tmp_path):
    """The lab fixture of ``test_wave_h_lab`` with the certifying plans' split calls and
    TempSeg's saved result in place (as recertify has it), so the Captioner alone runs."""
    monkeypatch.setattr(wh, "LAB_OUT", tmp_path / "out")
    monkeypatch.setattr(wh, "lab_calls", lambda kind, spec=None: wh.LAB_CALLS[kind])
    st = TL._done_state()
    ctx = wh.Ctx(st, TL._manifest(), tmp_path, state_path=tmp_path / "state.json", step="lab",
                 global_cap=10.0, caps={**wh.CAPS_USD, "lab": 2.85})
    recs = lambda ctx, job: [SimpleNamespace(  # noqa: E731
        episode_id=h0.episode_id(g), stage="segment",
        provenance=SimpleNamespace(config_hash=TL.SEG_HASH),
        segments=(SimpleNamespace(start_s=0.0, end_s=1.0, nv_metadata={}),))
        for g in wh.EPISODES]
    wh.LAB_OUT.mkdir(parents=True)
    wh.lab_result_path("tempseg").write_text(json.dumps(TL._saved("tempseg")))
    return ctx, recs


def _baseline(**kw):
    h = wh.lab_engine_hashes("captioner")
    res = {"run": "wave-h-r", "kind": "captioner", "arm": "baseline",
           "baseline": {"l0": [{"ids_equal": True, "grid_equal": True}],
                        "batcher": {"requests": 72}, "episodes": [], "gpu": "H100",
                        "episodes_requested": EPS},
           "engine_hashes": {"baseline": h["baseline"]},
           "compile_cache": {"baseline": {"warm": False}}, "segments_from": TL.BOUND,
           "gate_rel": GATE_REL, "gate_sha256": SHA, "container_s": 900.0,
           "usd_estimate": 1.2}
    res.update(kw)
    return res


def _plugin(**kw):
    h = wh.lab_engine_hashes("captioner")
    res = {"run": "wave-h-r", "kind": "captioner", "arm": "plugin", "gate_rel": GATE_REL,
           "gate_sha256": SHA, "baseline_engine_hash": h["baseline"],
           "engine_hashes": {"plugin": h["plugin"]},
           "compile_cache": {"plugin": {"warm": False}}, "dtbias": {"engine_build_s": 400.0},
           "gate_a": {"requests": 6, "max_abs_dlogit": 0.01}, "container_s": 500.0,
           "usd_estimate": 0.7}
    res.update(kw)
    return res


class _SplitLab(TL._Lab):
    """A fake ``balanced_lab`` whose results are per (kind, arm): a result that is an
    exception is raised by ``get`` (a client deadline, say)."""

    def __init__(self, results):
        super().__init__(results)
        self.calls = []

    def _spawn(self, kind, args):
        arm = args[6] if len(args) > 6 else "both"
        self.spawned.append((kind, args))
        res = self.results[(kind, arm)]
        call = TL._Call(res)
        if isinstance(res, BaseException):
            def boom(timeout, call=call, res=res):
                call.timeout = timeout
                raise res
            call.get = boom
        self.calls.append(call)
        return call


def test_the_captioner_runs_a_baseline_then_a_plugin_call_merged_into_one_result(split_ctx):
    ctx, recs = split_ctx
    fake = _SplitLab({("captioner", "baseline"): _baseline(),
                      ("captioner", "plugin"): _plugin()})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert res["ok"], res
    (k1, a1), (k2, a2) = fake.spawned
    assert (k1, k2) == ("captioner", "captioner")
    assert a1[6:] == ("baseline", "") and a2[6:] == ("plugin", GATE_REL)
    assert a1[2] == a2[2] == EPS and set(a1[4]) == set(EPS) and a1[5] == TL.BOUND
    # each call its own deadline, under the 1200 s container timeout
    assert [c.timeout for c in fake.calls] == [
        wh.LAB_CALL_DEADLINE_S[("captioner", "baseline")],
        wh.LAB_CALL_DEADLINE_S[("captioner", "plugin")]]
    assert all(c.timeout <= wh.LAB_TIMEOUT_S for c in fake.calls)
    assert fake.runs == [f"flora-wh-{ctx.st.run_id}-lab-a1"]  # one app, both calls
    merged = json.loads(wh.lab_result_path("captioner").read_text())
    assert merged["engine_hashes"] == wh.lab_engine_hashes("captioner")
    assert merged["gate_a"]["requests"] == 6 and merged["split"] is True
    assert set(merged["compile_cache"]) == {"baseline", "plugin"}
    assert merged["container_s"] == 1400.0 and merged["usd_estimate"] == pytest.approx(1.9)
    assert set(merged["calls"]) == {"baseline", "plugin"} and "error" not in merged
    # it is evidence, as a one-call result is
    assert wh.reusable_lab_result("captioner", "r", EPS, segments_from=TL.BOUND) is not None
    assert wh.lab_already_done(wh.LAB_OUT)[0]
    kind = res["kinds"]["captioner"]
    assert kind["evidence"] and set(kind["calls"]) == {"baseline", "plugin"}
    assert kind["requests"] == 72
    assert res["tracked_usd"] == pytest.approx(sum(c["usd"] for c in kind["calls"].values()),
                                               abs=1e-4)
    # the baseline call is kept for a retry
    assert wh.reusable_baseline_call("captioner", "r", EPS,
                                     segments_from=TL.BOUND)["gate_rel"] == GATE_REL


def test_a_retry_after_a_plugin_deadline_reuses_the_baseline_call(split_ctx):
    ctx, recs = split_ctx
    fake = _SplitLab({("captioner", "baseline"): _baseline(),
                      ("captioner", "plugin"): TimeoutError()})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert not res["ok"] and "plugin: client deadline 662 s" in res["error"]
    first = json.loads(wh.lab_result_path("captioner").read_text())
    assert "gate_a" not in first and first["error"].startswith("plugin: client deadline")
    # attempt 2: the plugin call alone, on the baseline call's staged Gate A inputs
    ctx.st.step("lab").attempts.append(wh.Attempt(1.0))
    fake2 = _SplitLab({("captioner", "plugin"): _plugin()})
    again = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake2)
    assert again["ok"], again
    assert [a[6:] for _, a in fake2.spawned] == [("plugin", GATE_REL)]
    assert fake2.runs == [f"flora-wh-{ctx.st.run_id}-lab-a2"]
    assert again["kinds"]["captioner"]["calls"]["baseline"]["usd"] == 0.0
    assert wh.lab_already_done(wh.LAB_OUT)[0]
    # the dry run prices that retry at the plugin call alone
    spec = replace(wh.SPECS[wh.RECERTIFY], lab_dir=wh.LAB_OUT)
    wh.baseline_call_path("captioner").write_text(json.dumps(_baseline()))
    assert wh.saved_lab_calls("r", spec) == ("captioner:baseline",)
    p = wh.project("lab", wh.offline_manifest(0.0, wh.CERT_GOLDEN), spec,
                   reuse=("tempseg", "captioner:baseline"))
    assert [i["usd_upper"] > 0 for i in p.items] == [False, False, True]
    assert "plugin call" in p.items[2]["what"]


def test_a_saved_baseline_call_for_other_segments_or_engines_is_not_reused(split_ctx):
    ctx, recs = split_ctx
    path = wh.baseline_call_path("captioner")
    for bad in (_baseline(segments_from={"job_id": "other", "segment_hash": "b" * 64}),
                _baseline(engine_hashes={"baseline": "0" * 64}),
                _baseline(gate_rel=""), _baseline(error="baseline: child exit 1"),
                _baseline(run="wave-h-other")):
        path.write_text(json.dumps(bad))
        assert wh.reusable_baseline_call("captioner", "r", EPS, segments_from=TL.BOUND) is None
    fake = _SplitLab({("captioner", "baseline"): _baseline(),
                      ("captioner", "plugin"): _plugin()})
    assert wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)["ok"]
    assert [a[6] for _, a in fake.spawned] == ["baseline", "plugin"]


def test_a_failed_baseline_call_starts_no_plugin_call(split_ctx):
    ctx, recs = split_ctx
    fake = _SplitLab({("captioner", "baseline"): TimeoutError()})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert not res["ok"] and [a[6] for _, a in fake.spawned] == ["baseline"]
    assert "client deadline 1200 s" in res["kinds"]["captioner"]["error"]
    assert wh.reusable_baseline_call("captioner", "r", EPS, segments_from=TL.BOUND) is None
    # a baseline that staged no Gate A inputs: no plugin call either
    fake2 = _SplitLab({("captioner", "baseline"): _baseline(gate_rel=None)})
    res2 = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake2)
    assert not res2["ok"] and len(fake2.spawned) == 1
    assert "staged no Gate A inputs" in res2["kinds"]["captioner"]["error"]


def test_a_plugin_call_on_other_gate_inputs_is_no_evidence(split_ctx):
    ctx, recs = split_ctx
    for plug, why in ((_plugin(gate_sha256="6" * 64), "gate_sha256"),
                      (_plugin(baseline_engine_hash="0" * 64), "another baseline engine")):
        fake = _SplitLab({("captioner", "baseline"): _baseline(),
                          ("captioner", "plugin"): plug})
        res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
        assert not res["ok"] and why in res["error"]
        merged = json.loads(wh.lab_result_path("captioner").read_text())
        assert "gate_a" not in merged
        assert wh.reusable_lab_result("captioner", "r", EPS, segments_from=TL.BOUND) is None
        wh.baseline_call_path("captioner").unlink()


def test_the_lab_stop_rule_uses_the_first_call_it_would_start(split_ctx):
    ctx, recs = split_ctx
    ctx.global_cap = 1.2  # ~850 s of H100: under the baseline call's expected 960 s
    fake = _SplitLab({})
    with pytest.raises(wh.StopRule, match="captioner baseline call"):
        wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert fake.spawned == [] and fake.runs == []
    # with the baseline call saved, the plugin call (530 s) is the first and fits
    wh.baseline_call_path("captioner").write_text(json.dumps(_baseline()))
    fake2 = _SplitLab({("captioner", "plugin"): _plugin()})
    assert wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake2)["ok"]


# --- the projection and the plan -----------------------------------------------------------------


def test_recertify_prices_the_captioner_lab_as_two_calls_and_runs_it_first():
    m = wh.offline_manifest(wh.DURATION_HEADROOM, wh.CERT_GOLDEN)
    p = wh.project("lab", m, wh.SPECS[wh.RECERTIFY], reuse=("tempseg",))
    assert [i["gpu"] for i in p.items] == ["L40S", "H100", "H100"]
    assert ["baseline call" in p.items[1]["what"], "plugin call" in p.items[2]["what"]] == [
        True, True]
    rate = wh.LAB["captioner"].usd_per_s()
    assert p.usd_upper == pytest.approx(rate * sum(
        wh.LAB_CALL_DEADLINE_S[("captioner", a)] + wh.LAB_SLACK_S
        for a in ("baseline", "plugin")), abs=1e-3)
    # every call cold: the v1 one-call deadline (1131 s) no longer carries the v4 Captioner
    assert wh.LAB_CALL_EXPECTED_S[("captioner", "baseline")] > wh.LAB_DEADLINE_S["captioner"] - 200
    assert wh.LAB_CALL_DEADLINE_S[("captioner", "baseline")] == wh.LAB_TIMEOUT_S
    assert wh.STEP_HARD_TIMEOUT_S["lab"] >= 3 * wh.LAB_TIMEOUT_S
    # the v1 plans keep their one call
    v1 = wh.project("lab", wh.offline_manifest(wh.DURATION_HEADROOM))
    assert [i["gpu"] for i in v1.items] == ["L40S", "H100"]
    steps = wh.PLANS[wh.RECERTIFY]
    assert steps.index("noise_all") < steps.index("lab") < steps.index("balanced") \
        < steps.index("caption_iso") \
        < steps.index("certify")


def test_the_dry_run_says_when_the_steps_left_do_not_fit(monkeypatch):
    m = wh.offline_manifest(wh.DURATION_HEADROOM, wh.CERT_GOLDEN)
    projs = wh.plan_projections(wh.RECERTIFY, m)
    st = wh.State("r", 0.0, "x", plan=wh.RECERTIFY)
    st.step("noise_all").status = "done"
    monkeypatch.setattr(wh, "RECERT_GLOBAL_CAP_USD", 8.0)  # under the $8.86 left
    lines: list[str] = []
    wh.print_plan(st, projs, lines.append, manifest=m, plan=wh.RECERTIFY)
    text = "\n".join(lines)
    left = sum(projs[s].usd_upper for s in ("balanced", "lab", "caption_iso"))
    assert f"DOES NOT FIT: spent $0.00 + the upper of the steps not done ${left:.2f}" in text
    assert "captioner baseline 1200 s + plugin 662 s" in text
    st.cap_override, st.cap_override_plan = 13.0, wh.RECERTIFY
    lines.clear()
    wh.print_plan(st, projs, lines.append, manifest=m, plan=wh.RECERTIFY)
    assert "fits: spent $0.00" in "\n".join(lines)


# --- J1 publishes its compile cache, J1w does not -------------------------------------------------


def test_recertify_j1_publishes_and_its_forced_cold_witness_does_not(tmp_path, monkeypatch):
    import flora_deploy.config.receipts as R
    import flora_deploy.config.schema as S
    import flora_deploy.validate.certkey as K
    from flora_deploy.modal_app import balanced_app as ba

    monkeypatch.setattr(K, "GENERATOR_VERSIONS", ("4",))
    spec = replace(wh.SPECS[wh.RECERTIFY], state=tmp_path / "s.json", work=tmp_path,
                   records=tmp_path / "r", report=tmp_path / "rb.md")
    calls = []

    def stage_job(ctx, job, ids, **kw):
        cfg = tmp_path / f"{job}.yaml"
        cfg.write_text(f"project: wave-h-r-{job}\npreset: balanced\npipeline:\n"
                       "  stages: [segment, caption]\n")
        return cfg, f"wave-h-r-{job}"

    def run_job(ctx, step, cfg, job_id, **kw):
        calls.append((job_id, ctx.env.get(ba.COMPILE_CACHE_ENV), ctx.env.get(ba.PUBLISH_ENV)))
        return {"app": "a", "job_id": job_id, "state": "done", "spend": 0.25, "cfg": str(cfg)}

    monkeypatch.setattr(wh, "_stage_job", stage_job)
    monkeypatch.setattr(wh, "_index", lambda *a, **k: None)
    monkeypatch.setattr(wh, "_run_job", run_job)
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **kw: {})
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")
    monkeypatch.setattr(S, "load_config", lambda p, **kw: None)
    monkeypatch.setattr(R, "derived_content", lambda cfg: {})
    monkeypatch.setattr(wh, "balanced_episodes", lambda m, spec, **kw: spec.episodes)
    monkeypatch.delenv(ba.COMPILE_CACHE_ENV, raising=False)
    monkeypatch.delenv(ba.PUBLISH_ENV, raising=False)
    ctx = wh.Ctx(wh.State(run_id="r", created=0.0, billing_since="x"),
                 wh.offline_manifest(0.0, wh.CERT_GOLDEN), tmp_path, spec=spec,
                 caps=dict(wh.CAPS_USD))
    assert wh.run_balanced(ctx)["ok"]
    # J1: the cache on and publishing (the default); J1w: forced cold, so it never publishes
    assert calls == [("wave-h-r-rc-balanced-human", None, None),
                     ("wave-h-r-rc-cold-human-a1", "off", None)]
    assert ba.COMPILE_CACHE_ENV not in os.environ
    assert wh.COMPILE_CACHE_ENV == ba.COMPILE_CACHE_ENV
    off = ba.prepare_cache("caption", SimpleNamespace(), None, {ba.COMPILE_CACHE_ENV: "off"},
                           mount=tmp_path / "m", local_root=tmp_path / "l", image_fp="f",
                           log=lambda s: None)
    assert off.publish_enabled is False


# --- balanced_lab: the split arms -----------------------------------------------------------------


@pytest.fixture()
def lab_container(monkeypatch, tmp_path):
    """``balanced_lab.run_kind`` with its children, Volumes and paths faked under tmp_path:
    a child returns per-request logits and, as the baseline, pickles its Gate A requests."""
    from flora_deploy.engines.offline_llm import EngineSpec
    from flora_deploy.modal_app import warm

    built = []

    def child(args, env):
        spec = EngineSpec.for_gpu(args["kind"], "/m", args["gpu"], dtbias_fp32=args["plugin"],
                                  logprobs_mode="raw_logits", max_logprobs=20)
        built.append(args)
        gate = [{"ids": [1, 2], "logits": [{"1": 0.5 + args["plugin"]}]}]
        if not args["plugin"]:
            with open(args["gate_requests"], "wb") as f:
                pickle.dump(["req-1", "req-2"], f)
            return {"kind": args["kind"], "plugin": False, "gate_a": gate, "episodes": [],
                    "engine_hash": lab.engine_hash(spec), "l0": [{"ids_equal": True}]}
        with open(args["gate_requests"], "rb") as f:
            assert pickle.load(f) == ["req-1", "req-2"]  # the baseline's requests
        return {"kind": args["kind"], "plugin": True, "gate_a": gate,
                "engine_hash": lab.engine_hash(spec)}

    commits = []
    monkeypatch.setattr(lab, "_run_child", child)
    monkeypatch.setattr(lab, "cache", SimpleNamespace(commit=lambda: commits.append("cache")))
    monkeypatch.setattr(lab, "golden", SimpleNamespace(commit=lambda: commits.append("golden"),
                                                       reload=lambda: None))
    monkeypatch.setattr(lab, "cache_env", lambda kind, plugin: {})
    monkeypatch.setattr(lab, "compile_cache_state", lambda kind, plugin: {"warm": False})
    monkeypatch.setattr(lab, "gate_a_compare",
                        lambda a, b: {"requests": len(a), "base": a, "patched": b})
    monkeypatch.setattr(warm, "start_prefetch",
                        lambda *a, **k: SimpleNamespace(wait=lambda t: None))
    from flora_deploy.engines.patches import dt_bias_fp32

    monkeypatch.setattr(dt_bias_fp32, "install_entry_point", lambda d: {})
    monkeypatch.setattr(lab, "Path", lambda p: tmp_path / str(p).strip("/"))
    return built, commits, tmp_path


def test_run_kind_baseline_arm_stages_gate_inputs_and_plugin_arm_reads_them(lab_container):
    built, commits, root = lab_container
    segs = {g: [{"start_s": 0.0, "end_s": 1.0, "metadata": {}}] for g in EPS}
    base = lab.run_kind("captioner", "wave-h-r", "", EPS, True, segs, TL.BOUND,
                        arm="baseline")
    assert "error" not in base and base["arm"] == "baseline"
    assert [a["plugin"] for a in built] == [False]  # no plugin engine in this call
    assert base["gate_rel"] == lab.gate_dir_rel("wave-h-r", "captioner") == GATE_REL
    staged = root / str(lab.GOLDEN_MOUNT).strip("/") / GATE_REL
    want = hashlib.sha256((staged / lab.GATE_REQUESTS).read_bytes()).hexdigest()
    assert base["gate_sha256"] == want and "golden" in commits
    assert "gate_a" not in base["baseline"]  # the logits are staged, not returned
    plug = lab.run_kind("captioner", "wave-h-r", "", EPS, True, arm="plugin",
                        gate_rel=base["gate_rel"])
    assert "error" not in plug and [a["plugin"] for a in built] == [False, True]
    assert built[1]["gate_only"] is True
    assert plug["gate_sha256"] == want
    assert plug["gate_a"]["base"][0]["logits"] == [{"1": 0.5}]  # the staged baseline logits
    assert plug["gate_a"]["patched"][0]["logits"] == [{"1": 1.5}]
    merged = wh.merge_lab_calls(base, plug)
    assert "error" not in merged and merged["engine_hashes"] == wh.lab_engine_hashes("captioner")
    assert merged["gate_a"]["requests"] == 1


def test_run_kind_plugin_arm_without_staged_inputs_is_no_evidence(lab_container):
    built, _, _ = lab_container
    none = lab.run_kind("captioner", "wave-h-r", arm="plugin")
    assert none["no_evidence"] and "no gate_rel" in none["error"]
    gone = lab.run_kind("captioner", "wave-h-r", arm="plugin", gate_rel="balanced-lab/x/nope")
    assert gone["no_evidence"] and "Gate A inputs" in gone["error"] and built == []
    bad = lab.run_kind("captioner", "wave-h-r", arm="sideways")
    assert bad["no_evidence"] and "unknown arm" in bad["error"]


def test_run_kind_both_is_the_one_call_it_always_was(lab_container):
    built, _, _ = lab_container
    res = lab.run_kind("tempseg", "wave-h-r", gate_a=True)
    assert [a["plugin"] for a in built] == [False, True] and "arm" not in res
    assert res["gate_a"]["requests"] == 1 and "gate_rel" not in res


# --- the recertify state as 2026-10-23 left it ----------------------------------------------------


def test_the_failed_d2_prime_state_retries_the_lab_as_attempt_2_before_d_prime():
    """f2 done, d2' attempt 1 failed at its 1131 s client deadline ($1.60, no evidence), d' g e'
    pending: d2' runs next as attempt 2 (its own app), then d', and e' stays after g."""
    m = wh.offline_manifest(wh.DURATION_HEADROOM, wh.CERT_GOLDEN)
    projs = wh.plan_projections(wh.RECERTIFY, m)
    st = wh.State("10221548f1b8", 0.0, "x", plan=wh.RECERTIFY)
    st.cap_override, st.cap_override_plan = 16.40, wh.RECERTIFY  # offline: +10% durations
    f2 = st.step("noise_all")
    f2.status = "done"
    f2.attempts.append(wh.Attempt(1.0, "done", tracked_usd=5.87, finished=2.0))
    lab_s = st.step("lab")
    lab_s.status = "failed"
    lab_s.attempts.append(wh.Attempt(3.0, "failed", tracked_usd=1.6043, finished=4.0,
                                     apps=["flora-wh-10221548f1b8-rc-lab-a1"]))
    lab_s.result = {"ok": False, "error": "captioner: client deadline 1131 s"}
    d = wh.next_decision(st, wh.RECERTIFY, projs)
    assert d.step == "lab" and d.action == "run", d.message
    lab_s.attempts.append(wh.Attempt(5.0))
    ctx = wh.Ctx(st, m, wh.ROOT, spec=wh.SPECS[wh.RECERTIFY], step="lab")
    assert ctx.app_name() == "flora-wh-10221548f1b8-rc-lab-a2"
    lab_s.status = "done"
    nxt = wh.next_decision(st, wh.RECERTIFY, projs)
    assert nxt.step == "balanced" and nxt.action == "run", nxt.message
    st.step("balanced").status = "done"
    assert wh.next_decision(st, wh.RECERTIFY, projs).step == "caption_iso"
    # its per-attempt cap is the split lab's whole upper: attempt 1's spend is not taken off it
    assert wh.plan_caps(wh.RECERTIFY, projs)["lab"] >= projs["lab"].usd_upper
