"""(d2) the balanced lab against (d)'s finished job (``scripts/wave_h.py`` + ``balanced_lab``):
the golden path fix behind the 2026-10-22 zero-request run, fail-loud results, preconditions
before any container, the two-cold-compile deadline, ``--only lab`` and cold startups that are
not scaled twice. Fakes only; nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
from flora_deploy.modal_app import balanced_lab as lab  # noqa: E402


def _manifest(ids=wh.EPISODES) -> dict:
    eps = []
    for i in ids:
        pool, dataset, slug = i.split("/")
        eps.append({"id": i, "pool": pool, "dataset": dataset, "slug": slug,
                    "path": f"golden/v1/{i}", "oracle_eligible": True, "profile": "egodex",
                    "duration_s": 10.0, "files": {"main": {"file": "main.mp4"}}})
    return {"episodes": eps}


def _segments(ids=wh.EPISODES) -> dict:
    return {g: [{"start_s": 0.0, "end_s": 1.0, "metadata": {}}] for g in ids}


# --- root cause: the golden root ----------------------------------------------------------------


def test_lab_inputs_open_the_files_where_the_manifest_puts_them():
    inputs = lab.episode_inputs(_manifest(), list(wh.EPISODES))
    assert [e["id"] for e in inputs] == list(wh.EPISODES)  # exactly (d)'s, in order
    first = inputs[0]["videos"]["main"]
    assert first == f"/golden/golden/v1/{wh.EPISODES[0]}/main.mp4"
    assert lab.volume_paths(inputs)[0] == f"golden/v1/{wh.EPISODES[0]}/main.mp4"
    # the 2026-10-22 root (<mount>/golden/v1 + golden/v1/...) doubles the prefix
    from flora_deploy.stages.reference import golden_episode_inputs

    old = golden_episode_inputs(_manifest(), list(wh.EPISODES), lab.GOLDEN_MOUNT / "golden/v1")
    assert lab.volume_paths(old)[0].startswith("golden/v1/golden/v1/")
    with pytest.raises(ValueError):  # a path outside the mount is refused
        lab.volume_paths([{"videos": {"main": "/elsewhere/x.mp4"}}])


def test_zero_requests_is_no_evidence():
    assert lab.evidence_problems(["a"], [{"id": "a"}], 3) == []
    assert lab.evidence_problems(["a"], [], 0) == ["no episode ran (of 1 requested)",
                                                    "the stage sent zero requests to the engine"]


# --- run_kind: fail loudly, commit the compile cache per engine ---------------------------------


class _Vol:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        pass


@pytest.fixture()
def fake_lab(monkeypatch, tmp_path):
    """``run_kind`` with fake children, Volumes, prefetch and cache dirs."""
    from flora_deploy.modal_app import warm

    calls: list[dict] = []
    vols = {"cache": _Vol(), "golden": _Vol()}
    monkeypatch.setattr(lab, "cache", vols["cache"])
    monkeypatch.setattr(lab, "golden", vols["golden"])
    monkeypatch.setattr(lab, "GOLDEN_MOUNT", tmp_path / "golden")
    monkeypatch.setattr(lab, "cache_env", lambda kind, plugin: {
        "VLLM_CACHE_ROOT": str(tmp_path / f"c-{kind}-{plugin}")})
    monkeypatch.setattr(warm, "start_prefetch", lambda *a, **k: SimpleNamespace(
        wait=lambda t: None))
    from flora_deploy.engines.patches import dt_bias_fp32

    monkeypatch.setattr(dt_bias_fp32, "install_entry_point", lambda p: {})
    box: dict = {}

    def child(args, env):
        calls.append(args)
        return box["plugin" if args["plugin"] else "baseline"]

    monkeypatch.setattr(lab, "_run_child", child)
    return box, calls, vols


def test_a_baseline_with_zero_requests_errors_and_builds_no_plugin_engine(fake_lab, tmp_path):
    box, calls, vols = fake_lab
    box["baseline"] = {"error": "no episode ran (of 6 requested)", "no_evidence": True,
                       "episodes": [{"id": "x", "ok": False, "error": "FileNotFoundError"}]}
    res = lab.run_kind("tempseg", "r", episodes=list(wh.EPISODES))
    assert res["error"].startswith("baseline: no episode ran") and res["no_evidence"]
    assert len(calls) == 1 and "dtbias" not in res and "gate_a" not in res
    assert "segments_rel" not in res  # no empty segments file handed on
    assert not (tmp_path / "golden").exists() and vols["cache"].commits == 1


def test_run_kind_passes_inline_segments_and_commits_after_each_engine(fake_lab, tmp_path):
    box, calls, vols = fake_lab
    gate = [{"ids": [1, 2], "logits": [{"1": 0.5}, {"2": 0.25}], "key": "k", "error": None}]
    box["baseline"] = {"episodes": [{"id": "x", "ok": True, "rows": [{"start_s": 0}]}],
                       "gate_a": gate, "l0": [{"ids_equal": True}]}
    box["plugin"] = {"gate_a": gate}
    seg = _segments()
    res = lab.run_kind("captioner", "r", episodes=list(wh.EPISODES), segments=seg)
    assert [c["plugin"] for c in calls] == [False, True]
    assert all(c["segments"] == seg for c in calls)
    assert res["gate_a"]["requests"] == 1 and "error" not in res
    assert vols["cache"].commits == 2  # the baseline compile survives a plugin deadline
    assert res["compile_cache"]["baseline"] == {"hashes": [], "complete": [], "warm": False}


def test_compile_cache_state_reports_finished_entries(tmp_path):
    base = tmp_path / "balanced" / "captioner-dtbias" / "vllm" / "torch_compile_cache"
    (base / "torch_aot_compile").mkdir(parents=True)
    (base / "aaaa" / "rank_0_0" / "backbone").mkdir(parents=True)
    (base / "bbbb" / "rank_0_0" / "backbone").mkdir(parents=True)
    (base / "bbbb" / "rank_0_0" / "backbone" / "vllm_compile_cache.py").write_text("")
    st = lab.compile_cache_state("captioner", True, root=tmp_path)
    assert st == {"hashes": ["aaaa", "bbbb"], "complete": ["bbbb"], "warm": True}
    assert lab.compile_cache_state("tempseg", False, root=tmp_path)["warm"] is False


# --- preconditions, before any container --------------------------------------------------------


def test_preconditions_pass_for_d_s_episodes_files_and_segments():
    seen = []
    ok = wh.lab_preconditions(_manifest(), list(wh.EPISODES), _segments(),
                              missing=lambda paths: seen.extend(paths) or [])
    assert ok == [] and len(seen) == 6 and all(p.startswith("golden/v1/human/") for p in seen)


def test_preconditions_name_every_gap():
    eps = list(wh.EPISODES)
    gone = wh.lab_preconditions(_manifest(), eps, _segments(), missing=lambda p: list(p))
    assert any("6 of 6 camera files not on flora-golden" in w for w in gone)
    short = wh.lab_preconditions(_manifest(), eps, _segments(eps[:4]), missing=lambda p: [])
    assert any("cover 4 of 6 episodes" in w for w in short)
    unstaged = wh.lab_preconditions(_manifest(eps[1:]), eps, _segments(), missing=lambda p: [])
    assert any("not in the golden manifest" in w for w in unstaged)
    other = wh.lab_preconditions(_manifest(), eps[:2], _segments(), missing=lambda p: [])
    assert any("differ from (d)'s 6" in w for w in other)
    assert wh.lab_preconditions(_manifest(), [], {}, missing=lambda p: []) == [
        "no episodes: (d) recorded none"]


def test_deployed_segment_records_map_back_to_golden_ids():
    def rec(eid, stage="segment"):
        segs = (SimpleNamespace(start_s=0.0, end_s=2.0, nv_metadata={"m": 1}),)
        return SimpleNamespace(episode_id=eid, stage=stage, segments=segs)

    out = wh.deployed_segments([rec(h0.episode_id(g)) for g in wh.EPISODES]
                               + [rec("other-ep"), rec(h0.episode_id(wh.EPISODES[0]), "caption")],
                               wh.EPISODES)
    assert list(out) == list(wh.EPISODES)
    assert out[wh.EPISODES[0]] == [{"start_s": 0.0, "end_s": 2.0, "metadata": {"m": 1}}]


# --- run_lab ------------------------------------------------------------------------------------


def _done_state() -> wh.State:
    st = wh.State("r", 0.0, "x")
    b = st.step("balanced")
    b.status = "failed"  # the 2026-10-22 state: the job finished, the inline lab failed
    b.result = {"episodes": list(wh.EPISODES), "kinds": {},
                "jobs": {wh.JOB: {"state": "done", "job_id": "job-d", "records": {}}}}
    st.step("lab").attempts.append(wh.Attempt(0.0))
    return st


class _Call:
    def __init__(self, res):
        self.res = res

    def get(self, timeout):
        self.timeout = timeout
        return self.res

    def cancel(self):
        pass


class _Lab:
    """A fake ``balanced_lab`` module: records what was spawned."""

    def __init__(self, results):
        self.results, self.spawned = results, []
        self.app = SimpleNamespace(run=self._run)
        self.run_tempseg = SimpleNamespace(spawn=lambda *a: self._spawn("tempseg", a))
        self.run_captioner = SimpleNamespace(spawn=lambda *a: self._spawn("captioner", a))
        self.episode_inputs, self.volume_paths = lab.episode_inputs, lab.volume_paths

    def _run(self):
        import contextlib

        return contextlib.nullcontext()

    def _spawn(self, kind, args):
        self.spawned.append((kind, args))
        return _Call(self.results[kind])


@pytest.fixture()
def lab_ctx(monkeypatch, tmp_path):
    monkeypatch.setattr(wh, "LAB_OUT", tmp_path / "out")
    st = _done_state()
    ctx = wh.Ctx(st, _manifest(), tmp_path, state_path=tmp_path / "state.json", step="lab",
                 global_cap=10.0)
    recs = lambda ctx, job: [SimpleNamespace(  # noqa: E731
        episode_id=h0.episode_id(g), stage="segment",
        segments=(SimpleNamespace(start_s=0.0, end_s=1.0, nv_metadata={}),))
        for g in wh.EPISODES]
    return ctx, recs


def _good(kind):
    return {"baseline": {"l0": [{"ids_equal": True, "grid_equal": True}],
                         "batcher": {"requests": 12}, "episodes": []},
            "gate_a": {"requests": 6, "max_abs_dlogit": 0.01}, "usd_estimate": 0.4}


def test_lab_precondition_failure_starts_nothing_and_charges_zero(lab_ctx):
    ctx, recs = lab_ctx
    fake = _Lab({})
    res = wh.run_lab(ctx, missing=lambda p: list(p), records=recs, lab_module=fake)
    assert not res["ok"] and res["tracked_usd"] == 0.0 and res["apps"] == []
    assert "camera files not on flora-golden" in res["error"] and wh.NO_EVIDENCE in res["note"]
    assert fake.spawned == [] and ctx.st.step("lab").current.apps == []
    bad = wh.run_lab(ctx, missing=lambda p: [], records=lambda c, j: 1 / 0, lab_module=fake)
    assert "reading (d)'s segment records" in bad["error"] and bad["tracked_usd"] == 0.0


def test_lab_refuses_without_a_finished_deployed_job(lab_ctx):
    ctx, recs = lab_ctx
    ctx.st.step("balanced").result["jobs"][wh.JOB]["state"] = "failed"
    with pytest.raises(RuntimeError, match="finished deployed job"):
        wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=_Lab({}))


def test_lab_zero_requests_fails_loudly_and_skips_the_captioner(lab_ctx):
    ctx, recs = lab_ctx
    zero = {"baseline": {"l0": [], "batcher": {"requests": 0},
                         "error": "no episode ran (of 6 requested)"},
            "error": "baseline: no episode ran (of 6 requested)", "no_evidence": True}
    fake = _Lab({"tempseg": zero})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert not res["ok"] and [k for k, _ in fake.spawned] == ["tempseg"]
    assert res["kinds"]["tempseg"]["evidence"] is False
    assert res["kinds"]["captioner"]["usd"] == 0.0 and "zero requests" in res[
        "kinds"]["captioner"]["skipped"]
    assert res["tracked_usd"] == pytest.approx(res["kinds"]["tempseg"]["usd"], abs=1e-4)
    assert res["apps"] == ["flora-balanced-lab"]
    # an "ok"-looking result without L0 / Gate A is no evidence either (the 2026-10-22 shape)
    silent = {"baseline": {"l0": [], "batcher": {"requests": 0}},
              "gate_a": {"requests": 0, "positions": 0}}
    res2 = wh.run_lab(ctx, missing=lambda p: [], records=recs,
                      lab_module=_Lab({"tempseg": silent}))
    assert not res2["ok"] and res2["kinds"]["tempseg"]["error"] == wh.NO_EVIDENCE


def test_lab_runs_d_s_episodes_with_d_s_segments_and_the_cold_deadlines(lab_ctx):
    ctx, recs = lab_ctx
    fake = _Lab({"tempseg": _good("tempseg"), "captioner": _good("captioner")})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert res["ok"] and res["segments"] == {"source": "deployed job records", "episodes": 6,
                                             "rows": 6}
    (k1, a1), (k2, a2) = fake.spawned
    assert (k1, k2) == ("tempseg", "captioner")
    assert a1[2] == a2[2] == list(wh.EPISODES)
    assert a1[4] is None and set(a2[4]) == set(wh.EPISODES)  # captioner: (d)'s segments
    assert json.loads((wh.LAB_OUT / "balanced-captioner.json").read_text())["gate_a"]
    assert ctx.st.step("lab").current.apps == ["flora-balanced-lab"]


def test_lab_is_not_started_when_the_budget_cannot_carry_it(lab_ctx):
    ctx, recs = lab_ctx
    ctx.global_cap = 0.2  # ~250 s of L40S: under TempSeg's expected two compiles
    fake = _Lab({})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert fake.spawned == [] and res["tracked_usd"] == 0.0 and not res["ok"]
    assert all("budget left" in v["skipped"] for v in res["kinds"].values())


# --- argv size, reuse of saved results, per-kind persistence ------------------------------------


def test_child_args_go_through_a_file_not_argv(tmp_path):
    import os
    import subprocess

    big = {"out": str(tmp_path / "a.json"), "kind": "captioner",
           "segments": {f"ep{i}": [{"start_s": 0.0, "end_s": 1.0, "metadata": {"x": "y" * 900}}
                                   for _ in range(40)] for i in range(40)}}
    blob = json.dumps(big)
    limit = os.sysconf("SC_ARG_MAX") if hasattr(os, "sysconf") else 2 ** 21
    assert len(blob) > max(limit, 131072)  # inline, this one argv string could not exec
    argv = lab.child_argv(big)
    assert sum(len(a) for a in argv) < 4096 and blob not in argv
    assert json.loads(Path(argv[-1]).read_text()) == big
    # the child's entry reads it back from the file (child() swapped for an echo)
    code = argv[2].replace("from flora_deploy.modal_app.balanced_lab import child; ", "").replace(
        "child(", "print(len(").replace(")))", "))))")
    p = subprocess.run([sys.executable, "-c", code, argv[3]], capture_output=True, text=True)
    assert p.returncode == 0 and int(p.stdout) == len(big)


def _saved(kind, **kw):
    res = {"run": "wave-h-r", "kind": kind, **_good(kind)}
    res["baseline"] = {**res["baseline"], "episodes_requested": list(wh.EPISODES)}
    res.update(kw)
    return res


def test_reusable_lab_result_needs_evidence_and_the_same_run(tmp_path):
    p = tmp_path / "balanced-tempseg.json"

    def check(res, run="r", eps=wh.EPISODES):
        p.write_text(json.dumps(res))
        return wh.reusable_lab_result("tempseg", run, eps, path=p) is not None

    assert check(_saved("tempseg"))
    assert not check(_saved("tempseg"), run="other")
    assert not check(_saved("tempseg"), eps=wh.EPISODES[:5])
    assert not check(_saved("captioner"))  # another kind's file
    assert not check(_saved("tempseg", error="client deadline 1131 s"))
    assert not check(_saved("tempseg", gate_a={"requests": 0}))
    no_l0 = _saved("tempseg")
    no_l0["baseline"]["l0"] = []
    assert not check(no_l0)
    assert not check(_saved("tempseg", dtbias={"error": "child exit 1"}))
    p.write_text("{not json")
    assert wh.reusable_lab_result("tempseg", "r", wh.EPISODES, path=p) is None


def test_lab_reuses_a_good_saved_kind_and_runs_only_the_rest(lab_ctx):
    ctx, recs = lab_ctx
    wh.LAB_OUT.mkdir(parents=True)
    wh.lab_result_path("tempseg").write_text(json.dumps(_saved("tempseg")))
    fake = _Lab({"captioner": _good("captioner")})
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert res["ok"] and [k for k, _ in fake.spawned] == ["captioner"]
    assert res["kinds"]["tempseg"]["reused"] and res["kinds"]["tempseg"]["usd"] == 0.0
    assert res["kinds"]["tempseg"]["l0"]  # its evidence is carried into the step result
    # everything saved: nothing starts, no app is noted
    wh.lab_result_path("captioner").write_text(json.dumps(_saved("captioner")))
    ctx.st.step("lab").attempts.append(wh.Attempt(1.0))
    fake2 = _Lab({})
    again = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake2)
    assert again["ok"] and fake2.spawned == [] and again["apps"] == []
    assert ctx.st.step("lab").current.apps == []


def test_a_later_kind_crashing_keeps_the_earlier_kinds_evidence(lab_ctx, monkeypatch):
    ctx, recs = lab_ctx

    class Boom(_Call):
        def get(self, timeout):
            raise OSError(36, "File name too long")

    fake = _Lab({"tempseg": _good("tempseg")})
    orig = fake._spawn
    fake._spawn = lambda kind, args: Boom(None) if kind == "captioner" else orig(kind, args)
    fake.run_captioner = SimpleNamespace(spawn=lambda *a: fake._spawn("captioner", a))
    saved = []
    real_save = wh.save_state

    def spy(st, path=wh.STATE):
        saved.append(json.loads(json.dumps(st.step("lab").result, default=str)))
        real_save(st, path)

    monkeypatch.setattr(wh, "save_state", spy)
    res = wh.run_lab(ctx, missing=lambda p: [], records=recs, lab_module=fake)
    assert not res["ok"] and "File name too long" in res["kinds"]["captioner"]["error"]
    assert res["kinds"]["tempseg"]["evidence"] is True
    # persisted after TempSeg returned, before the Captioner call
    after_tempseg = [s for s in saved if "tempseg" in s.get("kinds", {})
                     and "captioner" not in s["kinds"]]
    assert after_tempseg and after_tempseg[0]["kinds"]["tempseg"]["evidence"] is True
    on_disk = json.loads(ctx.state_path.read_text())["steps"]["lab"]["result"]
    assert on_disk["kinds"]["tempseg"]["evidence"] is True


def test_only_lab_dry_run_prices_only_what_is_not_reused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wh, "LAB_OUT", tmp_path / "out")
    monkeypatch.setattr(h0, "golden_manifest", lambda: wh.offline_manifest(wh.DURATION_HEADROOM))
    st = _done_state()
    st.run_id = "r"
    state = tmp_path / "state.json"
    wh.save_state(st, state)
    wh.LAB_OUT.mkdir()
    wh.lab_result_path("tempseg").write_text(json.dumps(_saved("tempseg")))
    full = wh.project("lab", wh.offline_manifest(wh.DURATION_HEADROOM))
    assert wh.main(["--dry-run", "--only", "lab", "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert "lab reuses the saved tempseg result" in out
    assert "tempseg: saved result reused" in out
    cap_only = wh.project("lab", wh.offline_manifest(wh.DURATION_HEADROOM), reuse=("tempseg",))
    assert f"upper ${cap_only.usd_upper:.2f}" in out
    assert cap_only.usd_upper == pytest.approx(full.items[1]["usd_upper"], abs=1e-3)
    assert cap_only.peak_usd_per_s == wh.LAB["captioner"].usd_per_s()
    # the plan's global cap is still derived from the whole lab step, not the reduced one
    ps = {s: wh.project(s, wh.offline_manifest(wh.DURATION_HEADROOM)) for s in wh.STEPS}
    reduced = {**ps, "lab": cap_only}
    assert wh.plan_global_cap("balanced-only", reduced) == wh.plan_global_cap(
        "balanced-only", ps)


# --- deadline and projection --------------------------------------------------------------------


def test_lab_deadline_covers_two_cold_compiles_within_the_container_timeout():
    assert wh.LAB_TIMEOUT_S == lab.TIMEOUT_S
    for kind, (base, plugin) in wh.LAB_BUILD_S.items():
        exp, deadline = wh.lab_seconds(kind)
        assert exp >= base + plugin + wh.LAB_WORK_S[kind]
        assert exp < deadline <= wh.LAB_TIMEOUT_S
    assert wh.LAB_BUILD_S["captioner"] == (325.0, 325.0)  # the measured H100 cold start, x2
    assert wh.LAB_DEADLINE_S["captioner"] > 600  # the old deadline cut it off mid-compile
    p = wh.project("lab", wh.offline_manifest(wh.DURATION_HEADROOM))
    assert [i["gpu"] for i in p.items] == ["L40S", "H100"]
    assert p.usd_upper <= wh.CAPS_USD["lab"] < p.usd_upper + 0.05 + 1e-9


def test_only_lab_dry_run_prices_just_the_lab(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(h0, "golden_manifest", lambda: wh.offline_manifest(wh.DURATION_HEADROOM))
    monkeypatch.setattr(wh, "RUNNERS", {k: None for k in wh.STEPS})
    state = tmp_path / "state.json"
    wh.save_state(_done_state(), state)
    before = state.read_text()
    assert wh.main(["--dry-run", "--only", "lab", "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert "d2. balanced lab" in out and "balanced lab captioner" in out
    assert "d. balanced via flora deploy/run" not in out and "0. TempSeg" not in out
    assert "(d)'s deployed job is done" in out and state.read_text() == before  # dry: unsaved


def test_adopting_the_finished_deployed_job():
    st = _done_state()
    assert wh.adopt_balanced_job(st) and st.steps["balanced"].status == "done"
    assert not wh.adopt_balanced_job(st)  # once
    st2 = _done_state()
    st2.step("balanced").result["jobs"][wh.JOB]["state"] = "failed"
    assert not wh.adopt_balanced_job(st2) and st2.steps["balanced"].status == "failed"


# --- cold startups are scaled once --------------------------------------------------------------


def _entry(**kw):
    from flora_deploy.planner.rates import StartupEntry

    base = dict(preset="balanced", kind="caption", startup_s=325.0, uncertainty=0.5,
                status="measured", provenance="p", date="2026-10-22", compile_cache="cold")
    return StartupEntry(**{**base, **kw})


def test_a_measured_cold_startup_is_not_scaled_again():
    f = wh.COLD_COMPILE_FACTOR
    assert wh.cold_startup_s(_entry()) == (325.0, "measured cold")
    assert wh.cold_startup_s(_entry(compile_cache="warm"))[0] == pytest.approx(325.0 * f)
    assert wh.cold_startup_s(_entry(compile_cache=None))[0] == pytest.approx(325.0 * f)
    assert wh.cold_startup_s(_entry(status="projected", compile_cache=None))[0] == \
        pytest.approx(325.0 * f)
    assert wh.cold_startup_s(_entry(status="placeholder", compile_cache=None))[1].startswith(
        "placeholder x")


def test_shipped_balanced_startups_are_recorded_cold_and_priced_as_measured():
    from flora_deploy.planner.rates import RatesFileError, RateTable

    t = RateTable.load()
    for kind, sec in (("segment", 300.5), ("caption", 325.0)):
        st = t.startup_for("balanced", kind)
        assert (st.status, st.compile_cache, st.startup_s) == ("measured", "cold", sec)
    secs = wh.balanced_job_seconds(100.0, t)
    unscaled = wh.balanced_job_seconds(100.0, t, compile_factor=1.0)
    assert secs == unscaled  # measured cold: the factor does not apply
    p = wh.project("balanced", wh.offline_manifest(wh.DURATION_HEADROOM))
    assert p.usd_upper <= wh.CAPS_USD["balanced"]
    with pytest.raises(RatesFileError, match="compile_cache"):
        RateTable.from_dict({"schema": 1, "startup": [{
            "preset": "x", "kind": "*", "startup_s": 1, "uncertainty": 0, "status": "measured",
            "provenance": "p", "date": "d", "compile_cache": "lukewarm"}]}, source="t")
