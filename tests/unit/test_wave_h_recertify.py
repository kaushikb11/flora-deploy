"""``scripts/wave_h.py --plan recertify`` step f2 ``noise_all`` offline: the certified run's (b)
repeated on all 12 of its episodes from (b)'s own config and data dir (so the same config and
stage hashes), its own state, job, app and index, priced and capped by the runbook's rules, and
the certified state never written. Modal is never contacted (``_index`` / ``_run_job`` faked)."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wh = _load("wave_h")
rb = sys.modules["_runbook"]
RUN = "10221548f1b8"
B_JOB = f"wave-h-{RUN}-exact_e2e-human"


def _write_source(tmp_path: Path, **job_over) -> tuple[Path, dict]:
    """A finished certified state shaped like run 10221548f1b8's, with (b)'s real config text
    (made by this script) and a small data dir; (b)'s hashes are the ones this checkout plans."""
    work = tmp_path / "certified" / "work"
    data = work / wh.CERT_DATA
    data.mkdir(parents=True)
    for i, g in enumerate(wh.CERT_EPISODES):
        local = wh.SPECS[wh.CERTIFIED].local_id(g)
        (data / f"{local}.mp4").write_bytes(b"v" * (i + 1))
        (data / f"{local}.txt").write_text("task\n")
    ctx = wh.Ctx(rb.State(RUN, 0.0, "x"), {}, work)
    text, job_id = wh.config_text(ctx, "exact_e2e-human", embodiment="human",
                                  stages=("segment", "caption"), budget=4.0,
                                  derived={"captioner_bf16": "c" * 64}, data=wh.CERT_DATA)
    assert job_id == B_JOB
    cfg = work / "flora-exact_e2e-human.yaml"
    cfg.write_text(text)
    chash, shashes = wh.planned_hashes(text, work)
    (work / "summary-exact_e2e-human.json").write_text(json.dumps(
        {"counts": {"episodes": 12}, "video_s": 235.914156,
         "dataset": {"revision": "local-abc"}}))
    job = {"app": f"flora-{B_JOB}-x", "cfg": str(cfg), "config_hash": chash,
           "stage_hashes": shashes, "containers": [{"gpu": "H200", "kind": "segcap"}],
           "data": wh.CERT_DATA, "job_id": B_JOB, "preset": "exact", "state": "done",
           "video_s": 235.8, "records": {"segment": "rec/exact/human/segment.jsonl"}}
    job.update(job_over)
    st = rb.State(RUN, 1.0, "2026-10-21", plan=wh.CERTIFIED)
    for name, status in (("exact_e2e", "done"), ("noise", "done"), ("balanced", "done"),
                         ("lab", "done"), ("certify", "failed")):
        st.step(name).status = status
    st.step("exact_e2e").result = {"ok": True, "episodes": 12, "jobs": {"human": job}}
    st.step("noise").result = {"jobs": {"human": {"job_id": f"wave-h-{RUN}-noise-human"}}}
    st.step("balanced").result = {
        "jobs": {"human": {"job_id": f"wave-h-{RUN}-balanced-human"}},
        "witness_jobs": {"human": {"job_id": f"wave-h-{RUN}-cold-human-a2"}}}
    path = tmp_path / "certified" / "state.json"
    rb.save_state(st, path)
    return path, job


def _spec(tmp_path: Path, source: Path) -> wh.PlanSpec:
    root = tmp_path / "recertified"
    return replace(wh.SPECS[wh.RECERTIFY], source=source, state=root / "state.json",
                   work=source.parent / "work", records=root / "records",
                   report=root / "runbook.md")


def _fakes(monkeypatch, calls: list, *, summary: dict | None = None):
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})

    def index(ctx, cfg, job_id, tag, *, limit=None):
        calls.append(("index", job_id, tag, limit, Path(cfg)))
        (ctx.work / f"summary-{tag}.json").write_text(json.dumps(summary or {
            "counts": {"episodes": 12}, "video_s": 235.914156,
            "dataset": {"revision": "local-abc"}}))

    def run_job(ctx, step, cfg, job_id, **kw):
        calls.append(("run", step, job_id, Path(cfg).read_text(), kw))
        return {"app": f"app-{job_id}", "job_id": job_id, "state": "done", "spend": 2.5,
                "cfg": str(cfg)}

    monkeypatch.setattr(wh, "_index", index)
    monkeypatch.setattr(wh, "_run_job", run_job)


def _ctx(st, spec, attempts: int = 1) -> wh.Ctx:
    for _ in range(attempts):
        st.step("noise_all").attempts.append(rb.Attempt(started=1.0))
    projs = wh.plan_projections(wh.RECERTIFY, wh.offline_manifest(0.0, wh.CERT_GOLDEN),
                                spec=spec)
    return wh.Ctx(st, wh.offline_manifest(0.0, wh.CERT_GOLDEN), spec.work, spec=spec,
                  caps=wh.plan_caps(wh.RECERTIFY, projs), step="noise_all",
                  plan_steps=wh.PLANS[wh.RECERTIFY], projections=projs)


# ---------------------------------------------------------------------------------------------
# Plan shape, prices and caps


def test_recertify_plan_is_f2_alone_with_its_own_state_reading_the_certified_one():
    spec = wh.SPECS[wh.RECERTIFY]
    cert = wh.SPECS[wh.CERTIFIED]
    assert wh.PLANS[wh.RECERTIFY] == ("noise_all",)
    assert wh.STEPS == ("tempseg", "preflight", "exact_e2e", "balanced", "lab", "certify",
                        "noise")  # the other plans' steps are unchanged
    assert spec.state == wh.RECERT_ROOT / "state.json" and spec.state != cert.state
    assert spec.source == cert.state and spec.work == cert.work
    assert spec.episodes == cert.episodes and len(spec.episodes) == 12
    # the local ids (the n-/x- file names in (b)'s data dir) are the certified run's
    assert all(spec.local_id(g) == cert.local_id(g) for g in spec.episodes)
    assert wh.deferred_steps("full") == ()
    assert wh.titles(wh.RECERTIFY)["noise_all"].startswith("f2.")


def test_noise_all_is_priced_as_b_on_all_twelve_and_capped_by_the_rounding_rule():
    m = wh.offline_manifest(wh.DURATION_HEADROOM, wh.CERT_GOLDEN)
    projs = wh.plan_projections(wh.RECERTIFY, m, spec=wh.SPECS[wh.RECERTIFY])
    p, b = projs["noise_all"], projs["exact_e2e"]
    assert [i["seconds"] for i in p.items] == [i["seconds"] for i in b.items]  # (b)'s price
    assert [i["upper_seconds"] for i in p.items] == [i["upper_seconds"] for i in b.items]
    assert "all 12 ep" in p.items[0]["what"] and p.items[0]["gpu"] == "H200"
    assert p.usd == pytest.approx(2.739, abs=0.002)  # caption-prereg.md section 5, f2
    assert p.usd_upper == pytest.approx(4.166, abs=0.002)
    caps = wh.plan_caps(wh.RECERTIFY, projs)
    assert caps["noise_all"] == 4.20  # upper rounded up to $0.05
    assert wh.plan_global_cap(wh.RECERTIFY, projs) == 4.20  # sum rounded up to $0.10
    st = rb.State(RUN, 0.0, "x", plan=wh.RECERTIFY)
    d = wh.decide(st, wh.RECERTIFY, "noise_all", projs)
    assert d.action == "run" and "--plan recertify --yes-spend" in d.command


# ---------------------------------------------------------------------------------------------
# The step


def test_noise_all_repeats_b_on_every_episode_with_b_hashes(tmp_path, monkeypatch):
    source, job = _write_source(tmp_path)
    before = source.read_bytes()
    spec = _spec(tmp_path, source)
    st = wh.new_recert_state(spec)
    assert st.run_id == RUN and st.plan == wh.RECERTIFY and not st.steps
    calls: list = []
    _fakes(monkeypatch, calls)
    res = wh.run_noise_all(_ctx(st, spec))
    job_id = f"wave-h-{RUN}-noise12-human-a1"
    assert res["ok"] and res["episodes"] == 12 and res["stage_hashes_of_oracle"]
    (_, ijob, tag, limit, icfg), = [c for c in calls if c[0] == "index"]
    assert ijob == job_id and limit is None  # no --limit: every episode
    assert tag == "noise12-human-a1"  # its own local store / summary, not (b)'s or (f)'s
    cfg = Path(job["cfg"]).parent / "flora-noise12-human-a1.yaml"
    assert icfg == cfg
    (_, step, rjob, text, kw), = [c for c in calls if c[0] == "run"]
    assert step == "noise_all" and rjob == job_id
    assert kw["determinism"] == "nvidia" == wh.ORACLE_DETERMINISM
    assert kw["usd_per_s"] == pytest.approx(wh.pool_container("segcap").usd_per_s())
    assert text == Path(job["cfg"]).read_text().replace(B_JOB, job_id)  # project renamed only
    assert wh.planned_hashes(text, cfg.parent) == (job["config_hash"], job["stage_hashes"])
    assert res["apps"] == [f"app-{job_id}"]
    assert res["oracle"]["job_id"] == B_JOB and res["oracle"]["run_id"] == RUN
    assert res["oracle"]["stage_hashes"] == job["stage_hashes"]
    assert source.read_bytes() == before  # the certified state is never written
    # a retry is its own job, app and index
    calls.clear()
    st.step("noise_all").attempts.append(rb.Attempt(started=2.0))
    wh.run_noise_all(wh.Ctx(st, {}, spec.work, spec=spec, step="noise_all"))
    assert [c[2] for c in calls if c[0] == "run"] == [f"wave-h-{RUN}-noise12-human-a2"]


@pytest.mark.parametrize("over, match", [
    ({"stage_hashes": {"segcap": "0" * 64}}, "not \\(b\\)'s"),
    ({"config_hash": "1" * 64}, "not \\(b\\)'s"),
    ({"state": "failed"}, "not done"),
    ({"containers": [{"gpu": "H100"}]}, "not H200"),
    ({"preset": "balanced"}, "not exact"),
    ({"data": "elsewhere"}, "data dir"),
])
def test_noise_all_refuses_before_any_index_or_app(tmp_path, monkeypatch, over, match):
    source, _ = _write_source(tmp_path, **over)
    spec = _spec(tmp_path, source)
    st = rb.State(RUN, 0.0, "x", plan=wh.RECERTIFY)
    calls: list = []
    _fakes(monkeypatch, calls)
    with pytest.raises(RuntimeError, match=match):
        wh.run_noise_all(_ctx(st, spec))
    assert calls == []
    assert not list((source.parent / "work").glob("flora-noise12-*.yaml"))


def test_noise_all_refuses_a_changed_data_dir_another_run_or_a_short_index(tmp_path,
                                                                           monkeypatch):
    source, job = _write_source(tmp_path)
    spec = _spec(tmp_path, source)
    calls: list = []
    _fakes(monkeypatch, calls)
    # another run id: (b)'s config writes into (b)'s run store
    with pytest.raises(RuntimeError, match="not \\(b\\)'s run"):
        wh.run_noise_all(_ctx(rb.State("other", 0.0, "x"), spec))
    # the index does not hold (b)'s twelve episodes: refused before the app
    _fakes(monkeypatch, calls, summary={"counts": {"episodes": 6}, "video_s": 99.0,
                                        "dataset": {"revision": "local-abc"}})
    with pytest.raises(RuntimeError, match="noise_all index"):
        wh.run_noise_all(_ctx(rb.State(RUN, 0.0, "x"), spec))
    assert not [c for c in calls if c[0] == "run"]
    # a data dir whose content changed is another stage hash (the content is hashed)
    calls.clear()
    data = Path(job["cfg"]).parent / wh.CERT_DATA
    next(data.glob("*.mp4")).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="not \\(b\\)'s"):
        wh.run_noise_all(_ctx(rb.State(RUN, 0.0, "x"), spec))
    assert calls == []


def test_noise_all_index_check_needs_b_summary_exactly(tmp_path):
    src = {"episodes": 12, "job": {"video_s": 235.8}}
    s, o = tmp_path / "s.json", tmp_path / "o.json"
    o.write_text(json.dumps({"counts": {"episodes": 12}, "video_s": 235.9,
                             "dataset": {"revision": "r1"}}))
    s.write_text(json.dumps({"counts": {"episodes": 12}, "video_s": 235.9,
                             "dataset": {"revision": "r1"}}))
    assert wh.noise_all_index_problems(s, o, src) == ""
    s.write_text(json.dumps({"counts": {"episodes": 12}, "video_s": 235.9,
                             "dataset": {"revision": "r2"}}))
    assert "revision" in wh.noise_all_index_problems(s, o, src)
    s.write_text(json.dumps({"counts": {"episodes": 11}, "video_s": 220.0}))
    assert "11 episodes" in wh.noise_all_index_problems(s, o, src)
    # without (b)'s summary: (b)'s recorded video seconds within 1%
    s.write_text(json.dumps({"counts": {"episodes": 12}, "video_s": 236.5}))
    assert wh.noise_all_index_problems(s, tmp_path / "missing.json", src) == ""


# ---------------------------------------------------------------------------------------------
# main: dry run, status, refusals, a run (runner faked)


def _main_args(tmp_path, source, *extra):
    state = tmp_path / "recertified" / "state.json"
    return ["--plan", "recertify", "--source-state", str(source), "--state", str(state),
            *extra], state


def test_dry_run_and_status_show_the_hash_check_and_write_nothing(tmp_path, capsys):
    source, job = _write_source(tmp_path)
    before = source.read_bytes()
    args, state = _main_args(tmp_path, source, "--dry-run", "--offline")
    assert wh.main(args) == 0
    out = capsys.readouterr().out
    assert f"run {RUN}" in out and "f2. noise-12" in out and "cap $4.20" in out
    assert f"segcap {job['stage_hashes']['segcap']} == (b)'s" in out
    assert "no --limit" in out and "project wave-h-10221548f1b8-noise12-human-a1" in out
    assert "global cap $4.20" in out
    args, _ = _main_args(tmp_path, source, "--status", "--offline")
    assert wh.main(args) == 0
    out = capsys.readouterr().out
    assert "--plan recertify --yes-spend" in out and "== (b)'s" in out
    assert not state.exists() and source.read_bytes() == before
    assert not list((source.parent / "work").glob("flora-noise12-*"))


def test_dry_run_shows_a_refusal_when_the_checkout_no_longer_plans_b(tmp_path, capsys):
    source, _ = _write_source(tmp_path, stage_hashes={"segcap": "0" * 64})
    args, _ = _main_args(tmp_path, source, "--dry-run", "--offline")
    assert wh.main(args) == 0
    assert "f2 check: REFUSED before any app" in capsys.readouterr().out


def test_main_refuses_the_certified_state_and_misplaced_flags(tmp_path, capsys):
    source, _ = _write_source(tmp_path)
    with pytest.raises(SystemExit):
        wh.main(["--plan", "recertify", "--source-state", str(source), "--state",
                 str(source), "--dry-run", "--offline"])
    with pytest.raises(SystemExit):
        wh.main(["--plan", "certified", "--source-state", str(source), "--dry-run",
                 "--offline"])
    with pytest.raises(SystemExit):
        wh.main(["--plan", "recertify", "--embodiment", "yam", "--dry-run", "--offline"])
    # a recertify state is never run under the certified plan, nor the other way round
    other = tmp_path / "other.json"
    rb.save_state(rb.State(RUN, 0.0, "x", plan=wh.RECERTIFY), other)
    assert wh.main(["--plan", "certified", "--state", str(other), "--dry-run",
                    "--offline"]) == 2
    # no readable certified state: refused
    assert wh.main(["--plan", "recertify", "--source-state", str(tmp_path / "none.json"),
                    "--state", str(tmp_path / "s.json"), "--dry-run", "--offline"]) == 2
    assert "refused" in capsys.readouterr().out


def test_a_run_records_f2_in_its_own_state_on_b_run_id(tmp_path, monkeypatch):
    source, _ = _write_source(tmp_path)
    before = source.read_bytes()
    spec = _spec(tmp_path, source)
    monkeypatch.setitem(wh.SPECS, wh.RECERTIFY, spec)
    monkeypatch.setattr(wh, "load_manifest",
                        lambda offline, version: wh.offline_manifest(0.0, version))
    monkeypatch.setattr(wh, "billed_by_app", lambda since: {})
    monkeypatch.setattr(wh, "refresh_billing", lambda st, **kw: True)
    monkeypatch.setattr(wh, "stop_killed", lambda st: None)
    seen = {}

    def fake(ctx):
        seen.update(run_id=ctx.st.run_id, step=ctx.step, budget=ctx.budget("noise_all"))
        ctx.note_app("app-noise12-a1")
        return {"ok": True, "tracked_usd": 2.5, "apps": ["app-noise12-a1"], "episodes": 12}

    monkeypatch.setitem(wh.RUNNERS, "noise_all", fake)
    monkeypatch.chdir(tmp_path)
    assert wh.main(["--plan", "recertify", "--yes-spend"]) == 0
    st = wh.load_state(spec.state)
    assert st.run_id == RUN and st.plan == wh.RECERTIFY
    projs = wh.plan_projections(wh.RECERTIFY, wh.offline_manifest(0.0, wh.CERT_GOLDEN),
                                spec=spec)
    assert st.step("noise_all").status == "done"
    assert (st.gate_cap, st.gate_cap_plan) == (wh.plan_global_cap(wh.RECERTIFY, projs),
                                               wh.RECERTIFY)
    a, = st.step("noise_all").attempts
    assert a.apps == ["app-noise12-a1"] and a.tracked_usd == pytest.approx(2.5)
    assert seen["run_id"] == RUN and seen["step"] == "noise_all" and seen["budget"] > 0
    assert set(st.steps) == {"noise_all"}
    assert source.read_bytes() == before  # (b), (f), (d), (d2), (e) untouched
    assert spec.report.exists()
    # done: a second run starts nothing
    monkeypatch.setitem(wh.RUNNERS, "noise_all", lambda ctx: pytest.fail("re-ran"))
    assert wh.main(["--plan", "recertify", "--yes-spend"]) == 0
