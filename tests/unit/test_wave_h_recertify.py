"""``scripts/wave_h.py --plan recertify`` offline. Step f2 ``noise_all``: the certified run's (b)
repeated on all 12 of its episodes from (b)'s own config and data dir (so the same config and
stage hashes), its own state, job, app and index, priced and capped by the runbook's rules, and
the certified state never written. Then the stage-isolated caption re-certification (caption
pre-registration §3, §5, §6 item 13): the v4-code guard of d2' / d' / g, local ids and uids
unchanged from (b), the segment-only J1w, the staged oracle segments and the J2 / J2w configs,
the certify view on (b)'s oracle and f2's recorded run, the projections and the $11.70 cap
(f2 done at its actual cost). Modal is never contacted (``_index`` / ``_run_job`` faked)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.fakes.certrecords import BOUNDS, ORACLE_HASH, rec, segs

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fin = _load("wave_h_finalize")
wh = sys.modules["wave_h"]
rb = sys.modules["_runbook"]
h0 = sys.modules["h0_rehearsal"]
RC = wh.SPECS[wh.RECERTIFY]
B = wh.SPECS[wh.CERTIFIED]
B_RUN = "b1"
SEGCAP = ORACLE_HASH  # (b)'s segcap stage hash: its records' provenance.config_hash
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


def test_recertify_plan_starts_with_f2_with_its_own_state_reading_the_certified_one():
    spec = wh.SPECS[wh.RECERTIFY]
    cert = wh.SPECS[wh.CERTIFIED]
    assert wh.PLANS[wh.RECERTIFY][0] == "noise_all"  # f2 first; phase 2 after it
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
    # the plan's global cap: its uppers' sum rounded up to $0.10, at most the approved $11.70
    assert wh.plan_global_cap(wh.RECERTIFY, projs) == wh.RECERT_GLOBAL_CAP_USD == 11.70
    st = rb.State(RUN, 0.0, "x", plan=wh.RECERTIFY)
    d = wh.decide(st, wh.RECERTIFY, "noise_all", projs)
    # f2's upper with every phase-2 upper kept after it is $11.90 on today's rates: over the
    # approved $11.70, so a fresh f2 here is stopped (it runs from main's checkout instead,
    # under main's f2-only cap; this plan then finds it done at its actual cost) ...
    assert d.action == "stop" and "reserve for lab, balanced, caption_iso" in d.message
    # ... and runs with a raise the user approves
    st.cap_override, st.cap_override_plan = 12.0, wh.RECERTIFY
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
    assert "global cap $11.70" in out
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
    assert wh.main(["--plan", "recertify", "--yes-spend", "--only", "noise_all"]) == 0
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
    assert wh.main(["--plan", "recertify", "--yes-spend", "--only", "noise_all"]) == 0


# =============================================================================================
# Phase 2: the stage-isolated caption re-certification


def _manifest() -> dict:
    return wh.offline_manifest(wh.DURATION_HEADROOM, wh.CERT_GOLDEN)


def _uid(g: str) -> str:
    from flora_deploy.jobs.types import episode_uid

    return episode_uid(wh.CERT_DATA, B.local_id(g))


def _jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records))


def world(tmp_path: Path):
    """(b)'s certified state (oracle + old 6-episode noise, its config file and exported
    oracle segment records) and a recertify spec under ``tmp_path``."""
    broot = tmp_path / "certified"
    (broot / "work").mkdir(parents=True)
    ocfg = broot / "work" / "flora-exact_e2e-human.yaml"
    ocfg.write_text(h0.config_text(project=f"wave-h-{B_RUN}-exact_e2e-human",
                                   data_dir=wh.CERT_DATA, embodiment="human",
                                   stages=("segment", "caption"), store_volume=wh.STORE_VOLUME,
                                   media_volume=wh.MEDIA_VOLUME, run_id=B_RUN))
    seg_path = broot / "records" / "exact" / wh.JOB / "segment.jsonl"
    oracle = [rec(_uid(g), "segment", "exact", embodiment="human", fp=None, gpu=None,
                  segments=segs([(s + i, e + i) for s, e in BOUNDS]))
              for i, g in enumerate(B.episodes)]
    _jsonl(seg_path, list(reversed(oracle)))  # file order is not the staged order
    wh.records_source_path(seg_path).write_text(json.dumps(
        {"job_id": f"wave-h-{B_RUN}-exact_e2e-human"}))
    bst = wh.State(run_id=B_RUN, created=0.0, billing_since="x", plan=wh.CERTIFIED)
    for step, jid in (("exact_e2e", "exact_e2e-human"), ("noise", "noise-human")):
        s = bst.step(step)
        s.status = "done"
        s.result = {"jobs": {wh.JOB: {
            "job_id": f"wave-h-{B_RUN}-{jid}", "state": "done", "cfg": str(ocfg),
            "stage_hashes": {"segcap": SEGCAP}, "data": wh.CERT_DATA,
            "records": {"segment": str(seg_path)}}}}
    wh.save_state(bst, broot / "state.json")
    root = tmp_path / "recertified"
    spec = replace(RC, state=root / "state.json", work=root / "work",
                   records=root / "records", report=root / "runbook.md", lab_dir=root,
                   source=broot / "state.json")
    spec.work.mkdir(parents=True)
    (spec.work / wh.CERT_DATA).mkdir()
    st = wh.State(run_id="rc1", created=0.0, billing_since="x", plan=wh.RECERTIFY)
    ctx = wh.Ctx(st, _manifest(), spec.work, state_path=spec.state, spec=spec,
                 caps=wh.plan_caps(wh.RECERTIFY, wh.plan_projections(wh.RECERTIFY,
                                                                    _manifest())))
    return ctx, oracle, ocfg


@pytest.fixture
def v4(monkeypatch):
    import flora_deploy.validate.certkey as K

    monkeypatch.setattr(K, "GENERATOR_VERSIONS", ("4",))


@pytest.fixture
def v3(monkeypatch):
    import flora_deploy.validate.certkey as K

    monkeypatch.setattr(K, "GENERATOR_VERSIONS", ("3",))


# ---------------------------------------------------------------------------------------------
# Plan shape, ids


def test_recertify_steps_state_root_and_chain():
    assert wh.PLANS[wh.RECERTIFY] == ("noise_all", "lab", "balanced", "caption_iso", "certify")
    assert wh.plan_deps(wh.RECERTIFY) == {"noise_all": (), "lab": ("noise_all",),
                                         "balanced": ("lab",), "caption_iso": ("balanced",),
                                         "certify": ("caption_iso",)}
    assert RC.state == wh.RECERT_ROOT / "state.json" and RC.state.parent != B.state.parent
    assert RC.records.parent == RC.labs == wh.RECERT_ROOT and RC.work == B.work
    assert RC.source == wh.CERT_ROOT / "state.json"  # (b)'s, read only
    assert RC.job_prefix == "rc-" and B.job_prefix == ""  # (b)'s run id: never (b)'s jobs
    assert RC.golden == B.golden and RC.episodes == B.episodes and RC.data == wh.CERT_DATA
    # the noise run on all 12 (R2), the witnesses on (b)'s old noise subset
    assert RC.noise == RC.episodes and RC.noise_limit is None
    assert RC.witness == B.noise and RC.witness_limit == 6 and RC.segment_only_witness
    assert wh.max_parallel_for(wh.RECERTIFY, None, 3) == 1
    assert wh.runner(wh.RECERTIFY, "lab") is wh.run_recert_lab
    assert wh.runner(wh.RECERTIFY, "certify") is wh.run_flora_certify
    assert wh.runner(wh.RECERTIFY, "noise_all") is wh.run_noise_all
    assert wh.runner(wh.RECERTIFY, "caption_iso") is wh.run_caption_iso


def test_local_ids_stay_keyed_on_the_old_noise_subset_so_uids_pair_with_b(tmp_path):
    from flora_deploy.media.sources.mp4_dir import Mp4DirSource

    for g in RC.episodes:
        assert RC.local_id(g) == B.local_id(g)  # every uid as (b) had it
        assert RC.local_id(g).startswith(wh.NOISE_PREFIX) == (g in B.noise)
        (tmp_path / f"{RC.local_id(g)}.mp4").write_bytes(b"")
        (tmp_path / f"{RC.local_id(g)}.txt").write_text("t\n")
    # `flora index --limit 6` (the witnesses) still takes exactly the old noise subset
    eps = Mp4DirSource(tmp_path).list_episodes()
    assert sorted(e.episode_id for e in eps[:RC.witness_limit]) == sorted(
        RC.local_id(g) for g in RC.witness)
    # a spec whose noise grows to every episode keeps the ids only through `witness`
    grown = replace(B, noise=B.episodes)
    assert all(grown.local_id(g) == B.local_id(g) for g in B.episodes)


def test_j1w_is_segment_only_and_its_projection_one_pool(tmp_path):
    from flora_deploy.config.schema import load_config

    (tmp_path / wh.CERT_DATA).mkdir()
    text = h0.config_text(project="p", data_dir=wh.CERT_DATA, embodiment="human",
                          stages=("segment", "caption"), store_volume="s", media_volume="m",
                          run_id="r", preset="balanced")
    w = wh.witness_config_text(text, "p-cold", ("segment",))
    cfg = tmp_path / "w.yaml"
    cfg.write_text(w)
    c = load_config(cfg, allow_uncertified=True)
    assert c.project == "p-cold" and c.pipeline.stages == ("segment",)
    assert "caption:" not in w.split("resources:")[1].split("output:")[0]
    assert w.replace("project: p-cold", "project: p").replace("[segment]",
                                                             "[segment, caption]") == \
        text.replace("  caption: {max_containers: 1}\n", "")
    p = wh.project("balanced", _manifest(), RC)
    wit = [i for i in p.items if "cold witness" in i["what"]]
    assert len(wit) == 1 and "segment-only" in wit[0]["what"] and wit[0]["gpu"] == "L40S"
    dur = wh.durations(_manifest())
    n, u = wh.single_pool_seconds("segment", sum(dur[g] for g in RC.witness))
    assert (wit[0]["seconds"], wit[0]["upper_seconds"]) == (round(n, 1), round(u, 1))
    # the golden/v2 human certified plan keeps its full witness (its run is done)
    assert not B.segment_only_witness
    assert sum("cold witness" in i["what"] for i in wh.project("balanced", _manifest(),
                                                              B).items) == 2


def test_run_cold_witness_runs_the_segment_stage_alone(tmp_path, monkeypatch):
    ctx, _, _ = world(tmp_path)
    cfg = ctx.work / "flora-balanced-human.yaml"
    cfg.write_text(h0.config_text(project="wave-h-rc1-balanced-human", data_dir=wh.CERT_DATA,
                                  embodiment="human", stages=("segment", "caption"),
                                  store_volume="s", media_volume="m", run_id="rc1",
                                  preset="balanced"))
    calls = []
    monkeypatch.setattr(wh, "_index", lambda c, f, j, t, *, limit=None:
                        calls.append(("index", j, limit)))
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")

    def run_job(c, step, f, job_id, **kw):
        calls.append(("run", job_id, c.env.get("FLORA_COMPILE_CACHE"), kw["usd_per_s"],
                      Path(f).read_text()))
        return {"app": "a", "job_id": job_id, "state": "done", "spend": 0.1}

    monkeypatch.setattr(wh, "_run_job", run_job)
    w = wh.run_cold_witness(ctx, cfg, 9.9, spent=0.0)
    assert w["stages"] == ["segment"] and w["episodes"] == 6
    assert ("index", "wave-h-rc1-rc-cold-human-a1", 6) in calls
    run = next(c for c in calls if c[0] == "run")
    assert run[2] == "off" and "  stages: [segment]\n" in run[4]
    assert run[3] == pytest.approx(wh.pool_container("segment", "balanced").usd_per_s())


# ---------------------------------------------------------------------------------------------
# The v4-code guard of d2', d', g


def test_lab_balanced_and_caption_iso_refuse_without_the_v4_code(tmp_path, monkeypatch, v3):
    ctx, _, _ = world(tmp_path)
    monkeypatch.setattr(wh, "stage_oracle_segments", lambda *a, **k: pytest.fail("staged"))
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: pytest.fail("reached"))
    for fn in (wh.run_recert_lab, wh.run_balanced, wh.run_caption_iso):
        with pytest.raises(RuntimeError, match="v4 certification code"):
            fn(ctx)


def test_recert_lab_reuses_tempseg_and_runs_the_captioner_only(tmp_path, monkeypatch, v4):
    from tests.unit.test_wave_h_certified import _labs

    ctx, _, _ = world(tmp_path)
    saved = _labs(tmp_path / "saved")
    monkeypatch.setattr(wh, "LAB_OUT", saved)
    seen = {}

    def fake_run_lab(c, **kw):
        seen.update(kw)
        return {"ok": True, "kinds": {}, "tracked_usd": 1.2, "apps": ["x"]}

    monkeypatch.setattr(wh, "run_lab", fake_run_lab)
    res = wh.run_recert_lab(ctx)
    assert res["ok"] and seen["kinds"] == ("captioner",) and seen["lab_dir"] == ctx.spec.labs
    assert seen["source"].run_id == B_RUN  # (b)'s (d) job's segments
    assert (ctx.spec.labs / "balanced-tempseg.json").read_bytes() == (
        saved / "balanced-tempseg.json").read_bytes()
    # with the Captioner's result there too, the step is done at $0
    (ctx.spec.labs / "balanced-captioner.json").write_bytes(
        (saved / "balanced-captioner.json").read_bytes())
    again = wh.run_recert_lab(ctx)
    assert again["ok"] and again["tracked_usd"] == 0.0 and again["skipped"]
    # a TempSeg result for other engine settings is not reused
    (ctx.spec.labs / "balanced-captioner.json").unlink()
    (saved / "balanced-tempseg.json").write_text(json.dumps({"kind": "tempseg"}))
    with pytest.raises(RuntimeError, match="TempSeg's saved lab result is not reusable"):
        wh.run_recert_lab(ctx)


# ---------------------------------------------------------------------------------------------
# g: caption_iso


def _iso_fakes(monkeypatch, calls):
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **kw: {"captioner_bf16": "c" * 64})
    monkeypatch.setattr(wh, "stage_data", lambda ctx, data, ids: calls.append(("data", data)))
    monkeypatch.setattr(h0, "upload_dir", lambda vol, local, remote: calls.append(
        ("upload", vol, sorted(p.name for p in Path(local).iterdir()), remote)))
    monkeypatch.setattr(wh, "_index", lambda c, f, j, t, *, limit=None: calls.append(
        ("index", j, limit)))
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")

    def run_job(c, step, f, job_id, **kw):
        calls.append(("run", job_id, kw["side"], c.env.get("FLORA_COMPILE_CACHE"),
                      Path(f).read_text()))
        return {"app": f"app-{job_id}", "job_id": job_id, "state": "done", "spend": 0.4,
                "cfg": str(f)}

    monkeypatch.setattr(wh, "_run_job", run_job)


def test_caption_iso_stages_the_oracle_segments_and_runs_j2_then_j2w(tmp_path, monkeypatch, v4):
    from flora_deploy.config.schema import load_config

    ctx, oracle, _ = world(tmp_path)
    calls: list = []
    _iso_fakes(monkeypatch, calls)
    res = wh.run_caption_iso(ctx)
    assert res["ok"], res
    segs_ = res["oracle_segments"]
    # byte-exact StageRecord.to_json lines of (b)'s segment records, by uid; sha256 local
    want = "".join(r.to_json() + "\n" for r in sorted(oracle, key=lambda r: r.uid)).encode()
    staged = Path(segs_["path"])
    assert staged.read_bytes() == want and staged.name == f"{SEGCAP}.jsonl"
    assert segs_["sha256"] == hashlib.sha256(want).hexdigest()
    assert segs_["oracle_stage_hash"] == SEGCAP
    assert segs_["records"] == f"volume://{wh.STORE_VOLUME}/rc1/oracle-segments/{SEGCAP}.jsonl"
    assert ("upload", wh.STORE_VOLUME, [staged.name], "rc1/oracle-segments") in calls
    runs = [c for c in calls if c[0] == "run"]
    assert [(c[1], c[2], c[3]) for c in runs] == [
        ("wave-h-rc1-rc-iso-human", "isolated", None),
        ("wave-h-rc1-rc-isocold-human-a1", "isolated_witness", "off")]
    assert ("index", "wave-h-rc1-rc-iso-human", None) in calls
    assert ("index", "wave-h-rc1-rc-isocold-human-a1", 6) in calls
    for c in runs:  # J2 and J2w: the same pinned caption_segments, caption only
        cfg = ctx.work / "x.yaml"
        cfg.write_text(c[4])
        conf = load_config(cfg, allow_uncertified=True)
        assert conf.pipeline.stages == ("caption",) and conf.preset == "balanced"
        cs = conf.pipeline.caption_segments
        assert (cs.records, cs.sha256, cs.oracle_stage_hash) == (
            segs_["records"], segs_["sha256"], SEGCAP)
        assert cs.records.startswith(conf.output.sink + "/")
    assert res["jobs"][wh.JOB]["oracle_segments"] == segs_
    assert res["witness_jobs"][wh.JOB]["compile_cache"] == "off"
    assert res["tracked_usd"] == pytest.approx(0.8)


def test_caption_iso_reruns_only_j2w_when_j2_is_done(tmp_path, monkeypatch, v4):
    ctx, _, _ = world(tmp_path)
    calls: list = []
    _iso_fakes(monkeypatch, calls)
    first = wh.run_caption_iso(ctx)
    first["witness_jobs"][wh.JOB]["state"] = "failed"
    ctx.st.step("caption_iso").result = first
    ctx.st.step("caption_iso").attempts.append(wh.Attempt(started=1.0))
    ctx.st.step("caption_iso").attempts.append(wh.Attempt(started=2.0))
    assert wh.iso_witness_pending(ctx.st)
    calls.clear()
    monkeypatch.setattr(wh, "stage_oracle_segments", lambda *a, **k: pytest.fail("restaged"))
    res = wh.run_caption_iso(ctx)
    assert res["ok"] and res["witness_only"]
    assert [c[1] for c in calls if c[0] == "run"] == ["wave-h-rc1-rc-isocold-human-a2"]
    assert res["tracked_usd"] == pytest.approx(0.4)
    p = wh.project("caption_iso", _manifest(), RC, witness_only=True)
    assert len(p.items) == 1 and "cold witness" in p.items[0]["what"]


def test_oracle_segment_bytes_refuse_a_missing_or_foreign_record():
    r = rec("u1", "segment", "exact", fp=None, segments=segs(BOUNDS))
    other = rec("u2", "segment", "exact", fp=None, segments=segs(BOUNDS), config_hash="e" * 64)
    bal = rec("u3", "segment", "balanced", fp=None, segments=segs(BOUNDS),
              config_hash=ORACLE_HASH)
    assert wh.oracle_segment_bytes([r], ORACLE_HASH, ["u1"]) == (r.to_json() + "\n").encode()
    for recs in ([other], [bal], []):
        with pytest.raises(RuntimeError, match="no exact segment record"):
            wh.oracle_segment_bytes([r, *recs], ORACLE_HASH, ["u1", recs[0].uid if recs
                                                              else "u9"])


# ---------------------------------------------------------------------------------------------
# e': the certify view


def test_certify_view_takes_bs_oracle_f2_as_noise_and_bs_noise_as_repeat(tmp_path):
    ctx, _, _ = world(tmp_path)
    s = ctx.st.step("noise_all")
    s.status = "done"
    s.result = {"jobs": {wh.JOB: {"job_id": "wave-h-rc1-noise12-human-a1", "state": "done",
                                  "store": f"modal-volume://{wh.STORE_VOLUME}/{B_RUN}"}}}
    view = wh.certify_view(ctx.st, ctx.spec)
    b_store = f"modal-volume://{wh.STORE_VOLUME}/{B_RUN}"
    assert view.steps["exact_e2e"].result["jobs"][wh.JOB]["job_id"] == \
        f"wave-h-{B_RUN}-exact_e2e-human"
    assert view.steps["exact_e2e"].result["jobs"][wh.JOB]["store"] == b_store
    assert view.steps["noise"].result["jobs"][wh.JOB]["job_id"] == "wave-h-rc1-noise12-human-a1"
    assert view.steps["repeat"].result["jobs"][wh.JOB]["job_id"] == \
        f"wave-h-{B_RUN}-noise-human"
    assert "exact_e2e" not in ctx.st.steps  # the state itself is untouched
    assert wh.load_state(ctx.spec.source).steps["noise"].result["jobs"][wh.JOB].get(
        "store") is None  # nor (b)'s file
    # another plan's view is its state
    other = wh.State("r", 0.0, "x", plan=wh.CERTIFIED)
    assert wh.certify_view(other).steps == other.steps


# ---------------------------------------------------------------------------------------------
# Prices: §5 on its rates, the caps, the dry run

# The balanced rows planner/rates.json had when the pre-registration priced §5 (before run
# 10221548f1b8's measurement was folded in, a4c84e8): segment 0.367 c-s/v-s, 300.5 s cold
# start; caption 0.906, 325.0 s.
PREREG_RATES = {"segment": (0.367, 300.5), "caption": (0.906, 325.0)}
SECTION5 = {  # (expected, upper) per item, $ (caption pre-registration §5, human)
    "noise_all": [(2.739, 4.166)],
    "lab": [(0.0, 0.0), (1.281, 1.686)],
    "balanced": [(0.544, 0.873), (0.823, 1.554), (0.277, 0.585)],
    "caption_iso": [(0.823, 1.554), (0.634, 1.269)],
    "certify": []}


@pytest.fixture
def prereg_rates(monkeypatch):
    from flora_deploy.planner.rates import RateTable

    data = json.loads((REPO / "src/flora_deploy/planner/rates.json").read_text())
    for r in data["rates"]:
        if r["preset"] == "balanced" and r["kind"] in PREREG_RATES:
            r["container_s_per_video_s"] = PREREG_RATES[r["kind"]][0]
    for r in data["startup"]:
        if (r["preset"] == "balanced" and r["kind"] in PREREG_RATES
                and r.get("compile_cache") != "warm"):
            r["startup_s"] = PREREG_RATES[r["kind"]][1]
    t = RateTable.from_dict(data, source="prereg")
    monkeypatch.setattr(RateTable, "load", classmethod(lambda cls, *a, **k: t))


def test_projections_reproduce_section_5_on_its_rates(prereg_rates):
    ps = wh.plan_projections(wh.RECERTIFY, _manifest())
    for step, want in SECTION5.items():
        got = [(i["usd"], i["usd_upper"]) for i in ps[step].items]
        assert len(got) == len(want), step
        for (a, b), (x, y) in zip(got, want, strict=True):
            assert a == pytest.approx(x, abs=6e-4) and b == pytest.approx(y, abs=6e-4), step
    steps = wh.PLANS[wh.RECERTIFY]
    assert sum(ps[s].usd for s in steps) == pytest.approx(7.12, abs=0.005)
    assert sum(ps[s].usd_upper for s in steps) == pytest.approx(11.69, abs=0.005)
    assert ps["noise_all"].usd_upper == pytest.approx(4.166, abs=6e-4)  # f2 is inside it
    assert wh.plan_global_cap(wh.RECERTIFY, ps) == 11.70  # derived: 11.686 -> 11.70
    caps = wh.plan_caps(wh.RECERTIFY, ps)
    assert {s: caps[s] for s in steps} == {"noise_all": 4.20, "lab": 1.70, "balanced": 3.05,
                                          "caption_iso": 2.85, "certify": 0.0}
    st = wh.State("r", 0.0, "x", plan=wh.RECERTIFY)
    for s in steps:  # every step fits its cap and the global cap with the later ones kept
        assert wh.decide(st, wh.RECERTIFY, s, ps).runs, s


def test_global_cap_is_never_above_the_approved_11_70():
    ps = wh.plan_projections(wh.RECERTIFY, _manifest())
    steps = wh.PLANS[wh.RECERTIFY]
    up = sum(wh._cap_upper(ps[s]) for s in steps)
    g = wh.plan_global_cap(wh.RECERTIFY, ps)
    assert g <= wh.RECERT_GLOBAL_CAP_USD == 11.70
    assert g == min(11.70, round(-(-round(up, 6) * 10 // 1) / 10, 2))
    # the lab's reused TempSeg is not in its cap (by design); f2 is in the global sum
    assert ps["lab"].cap_usd_upper is None
    assert ps["lab"].usd_upper == pytest.approx(wh.LAB["captioner"].usd_per_s()
                                                * (wh.LAB_DEADLINE_S["captioner"]
                                                   + wh.LAB_SLACK_S), abs=1e-3)
    # only an approved override raises it
    assert wh.plan_global_cap(wh.RECERTIFY, ps, 12.0) == 12.0
    assert wh.check_global_cap(12.0, spent=0.0, derived=g) is None


def test_dry_run_shows_the_whole_sequence_with_caps_and_the_split(tmp_path, capsys):
    source, _ = _write_source(tmp_path)
    args, state = _main_args(tmp_path, source, "--dry-run", "--offline")
    assert wh.main(args) == 0
    out = capsys.readouterr().out
    assert "plan recertify (f2 -> d2' -> d' -> g -> e')" in out
    for title in ("f2. noise-12", "d2'. Captioner lab", "d'. balanced J1 (12 ep) + segment-only",
                  "g. caption-only isolated J2 (12 ep)", "e'. flora certify (v4)"):
        assert title in out, title
    assert "split: f2 noise_all nominal $2.74, upper $4.17; the rest (phase 2)" in out
    assert "global cap $11.70" in out
    assert "cold witness segment-only human (6 ep" in out
    assert "isolated caption cold witness human (6 ep" in out
    assert not state.exists()


def test_phase_2_picks_up_f2_as_recorded_and_fits_the_cap_at_its_actual_cost(tmp_path,
                                                                             monkeypatch):
    """After merge the recertify state on disk already has f2 done (``run_noise_all``'s record:
    ``noise_all.result.jobs.human``, on (b)'s run id): phase 2 runs against $11.70 minus what
    f2 cost, and the certify view takes f2's job as the noise run, (b)'s store."""
    source, job = _write_source(tmp_path)
    spec = _spec(tmp_path, source)
    calls: list = []
    _fakes(monkeypatch, calls)
    st = wh.new_recert_state(spec)
    res = wh.run_noise_all(_ctx(st, spec))  # what main's f2 records
    s = st.step("noise_all")
    s.status, s.result = "done", res
    a = s.attempts[-1]
    a.status, a.finished, a.tracked_usd = "done", 2.0, 2.74
    m = wh.offline_manifest(wh.DURATION_HEADROOM, wh.CERT_GOLDEN)
    projs = wh.plan_projections(wh.RECERTIFY, m, spec=spec)
    fresh = rb.State(RUN, 0.0, "x", plan=wh.RECERTIFY)
    assert wh.decide(fresh, wh.RECERTIFY, "noise_all", projs).action == "stop"  # $11.90 > $11.70
    assert st.spent == pytest.approx(2.74)
    for name in wh.PLANS[wh.RECERTIFY]:
        d = wh.decide(st, wh.RECERTIFY, name, projs)
        assert d.action == ("done" if name == "noise_all" else "run"), (name, d.message)
    rest = sum(projs[n].usd_upper for n in wh.PLANS[wh.RECERTIFY] if n != "noise_all")
    assert 2.74 + rest <= wh.RECERT_GLOBAL_CAP_USD
    view = wh.certify_view(st, spec)
    nz = view.steps["noise"].result["jobs"][wh.JOB]
    assert nz["job_id"] == f"wave-h-{RUN}-noise12-human-a1" and nz["state"] == "done"
    assert view.steps["exact_e2e"].result["jobs"][wh.JOB]["job_id"] == B_JOB
    assert view.steps["repeat"].result["jobs"][wh.JOB]["job_id"] == f"wave-h-{RUN}-noise-human"
    store = f"modal-volume://{wh.STORE_VOLUME}/{RUN}"
    assert {nz.get("store", wh.store_uri(view)),
            view.steps["exact_e2e"].result["jobs"][wh.JOB]["store"]} == {store}
    lines: list[str] = []
    wh.print_plan(st, projs, lines.append, manifest=m, plan=wh.RECERTIFY, spec=spec)
    text = "\n".join(lines)
    assert "fits: spent $2.74 + the upper of the steps not done" in text
    assert "NOT HONEST" not in text and "f2 check" not in text  # done: not re-checked
