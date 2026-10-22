"""``scripts/wave_h.py --plan certified`` offline: the golden/v2 episode sets, the noise subset
that ``flora index --limit`` selects, pairing uids, the derived caps and the "lab already done"
skip, and an end-to-end certify step on fake records shaped like the real ones that comes out
**certified** (not provisional) when the numbers agree. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.fakes.certrecords import BOUNDS, CONTENT, rec, segs

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
h0 = sys.modules["h0_rehearsal"]
SPEC = wh.SPECS[wh.CERTIFIED]
T0 = 1_792_000_000.0


def _manifest() -> dict:
    return wh.offline_manifest(0.0, wh.CERT_GOLDEN)


# ---------------------------------------------------------------------------------------------
# Plan shape


def test_certified_plan_runs_b_f_d_then_certify_on_golden_v2():
    assert wh.PLANS[wh.CERTIFIED] == ("exact_e2e", "noise", "balanced", "lab", "certify")
    assert SPEC.golden == "golden/v2" and SPEC.shared_data
    assert len(SPEC.episodes) >= 12 and len(set(SPEC.episodes)) == len(SPEC.episodes)
    assert len(SPEC.noise) == wh.MIN_NOISE_EPISODES + wh.NOISE_SPARES
    assert set(SPEC.noise) < set(SPEC.episodes)
    assert set(wh.EPISODES) <= set(SPEC.episodes)  # v1's six are kept
    assert SPEC.lab_episodes == wh.EPISODES
    assert SPEC.state != wh.STATE and SPEC.state.parent == wh.CERT_ROOT
    assert wh.balanced_episodes(_manifest(), SPEC) == SPEC.episodes


def test_noise_subset_is_exactly_what_index_limit_selects(tmp_path):
    """The noise job indexes the oracle's own data dir with ``--limit 5``: the first five
    episodes in ``mp4_dir`` order must be the noise subset (same config, same stage hash)."""
    from flora_deploy.media.sources.mp4_dir import Mp4DirSource

    for g in SPEC.episodes:
        (tmp_path / f"{SPEC.local_id(g)}.mp4").write_bytes(b"")
        (tmp_path / f"{SPEC.local_id(g)}.txt").write_text("t\n")
    eps = Mp4DirSource(tmp_path).list_episodes()
    assert len(eps) == len(SPEC.episodes)
    first = [e.episode_id for e in eps[:SPEC.noise_limit]]
    assert sorted(first) == sorted(SPEC.local_id(g) for g in SPEC.noise)
    assert SPEC.noise_limit == len(SPEC.noise)
    # v1 specs keep their ids and repeat every episode
    v1 = wh.SPECS["full"]
    assert v1.local_id(wh.EPISODES[0]) == h0.episode_id(wh.EPISODES[0]) and v1.noise_limit is None


def test_oracle_noise_and_balanced_share_one_data_dir_so_uids_pair(tmp_path):
    from flora_deploy.config.schema import load_config
    from flora_deploy.jobs.indexing import DatasetSpec
    from flora_deploy.jobs.types import episode_uid

    (tmp_path / wh.CERT_DATA).mkdir()
    names = {}
    for step, preset in (("exact_e2e", "exact"), ("balanced", "balanced")):
        assert SPEC.data_dir(step) == wh.CERT_DATA
        cfg = tmp_path / f"flora-{step}.yaml"
        cfg.write_text(h0.config_text(project=f"p-{step}", data_dir=SPEC.data_dir(step),
                                      embodiment="human", stages=("segment", "caption"),
                                      store_volume="s", media_volume="m", run_id="r",
                                      preset=preset))
        c = load_config(cfg, allow_uncertified=True)
        names[step] = DatasetSpec.from_config(c.input).name
    assert names["exact_e2e"] == names["balanced"]
    uid = episode_uid(names["balanced"], SPEC.local_id(SPEC.noise[0]))
    assert uid.startswith(f"{wh.CERT_DATA}--{wh.NOISE_PREFIX}")
    # full stages one dir per step: its uids would never pair (why certified exists)
    assert wh.SPECS["full"].data_dir("exact_e2e") != wh.SPECS["full"].data_dir("balanced")


def test_caps_are_each_steps_upper_and_the_global_cap_their_rounded_sum():
    ps = wh.plan_projections(wh.CERTIFIED, _manifest())
    caps = wh.plan_caps(wh.CERTIFIED, ps)
    for s in wh.PLANS[wh.CERTIFIED]:
        assert ps[s].usd_upper <= caps[s] < ps[s].usd_upper + 0.05 + 1e-9, s
    up = sum(ps[s].usd_upper for s in wh.PLANS[wh.CERTIFIED])
    g = wh.plan_global_cap(wh.CERTIFIED, ps)
    assert up <= g < up + 0.1 + 1e-9
    st = wh.State(run_id="r", created=T0, billing_since="2026-10-21")
    for s in wh.PLANS[wh.CERTIFIED]:  # no step is refused by its own stop rule
        wh.check_before(st, s, ps[s], caps=caps, global_cap=g)
    # (f) prices the 5 noise episodes only; the oracle all of them
    dur = wh.durations(_manifest())
    assert f"({len(SPEC.noise)} ep, {sum(dur[g] for g in SPEC.noise):.1f} s" in \
        ps["noise"].items[0]["what"]
    assert f"({len(SPEC.episodes)} ep" in ps["exact_e2e"].items[0]["what"]
    # with the lab already done, (d2) is $0 and the global cap drops by its upper price
    done = wh.plan_projections(wh.CERTIFIED, _manifest(), lab_done=True)
    assert done["lab"].usd_upper == 0.0
    assert wh.plan_global_cap(wh.CERTIFIED, done) < g


def test_too_few_episodes_or_noise_refuse_before_any_app():
    from dataclasses import replace

    with pytest.raises(RuntimeError, match="at least 10"):
        wh.balanced_episodes(_manifest(), replace(SPEC, episodes=SPEC.episodes[:9],
                                                  noise=SPEC.episodes[:5]))
    with pytest.raises(RuntimeError, match="noise subset"):
        wh.balanced_episodes(_manifest(), replace(SPEC, noise=SPEC.noise[:4]))


def test_noise_index_check(tmp_path):
    m = _manifest()
    dur = wh.durations(m)
    p = tmp_path / "summary.json"
    want = sum(dur[g] for g in SPEC.noise)
    p.write_text(json.dumps({"counts": {"episodes": len(SPEC.noise)}, "video_s": want}))
    assert wh.noise_index_problems(p, m, SPEC) == ""
    p.write_text(json.dumps({"counts": {"episodes": 12}, "video_s": want * 3}))
    assert f"not the {len(SPEC.noise)} noise episodes" in wh.noise_index_problems(p, m, SPEC)


# ---------------------------------------------------------------------------------------------
# Lab already done


def _lab(kind: str, gpu: str, ok: bool = True) -> dict:
    from flora_deploy.engines.offline_llm import normalize_gpu
    from tests.fakes.certrecords import lab_hashes

    return {"kind": kind, "baseline": {"gpu": gpu, "l0": [
        {"key": f"k{i}", "ids_equal": ok, "render_ids_equal": True, "grid_equal": True}
        for i in range(6)]},
        "gate_a": {"requests": 6, "greedy_identical": 6, "positions": 120,
                   "max_abs_dlogit": 0.1, "mean_abs_dlogit": 0.01, "first_divergence": []},
        "engine_hashes": lab_hashes(kind, normalize_gpu(gpu))}


def _labs(d: Path, ok: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "balanced-tempseg.json").write_text(json.dumps(_lab("tempseg", "NVIDIA L40S", ok)))
    (d / "balanced-captioner.json").write_text(
        json.dumps(_lab("captioner", "NVIDIA H100 80GB HBM3")))
    return d


def test_lab_already_done_needs_passing_l0_and_gate_a_per_kind(tmp_path, monkeypatch):
    assert wh.lab_already_done(tmp_path)[0] is False
    _labs(tmp_path, ok=False)
    done, why = wh.lab_already_done(tmp_path)
    assert not done and "tempseg: L0 not passing" in why
    _labs(tmp_path)
    assert wh.lab_already_done(tmp_path)[0] is True
    (tmp_path / "balanced-captioner.json").write_text(json.dumps({"error": "deadline"}))
    assert "captioner: no L0 rows" in wh.lab_already_done(tmp_path)[1]
    # the certified lab step skips at $0 without starting anything
    _labs(tmp_path)
    monkeypatch.setattr(wh, "LAB_OUT", tmp_path)
    ctx = wh.Ctx(wh.State(run_id="r", created=T0, billing_since="x"), _manifest(), tmp_path,
                 spec=SPEC)
    res = wh.run_lab(ctx, lab_module=object())
    assert res["ok"] and res["apps"] == [] and res["tracked_usd"] == 0.0
    assert res["skipped"].startswith("lab already done")


# ---------------------------------------------------------------------------------------------
# End to end: the certify step on records shaped like the real ones


def _uid(g: str) -> str:
    from flora_deploy.jobs.types import episode_uid

    return episode_uid(wh.CERT_DATA, SPEC.local_id(g))


def _records(ids, preset: str, stage: str, *, shift: float = 0.0, noise_ref: bool = False,
             **kw):
    out = []
    for g in ids:
        i = SPEC.episodes.index(g)  # captions follow the episode, not its place in the file
        caps = [f"c{i}-{j}" for j in range(len(BOUNDS))] if stage == "caption" else None
        r = rec(_uid(g), stage, preset, embodiment="human", fp=None, gpu=None,
                segments=segs([(s + shift, e + shift) for s, e in BOUNDS], caps),
                audit_ref=(f"noise-{i}" if noise_ref else None), **kw)
        out.append(r)
    return out


def _jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records))


def _job(step: str, cfg: Path | None) -> dict:
    return {"app": f"app-{step}", "job_id": f"wave-h-r-{step}", "state": "done", "spend": 0.5,
            "containers": [], "video_hours": {"done": 0.06}, "cfg": str(cfg) if cfg else None}


def certified_world(tmp_path: Path, *, noise_n: int | None = None, shift: float = 0.01,
                    witness: bool = True):
    """A certified-plan state after (b), (f), (d): the records under the plan's records dir
    (the cold witness's too, unless ``witness=False``), the balanced job's real-format config,
    the lab JSON."""
    from dataclasses import replace

    spec = replace(SPEC, state=tmp_path / "state.json", work=tmp_path / "work",
                   records=tmp_path / "records", report=tmp_path / "runbook.md")
    spec.work.mkdir(parents=True)
    (spec.work / wh.CERT_DATA).mkdir()
    cfg = spec.work / "flora-balanced-human.yaml"
    cfg.write_text(h0.config_text(project="p", data_dir=wh.CERT_DATA, embodiment="human",
                                  stages=("segment", "caption"), store_volume="s",
                                  media_volume="m", run_id="r", preset="balanced"))
    st = wh.State(run_id="r", created=T0, billing_since="2026-10-21", plan=wh.CERTIFIED)
    for i, step in enumerate(("exact_e2e", "noise", "balanced")):
        s = st.step(step)
        s.status = "done"
        s.result = {"jobs": {wh.JOB: _job(step, cfg if step == "balanced" else None)}}
        if step == "balanced":
            s.result["derived_content"] = dict(CONTENT)
            if witness:
                s.result["witness_jobs"] = {wh.JOB: {**_job("cold", cfg),
                                                     "compile_cache": "off"}}
        s.attempts.append(wh.Attempt(started=T0 + i * 1000, status="done",
                                     finished=T0 + i * 1000 + 900, tracked_usd=0.5))
    wh.save_state(st, spec.state)
    noise = SPEC.noise[:noise_n]
    for stage in wh.CERT_STAGES:
        if witness:
            _jsonl(spec.records / "witness" / wh.JOB / f"{stage}.jsonl",
                   _records(SPEC.noise, "balanced", stage, shift=shift,
                            compile_cache="cold:forced"))
        _jsonl(spec.records / "exact" / wh.JOB / f"{stage}.jsonl",
               _records(SPEC.episodes, "exact", stage))
        _jsonl(spec.records / "noise" / wh.JOB / f"{stage}.jsonl",
               _records(noise, "exact", stage, noise_ref=True))
        _jsonl(spec.records / "balanced" / wh.JOB / f"{stage}.jsonl",
               _records(SPEC.episodes, "balanced", stage, shift=shift))
    labs = _labs(tmp_path / "lab")
    ctx = wh.Ctx(st, _manifest(), spec.work, state_path=spec.state, spec=spec,
                 caps=wh.plan_caps(wh.CERTIFIED, wh.plan_projections(wh.CERTIFIED,
                                                                    _manifest())))
    return ctx, labs


class Sh:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append(args)
        return type("P", (), {"returncode": 0})()


@pytest.fixture
def no_fetch(monkeypatch):
    monkeypatch.setattr(fin, "fetch_records", lambda paths, st, run=None: None)
    monkeypatch.setitem(sys.modules, "wave_h_finalize", fin)  # what run_flora_certify imports


def test_certify_step_comes_out_certified_when_the_numbers_agree(tmp_path, no_fetch):
    ctx, labs = certified_world(tmp_path)
    sh = Sh()
    res = wh.run_flora_certify(ctx, finalize=fin, sh=sh, lab_dir=labs)
    assert res["ok"], res
    for stage in wh.CERT_STAGES:
        v = res["stages"][stage]
        assert v["verdict"] == "certified" and v["reasons"] == [], v
        assert v["episodes"] == len(SPEC.episodes) and v["noise_episodes"] == len(SPEC.noise)
    certs = [json.loads(p.read_text()) for p in (ctx.work / "certify").glob("cert_*.json")]
    assert {c["stage"] for c in certs} == set(wh.CERT_STAGES)
    for c in certs:
        assert c["unlocks"] and c["covered_pools"] == ["human"]
        assert c["oracle"]["determinism"] == "nvidia" and c["oracle"]["gpu"] == "H200"
        assert c["metrics"]["coverage"] == 1.0
    # then wave_h_finalize.py --write on this state and records
    assert len(sh.calls) == 1
    args = sh.calls[0]
    assert args[1].endswith("wave_h_finalize.py") and "--write" in args
    assert args[args.index("--state") + 1] == str(ctx.state_path)
    assert args[args.index("--records") + 1] == str(ctx.spec.records)


def test_four_noise_episodes_are_provisional_and_write_nothing(tmp_path, no_fetch):
    ctx, labs = certified_world(tmp_path, noise_n=4)
    sh = Sh()
    res = wh.run_flora_certify(ctx, finalize=fin, sh=sh, lab_dir=labs)
    assert not res["ok"] and sh.calls == []
    assert {v["verdict"] for v in res["stages"].values()} == {"provisional"}
    assert all("too_few_noise_samples" in v["reasons"] for v in res["stages"].values())
    assert "at least 5 noise-run episodes" in res["error"]


def test_a_divergent_candidate_fails_and_writes_nothing(tmp_path, no_fetch):
    ctx, labs = certified_world(tmp_path, shift=0.6)  # boundary error 0.6 s > 1.5 x 0.1 s
    sh = Sh()
    res = wh.run_flora_certify(ctx, finalize=fin, sh=sh, lab_dir=labs)
    assert not res["ok"] and sh.calls == []
    assert res["stages"]["segment"]["verdict"] == "fail"


def test_main_runs_the_certified_certify_step_and_finalize_reads_its_plan(
        tmp_path, no_fetch, monkeypatch):
    ctx, labs = certified_world(tmp_path)
    monkeypatch.setattr(wh, "LAB_OUT", labs)
    monkeypatch.setattr(wh, "billed_by_app", lambda since: {})
    sh = Sh()
    monkeypatch.setattr(wh.Ctx, "sh", lambda self, args, **kw: sh(args, **kw))
    monkeypatch.setattr(h0, "fetch_dataset", lambda *a, **k: pytest.fail("no staging"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wh, "load_manifest", lambda offline, version: _manifest())
    monkeypatch.setitem(wh.SPECS, wh.CERTIFIED, ctx.spec)  # tmp state, work, records
    rc = wh.main(["--plan", "certified", "--yes-spend", "--only", "certify"])
    st = wh.load_state(ctx.state_path)
    assert rc == 0, st.steps["certify"].notes
    assert st.plan == wh.CERTIFIED and st.steps["certify"].status == "done"
    assert {v["verdict"] for v in st.steps["certify"].result["stages"].values()} == {
        "certified"}
    assert fin.run_plan(st) == wh.CERTIFIED
    assert any("wave_h_finalize.py" in str(a) for a in sh.calls[-1])
    # a v1 plan never runs on the certified state
    assert wh.main(["--plan", "full", "--dry-run", "--offline", "--state",
                    str(ctx.state_path)]) == 2


def test_certified_balanced_step_prices_cold_from_the_merged_rate_table(capsys):
    """(d) is priced on the measured balanced rows and the measured cold startups, never the
    warm compile-cache placeholders, and runs the batch-fill engine limits."""
    from flora_deploy.engines.batch_policy import policy_for
    from flora_deploy.engines.offline_llm import GPU_LIMITS
    from flora_deploy.planner.rates import RateTable

    t = RateTable.load()
    for kind in ("segment", "caption"):
        assert t.rate("balanced", kind, 1, 1).status == "measured"
        assert t.startup_for("balanced", kind).compile_cache == "cold"
        assert t.startup_for("balanced", kind, warm=True).compile_cache == "warm"
    secs = wh.balanced_job_seconds(100.0, t)
    assert secs["segment"][0] > wh.balanced_job_seconds(100.0, t, warm_kinds=("segment",))[
        "segment"][0]
    lim = wh.balanced_engine_limits()
    assert lim["segment"]["max_num_seqs"] == GPU_LIMITS[("tempseg", "L40S")].max_num_seqs
    assert lim["caption"]["max_num_seqs"] == GPU_LIMITS[("captioner", "H100")].max_num_seqs
    assert lim["segment"]["infer_episodes"] == policy_for("segment").infer_episodes
    man = wh.offline_manifest(wh.DURATION_HEADROOM, wh.SPECS[wh.CERTIFIED].golden)
    st = wh.State("r", 0.0, "x")
    wh.print_plan(st, wh.plan_projections(wh.CERTIFIED, man), manifest=man, plan=wh.CERTIFIED)
    out = capsys.readouterr().out
    assert "batch-fill engine limits" in out and "max_num_seqs 48" in out
    assert "never write it back" not in out
