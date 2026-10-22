"""Review 13 for ``scripts/wave_h.py --plan certified`` and ``scripts/wave_h_finalize.py``:

- S3-1: lab evidence counts (reuse, "already done", the certificate's L0) only when it was
  measured under the current engine settings (``certkey.engine_settings_hash``), so the
  certified plan runs and prices the lab;
- S3-3: the noise subset has one spare, all of it in the oracle's episodes;
- W1: (d) adds a cold witness job on the noise subset with ``FLORA_COMPILE_CACHE=off``, and
  (e) needs it; the finalize step reports preemptions and the compile-cache mix.
Nothing here contacts Modal."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import pytest

from tests.unit.test_wave_h_certified import (
    SPEC,
    Sh,
    _jsonl,
    _lab,
    _labs,
    _manifest,
    _records,
    certified_world,
    fin,
    wh,
)


def _old_hash(kind: str) -> str:
    """The engine-settings hash of the lab files measured before batch fill (32 / 16)."""
    from flora_deploy.engines.offline_llm import EngineSpec
    from flora_deploy.validate import certkey

    h = EngineSpec.for_gpu(kind, "", wh.LAB[kind].gpu).hashed()
    h["engine_kwargs"]["max_num_seqs"] = {"tempseg": 32, "captioner": 16}[kind] + 1
    return certkey.engine_settings_hash(h)


# ---------------------------------------------------------------------------------------------
# S3-1: stale lab evidence


def test_lab_engine_hashes_are_the_current_engines():
    from flora_deploy.engines.offline_llm import EngineSpec
    from flora_deploy.validate import certkey

    want = certkey.engine_settings_hash(EngineSpec.for_gpu("tempseg", "", "L40S").hashed())
    assert wh.lab_engine_hashes("tempseg")["baseline"] == want
    assert set(wh.lab_engine_hashes("captioner")) == {"baseline", "plugin"}


def test_lab_reuse_needs_the_current_engine_hash(tmp_path):
    from tests.unit.test_wave_h_lab import _saved

    p = tmp_path / "balanced-tempseg.json"

    def reused(res):
        p.write_text(json.dumps(res))
        return wh.reusable_lab_result("tempseg", "r", wh.EPISODES, path=p) is not None

    assert reused(_saved("tempseg", engine_hashes=wh.lab_engine_hashes("tempseg")))
    assert not reused(_saved("tempseg", engine_hashes=None))  # before the hash: never reused
    stale = {**wh.lab_engine_hashes("tempseg"), "baseline": _old_hash("tempseg")}
    assert not reused(_saved("tempseg", engine_hashes=stale))
    plugin = {**wh.lab_engine_hashes("tempseg"), "plugin": "0" * 64}
    assert not reused(_saved("tempseg", engine_hashes=plugin, dtbias={"spec": {}}))


def test_lab_already_done_needs_the_current_engine_hash(tmp_path):
    _labs(tmp_path)
    assert wh.lab_already_done(tmp_path)[0] is True
    old = _lab("tempseg", "NVIDIA L40S")
    old.pop("engine_hashes")
    (tmp_path / "balanced-tempseg.json").write_text(json.dumps(old))
    done, why = wh.lab_already_done(tmp_path)
    assert not done and "tempseg: measured under other engine settings" in why
    old["engine_hashes"] = {"baseline": _old_hash("tempseg")}
    (tmp_path / "balanced-tempseg.json").write_text(json.dumps(old))
    assert not wh.lab_already_done(tmp_path)[0]


def test_the_certified_dry_run_runs_and_prices_the_lab_on_stale_evidence(
        tmp_path, monkeypatch, capsys):
    """The lab files from before batch fill carry no engine hash: the certified plan runs the
    lab under the current settings and its dry run prices it."""
    stale = tmp_path / "lab"
    _labs(stale)
    for kind in ("tempseg", "captioner"):
        p = stale / f"balanced-{kind}.json"
        d = json.loads(p.read_text())
        d.pop("engine_hashes")
        p.write_text(json.dumps(d))
    monkeypatch.setattr(wh, "LAB_OUT", stale)
    monkeypatch.setitem(wh.SPECS, wh.CERTIFIED, replace(
        SPEC, state=tmp_path / "s.json", work=tmp_path / "w", records=tmp_path / "r",
        report=tmp_path / "rb.md"))
    assert wh.main(["--plan", "certified", "--dry-run", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "lab (d2): runs" in out and "measured under other engine settings" in out
    ps = wh.plan_projections(wh.CERTIFIED, _manifest(), lab_done=wh.lab_already_done()[0])
    assert ps["lab"].usd > 1.0 and ps["lab"].usd_upper > ps["lab"].usd


# ---------------------------------------------------------------------------------------------
# S3-3: the noise subset and its spare


def test_noise_subset_has_one_spare_from_the_oracle_episodes():
    sel = wh.golden_selection(wh.CERT_GOLDEN)["noise_subset"]
    assert SPEC.noise[:len(sel)] == tuple(sel)
    assert len(SPEC.noise) == wh.MIN_NOISE_EPISODES + wh.NOISE_SPARES == len(sel) + 1
    spare = SPEC.noise[-1]
    assert spare in SPEC.episodes and spare not in sel
    assert set(SPEC.noise) <= set(SPEC.episodes) and SPEC.noise_limit == len(SPEC.noise)


@pytest.mark.parametrize("drop", ["exact_noise_episode", "exact_other", "noise", "balanced",
                                  "witness"])
def test_certify_step_survives_one_missing_episode_at_each_stage(tmp_path, monkeypatch, drop):
    """One episode missing from the oracle (a noise-subset one, or another), the noise run, the
    balanced run or the cold witness: the spares fill in and the step is still certified."""
    monkeypatch.setattr(fin, "fetch_records", lambda paths, st, run=None: None)
    ctx, labs = certified_world(tmp_path)
    root = ctx.spec.records
    eps, noise = list(SPEC.episodes), list(SPEC.noise)
    sets = {"exact": eps, "noise": noise, "balanced": eps, "witness": noise}
    gone = {"exact_noise_episode": ("exact", noise[0]),
            "exact_other": ("exact", next(g for g in eps if g not in noise)),
            "noise": ("noise", noise[2]), "balanced": ("balanced", eps[-1]),
            "witness": ("witness", noise[1])}[drop]
    side, g = gone
    keep = [x for x in sets[side] if x != g]
    for stage in wh.CERT_STAGES:
        kw = {"exact": {}, "noise": {"noise_ref": True}, "balanced": {"shift": 0.01},
              "witness": {"shift": 0.01, "compile_cache": "cold:forced"}}[side]
        _jsonl(root / side / wh.JOB / f"{stage}.jsonl",
               _records(keep, "balanced" if side in ("balanced", "witness") else "exact",
                        stage, **kw))
    res = wh.run_flora_certify(ctx, finalize=fin, sh=Sh(), lab_dir=labs)
    assert res["ok"], res
    for stage, v in res["stages"].items():
        assert v["verdict"] == "certified", (stage, v)
        assert v["noise_episodes"] >= wh.MIN_NOISE_EPISODES


# ---------------------------------------------------------------------------------------------
# W1: the cold witness


@pytest.mark.parametrize("emb", ["human", "yam"])
def test_balanced_step_runs_the_cold_witness_on_the_noise_subset(tmp_path, monkeypatch, emb):
    calls = []
    spec = SPEC if emb == "human" else replace(
        wh.CERT_SPECS[emb], state=tmp_path / "s.json", work=tmp_path, records=tmp_path / "r",
        report=tmp_path / "rb.md")

    def stage_job(ctx, job, ids, **kw):
        calls.append(("stage", job, tuple(ids)))
        cfg = tmp_path / f"{job}.yaml"
        cfg.write_text(f"# header\nproject: wave-h-r-{job}\npreset: balanced\n")
        return cfg, f"wave-h-r-{job}"

    def index(ctx, cfg, job_id, tag, *, limit=None):
        calls.append(("index", job_id, limit))

    def run_job(ctx, step, cfg, job_id, **kw):
        calls.append(("run", job_id, os.environ.get("FLORA_COMPILE_CACHE")))
        return {"app": "a", "job_id": job_id, "state": "done", "spend": 0.25, "cfg": str(cfg)}

    monkeypatch.setattr(wh, "_stage_job", stage_job)
    monkeypatch.setattr(wh, "_index", index)
    monkeypatch.setattr(wh, "_run_job", run_job)
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **kw: {})
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")
    import flora_deploy.config.receipts as R
    import flora_deploy.config.schema as S

    monkeypatch.setattr(S, "load_config", lambda p, **kw: None)
    monkeypatch.setattr(R, "derived_content", lambda cfg: {})
    monkeypatch.delenv("FLORA_COMPILE_CACHE", raising=False)
    monkeypatch.setattr(wh, "balanced_episodes", lambda m, spec, **kw: spec.episodes)
    ctx = wh.Ctx(wh.State(run_id="r", created=0.0, billing_since="x"), _manifest(), tmp_path,
                 spec=spec, caps=dict(wh.CAPS_USD))
    res = wh.run_balanced(ctx)
    assert res["ok"], res
    main = f"wave-h-r-balanced-{emb}"
    cold = f"wave-h-r-cold-{emb}-a1"  # this attempt's own job and app
    assert [c for c in calls if c[0] == "run"] == [
        ("run", main, None), ("run", cold, "off")]
    assert ("index", cold, len(spec.noise)) in calls
    wcfg = (tmp_path / f"flora-cold-{emb}.yaml").read_text()
    assert f"project: {cold}\n" in wcfg and "preset: balanced" in wcfg
    assert "FLORA_COMPILE_CACHE" not in os.environ  # restored
    assert set(res["witness_jobs"]) == {wh.JOB} and res["tracked_usd"] == pytest.approx(0.5)
    assert res["witness_jobs"][wh.JOB]["compile_cache"] == "off"


def test_balanced_projection_prices_the_cold_witness():
    p = wh.project("balanced", _manifest(), SPEC)
    whats = [i["what"] for i in p.items]
    assert any("cold witness segment" in w and f"({len(SPEC.noise)} ep" in w for w in whats)
    assert any("cold witness caption" in w for w in whats)
    v1 = wh.project("balanced", wh.offline_manifest(0.0), wh.SPECS["full"])
    assert not any("cold witness" in i["what"] for i in v1.items)


def test_certify_step_needs_the_cold_witness_and_reports_engine_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(fin, "fetch_records", lambda paths, st, run=None: None)
    ctx, labs = certified_world(tmp_path)
    res = wh.run_flora_certify(ctx, finalize=fin, sh=Sh(), lab_dir=labs)
    assert res["ok"], res
    for v in res["stages"].values():
        assert v["witness_episodes"] == len(SPEC.noise)
        assert v["preemptions"] == {"preempted": 0, "unknown": 0}
        assert v["compile_cache"] == ["cold", "cold:forced"]
    ctx, labs = certified_world(tmp_path / "nowit", witness=False)
    res = wh.run_flora_certify(ctx, finalize=fin, sh=Sh(), lab_dir=labs)
    assert not res["ok"] and "cold witness" in res["error"]


def test_finalize_fetches_and_passes_the_witness(tmp_path, monkeypatch):
    ctx, labs = certified_world(tmp_path)
    paths = fin.Paths(root=wh.ROOT, state=ctx.state_path, records=ctx.spec.records, lab=labs)
    ran = []
    monkeypatch.setitem(sys.modules, "h0_rehearsal", wh.h0)
    fin.fetch_records(paths, ctx.st, run=lambda args, check: ran.append(args))
    outs = {a[a.index("--job") + 1]: a[a.index("--out") + 1] for a in ran}
    assert outs["wave-h-r-cold"].endswith(f"witness/{wh.JOB}/caption.jsonl")
    seen = []

    def cli(args):
        seen.append(args)
        return 2, ""

    fin.certify_all(paths, ctx.st, tmp_path / "c", derived={}, cli=cli)
    assert seen and all("--witness" in a for a in seen)
    for a in seen:
        stage = a[a.index("--stage") + 1]
        assert a[a.index("--witness") + 1].endswith(f"witness/{wh.JOB}/{stage}.jsonl")


def test_finalize_prints_preemptions_and_compile_caches():
    cert = {"stage": "segment", "preemptions": {"preempted": 1, "unknown": 0},
            "compile_cache": {"covers": ["cold", "cold:forced"]}}
    line = fin.engine_facts_line(cert)
    assert "1 preempted" in line and "cold:forced" in line
