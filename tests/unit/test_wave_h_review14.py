"""Review 14 for ``scripts/wave_h.py --plan certified``:

- M1: (d) is done only when its deployed job **and** its cold witness are done. A failed or
  missing witness re-runs the witness alone (the main job's records are reused), priced and
  named as its own attempt; d2 and e wait for it.
Nothing here contacts Modal."""

from __future__ import annotations

import os
import sys

import pytest

from tests.unit.test_wave_h_certified import SPEC, _manifest, wh


def _state(main: str = "done", witness: str | None = "failed", *, status: str = "failed",
           plan: str | None = wh.CERTIFIED, cfg: str = "flora-balanced-human.yaml"):
    st = wh.State(run_id="r", created=0.0, billing_since="x", plan=plan)
    s = st.step("balanced")
    s.status = status
    s.result = {"ok": False, "jobs": {wh.JOB: {"job_id": "wave-h-r-balanced-human",
                                               "state": main, "spend": 1.5, "cfg": cfg,
                                               "app": "main-app"}},
                "episodes": list(SPEC.episodes), "derived_content": {"k": "v"}}
    if witness is not None:
        s.result["witness_jobs"] = {wh.JOB: {"job_id": "wave-h-r-cold-human-a1",
                                             "state": witness, "spend": 0.3}}
    s.attempts.append(wh.Attempt(started=1.0, status="failed", finished=2.0, tracked_usd=1.8))
    return st


def test_failed_witness_is_not_adopted_as_done():  # the probe (review14/adopt.py)
    st = _state(plan=None)
    assert wh.witness_pending(st)
    assert wh.adopt_balanced_job(st) is False
    assert st.step("balanced").status == "failed"


def test_missing_witness_on_the_certified_plan_is_pending():
    st = _state(witness=None)
    assert wh.witness_pending(st) and not wh.adopt_balanced_job(st)
    v1 = _state(witness=None, plan="full")  # other plans have no witness: adopted as before
    assert not wh.witness_pending(v1) and wh.adopt_balanced_job(v1)


def test_main_and_witness_done_is_adopted():
    st = _state(witness="done")
    assert not wh.witness_pending(st)
    assert wh.adopt_balanced_job(st) and st.step("balanced").status == "done"


def test_failed_main_job_is_not_witness_pending():
    st = _state(main="failed")
    assert not wh.witness_pending(st) and not wh.adopt_balanced_job(st)


def _fake_jobs(monkeypatch, tmp_path, calls):
    def stage_job(ctx, job, ids, **kw):
        calls.append(("stage", job))
        cfg = tmp_path / f"flora-{job}.yaml"
        cfg.write_text(f"project: wave-h-r-{job}\npreset: balanced\n")
        return cfg, f"wave-h-r-{job}"

    def run_job(ctx, step, cfg, job_id, **kw):
        calls.append(("run", job_id, ctx.env.get("FLORA_COMPILE_CACHE"),
                      kw.get("spent_in_step")))
        return {"app": f"app-{job_id}", "job_id": job_id, "state": "done", "spend": 0.25}

    monkeypatch.setattr(wh, "_stage_job", stage_job)
    monkeypatch.setattr(wh, "_index", lambda ctx, cfg, job_id, tag, *, limit=None:
                        calls.append(("index", job_id, limit)))
    monkeypatch.setattr(wh, "_run_job", run_job)
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **kw: {})
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")
    monkeypatch.setattr(wh, "balanced_episodes", lambda m, spec, **kw: spec.episodes)
    import flora_deploy.config.receipts as R
    import flora_deploy.config.schema as S

    monkeypatch.setattr(S, "load_config", lambda p, **kw: None)
    monkeypatch.setattr(R, "derived_content", lambda cfg: {})
    monkeypatch.delenv("FLORA_COMPILE_CACHE", raising=False)


def test_resume_reruns_only_the_witness_and_reuses_the_main_job(tmp_path, monkeypatch):
    calls: list = []
    _fake_jobs(monkeypatch, tmp_path, calls)
    cfg = tmp_path / "flora-balanced-human.yaml"
    cfg.write_text("project: wave-h-r-balanced-human\npreset: balanced\n")
    st = _state(cfg=str(cfg))
    main = dict(st.step("balanced").result["jobs"][wh.JOB])
    st.step("balanced").attempts.append(wh.Attempt(started=3.0))  # the retry's attempt
    ctx = wh.Ctx(st, _manifest(), tmp_path, spec=SPEC, caps=dict(wh.CAPS_USD))
    res = wh.run_balanced(ctx)
    assert res["ok"], res
    assert [c for c in calls if c[0] in ("stage", "run")] == [
        ("run", "wave-h-r-cold-human-a2", "off", 0.0)]  # its own job and app; main not re-run
    assert ("index", "wave-h-r-cold-human-a2", len(SPEC.noise)) in calls
    assert res["jobs"][wh.JOB] == main  # reused, records and all
    assert res["witness_jobs"][wh.JOB]["state"] == "done"
    assert res["tracked_usd"] == pytest.approx(0.25)  # this attempt: the witness only
    assert res["apps"] == ["app-wave-h-r-cold-human-a2"]
    assert res["episodes"] == list(SPEC.episodes) and res["derived_content"] == {"k": "v"}
    assert "FLORA_COMPILE_CACHE" not in os.environ


def test_failed_main_job_reruns_both_as_today(tmp_path, monkeypatch):
    calls: list = []
    _fake_jobs(monkeypatch, tmp_path, calls)
    st = _state(main="failed")
    ctx = wh.Ctx(st, _manifest(), tmp_path, spec=SPEC, caps=dict(wh.CAPS_USD))
    res = wh.run_balanced(ctx)
    assert res["ok"]
    assert [c[1] for c in calls if c[0] == "run"] == ["wave-h-r-balanced-human",
                                                     "wave-h-r-cold-human-a1"]


def test_witness_only_retry_is_priced_as_the_witness(tmp_path):
    full = wh.project("balanced", _manifest(), SPEC)
    p = wh.project("balanced", _manifest(), SPEC, witness_only=True)
    assert p.items and all("cold witness" in i["what"] for i in p.items)
    assert p.usd_upper == pytest.approx(sum(i["usd_upper"] for i in full.items
                                            if "cold witness" in i["what"]))
    assert p.cap_usd_upper == pytest.approx(full.usd_upper)  # the global cap is unchanged
    pj = wh.plan_projections(wh.CERTIFIED, _manifest(), witness_only=True)
    base = wh.plan_projections(wh.CERTIFIED, _manifest())
    assert wh.plan_global_cap(wh.CERTIFIED, pj) == wh.plan_global_cap(wh.CERTIFIED, base)
    caps = wh.plan_caps(wh.CERTIFIED, pj)
    assert caps["balanced"] < wh.plan_caps(wh.CERTIFIED, base)["balanced"]
    assert caps["balanced"] >= p.usd_upper  # budget_for's attempt cap fits the witness
    d = wh.decide(_state(), wh.CERTIFIED, "balanced", pj)
    assert d.action == "run" and d.budget.usd >= p.usd_upper


def test_status_says_witness_pending_and_d2_e_wait(tmp_path):
    st = _state()
    pj = wh.plan_projections(wh.CERTIFIED, _manifest(), witness_only=True)
    for s in ("exact_e2e", "noise"):
        st.step(s).status = "done"
    lines: list[str] = []
    d = wh.print_status(st, pj, plan=wh.CERTIFIED, out=lines.append, spec=SPEC)
    assert d is not None and d.step == "balanced"  # not lab, not certify
    text = "\n".join(lines)
    assert f"witness pending: will re-run the {len(SPEC.noise)}-episode cold job only" in text


def test_lab_refuses_while_the_witness_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "lab_already_done", lambda *a: (False, ""))
    ctx = wh.Ctx(_state(), _manifest(), tmp_path, spec=SPEC, caps=dict(wh.CAPS_USD))
    with pytest.raises(RuntimeError, match="cold witness"):
        wh.run_lab(ctx, lab_module=object())


# ---------------------------------------------------------------------------------------------
# W4: rates.json without camera_scaling


def test_import_without_camera_scaling_prices_at_one_and_warns(monkeypatch, capsys):
    import dataclasses
    import importlib.util

    import flora_deploy.planner.rates as R

    real = R.RateTable.load
    monkeypatch.setattr(R.RateTable, "load", classmethod(
        lambda cls, *a, **k: dataclasses.replace(real(*a, **k), camera_scaling=None)))
    spec = importlib.util.spec_from_file_location("wave_h_noscale", wh.__file__)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "wave_h_noscale", mod)  # dataclasses look it up
    spec.loader.exec_module(mod)  # no AttributeError at import
    assert mod.CERT_SPECS["yam"].rate_scale == () and mod.NO_CAMERA_SCALING
    assert mod.main(["--plan", "certified", "--embodiment", "yam", "--dry-run",
                     "--offline"]) == 0
    out = capsys.readouterr().out
    assert "no camera_scaling" in out and "factor 1.0" in out
    assert not wh.NO_CAMERA_SCALING  # the shipped table has it
