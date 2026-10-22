"""``scripts/wave_h_finalize.py`` on a synthetic Wave H-lite state + records: rates, certificates,
report and docs blocks; refusals (too few samples), a missing L0, billing lag / mismatch, and
idempotence. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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

SECRET = "SECRET-NV-CAPTION"  # stands for NVIDIA text: must never leave the records
T0 = 1_792_000_000.0  # 2026-10-14 (UTC), the synthetic run's start
JOBS = {"human": 2, "yam": 4}  # episodes per job, as in the runbook
VIDEO = {"human": 30.7, "yam": 45.8}
BUSY = {"exact": {"human": 6.4, "yam": 6.6}, "segment": {"human": 0.5, "yam": 0.6},
        "caption": {"human": 1.4, "yam": 1.0}}
STARTUP = {"exact": (80.0, 90.0), "segment": (110.0, 130.0), "caption": (140.0, 150.0)}
GPU = {"exact": "H200", "segment": "L40S", "caption": "H100!"}
DERIVED = [a for n, h in CONTENT.items() for a in ("--derived-content", f"{n}={h}")]


def _container(kind: str, job: str, i: int, preset: str) -> dict:
    key = "exact" if preset == "exact" else kind
    busy = BUSY[key][job] * VIDEO[job]
    start = STARTUP[key][i]
    c = wh.pool_container(kind, preset)
    secs = start + busy + 30.0
    return {"container_id": f"ta-{preset}-{kind}-{job}", "gpu": GPU[key], "gpu_count": 1,
            "usd": round(c.usd_per_s() * secs, 4), "stopped": True, "kind": kind,
            "seconds": secs, "startup_s": start, "busy_s": busy}


def _job(preset: str, job: str, i: int, cfg: Path | None, *, step: str) -> dict:
    kinds = ("segcap",) if preset == "exact" else ("segment", "caption")
    cs = [_container(k, job, i, preset) for k in kinds]
    return {"app": f"flora-{preset}-{job}", "job_id": f"wave-h-r1-{step}-{job}",
            "state": "done", "spend": round(sum(c["usd"] for c in cs), 4), "containers": cs,
            "video_hours": {"planned": round(VIDEO[job] / 3600, 6),
                            "done": round(VIDEO[job] / 3600, 6)},
            "cfg": str(cfg) if cfg else None}


def _attempt(t: float, usd: float, apps: list[str]) -> wh.Attempt:
    return wh.Attempt(started=t, status="done", finished=t + 900, heartbeat=t + 900,
                      tracked_usd=usd, apps=apps, budget_usd=1.0)


def _l0_rows(ok: bool = True) -> list[dict]:
    return [{"key": f"k{i}", "ids_equal": ok, "render_ids_equal": True, "grid_equal": True}
            for i in range(6)]


def _lab(kind: str, gpu: str, dlogit: float) -> dict:
    from flora_deploy.engines.offline_llm import normalize_gpu
    from tests.fakes.certrecords import lab_hashes

    return {"run": "wave-h-r1", "kind": kind, "baseline": {
        "gpu": gpu, "l0": _l0_rows(), "episodes": [{"id": "x", "ok": True,
                                                    "rows": [{"caption": SECRET}]}]},
        "gate_a": {"requests": 6, "greedy_identical": 6, "positions": 120,
                   "max_abs_dlogit": dlogit, "mean_abs_dlogit": dlogit / 10,
                   "first_divergence": []},
        "engine_hashes": lab_hashes(kind, normalize_gpu(gpu))}


def _records(n: int, preset: str, stage: str, emb: str, *, shift: float = 0.0) -> list:
    out = []
    for i in range(n):
        caps = [f"{SECRET} {i} {j}" for j in range(len(BOUNDS))] if stage == "caption" else None
        out.append(rec(f"{emb}-u{i}", stage, preset, embodiment=emb, fp=None,
                       segments=segs([(s + shift, e + shift) for s, e in BOUNDS], caps)))
    return out


def _jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records))


def build(tmp: Path, *, jobs=("human", "yam"), labs=("tempseg", "captioner"),
          billed_lag_s: float = 7200.0, billed: dict | None = None) -> Path:
    """A repo root with the files the script owns, a runbook state, records and lab JSON."""
    for rel in ("src/flora_deploy/planner/rates.json", "docs/balanced-engines.md", "README.md"):
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp / rel)
    w = tmp / "reports" / "wave-h"
    (w / "work").mkdir(parents=True)
    cfgs = {}
    for job in JOBS:
        d = {"project": f"wave-h-r1-balanced-{job}", "license_ack": "research-only",
             "modal": {"max_gpus": 4, "budget_usd": 4}, "preset": "balanced",
             "allow_uncertified": True,
             "input": {"format": "lerobot_v3", "repo": "org/ds",
                       "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"},
             "pipeline": {"stages": ["segment", "caption"], "embodiment": job},
             "output": {"sink": "s3://b/p"}}
        cfgs[job] = w / "work" / f"flora-balanced-{job}.yaml"
        cfgs[job].write_text(yaml.safe_dump(d))
    st = wh.State(run_id="r1", created=T0, billing_since="2026-10-13")
    t = T0
    for name in (n for n in wh.STEPS if n != "lab"):  # the 2026-10-22 shape: lab inside (d)
        s = st.step(name)
        s.status = "done"
        if name == "exact_e2e":
            s.result = {"jobs": {j: _job("exact", j, i, None, step=name)
                                 for i, j in enumerate(JOBS)}}
        elif name == "balanced":
            s.result = {"jobs": {j: _job("balanced", j, i, cfgs[j], step=name)
                                 for i, j in enumerate(JOBS) if j in jobs},
                        "kinds": {k: {"wall_s": 400.0} for k in ("tempseg", "captioner")}}
        elif name == "noise":
            s.result = {"jobs": {"human": _job("exact", "human", 1, None, step=name)}}
        spend = sum(float(r["spend"]) for r in s.result.get("jobs", {}).values())
        apps = sorted({r["app"] for r in s.result.get("jobs", {}).values()})
        if name == "balanced":
            spend += 0.9
            apps.append("flora-balanced-lab")
        s.attempts.append(_attempt(t, round(spend, 4), apps))
        t += 1000
    exact_done = st.steps["exact_e2e"].attempts[-1].finished
    for name, s in st.steps.items():
        if name in ("exact_e2e", "tempseg"):
            s.billed_usd = (billed or {}).get(name, s.tracked_usd)
    st.billed_total = round(sum(s.tracked_usd for s in st.steps.values()), 4)
    st.billed_at = exact_done + billed_lag_s  # the lag as seen by step (b)
    wh.save_state(st, w / "state.json")
    r = w / "records"
    for job, n in JOBS.items():
        for stage in ("segment", "caption"):
            _jsonl(r / "exact" / job / f"{stage}.jsonl", _records(n, "exact", stage, job))
            if job in jobs:
                _jsonl(r / "balanced" / job / f"{stage}.jsonl",
                       _records(n, "balanced", stage, job, shift=0.01))
    for stage in ("segment", "caption"):
        _jsonl(r / "noise" / "human" / f"{stage}.jsonl", _records(2, "exact", stage, "human"))
    if "tempseg" in labs:
        (w / "balanced-tempseg.json").write_text(json.dumps(_lab("tempseg", "NVIDIA L40S", 0.12)))
    if "captioner" in labs:
        (w / "balanced-captioner.json").write_text(
            json.dumps(_lab("captioner", "NVIDIA H100 80GB HBM3", 0.31)))
    return tmp


def run(root: Path, *extra: str, capsys=None) -> tuple[int, str]:
    rc = fin.main(["--root", str(root), *DERIVED, *extra])
    return rc, (capsys.readouterr().out if capsys else "")


INPUTS = ("records", "work", "state.json", "balanced-tempseg.json", "balanced-captioner.json")


def owned(root: Path) -> dict[str, bytes]:
    """Every file the script may write (the runbook's inputs left out)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not set(INPUTS) & set(p.parts):
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def certs(root: Path) -> list[dict]:
    d = root / "src" / "flora_deploy" / "config" / "certificates"
    return [json.loads(p.read_text()) for p in sorted(d.glob("cert_*.json"))]


# ---------------------------------------------------------------------------------------------


def test_happy_path_updates_everything_consistently_and_is_idempotent(tmp_path, capsys):
    root = build(tmp_path)
    before = owned(root)
    rc, out = run(root, capsys=capsys)  # dry run
    assert rc == 0 and "would update" in out and "dry run: nothing written" in out
    assert owned(root) == before

    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    # 1. rates: the balanced slot measured, loadable, with per-kind startup rows
    from flora_deploy.planner.rates import RateTable

    t = RateTable.load(root / "src/flora_deploy/planner/rates.json")
    seg, cap = t.rate("balanced", "segment", 1), t.rate("balanced", "caption", 3)
    want_seg = round((0.5 * 30.7 + 0.6 * 45.8) / 76.5, 3)
    want_cap = round((1.4 * 30.7 + 1.0 * 45.8) / 76.5, 3)
    assert (seg.status, seg.container_s_per_video_s) == ("measured", want_seg)
    assert (cap.status, cap.container_s_per_video_s) == ("measured", want_cap)
    assert seg.uncertainty == 0.25  # range/mean 0.18 is below the floor
    assert cap.uncertainty == round((1.4 - 1.0) / 1.2, 2)  # the range: above the floor
    assert seg.source.startswith("docs/wave-h-report.md") and seg.date == "2026-10-14"
    assert "n=2 job(s) (76.5 video s) on L40S" in seg.provenance and "0.500-0.600" in seg.provenance
    assert t.startup_for("balanced", "segment").startup_s == 120.0
    assert t.startup_for("balanced", "caption").startup_s == 145.0
    assert t.startup_for("balanced", "caption").status == "measured"
    exact_row = t.rate("exact", "segcap", 3, 4)
    assert exact_row.container_s_per_video_s == 6.4  # exact rows are not touched

    # 2. certificates: per (stage, embodiment), provisional, metrics only
    cs = certs(root)
    assert sorted((c["stage"], c["covered_pools"][0]) for c in cs) == [
        ("caption", "human"), ("caption", "yam"), ("segment", "human"), ("segment", "yam")]
    assert {c["verdict"] for c in cs} == {"provisional"}
    by = {(c["stage"], c["covered_pools"][0]): c for c in cs}
    assert set(by[("segment", "human")]["reasons"]) == {"too_few_noise_samples",
                                                        "too_few_samples"}
    assert "no_noise_run" in by[("segment", "yam")]["reasons"]
    assert by[("segment", "yam")]["l0"]["status"] == "pass"
    assert by[("caption", "yam")]["gpu"] == "H100!"
    assert by[("segment", "human")]["gate_a"]["max_abs_dlogit"] == 0.12
    assert {c["generated_at"] for c in cs} == {wh_stamp(root)}
    assert "certificate verdict: provisional" in out
    assert "at least 10 candidate episodes of the pool (has 2)" in out
    assert "at least 5 noise-run episodes (has 2)" in out

    # 3. report
    rep = (root / "docs/wave-h-report.md").read_text()
    for head in ("## Spend per step", "## Measured rates", "## $/video-hour at list prices",
                 "## L0", "## Gate A", "## Certificate"):
        assert head in rep
    seg_h = wh.pool_container("segment", "balanced").usd_per_h()
    cap_h = wh.pool_container("caption", "balanced").usd_per_h()
    # from the unrounded rates, rounded once (review 12 F9: not a sum of per-term cents)
    exact_seg, exact_cap = (0.5 * 30.7 + 0.6 * 45.8) / 76.5, (1.4 * 30.7 + 1.0 * 45.8) / 76.5
    bal_cost = round(exact_seg * seg_h + exact_cap * cap_h, 2)
    assert f"**${bal_cost:,.2f}**" in rep
    assert "3600 x sum over the preset's kinds" in rep
    assert "**Recommendation: plugin off**" in rep and "0.31 is within the reference 0.5" in rep
    assert "Overall: **provisional**" in rep and "reconciled" in rep
    assert "| tempseg | pass | 6 | 6 | 6 | 6 |" in rep

    # 4. docs blocks
    doc = (root / "docs/balanced-engines.md").read_text()
    block = doc.split("<!-- wave-h:results")[1].split("<!-- /wave-h:results -->")[0]
    assert f"${bal_cost:.2f}/video-h" in block and "plugin **off**" in block
    assert "tempseg: pass (6/6 ids, 6/6 grid)" in block and "not run" not in block
    assert doc.count("<!-- wave-h:results") == 1
    readme = (root / "README.md").read_text()
    cost = readme.split("<!-- wave-h:cost -->")[1].split("<!-- /wave-h:cost -->")[0]
    assert f"${bal_cost:.2f} per video-hour" in cost and "#" not in cost
    assert 2 <= len(cost.strip().splitlines()) <= 3
    readme0 = (REPO / "README.md").read_text()
    outside = readme0.split("<!-- wave-h:cost -->")
    assert readme.startswith(outside[0]) and readme.endswith(outside[1].split(
        "<!-- /wave-h:cost -->")[1])  # only the block changed
    assert readme.count("## Cost") == readme0.count("## Cost") == 1

    # no NVIDIA text anywhere we wrote
    for name, data in owned(root).items():
        assert SECRET.encode() not in data, name

    # idempotent: a second run changes nothing
    after = owned(root)
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0 and "nothing to change" in out
    assert owned(root) == after


def wh_stamp(root: Path) -> str:
    st = wh.State.from_json(json.loads((root / "reports/wave-h/state.json").read_text()))
    return fin.utc(fin.run_stamp(st))


def test_too_little_video_refuses_the_rates_and_keeps_the_placeholders(tmp_path, capsys):
    root = build(tmp_path, jobs=("human",))
    st = wh.State.from_json(json.loads((root / "reports/wave-h/state.json").read_text()))
    st.steps["balanced"].result["jobs"]["human"]["video_hours"]["done"] = 20.0 / 3600
    wh.save_state(st, root / "reports/wave-h/state.json")
    rates0 = (root / "src/flora_deploy/planner/rates.json").read_bytes()
    readme0 = (root / "README.md").read_bytes()
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 3
    assert "rates: REFUSED" in out
    assert "balanced segment on L40S: 20.0 video s behind the rate; at least 30 s" in out
    assert (root / "src/flora_deploy/planner/rates.json").read_bytes() == rates0
    assert (root / "README.md").read_bytes() == readme0  # the placeholder stays
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "`planner/rates.json` **not updated**" in rep
    assert "| balanced | not measured |" in rep
    doc = (root / "docs/balanced-engines.md").read_text()
    assert "**not measured** (balanced segment on L40S: 20.0 video s" in doc
    # the human job is still certified (provisionally); yam has no balanced records
    assert sorted(c["covered_pools"][0] for c in certs(root)) == ["human", "human"]
    assert "segment (yam): refused" not in rep  # no balanced yam job: nothing to certify
    rc2, out2 = run(root, "--write", capsys=capsys)
    assert rc2 == 3 and "nothing to change" in out2


def test_a_single_balanced_job_is_measured_with_n_1_and_a_50_percent_band(tmp_path, capsys):
    """The balanced-only GPU session: one job, one container per kind. Measured, honestly:
    n=1 and ±50% in rates.json (rate and cold start), the report and `flora plan`."""
    from flora_deploy.planner.options import PlanRequest, WorkloadSummary, plan_options, rates_label
    from flora_deploy.planner.rates import RateTable

    root = build(tmp_path, jobs=("human",))
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    path = root / "src/flora_deploy/planner/rates.json"
    data = json.loads(path.read_text())
    for kind in ("segment", "caption"):
        row = next(r for r in data["rates"] if r["preset"] == "balanced" and r["kind"] == kind)
        assert (row["status"], row["n"], row["uncertainty"]) == ("measured", 1, 0.5)
        assert "n=1 job(s)" in row["provenance"] and "±50%" in row["provenance"]
        srow = next(r for r in data["startup"] if r["preset"] == "balanced"
                    and r["kind"] == kind)
        assert (srow["status"], srow["n"], srow["uncertainty"]) == ("measured", 1, 0.5)
        assert "one container" in srow["provenance"]
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "a single sample (n=1) has no range, so its band is ±50%" in rep
    assert "| balanced | segment | L40S | 1 | 30.7 |" in rep and "±50%" in rep
    t = RateTable.load(path)
    w = WorkloadSummary.from_dict({"groups": [{"pool": "human", "cameras": 1,
                                               "video_s": 3600.0, "episodes": 100}]})
    opts = plan_options(PlanRequest(workload=w, stages=("segment", "caption"),
                                    presets=("balanced",), max_gpus=4), table=t)
    assert opts and {rates_label(o) for o in opts} == {"measured (n=1, ±50%)"}
    assert all(k.rate_samples == "n=1, ±50%" for o in opts for k in o.kinds)


def test_without_oracle_or_noise_records_certificates_are_deferred(tmp_path, capsys):
    """Balanced only: no exact / noise records. Rates are still measured; certification is
    deferred to the certified run, and nothing written earlier is removed."""
    root = build(tmp_path)
    assert run(root, "--write", capsys=capsys)[0] == 0
    before = sorted(p.name for p in _cert_dir(root).glob("cert_*.json"))
    shutil.rmtree(root / "reports/wave-h/records/exact")
    shutil.rmtree(root / "reports/wave-h/records/noise")
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    assert "deferred to the certified run" in out and "remove" not in out
    assert sorted(p.name for p in _cert_dir(root).glob("cert_*.json")) == before
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "deferred to the certified run (no oracle records" in rep
    assert "refused" not in rep.split("## Certificate")[1]


def test_missing_l0_is_reported_and_downgrades_the_segment_certificate(tmp_path, capsys):
    root = build(tmp_path, labs=("captioner",))
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    by = {(c["stage"], c["covered_pools"][0]): c for c in certs(root)}
    assert "no_l0" in by[("segment", "yam")]["reasons"]
    assert by[("segment", "yam")]["l0"]["status"] == "missing"
    assert by[("caption", "yam")]["l0"]["status"] == "pass"
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "| tempseg | missing | - |" in rep
    assert "the L0 JSON of tempseg" in rep
    assert "| tempseg | not measured |" in rep and "(tempseg not measured)" in rep
    assert "tempseg: missing" in (root / "docs/balanced-engines.md").read_text()


def test_gate_a_over_the_reference_recommends_the_plugin(tmp_path):
    labs = {"tempseg": _lab("tempseg", "NVIDIA L40S", 0.9), "captioner": None}
    g = fin.gate_a(labs)
    assert g["recommendation"] == "on" and "tempseg" in g["why"]
    assert fin.gate_a({"tempseg": None, "captioner": None})["recommendation"] == "off"


@pytest.mark.parametrize(("lag", "billed", "want"), [
    (600.0, {"exact_e2e": 0.40}, "partial: read"),  # within the billing lag
    (7200.0, {"exact_e2e": 5.00}, "MISMATCH"),  # reconciled, far from tracked
])
def test_billing_lag_and_mismatch_are_flagged(tmp_path, capsys, lag, billed, want):
    root = build(tmp_path, billed_lag_s=lag, billed=billed)
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0
    rep = (root / "docs/wave-h-report.md").read_text()
    row = next(ln for ln in rep.splitlines() if ln.startswith("| b. exact oracle"))
    assert want in row
    assert "Warnings:" in rep and "WARNING exact_e2e" in out
    other = next(ln for ln in rep.splitlines() if ln.startswith("| 0. TempSeg"))
    if lag < wh.BILLING_LAG_S:
        assert "re-run the runbook's billing read after" in rep
        assert "partial" in other  # read under an hour after it too
    else:
        assert "reconciled" in other  # billed == tracked: only (b) is flagged


def test_blocks_fill_only_between_markers_and_refuse_without_them():
    text = "## Cost\n\nintro\n\n<!-- wave-h:cost -->\n<!-- /wave-h:cost -->\n\n## Next\n"
    out = fin.replace_block(text, "cost", "about $1\nper video-hour")
    assert out == ("## Cost\n\nintro\n\n<!-- wave-h:cost -->\nabout $1\nper video-hour\n"
                   "<!-- /wave-h:cost -->\n\n## Next\n")
    assert fin.replace_block(out, "cost", "about $1\nper video-hour") == out
    with pytest.raises(SystemExit):
        fin.replace_block("## Run on Modal\n\n## Config\n", "cost", "x")  # never inserts


def test_l0_copy_takes_the_pool_gpu_name_and_drops_episode_rows():
    lab = _lab("captioner", "NVIDIA H100 80GB HBM3", 0.2)
    got = fin.l0_for_certify(lab, "captioner", "H100!")
    assert got["baseline"]["gpu"] == "H100!" and "episodes" not in got["baseline"]
    assert SECRET not in json.dumps(got)
    other = fin.l0_for_certify(_lab("tempseg", "NVIDIA H200", 0.1), "tempseg", "L40S")
    assert other["baseline"]["gpu"] == "NVIDIA H200"  # left for certify to refuse


# ---------------------------------------------------------------------------------------------
# Review 11a


def _state(root: Path) -> wh.State:
    return wh.State.from_json(json.loads((root / "reports/wave-h/state.json").read_text()))


def _save(root: Path, st: wh.State) -> None:
    wh.save_state(st, root / "reports/wave-h/state.json")


def _cert_dir(root: Path) -> Path:
    return root / "src" / "flora_deploy" / "config" / "certificates"


def run_with(root: Path, cli, *extra: str, capsys=None) -> tuple[int, str]:
    rc = fin.main(["--root", str(root), *DERIVED, *extra], cli=cli)
    return rc, (capsys.readouterr().out if capsys else "")


def test_receipts_without_the_report_contract_fields_refuse_with_a_clear_message(tmp_path,
                                                                                 capsys):
    """H1: finalize reads ``Job.report()``'s per-container ``kind``, ``startup_s``, ``busy_s``,
    ``seconds``; a report without them is named as such, never measured from nothing."""
    root = build(tmp_path)
    st = _state(root)
    for step in ("exact_e2e", "balanced", "noise"):
        for r in st.steps[step].result["jobs"].values():
            r["containers"] = [{k: c[k] for k in ("container_id", "gpu", "gpu_count", "usd",
                                                  "stopped")} for c in r["containers"]]
    _save(root, st)
    rc, out = run(root, capsys=capsys)
    assert rc == 3 and "rates: REFUSED" in out
    assert "lacks `kind`, `startup_s`, `busy_s`, `seconds`" in out
    assert "Job.report() per-container" in out
    for step in ("exact_e2e", "balanced", "noise"):
        for r in st.steps[step].result["jobs"].values():
            r["containers"] = []
    _save(root, st)
    rc, out = run(root, capsys=capsys)
    assert rc == 3
    assert "balanced.jobs.human: `flora report` lists no per-container receipts" in out


def test_write_with_missing_records_never_deletes_certificates(tmp_path, capsys):
    """M1: records gone (a fresh checkout, --fetch not run): every certificate stays."""
    root = build(tmp_path)
    assert run(root, "--write", capsys=capsys)[0] == 0
    before = sorted(p.name for p in _cert_dir(root).glob("cert_*.json"))
    assert len(before) == 4
    shutil.rmtree(root / "reports/wave-h/records")
    rc, out = run(root, "--write", capsys=capsys)
    assert "remove" not in out
    assert sorted(p.name for p in _cert_dir(root).glob("cert_*.json")) == before
    man = json.loads((root / "reports/wave-h/finalize.json").read_text())
    assert man["certificates"] == [Path(b).stem for b in before]  # still owned
    # L6: the refusals name inputs relative to the root, never the local path
    rep = (root / "docs/wave-h-report.md").read_text()
    assert ("deferred to the certified run (no oracle records: "
            "reports/wave-h/records/exact/human/segment.jsonl" in rep)
    assert str(root) not in rep and str(tmp_path) not in rep


def test_a_new_certificate_replaces_only_the_one_with_its_key(tmp_path, capsys):
    root = build(tmp_path)
    assert run(root, "--write", capsys=capsys)[0] == 0
    old = {(c["stage"], c["covered_pools"][0]): c["id"] for c in certs(root)}
    # the human records go missing; yam's are re-certified at a new stamp (new ids, same keys)
    for stage in ("segment", "caption"):
        (root / f"reports/wave-h/records/balanced/human/{stage}.jsonl").unlink()
    st = _state(root)
    t = fin.run_stamp(st)
    for s in st.steps.values():
        for a in s.attempts:
            if a.finished == t:
                a.finished += 60  # a later run: a new stamp, so new certificate ids
    _save(root, st)
    rc, out = run(root, "--write", capsys=capsys)
    new = {(c["stage"], c["covered_pools"][0]): c["id"] for c in certs(root)}
    assert len(new) == 4 == len(certs(root))
    assert new[("segment", "human")] == old[("segment", "human")]  # kept: nothing replaced it
    assert new[("segment", "yam")] != old[("segment", "yam")]  # replaced by its new run
    assert out.count("remove superseded") == 2
    man = json.loads((root / "reports/wave-h/finalize.json").read_text())
    assert sorted(man["certificates"]) == sorted(new.values())


def test_blocks_match_whole_marker_lines_only():
    """L4: a marker quoted in prose, or a block whose name starts with this one, is not it."""
    txt = ("intro `<!-- wave-h:cost -->` explains the block\n...\n<!-- wave-h:cost -->\nold\n"
           "<!-- /wave-h:cost -->\n")
    new = fin.replace_block(txt, "cost", "NEW")
    assert new == ("intro `<!-- wave-h:cost -->` explains the block\n...\n<!-- wave-h:cost -->\n"
                   "NEW\n<!-- /wave-h:cost -->\n")
    txt2 = ("<!-- wave-h:costs -->\nA\n<!-- /wave-h:costs -->\n<!-- wave-h:cost -->\nB\n"
            "<!-- /wave-h:cost -->\n")
    assert fin.replace_block(txt2, "cost", "NEW") == txt2.replace("\nB\n", "\nNEW\n")
    with_note = "<!-- wave-h:results (owned by x) -->\nold\n<!-- /wave-h:results -->\n"
    assert "\nNEW\n" in fin.replace_block(with_note, "results", "NEW")
    for bad in ("<!-- wave-h:cost -->\n<!-- /wave-h:cost -->\n" * 2,  # two blocks
                "<!-- /wave-h:cost -->\n<!-- wave-h:cost -->\n"):  # close before open
        with pytest.raises(SystemExit):
            fin.replace_block(bad, "cost", "x")


def test_report_carries_structured_certify_fields_only(tmp_path, capsys):
    """No raw ``flora certify`` output reaches the report: a refusal is its exit code and a
    category; a non-zero exit with a printed list is not accepted (L5)."""
    root = build(tmp_path)

    def leaky(args):
        return 2, f"error: {SECRET} {root}/reports/wave-h/records/x.jsonl"

    run_with(root, leaky, "--write", capsys=capsys)
    rep = (root / "docs/wave-h-report.md").read_text()
    assert SECRET not in rep and str(root) not in rep
    assert "refused: flora certify exited 2" in rep
    for name, data in owned(root).items():
        assert SECRET.encode() not in data, name
    assert certs(root) == []

    def lies(args):
        _, text = fin.run_cli(args)
        return 1, text  # a certificate list, but a failing exit code

    run_with(root, lies, "--write", capsys=capsys)
    assert certs(root) == []
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "flora certify exited 1 with verdict provisional" in rep
    assert SECRET not in rep


def test_a_wrong_gpu_worker_refuses_its_jobs_rates(tmp_path, capsys):
    root = build(tmp_path)
    st = _state(root)
    j = st.steps["balanced"].result["jobs"]["yam"]
    cap = next(c for c in j["containers"] if c["kind"] == "caption")
    j["containers"].insert(0, dict(cap, container_id="ta-early", startup_s=5.0))
    cap["gpu_seen"] = "NVIDIA H200"  # jobs.containers: the receipts' nvidia-smi name
    _save(root, st)
    rates0 = (root / "src/flora_deploy/planner/rates.json").read_bytes()
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 3
    assert ("balanced.jobs.yam: caption rates refused: container ta-balanced-caption-yam ran "
            "on 'NVIDIA H200', not the balanced caption pool's H100") in out
    assert (root / "src/flora_deploy/planner/rates.json").read_bytes() == rates0
    m = fin.measure(_state(root))[("balanced", "caption")]
    assert [s[0] for s in m.samples] == ["human"]
    assert 5.0 not in m.startups  # nothing from the refused job, not even its good container
    # nvidia-smi's name for the pool's own GPU is the same GPU
    cap["gpu_seen"] = "NVIDIA H100 80GB HBM3"
    j["containers"].pop(0)
    _save(root, st)
    assert run(root, capsys=capsys)[0] == 0


def test_global_cap_is_the_state_plans_cap_not_the_full_plans(tmp_path, capsys):
    """A balanced-only run was gated on its own cap: the sum of the upper projections the
    runbook stored per step, rounded up to $0.10 (``wave_h.plan_global_cap``), not
    ``GLOBAL_CAP_USD`` (the full plan's)."""
    st = wh.State(run_id="r", created=T0, billing_since="2026-10-13")
    for name, up in (("tempseg", 0.7189), ("preflight", 0.0053), ("balanced", 3.6646)):
        st.step(name).status = "done"
        st.step(name).projected_upper_usd = up
    st.step("balanced").result = {"kinds": {}}  # pre-(d2): the lab ran inside (d)
    assert fin.run_plan(st) == "balanced-only" and fin.inline_lab(st)
    assert fin.global_cap(st) == 4.40 != wh.GLOBAL_CAP_USD
    # the same relationship as the runbook: cap = upper sum rounded up to $0.10
    ps = {n: SimpleNamespace(usd_upper=st.steps[n].projected_upper_usd)
          for n in wh.PLANS["balanced-only"] if n != "lab"}
    ps["lab"] = SimpleNamespace(usd_upper=0.0)
    assert fin.global_cap(st) == wh.plan_global_cap("balanced-only", ps)
    st.step("lab").projected_upper_usd = 2.3  # a (d2) state: the lab step counts
    assert not fin.inline_lab(st) and fin.global_cap(st) == 6.70
    # a step the run never reached is priced as the runbook prices it
    st2 = wh.State(run_id="r", created=T0, billing_since="2026-10-13")
    st2.step("tempseg").projected_upper_usd = 0.7189
    up = 0.7189 + sum(wh.project(n, wh.offline_manifest(wh.DURATION_HEADROOM)).usd_upper
                      for n in ("preflight", "balanced", "lab"))
    assert up <= fin.global_cap(st2) < up + 0.1 + 1e-9
    # a started full-plan step: the full plan's cap
    st.step("exact_e2e").attempts.append(_attempt(T0, 0.1, []))
    assert fin.run_plan(st) == "full" and fin.global_cap(st) == wh.GLOBAL_CAP_USD

    # end to end: the report and the summary line name the plan's cap
    root = build(tmp_path)
    path = root / "reports/wave-h/state.json"
    d = json.loads(path.read_text())
    for name in wh.deferred_steps("balanced-only"):
        d["steps"].pop(name)
    for name, up in (("tempseg", 0.7189), ("preflight", 0.0053), ("balanced", 3.6646)):
        d["steps"][name]["projected_upper_usd"] = up
    path.write_text(json.dumps(d))
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    assert "(plan balanced-only)" in out and "of $4.40" in out and "$6.50" not in out
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "of the $4.40 global cap" in rep and "$6.50" not in rep


# --- review 12 F8 / F9: rounding, Gate A without requests, the spend figures ------------------


def test_usd_per_video_h_rounds_once_from_unrounded_rates():
    # the 2026-10-22 run: 65.4 / 161.5 busy s over 178.2 video s gave $5.62 by summing cents
    def m(kind, busy):
        real = fin.Measurement("balanced", kind, "GPU", [("human", busy, 178.2)])
        return SimpleNamespace(rate=real.rate, rate_exact=real.rate_exact, refusals=[],
                               uncertainty=0.5, gpu="GPU")

    ms = {("balanced", "segment"): m("segment", 65.4), ("balanced", "caption"): m("caption", 161.5)}
    assert ms[("balanced", "segment")].rate == 0.367  # the rounded rate rates.json keeps
    c = fin.usd_per_video_h(ms, "balanced")
    seg_s = wh.pool_container("segment", "balanced").usd_per_s()
    cap_s = wh.pool_container("caption", "balanced").usd_per_s()
    want = 3600 * (65.4 / 178.2 * seg_s + 161.5 / 178.2 * cap_s)
    assert c["usd_per_video_h"] == round(want, 2)
    double = round(sum(round(3600 * r * p, 2) for r, p in ((0.367, seg_s), (0.906, cap_s))), 2)
    if round(want, 2) != double:  # with today's prices: $5.61, not $5.62
        assert c["usd_per_video_h"] != double


def test_gate_a_without_requests_is_not_measured():
    zero = {"gate_a": {"requests": 0, "greedy_identical": 0, "max_abs_dlogit": None,
                       "mean_abs_dlogit": None}}
    g = fin.gate_a({"tempseg": zero, "captioner": None})
    assert g["kinds"] == {"tempseg": None, "captioner": None}
    assert g["why"].startswith("Gate A was not measured")


def test_run_cost_without_cold_starts():
    st = wh.State("r", 0.0, "x")
    st.step("balanced").result = {"jobs": {"human": {
        "spend": 1.6516, "video_s": 178.2,
        "containers": [{"seconds": 665.9, "startup_s": 300.5, "usd": 0.5016},
                       {"seconds": 540.4, "startup_s": 325.0, "usd": 0.7649}]}}}
    assert fin.run_usd_per_video_h(st, "balanced") == 33.37
    assert fin.run_usd_per_video_h_after_startup(st, "balanced") == 11.72


def test_report_spend_shows_tracked_charged_and_billed(tmp_path, capsys):
    root = build(tmp_path)
    st_path = root / "reports/wave-h/state.json"
    st = wh.State.from_json(json.loads(st_path.read_text()))
    lab = st.step("lab")
    lab.status = "failed"  # a crashed attempt: charged at its budget until billing reads it
    end = st.billed_at - 60  # billing has not caught up with it yet
    lab.attempts.append(wh.Attempt(started=end - 600, finished=end, tracked_usd=0.72,
                                   status="failed", budget_usd=2.35, report_error=True))
    st_path.write_text(json.dumps(st.to_json()))
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    rep = (root / "docs/wave-h-report.md").read_text()
    tracked = sum(s.tracked_usd for s in st.steps.values())
    charged = sum(s.charged(st.watermark) for s in st.steps.values())
    assert f"Total: **${st.spent:,.2f}**" in rep
    assert f"- tracked ${tracked:,.2f}" in rep and f"- charged ${charged:,.2f}" in rep
    assert "over tracked" in rep and f"- billed ${st.billed_total:,.2f}" in rep


README_BLOCKS = ("certified", "allow-uncertified", "cost")


def _strip_blocks(text: str) -> str:
    """``text`` with every ``wave-h:*`` block body emptied (markers kept)."""
    import re

    return re.sub(r"(^<!-- wave-h:([\w-]+)(?: [^>]*)? -->\n).*?(^<!-- /wave-h:\2 -->)", r"\1\3",
                  text, flags=re.M | re.S)


def test_write_regenerates_every_readme_block_and_nothing_else(tmp_path, capsys):
    import re

    root = build(tmp_path)
    readme = root / "README.md"
    names = re.findall(r"^<!-- wave-h:([\w-]+)(?: [^>]*)? -->$", readme.read_text(), re.M)
    assert sorted(names) == sorted(README_BLOCKS)  # a new block needs a regenerator here
    stale = readme.read_text()
    for n in README_BLOCKS:  # every block body stale: --write must rewrite each one
        stale = re.sub(rf"(^<!-- wave-h:{n} -->\n).*?(^<!-- /wave-h:{n} -->)",
                       r"\1STALE\n\2", stale, flags=re.M | re.S)
    readme.write_text(stale)
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    new = readme.read_text()
    assert "STALE" not in new
    assert _strip_blocks(new) == _strip_blocks(stale)  # nothing outside the blocks changed
    certs_dir = root / "src/flora_deploy/config/certificates"
    blocks = fin.certification_blocks(stale, [certs_dir])
    for n in ("certified", "allow-uncertified"):
        want = blocks.split(f"<!-- wave-h:{n} -->\n")[1].split(f"<!-- /wave-h:{n} -->")[0]
        assert new.split(f"<!-- wave-h:{n} -->\n")[1].split(f"<!-- /wave-h:{n} -->")[0] == want


def test_write_locks_every_certificate_it_writes(tmp_path, capsys):
    from flora_deploy.validate import certfreeze as F
    from flora_deploy.validate import certkey as K

    root = build(tmp_path)
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    d = root / "src/flora_deploy/config/certificates"
    cs = certs(root)
    assert cs and "lock " in out
    for c in cs:
        lock = F.read_freeze(d, c)
        assert lock is not None, c["id"]  # the drift gate can name the changed modules
        assert K.sha256_hex(lock["key_document"]) == c["engine_spec_hash"]
    assert run(root, "--write", capsys=capsys)[1].count("would lock") == 0


def test_write_refuses_a_certificate_this_tree_cannot_lock(tmp_path, capsys, monkeypatch):
    from flora_deploy.validate import certfreeze as F
    from flora_deploy.validate import certkey as K

    root = build(tmp_path)
    before = owned(root)

    def drifted(cert, source_dir=None):
        raise K.CertKeyError(f"{cert['id']}: this tree no longer computes its key")

    monkeypatch.setattr(F, "lock_document", drifted)
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 1 and "REFUSED freeze lock" in out and "nothing written" in out
    assert owned(root) == before


def test_new_rates_regenerate_the_cost_guide(tmp_path, capsys):
    """When the rates change, docs/costs.md's generated tables follow them, prose untouched."""
    from dataclasses import replace

    from flora_deploy.planner.rates import RateTable
    from scripts import cost_table

    root = build(tmp_path)
    shutil.copyfile(REPO / "docs/costs.md", root / "docs/costs.md")
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0, out
    assert "docs/costs.md" in out
    table = replace(RateTable.load(root / "src/flora_deploy/planner/rates.json"),
                    source="planner/rates.json")
    doc0 = (REPO / "docs/costs.md").read_text(encoding="utf-8")
    doc = (root / "docs/costs.md").read_text(encoding="utf-8")
    assert doc != doc0 and doc == cost_table.render(doc0, table=table)
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 0 and "costs.md" not in out  # idempotent
