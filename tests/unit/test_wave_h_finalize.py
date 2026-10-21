"""``scripts/wave_h_finalize.py`` on a synthetic Wave H-lite state + records: rates, certificates,
report and docs blocks; refusals (too few samples), a missing L0, billing lag / mismatch, and
idempotence. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

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
    return {"run": "wave-h-r1", "kind": kind, "baseline": {
        "gpu": gpu, "l0": _l0_rows(), "episodes": [{"id": "x", "ok": True,
                                                    "rows": [{"caption": SECRET}]}]},
        "gate_a": {"requests": 6, "greedy_identical": 6, "positions": 120,
                   "max_abs_dlogit": dlogit, "mean_abs_dlogit": dlogit / 10,
                   "first_divergence": []}}


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
    for name in wh.STEPS:
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
        if name in ("exact_e2e", "rewards"):
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
    assert "2 jobs (76.5 video s) on L40S" in seg.provenance and "0.500-0.600" in seg.provenance
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
    bal_cost = round(round(want_seg * seg_h, 2) + round(want_cap * cap_h, 2), 2)
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


def test_too_few_samples_refuses_the_rates_and_keeps_the_placeholders(tmp_path, capsys):
    root = build(tmp_path, jobs=("human",))
    rates0 = (root / "src/flora_deploy/planner/rates.json").read_bytes()
    readme0 = (root / "README.md").read_bytes()
    rc, out = run(root, "--write", capsys=capsys)
    assert rc == 3
    assert "rates: REFUSED" in out
    assert "balanced segment on L40S: 1 job sample(s); an error band needs a range" in out
    assert "balanced caption on H100: 1 cold-start sample(s)" in out
    assert (root / "src/flora_deploy/planner/rates.json").read_bytes() == rates0
    assert (root / "README.md").read_bytes() == readme0  # the placeholder stays
    rep = (root / "docs/wave-h-report.md").read_text()
    assert "`planner/rates.json` **not updated**" in rep
    assert "| balanced | not measured |" in rep
    doc = (root / "docs/balanced-engines.md").read_text()
    assert "**not measured** (balanced segment on L40S: 1 job sample(s)" in doc
    # the human job is still certified (provisionally); yam has no balanced records
    assert sorted(c["covered_pools"][0] for c in certs(root)) == ["human", "human"]
    assert "segment (yam): refused" not in rep  # no balanced yam job: nothing to certify
    rc2, out2 = run(root, "--write", capsys=capsys)
    assert rc2 == 3 and "nothing to change" in out2


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
    rewards = next(ln for ln in rep.splitlines() if ln.startswith("| c. rewards"))
    if lag < wh.BILLING_LAG_S:
        assert "re-run the runbook's billing read after" in rep
        assert "partial" in rewards  # read under an hour after it too
    else:
        assert "reconciled" in rewards  # billed == tracked: only (b) is flagged


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
