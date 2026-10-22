"""Per-camera-layout pricing: ``rates.json`` ``camera_scaling`` in the planner, ``flora plan``,
``docs/costs.md`` and ``scripts/wave_h.py``'s yam certified plan (one set of numbers)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import template_text
from flora_deploy.planner.options import (
    PlanRequest,
    WorkloadGroup,
    WorkloadSummary,
    build_option,
    rates_label,
)
from flora_deploy.planner.rates import RatesFileError, RateTable

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
RATES = REPO / "src" / "flora_deploy" / "planner" / "rates.json"
HOUR = 3600.0


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _groups(*layouts: tuple[int, float]) -> WorkloadSummary:
    return WorkloadSummary(tuple(WorkloadGroup("yam" if c > 1 else "human", c, h * HOUR,
                                               int(h * 60)) for c, h in layouts))


def _option(workload: WorkloadSummary, preset: str = "balanced", conc: int = 1,
            table: RateTable | None = None):
    from flora_deploy.jobs.budget import pinned_rates

    t = table or RateTable.load()
    req = PlanRequest(workload=workload, presets=(preset,))
    return build_option(req, t, pinned_rates(), preset, conc)


def _work(o) -> dict[str, float]:
    return {k.kind: k.work_s for k in o.kinds}


def test_table_is_an_estimate_with_provenance():
    cs = RateTable.load().camera_scaling
    assert cs is not None and cs.status == "estimate"
    assert "golden-manifest-v3-yam" in cs.source and cs.date and cs.derivation
    assert cs.factors(1) == {}
    f = cs.factors(3)
    assert set(f) == {"segment", "caption", "segcap"}
    assert f["segment"] == pytest.approx(2.56, abs=0.01)
    assert f["caption"] == pytest.approx(1.42, abs=0.01)
    assert f["segcap"] == pytest.approx(1.30, abs=0.01)


def test_single_view_is_unchanged():
    t = RateTable.load()
    bare = replace(t, camera_scaling=None)
    for preset, conc in (("balanced", 1), ("exact", 4)):
        a, b = _option(_groups((1, 10)), preset, conc, t), _option(_groups((1, 10)), preset,
                                                                   conc, bare)
        assert a == b
        assert not a.camera_scaling and all(not k.camera_scale for k in a.kinds)
        assert not any("camera scaling" in f for f in a.flags)
        assert "camera" not in rates_label(a)
        for k in a.kinds:
            assert k.work_s == pytest.approx(10 * HOUR * t.rate(preset, k.kind, 1, conc)
                                             .container_s_per_video_s)


@pytest.mark.parametrize(("preset", "conc"), [("balanced", 1), ("exact", 4)])
def test_three_view_scales_by_the_table(preset, conc):
    t = RateTable.load()
    one, three = _option(_groups((1, 10)), preset, conc), _option(_groups((3, 10)), preset, conc)
    f = t.camera_scaling.factors(3)
    for kind, w in _work(three).items():
        assert w == pytest.approx(_work(one)[kind] * f[kind], rel=1e-6)
    assert three.cost_usd > one.cost_usd
    assert three.camera_scaling == "estimate"
    assert all(k.camera_scale == f"3-view x{f[k.kind]:g}" for k in three.kinds)


def test_mixed_layouts_priced_per_layout_then_summed():
    one, three = _option(_groups((1, 4))), _option(_groups((3, 6)))
    mixed = _option(_groups((1, 4), (3, 6)))
    for kind, w in _work(mixed).items():
        assert w == pytest.approx(_work(one)[kind] + _work(three)[kind], rel=1e-6)
    assert mixed.camera_scaling == "estimate"


def test_rate_measured_on_the_layout_is_not_scaled():
    t = RateTable.load()
    rows = list(t.rates)
    seg = next(e for e in rows if e.preset == "balanced" and e.kind == "segment")
    rows.append(replace(seg, cameras=3, container_s_per_video_s=1.0))
    o = _option(_groups((3, 1)), table=replace(t, rates=tuple(rows)))
    k = {k.kind: k for k in o.kinds}
    assert k["segment"].work_s == pytest.approx(HOUR) and not k["segment"].camera_scale
    assert k["caption"].camera_scale  # still on the "*" row: scaled


def test_unlisted_layout_is_flagged_not_scaled():
    o = _option(_groups((2, 1)))
    assert not o.camera_scaling
    assert any(f.startswith("no camera scaling for 2-view") for f in o.flags)


def test_bad_camera_scaling_is_refused():
    base = json.loads(RATES.read_text())
    for patch in ({"status": "guess"}, {"source": ""}, {"layouts": {"3": {"segment": 0}}},
                  {"layouts": {"1": {"segment": 2.0}}}, {"layouts": {}}):
        d = copy.deepcopy(base)
        d["camera_scaling"].update(patch)
        with pytest.raises(RatesFileError, match="camera_scaling"):
            RateTable.from_dict(d, source="x.json")
    d = copy.deepcopy(base)
    del d["camera_scaling"]
    assert RateTable.from_dict(d, source="x.json").camera_scaling is None


def test_plan_output_shows_scaling_and_its_status(tmp_path):
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(template_text("exact_mp4_dir"))
    status = RateTable.load().camera_scaling.status
    runner = CliRunner()
    args = ["plan", "-c", str(cfg), "--video-hours", "10", "--budget", "6000",
            "--max-gpus", "10"]
    r = runner.invoke(app, [*args, "--cameras", "3"], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    out = " ".join(r.output.split())
    assert f"camera scaling ({status}" in out and "3-view segment x2.56" in out
    assert f"+ camera scaling ({status})" in out and "3-view 10.0 h" in out
    r1 = runner.invoke(app, [*args, "--cameras", "1"], env={"COLUMNS": "200"})
    assert r1.exit_code == 0 and "camera scaling" not in r1.output
    rj = runner.invoke(app, [*args, "--cameras", "3", "--json"])
    d = json.loads(rj.output)
    assert all(o["camera_scaling"] == status and o["rates"].endswith(f"({status})")
               for o in d["options"])


def test_index_summary_layouts_price_separately():
    """The ``groups`` of ``flora index``'s summary.json (pool, cameras) feed the same scaling."""
    w = WorkloadSummary.from_dict({"groups": [
        {"pool": "human", "cameras": 1, "video_s": 3600, "episodes": 60},
        {"pool": "yam", "cameras": 3, "video_s": 7200, "episodes": 120}]})
    assert _work(_option(w)) == pytest.approx(
        {k: a + b for (k, a), b in zip(_work(_option(_groups((1, 1)))).items(),
                                       _work(_option(_groups((3, 2)))).values(), strict=True)})


def test_cost_doc_has_a_three_camera_row():
    C = _load("cost_table")
    block = C.block_cameras(RateTable.load())
    assert "\n| 3 | " in block and "\n| 1 | " in block and "estimate" in block


def test_wave_h_and_planner_use_identical_numbers():
    wh = _load("wave_h")
    planner = RateTable.load().camera_scaling.factors(3)
    assert wh.CERT_SPECS["yam"].scale == planner
    # the stored numbers are what wave_h's derivation (three_view_scale) gives
    assert wh.three_view_scale() == planner
