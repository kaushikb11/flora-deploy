"""``scripts/wave_h.py --plan certified --embodiment``: the yam certified plan on golden/v3-yam
(its own episodes, data dir and state), its three-view prices, the human plan left exactly as
it was, and the v3-yam selection itself. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from flora_deploy.validate.golden import (
    SELECTIONS,
    entry_digest,
    entry_id,
    load_selection,
    prefix_for,
    validate_selection,
)

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


wh = _load("wave_h")
h0 = sys.modules["h0_rehearsal"]
HUMAN, YAM = wh.CERT_SPECS["human"], wh.CERT_SPECS["yam"]


def _manifest(spec) -> dict:
    return wh.offline_manifest(wh.DURATION_HEADROOM, spec.golden)


@pytest.fixture
def no_lab(monkeypatch, tmp_path):
    """The lab evidence as in the main checkout (or not): the same for every call here."""
    monkeypatch.setattr(wh, "LAB_OUT", tmp_path / "no-lab")


# ---------------------------------------------------------------------------------------------
# The v3-yam selection


def test_v3_yam_is_twelve_three_view_episodes_from_pinned_datasets():
    v3 = validate_selection(load_selection(SELECTIONS["golden/v3-yam"]))
    v1 = load_selection()
    assert v3["version"] == "golden/v3-yam" and str(prefix_for(v3["version"])) == "golden/v3-yam"
    eps = v3["episodes"]
    assert len(eps) >= 12 and {e["pool"] for e in eps} == {"yam"}
    for e in eps:
        assert e["profile"] == "yam" and set(e["cameras"]) == {"main", "left_wrist",
                                                               "right_wrist"}
        assert e["expected_s"] <= 30.0  # short: the oracle is priced per video second
    # only datasets v1 already pins, at the same revision
    for k, ds in v3["datasets"].items():
        assert ds == v1["datasets"][k], k
    by1 = {entry_id(e): e for e in v1["episodes"]}
    kept = [e for e in eps if entry_id(e) in by1]
    assert len(kept) == 8
    for e in kept:  # unchanged entries: the same staged bytes as v1
        assert entry_digest(e, v3["datasets"][e["dataset"]]) == entry_digest(
            by1[entry_id(e)], v1["datasets"][e["dataset"]])
    tasks = {e["episode"].split("/")[1] if e["dataset"] == "manirewardbench_yam"
             else e["dataset"] for e in eps}  # ManiRewardBench: YAM/<task>/<episode>
    assert len(tasks) == len(eps)  # one episode per task
    noise = v3["noise_subset"]
    assert len(noise) == wh.MIN_NOISE_EPISODES and set(noise) <= {entry_id(e) for e in eps}
    assert len({n.split("/")[1] for n in noise}) >= 3  # ALOHA, RoboCOIN and ManiRewardBench


def test_v3_yam_stages_under_its_own_prefix():
    v3 = load_selection(SELECTIONS["golden/v3-yam"])
    from flora_deploy.validate.golden import episode_relpath

    paths = {str(episode_relpath(e, prefix_for(v3["version"]))) for e in v3["episodes"]}
    assert all(p.startswith("golden/v3-yam/yam/") for p in paths)


# ---------------------------------------------------------------------------------------------
# Plan shape: separate episodes, data dir and state


def test_yam_spec_has_its_own_episodes_data_dir_and_state():
    assert YAM.name == HUMAN.name == wh.CERTIFIED and YAM.shared_data
    assert YAM.golden == "golden/v3-yam" and YAM.embodiment == "yam"
    assert len(YAM.episodes) >= wh.MIN_CERT_EPISODES
    # v4 (R2): the noise run repeats the oracle on all 12; the cold witnesses keep the
    # selection's noise subset (5 + a spare) and its n- local ids
    assert YAM.noise == YAM.episodes and YAM.noise_limit is None
    assert set(YAM.witness) < set(YAM.episodes) and YAM.witness == wh.CERT_YAM_NOISE
    assert len(YAM.witness) == wh.MIN_NOISE_EPISODES + wh.NOISE_SPARES
    assert YAM.witness_limit == len(YAM.witness) and YAM.segment_only_witness
    assert "caption_iso" in wh.PLANS[wh.CERTIFIED]
    assert wh.balanced_episodes(_manifest(YAM), YAM) == YAM.episodes
    # nothing shared with the human run: state, work dir, records, report, data dir (uids)
    for attr in ("state", "work", "records", "report"):
        assert getattr(YAM, attr) != getattr(HUMAN, attr), attr
    assert YAM.state.parent != HUMAN.state.parent
    assert YAM.data_dir("exact_e2e") == YAM.data_dir("balanced") == "cert-yam"
    assert HUMAN.data_dir("exact_e2e") == wh.CERT_DATA == "cert-human"
    assert set(YAM.lab_episodes) <= set(YAM.episodes) and HUMAN.lab_episodes == wh.EPISODES
    # the default is the human plan, unchanged
    assert wh.spec_for(wh.CERTIFIED) is wh.SPECS[wh.CERTIFIED] is HUMAN
    assert HUMAN.embodiment == "human" and HUMAN.scale == {}


def test_yam_witness_subset_is_what_index_limit_selects(tmp_path):
    """Three-view episodes are directories; ``flora index --limit 6`` still takes the witness
    subset first."""
    from flora_deploy.media.sources.mp4_dir import Mp4DirSource

    for g in YAM.episodes:
        d = tmp_path / YAM.local_id(g)
        d.mkdir()
        for role in ("main", "left_wrist", "right_wrist"):
            (d / f"{role}.mp4").write_bytes(b"")
        (d / "task.txt").write_text("t\n")
    eps = Mp4DirSource(tmp_path).list_episodes()
    assert len(eps) == len(YAM.episodes)
    assert all(len(e.cameras) == 3 for e in eps)
    first = [e.episode_id for e in eps[:YAM.witness_limit]]
    assert sorted(first) == sorted(YAM.local_id(g) for g in YAM.witness)


def test_stage_local_renames_three_view_directories(tmp_path, monkeypatch):
    def fake_fetch(manifest, work, *, episodes, poison_job):
        for data, ids in episodes.items():
            for g in ids:
                d = work / data / h0.episode_id(g)
                d.mkdir(parents=True)
                for role in ("main", "left_wrist", "right_wrist"):
                    (d / f"{role}.mp4").write_bytes(b"x")
                (d / "task.txt").write_text("t\n")

    calls = []
    monkeypatch.setattr(h0, "fetch_dataset",
                        lambda *a, **k: (calls.append(k["episodes"]), fake_fetch(*a, **k)))
    st = wh.State("r", 0.0, "x")
    ctx = wh.Ctx(st, _manifest(YAM), tmp_path, spec=YAM)
    root = wh.stage_local(ctx, YAM.data, YAM.episodes)
    for g in YAM.episodes:
        assert (root / YAM.local_id(g) / "main.mp4").is_file()
        assert not (root / h0.episode_id(g)).exists()
    wh.stage_local(ctx, YAM.data, YAM.episodes)  # idempotent: nothing fetched again
    assert len(calls) == 1


def test_yam_job_config_is_three_view_yam(tmp_path):
    from flora_deploy.config.schema import load_config

    cfg = tmp_path / "flora.yaml"
    cfg.write_text(h0.config_text(project="p", data_dir=YAM.data_dir("balanced"),
                                  embodiment=YAM.embodiment, stages=("segment", "caption"),
                                  store_volume="s", media_volume="m", run_id="r",
                                  preset="balanced"))
    c = load_config(cfg, allow_uncertified=True)
    assert c.pipeline.embodiment == "yam"


# ---------------------------------------------------------------------------------------------
# The flag


def test_embodiment_flag_picks_the_yam_plan_and_its_state(no_lab, tmp_path, capsys):
    state = tmp_path / "yam-state.json"
    assert wh.main(["--plan", "certified", "--embodiment", "yam", "--dry-run", "--offline",
                    "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert "(yam only), plan certified" in out
    assert f"episodes: {len(YAM.episodes)} yam" in out and "golden/v3-yam selection" in out
    assert "segcap yam (12 ep" in out and "3-view work" in out
    assert "assumption: yam is three-view" in out
    assert not state.exists()  # a dry run saves nothing


@pytest.mark.parametrize("emb", ["human", "yam"])
def test_status_shows_the_embodiments_episodes_and_command(no_lab, tmp_path, capsys, emb):
    spec = HUMAN if emb == "human" else YAM
    state = tmp_path / f"{emb}-state.json"
    assert wh.main(["--plan", "certified", "--embodiment", emb, "--status", "--offline",
                    "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert f"({emb}), plan certified" in out
    assert f"episodes: {len(spec.episodes)} {emb} ({spec.golden}), noise subset " \
           f"{len(spec.noise)}" in out
    assert f"b. exact oracle, {len(spec.episodes)} {emb} episodes ({spec.golden})" in out
    cmd = next(ln for ln in out.splitlines() if "command:" in ln)
    assert ("--embodiment yam" in cmd) == (emb == "yam")
    assert not state.exists()


def test_unsupported_embodiments_are_refused(capsys):
    for emb in ("dex", "tianji", "yam32"):
        with pytest.raises(SystemExit) as e:
            wh.main(["--plan", "certified", "--embodiment", emb, "--dry-run", "--offline"])
        assert e.value.code == 2
        assert "no golden certification set" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # another plan takes the human episodes only
        wh.main(["--plan", "full", "--embodiment", "yam", "--dry-run", "--offline"])
    assert "only --plan certified" in capsys.readouterr().err


def test_another_embodiments_state_is_refused(capsys):
    with pytest.raises(SystemExit) as e:
        wh.main(["--plan", "certified", "--embodiment", "yam", "--dry-run", "--offline",
                 "--state", str(HUMAN.state)])
    assert e.value.code == 2 and "another embodiment" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        wh.main(["--plan", "certified", "--dry-run", "--offline", "--state", str(YAM.state)])


def test_human_dry_run_is_unchanged_by_the_flag(no_lab, tmp_path, capsys):
    """``--embodiment human`` and no flag print the same plan; nothing in it is scaled."""
    state = tmp_path / "s.json"
    state.write_text(json.dumps({"run_id": "r1", "created": 0.0, "billing_since": "x",
                                 "plan": "certified", "steps": {}}))
    outs = []
    for extra in ([], ["--embodiment", "human"]):
        assert wh.main(["--plan", "certified", "--dry-run", "--offline", "--state",
                        str(state), *extra]) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]
    assert "3-view" not in outs[0] and "(human only)" in outs[0]
    assert f"b. exact oracle, {len(HUMAN.episodes)} human episodes (golden/v2)" in outs[0]


# ---------------------------------------------------------------------------------------------
# Prices


def test_three_view_scale_from_measured_prompt_lengths_and_generate_times():
    sc = wh.three_view_scale()
    assert YAM.scale == sc
    for kind in ("segment", "caption"):
        g = wh.GENERATE_S[kind] / wh.BUSY_S[kind]
        r = wh.PROMPT_TOKENS_3VIEW[kind] / wh.PROMPT_TOKENS_1VIEW[kind]
        assert 0 < g < 1 and r > 1
        assert sc[kind] == pytest.approx((1 - g) + g * r, abs=1e-3)
        assert 1 < sc[kind] < r  # only generation scales
    assert sc["segment"] == pytest.approx(2.556, abs=0.01)
    assert sc["caption"] == pytest.approx(1.422, abs=0.01)
    assert 1 < sc["segcap"] < 1.5  # the measured exact rate is already half 3-view


def test_yam_dry_run_cost_is_scaled_and_capped(no_lab):
    man_y = _manifest(YAM)
    ps = wh.plan_projections(wh.CERTIFIED, man_y, spec=YAM, lab_done=True)
    flat = wh.plan_projections(wh.CERTIFIED, man_y, spec=replace(YAM, rate_scale=()),
                               lab_done=True)
    for step in ("exact_e2e", "noise", "balanced"):
        assert ps[step].usd > flat[step].usd and ps[step].usd_upper > flat[step].usd_upper
    v = sum(wh.durations(man_y)[g] for g in YAM.episodes)
    assert f"(12 ep, {v:.1f} s x{YAM.scale['segcap']:.2f} 3-view work" in \
        ps["exact_e2e"].items[0]["what"]
    # the work, not the startup, is what scales
    nominal, _ = wh.exact_job_seconds(v * YAM.scale["segcap"])
    assert ps["exact_e2e"].items[0]["seconds"] == pytest.approx(nominal, abs=0.1)
    caps = wh.plan_caps(wh.CERTIFIED, ps)
    g = wh.plan_global_cap(wh.CERTIFIED, ps)
    st = wh.State(run_id="r", created=0.0, billing_since="x")
    for s in wh.PLANS[wh.CERTIFIED]:
        assert ps[s].usd_upper <= caps[s] < ps[s].usd_upper + 0.05 + 1e-9, s
        wh.check_before(st, s, ps[s], caps=caps, global_cap=g)
    up = sum(ps[s].usd_upper for s in wh.PLANS[wh.CERTIFIED])
    assert up <= g < up + 0.1 + 1e-9
    # the human plan's own projections are untouched by the yam spec
    man_h = _manifest(HUMAN)
    a = wh.plan_projections(wh.CERTIFIED, man_h)
    b = wh.plan_projections(wh.CERTIFIED, man_h, spec=HUMAN)
    assert {k: (p.items, p.peak_usd_per_s) for k, p in a.items()} == {
        k: (p.items, p.peak_usd_per_s) for k, p in b.items()}


def test_yam_certify_step_does_not_run_finalize(tmp_path):
    """finalize --write writes the human rates, report and docs: not for a yam run."""

    class Fin:
        def Paths(self, **kw):  # noqa: N802 - mimics the module's class
            return kw

        def fetch_records(self, paths, st):
            pass

        def certify_all(self, paths, st, out_dir, *, derived):
            return [{"stage": s, "cert": {"verdict": "certified",
                                          "metrics": {"witness": {"n": len(YAM.noise)}}}}
                    for s in wh.CERT_STAGES]

        def derived_content_used(self, st):
            return {}

        def missing_for_certified(self, c):
            return []

    calls = []
    st = wh.State("r", 0.0, "x")
    ctx = wh.Ctx(st, _manifest(YAM), tmp_path, state_path=tmp_path / "s.json", spec=YAM)
    res = wh.run_flora_certify(ctx, finalize=Fin(), sh=lambda *a, **k: calls.append(a))
    assert res["ok"] and calls == [] and "not run for yam" in res["finalize"]


def test_yam_noise_is_all_twelve_and_it_runs_the_isolated_caption_steps(no_lab):
    """v4 (pre-registration §3.3, yam): f on all 12 (no ``--limit``), d's cold witness the
    segment stage alone, g's J2 (12) and J2w (the 6 witnesses), priced single-pool with the
    three-view caption scale; the 3-view lab (d2) stays in the plan, skipped when done."""
    man = _manifest(YAM)
    dur = wh.durations(man)
    v = sum(dur[g] for g in YAM.episodes)
    wv = sum(dur[g] for g in YAM.witness)
    ps = wh.plan_projections(wh.CERTIFIED, man, spec=YAM, lab_done=True)
    assert "(12 ep" in ps["noise"].items[0]["what"]
    assert ps["noise"].usd == ps["exact_e2e"].usd  # the oracle's job, all of it
    wit = [i for i in ps["balanced"].items if "cold witness" in i["what"]]
    assert len(wit) == 1 and "segment-only" in wit[0]["what"]
    n, u = wh.single_pool_seconds("segment", wv, scale=YAM.scale)
    assert (wit[0]["seconds"], wit[0]["upper_seconds"]) == (round(n, 1), round(u, 1))
    j2, j2w = ps["caption_iso"].items
    assert "(12 ep" in j2["what"] and "(6 ep" in j2w["what"] and "3-view" in j2["what"]
    for item, vs in ((j2, v), (j2w, wv)):
        n, u = wh.single_pool_seconds("caption", vs, scale=YAM.scale)
        assert (item["seconds"], item["upper_seconds"]) == (round(n, 1), round(u, 1))
    assert ps["lab"].usd_upper == 0.0  # lab_done: the optional 3-view lab is skipped
    assert wh.plan_projections(wh.CERTIFIED, man, spec=YAM)["lab"].usd_upper > 0
    assert wh.cert_titles(YAM)["caption_iso"].startswith("g. caption-only isolated J2")
