"""``scripts/scale_test.py --profile lite`` (the default): the dry-run cost per profile, the
subset's determinism, the lite criteria, and ``--finalize`` folding the run's per-container
rates into ``rates.json`` as more samples. Nothing here contacts Modal or the Hub."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
RATES = Path(__file__).resolve().parents[2] / "src" / "flora_deploy" / "planner" / "rates.json"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load("scale_test")
wh = sys.modules["wave_h"]
wf = _load("wave_h_finalize")


@pytest.fixture(autouse=True)
def lite_profile():
    prev = sc.use_profile("lite")
    yield
    sc.use_profile(prev)


def _table(seg=0.367, cap=0.906, seg_start=300.5, cap_start=325.0):
    """Today's rates.json with the balanced rows pinned, so the numbers below do not move when
    a later run folds new samples into the real file."""
    from flora_deploy.planner.rates import RateTable

    d = json.loads(RATES.read_text())
    for r in d["rates"]:
        if r["preset"] == "balanced" and r["kind"] in ("segment", "caption"):
            r.update(container_s_per_video_s=seg if r["kind"] == "segment" else cap,
                     uncertainty=0.5)
    for r in d["startup"]:
        if r["preset"] == "balanced" and r.get("compile_cache") == "cold":
            r.update(startup_s=seg_start if r["kind"] == "segment" else cap_start,
                     uncertainty=0.5)
    return RateTable.from_dict(d, source="test")


# --------------------------------------------------------------------------------------------
# Profiles and the dry run


def test_lite_is_the_default_and_the_dry_run_starts_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = tmp_path / "state.json"
    assert sc.main(["--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert not state.exists() and "dry run: nothing started" in out
    assert "lite profile" in out and "first 32 of 128 episodes" in out
    assert "segment 2 x L40S, caption 2 x H100!" in out and "chaos none" in out
    assert "--limit" in out
    assert sc.PROFILE.name == "lite"  # main restored the profile it found
    sc.main(["--state", str(state), "--profile", "full"])
    out = capsys.readouterr().out
    assert "full profile" in out and "128 episodes" in out and "caption 3 x H100!" in out
    assert sc.state_path("real").parts[-3:] == ("lite", "real", "state.json")
    with sc.profile_ctx("full"):
        assert sc.state_path("real").parts[-3:] == ("scale-test", "real", "state.json")


def test_dry_run_cost_per_profile():
    t = _table()
    totals = {}
    for name in sc.PROFILES:
        with sc.profile_ctx(name):
            ps = sc.projections("real", t)
            for step, p in ps.items():  # honest caps in every profile and mode
                assert p.usd_upper <= sc.caps("real")[step] + 1e-9, (name, step)
            totals[name] = (sum(p.usd for p in ps.values()),
                            sum(p.usd_upper for p in ps.values()),
                            sc.run_wall_s("real", table=t), ps["run"])
    lite, full = totals["lite"], totals["full"]
    assert lite[0] == pytest.approx(2.77, abs=0.02) and lite[1] == pytest.approx(6.32, abs=0.02)
    assert full[3].usd == pytest.approx(11.46, abs=0.02)  # full is unchanged by the profiles
    assert full[3].usd_upper == pytest.approx(19.15, abs=0.02)
    assert lite[1] <= sc.LITE_UPPER_MAX_USD < full[1]
    assert lite[2] < full[2]
    whats = " | ".join(i["what"] for i in lite[3].items)
    assert "preemption" not in whats and "reconciler gap" not in whats  # lite: no chaos
    assert "segment: 2 cold starts" in whats and "caption: 2 cold starts" in whats
    with sc.profile_ctx("lite"):
        assert sc.max_gpus() == 4 and sc.INDEX_UNITS == 3
        assert sc.global_cap("real") <= sc.LITE_UPPER_MAX_USD + 0.2


def test_size_options_recommend_32_episodes_on_two_plus_two():
    rows = sc.size_options(table=_table())
    sizes = {(x["episodes"], x["segment"], x["caption"]) for x in rows}
    assert sizes == {(e, a, b) for e in sc.LITE_EPISODES for a, b in sc.LITE_POOLS}
    for x in rows:
        assert x["usd_upper"] > x["usd"] > 0 and x["wall_upper_s"] > x["wall_s"]
    rec = sc.recommend_size(rows)
    assert (rec["episodes"], rec["segment"], rec["caption"]) == (32, 2, 2)
    assert rec["usd_upper"] <= sc.LITE_UPPER_MAX_USD
    by = {(x["episodes"], x["segment"], x["caption"]): x for x in rows}
    assert by[(16, 2, 2)]["peak"]["caption"] < 2  # too few shards: one caption container idles
    assert by[(48, 2, 2)]["usd_upper"] > sc.LITE_UPPER_MAX_USD  # over the money left
    assert by[(32, 1, 2)]["segment"] < 2  # one segment container: no multi-container rate


# --------------------------------------------------------------------------------------------
# The subset


def test_the_subset_is_the_first_n_episodes_and_deterministic():
    d = sc.DATASET
    assert len(sc.EPISODE_FRAMES) == len(sc.EPISODE_FILE) == d["total_episodes"]
    assert sum(sc.EPISODE_FRAMES) == d["total_frames"]
    assert list(sc.EPISODE_FILE) == sorted(sc.EPISODE_FILE)  # source files in episode order
    assert sc.subset_video_s(d["total_episodes"]) == sc.FULL_VIDEO_S
    assert sc.VIDEO_S == pytest.approx(801.67, abs=0.01)  # 24,050 frames at 30 fps
    assert sc.index_limit_args() == ["--limit", "32"]
    with sc.profile_ctx("full"):
        assert sc.index_limit_args() == []
    a, b = sc.exact_plan(32, 2.0), sc.exact_plan(32, 2.0)
    assert a == b  # pure: the same shards every time
    assert sum(a["segment"]) == pytest.approx(sc.VIDEO_S)
    assert sum(v for v, _ in a["caption"]) == pytest.approx(sc.VIDEO_S)
    assert len(a["segment"]) >= 2 * sc.POOLS["segment"]["max_containers"]
    assert len(a["caption"]) >= 2 * sc.POOLS["caption"]["max_containers"]
    for _, deps in a["caption"]:
        assert deps and all(0 <= i < len(a["segment"]) for i in deps)
    # durations are spread, not cherry-picked: short and long episodes both well represented
    durs = [f / d["fps"] for f in sc.EPISODE_FRAMES[:32]]
    assert sum(x < 15 for x in durs) >= 8 and sum(x >= 30 for x in durs) >= 8
    assert max(durs) / min(durs) > 5 and statistics.pstdev(durs) > 10
    # the projection uses those exact shards; a different indexed length scales them
    assert sc.shard_counts() == {"segment": len(a["segment"]), "caption": len(a["caption"])}
    s2 = sc.shard_plan(2 * sc.VIDEO_S)
    assert sum(s2["segment"]) == pytest.approx(2 * sc.VIDEO_S)


def test_index_checks_it_got_the_subset():
    ok = {"episodes": 32, "indexed": 32}
    assert sc.subset_problems(ok, 801.9) == []
    assert sc.subset_problems({"episodes": 128, "indexed": 128}, sc.FULL_VIDEO_S)
    assert sc.subset_problems(ok, 900.0)  # another subset of the same count
    assert sc.subset_problems(ok, None)


def test_lite_config_matches_the_profile():
    import yaml

    cfg = yaml.safe_load(sc.config_text("real", project="p", run_id="r", budget_usd=6.3))
    assert cfg["modal"]["max_gpus"] == 4
    assert cfg["resources"]["caption"]["max_containers"] == 2
    assert cfg["pipeline"]["shard_target_gpu_minutes"] == 2
    assert sc.job_id_for("real", "abc") == "scale-lite-abc"
    with sc.profile_ctx("full"):
        assert sc.job_id_for("real", "abc") == "scale-abc"


# --------------------------------------------------------------------------------------------
# Criteria at lite size


def _lite_good_run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from test_scale_test_runbook import _good_run

    r = _good_run()
    r.update(profile="lite", chaos_enabled=[], chaos={}, orchestrator={})
    return r


def test_lite_marks_the_chaos_criteria_na_with_a_reason_and_judges_the_rest():
    rows = {x["id"]: x for x in sc.evaluate(_lite_good_run(), "real")}
    for i in (10, 11):
        assert rows[i]["verdict"] == "n.a." and "lite profile" in rows[i]["reason"]
    assert {rows[i]["verdict"] for i in rows if i not in (10, 11)} == {"PASS"}
    assert all(x["auto"] for x in rows.values())


def test_lite_proofs_fail_on_their_own():
    r = _lite_good_run()
    r["measure"]["containers"] = [c for c in r["measure"]["containers"]
                                  if not c["container_id"].endswith("-1")]
    assert {x["id"]: x["verdict"] for x in sc.evaluate(r, "real")}[15] == "FAIL"
    r = _lite_good_run()
    r["cache"] = {k: {"ok": None, "detail": "skipped", "states": {}} for k in sc.POOLS}
    row = {x["id"]: x for x in sc.evaluate(r, "real")}[18]
    assert row["verdict"] == "FAIL" and "not recorded" in row["reason"]
    r = _lite_good_run()
    r["batch_fill"]["caption"]["mean_batch"] = 1.0
    assert {x["id"]: x["verdict"] for x in sc.evaluate(r, "real")}[19] == "FAIL"


def test_batch_fill_and_cache_from_audit_rows():
    audits = [
        {"shard_audit": True, "shard_id": "s1", "stage": "segment",
         "batcher": {"rounds": 4, "mean_batch": 10.0, "max_batch": 20, "generate_s": 40.0,
                     "wall_s": 50.0}, "compile_cache": {"state": "warm"}},
        {"shard_audit": True, "shard_id": "s2", "stage": "segment",
         "batcher": {"rounds": 6, "mean_batch": 5.0, "max_batch": 9, "generate_s": 30.0,
                     "wall_s": 50.0}, "compile_cache": {"state": "warm"}},
        {"shard_audit": True, "shard_id": "c1", "stage": "caption",
         "batcher": {"rounds": 2, "mean_batch": 8.0, "max_batch": 16, "generate_s": 9.0,
                     "wall_s": 10.0}, "compile_cache": {"state": "cold"}},
        {"uid": "e1", "stage": "caption", "calls": {}},  # an episode row: not a shard row
    ]
    bf = sc.batch_fill(audits)
    assert bf["segment"]["mean_batch"] == 7.0  # (4 x 10 + 6 x 5) / 10 rounds
    assert bf["segment"]["busy_fraction"] == 0.7 and bf["segment"]["max_batch"] == 20
    assert bf["segment"]["fill"] == round(7.0 / sc.ENGINE_MAX_NUM_SEQS["segment"], 3)
    assert bf["caption"]["shards"] == 1
    c = sc.cache_by_kind(audits, [], warm_kinds=("segment",))
    assert c["segment"]["ok"] and c["segment"]["states"] == {"warm": 2}
    assert c["caption"]["ok"] and not c["caption"]["published_before"]
    c = sc.cache_by_kind(audits, [], warm_kinds=("caption",))  # published, yet compiled cold
    assert c["caption"]["ok"] is False
    assert sc.cache_by_kind([], [])["segment"]["ok"] is None  # nothing recorded


def test_container_ends_and_the_tail_watch():
    mon = sc.Monitor.__new__(sc.Monitor)
    mon.app, mon.containers, mon.kinds = "app", {"a": {"start": 0.0, "end": 100.0},
                                                 "b": {"start": 0.0, "end": 60.0}}, {"a": "x"}
    mon.last_poll = 100.0
    seen = mon.seen()
    assert seen["a"]["ended_at"] is None and seen["b"]["ended_at"] == 60.0
    assert mon.live_workers() == ["a"]
    polls = []

    def poll():
        polls.append(1)
        mon.containers["a"]["end"] = 100.0  # gone: not seen again
        mon.last_poll = 105.0
        return 0
    mon.poll_containers = poll
    mon.watch_tail(until_s=60, every_s=0.0, sleep=lambda s: None)
    assert polls == [1] and mon.live_workers() == []


# --------------------------------------------------------------------------------------------
# --finalize: fold the per-container rates into rates.json


def _finished_state(*, containers=2, state="done", states=("cold",)) -> wh.State:
    st = wh.State("lite01", 0.0, "")
    rep = st.step("report")
    rep.status = "done"
    rep.attempts.append(wh.Attempt(started=1.0e9, finished=1.7613e9, status="done"))
    conts, rows = [], []
    for kind, gpu, rate in (("segment", "L40S", 0.30), ("caption", "H100", 0.80)):
        for i in range(containers):
            cid = f"ta-{kind}-{i:04d}"
            video = 300.0 + 100 * i
            conts.append({"container_id": cid, "kind": kind, "shards": 3, "video_s": video,
                          "busy_s": video * (rate + 0.1 * i), "prewarmed": kind == "caption"
                          and i == 0})
            rows.append({"container_id": cid, "kind": kind, "gpu": gpu,
                         "gpu_seen": f"NVIDIA {gpu}", "startup_s": 200.0 + 20 * i,
                         "busy_s": video * (rate + 0.1 * i), "seconds": video + 400})
    rep.result = {"state": state, "records": {"missing": 0, "duplicates": 0},
                  "profile": "lite", "measure": {"containers": conts},
                  "report_containers": rows,
                  "cache": {k: {"states": dict.fromkeys(states, 3)} for k in sc.POOLS}}
    return st


def _args(tmp_path, st, rates, *, write=False):
    p = tmp_path / "state.json"
    wh.save_state(st, p)
    return argparse.Namespace(mode="real", state=p, rates=rates, write=write)


def _row(data, table, kind, cache="cold"):
    if table == "rates":
        return next(r for r in data["rates"] if r["preset"] == "balanced" and r["kind"] == kind)
    return next(r for r in data["startup"] if r["preset"] == "balanced" and r["kind"] == kind
                and r.get("compile_cache") == cache)


def test_finalize_folds_samples_and_keeps_the_certified_record(tmp_path):
    rates = tmp_path / "rates.json"
    rates.write_text(RATES.read_text())
    before = json.loads(rates.read_text())
    out: list[str] = []
    st = _finished_state()
    assert sc.finalize_main(_args(tmp_path, st, rates), out=out.append) == 0
    assert rates.read_text() == RATES.read_text()  # dry run: nothing written
    assert sc.finalize_main(_args(tmp_path, st, rates, write=True), out=out.append) == 0
    data = json.loads(rates.read_text())
    from flora_deploy.planner.rates import RateTable

    RateTable.from_dict(data, source="t")  # the loader accepts the folded file
    for kind, lo, hi in (("segment", 0.30, 0.40), ("caption", 0.80, 0.90)):
        old, new = _row(before, "rates", kind), _row(data, "rates", kind)
        base, *folds = new["samples"]
        # the committed record's video seconds, as its provenance states them
        video = float(re.search(r"\(([\d.]+) video s\)", old["provenance"]).group(1))
        # the certified run's record is kept verbatim as the base sample
        assert base["origin"] == "base" and base["value"] == old["container_s_per_video_s"]
        assert base["source"] == old["source"] and base["n"] == old["n"]
        assert base["provenance"] == old["provenance"] and base["video_s"] == video
        assert new["source"] == old["source"]
        assert [f["value"] for f in folds] == [lo, hi] and new["n"] == old["n"] + 2
        vals = [old["container_s_per_video_s"], lo, hi]
        assert new["uncertainty"] == round(max(wf.UNCERTAINTY_FLOOR,
                                               (max(vals) - min(vals)) / statistics.mean(vals)),
                                           2)
        pooled = ((old["container_s_per_video_s"] * video + lo * 300 + hi * 400)
                  / (video + 700))
        assert new["container_s_per_video_s"] == round(pooled, 3)
    # startups: cold row; the pre-warmed caption container gives no startup sample
    seg, cap = _row(data, "startup", "segment"), _row(data, "startup", "caption")
    assert [x["value"] for x in seg["samples"][1:]] == [200.0, 220.0] and seg["n"] == 3
    assert [x["value"] for x in cap["samples"][1:]] == [220.0] and cap["n"] == 2
    assert _row(data, "startup", "caption", "warm") == _row(before, "startup", "caption", "warm")
    # idempotent: folding the same run again changes nothing
    out.clear()
    assert sc.finalize_main(_args(tmp_path, st, rates, write=True), out=out.append) == 0
    assert any("nothing to change" in x for x in out)
    assert json.loads(rates.read_text()) == data


def test_a_warm_run_fills_the_warm_startup_row(tmp_path):
    rates = tmp_path / "rates.json"
    rates.write_text(RATES.read_text())
    st = _finished_state(states=("warm",))
    assert sc.finalize_main(_args(tmp_path, st, rates, write=True), out=lambda s: None) == 0
    data = json.loads(rates.read_text())
    w = _row(data, "startup", "segment", "warm")
    assert w["status"] == "measured" and w["n"] == 2 and w["startup_s"] == 210.0
    assert all(x["origin"] == "fold" for x in w["samples"])  # a placeholder is no record
    assert w["source"].startswith("reports/scale-test (scale test lite run lite01)")


def test_finalize_refuses_what_it_cannot_measure(tmp_path):
    rates = tmp_path / "rates.json"
    rates.write_text(RATES.read_text())
    out: list[str] = []
    assert sc.finalize_main(_args(tmp_path, _finished_state(containers=1), rates),
                            out=out.append) == 1
    assert any("at least 2" in x for x in out)
    out.clear()
    assert sc.finalize_main(_args(tmp_path, _finished_state(state="failed"), rates),
                            out=out.append) == 1
    assert any("'failed'" in x for x in out)
    st = _finished_state()
    for r in st.step("report").result["report_containers"]:
        r["gpu_seen"] = "NVIDIA A100"  # ran on another GPU: not this pool's rate
    out.clear()
    assert sc.finalize_main(_args(tmp_path, st, rates), out=out.append) == 1
    assert any("ran on" in x for x in out)
    a = _args(tmp_path, _finished_state(), rates)
    a.mode = "local"
    assert sc.finalize_main(a, out=lambda s: None) == 2
    assert rates.read_text() == RATES.read_text()


def test_a_later_certified_rewrite_keeps_the_folded_samples():
    data = json.loads(RATES.read_text())
    samples, _, refusals = sc.finalize_samples(_finished_state())
    assert not refusals
    folded = sc.folded_rates(data, samples)
    ms = {}
    for preset, kind in (("balanced", "segment"), ("balanced", "caption")):
        m = wf.Measurement(preset, kind, "L40S" if kind == "segment" else "H100")
        m.samples = [("j1", 40.0, 100.0), ("j2", 50.0, 100.0)]
        m.startups = [290.0, 310.0]
        ms[(preset, kind)] = m
    ms[("exact", "segcap")] = wf.Measurement("exact", "segcap", "H200")
    new = wf.updated_rates(folded, ms, run_id="cert2", date="2026-11-01", source="cert2 doc")
    seg = _row(new, "rates", "segment")
    base, *rest = seg["samples"]
    assert base["source"] == "cert2 doc" and base["n"] == 2 and base["range"] == [0.4, 0.5]
    assert [x["value"] for x in rest] == [0.3, 0.4] and seg["n"] == 4
    again = wf.updated_rates(copy.deepcopy(new), ms, run_id="cert2", date="2026-11-01",
                             source="cert2 doc")
    assert again == new  # idempotent
    # without folded samples the certified rewrite is what it always was (no samples key)
    plain = wf.updated_rates(json.loads(RATES.read_text()), ms, run_id="c", date="d",
                             source="s")
    assert "samples" not in _row(plain, "rates", "segment")


def test_fold_refuses_a_rate_sample_without_video_seconds():
    row = {"preset": "balanced", "kind": "segment", "container_s_per_video_s": 0.4,
           "status": "measured", "n": 1, "source": "s", "date": "d", "provenance": "no length"}
    with pytest.raises(wf.FoldRefused):
        wf.fold_samples(row, [{"source": "x", "value": 0.3, "video_s": 100.0, "n": 1}],
                        kind="rate")
