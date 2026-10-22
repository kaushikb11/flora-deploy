"""``scripts/h0_rehearsal.py``'s checks of the behaviours H0 had never run on Modal: job-end
scale-to-zero (observed container lifetimes vs the recorded drains), the caption pre-warm, the
batch-fill audit fields and compile-cache provenance. Fake job reports and a local run of the
deployed balanced fakes; nothing here contacts Modal."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flora_deploy.jobs.containers import read_startup_facts, write_startup_facts
from flora_deploy.jobs.pools import PoolScaler, drains_path, read_prewarm_plans
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.jobs.types import Shard

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
APP = "flora-h0-r1-bal-human-fake-abc"
CLASSES = {"segment": "FakeSegmentWorker", "caption": "FakeCaptionWorker",
           "reward.terminal": "FakeRewardWorker", "reward.dense": "FakeRewardWorker"}


@pytest.fixture(scope="module")
def h0():
    if "h0_rehearsal" in sys.modules:
        return sys.modules["h0_rehearsal"]
    spec = importlib.util.spec_from_file_location("h0_rehearsal", SCRIPTS / "h0_rehearsal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h0_rehearsal"] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def _seen(**ends):
    """Watch records: ``ta-<name>=(started_at, ended_at | None)``."""
    return {f"ta-{k}": {"task_id": f"ta-{k}", "app": APP, "started_at": s, "ended_at": e,
                        "first_seen": s, "last_seen": e or 2000.0} for k, (s, e) in ends.items()}


ROWS = [{"container_id": "ta-seg", "kind": "segment", "started_utc": None},
        {"container_id": "ta-cap", "kind": "caption", "started_utc": None},
        {"container_id": "ta-rew", "kind": "reward.terminal", "started_utc": None}]
# segment drained mid-job at 500; caption and reward at the job end (900)
DRAINS = {"segment": ((500.0, None),), "caption": ((900.0, None),),
          "reward.terminal": ((900.0, None),), "reward.dense": ((900.0, None),)}


# ----------------------------------------------------------------------------------------------
# drain


def test_drain_passes_when_every_container_left_soon_after_its_class_drained(h0):
    seen = _seen(seg=(100.0, 505.0), cap=(300.0, 912.0), rew=(600.0, 925.0),
                 orch=(50.0, 903.0))
    rows = h0.drain_rows(APP, ROWS, seen, DRAINS, CLASSES, observed_until=1000.0)
    by = {r["container_id"]: r for r in rows}
    assert by["ta-seg"]["gap_s"] == 5.0 and by["ta-seg"]["drain_at"] == 500.0
    assert by["ta-cap"]["gap_s"] == 12.0 and by["ta-rew"]["gap_s"] == 25.0
    assert by["ta-orch"]["class"] == "other" and by["ta-orch"]["gap_s"] == 3.0
    ok, detail = h0.drain_check(rows)
    assert ok, detail
    assert "3 worker container(s) measured" in detail and "seg segment: 5 s" in detail


def test_drain_fails_when_an_idle_container_kept_the_old_window(h0):
    # the question H0 answers: the lowered window did not reach the already-idle segment
    # container, which stayed for the deployed 300 s (still alive when the watch stopped)
    seen = _seen(seg=(100.0, None), cap=(300.0, 905.0), rew=(600.0, 910.0))
    rows = h0.drain_rows(APP, ROWS, seen, DRAINS, CLASSES, observed_until=945.0)
    seg = next(r for r in rows if r["container_id"] == "ta-seg")
    assert seg["alive"] and seg["gap_s"] == 445.0 and not seg["ok"]
    ok, detail = h0.drain_check(rows)
    assert not ok and "seg segment: >=445 s" in detail
    late = h0.drain_rows(APP, ROWS, _seen(seg=(100.0, 540.0), cap=(300.0, 905.0)), DRAINS,
                         CLASSES, observed_until=945.0)
    assert not h0.drain_check(late)[0]  # 40 s > the 30 s bound


def test_a_preempted_container_that_ended_before_any_drain_passes(h0):
    rows_in = [*ROWS, {"container_id": "ta-seg0", "kind": "segment"}]
    seen = _seen(seg0=(100.0, 200.0), seg=(210.0, 503.0), cap=(300.0, 905.0),
                 rew=(600.0, 910.0))
    rows = h0.drain_rows(APP, rows_in, seen, DRAINS, CLASSES, observed_until=945.0)
    first = next(r for r in rows if r["container_id"] == "ta-seg0")
    assert first["gap_s"] is None and first["ok"]
    assert h0.drain_check(rows)[0]


def test_drain_uses_the_latest_drain_before_the_end_after_a_restore(h0):
    drains = {**DRAINS, "segment": ((300.0, 400.0), (700.0, None))}  # drained, respawned
    seen = _seen(seg=(100.0, 710.0), cap=(300.0, 905.0), rew=(600.0, 910.0))
    rows = h0.drain_rows(APP, ROWS, seen, drains, CLASSES, observed_until=945.0)
    assert next(r for r in rows if r["kind"] == "segment")["gap_s"] == 10.0


def test_drain_fails_without_any_recorded_drain_or_other_apps_containers(h0):
    seen = _seen(seg=(100.0, 505.0), orch=(50.0, 903.0))
    seen["ta-other-app"] = {"task_id": "ta-other-app", "app": "flora-real-app",
                            "started_at": 1.0, "ended_at": None}
    rows = h0.drain_rows(APP, ROWS, seen, {}, CLASSES, observed_until=945.0)
    assert {r["container_id"] for r in rows} == {"ta-seg", "ta-orch"}  # this app only
    ok, detail = h0.drain_check(rows)
    assert not ok and "no worker container measured" in detail


def test_container_watch_records_lifetimes_from_task_list_polls(h0):
    t = [0.0]
    polls = [[{"task_id": "ta-a", "app": APP, "started_at": 1.0},
              {"task_id": "ta-x", "app": "someone-else", "started_at": 1.0}],
             [{"task_id": "ta-a", "app": APP, "started_at": 1.0},
              {"task_id": "ta-b", "app": APP, "started_at": 6.0}],
             RuntimeError("api hiccup"),
             [{"task_id": "ta-b", "app": APP, "started_at": 6.0}],
             []]

    def list_fn():
        p = polls.pop(0)
        if isinstance(p, Exception):
            raise p
        return p

    w = h0.ContainerWatch(lambda app: app == APP, list_fn=list_fn, clock=lambda: t[0])
    for now in (5.0, 10.0, 15.0, 20.0):
        t[0] = now
        w.poll()
    assert w.errors == 1 and sorted(w.seen) == ["ta-a", "ta-b"]
    assert w.seen["ta-a"]["ended_at"] == 20.0 and w.seen["ta-a"]["last_seen"] == 10.0
    assert w.alive() == ["ta-b"]
    t[0] = 25.0
    w.wait_drained(100.0, sleep=lambda s: (w.poll(), t.__setitem__(0, t[0] + s)))
    assert w.alive() == [] and w.seen["ta-b"]["ended_at"] == 25.0 and t[0] < 100.0


# ----------------------------------------------------------------------------------------------
# pre-warm


def _recorded_prewarm(tmp_path):
    """A real PoolScaler records its caption pre-warm plan and start in the job's pool file."""
    store = LocalStore(tmp_path / "store")
    path = drains_path(JobPaths("job").root, "cfg")
    clock = [1000.0]

    class Auto:
        def set_scaledown(self, *a):
            pass

        def set_min_containers(self, *a):
            pass

    s = PoolScaler(job_id="job", classes=CLASSES, static_s=300, autoscaler=Auto(), kv=_KV(),
                   clock=lambda: clock[0], store=store, path=path,
                   startup_s={"segment": 300.0, "caption": 325.0}.__getitem__,
                   work_s=lambda sh: 10.0)
    shards = [Shard("segment-0", "segment", ("a",), 60.0),
              Shard("caption-0", "caption", ("a",), 60.0, depends_on=("segment-0",))]
    s.tick_prewarm(shards, satisfied=set(), live_spawned={"segment-0": 990.0},
                   called_kinds={"segment"})
    return store, s


class _KV(dict):
    def put(self, k, v):
        self[k] = v

    def items(self):
        return list(super().items())


def test_prewarm_plan_is_persisted_and_reloaded(tmp_path):
    store, s = _recorded_prewarm(tmp_path)
    plans = read_prewarm_plans(store, JobPaths("job").root, "cfg")
    assert plans["caption"]["decision"] == "prewarm" and plans["caption"]["t"] == 1000.0
    assert plans["caption"]["start_at"] == 990.0 + 300.0 + 10.0 - 325.0
    again = PoolScaler(job_id="job", classes=CLASSES, static_s=300, autoscaler=None, kv=_KV(),
                       store=store, path=drains_path(JobPaths("job").root, "cfg"))
    assert again._prewarm_plan["caption"]["decision"] == "prewarm"  # not decided again
    assert "t" not in again._prewarm_plan["caption"]


def test_prewarm_check_on_a_recorded_plan(h0, tmp_path):
    store, s = _recorded_prewarm(tmp_path)
    plans = read_prewarm_plans(store, JobPaths("job").root, "cfg")
    prewarms = s.prewarms()
    seen = {"ta-cap": {"started_at": 1040.0}}
    rows = [{"container_id": "ta-cap", "kind": "caption", "started_utc": None}]
    ok, detail = h0.prewarm_check(plans, prewarms, rows, seen)
    assert ok, detail
    assert "+40 s" in detail  # planned start in the past: "now" (the decision, 1000)
    late = h0.prewarm_check(plans, prewarms, rows, {"ta-cap": {"started_at": 1400.0}})
    assert not late[0]
    # falls back to the attempt marker's time when Modal's start was not seen
    marker = [{"container_id": "ta-cap", "kind": "caption",
               "started_utc": "1970-01-01T00:17:30.000000Z"}]  # 1050
    assert h0.prewarm_check(plans, prewarms, marker, {})[0]


def test_prewarm_check_fails_without_a_plan_a_start_or_a_container(h0):
    rows = [{"container_id": "ta-cap", "kind": "caption"}]
    seen = {"ta-cap": {"started_at": 10.0}}
    assert not h0.prewarm_check({}, {}, rows, seen)[0]
    skip = {"caption": {"decision": "skip", "reason": "its first shard is runnable now"}}
    ok, detail = h0.prewarm_check(skip, {}, rows, seen)
    assert not ok and "runnable now" in detail
    plan = {"caption": {"decision": "prewarm", "start_at": 5.0, "ready_at": 50.0, "t": 1.0}}
    assert not h0.prewarm_check(plan, {}, rows, seen)[0]  # never started
    assert not h0.prewarm_check(plan, {"caption": (5.0, 9.0)}, [], seen)[0]
    assert h0.prewarm_check(plan, {"caption": (5.0, 9.0)}, rows, seen)[0]


# ----------------------------------------------------------------------------------------------
# batch fill: a local run of the deployed balanced fakes, with resource sampling on as on Modal


@pytest.fixture(scope="module")
def recorded_bal(tmp_path_factory):
    from flora_deploy.config.schema import load_config
    from flora_deploy.local import balanced as LB
    from flora_deploy.local import sample_dataset as make_sample
    from flora_deploy.modal_app import factory as F
    from tests.unit.test_balanced_deps import balanced_yaml
    from tests.unit.test_h0_balanced_fakes import TEMPLATE, _DeployedFakes

    d = tmp_path_factory.mktemp("h0new")
    text = balanced_yaml(TEMPLATE.read_text()).replace(
        "stage_to: null", "stage_to: volume://flora-h0-media/run/videos")
    (d / "flora.yaml").write_text(text)
    make_sample(d / "videos")
    cfg = load_config(d / "flora.yaml")
    plan = F.plan_app(cfg, fake_models=True)
    env = dict(plan.object("FakeSegmentWorker").env)
    store_dir = d / "store"
    env.update({"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(store_dir)})
    mp = pytest.MonkeyPatch()
    mp.setattr(LB, "fake_balanced_runner",
               functools.partial(LB.fake_balanced_runner, sample_resources=True))
    r = _DeployedFakes(cfg, config_path=d / "flora.yaml", store=store_dir, env=env,
                       app_plan=plan)
    try:
        r.index()
        r.plan()
        outcome = r.orchestrate()
        r.compose(outcome)
    finally:
        r.close()
        mp.undo()
    return LocalStore(store_dir), r.job_id


def test_batch_fill_fields_of_a_recorded_balanced_run(h0, recorded_bal):
    store, job_id = recorded_bal
    audits = h0.job_audits(store, JobPaths(job_id).root)
    facts = read_startup_facts(store, job_id)
    assert {f["kind"] for f in facts.values()} == {"segment", "caption"}
    ok, detail = h0.batch_fill_check(audits, facts)
    assert ok, detail
    assert "GPU memory n/a (CPU)" in detail and "peak RSS" in detail


def test_batch_fill_check_catches_missing_or_wrong_fields(h0, recorded_bal):
    store, job_id = recorded_bal
    audits = h0.job_audits(store, JobPaths(job_id).root)
    facts = read_startup_facts(store, job_id)
    no_res = [{k: v for k, v in a.items() if k != "resources"} for a in audits]
    assert not h0.batch_fill_check(no_res, facts)[0]
    one_phase = [{**a, "phased": False} if a.get("shard_audit") else a for a in audits]
    assert not h0.batch_fill_check(one_phase, facts)[0]
    wrong = [{**a, "batch_policy": "0" * 16} if not a.get("shard_audit") and "uid" in a
             and a.get("stage") == "caption" else a for a in audits]
    ok, detail = h0.batch_fill_check(wrong, facts)
    assert not ok and "episode batch_policy" in detail
    ok, detail = h0.batch_fill_check(audits, {})
    assert not ok and "no segment startup facts" in detail
    gpu = {**next(a for a in audits if a.get("shard_audit"))["resources"],
           "gpu_mem_used_max_mib": 90000.0, "gpu_mem_total_mib": 81559.0}
    assert not h0.resources_ok(gpu)[0]
    assert h0.resources_ok({**gpu, "gpu_mem_used_max_mib": 70000.0})[0]


def test_startup_facts_round_trip(tmp_path):
    store = LocalStore(tmp_path)
    write_startup_facts(store, "job", "ta-1", "segment", {"phased": True}, now=0.0)
    store.put_bytes(f"{JobPaths('job').root}/containers/junk.json", b"{not json")
    got = read_startup_facts(store, "job")
    assert list(got) == ["ta-1"] and got["ta-1"]["startup"] == {"phased": True}
    assert got["ta-1"]["written_utc"].startswith("1970-01-01T00:00:00")


# ----------------------------------------------------------------------------------------------
# compile-cache provenance (Fix-13) and the projection


def test_cache_provenance_is_skipped_until_shards_record_it(h0, recorded_bal):
    store, job_id = recorded_bal
    audits = h0.job_audits(store, JobPaths(job_id).root)
    ok, detail = h0.cache_provenance_check(audits, [])
    assert ok is None and "skipped" in detail
    rows = [a for a in audits if a.get("shard_audit")]
    some = [{**a, "compile_cache": {"state": "warm"}} if i == 0 else a
            for i, a in enumerate(rows)]
    ok, detail = h0.cache_provenance_check(some, [])
    assert ok is False and "missing" in detail
    every = [{**a, "compile_cache": {"state": "cold", "reason": "no cache"}} for a in rows]
    assert h0.cache_provenance_check(every, [])[0] is True
    receipts = [{"kind": a["stage"], "shard_id": a["shard_id"], "compile_cache": "warm"}
                for a in rows]
    assert h0.cache_provenance_check(rows, receipts)[0] is True
    assert h0.cache_provenance_check(rows, [{**receipts[0], "compile_cache": {}}])[0] is False


def test_projected_cost_of_the_default_balanced_run(h0, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("human", "yam"):
        (tmp_path / d).mkdir()
    bal = {j.name: h0.plans(h0.write_config(tmp_path, j.name, "r1"))["fake"]
           for j in h0.JOBS if j.preset == "balanced"}
    proj = h0.projected_cost(bal)
    assert 0.3 <= proj["usd"] <= 0.5, proj
    assert proj["usd_undrained"] > proj["usd"]
    # the preempted jobs need one more container than the clean one
    assert proj["jobs"]["bal-yam-seg"] > proj["jobs"]["bal-human"]
    assert json.dumps(proj)


def test_balanced_is_the_default_preset(h0, monkeypatch, tmp_path):
    got = {}
    orig = h0.argparse.ArgumentParser.parse_args

    def capture(self, argv=None, namespace=None):
        got["args"] = orig(self, argv, namespace)
        raise SystemExit(0)  # stop before Modal is contacted

    monkeypatch.setattr(h0.argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        h0.main(["--work", str(tmp_path)])
    assert got["args"].preset == "balanced"
