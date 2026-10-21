"""G3 (``docs/exact-cost.md``): exact ``segcap`` longest-first episode order leaves every model
input unchanged, keeps the ``ShardWorker`` contract, is off by default and reaches the worker
(config -> factory env -> worker settings). G3's shard overlap was removed (review 7b).

The engine is the production ``FastExactEngine`` composite over the fake NVIDIA family of
``test_reference_concurrency`` (NVIDIA's call structure; its ``generate`` logs NVIDIA's
``input_tensor_digest``), as in ``test_segcap_worker``."""

from __future__ import annotations

import json
import threading
import time

import pytest

from flora_deploy.accel.identity import compare_runs
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.attempts import JobFiles
from flora_deploy.jobs.manifest import save_manifest
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard
from flora_deploy.workers_impl.segcap import (
    SegcapShardRunner,
    SegcapWorkerSettings,
    VideoResolver,
    order_episodes,
    scheduling_from_env,
)
from tests.conformance.registry import segcap_fake_engine
from tests.conformance.worker_suite import Script
from tests.unit.test_segcap_worker import CFG, DS, JOB, PATHS, receipt, uid

# two shards; durations chosen so that plan order != longest-first order
SHARDS = {"segcap-a": [("a0", 3.0), ("a1", 9.0), ("a2", 5.0)],
          "segcap-b": [("b0", 2.0), ("b1", 7.0), ("b2", 4.0), ("b3", 8.0)]}


def world(tmp_path, shards=SHARDS, store=None):
    store = store or LocalStore(tmp_path / "store")
    refs, out = [], []
    for sid, eps in shards.items():
        uids = []
        for eid, dur in eps:
            r = EpisodeRef(uid(eid), DS, eid, "human", dur, True, True, PATHS.index(uid(eid)))
            store.put_bytes(r.index_ref, json.dumps(receipt(eid)).encode())
            refs.append(r)
            uids.append(r.uid)
        out.append(Shard(sid, "segcap", tuple(uids), 1.0))
    save_manifest(store, Manifest(JOB, CFG, tuple(refs), tuple(out),
                                  "2026-10-21T00:00:00.000000Z"))
    return store, out


def make_runner(tmp_path, store, *, order="plan", engine=None, sleeps=None, container="ta-1",
                starts=None):
    eng = engine or segcap_fake_engine(Script(), concurrency=2, sleeps=sleeps)
    if starts is not None:  # record the start order NVIDIA's annotate sees
        inner = eng.nv.pipeline.annotate

        def annotate(videos, *a, **k):
            starts.append((time.perf_counter(), str(videos["main"].path)))
            return inner(videos, *a, **k)

        eng.nv.pipeline.annotate = annotate
    s = SegcapWorkerSettings(container_id=container, gpu="H200", models={"tempseg": "rev"},
                             media_root="/media", local_dir=str(tmp_path / "local"),
                             episode_order=order)
    resolver = VideoResolver("/media", tmp_path / "cuts", cutter=lambda src, dest: dest)
    return SegcapShardRunner(store, eng, s, call_id=lambda: "fc-1", resolver=resolver), eng


def stored(store, shards):
    """uid -> (record JSONs, model_calls) from the store, for every shard."""
    jf = JobFiles(store, JOB, CFG)
    out: dict[str, dict] = {}
    for sh in shards:
        for p in P.shard_parts(jf, sh).parts:
            for r in P.read_part(jf, p):
                out.setdefault(r.uid, {"records": set(), "calls": None})["records"].add(
                    r.to_json())
        prefix = PATHS.audit(CFG, "segcap", sh.shard_id, "x", 0)[:-len("x-0.jsonl")]
        for p in store.list(prefix):
            for line in store.get_bytes(p).decode().splitlines():
                if line.strip():
                    a = json.loads(line)
                    out[a["uid"]]["calls"] = a["result"]["flora_reference"]["model_calls"]
                    out[a["uid"]]["scheduling"] = a["runtime"]["scheduling"]
    return out


def results(st):
    """uid -> persisted ``model_calls`` (``accel.identity.compare_runs`` input)."""
    return {u: v["calls"] for u, v in st.items()}


# --- ordering -------------------------------------------------------------------------------


def test_longest_first_is_lpt_with_plan_order_ties():
    refs = {u: EpisodeRef(u, DS, u, "human", d, True, True, "i") for u, d in
            [("x", 2.0), ("y", 9.0), ("z", 2.0), ("w", 5.0)]}
    assert order_episodes(["x", "y", "z", "w"], refs, "plan") == ["x", "y", "z", "w"]
    assert order_episodes(["x", "y", "z", "w"], refs, "longest_first") == ["y", "w", "x", "z"]
    with pytest.raises(ValueError):
        order_episodes(["x"], refs, "shortest_first")


def test_scheduling_env_defaults_and_validation():
    assert scheduling_from_env({}) == {"episode_order": "plan"}
    env = {"FLORA_SEGCAP_SCHEDULING": json.dumps({"episode_order": "longest_first"})}
    assert scheduling_from_env(env) == {"episode_order": "longest_first"}
    with pytest.raises(ValueError):
        scheduling_from_env({"FLORA_SEGCAP_SCHEDULING": '{"episode_order": "random"}'})


def test_defaults_are_off_and_overlap_is_gone():
    assert SegcapWorkerSettings().episode_order == "plan"
    assert not hasattr(SegcapWorkerSettings(), "max_shards")


def test_longest_first_starts_longest_and_keeps_inputs(tmp_path):
    base_store, shards = world(tmp_path / "base")
    r, _ = make_runner(tmp_path / "base", base_store)
    for sh in shards:
        r.run_shard(JOB, sh.shard_id, CFG)

    starts: list = []
    store, _ = world(tmp_path / "lpt")
    r2, eng = make_runner(tmp_path / "lpt", store, order="longest_first", starts=starts)
    eng.concurrency = 1  # sequential: the start order is the order
    receipt_a = r2.run_shard(JOB, "segcap-a", CFG)
    assert [p.rsplit("/", 1)[-1] for _, p in starts] == ["a1-main.mp4", "a2-main.mp4",
                                                          "a0-main.mp4"]
    assert receipt_a.episodes_done == shards[0].episodes  # receipt keeps plan order
    r2.run_shard(JOB, "segcap-b", CFG)

    a, b = stored(base_store, shards), stored(store, shards)
    assert {u: v["records"] for u, v in a.items()} == {u: v["records"] for u, v in b.items()}
    ident = compare_runs(results(a), results(b))
    assert ident["ok"] and ident["paired"] > 0 and not ident["violations"], ident
    assert all(v["scheduling"]["episode_order"] == "longest_first" for v in b.values())


# --- Volume refresh guard -------------------------------------------------------------------


class _VolumeLike(LocalStore):
    """A store with a mount and a refresh (``VolumeStore``'s surface) that counts reloads."""

    def __init__(self, root):
        super().__init__(root)
        self.mount = root
        self.reloads = 0

    def refresh(self):
        self.reloads += 1
        return True


def test_refresh_is_skipped_while_episode_threads_still_read_store_media(tmp_path):
    """Review 7b G3-5: the guard keys on live episode threads (a cancelled or interrupted
    shard's in-flight episodes outlive its run_shard), not on running shards."""
    store = _VolumeLike(tmp_path / "vol")
    store, shards = world(tmp_path, store=store)
    eng = segcap_fake_engine(Script(), concurrency=2)
    s = SegcapWorkerSettings(container_id="ta-1", gpu="H200", models={"tempseg": "rev"},
                             media_root=str(tmp_path / "vol" / "media"),
                             local_dir=str(tmp_path / "local"))
    resolver = VideoResolver(s.media_root, tmp_path / "cuts", cutter=lambda src, dest: dest)
    r = SegcapShardRunner(store, eng, s, call_id=lambda: "fc", resolver=resolver)
    release = threading.Event()
    zombie = threading.Thread(target=release.wait, name="flora-batch", daemon=True)
    zombie.start()
    eng._thread = zombie  # an earlier batch's episodes are still running
    threading.Timer(0.5, release.set).start()  # ... and finish while this shard waits
    r.run_shard(JOB, "segcap-a", CFG)
    assert store.reloads == 0 and r.skipped_refreshes == 1
    zombie.join(5)
    r.run_shard(JOB, "segcap-b", CFG)  # no live episodes: reload as usual
    assert store.reloads == 1 and r.skipped_refreshes == 1


# --- config ---------------------------------------------------------------------------------


def test_exact_scheduling_config_is_validated_and_not_hashed():
    from flora_deploy.config.schema import exact_scheduling, parse_config
    from tests.unit.test_config_hashing import base, h, set_, sh

    d = base()
    cfg = parse_config(d, allow_uncertified=True)
    assert exact_scheduling(cfg) == {"episode_order": "plan"}
    on = set_(d, "pipeline.exact_scheduling", {"episode_order": "longest_first"})
    cfg_on = parse_config(on, allow_uncertified=True)
    assert exact_scheduling(cfg_on) == {"episode_order": "longest_first"}
    # start order leaves every model input unchanged: not in any hash
    assert h(on) == h(d) and sh(on) == sh(d)
    with pytest.raises(ValueError):  # overlap is gone (review 7b), and so is its knob
        parse_config(set_(d, "pipeline.exact_scheduling", {"shard_overlap": True}),
                     allow_uncertified=True)
    with pytest.raises(ValueError, match="episode_concurrency"):  # H2: one input/container
        parse_config(set_(d, "resources", {"segcap": {"max_inputs": 2}}),
                     allow_uncertified=True)
    with pytest.raises(ValueError):
        parse_config(set_(d, "pipeline.exact_scheduling", {"episode_order": "random"}),
                     allow_uncertified=True)
    rewards_only = set_(set_(d, "pipeline.stages", ["reward.terminal"]),
                        "pipeline.exact_scheduling", {"episode_order": "longest_first"})
    with pytest.raises(ValueError, match="segcap worker only"):
        parse_config(rewards_only, allow_uncertified=True)


def test_longest_first_reaches_the_worker():
    """Wired end to end: config -> factory env -> worker settings; segcap max_inputs stays 1."""
    from flora_deploy.config.schema import parse_config
    from flora_deploy.modal_app import factory as f
    from flora_deploy.modal_app import reference_app as ra
    from tests.unit.test_factory import RECEIPTS, template

    d = template("exact_mp4_dir.yaml")
    d["pipeline"]["exact_scheduling"] = {"episode_order": "longest_first"}
    plan = f.plan_app(parse_config(d), derived_receipts=RECEIPTS)
    (spec,) = [o for o in plan.objects if o.kind == "cls" and "segcap" in o.shard_kinds]
    assert spec.max_inputs is None  # no @modal.concurrent: one shard per container
    assert ra.segcap_settings_from_env(spec.env, gpu="H200").episode_order == "longest_first"
    assert ra.segcap_settings_from_env({}, gpu=None).episode_order == "plan"
