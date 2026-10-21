"""G3 (``docs/exact-cost.md``): exact ``segcap`` scheduling — longest-first episode order and
shard overlap (several shards per container sharing the episode slots) — leaves every model
input unchanged, keeps the ``ShardWorker`` contract per shard, and is off by default.

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
from flora_deploy.jobs.attempts import JobFiles, read_attempt
from flora_deploy.jobs.manifest import save_manifest
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard
from flora_deploy.stages import reference as ref_stage
from flora_deploy.stages.reference import EpisodeNotStarted, EpisodeSlots
from flora_deploy.workers_impl.segcap import (
    ConcurrentShardRefused,
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


def make_runner(tmp_path, store, *, order="plan", max_shards=1, overlap=False, engine=None,
                sleeps=None, container="ta-1", starts=None):
    eng = engine or segcap_fake_engine(Script(), concurrency=2, sleeps=sleeps)
    if overlap:
        eng.overlap = True
        eng.__post_init__()
    if starts is not None:  # record the start order NVIDIA's annotate sees
        inner = eng.nv.pipeline.annotate

        def annotate(videos, *a, **k):
            starts.append((time.perf_counter(), str(videos["main"].path)))
            return inner(videos, *a, **k)

        eng.nv.pipeline.annotate = annotate
    s = SegcapWorkerSettings(container_id=container, gpu="H200", models={"tempseg": "rev"},
                             media_root="/media", local_dir=str(tmp_path / "local"),
                             episode_order=order, max_shards=max_shards)
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
    assert scheduling_from_env({}) == {"episode_order": "plan", "max_shards": 1}
    env = {"FLORA_SEGCAP_SCHEDULING": json.dumps({"episode_order": "longest_first",
                                                  "max_shards": 2})}
    assert scheduling_from_env(env) == {"episode_order": "longest_first", "max_shards": 2}
    with pytest.raises(ValueError):
        scheduling_from_env({"FLORA_SEGCAP_SCHEDULING": '{"max_shards": 9}'})


# --- EpisodeSlots ---------------------------------------------------------------------------


def test_slots_cap_running_and_grant_the_earliest_key_first():
    slots = EpisodeSlots(1)
    slots.acquire((0, 0))
    got: list = []

    def take(key):
        slots.acquire(key)
        got.append(key)
        slots.release()

    late = threading.Thread(target=take, args=((2, 0),))
    late.start()
    time.sleep(0.05)
    early = threading.Thread(target=take, args=((1, 5),))
    early.start()
    time.sleep(0.05)
    assert got == []  # capacity 1 is held
    slots.release()
    late.join(2)
    early.join(2)
    assert got == [(1, 5), (2, 0)]  # an earlier batch's episode always goes first
    assert slots.stats()["max_running"] == 1 and slots.stats()["waiting"] == 0


def test_slots_cancel_gives_up_and_leaves_no_waiter():
    slots = EpisodeSlots(1)
    slots.acquire((0, 0))
    stop = threading.Event()
    box = {}

    def wait():
        try:
            slots.acquire((1, 0), stop.is_set)
        except EpisodeNotStarted as e:
            box["e"] = e

    t = threading.Thread(target=wait)
    t.start()
    stop.set()
    t.join(2)
    assert isinstance(box.get("e"), EpisodeNotStarted)
    assert slots.stats()["waiting"] == 0
    slots.release()
    slots.acquire((2, 0))  # still usable
    slots.release()


# --- runner ---------------------------------------------------------------------------------


def test_defaults_are_off_and_overlap_needs_an_overlap_engine(tmp_path):
    assert SegcapWorkerSettings().episode_order == "plan"
    assert SegcapWorkerSettings().max_shards == 1
    store, _ = world(tmp_path)
    with pytest.raises(ValueError, match="max_shards=2"):
        make_runner(tmp_path, store, max_shards=2)  # engine without overlap
    with pytest.raises(ValueError, match="concurrency >= 2"):
        eng = segcap_fake_engine(Script(), concurrency=1)
        eng.overlap = True
        eng.__post_init__()


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


def test_overlapped_shards_share_slots_and_keep_every_input(tmp_path):
    """Baseline: one shard after the other, plan order, no overlap. Overlap: both shards in one
    container at once (two ``run_shard`` threads, as ``@modal.concurrent`` delivers them),
    longest first, two shared episode slots. Same records, same model-call input digests; the
    second shard's episodes start while the first one is still running; never more than two
    episodes at once."""
    base_store, shards = world(tmp_path / "base")
    r, _ = make_runner(tmp_path / "base", base_store)
    for sh in shards:
        r.run_shard(JOB, sh.shard_id, CFG)

    store, _ = world(tmp_path / "ov")
    starts: list = []
    # a1 (the longest, so first) runs long: shard b's episodes must take the freed slot
    r2, eng = make_runner(tmp_path / "ov", store, order="longest_first", max_shards=2,
                          overlap=True, sleeps={"a1-": 0.6}, starts=starts)
    out: dict = {}

    def go(sid):
        out[sid] = r2.run_shard(JOB, sid, CFG)

    ta = threading.Thread(target=go, args=("segcap-a",))
    ta.start()
    time.sleep(0.05)
    tb = threading.Thread(target=go, args=("segcap-b",))
    tb.start()
    ta.join(30)
    tb.join(30)
    assert out["segcap-a"] is not None and out["segcap-b"] is not None
    assert len(out["segcap-a"].episodes_done) == 3 and len(out["segcap-b"].episodes_done) == 4
    # overlap happened: some b episode started before shard a's last episode finished
    first_b = min(t for t, p in starts if "/b" in p)
    a_end = max(t for t, p in starts if "/a" in p)
    assert first_b < a_end + 0.6
    assert eng._slots.stats()["max_running"] == 2  # the cap held across both shards
    assert r2.skipped_refreshes == 0  # LocalStore: nothing to reload

    a, b = stored(base_store, shards), stored(store, shards)
    assert set(a) == set(b) and len(a) == 7
    assert {u: v["records"] for u, v in a.items()} == {u: v["records"] for u, v in b.items()}
    ident = compare_runs(results(a), results(b))
    assert ident["ok"] and ident["paired"] == sum(len(v["calls"]) for v in a.values()), ident
    for u in a:  # strict, call by call
        assert [c["input_digest"] for c in a[u]["calls"]] == \
            [c["input_digest"] for c in b[u]["calls"]]


def test_more_shards_than_max_shards_are_refused(tmp_path):
    store, _ = world(tmp_path)
    r, _ = make_runner(tmp_path, store, max_shards=2, overlap=True, sleeps={"a": 0.3})
    errs: list = []
    gate = threading.Barrier(3)

    def go(sid):
        gate.wait()
        try:
            r.run_shard(JOB, sid, CFG)
        except ConcurrentShardRefused as e:
            errs.append(e)

    threads = [threading.Thread(target=go, args=(s,)) for s in ("segcap-a", "segcap-a",
                                                                "segcap-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(errs) == 1


def test_preempt_marks_every_overlapped_shard(tmp_path):
    store, shards = world(tmp_path)
    r, eng = make_runner(tmp_path, store, max_shards=2, overlap=True,
                         sleeps={"a": 0.4, "b": 0.4})
    excs: dict = {}

    def go(sid):
        try:
            r.run_shard(JOB, sid, CFG)
        except BaseException as e:  # noqa: BLE001 - the cancelled batch raises
            excs[sid] = e

    ts = [threading.Thread(target=go, args=(s.shard_id,)) for s in shards]
    for t in ts:
        t.start()
    time.sleep(0.2)
    assert r.preempt()
    for t in ts:
        t.join(30)
    jf = JobFiles(store, JOB, CFG)
    for sh in shards:
        assert read_attempt(jf, "segcap", sh.shard_id, "ta-1").preempted
        assert not store.exists(PATHS.receipt(CFG, "segcap", sh.shard_id))
    assert eng._slots.stats()["running"] == 0 and eng._slots.stats()["waiting"] == 0


class _VolumeLike(LocalStore):
    """A store with a mount and a refresh (``VolumeStore``'s surface) that counts reloads."""

    def __init__(self, root):
        super().__init__(root)
        self.mount = root
        self.reloads = 0

    def refresh(self):
        self.reloads += 1
        return True


def test_refresh_is_skipped_while_another_shard_reads_media_from_the_store_volume(tmp_path):
    store = _VolumeLike(tmp_path / "vol")
    store, shards = world(tmp_path, store=store)
    eng = segcap_fake_engine(Script(), concurrency=2, sleeps={"a": 0.3})
    eng.overlap = True
    eng.__post_init__()
    s = SegcapWorkerSettings(container_id="ta-1", gpu="H200", models={"tempseg": "rev"},
                             media_root=str(tmp_path / "vol" / "media"),
                             local_dir=str(tmp_path / "local"), max_shards=2)
    resolver = VideoResolver(s.media_root, tmp_path / "cuts", cutter=lambda src, dest: dest)
    r = SegcapShardRunner(store, eng, s, call_id=lambda: "fc", resolver=resolver)
    ta = threading.Thread(target=r.run_shard, args=(JOB, "segcap-a", CFG))
    ta.start()
    time.sleep(0.1)
    r.run_shard(JOB, "segcap-b", CFG)
    ta.join(30)
    assert store.reloads == 1 and r.skipped_refreshes == 1


def test_annotate_batch_slots_need_the_concurrent_composite():
    from tests.unit.test_reference_concurrency import GpuProbe, fake_nv

    nv = fake_nv(GpuProbe())
    with pytest.raises(RuntimeError, match="needs the resident session"):
        ref_stage.annotate_batch(nv, None, [], slots=EpisodeSlots(2))


# --- config ---------------------------------------------------------------------------------


def test_exact_scheduling_config_is_validated_and_not_hashed():
    from flora_deploy.config.schema import exact_scheduling, parse_config
    from tests.unit.test_config_hashing import base, h, set_, sh

    d = base()
    cfg = parse_config(d, allow_uncertified=True)
    assert exact_scheduling(cfg) == {"episode_order": "plan", "max_shards": 1}
    on = set_(set_(d, "pipeline.exact_scheduling", {"episode_order": "longest_first",
                                                    "shard_overlap": True}),
              "resources", {"segcap": {"max_inputs": 2}})
    cfg_on = parse_config(on, allow_uncertified=True)
    assert exact_scheduling(cfg_on) == {"episode_order": "longest_first", "max_shards": 2}
    # start order and shard overlap leave every model input unchanged: not in any hash
    assert h(on) == h(d) and sh(on) == sh(d)
    with pytest.raises(ValueError, match="max_inputs"):
        parse_config(set_(d, "pipeline.exact_scheduling", {"shard_overlap": True}),
                     allow_uncertified=True)
    with pytest.raises(ValueError):
        parse_config(set_(d, "pipeline.exact_scheduling", {"episode_order": "random"}),
                     allow_uncertified=True)
    rewards_only = set_(set_(d, "pipeline.stages", ["reward.terminal"]),
                        "pipeline.exact_scheduling", {"episode_order": "longest_first"})
    with pytest.raises(ValueError, match="segcap worker only"):
        parse_config(rewards_only, allow_uncertified=True)
