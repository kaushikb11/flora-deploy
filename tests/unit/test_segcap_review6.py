"""Review 6 (SegcapWorker after Fix-S): one reproduction per finding.

H1/M2  NVIDIA's rejections are found by origin (``nv.errors``), not by a ``flora_nv_`` prefix:
       terminal in the runner and end to end (a local job no longer fails on one episode).
H3     Modal input cancellation (``InputCancellation``, a ``BaseException``): the batch is
       cancelled, the shard marked, and its queued episodes are never revived by the next shard.
M1     A deterministic CUDA fault / per-episode OOM counts toward D15 and ends isolated
       (poison pill), instead of crash-looping or failing the job.
L1     The time stop keys on the failing (last) call and becomes terminal when it repeats.
L2     A decode-type rejection is retried once before it is excused.
L3     ``audit_ref`` points at the episode's audit line, identically from every container.
"""

from __future__ import annotations

import json
import shutil
import signal
import threading
import time
from pathlib import Path

import pytest

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.attempts import JobFiles
from flora_deploy.jobs.manifest import save_manifest
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard, episode_uid
from flora_deploy.workers_impl.reward import ShardInfraError
from tests.conformance.registry import segcap_fake_engine
from tests.conformance.worker_suite import Script
from tests.fakes import fake_nvidia as F
from tests.unit.test_segcap_worker import (
    CFG,
    JOB,
    SID,
    errors,
    marker,
    receipt,
    records,
    runner,
    uid,
    world,
)

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"


def nvidia(exc):
    """An exception factory that raises ``exc`` from a frame of 'NVIDIA's code'."""
    def make():
        try:
            F.nvidia_raise(exc)
        except BaseException as e:  # noqa: BLE001
            return e
    return make


# --- H1 / M2: classification by origin, in the runner -----------------------------------------


@pytest.mark.parametrize("make", [
    lambda: F.FineSegmentationOutputError("could not be repaired by bounded fine retry"),
    lambda: F.SubtaskAnnotationAdapterError("response identity mismatch"),
    lambda: F.FineWorkerError("nearest decoded frame is farther than half a slot"),
    lambda: F.U2WorkerError("U2 frame budget 999 exceeds 512"),
    nvidia(AssertionError("Real temporal slots must not duplicate evidence")),
    nvidia(KeyError("u2_context")),
    nvidia(TypeError("confidence must be a number in [0, 1]")),
    nvidia(StopIteration()),
], ids=["FineSegmentationOutputError", "SubtaskAnnotationAdapterError", "FineWorkerError",
        "U2WorkerError", "AssertionError", "KeyError", "TypeError", "StopIteration"])
def test_nvidia_rejection_by_origin_is_terminal(tmp_path, make):
    eps = [("e0", receipt("e0")), ("e1", receipt("e1"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), raises={"e1-": make})
    r, _ = runner(tmp_path, store, engine=eng)
    rec = r.run_shard(JOB, shard.shard_id, CFG)
    assert rec.episodes_done == (uid("e0"),) and rec.episodes_failed == (uid("e1"),)
    e = next(x for x in errors(store, shard) if x.uid == uid("e1"))
    assert e.kind == "nv_failure" and e.diagnostics["class"] == "nv_failure"


def test_our_builtin_errors_are_not_excused(tmp_path):
    """A ValueError raised by our code (not an NVIDIA frame) is ours: retried, never excused."""
    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), raises={"e0-": lambda: ValueError("our bug")})
    r, _ = runner(tmp_path, store, engine=eng)
    with pytest.raises(ShardInfraError):
        r.run_shard(JOB, shard.shard_id, CFG)
    (e, _) = errors(store, shard)
    assert e.kind == "infra" and e.diagnostics["class"] == "python"


def test_local_job_survives_one_nvidia_rejection(tmp_path, monkeypatch):
    """Review 6 p3: one episode NVIDIA deterministically rejects with its own class. The job
    completes with that episode excused after ONE run (it used to run 12x and fail the job)."""
    from flora_deploy.local import engines as LE
    from flora_deploy.local import run_local, sample_dataset

    runs = {"n": 0}
    orig = LE.FakeSegcapEngine.result_for

    def result_for(self, item):
        if item["episode"].endswith("--ep003"):
            runs["n"] += 1
            F.nvidia_raise(F.FineSegmentationOutputError("retry_output_rejected"))
        return orig(self, item)

    monkeypatch.setattr(LE.FakeSegcapEngine, "result_for", result_for)
    shutil.copy(TEMPLATE, tmp_path / "flora.yaml")
    sample_dataset(tmp_path / "videos")
    res = run_local(tmp_path / "flora.yaml", store=tmp_path / "store")
    assert res.state in ("done", "completed_with_errors"), res.state
    assert runs["n"] == 1
    assert not [e for e in res.events if e["event"] == "shard_failed"]


# --- H3: Modal input cancellation --------------------------------------------------------------


from modal.exception import InputCancellation  # noqa: E402 - what Modal raises on cancel


def _second_job(store):
    job2, ds2 = "job-s2", "lerobot/ds2"
    paths = JobPaths(job2)
    refs = []
    for i in range(2):
        u = episode_uid(ds2, f"f{i}")
        ref = EpisodeRef(u, ds2, f"f{i}", "human", 4.0, True, True, paths.index(u))
        rec = receipt(f"f{i}")
        rec.update(uid=u, dataset={"name": ds2})
        store.put_bytes(ref.index_ref, json.dumps(rec).encode())
        refs.append(ref)
    shard = Shard("segcap-00000", "segcap", tuple(r.uid for r in refs), 1.0)
    save_manifest(store, Manifest(job2, CFG, tuple(refs), (shard,), "2026-10-21T00:00:00.000000Z"))
    return job2, shard


def test_input_cancellation_cancels_the_batch_and_never_revives_it(tmp_path):
    """Review 6 H3 (p1 B): a real SIGUSR1 -> InputCancellation in the main thread, as Modal's
    container entrypoint raises it for ``FunctionCall.cancel()`` (quarantine, job cancel)."""
    n, ep_s = 8, 1.0
    store, shard = world(tmp_path, [(f"e{i}", receipt(f"e{i}")) for i in range(n)])
    eng = segcap_fake_engine(Script(), concurrency=4,
                             sleeps={f"e{i}-": ep_s for i in range(n)})
    started = []
    orig_start = eng._on_start

    def on_start(ep):
        orig_start(ep)
        started.append((ep["episode"], time.monotonic()))

    eng._on_start = on_start
    job2, shard2 = _second_job(store)
    r, _ = runner(tmp_path, store, engine=eng)

    def handler(signum, frame):
        raise InputCancellation("Input was cancelled by user")

    old = signal.signal(signal.SIGUSR1, handler)
    main = threading.get_ident()
    sent = {}

    def fire():
        time.sleep(0.4)
        sent["t"] = time.monotonic()
        signal.pthread_kill(main, signal.SIGUSR1)

    threading.Thread(target=fire, daemon=True).start()
    try:
        with pytest.raises(InputCancellation):
            r.run_shard(JOB, shard.shard_id, CFG)
    finally:
        signal.signal(signal.SIGUSR1, old)
    m = marker(store)
    assert m is not None and m.preempted and m.ended_utc  # marked at once, not D15-counted
    assert time.monotonic() - sent["t"] < ep_s / 2
    assert A.counted_starts(JobFiles(store, JOB, CFG), "segcap", SID) == 0
    # the next input lands in this container: it runs alone, after the cancelled batch drained
    rec2 = r.run_shard(job2, shard2.shard_id, CFG)
    assert rec2 is not None and len(rec2.episodes_done) == 2
    time.sleep(ep_s)
    old_after_cancel = [u for u, t in started if t > sent["t"] and "ds--" in u]
    assert old_after_cancel == [], "a cancelled shard's queued episodes were revived"
    assert len([u for u, _ in started if "ds--" in u]) == 4  # only the 4 in flight ever ran
    assert P.read_receipt(JobFiles(store, JOB, CFG), "segcap", SID) is None
    assert {x.uid for x in records(store, shard)} == set()  # a cancelled shard writes nothing


# --- M1: CUDA faults and OOM are D15 crashes ---------------------------------------------------


def _local_job(tmp_path, monkeypatch, exc_factory, max_ticks=400):
    from flora_deploy.local import LocalRunner, sample_dataset
    from flora_deploy.local import engines as LE

    runs = {"n": 0}
    orig = LE.FakeSegcapEngine.result_for

    def result_for(self, item):
        if item["episode"].endswith("--ep003"):
            runs["n"] += 1
            raise exc_factory()
        return orig(self, item)

    monkeypatch.setattr(LE.FakeSegcapEngine, "result_for", result_for)
    shutil.copy(TEMPLATE, tmp_path / "flora.yaml")
    sample_dataset(tmp_path / "videos")
    lr = LocalRunner(tmp_path / "flora.yaml", store=tmp_path / "store")
    try:
        lr.index()
        lr.plan()
        outcome = lr.orchestrate(max_ticks=max_ticks)
        events = {}
        for e in lr.events:
            events[e["event"]] = events.get(e["event"], 0) + 1
        return outcome, runs["n"], events, lr.spawner.stats()
    finally:
        lr.close()


@pytest.mark.parametrize("make", [
    lambda: RuntimeError("CUDA error: an illegal memory access was encountered"),
    lambda: type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})(
        "CUDA out of memory. Tried to allocate 20.00 GiB"),
], ids=["cuda_fault", "torch_oom"])
def test_deterministic_gpu_fault_ends_isolated(tmp_path, monkeypatch, make):
    outcome, runs, events, stats = _local_job(tmp_path, monkeypatch, make)
    assert outcome != "max_ticks", (outcome, runs, stats)
    assert events.get("poisoned") == 1 and "shard_failed" not in events, events
    assert runs <= 2 * 3 + 3  # K=3 per shard level (parent, then the single-episode child)


def test_crash_marker_counts_toward_quarantine(tmp_path):
    eps = [("e0", receipt("e0")), ("e1", receipt("e1"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), raises={"e1-": lambda: RuntimeError(
        "CUDA error: an illegal memory access was encountered")})
    seen = []
    r, _ = runner(tmp_path, store, engine=eng, fatal=lambda exc: seen.append(marker(store)))
    with pytest.raises(ShardInfraError):
        r.run_shard(JOB, shard.shard_id, CFG)
    assert len(seen) == 1 and seen[0].error and seen[0].crashed  # marked, then exit
    assert A.counted_starts(JobFiles(store, JOB, CFG), "segcap", SID) == 1


# --- L1 / L2 ----------------------------------------------------------------------------------


def test_time_stop_uses_the_failing_call_and_is_terminal_when_repeated(tmp_path):
    ife = lambda: F.InputErrorForModel(  # noqa: E731
        "The model did not return a complete output for this input.")
    store, shard = world(tmp_path, [("e0", receipt("e0"))])
    # an earlier complete call was long, the failing (last) one short: NVIDIA's verdict
    eng = segcap_fake_engine(Script(), raises={"e0-": ife})
    eng.episode_timing = lambda: {"gpu_calls": 2, "max_call_s": 300.2, "last_call_s": 20.0,
                                  "timing_dependent": True}
    r, _ = runner(tmp_path, store, engine=eng)
    assert r.run_shard(JOB, shard.shard_id, CFG).episodes_failed == (uid("e0"),)

    store2, shard2 = world(tmp_path / "b", [("e0", receipt("e0"))])
    eng2 = segcap_fake_engine(Script(), raises={"e0-": ife})
    eng2.episode_timing = lambda: {"gpu_calls": 2, "max_call_s": 300.2, "last_call_s": 300.2,
                                   "timing_dependent": True}
    r2, _ = runner(tmp_path / "b", store2, engine=eng2)
    with pytest.raises(ShardInfraError):  # first time stop: retried
        r2.run_shard(JOB, shard2.shard_id, CFG)
    r3, _ = runner(tmp_path / "b", store2, engine=eng2, container="ta-2")
    rec = r3.run_shard(JOB, shard2.shard_id, CFG)  # the same time stop again: terminal
    assert rec.episodes_failed == (uid("e0"),)
    last = [e for e in errors(store2, shard2) if e.kind == "nv_failure"]
    assert last and last[0].diagnostics["time_stop"] and last[0].diagnostics["repeats"] == 2


def test_decode_rejection_is_retried_once_before_it_is_excused(tmp_path):
    store, shard = world(tmp_path, [("e0", receipt("e0"))])
    eng = segcap_fake_engine(Script(), raises={"e0-": lambda: F.InputError(
        "A requested source frame could not be decoded.")})
    r, _ = runner(tmp_path, store, engine=eng)
    with pytest.raises(ShardInfraError):  # maybe a short read on the Volume: retried
        r.run_shard(JOB, shard.shard_id, CFG)
    r2, _ = runner(tmp_path, store, engine=eng, container="ta-2")
    assert r2.run_shard(JOB, shard.shard_id, CFG).episodes_failed == (uid("e0"),)


# --- L3: audit_ref -----------------------------------------------------------------------------


def test_audit_ref_points_at_the_audit_line(tmp_path):
    store, shard = world(tmp_path, [("e0", receipt("e0"))])
    r, _ = runner(tmp_path, store)
    r.run_shard(JOB, shard.shard_id, CFG)
    recs = records(store, shard)
    refs = {x.audit_ref for x in recs}
    assert len(refs) == 1 and None not in refs
    (ref,) = refs
    prefix, _, frag = ref.partition("#")
    assert frag == f"uid={uid('e0')}" and "ta-1" not in ref  # container-independent
    lines = [json.loads(line) for p in store.list(prefix)
             for line in store.get_bytes(p).decode().splitlines() if line.strip()]
    assert [a["uid"] for a in lines] == [uid("e0")]
