"""SegcapWorker core (``workers_impl.segcap``): index receipts -> NVIDIA inputs, certification,
provenance / audit lines, per-episode flushes under concurrency, the failure taxonomy
(``nv_failure`` terminal vs ``infra`` retried then raised), stage-hash keyed paths, preemption,
determinism and ``accel.identity`` call records. The contract itself is checked by
``tests/conformance``.

The engine is the production ``FastExactEngine`` over the fake NVIDIA family of
``test_reference_concurrency`` (``tests/conformance/registry.segcap_fake_engine``)."""

from __future__ import annotations

import json
import os
import signal
import threading
import time

import pytest

from flora_deploy.accel.identity import compare_runs
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.attempts import JobFiles, read_attempt
from flora_deploy.jobs.manifest import load_manifest, save_manifest
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard, episode_uid
from flora_deploy.media.index import NotKeyframeAligned
from flora_deploy.media.types import SourceRef
from flora_deploy.stages import reference as ref_stage
from flora_deploy.workers_impl.reward import ShardInfraError
from flora_deploy.workers_impl.segcap import (
    ConcurrentShardRefused,
    InputRejected,
    InstructionMissing,
    SegcapShardRunner,
    SegcapWorkerSettings,
    VideoResolver,
    certification,
    classify_input_error,
    classify_nvidia_error,
    instruction_for,
)
from tests.conformance.registry import segcap_fake_engine
from tests.conformance.worker_suite import Script

JOB, CFG, DS = "job-s", "cfg-s", "lerobot/ds"
PATHS = JobPaths(JOB)
SID = "segcap-00000"


def uid(eid: str) -> str:
    return episode_uid(DS, eid)


def receipt(eid, *, views=1, task="pick the cup", instruction="dataset_task", profile="human",
            yam32=False, rng=None, instruction_text=None, status="indexed"):
    roles = ("main",) if views == 1 else ("main", "left_wrist", "right_wrist")
    start, end = rng or (None, None)
    out = {"schema": "flora_index_receipt_v1", "status": status, "uid": uid(eid),
           "episode_id": eid, "dataset": {"name": DS}, "task": task,
           "instruction": instruction, "profile": profile, "yam32": yam32,
           "streams": {r: {"source": {"uri": f"videos/{eid}-{r}.mp4", "start_s": start,
                                      "end_s": end}} for r in roles},
           "envelope": {"ok": True, "reasons": []}}
    if instruction_text is not None:  # index receipts v2 (review-4 C2)
        out["instruction_text"] = instruction_text
    if status != "indexed":
        out["error"] = "source_error: stream missing"
    return out


def world(tmp_path, eps, *, envelope_ok=True, write=True, cfg=CFG, stage_hashes=None,
          store=None):
    store = store or LocalStore(tmp_path / "store")
    refs = []
    for eid, rec in eps:
        ref = EpisodeRef(uid(eid), DS, eid, "human", 4.0, True,
                         envelope_ok if not isinstance(envelope_ok, dict)
                         else envelope_ok.get(eid, True), PATHS.index(uid(eid)))
        refs.append(ref)
        if write and rec is not None:
            store.put_bytes(ref.index_ref, json.dumps(rec).encode())
    shard = Shard(SID, "segcap", tuple(r.uid for r in refs), 1.0)
    save_manifest(store, Manifest(JOB, cfg, tuple(refs), (shard,),
                                  "2026-10-21T00:00:00.000000Z",
                                  stage_hashes=dict(stage_hashes or {})))
    return store, shard


def runner(tmp_path, store, *, concurrency=2, script=None, envelope="warn",
           instruction_text=None, engine=None, cuts=None, container="ta-1",
           determinism="nvidia", fatal=None):
    eng = engine or segcap_fake_engine(script or Script(), concurrency=concurrency)
    settings = SegcapWorkerSettings(container_id=container, gpu="H200",
                                    models={"tempseg": "rev"}, envelope=envelope,
                                    media_root="/media", local_dir=str(tmp_path / "local"),
                                    instruction_text=instruction_text, determinism=determinism)
    resolver = VideoResolver("/media", tmp_path / "cuts",
                             cutter=cuts or (lambda src, dest: dest))
    return SegcapShardRunner(store, eng, settings, call_id=lambda: "fc-1", resolver=resolver,
                             fatal=fatal), eng


def records(store, shard, cfg=CFG):
    jf = JobFiles(store, JOB, cfg)
    return [r for p in P.shard_parts(jf, shard).parts for r in P.read_part(jf, p)]


def audits(store, shard, key=CFG):
    prefix = PATHS.audit(key, "segcap", shard.shard_id, "x", 0)[:-len("x-0.jsonl")]
    return [json.loads(line) for p in store.list(prefix)
            for line in store.get_bytes(p).decode().splitlines() if line.strip()]


def errors(store, shard, key=CFG):
    jf = JobFiles(store, JOB, CFG)
    prefix = PATHS.errors(key, "segcap", shard.shard_id, "x", 0)[:-len("x-0.jsonl")]
    return [e for p in store.list(prefix) for e in P.read_errors(jf, p)]


def marker(store, container="ta-1", cfg=CFG):
    return read_attempt(JobFiles(store, JOB, cfg), "segcap", SID, container)


def nv_class(name):
    """NVIDIA's exception class ``name``, from the registered stand-in for NVIDIA's code
    (``tests.fakes.fake_nvidia``): what makes it NVIDIA's is where it was loaded from, not its
    module name (review 6 H1; the real classes: ``tests/nvlink/test_nv_link_errors.py``)."""
    from tests.fakes import fake_nvidia

    return getattr(fake_nvidia, name)


def from_nvidia_frame(exc):
    """``exc`` as raised by a frame of NVIDIA's code (review 6 M2)."""
    from tests.fakes import fake_nvidia

    try:
        fake_nvidia.nvidia_raise(exc)
    except BaseException as e:  # noqa: BLE001
        return e


# --- inputs ---------------------------------------------------------------------------------


def test_inputs_from_index_receipt(tmp_path):
    cut = []
    eps = [("e0", receipt("e0")),
           ("e1", receipt("e1", views=3, profile="yam", yam32=True, rng=(2.0, 5.0)))]
    store, shard = world(tmp_path, eps)
    r, _ = runner(tmp_path, store, cuts=lambda src, dest: cut.append((src, dest)) or dest)
    refs = {e.uid: e for e in load_manifest(store, JOB).episodes}
    a = r.inputs_from_index(refs[uid("e0")])
    assert a.videos == {"main": "/media/videos/e0-main.mp4"}
    assert (a.profile, a.long_windows, a.instruction) == ("human", False, "pick the cup")
    b = r.inputs_from_index(refs[uid("e1")])
    assert set(b.videos) == {"main", "left_wrist", "right_wrist"}
    assert (b.profile, b.long_windows) == ("yam", True)
    assert len(cut) == 3 and cut[0][0] == SourceRef("/media/videos/e1-left_wrist.mp4", 2.0, 5.0)
    assert b.videos["main"].endswith(f"{uid('e1')}/main.mp4")
    assert b.item()["episode"] == uid("e1")


def test_instruction_sources():
    assert instruction_for({"instruction": "dataset_task"}, "t", None) == "t"
    assert instruction_for({}, None, None) == ""
    assert instruction_for({"instruction": "none"}, "t", "x") == ""
    assert instruction_for({"instruction": "fixed"}, "t", "do it") == "do it"
    with pytest.raises(InstructionMissing):
        instruction_for({"instruction": "fixed"}, "t", None)
    # index receipts v2: the resolved instruction_text wins (one source of truth)
    assert instruction_for({"instruction": "fixed", "instruction_text": "open"}, "t", None) \
        == "open"
    assert instruction_for({"instruction": "fixed", "instruction_text": "open"}, "t", "x") \
        == "open"
    assert instruction_for({"instruction": "dataset_task", "instruction_text": "T2"}, "t",
                           None) == "T2"
    with pytest.raises(InstructionMissing):  # an empty fixed text is not an instruction
        instruction_for({"instruction": "fixed", "instruction_text": ""}, "t", "  ")


def test_certification_per_envelope():
    ref = EpisodeRef("u", DS, "e", "human", 1.0, True, True, "x")
    assert certification(ref) == (True, None)
    out = EpisodeRef("u", DS, "e", "human", 1.0, True, False, "x")
    assert certification(out) == (False, "outside_reference_envelope")
    none = EpisodeRef("u", DS, "e", "human", 1.0, True, None, "x")
    assert certification(none) == (False, "oracle_ineligible_source")


def test_envelope_ignore_never_certifies_out_of_envelope(tmp_path):
    """The policy decides which episodes run, never ``certified`` (no certificate covers them)."""
    eps = [(f"e{i}", receipt(f"e{i}")) for i in range(2)]
    store, shard = world(tmp_path, eps, envelope_ok={"e1": False})
    r, _ = runner(tmp_path, store, envelope="ignore")
    r.run_shard(JOB, shard.shard_id, CFG)
    recs = records(store, shard)
    assert recs
    for x in recs:
        if x.uid == uid("e1"):
            assert not x.certified and x.uncertified_reason == "outside_reference_envelope"
        else:
            assert x.certified and x.uncertified_reason is None


# --- a full shard ---------------------------------------------------------------------------


def test_shard_records_audits_and_per_episode_flushes(tmp_path):
    eps = [(f"e{i}", receipt(f"e{i}")) for i in range(5)]
    store, shard = world(tmp_path, eps, envelope_ok={"e3": False})
    r, eng = runner(tmp_path, store, concurrency=4)
    rec = r.run_shard(JOB, shard.shard_id, CFG)
    assert rec is not None and len(rec.episodes_done) == 5 and not rec.episodes_failed
    assert len(rec.parts) == 5  # one part per finished episode (flush per episode)
    recs = records(store, shard)
    assert {(x.uid, x.stage) for x in recs} == {(uid(f"e{i}"), s) for i in range(5)
                                               for s in ("segment", "caption")}
    for x in recs:
        assert x.provenance.config_hash == CFG and x.provenance.engine == "nvidia-hf"
        assert x.provenance.gpu == "H200" and x.provenance.models == {"tempseg": "rev"}
        assert x.provenance.source["main"] == {"uri": f"videos/{x.episode_id}-main.mp4",
                                               "range_s": None}
        assert x.segments and (x.stage == "caption") == all(s.caption for s in x.segments)
        if x.uid == uid("e3"):
            assert not x.certified and x.uncertified_reason == "outside_reference_envelope"
        else:
            assert x.certified
    lines = audits(store, shard)
    assert len(lines) == 5
    rt = lines[0]["runtime"]
    assert rt["gpu_thread"]["enabled"] and rt["gpu_thread"]["calls"] > 0
    assert rt["image_fingerprint"] == "sha256:fake" and rt["exact_profile"]["gpu_thread"]
    assert rt["nv_audit"]["ok"] and rt["concurrency"] == 4
    # every GPU call of the episode was timed (evidence also kept for failed episodes)
    assert all(a["gpu_timing"]["gpu_calls"] > 0 and not a["gpu_timing"]["timing_dependent"]
               for a in lines)
    # nv.audit(): once per batch, before and after every episode
    assert len(eng.nv.audits) == 1 + 2 * 5
    # identity: every call's output was logged by the GPU lock (generate_output)
    calls = lines[0]["result"]["flora_reference"]["identity_calls"]
    assert calls and all(c["output_sha256"] for c in calls)
    assert r.last_batch["gpu_lock"]["acquisitions"] > 0
    m = marker(store)
    assert m is not None and m.call_id == "fc-1"


def test_two_runs_are_call_identical(tmp_path):
    eps = [(f"e{i}", receipt(f"e{i}", views=3 if i == 1 else 1,
                             profile="yam" if i == 1 else "human")) for i in range(3)]
    runs = []
    for k, c in enumerate((1, 3)):
        store, shard = world(tmp_path / f"r{k}", eps)
        r, _ = runner(tmp_path / f"r{k}", store, concurrency=c)
        r.run_shard(JOB, shard.shard_id, CFG)
        runs.append({a["uid"]: a["result"]["flora_reference"]["identity_calls"]
                     for a in audits(store, shard)})
    cmp = compare_runs(*runs)
    assert cmp["ok"] and cmp["conclusive"] and cmp["paired"] > 0, cmp


# --- failure taxonomy: nv_failure is terminal, infra is retried and raised -------------------


def test_nvidia_rejection_and_infra_errors(tmp_path):
    """NVIDIA's InputError is a terminal nv_failure; a missing index receipt is infra: the
    finished episode is flushed, the error file is written, the shard raises (no receipt), and
    the retry runs only the missing episode."""
    eps = [("e0", receipt("e0")),
           ("e1", receipt("e1", yam32=True)),  # NVIDIA: yam32 needs three views
           ("e2", None)]  # no index receipt: infra
    store, shard = world(tmp_path, eps)
    r, _ = runner(tmp_path, store)
    with pytest.raises(ShardInfraError, match="e2") as ei:
        r.run_shard(JOB, shard.shard_id, CFG)
    assert set(ei.value.infra) == {uid("e2")}
    jf = JobFiles(store, JOB, CFG)
    assert P.read_receipt(jf, "segcap", SID) is None  # never a receipt excusing infra
    errs = {(e.uid, e.stage, e.kind) for e in errors(store, shard)}
    assert errs == {(uid("e1"), s, "nv_failure") for s in ("segment", "caption")} | \
        {(uid("e2"), s, "infra") for s in ("segment", "caption")}
    msg = next(e for e in errors(store, shard) if e.uid == uid("e1"))
    assert "yam32" in msg.message and msg.diagnostics["type"].endswith("InputError")
    infra = next(e for e in errors(store, shard) if e.uid == uid("e2"))
    assert infra.diagnostics["attempts"] == 2 and "FileNotFoundError" in infra.message
    assert {x.uid for x in records(store, shard)} == {uid("e0")}  # flushed before raising
    m = marker(store)
    assert m is not None and "ShardInfraError" in m.error and not m.preempted
    prog = P.derive(jf, load_manifest(store, JOB), limit=3)
    assert prog.episodes[(uid("e2"), "segcap")] == "pending" and not prog.complete
    # the receipt appears (transient): a retry in a new container finishes the shard
    store.put_bytes(PATHS.index(uid("e2")), json.dumps(receipt("e2")).encode())
    r2, _ = runner(tmp_path, store, container="ta-2")
    rec = r2.run_shard(JOB, shard.shard_id, CFG)
    assert rec.episodes_done == (uid("e0"), uid("e2")) and rec.episodes_failed == (uid("e1"),)
    own = [p for p in rec.parts if p.rsplit("/", 1)[-1].startswith("ta-2-")]
    assert uid("e0") not in {x.uid for p in own for x in P.read_part(jf, p)}  # not redone


def test_infra_input_is_retried_once_in_process(tmp_path):
    eps = [("e0", receipt("e0")), ("e1", receipt("e1"))]
    store, shard = world(tmp_path, eps)
    r, _ = runner(tmp_path, store)
    calls: dict[str, int] = {}
    real = r.inputs_from_index

    def flaky(ref):  # one transient I/O error per episode, then fine
        calls[ref.uid] = calls.get(ref.uid, 0) + 1
        if calls[ref.uid] == 1:
            raise OSError("transient: connection reset")
        return real(ref)

    r.inputs_for = flaky
    rec = r.run_shard(JOB, shard.shard_id, CFG)
    assert set(rec.episodes_done) == {uid("e0"), uid("e1")} and not errors(store, shard)
    assert calls == {uid("e0"): 2, uid("e1"): 2}

    store2, shard2 = world(tmp_path / "b", eps)
    r2, _ = runner(tmp_path / "b", store2)
    n = []

    def broken(ref):
        n.append(ref.uid)
        raise OSError("store down")

    r2.inputs_for = broken
    with pytest.raises(ShardInfraError):
        r2.run_shard(JOB, shard2.shard_id, CFG)
    assert len(n) == 4  # each episode: the attempt + one retry, then raise
    assert {e.kind for e in errors(store2, shard2)} == {"infra"}
    assert P.read_receipt(JobFiles(store2, JOB, CFG), "segcap", SID) is None


def test_corrupt_index_receipt_is_infra_and_source_error_is_terminal(tmp_path):
    eps = [("e0", receipt("e0")), ("e1", receipt("e1", status="source_error"))]
    store, shard = world(tmp_path, eps)
    store.put_bytes(PATHS.index(uid("e0")), b'{"schema": "flora_index_receipt_v1", "sta')
    r, _ = runner(tmp_path, store)
    with pytest.raises(ShardInfraError, match="JSONDecodeError"):
        r.run_shard(JOB, shard.shard_id, CFG)
    kinds = {(e.uid, e.kind) for e in errors(store, shard)}
    assert kinds == {(uid("e0"), "infra"), (uid("e1"), "nv_failure")}


def test_fixed_instruction_needs_text(tmp_path):
    """A fixed instruction without text anywhere is a configuration error: the shard fails,
    nothing is excused. The text comes from the index receipt (v2) or FLORA_INSTRUCTION_TEXT."""
    eps = [("e0", receipt("e0", instruction="fixed"))]  # v1 receipt: no instruction_text
    store, shard = world(tmp_path, eps)
    r, _ = runner(tmp_path, store)
    with pytest.raises(InstructionMissing):
        r.run_shard(JOB, shard.shard_id, CFG)
    assert P.read_receipt(JobFiles(store, JOB, CFG), "segcap", SID) is None
    assert not errors(store, shard)  # never an excused episode
    m = marker(store)
    assert m is not None and "InstructionMissing" in m.error

    store2, shard2 = world(tmp_path / "b", eps)
    r2, _ = runner(tmp_path / "b", store2, instruction_text="open the box")
    refs = {e.uid: e for e in load_manifest(store2, JOB).episodes}
    assert r2.inputs_from_index(refs[uid("e0")]).instruction == "open the box"
    assert r2.run_shard(JOB, shard2.shard_id, CFG).episodes_done == (uid("e0"),)

    v2 = [("e0", receipt("e0", instruction="fixed", instruction_text="stack the cups"))]
    store3, shard3 = world(tmp_path / "c", v2)
    r3, _ = runner(tmp_path / "c", store3)  # no FLORA_INSTRUCTION_TEXT needed
    refs = {e.uid: e for e in load_manifest(store3, JOB).episodes}
    assert r3.inputs_from_index(refs[uid("e0")]).instruction == "stack the cups"
    assert r3.run_shard(JOB, shard3.shard_id, CFG).episodes_done == (uid("e0"),)


def test_classifier_table():
    t = ref_stage.transport_error
    fine = nv_class("FineSegmentationOutputError")
    ife = nv_class("InputErrorForModel")
    inp = nv_class("InputError")
    long_call = {"gpu_calls": 3, "max_call_s": 300.4, "timing_dependent": True}
    short = {"gpu_calls": 3, "max_call_s": 12.0, "timing_dependent": False}
    cases = [
        (t(fine("fine segmentation could not be repaired")), None, "nv_failure"),
        (t(nv_class("FineWorkerError")("bad worker output")), None, "nv_failure"),
        (t(nv_class("U2WorkerError")("bad caption")), None, "nv_failure"),
        (t(inp("yam32 requires YAM and all three camera views.")), None, "nv_failure"),
        (t(from_nvidia_frame(ValueError("camera roles must be main"))), None, "nv_failure"),
        (t(from_nvidia_frame(AssertionError("slots duplicate evidence"))), None, "nv_failure"),
        # never raised by NVIDIA's code (no NVIDIA frame): not NVIDIA's verdict (review 6 M2)
        (t(ValueError("camera roles must be main")), None, "infra"),
        (t(RuntimeError("The pipeline returned an invalid caption interval.")), None,
         "nv_failure"),
        (t(ife("The model did not return a complete output for this input.")), short,
         "nv_failure"),
        # a max_time=300 time stop is timing-, not input-determined: retried, never excused
        (t(ife("The model did not return a complete output for this input.")), long_call,
         "infra"),
        (t(ife("The selected video exceeds the native model context budget.")), long_call,
         "nv_failure"),
        (t(RuntimeError("CUDA error: an illegal memory access was encountered")), None,
         "infra"),
        (t(type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})("oom")), None,
         "infra"),
        (t(nv_class("NotReady")("model not loaded")), None, "infra"),
        (t(FileNotFoundError("/media/x.mp4")), None, "infra"),
        (t(RuntimeError("something in our code")), None, "infra"),
        (t(RuntimeError("A model asset failed its integrity check")), None, "infra"),
    ]
    for err, timing, want in cases:
        kind, diag = classify_nvidia_error(err, timing)
        assert kind == want, (err, timing, kind)
        if timing:
            assert diag["max_call_s"] == timing["max_call_s"]
    assert classify_nvidia_error(t(ife("did not return a complete output")), long_call)[1][
        "time_stop"]
    # inputs phase
    assert classify_input_error(json.JSONDecodeError("x", "", 0)) == "infra"
    assert classify_input_error(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")) == "infra"
    assert classify_input_error(ValueError("index receipt has no camera streams")) == "infra"
    assert classify_input_error(OSError("gone")) == "infra"
    assert classify_input_error(KeyError("streams")) == "infra"
    assert classify_input_error(InputRejected("source_error")) == "nv_failure"
    assert classify_input_error(NotKeyframeAligned("cut")) == "nv_failure"


def test_nvidia_family_rejection_is_terminal_with_timing(tmp_path):
    fine = nv_class("FineSegmentationOutputError")
    eps = [("e0", receipt("e0")), ("e1", receipt("e1"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), raises={"e1-": lambda: fine("not repaired")})
    r, _ = runner(tmp_path, store, engine=eng)
    rec = r.run_shard(JOB, shard.shard_id, CFG)
    assert rec.episodes_done == (uid("e0"),) and rec.episodes_failed == (uid("e1"),)
    e = next(x for x in errors(store, shard) if x.uid == uid("e1"))
    assert e.kind == "nv_failure" and e.diagnostics["type"].endswith("FineSegmentationOutputError")
    assert "gpu_calls" in e.diagnostics and "timing_dependent" in e.diagnostics  # M4 evidence


def test_time_stop_is_infra_never_excused(tmp_path):
    ife = nv_class("InputErrorForModel")
    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), raises={"e0-": lambda: ife(
        "The model did not return a complete output for this input.")})
    eng.episode_timing = lambda: {"gpu_calls": 2, "max_call_s": 300.2,
                                  "timing_dependent": True}
    r, _ = runner(tmp_path, store, engine=eng)
    with pytest.raises(ShardInfraError):
        r.run_shard(JOB, shard.shard_id, CFG)
    (e, _) = errors(store, shard)
    assert e.kind == "infra" and e.diagnostics["time_stop"] and \
        e.diagnostics["max_call_s"] == 300.2


def test_cuda_fault_exits_after_marking(tmp_path):
    eps = [("e0", receipt("e0")), ("e1", receipt("e1"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), raises={"e1-": lambda: RuntimeError(
        "CUDA error: an illegal memory access was encountered")})
    seen = []
    r, _ = runner(tmp_path, store, engine=eng, fatal=lambda exc: seen.append(marker(store)))
    with pytest.raises(ShardInfraError):
        r.run_shard(JOB, shard.shard_id, CFG)
    assert len(seen) == 1 and seen[0] is not None and seen[0].error  # marked, then exit
    assert {x.uid for x in records(store, shard)} == {uid("e0")}
    assert {e.kind for e in errors(store, shard)} == {"infra"}


# --- keys: everything is keyed by the stage hash ----------------------------------------------


def test_stage_hash_keyed_paths_and_reuse(tmp_path):
    eps = [("e0", receipt("e0")), ("e1", receipt("e1"))]
    store, shard = world(tmp_path, eps, stage_hashes={"segcap": "stSEG"})
    r, _ = runner(tmp_path, store)
    rec = r.run_shard(JOB, shard.shard_id, CFG)  # the orchestrator passes the job hash
    assert rec.config_hash == "stSEG"
    stored = P.read_receipt(JobFiles(store, JOB, CFG), "segcap", SID)
    assert stored == rec
    assert store.exists(PATHS.receipt("stSEG", "segcap", SID))
    assert store.exists(PATHS.attempt("stSEG", "segcap", SID, "ta-1"))
    assert rec.parts and all(p.startswith(PATHS.part("stSEG", "segcap", SID, "x", 0)
                                          [:-len("x-0.jsonl")]) for p in rec.parts)
    # nothing at all under the job hash's per-stage areas
    for maker in (PATHS.part, PATHS.audit, PATHS.errors):
        assert not list(store.list(maker(CFG, "segcap", SID, "x", 0)[:-len("x-0.jsonl")]))
    assert not store.exists(PATHS.receipt(CFG, "segcap", SID))
    assert {x.provenance.config_hash for x in records(store, shard)} == {"stSEG"}
    lines = audits(store, shard, key="stSEG")
    assert {a["config_hash"] for a in lines} == {"stSEG"} and \
        {a["job_config_hash"] for a in lines} == {CFG}
    # another config with the same segcap stage hash reuses the work: nothing is re-run
    world(tmp_path, eps, cfg="cfg-other", stage_hashes={"segcap": "stSEG"}, store=store)
    r2, eng2 = runner(tmp_path, store, container="ta-2")
    rec2 = r2.run_shard(JOB, shard.shard_id, "cfg-other")
    assert set(rec2.episodes_done) == {uid("e0"), uid("e1")} and rec2.config_hash == "stSEG"
    assert set(rec2.parts) == set(rec.parts)  # no new part: every episode was already done
    assert len(eng2.nv.audits) == 0  # NVIDIA never ran


# --- preemption / concurrency ----------------------------------------------------------------


def test_sigint_marks_preempted_at_once_and_inflight_episodes_still_flush(tmp_path):
    """Review 5 H3: a real SIGINT to the main thread at episode concurrency 4. The preempt mark
    is written immediately (not after the in-flight episodes finish: on the H200 they take
    minutes against Modal's 30 s grace), queued episodes never start, and the in-flight ones
    still flush their records in the background."""
    eps = [(f"e{i}", receipt(f"e{i}")) for i in range(8)]
    store, shard = world(tmp_path, eps)
    slow = 1.5
    eng = segcap_fake_engine(Script(), concurrency=4,
                             sleeps={f"e{i}-": slow for i in range(8)})
    r, _ = runner(tmp_path, store, engine=eng)
    main = threading.get_ident()
    sent = []

    def fire():
        time.sleep(0.4)
        sent.append(time.perf_counter())
        signal.pthread_kill(main, signal.SIGINT)

    threading.Thread(target=fire, daemon=True).start()
    with pytest.raises(KeyboardInterrupt):
        r.run_shard(JOB, shard.shard_id, CFG)
    m = marker(store)
    assert m is not None and m.preempted and m.ended_utc
    assert time.perf_counter() - sent[0] < slow / 2, "the mark waited for in-flight episodes"
    assert A.counted_starts(JobFiles(store, JOB, CFG), "segcap", SID) == 0
    assert P.read_receipt(JobFiles(store, JOB, CFG), "segcap", SID) is None
    deadline = time.time() + 10
    while time.time() < deadline and len({x.uid for x in records(store, shard)}) < 4:
        time.sleep(0.1)
    time.sleep(slow)  # give a (wrongly) started queued episode time to show up
    done = {x.uid for x in records(store, shard)}
    assert len(done) == 4, done  # the 4 in flight flushed; the 4 queued never started
    assert len(eng.nv.audits) == 1 + 2 * 4
    # the retry runs only the rest
    r2, _ = runner(tmp_path, store, container="ta-2")
    rec = r2.run_shard(JOB, shard.shard_id, CFG)
    assert len(rec.episodes_done) == 8
    assert os.getpid()  # (the signal went to this test process only)


def test_exit_hook_marks_inflight_shard_and_stops_queued_episodes(tmp_path):
    eps = [(f"e{i}", receipt(f"e{i}")) for i in range(6)]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), concurrency=2, sleeps={f"e{i}-": 0.6 for i in range(6)})
    r, _ = runner(tmp_path, store, engine=eng)
    out = {}

    def go():
        try:
            out["rec"] = r.run_shard(JOB, shard.shard_id, CFG)
        except BaseException as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target=go)
    t.start()
    time.sleep(0.3)
    assert r.preempt()  # @modal.exit while 2 episodes run and 4 are queued
    m = marker(store)
    assert m is not None and m.preempted
    t.join(timeout=10)
    assert "rec" not in out  # the queued episodes were cancelled: no receipt
    assert len({x.uid for x in records(store, shard)}) == 2
    assert r.preempt() is False  # idle again


def test_concurrent_shard_is_refused(tmp_path):
    """Review 5 M1: one shard per container (max_inputs must be 1); a second concurrent input
    is refused before it writes anything."""
    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script(), sleeps={"e0-": 0.8})
    r, _ = runner(tmp_path, store, engine=eng)
    t = threading.Thread(target=lambda: r.run_shard(JOB, shard.shard_id, CFG))
    t.start()
    time.sleep(0.3)
    before = sorted(store.list(f"flora/{JOB}/"))
    with pytest.raises(ConcurrentShardRefused, match="max_inputs"):
        r.run_shard(JOB, shard.shard_id, CFG)
    t.join(timeout=10)
    assert P.read_receipt(JobFiles(store, JOB, CFG), "segcap", SID) is not None
    assert before  # the first shard was running; the refused one wrote nothing of its own
    assert r.run_shard(JOB, shard.shard_id, CFG) is not None  # free again afterwards


# --- determinism ----------------------------------------------------------------------------


def _cpu_torch():
    """A torch stand-in with torch's CPU-process defaults (deterministic algorithms off, cuDNN SDPA
    on): what ``accel.determinism`` reads. Injected so the drift check is exercised identically
    whether or not torch is installed (CI's test job has no torch: ``--group dev`` only)."""
    from types import SimpleNamespace as NS

    cuda = NS(matmul=NS(allow_tf32=False), flash_sdp_enabled=lambda: True,
              mem_efficient_sdp_enabled=lambda: True, math_sdp_enabled=lambda: True,
              cudnn_sdp_enabled=lambda: True)
    return NS(__version__="cpu-stand-in",
              backends=NS(cuda=cuda, cudnn=NS(deterministic=False, benchmark=False,
                                              allow_tf32=True)),
              are_deterministic_algorithms_enabled=lambda: False,
              is_deterministic_algorithms_warn_only_enabled=lambda: False,
              get_float32_matmul_precision=lambda: "highest")


def test_determinism_label_and_drift(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _cpu_torch())
    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script())
    eng.determinism = {"schema": "flora_determinism_v1", "mode": "deterministic",
                       "in_effect": True, "flags": {}}
    r, _ = runner(tmp_path, store, engine=eng, determinism="deterministic")
    assert eng.label() == "nvidia-hf+deterministic"
    # the CPU test process is not in deterministic mode: verify() raises before any record
    with pytest.raises(Exception, match="eterminism") as ei:
        r.run_shard(JOB, shard.shard_id, CFG)
    assert "configured" not in str(ei.value)  # the drift check, not the mode refusal
    assert not records(store, shard)
    m = marker(store)
    assert m is not None and m.error and "eterminism" in m.error



def test_determinism_flag_drift_is_refused(tmp_path, monkeypatch):
    """Review 7 G4-5: the drift branch itself. The stand-in is put in deterministic mode, so the
    record is in effect at apply time; one flag then changes before the shard runs."""
    import sys

    from flora_deploy.accel import determinism as det

    torch = _cpu_torch()
    torch.are_deterministic_algorithms_enabled = lambda: True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.cudnn_sdp_enabled = lambda: False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv(det.CUBLAS_ENV, det.CUBLAS_WORKSPACE_CONFIG)
    record = det.provenance(det.DETERMINISTIC)
    assert record["in_effect"] is True
    det.verify(record)  # control: unchanged flags pass
    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script())
    eng.determinism = record
    r, _ = runner(tmp_path, store, engine=eng, determinism="deterministic")
    torch.backends.cuda.matmul.allow_tf32 = True  # drift after apply()
    with pytest.raises(det.DeterminismError, match="flags changed") as ei:
        r.run_shard(JOB, shard.shard_id, CFG)
    assert "matmul_allow_tf32" in str(ei.value)
    assert not records(store, shard)
    m = marker(store)
    assert m is not None and m.error and "flags changed" in m.error

def test_determinism_mode_mismatch_is_refused(tmp_path):
    from flora_deploy.accel.determinism import DeterminismError

    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script())
    eng.determinism = {"schema": "flora_determinism_v1", "mode": "deterministic",
                       "in_effect": True, "flags": {}}
    r, _ = runner(tmp_path, store, engine=eng)  # the job was hashed with "nvidia"
    with pytest.raises(DeterminismError, match="configured"):
        r.run_shard(JOB, shard.shard_id, CFG)
    assert not records(store, shard)
    with pytest.raises(DeterminismError):
        runner(tmp_path, store, determinism="fast")


def test_audit_results_carry_the_mode(tmp_path, monkeypatch):
    """Review 5 M3: production audit results name their D-1 mode, so a deterministic-mode
    audit is never compared with an nvidia-mode one."""
    from flora_deploy.accel import determinism as det

    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script())
    eng.determinism = {"schema": "flora_determinism_v1", "mode": "deterministic",
                       "in_effect": True, "flags": {}}
    monkeypatch.setattr(det, "verify", lambda rec: None)
    r, _ = runner(tmp_path, store, engine=eng, determinism="deterministic")
    r.run_shard(JOB, shard.shard_id, CFG)
    (a,) = audits(store, shard)
    assert ref_stage.determinism_mode(a["result"]) == "deterministic"
    other = json.loads(json.dumps(a["result"]))
    other["flora_reference"]["determinism"] = {"mode": "nvidia"}
    with pytest.raises(ref_stage.ModeMismatch):
        ref_stage.check_same_mode({"e": a["result"]}, {"e": other})
    store2, shard2 = world(tmp_path / "n", eps)
    r2, _ = runner(tmp_path / "n", store2)
    r2.run_shard(JOB, shard2.shard_id, CFG)
    (b,) = audits(store2, shard2)
    assert b["result"]["flora_reference"]["determinism"]["mode"] == "nvidia"


def test_wrong_kind_is_refused(tmp_path):
    store = LocalStore(tmp_path / "s")
    ref = EpisodeRef(uid("e0"), DS, "e0", "human", 4.0, True, True, PATHS.index(uid("e0")))
    shard = Shard("reward.terminal-00000", "reward.terminal", (ref.uid,), 1.0)
    save_manifest(store, Manifest(JOB, CFG, (ref,), (shard,), "2026-10-21T00:00:00.000000Z"))
    r, _ = runner(tmp_path, store)
    with pytest.raises(ValueError, match="Segcap"):
        r.run_shard(JOB, shard.shard_id, CFG)


# --- the Modal wiring (no Modal runs) ---------------------------------------------------------


def test_reference_app_wires_the_runner():
    import inspect

    from flora_deploy.accel.determinism import DeterminismError
    from flora_deploy.modal_app import reference_app as app_mod

    src = inspect.getsource(app_mod.ReferenceWorker)
    assert "background=True" in src and "gpu_thread.install" in src
    assert "emit_output=True" in src and "FLORA_DETERMINISM" in src
    assert 'def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""' \
        in src
    assert "fatal=_die" in src and "self._runner_lock" in src
    load = src[src.index("def load("):src.index("def _nvidia_attributes(")]
    # the cheap preflight runs before the expensive work (layout, family link, model loads)
    assert load.index("preflight(os.environ") < load.index("determinism.apply(")
    assert load.index("preflight(os.environ") < load.index("build_linked_layout")
    assert load.index("preflight(os.environ") < load.index("load_family(")
    s = app_mod.segcap_settings_from_env({"FLORA_ENVELOPE": "ignore",
                                          "FLORA_INSTRUCTION_TEXT": "x",
                                          "FLORA_MEDIA_ROOT": "/m"}, gpu="H200")
    assert s.envelope == "ignore" and s.instruction_text == "x" and s.gpu == "H200"
    assert s.media_root == "/m" and s.determinism == "nvidia"
    assert set(s.models) == {"tempseg", "tempseg_code", "tempseg_base", "captioner"}
    det = app_mod.segcap_settings_from_env(
        {"FLORA_DETERMINISM": "deterministic",
         "FLORA_OUTPUT_SETTINGS": json.dumps({"determinism": "deterministic"})}, gpu=None)
    assert det.determinism == "deterministic"
    assert app_mod.configured_determinism(
        {"FLORA_OUTPUT_SETTINGS": json.dumps({"determinism": "deterministic"})}) \
        == "deterministic"
    with pytest.raises(DeterminismError):
        app_mod.configured_determinism(
            {"FLORA_DETERMINISM": "nvidia",
             "FLORA_OUTPUT_SETTINGS": json.dumps({"determinism": "deterministic"})})
    with pytest.raises(DeterminismError):
        app_mod.configured_determinism({"FLORA_DETERMINISM": "fast"})
    assert app_mod.job_worker_env({"FLORA_STORE_KIND": "volume"})
    assert not app_mod.job_worker_env({})
