"""SegcapWorker core (``workers_impl.segcap``): index receipts -> NVIDIA inputs, certification,
provenance / audit lines, per-episode flushes under concurrency, error kinds, determinism and
``accel.identity`` call records. The contract itself is checked by ``tests/conformance``.

The engine is the production ``FastExactEngine`` over the fake NVIDIA family of
``test_reference_concurrency`` (``tests/conformance/registry.segcap_fake_engine``)."""

from __future__ import annotations

import json

import pytest

from flora_deploy.accel.identity import compare_runs
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.attempts import JobFiles, read_attempt
from flora_deploy.jobs.manifest import load_manifest, save_manifest
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard, episode_uid
from flora_deploy.media.types import SourceRef
from flora_deploy.workers_impl.segcap import (
    SegcapShardRunner,
    SegcapWorkerSettings,
    VideoResolver,
    certification,
    instruction_for,
)
from tests.conformance.registry import segcap_fake_engine
from tests.conformance.worker_suite import Script

JOB, CFG, DS = "job-s", "cfg-s", "lerobot/ds"
PATHS = JobPaths(JOB)


def uid(eid: str) -> str:
    return episode_uid(DS, eid)


def receipt(eid, *, views=1, task="pick the cup", instruction="dataset_task", profile="human",
            yam32=False, rng=None):
    roles = ("main",) if views == 1 else ("main", "left_wrist", "right_wrist")
    start, end = rng or (None, None)
    return {"schema": "flora_index_receipt_v1", "status": "indexed", "uid": uid(eid),
            "episode_id": eid, "dataset": {"name": DS}, "task": task,
            "instruction": instruction, "profile": profile, "yam32": yam32,
            "streams": {r: {"source": {"uri": f"videos/{eid}-{r}.mp4", "start_s": start,
                                       "end_s": end}} for r in roles},
            "envelope": {"ok": True, "reasons": []}}


def world(tmp_path, eps, *, envelope_ok=True, write=True):
    store = LocalStore(tmp_path / "store")
    refs = []
    for eid, rec in eps:
        ref = EpisodeRef(uid(eid), DS, eid, "human", 4.0, True,
                         envelope_ok if not isinstance(envelope_ok, dict)
                         else envelope_ok.get(eid, True), PATHS.index(uid(eid)))
        refs.append(ref)
        if write and rec is not None:
            store.put_bytes(ref.index_ref, json.dumps(rec).encode())
    shard = Shard("segcap-00000", "segcap", tuple(r.uid for r in refs), 1.0)
    save_manifest(store, Manifest(JOB, CFG, tuple(refs), (shard,),
                                  "2026-10-21T00:00:00.000000Z"))
    return store, shard


def runner(tmp_path, store, *, concurrency=2, script=None, envelope="warn",
           instruction_text=None, engine=None, cuts=None):
    eng = engine or segcap_fake_engine(script or Script(), concurrency=concurrency)
    settings = SegcapWorkerSettings(container_id="ta-1", gpu="H200",
                                    models={"tempseg": "rev"}, envelope=envelope,
                                    media_root="/media", local_dir=str(tmp_path / "local"),
                                    instruction_text=instruction_text)
    resolver = VideoResolver("/media", tmp_path / "cuts",
                             cutter=cuts or (lambda src, dest: dest))
    return SegcapShardRunner(store, eng, settings, call_id=lambda: "fc-1", resolver=resolver), eng


def records(store, shard):
    jf = JobFiles(store, JOB, CFG)
    return [r for p in P.shard_parts(jf, shard).parts for r in P.read_part(jf, p)]


def audits(store, shard):
    prefix = PATHS.audit(CFG, "segcap", shard.shard_id, "x", 0)[:-len("x-0.jsonl")]
    return [json.loads(line) for p in store.list(prefix)
            for line in store.get_bytes(p).decode().splitlines() if line.strip()]


def errors(store, shard):
    jf = JobFiles(store, JOB, CFG)
    prefix = PATHS.errors(CFG, "segcap", shard.shard_id, "x", 0)[:-len("x-0.jsonl")]
    return [e for p in store.list(prefix) for e in P.read_errors(jf, p)]


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
    with pytest.raises(RuntimeError):
        instruction_for({"instruction": "fixed"}, "t", None)


def test_certification_per_envelope():
    ref = EpisodeRef("u", DS, "e", "human", 1.0, True, True, "x")
    assert certification(ref) == (True, None)
    out = EpisodeRef("u", DS, "e", "human", 1.0, True, False, "x")
    assert certification(out) == (False, "outside_reference_envelope")
    assert certification(out, "ignore") == (True, None)
    none = EpisodeRef("u", DS, "e", "human", 1.0, True, None, "x")
    assert certification(none) == (False, "oracle_ineligible_source")


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
    # nv.audit(): once per batch, before and after every episode
    assert len(eng.nv.audits) == 1 + 2 * 5
    # identity: every call's output was logged by the GPU lock (generate_output)
    calls = lines[0]["result"]["flora_reference"]["identity_calls"]
    assert calls and all(c["output_sha256"] for c in calls)
    assert r.last_batch["gpu_lock"]["acquisitions"] > 0
    m = read_attempt(JobFiles(store, JOB, CFG), "segcap", shard.shard_id, "ta-1")
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


def test_nvidia_rejection_and_infra_errors(tmp_path):
    eps = [("e0", receipt("e0")),
           ("e1", receipt("e1", yam32=True)),  # NVIDIA: yam32 needs three views
           ("e2", None)]  # no index receipt: infra
    store, shard = world(tmp_path, eps)
    r, _ = runner(tmp_path, store)
    rec = r.run_shard(JOB, shard.shard_id, CFG)
    assert rec.episodes_done == (uid("e0"),)
    assert set(rec.episodes_failed) == {uid("e1"), uid("e2")}
    errs = {(e.uid, e.stage, e.kind) for e in errors(store, shard)}
    assert {(uid("e1"), s, "nv_failure") for s in ("segment", "caption")} <= errs
    assert {(uid("e2"), s, "infra") for s in ("segment", "caption")} <= errs
    msg = next(e for e in errors(store, shard) if e.uid == uid("e1"))
    assert "yam32" in msg.message and msg.diagnostics["type"].endswith("InputError")


def test_fixed_instruction_needs_text(tmp_path):
    eps = [("e0", receipt("e0", instruction="fixed"))]
    store, shard = world(tmp_path, eps)
    r, _ = runner(tmp_path, store)
    assert r.run_shard(JOB, shard.shard_id, CFG).episodes_failed == (uid("e0"),)
    store2, shard2 = world(tmp_path / "b", eps)
    r2, _ = runner(tmp_path / "b", store2, instruction_text="open the box")
    assert r2.run_shard(JOB, shard2.shard_id, CFG).episodes_done == (uid("e0"),)


def test_determinism_label_and_drift(tmp_path):
    eps = [("e0", receipt("e0"))]
    store, shard = world(tmp_path, eps)
    eng = segcap_fake_engine(Script())
    eng.determinism = {"schema": "flora_determinism_v1", "mode": "deterministic",
                       "in_effect": True, "flags": {}}
    r, _ = runner(tmp_path, store, engine=eng)
    assert eng.label() == "nvidia-hf+deterministic"
    # the CPU test process is not in deterministic mode: verify() raises before any record
    with pytest.raises(Exception, match="determinism"):
        r.run_shard(JOB, shard.shard_id, CFG)
    assert not records(store, shard)
    m = read_attempt(JobFiles(store, JOB, CFG), "segcap", shard.shard_id, "ta-1")
    assert m is not None and m.error and "eterminism" in m.error


def test_wrong_kind_is_refused(tmp_path):
    store = LocalStore(tmp_path / "s")
    ref = EpisodeRef(uid("e0"), DS, "e0", "human", 4.0, True, True, PATHS.index(uid("e0")))
    shard = Shard("reward.terminal-00000", "reward.terminal", (ref.uid,), 1.0)
    save_manifest(store, Manifest(JOB, CFG, (ref,), (shard,), "2026-10-21T00:00:00.000000Z"))
    r, _ = runner(tmp_path, store)
    with pytest.raises(ValueError, match="Segcap"):
        r.run_shard(JOB, shard.shard_id, CFG)


def test_reference_app_wires_the_runner():
    import inspect

    from flora_deploy.modal_app import reference_app as app_mod

    src = inspect.getsource(app_mod.ReferenceWorker)
    assert "background=True" in src and "gpu_thread.install" in src
    assert "emit_output=True" in src and "FLORA_DETERMINISM" in src
    assert 'def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""' \
        in src
    s = app_mod.segcap_settings_from_env({"FLORA_ENVELOPE": "ignore",
                                          "FLORA_INSTRUCTION_TEXT": "x"}, gpu="H200")
    assert s.envelope == "ignore" and s.instruction_text == "x" and s.gpu == "H200"
    assert set(s.models) == {"tempseg", "tempseg_code", "tempseg_base", "captioner"}
