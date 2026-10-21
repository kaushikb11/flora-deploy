"""Exact-preset concurrency: GPU-lock semantics, per-thread processors, identical outputs.

A fake NVIDIA family mirrors the structure that matters: ``pipeline`` functions look up
``pipeline.generate`` at call time (NVIDIA imports it into ``pipeline`` at module level), the
segmenter calls its predictor through ``asyncio.to_thread``, input building goes through a
stateful processor, and ``generate`` prints ``FLORA_RUNTIME`` events.
"""

import asyncio
import contextlib
import hashlib
import json
import random
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from flora_deploy.nv.session import CachedSession
from flora_deploy.observe import Patcher, TimedLock
from flora_deploy.stages import reference as ref


class InputError(ValueError):
    pass


class FakeProc:
    """Stateful like a fast tokenizer: per-call state lives on the object (racy if shared)."""

    def __init__(self, kind):
        self.kind = kind
        self._state = None

    def __call__(self, text):
        self._state = text
        time.sleep(random.uniform(0, 0.002))
        return {"input_ids": hashlib.sha256(f"{self.kind}:{self._state}".encode()).hexdigest()}


def runtime_event(event, **values):
    """NVIDIA's ``models.runtime_event`` format (``FLORA_RUNTIME {json}`` with ``utc``)."""
    from datetime import UTC, datetime

    print("FLORA_RUNTIME " + json.dumps(dict(event=event, utc=datetime.now(UTC).isoformat(),
                                             **values), sort_keys=True), flush=True)


class GpuProbe:
    """Fake ``models.generate``: deterministic output; records overlap and procs seen; logs
    ``generation_start`` (with an input digest) and ``generation_end`` like NVIDIA's."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.procs: dict[int, set[int]] = {}
        self._lock = threading.Lock()

    def __call__(self, model, proc, inputs, budget):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
            self.procs.setdefault(threading.get_ident(), set()).add(id(proc))
        try:
            digest = hashlib.sha256(str(inputs["input_ids"]).encode()).hexdigest()
            runtime_event("generation_start", input_tokens=10, output_limit=budget,
                          input_tensor_digest=digest)
            time.sleep(random.uniform(0.001, 0.004))
            text = hashlib.sha256(f"{model}:{inputs['input_ids']}:{budget}".encode()).hexdigest()
            runtime_event("generation_end", input_tokens=10, generated_tokens=len(text) % 7 + 1,
                          output_limit=budget, complete=True, elapsed_s=0.01)
            return text[:16]
        finally:
            with self._lock:
                self.active -= 1


def _video(path, n=40):
    pts = tuple(Fraction(i * 250000) for i in range(n))
    return SimpleNamespace(path=Path(path), sha256=hashlib.sha256(str(path).encode()).hexdigest(),
                           rate=Fraction(4), pts_us=pts, duration_us=n * 250000,
                           duration_s=n / 4, size=(64, 48), cfr=True)


def fake_nv(probe: GpuProbe, calls: list | None = None):
    """A fake SEGCAP family with NVIDIA's call structure: ``api.DataEnginePipeline().annotate``
    takes ``api._REQUEST_LOCK`` (looked up at call time), runs ``_inputs`` then
    ``pipeline.annotate``, which builds ``pipeline.Session()`` (looked up at call time) and
    closes it in ``finally``; ``pipeline`` functions call ``pipeline.generate``."""
    calls = [] if calls is None else calls
    profiles = {"human": "Human", "yam": "YAM"}
    pools = {"Human": "human", "YAM": "yam"}
    pipeline = SimpleNamespace(generate=probe)
    api = SimpleNamespace(_REQUEST_LOCK=threading.RLock())

    class NvSession:  # NVIDIA's own Session: fresh processors per video
        def __init__(self):
            calls.append(("fresh_session",))
            self.audit = {}

        def load(self, kind):
            return f"model-{kind}", FakeProc(kind)

        def close(self):
            calls.append(("fresh_close",))

    pipeline.Session = NvSession

    def _inputs(video, profile, left, right, yam32, instruction):
        if bool(left) != bool(right):
            raise InputError("Provide both left and right wrist videos together.")
        if yam32 and not left:
            raise InputError("yam32 requires YAM and all three camera views.")
        paths = {"main": video} | ({"left_wrist": left, "right_wrist": right} if left else {})
        return {r: _video(p) for r, p in paths.items()}, profiles[profile]

    def frames_for(videos, wanted):
        time.sleep(random.uniform(0, 0.003))  # "CPU work"
        return {i: f"{videos['main'].sha256[:8]}:{i}" for i in sorted(wanted)}

    pipeline.frames_for = frames_for

    def segment(videos, pool, session, instruction, progress):
        model, proc = session.load("tempseg")
        windows = 2 + int(videos["main"].sha256, 16) % 4

        def predict():  # NVIDIA's segmenter runs its predictor in asyncio.to_thread
            outs = []
            for w in range(windows):
                frames = pipeline.frames_for(videos, {w, w + 1})
                inputs = proc(text=f"{pool}:{w}:{sorted(frames.values())}")
                outs.append(pipeline.generate(model, proc, inputs, 1024))
            return outs

        async def run():
            return await asyncio.to_thread(predict)

        outs = asyncio.run(run())
        segs = [{"start_s": float(i), "end_s": float(i) + 1.0, "metadata": {"o": o}}
                for i, o in enumerate(outs)]
        return segs, [{"window": i, "output": o} for i, o in enumerate(outs)]

    def caption_single(videos, segments, profile, session, instruction, progress):
        model, proc = session.load("captioner")
        rows, audit = [], []
        for i, s in enumerate(segments):
            frames = pipeline.frames_for(videos, {i})
            inputs = proc(text=f"{profile}:{s['metadata']['o']}:{list(frames.values())}")
            cap = pipeline.generate(model, proc, inputs, 512)
            rows.append(dict(start_s=s["start_s"], end_s=s["end_s"], caption=cap,
                             status="complete"))
            audit.append({"event_id": f"event_{i:04d}"})
        return rows, audit

    def caption_three(videos, segments, session, instruction, progress, profile="YAM"):
        calls.append(("caption_three", profile))
        return caption_single(videos, segments, profile, session, instruction, progress)

    def annotate(videos, profile, long_windows, instruction, progress):
        if long_windows and len(videos) != 3:
            raise InputError("The 32-second profile requires three synchronized camera views.")
        pool = "yam32" if long_windows else pools[profile]
        session = pipeline.Session()
        try:
            progress(0.1, desc="Loading the action segmenter")
            segments, temporal = pipeline.segment(videos, pool, session, instruction, progress)
            runtime_event("segmentation_complete", windows=len(temporal), segments=len(segments))
            if len(videos) == 3:
                captions, cap_audit = pipeline.caption_three(videos, segments, session,
                                                             instruction, progress,
                                                             profile=profile)
            else:
                captions, cap_audit = pipeline.caption_single(videos, segments, profile, session,
                                                              instruction, progress)
            for row in captions:
                row["segmentation_provenance"] = "fine"
            runtime_event("annotation_complete", segments=len(captions))
            return dict(schema_version="fake_v1", duration_s=videos["main"].duration_s,
                        segments=captions, elapsed_s=random.random(), camera_roles=list(videos),
                        profile=profile, input_sha256={r: v.sha256 for r, v in videos.items()},
                        pipeline=dict(temporal_windows=temporal, caption_inputs=cap_audit))
        finally:
            session.close()

    class DataEnginePipeline:
        def annotate(self, video, *, profile="human", left_wrist=None, right_wrist=None,
                     yam32=False, instruction="", progress=None, include_audit=False):
            calls.append(("api.annotate", profile, yam32))
            with api._REQUEST_LOCK:
                videos, canonical = _inputs(video, profile, left_wrist, right_wrist, yam32,
                                            instruction)
                result = pipeline.annotate(videos, canonical, yam32, instruction,
                                           progress or (lambda *a, **k: None))
            if not include_audit:
                result.pop("pipeline", None)
            return result

    pipeline.segment, pipeline.caption_single = segment, caption_single
    pipeline.caption_three, pipeline.annotate = caption_three, annotate
    api.DataEnginePipeline = DataEnginePipeline
    return SimpleNamespace(
        api=api,
        media=SimpleNamespace(InputError=InputError),
        config=SimpleNamespace(PROFILE_POOLS=pools, MODELS={"tempseg": {"revision": "r"}}),
        models=SimpleNamespace(runtime_event=runtime_event),
        pipeline=pipeline,
    )


class FakeNvSession:
    def __init__(self):
        self.audit = {}

    def load(self, kind):
        return f"model-{kind}", FakeProc(kind)

    def close(self):
        pass


EPISODES = [
    {"episode": f"ep{i}", "profile": "yam" if i % 3 == 0 else "human",
     "videos": ({"main": f"/v/{i}.mp4", "left_wrist": f"/v/{i}l.mp4",
                 "right_wrist": f"/v/{i}r.mp4"} if i % 3 == 0 else {"main": f"/v/{i}.mp4"})}
    for i in range(7)
]


def setup(lock: bool = True):
    probe = GpuProbe()
    nv = fake_nv(probe)
    session = CachedSession(FakeNvSession)
    gpu_lock, patcher = TimedLock("gpu"), Patcher()
    if lock:
        ref.install_gpu_lock(nv, gpu_lock, patcher)
    ref.install_nvidia_entry(nv, patcher, request_lock=False)
    return probe, nv, session, gpu_lock, patcher


def test_concurrent_outputs_identical_to_sequential():
    probe, nv, session, gpu_lock, patcher = setup()
    seq = ref.annotate_batch(nv, session, EPISODES, concurrency=1, gpu_lock=gpu_lock)
    calls_seq = probe.calls
    conc = ref.annotate_batch(nv, session, EPISODES, concurrency=4, gpu_lock=gpu_lock,
                              measure=True)
    assert not seq["errors"] and not conc["errors"]
    assert list(conc["results"]) == [e["episode"] for e in EPISODES]  # input order kept
    cmp = ref.compare_runs(seq["results"], conc["results"])
    assert cmp["identical"], cmp
    assert probe.calls == 2 * calls_seq
    assert probe.max_active == 1  # the GPU lock serialized every generate()
    lock = conc["batch"]["gpu_lock"]
    assert lock["acquisitions"] == calls_seq and lock["contended"] > 0
    assert 0 < conc["batch"]["gpu_busy_fraction"] <= 1
    assert conc["batch"]["processor_clones"]["new_clones"] >= 2
    patcher.restore()
    assert nv.pipeline.generate is probe


def test_runtime_events_stay_with_their_episode():
    probe, nv, session, gpu_lock, _ = setup()
    seq = ref.annotate_batch(nv, session, EPISODES, concurrency=1, gpu_lock=gpu_lock)
    conc = ref.annotate_batch(nv, session, EPISODES, concurrency=5, gpu_lock=gpu_lock)
    for name, result in seq["results"].items():
        a = result["flora_reference"]["generation"]
        b = conc["results"][name]["flora_reference"]["generation"]
        assert a == b  # token counts from each episode's own FLORA_RUNTIME lines only
        assert a["segment"]["calls"] == len(result["pipeline"]["temporal_windows"])
        assert a["caption"]["calls"] == len(result["segments"])
        tok = result["flora_reference"]["tokens"]["caption"]
        assert tok["output_limit"] == [512] and tok["output_tokens"]["count"] == len(
            result["segments"])


def test_each_worker_thread_uses_its_own_processor():
    probe, nv, session, gpu_lock, _ = setup()
    shared = {id(session.load(k)[1]) for k in ("tempseg", "captioner")}
    ref.annotate_batch(nv, session, EPISODES, concurrency=3, gpu_lock=gpu_lock)
    used = set().union(*probe.procs.values())
    assert not used & shared  # never the shared processors
    assert 2 < len(used) <= 2 * 3  # one clone per kind per worker thread
    # sequential: the original processors, no clones
    probe.procs.clear()
    out = ref.annotate_batch(nv, session, EPISODES[:2], concurrency=1, gpu_lock=gpu_lock)
    assert set().union(*probe.procs.values()) <= shared
    assert out["batch"]["processor_clones"] is None


def test_concurrency_requires_gpu_lock():
    _, nv, session, _, _ = setup(lock=False)
    with pytest.raises(RuntimeError, match="GPU lock"):
        ref.annotate_batch(nv, session, EPISODES, concurrency=2)
    ref.annotate_batch(nv, session, EPISODES[:1], concurrency=1)  # sequential needs no lock


def test_install_gpu_lock_twice_refuses():
    _, nv, _, gpu_lock, patcher = setup()
    with pytest.raises(RuntimeError, match="already"):
        ref.install_gpu_lock(nv, gpu_lock, Patcher())
    patcher.restore()
    ref.install_gpu_lock(nv, gpu_lock, patcher)


def test_without_lock_generate_overlaps():
    """Control: the fake really would overlap without the lock (the test above is sensitive)."""
    probe, nv, session, _, _ = setup(lock=False)
    lock = TimedLock("fake")
    probe.__flora_gpu_lock__ = lock  # passes the guard without serializing anything
    ref.annotate_batch(nv, session, EPISODES, concurrency=4, gpu_lock=lock)
    assert probe.max_active > 1


def test_errors_are_per_episode():
    probe, nv, session, gpu_lock, _ = setup()
    bad = [*EPISODES[:2], {"episode": "bad", "profile": "human",
                           "videos": {"main": "/m", "left_wrist": "/l"}}]
    out = ref.annotate_batch(nv, session, bad, concurrency=2, gpu_lock=gpu_lock)
    assert set(out["results"]) == {"ep0", "ep1"}
    assert "camera roles" in out["errors"]["bad"]
    with pytest.raises(ValueError, match="duplicate"):
        ref.annotate_batch(nv, session, [EPISODES[0], EPISODES[0]], concurrency=1)
    with pytest.raises(ValueError):
        ref.annotate_batch(nv, session, EPISODES, concurrency=0)


def test_measurement_attaches_per_episode_tables():
    probe, nv, session, gpu_lock, _ = setup()
    with Patcher() as p:
        p.time(nv.pipeline, "frames_for", "media.frames_for")
        out = ref.annotate_batch(nv, session, EPISODES[:3], concurrency=3, gpu_lock=gpu_lock,
                                 measure=True)
    for result in out["results"].values():
        obs = result["flora_reference"]["observe"]
        calls = obs["calls"]
        n_gen = sum(result["flora_reference"]["generation"][s]["calls"]
                    for s in ("segment", "caption"))
        assert calls["models.generate"]["calls"] == n_gen
        assert calls["lock:gpu"]["calls"] == n_gen
        assert calls["media.frames_for"]["calls"] == n_gen  # one per model call in the fake
        assert obs["gpu_lock"]["calls"] == n_gen
        assert obs["untimed_s"] >= -0.05
    merged = out["batch"]["calls"]
    assert merged["models.generate"]["calls"] == probe.calls


def test_compare_runs_reports_paths_not_values():
    a = {"e": {"segments": [{"caption": "secret one"}], "elapsed_s": 1.0,
               "flora_reference": {"x": 1}}}
    b = {"e": {"segments": [{"caption": "secret two"}], "elapsed_s": 9.0,
               "flora_reference": {"x": 2}}}
    cmp = ref.compare_runs(a, b)
    assert not cmp["identical"]
    assert cmp["episodes"]["e"]["diff_paths"] == ["$.segments[0].caption"]
    assert "secret" not in json.dumps(cmp)
    b["e"]["segments"][0]["caption"] = "secret one"
    assert ref.compare_runs(a, b)["identical"]  # elapsed_s and flora_reference ignored
    assert ref.result_digest(a["e"]) == ref.result_digest(b["e"])
    assert not ref.compare_runs(a, {})["identical"]
    assert ref.diff_paths([1, 2], [1]) == ["$[len]"]


def test_processor_clone_leases():
    clones = ref.ProcessorClones()
    proc = FakeProc("k")
    with clones.lease() as one:
        a = one.get("k", proc)
        assert a is not proc and one.get("k", proc) is a
        with clones.lease() as two:  # a concurrent episode never shares a clone
            b = two.get("k", proc)
            assert b is not a
        assert one.get("k", FakeProc("k")) is not a  # a new base proc gets a fresh clone
    assert clones.stats() == {"clones": 3, "seconds": clones.stats()["seconds"], "sets": 2}
    with clones.lease() as again:  # sets are reused (no new set is made)
        again.get("k", proc)
    assert clones.stats()["clones"] == 4  # the reused set held a clone of the other proc


def test_clone_pool_is_reused_across_batches():
    probe, nv, session, gpu_lock, _ = setup()
    pool = ref.ProcessorClones()
    first = ref.annotate_batch(nv, session, EPISODES, concurrency=3, gpu_lock=gpu_lock,
                               clones=pool)
    second = ref.annotate_batch(nv, session, EPISODES, concurrency=3, gpu_lock=gpu_lock,
                                clones=pool)
    assert 0 < first["batch"]["processor_clones"]["new_clones"] <= 2 * 3
    assert second["batch"]["processor_clones"]["new_clones"] == 0
    assert ref.compare_runs(first["results"], second["results"])["identical"]


def test_measurement_targets_cover_hot_path():
    nv = SimpleNamespace(media=SimpleNamespace(probe=1, decode_selected=2),
                         pipeline=SimpleNamespace(frames_for=3, human_transport=4,
                                                  temporal_inputs=5,
                                                  three_view_caption_inputs=6))
    labels = {t[2] for t in ref.measurement_targets(nv, [FakeProc("a"), FakeProc("b")])
              if isinstance(t[2], str)}
    assert {"media.decode_selected", "media.frames_for", "preprocessing.human_transport",
            "preprocessing.temporal_inputs", "preprocessing.three_view_caption_inputs",
            "hf.processor", "pil.resize"} <= labels
    procs = [t for t in ref.measurement_targets(nv, [FakeProc("a"), FakeProc("b")])
             if t[2] == "hf.processor"]
    assert len(procs) == 1  # one patch per processor class


def test_pil_save_is_timed_by_format(tmp_path):
    from PIL import Image

    from flora_deploy.observe import Recorder, call_table, recording

    nv = SimpleNamespace(media=SimpleNamespace(probe=1, decode_selected=2),
                         pipeline=SimpleNamespace(frames_for=3, human_transport=4,
                                                  temporal_inputs=5,
                                                  three_view_caption_inputs=6))
    rec = Recorder()
    with Patcher() as p:
        ref.install_measurement(nv, p)
        img = Image.new("RGB", (8, 8))
        with recording(rec):
            img.save(tmp_path / "a.png", format="PNG")
            img.save(tmp_path / "b.jpg")
            img.resize((4, 4))
    assert "save" not in vars(Image.Image) or not hasattr(Image.Image.save, "__wrapped__")
    table = call_table(rec.events)
    assert table["pil.save.PNG"]["calls"] == 1 and table["pil.save.JPG"]["calls"] == 1
    assert table["pil.resize"]["calls"] == 1


# --- S1: NVIDIA's own entry point, fresh vs resident session, identity summary ---------------


def test_concurrency_needs_request_lock_swapped():
    probe = GpuProbe()
    nv = fake_nv(probe)
    gpu_lock, patcher = TimedLock("gpu"), Patcher()
    ref.install_gpu_lock(nv, gpu_lock, patcher)
    ref.install_nvidia_entry(nv, patcher)  # NVIDIA's whole-request RLock kept
    session = CachedSession(FakeNvSession)
    with pytest.raises(RuntimeError, match="request_lock"):
        ref.annotate_batch(nv, session, EPISODES, concurrency=2, gpu_lock=gpu_lock)
    out = ref.annotate_batch(nv, session, EPISODES[:2], concurrency=1, gpu_lock=gpu_lock)
    assert not out["errors"]
    with pytest.raises(RuntimeError, match="already bound"):
        ref.install_nvidia_entry(nv, Patcher())


def test_fresh_nvidia_session_matches_resident_concurrent():
    """The identity-proof shape: NVIDIA's own fresh Session per video, sequential, no swaps,
    vs the resident session with 4 threads under the GPU lock."""
    probe = GpuProbe()
    calls: list = []
    nv = fake_nv(probe, calls)
    original = (nv.pipeline.Session, nv.pipeline.generate, nv.api._REQUEST_LOCK)
    gpu_lock, patcher = TimedLock("gpu"), Patcher()
    ref.install_gpu_lock(nv, gpu_lock, patcher)
    ref.install_nvidia_entry(nv, patcher, request_lock=False)
    session = CachedSession(FakeNvSession)
    fast = ref.annotate_batch(nv, session, EPISODES, concurrency=4, gpu_lock=gpu_lock)
    assert not [c for c in calls if c[0] == "fresh_session"]  # resident session only
    with pytest.raises(RuntimeError, match="restore"):
        ref.annotate_nvidia(nv, None, EPISODES[0]["videos"], "human")
    patcher.restore()
    assert (nv.pipeline.Session, nv.pipeline.generate, nv.api._REQUEST_LOCK) == original
    with pytest.raises(RuntimeError, match="not active"):
        ref.annotate_nvidia(nv, session, EPISODES[0]["videos"], "human")
    with pytest.raises(RuntimeError, match="resident"):
        ref.annotate_batch(nv, None, EPISODES, concurrency=2)
    calls.clear()
    nvidia = ref.annotate_batch(nv, None, EPISODES, concurrency=1)
    assert sum(c[0] == "fresh_session" for c in calls) == len(EPISODES)
    assert sum(c[0] == "fresh_close" for c in calls) == len(EPISODES)
    ident = ref.identity_summary(nvidia["results"], fast["results"])
    assert ident["gate_ok"] and ident["compared"] == len(EPISODES)
    assert ident["strict_episodes"] == ident["output_identical_episodes"] == len(EPISODES)
    assert ident["same_input_different_output"] == 0
    assert ident["gated_calls"] == ident["calls_a"] == probe.calls // 2
    for r in nvidia["results"].values():
        assert r["flora_reference"]["session"] == "nvidia_fresh"
        mc = r["flora_reference"]["model_calls"]
        assert all(c["output_sha256"] and c["input_digest"] for c in mc)
        t = r["flora_reference"]["timings_s"]
        assert all(t[k] is not None and t[k] >= 0 for k in ("probe", "segment", "caption"))


def test_accel_episode_scope_and_provenance():
    probe, nv, session, gpu_lock, _ = setup()

    class FakeAccel:
        depth = max_depth = 0

        @contextlib.contextmanager
        def episode(self):
            FakeAccel.depth += 1
            FakeAccel.max_depth = max(FakeAccel.max_depth, FakeAccel.depth)
            try:
                yield
            finally:
                FakeAccel.depth -= 1

        def provenance(self):
            return {"schema": "flora_accel_patches_v1", "patches": []}

    out = ref.annotate_batch(nv, session, EPISODES, concurrency=3, gpu_lock=gpu_lock,
                             accel=FakeAccel())
    assert not out["errors"] and FakeAccel.depth == 0 and FakeAccel.max_depth >= 2
    assert all(r["flora_reference"]["accel"]["schema"] == "flora_accel_patches_v1"
               for r in out["results"].values())


def _call(d, out="o", complete=True, tok=3):
    return {"input_digest": d, "output_sha256": out, "complete": complete,
            "generated_tokens": tok}


def test_compare_model_inputs_gate():
    a = [_call("x"), _call("y"), _call("z")]
    assert ref.compare_model_inputs(a, list(a))["strict"]
    # an output differs at call 1: call 2's input may differ (downstream), the gate holds
    b = [_call("x"), _call("y", out="p"), _call("w")]
    r = ref.compare_model_inputs(a, b)
    assert r["gate_ok"] and not r["strict"] and r["first_output_difference_call"] == 1
    assert r["gated_calls"] == 2 and r["same_input_different_output"] == 1
    # an input differs while everything upstream agreed: violation
    c = [_call("x"), _call("q"), _call("z")]
    r = ref.compare_model_inputs(a, c)
    assert not r["gate_ok"] and r["gate_violation_call"] == 1
    # one run made more calls with everything else equal: violation
    r = ref.compare_model_inputs(a, a[:2])
    assert not r["gate_ok"] and r["gate_violation_call"] == 2
    assert ref.compare_model_inputs(a, a[::-1])["multiset"]
