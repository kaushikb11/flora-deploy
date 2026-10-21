"""Exact-preset concurrency: GPU-lock semantics, per-thread processors, identical outputs.

A fake NVIDIA family mirrors the structure that matters: ``pipeline`` functions look up
``pipeline.generate`` at call time (NVIDIA imports it into ``pipeline`` at module level), the
segmenter calls its predictor through ``asyncio.to_thread``, input building goes through a
stateful processor, and ``generate`` prints ``FLORA_RUNTIME`` events.
"""

import asyncio
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


class GpuProbe:
    """Fake ``models.generate``: deterministic output; records overlap and procs seen."""

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
            time.sleep(random.uniform(0.001, 0.004))
            text = hashlib.sha256(f"{model}:{inputs['input_ids']}:{budget}".encode()).hexdigest()
            print("FLORA_RUNTIME " + json.dumps({"event": "generation_end", "input_tokens": 10,
                                                 "generated_tokens": len(text) % 7 + 1,
                                                 "output_limit": budget, "complete": True,
                                                 "elapsed_s": 0.01}), flush=True)
            return text[:16]
        finally:
            with self._lock:
                self.active -= 1


def _video(path, n=40):
    pts = tuple(Fraction(i * 250000) for i in range(n))
    return SimpleNamespace(path=Path(path), sha256=hashlib.sha256(str(path).encode()).hexdigest(),
                           rate=Fraction(4), pts_us=pts, duration_us=n * 250000,
                           duration_s=n / 4, size=(64, 48), cfr=True)


def fake_nv(probe: GpuProbe):
    profiles = {"human": "Human", "yam": "YAM"}
    pipeline = SimpleNamespace(generate=probe)

    def _inputs(video, profile, left, right, yam32, instruction):
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
        return caption_single(videos, segments, profile, session, instruction, progress)

    pipeline.segment, pipeline.caption_single = segment, caption_single
    pipeline.caption_three = caption_three
    return SimpleNamespace(
        api=SimpleNamespace(DataEnginePipeline=SimpleNamespace(_inputs=_inputs)),
        media=SimpleNamespace(check_synchronized=lambda v: None, InputError=InputError),
        preprocessing=SimpleNamespace(instruction_suffix=lambda s: ""),
        config=SimpleNamespace(PROFILE_POOLS={"Human": "human", "YAM": "yam"},
                               MODELS={"tempseg": {"revision": "r"}}),
        models=SimpleNamespace(runtime_event=lambda e, **kw: None),
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
    assert conc["batch"]["processor_clones"]["clones"] >= 2
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


def test_processor_clones_per_thread():
    clones = ref.ProcessorClones()
    proc = FakeProc("k")
    a = clones.get("k", proc)
    assert a is not proc and clones.get("k", proc) is a
    seen = []
    t = threading.Thread(target=lambda: seen.append(clones.get("k", proc)))
    t.start()
    t.join()
    assert seen[0] is not a and clones.clones == 2
    other = FakeProc("k")
    assert clones.get("k", other) is not a  # a different base proc gets a fresh clone


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
