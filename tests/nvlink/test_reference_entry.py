"""S1 on CPU: NVIDIA's own entry point, unmodified vs the fast ``exact`` composite.

The CPU twin of ``ReferenceWorker.identity_proof``. Both arms run NVIDIA's real
``api.DataEnginePipeline().annotate`` and NVIDIA's real ``models.generate`` (which computes and
logs ``input_tensor_digest``), with the real Flora processors and synthetic videos; only the
model is a fake (CPU, deterministic text):

- **unmodified**: NVIDIA's code with no swap at all; a fresh session per video (standing in for
  NVIDIA's ``Session``, which needs the weights), sequential;
- **composite**: GPU lock + ``pipeline.Session`` bound to one resident ``CachedSession`` +
  NVIDIA's request lock swapped + every OPT-1 ``accel`` patch, 3 episode threads.

``identity_summary`` must report every model call's input digest identical (strict), and with a
deterministic model the full NVIDIA results identical too. NVIDIA's own errors must propagate
unchanged through the composite.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from flora_deploy.media import synth
from flora_deploy.media.synth import Spec
from flora_deploy.nv import Family
from flora_deploy.nv.linker import load_family
from flora_deploy.nv.session import CachedSession
from flora_deploy.observe import Patcher, TimedLock
from flora_deploy.stages import reference as ref

pytestmark = pytest.mark.nvlink
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROLES3 = ("main", "left_wrist", "right_wrist")


@pytest.fixture(scope="module")
def nv_env():
    import os

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[k]) for k in ("tempseg_code", "captioner_code")}
    for key, path in dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    nv = load_family(Family.SEGCAP, dirs["tempseg_code"])
    procs = {"tempseg": nv.models.processor("tempseg", dirs["tempseg_code"]),
             "captioner": nv.models.processor("captioner", dirs["captioner_code"])}
    return nv, procs


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1")
    specs = {
        "human_h264": [Spec(240, size=(112, 64), gop=24, bframes=2, seed=1)],
        "yam_3v": [Spec(240, size=(112, 64), gop=20, seed=10 + k) for k in range(3)],
        "dex_3v_av1": [Spec(150, codec="libsvtav1", size=(64, 64), gop=25, seed=30 + k)
                       for k in range(3)],
    }
    out = {}
    for name, ss in specs.items():
        roles = ROLES3 if len(ss) == 3 else ("main",)
        out[name] = {r: str(synth.encode(root / f"{name}_{r}.mp4", s))
                     for r, s in zip(roles, ss, strict=True)}
    return out


class FakeModel(torch.nn.Module):
    """Deterministic stand-in for TempSeg / the Captioner behind NVIDIA's real ``generate``."""

    def __init__(self, tokenizer):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.tok = tokenizer

    def generate(self, input_ids, max_new_tokens, eos_token_id, **kwargs):
        text = '{"segments":[]}' if max_new_tokens == 1024 else (
            f"pick up item {int(input_ids.sum()) % 97}")
        eos = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]
        ids = self.tok(text, add_special_tokens=False)["input_ids"] + [eos]
        return torch.cat([input_ids, torch.tensor([ids], dtype=input_ids.dtype)], dim=1)


def canned_segments(videos):
    d = videos["main"].duration_s
    return [dict(start_s=0.0, end_s=min(1.5, d / 4), metadata={}),
            dict(start_s=d / 4, end_s=d / 2, metadata={}),
            dict(start_s=d / 2, end_s=d - 0.2, metadata={})]


@pytest.fixture
def harness(nv_env, monkeypatch):
    nv, procs = nv_env
    models = {k: FakeModel(p.tokenizer) for k, p in procs.items()}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # api.py's CUDA check
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    # NVIDIA's segmenter yields no segments for the fake TempSeg; keep its real TempSeg calls
    # and add segments so the caption stage runs (same wrapper in both arms).
    real_segment = nv.pipeline.segment

    def segment(videos, pool, session, instruction, progress):
        segs, audit = real_segment(videos, pool, session, instruction, progress)
        return (segs or canned_segments(videos)), audit

    monkeypatch.setattr(nv.pipeline, "segment", segment)

    class FreshSession:  # stands in for NVIDIA's Session: new processors per video
        def __init__(self):
            self.audit = {}

        def load(self, kind):
            return models[kind], copy.deepcopy(procs[kind])

        def close(self):
            pass

    monkeypatch.setattr(nv.pipeline, "Session", FreshSession)
    return nv, FreshSession


def episodes(clips):
    return [
        {"episode": "human", "videos": clips["human_h264"], "profile": "human",
         "instruction": "Fold the towel."},
        {"episode": "yam3", "videos": clips["yam_3v"], "profile": "yam"},
        {"episode": "dex3", "videos": clips["dex_3v_av1"], "profile": "dex",
         "instruction": "Stack the cups."},
        {"episode": "human_blank", "videos": clips["human_h264"], "profile": "human"},
    ]


def test_composite_model_inputs_identical_to_unmodified(harness, clips):
    from flora_deploy.accel import ALL_PATCHES, apply_patches

    nv, fresh = harness
    eps = episodes(clips)
    unmodified = ref.annotate_batch(nv, None, eps, concurrency=1)
    assert not unmodified["errors"], unmodified["errors"]

    patcher, lock = Patcher(), TimedLock("gpu")
    ref.install_gpu_lock(nv, lock, patcher)
    ref.install_nvidia_entry(nv, patcher, request_lock=False)
    accel = apply_patches(nv, ALL_PATCHES)
    try:
        fast = ref.annotate_batch(nv, CachedSession(fresh), eps, concurrency=3, gpu_lock=lock,
                                  accel=accel, clones=ref.ProcessorClones())
    finally:
        accel.restore()
        patcher.restore()
    assert not fast["errors"], fast["errors"]
    ident = ref.identity_summary(unmodified["results"], fast["results"])
    assert ident["compared"] == len(eps)
    assert ident["gate_ok"] and ident["strict_episodes"] == len(eps), ident
    assert ident["output_identical_episodes"] == len(eps), ident
    for name, r in fast["results"].items():
        calls = r["flora_reference"]["model_calls"]
        assert {c["model"] for c in calls} == {"tempseg", "captioner"}, name
        assert all(c["input_digest"] and c["output_sha256"] for c in calls), name
    assert fast["batch"]["gpu_lock"]["acquisitions"] == ident["calls_a"]
    assert nv.pipeline.Session is fresh  # restored


def test_nvidia_errors_unchanged_through_composite(harness, clips):
    nv, fresh = harness
    bad = [  # each rejected by NVIDIA's own api._inputs / pipeline.annotate
        dict(videos={"main": clips["human_h264"]["main"]}, profile="robot"),
        dict(videos={"main": clips["human_h264"]["main"]}, profile="yam", long_windows=True),
        dict(videos=clips["yam_3v"], profile="human"),
        dict(videos={"main": "/does/not/exist.mp4"}, profile="human"),
    ]

    def outcome(session):
        out = []
        for b in bad:
            try:
                ref.annotate_nvidia(nv, session, b["videos"], b["profile"],
                                    b.get("long_windows", False))
                out.append(None)
            except Exception as e:  # noqa: BLE001 - comparing NVIDIA's errors
                out.append((type(e).__qualname__, str(e)))
        return out

    reference = outcome(None)
    patcher = Patcher()
    ref.install_nvidia_entry(nv, patcher, request_lock=False)
    try:
        got = outcome(CachedSession(fresh))
    finally:
        patcher.restore()
    assert got == reference and all(reference)
    assert {r[0] for r in reference} >= {"InputError"}
