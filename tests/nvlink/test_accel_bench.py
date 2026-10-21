"""OPT-1 microbenchmark: one episode through NVIDIA's pipeline, unpatched vs all patches.

Runs ``pipeline.segment`` followed by ``caption_single`` (1 view, ``human``) or
``caption_three`` (3 views, ``yam``) with the real Flora processors and a canned ``generate``
(no model), so the wall time is NVIDIA's CPU work. Reports decoder passes, frames decoded and
wall time per stage, asserts identical outputs, fewer decoded frames and a faster episode.
Set ``FLORA_ACCEL_BENCH_OUT=path.json`` to keep the numbers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from flora_deploy.accel import ALL_PATCHES, apply_patches
from flora_deploy.media import synth
from flora_deploy.media.synth import Spec
from flora_deploy.nv import Family
from flora_deploy.nv.linker import load_family
from flora_deploy.nv.session import CachedSession

pytestmark = pytest.mark.nvlink

TENSOR_KEYS = ("input_ids", "attention_mask", "video_grid_thw", "pixel_values_videos")
EPISODES = {
    # id: (spec per role, pool, caption profile)
    "1view-640x360-45s": ([Spec(1350, size=(640, 360), gop=60, seed=1)], "human", "Human"),
    "3view-320x240-20s": (
        [Spec(600, size=(320, 240), gop=60, seed=2 + k) for k in range(3)],
        "yam",
        "YAM",
    ),
}


@pytest.fixture(scope="module")
def nv_dirs():
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[k]) for k in ("tempseg_code", "captioner_code")}
    for key, path in dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return dirs


@pytest.fixture(scope="module")
def nv(nv_dirs):
    return load_family(Family.SEGCAP, nv_dirs["tempseg_code"])


@pytest.fixture(scope="module")
def procs(nv, nv_dirs):
    return {
        "tempseg": nv.models.processor("tempseg", nv_dirs["tempseg_code"]),
        "captioner": nv.models.processor("captioner", nv_dirs["captioner_code"]),
    }


def _episode(nv, procs, monkeypatch, videos, pool, profile):
    log = []

    class Fake:
        audit: dict = {}

        def load(self, kind):
            return object(), procs[kind]

        def close(self):
            pass

    def generate(model, proc, inputs, budget):
        log.append({k: inputs[k].numpy().tobytes() for k in TENSOR_KEYS if k in inputs})
        return '{"segments":[]}' if budget == 1024 else f"caption {len(log)}"

    monkeypatch.setattr(nv.pipeline, "generate", generate)
    session = CachedSession(Fake)
    progress = lambda *a, **k: None  # noqa: E731
    d = videos["main"].duration_s
    segments = [
        dict(start_s=a, end_s=min(d, a + 2.5), metadata={}) for a in range(1, int(d) - 1, 3)
    ]
    t0 = time.perf_counter()
    seg = nv.pipeline.segment(videos, pool, session, "", progress)
    t1 = time.perf_counter()
    if len(videos) == 3:
        cap = nv.pipeline.caption_three(videos, segments, session, "", progress, profile=profile)
    else:
        cap = nv.pipeline.caption_single(videos, segments, profile, session, "", progress)
    t2 = time.perf_counter()
    return (seg, cap, log), dict(segment_s=t1 - t0, caption_s=t2 - t1, total_s=t2 - t0)


@pytest.mark.parametrize("episode", sorted(EPISODES))
def test_accel_benchmark(nv, procs, monkeypatch, tmp_path_factory, episode):
    specs, pool, profile = EPISODES[episode]
    root = tmp_path_factory.mktemp("bench")
    roles = ("main", "left_wrist", "right_wrist")[: len(specs)]
    videos = {
        role: nv.media.probe(synth.encode(root / f"{role}.mp4", spec))
        for role, spec in zip(roles, specs, strict=True)
    }

    calls = []
    real = nv.media.decode_selected

    def counting(video, wanted, canonical=False):
        wanted = set(wanted)
        calls.append(max(wanted) + 1 if wanted else 0)  # NVIDIA decodes 0..max(wanted)
        return real(video, wanted, canonical)

    monkeypatch.setattr(nv.media, "decode_selected", counting)
    ref, ref_t = _episode(nv, procs, monkeypatch, videos, pool, profile)
    unpatched = dict(ref_t, decoder_passes=len(calls), frames_decoded=sum(calls))
    calls.clear()
    with apply_patches(nv, ALL_PATCHES) as active:
        with active.episode():
            got, got_t = _episode(nv, procs, monkeypatch, videos, pool, profile)
            stats = active.stats()
    assert not calls  # NVIDIA's decode_selected never ran
    patched = dict(
        got_t,
        decoder_passes=stats.get("decoder_restarts_from_0", 0)
        + stats.get("decoder_restarts_random_access", 0),
        frames_decoded=stats["frames_decoded"],
    )
    assert ref == got  # segments, captions, audits and every processor tensor
    result = dict(
        episode=episode,
        unpatched=unpatched,
        patched=patched,
        speedup=dict(
            segment=unpatched["segment_s"] / patched["segment_s"],
            caption=unpatched["caption_s"] / patched["caption_s"],
            total=unpatched["total_s"] / patched["total_s"],
        ),
        stats=stats,
        cpu_count=os.cpu_count(),
    )
    print(json.dumps(result, indent=1, default=str))
    out = os.environ.get("FLORA_ACCEL_BENCH_OUT")
    if out:
        path = Path(out)
        data = json.loads(path.read_text()) if path.is_file() else {}
        data[episode] = result
        path.write_text(json.dumps(data, indent=1, default=str))
    assert patched["frames_decoded"] < unpatched["frames_decoded"]
    assert patched["total_s"] < unpatched["total_s"]
