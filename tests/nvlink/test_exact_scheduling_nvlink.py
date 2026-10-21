"""G3 on NVIDIA's real code (CPU, fake models; ``docs/exact-cost.md``).

Same harness as ``test_reference_entry`` (NVIDIA's real ``api.DataEnginePipeline().annotate``,
real ``models.generate`` logging ``input_tensor_digest``, real Flora processors, synthetic
H.264/AV1 clips, a deterministic fake model):

1. **Shard overlap + longest-first**: two batches run at once in one process, sharing two
   ``EpisodeSlots`` (as two overlapped shards in one container), episodes started in reverse
   plan order, against NVIDIA unmodified (fresh sessions, sequential, plan order). Every
   model call's input digest and every full NVIDIA result must be identical.
2. **Split TempSeg / Captioner** (``workers_impl.exact_split``): NVIDIA's ``annotate`` run as a
   TempSeg step, its segment result persisted as bytes, and a separate Captioner step (own
   fresh NVIDIA ``Session``, own probe) fed from those bytes. The Captioner calls' input
   digests, the TempSeg calls' input digests and the full result must equal the
   single-process run.
"""

from __future__ import annotations

import threading

import pytest

from flora_deploy.nv.session import CachedSession
from flora_deploy.observe import Patcher, TimedLock
from flora_deploy.stages import reference as ref
from tests.nvlink.test_reference_entry import clips, episodes, harness, nv_env  # noqa: F401

pytestmark = pytest.mark.nvlink


def digests(result, model=None):
    return [c["input_digest"] for c in result["flora_reference"]["model_calls"]
            if model is None or c["model"] == model]


def test_overlapped_batches_longest_first_identical_to_unmodified(harness, clips):  # noqa: F811
    from flora_deploy.accel import ALL_PATCHES, apply_patches

    nv, fresh = harness
    eps = episodes(clips)
    unmodified = ref.annotate_batch(nv, None, eps, concurrency=1)
    assert not unmodified["errors"], unmodified["errors"]

    patcher, lock = Patcher(), TimedLock("gpu")
    ref.install_gpu_lock(nv, lock, patcher)
    ref.install_nvidia_entry(nv, patcher, request_lock=False)
    accel = apply_patches(nv, ALL_PATCHES)
    slots, clones, session = ref.EpisodeSlots(2), ref.ProcessorClones(), CachedSession(fresh)
    shards = [list(reversed(eps[:2])), list(reversed(eps[2:]))]  # not plan order
    out: dict[int, dict] = {}
    try:
        def run(i):
            out[i] = ref.annotate_batch(nv, session, shards[i], concurrency=2, gpu_lock=lock,
                                        accel=accel, clones=clones, slots=slots,
                                        slot_key=i, reset_lock=i == 0)

        ts = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(600)
    finally:
        accel.restore()
        patcher.restore()
    fast = {**out[0]["results"], **out[1]["results"]}
    assert not out[0]["errors"] and not out[1]["errors"]
    assert slots.stats()["max_running"] <= 2 and slots.stats()["grants"] == len(eps)
    ident = ref.identity_summary(unmodified["results"], fast)
    assert ident["gate_ok"] and ident["strict_episodes"] == len(eps), ident
    assert ident["output_identical_episodes"] == len(eps), ident


def test_split_tempseg_captioner_identical_to_single_process(harness, clips):  # noqa: F811
    from flora_deploy.workers_impl import exact_split as X

    nv, _fresh = harness
    eps = episodes(clips)
    single = ref.annotate_batch(nv, None, eps, concurrency=1)
    assert not single["errors"], single["errors"]

    patcher = Patcher()
    X.install_split(nv, patcher)
    try:
        for ep in eps:
            doc = X.segment_episode(nv, None, ep)  # "TempSeg pool": NVIDIA's own Session
            blob = X.encode(doc)  # what the TempSeg worker would persist
            back = X.decode(blob)
            assert back == doc and X.encode(back) == blob  # faithful, types included
            result = X.caption_episode(nv, None, ep, back)  # "Captioner pool": new Session
            one = single["results"][ep["episode"]]
            assert [c["input_digest"] for c in doc["model_calls"]] == digests(one, "tempseg")
            assert all(c["model"] == "tempseg" for c in doc["model_calls"])
            assert digests(result, "captioner") == digests(one, "captioner"), ep["episode"]
            assert digests(result, "tempseg") == []  # no TempSeg call in the Captioner step
            assert digests(one, "captioner"), ep["episode"]  # the Captioner really ran
            assert ref.comparable_result(result) == ref.comparable_result(one), ep["episode"]
        # NVIDIA's result for an episode passes through untouched when no step is active
        plain = ref.annotate_nvidia(nv, None, eps[0]["videos"], eps[0]["profile"],
                                    instruction=eps[0].get("instruction", ""))
        assert ref.comparable_result(plain) == ref.comparable_result(
            single["results"][eps[0]["episode"]])
        # a document for other inputs is refused
        doc = X.segment_episode(nv, None, eps[1])
        with pytest.raises(X.SplitError):
            X.caption_episode(nv, None, eps[0], dict(doc, episode=eps[0]["episode"]))
    finally:
        patcher.restore()
