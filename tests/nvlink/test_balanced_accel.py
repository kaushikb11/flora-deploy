"""OPT-1 accel in the balanced workers: every model input is byte-identical with and without it.

The balanced episode jobs (``workers_impl.segment.SegmentEpisodeJob`` /
``workers_impl.caption.CaptionEpisodeJob``) run NVIDIA's real code (probe, ``Predictor.predict``,
``caption_single`` / ``caption_three``, NVIDIA's decode of real H.264 files made by
``media.synth``) through the real Flora processors and the micro-batcher, with and without
``workers_impl.balanced.wrap_job(..., active=apply_accel(nv))`` (all registered patches, a
per-episode ``active.episode()`` scope in each episode coroutine). Two episodes (1-view human,
3-view YAM) overlap on one event loop. The engine answers each request as a function of its
content digest (``stages.segment.request_digest``: prompt + every frame + metadata), so equal
inputs give equal answers; the test asserts the multiset of request digests, the rows and the
audits are identical, and that the patches really ran (decode-once lookahead used). It also
asserts the multiset of **literal** digests of what vLLM receives
(``tests.fakes.request_digest.prompt_dict_digest``: the wrapped prompt, every frame array with
its dtype and shape, the metadata and ``mm_processor_kwargs``; review 10 L3).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from flora_deploy.engines.base import Completion
from flora_deploy.engines.microbatch import MicroBatcher
from flora_deploy.jobs.types import EpisodeRef
from flora_deploy.media import synth
from flora_deploy.media.synth import Spec
from flora_deploy.nv import Family
from flora_deploy.nv.linker import load_family
from flora_deploy.stages import caption as C
from flora_deploy.stages import segment as S
from flora_deploy.workers_impl.balanced import AccelJob, apply_accel, wrap_job
from flora_deploy.workers_impl.caption import CaptionEpisodeJob
from flora_deploy.workers_impl.segment import EpisodeInputs, SegmentEpisodeJob, StageInfo
from tests.fakes.request_digest import prompt_dict_digest

pytestmark = pytest.mark.nvlink

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROLES3 = ("main", "left_wrist", "right_wrist")


@pytest.fixture(scope="module")
def nv_dirs():
    import os

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[k]) for k in ("tempseg_code", "captioner_code")}
    for key, path in dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    return dirs


@pytest.fixture(scope="module")
def nv(nv_dirs):
    return load_family(Family.SEGCAP, nv_dirs["tempseg_code"])


@pytest.fixture(scope="module")
def procs(nv, nv_dirs):
    return {"tempseg": nv.models.processor("tempseg", nv_dirs["tempseg_code"]),
            "captioner": nv.models.processor("captioner", nv_dirs["captioner_code"])}


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    root = tmp_path_factory.mktemp("balanced-accel")
    human = {"main": str(synth.encode(root / "human_main.mp4",
                                      Spec(300, size=(112, 64), gop=24, bframes=2, seed=1)))}
    yam = {role: str(synth.encode(root / f"yam_{role}.mp4",
                                  Spec(300, size=(112, 64), gop=20, bframes=2, seed=10 + k)))
           for k, role in enumerate(ROLES3)}
    return [("ds--human", human, "human", ""), ("ds--yam", yam, "yam", "Fold the towel.")]


class DigestEngine:
    """Answers by request content digest; records every digest it saw."""

    name = "digest"

    def __init__(self, tokenizer, kind: str):
        self.tok, self.kind = tokenizer, kind
        self.eos = tokenizer.eos_token_id
        self.digests: list[str] = []
        self.literal: list[str] = []  # prompt_dict() as vLLM gets it (L3)

    def generate_batch(self, requests, samplings):
        out = []
        for r in requests:
            d = S.request_digest(r.prompt, r.videos)
            self.digests.append(d)
            self.literal.append(prompt_dict_digest(r))
            text = '{"segments": []}' if self.kind == "tempseg" else f"the arm moves {d[:6]}"
            ids = tuple(self.tok.encode(text, add_special_tokens=False)) + (self.eos,)
            out.append(Completion(key=r.key, token_ids=ids, finish_reason="stop",
                                  prompt_tokens=1000))
        return out


def _ref(uid, pool):
    return EpisodeRef(uid=uid, dataset="ds", episode_id=uid.split("--")[1], pool=pool,
                      duration_s=10.0, oracle_eligible=True, envelope_ok=True, index_ref="x")


def _inputs(episodes):
    by = {uid: EpisodeInputs(uid=uid, dataset="ds", episode_id=uid.split("--")[1], paths=paths,
                             profile=pool, yam32=False, instruction=instr)
          for uid, paths, pool, instr in episodes}
    return lambda ref: by[ref.uid]


def _segments_record(ref, _ctx):
    from flora_deploy.records import Provenance

    rows = [dict(start_s=0.5, end_s=3.0, metadata={"segmentation_provenance": "fine"}),
            dict(start_s=3.5, end_s=7.0, metadata={}), dict(start_s=7.5, end_s=9.5, metadata={})]
    prov = Provenance(models={}, engine="t", preset="balanced", config_hash="c",
                      logic_version="t", framework="t")
    return S.segment_record(rows, uid=ref.uid, dataset="ds", episode_id=ref.episode_id,
                            provenance=prov)


def _run(nv, procs, episodes, kind, accel: bool):
    proc = procs[kind]
    engine = DigestEngine(proc.tokenizer, kind)
    info = StageInfo(engine_label="vllm-test")
    inputs = _inputs(episodes)
    if kind == "tempseg":
        stage = S.SegmentStage(nv, proc)
        job = SegmentEpisodeJob(stage, info, inputs, budget=1024)
    else:
        stage = C.CaptionStage(nv, proc)
        job = CaptionEpisodeJob(stage, info, inputs, segments=_segments_record, budget=512)
    active = apply_accel(nv) if accel else None
    job = wrap_job(job, active=active)
    if accel:
        assert isinstance(job, AccelJob)

    async def one(mb, uid, pool):
        async with mb.episode():
            try:
                out = await job.run(_ref(uid, pool), mb, None)
                return ("ok", [r.to_json() for r in out.records],
                        json.dumps([{k: v for k, v in a.items() if k not in ("timings_s",
                                                                              "accel")}
                                    for a in out.audits], sort_keys=True, default=str))
            except Exception as exc:  # noqa: BLE001 - compared across runs
                return ("error", type(exc).__name__, str(exc))

    async def main():
        async with MicroBatcher(engine, max_wait_s=30) as mb:
            return await asyncio.gather(*(one(mb, uid, pool)
                                          for uid, _p, pool, _i in episodes))

    try:
        outs = asyncio.run(main())
        stats = active.stats() if active is not None else {}
    finally:
        stage.close()
        if active is not None:
            active.restore()
    return outs, (sorted(engine.digests), sorted(engine.literal)), stats


@pytest.mark.parametrize("kind", ["tempseg", "captioner"])
def test_balanced_inputs_identical_with_and_without_accel(nv, procs, episodes, kind):
    ref_out, ref_digests, _ = _run(nv, procs, episodes, kind, accel=False)
    got_out, got_digests, stats = _run(nv, procs, episodes, kind, accel=True)
    assert ref_digests[0], "no model request was made"
    assert got_digests[0] == ref_digests[0]  # every request: prompt, every frame, metadata
    assert got_digests[1] == ref_digests[1]  # literally: wrapped prompt, dtypes, mm kwargs
    assert got_out == ref_out  # records and NVIDIA's audits
    assert all(o[0] == "ok" for o in ref_out), ref_out
    assert stats.get("frames_for_calls", 0) >= 1  # the patches were on the decode path
    assert stats.get("active_episodes", 0) == 0  # every episode scope was released
