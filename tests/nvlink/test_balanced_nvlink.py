"""The balanced stages reproduce NVIDIA's own sync pipeline for canned model outputs.

Reference: NVIDIA's ``pipeline.segment`` / ``caption_three`` / ``caption_single`` exactly as
shipped (sync ``Predictor.predict``, one ``generate`` per window or caption plan), with two
substitutions: ``media.decode_selected`` returns synthetic frames and ``pipeline.generate``
answers from a canned model. The processor is NVIDIA's real one behind a ``CaptureProcessor``
(the capture parity tests prove that changes nothing upstream of ``generate``), so every call
the reference makes is keyed by its content digest (prompt, frames, metadata) -> canned text.

Batched: ``stages.segment.SegmentStage`` / ``stages.caption.CaptionStage`` with the micro-batcher
over a fake engine that answers each request **by its content digest** from that table, as
token ids from the real Flora tokenizer plus EOS. A request the reference never made is a
KeyError, so the batched path must build exactly NVIDIA's requests. Rows, NVIDIA's audit
(``Predictor.calls`` / caption inputs) and failure types must be identical, including NVIDIA's
recentered and directional retry rounds and its conservative completion.
"""

from __future__ import annotations

import asyncio
from fractions import Fraction
from pathlib import Path

import pytest

from flora_deploy.engines.base import Completion
from flora_deploy.engines.microbatch import MicroBatcher
from flora_deploy.engines.offline_llm import NV_VIDEO_KWARGS
from flora_deploy.nv import Family
from flora_deploy.nv.capture import CaptureProcessor
from flora_deploy.nv.linker import load_family
from flora_deploy.nv.session import CachedSession
from flora_deploy.stages import caption as C
from flora_deploy.stages import segment as S

pytestmark = pytest.mark.nvlink

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.nvlink.test_nv_link_capture import FakeNvSession, synthetic_frame  # noqa: E402
from tests.nvlink.test_nv_link_segmenter import SCENARIOS, Layout  # noqa: E402

FPS = 30
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


def make_videos(nv, roles, duration_s: float, tag: str = ""):
    n = int(round(duration_s * FPS))
    return {
        role: nv.media.Video(
            path=Path(f"{tag}{role}.mp4"), sha256=f"{i + len(tag):02d}" * 32, rate=Fraction(FPS),
            pts_us=tuple(Fraction(k * 1_000_000, FPS) for k in range(n)),
            duration_us=int(round(duration_s * 1_000_000)), size=(640, 480), cfr=True)
        for i, role in enumerate(roles)
    }


def fake_decode(video, wanted, canonical=False):
    stem = video.path.stem
    role = next((r for r in ROLES3 if stem.endswith(r)), stem)
    salt = sum(map(ord, stem[: len(stem) - len(role)]))  # different episodes, different pixels
    return {i: synthetic_frame(role, i + 7919 * salt) for i in wanted}


def no_progress(*_a, **_k):
    return None


class Canned:
    """digest -> texts in call order (None: the call fails like a missing EOS); filled by the
    reference. A real model is a function of its inputs; NVIDIA's retry scenarios replay one
    window with a different canned answer, so answers are kept per occurrence."""

    def __init__(self):
        self.table: dict[str, list[str | None]] = {}
        self.prompt_tokens: dict[str, int] = {}
        self.served: dict[str, int] = {}

    def record(self, req, text):
        d = S.request_digest(req.prompt, req.videos)
        self.table.setdefault(d, []).append(text)
        return d

    def answer(self, d):
        k = self.served.get(d, 0)
        self.served[d] = k + 1
        return self.table[d][k]  # KeyError / IndexError: a request NVIDIA never made


class CannedEngine:
    name = "canned"

    def __init__(self, canned: Canned, tokenizer):
        self.canned, self.tok = canned, tokenizer
        self.eos = tokenizer.eos_token_id
        self.batches: list[list[str]] = []

    def generate_batch(self, requests, samplings):
        self.batches.append([r.key for r in requests])
        out = []
        for r, sp in zip(requests, samplings, strict=True):
            d = S.request_digest(r.prompt, r.videos)
            text = self.canned.answer(d)
            n = self.canned.prompt_tokens.get(d, 1000)
            if text is None:  # cut off at the budget, no EOS
                ids = tuple(self.tok.encode("x " * sp.max_tokens, add_special_tokens=False)
                            [: sp.max_tokens])
                out.append(Completion(key=r.key, token_ids=ids, finish_reason="length",
                                      prompt_tokens=n))
                continue
            ids = tuple(self.tok.encode(text, add_special_tokens=False)) + (self.eos,)
            out.append(Completion(key=r.key, token_ids=ids, finish_reason="stop",
                                  prompt_tokens=n))
        return out


# ---------------------------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------------------------


def reference_segment(nv, procs, monkeypatch, videos, pool, text_for, canned, *,
                      instruction=""):
    """NVIDIA's ``pipeline.segment``; ``text_for(round, window)`` is the canned model."""
    pipeline = nv.pipeline
    stash, rounds = [], [0]
    original_ti, original_predict = pipeline.temporal_inputs, pipeline.Predictor.predict

    def temporal_inputs(frames, window, *a, **k):
        stash.append(window)
        return original_ti(frames, window, *a, **k)

    def predict(self, request):
        try:
            return original_predict(self, request)
        finally:
            rounds[0] += 1

    def generate(model, proc, inputs, budget):
        assert budget == 1024
        req = proc.pop_request()
        text = text_for(rounds[0], stash[-1])
        d = canned.record(req, text)
        if text is None or canned.prompt_tokens.get(d, 0) + budget > 32768:
            raise nv.models.InputErrorForModel("canned failure")
        return text

    with monkeypatch.context() as m:
        m.setattr(nv.media, "decode_selected", fake_decode)
        m.setattr(pipeline, "temporal_inputs", temporal_inputs)
        m.setattr(pipeline.Predictor, "predict", predict)
        m.setattr(pipeline, "generate", generate)
        session = CachedSession(lambda: FakeNvSession(
            {"tempseg": CaptureProcessor(procs["tempseg"], budget=1024)}))
        try:
            return ("ok",) + pipeline.segment(videos, pool, session, instruction, no_progress)
        except Exception as exc:
            return ("error", exc)


def batched(nv, procs, monkeypatch, episodes, canned, kind="tempseg"):
    """Run the balanced stage on ``episodes`` (list of (EpisodeVideos, segments|None)) together
    on one micro-batcher; returns (outcomes, engine)."""
    engine = CannedEngine(canned, procs[kind].tokenizer)
    stage = (S.SegmentStage if kind == "tempseg" else C.CaptionStage)(nv, procs[kind])

    async def one(mb, ep, segments):
        async with mb.episode():
            try:
                if kind == "tempseg":
                    r = await stage.segment(mb, ep)
                    return ("ok", r.rows, r.calls)
                r = await stage.caption(mb, ep, segments)
                return ("ok", r.rows, r.audit)
            except Exception as exc:
                return ("error", exc)

    async def main():
        async with MicroBatcher(engine, max_wait_s=30) as mb:
            return await asyncio.gather(*(one(mb, ep, sg) for ep, sg in episodes))

    with monkeypatch.context() as m:
        m.setattr(nv.media, "decode_selected", fake_decode)
        try:
            return asyncio.run(main()), engine
        finally:
            stage.close()


def episode(videos, pool, profile, instruction=""):
    return S.EpisodeVideos(videos=videos, profile=profile, pool=pool, instruction=instruction)


CANONICAL = {"human": "Human", "yam": "YAM", "yam32": "YAM", "dex": "Dex"}


@pytest.mark.parametrize(("pool", "n_views"), [("human", 1), ("yam", 3), ("yam32", 3)])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_segment_matches_nvidia_sync_path(nv, procs, monkeypatch, pool, n_views, scenario):
    if pool == "yam32" and scenario not in ("clean", "directional_retry"):
        pytest.skip("yam32: two scenarios are enough (long windows, slow synthetic frames)")
    renderer, n_requests, _ = SCENARIOS[scenario]
    L = Layout(nv, pool)
    videos = make_videos(nv, ROLES3[:n_views], L.duration_s)
    canned = Canned()
    ref = reference_segment(nv, procs, monkeypatch, videos, pool,
                            lambda r, w: renderer(L, r, w), canned)
    assert ref[0] == "ok", ref
    (out,), engine = batched(nv, procs, monkeypatch,
                             [(episode(videos, pool, CANONICAL[pool]), None)], canned)
    assert out[0] == "ok", out
    assert out[1] == ref[1]  # segments: spans + NVIDIA's metadata
    assert out[2] == ref[2]  # NVIDIA's temporal audit, outputs included
    assert len(engine.batches) == n_requests  # grid round + NVIDIA's retry rounds
    assert len(engine.batches[0]) == sum(1 for c in ref[2] if c["window"]["window_index"]
                                         is not None) - sum(len(b) for b in engine.batches[1:])


def test_segment_failure_semantics_match(nv, procs, monkeypatch):
    """A window without EOS fails the episode with NVIDIA's InputErrorForModel on both paths;
    so does a prompt over NVIDIA's 32k guard. (NVIDIA stops at the failing window; the batched
    path has already generated the request's other windows, which changes nothing.)"""
    from tests.nvlink.test_nv_link_segmenter import render

    L = Layout(nv, "human")
    videos = make_videos(nv, ("main",), L.duration_s)
    clean = Canned()
    reference_segment(nv, procs, monkeypatch, videos, "human",
                      lambda r, w: render(w, L.events), clean)
    bad = list(clean.table)[1]  # window 1 of the grid round
    for mode in ("no_eos", "budget"):
        ref_canned, canned = Canned(), Canned()
        canned.table = {d: list(v) for d, v in clean.table.items()}
        if mode == "no_eos":
            canned.table[bad] = [None]
        else:
            canned.prompt_tokens[bad] = ref_canned.prompt_tokens[bad] = 32000

        def text_for(r, w, mode=mode):
            return None if (mode == "no_eos" and w.window_index == 1) else render(w, L.events)

        ref = reference_segment(nv, procs, monkeypatch, videos, "human", text_for, ref_canned)
        (out,), _ = batched(nv, procs, monkeypatch, [(episode(videos, "human", "Human"), None)],
                            canned)
        assert ref[0] == out[0] == "error", (mode, ref, out)
        assert isinstance(ref[1], nv.models.InputErrorForModel)
        assert isinstance(out[1], nv.models.InputErrorForModel), out


def test_episodes_share_rounds(nv, procs, monkeypatch):
    """Two episodes (1-view human, 3-view YAM) batched together: each equals its own NVIDIA
    run, and the grid windows of both go into one engine round."""
    renderer = SCENARIOS["recentered_retry"][0]
    eps, refs, canned = [], [], Canned()
    for pool, n in (("human", 1), ("yam", 3)):
        L = Layout(nv, pool)
        videos = make_videos(nv, ROLES3[:n], L.duration_s, tag=f"{pool}-")
        ref = reference_segment(nv, procs, monkeypatch, videos, pool,
                                lambda r, w, L=L: renderer(L, r, w), canned)
        assert ref[0] == "ok"
        refs.append(ref)
        eps.append((episode(videos, pool, CANONICAL[pool]), None))
    outs, engine = batched(nv, procs, monkeypatch, eps, canned)
    for out, ref in zip(outs, refs, strict=True):
        assert out[0] == "ok" and out[1] == ref[1] and out[2] == ref[2]
    grid = sum(len(r[2]) for r in refs) - sum(len(b) for b in engine.batches[1:])
    assert len(engine.batches[0]) == grid
    assert len(engine.batches) == 2  # one shared grid round, one shared retry round


# ---------------------------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------------------------

SEGMENTS = [dict(start_s=2.0, end_s=5.0, metadata={"segmentation_provenance": "fine"}),
            dict(start_s=6.0, end_s=9.5, metadata={"segmentation_provenance":
                                                   "fine_repaired_by_fine_retry"}),
            dict(start_s=11.0, end_s=11.02, metadata={}),
            dict(start_s=12.0, end_s=16.0, metadata={})]


def reference_caption(nv, procs, monkeypatch, videos, profile, canned, *, instruction="",
                      fail_call: int | None = None):
    pipeline = nv.pipeline
    n = [0]

    def generate(model, proc, inputs, budget):
        assert budget == 512
        req = proc.pop_request()
        text = None if n[0] == fail_call else f"caption number {n[0]}: the arm lifts the cup"
        n[0] += 1
        canned.record(req, text)
        if text is None:
            raise nv.models.InputErrorForModel("canned failure")
        return text

    with monkeypatch.context() as m:
        m.setattr(nv.media, "decode_selected", fake_decode)
        m.setattr(pipeline, "generate", generate)
        session = CachedSession(lambda: FakeNvSession(
            {"captioner": CaptureProcessor(procs["captioner"], budget=512)}))
        try:
            if len(videos) == 3:
                rows, audit = pipeline.caption_three(videos, [dict(s) for s in SEGMENTS],
                                                     session, instruction, no_progress,
                                                     profile=profile)
            else:
                rows, audit = pipeline.caption_single(videos, [dict(s) for s in SEGMENTS],
                                                      profile, session, instruction,
                                                      no_progress)
        except Exception as exc:
            return ("error", exc)
    for row, seg in zip(rows, SEGMENTS, strict=True):
        row["segmentation_provenance"] = seg["metadata"].get("segmentation_provenance", "fine")
    return ("ok", rows, audit)


@pytest.mark.parametrize(("profile", "pool", "n_views", "instruction"), [
    ("YAM", "yam", 3, ""),
    ("Dex", "dex", 3, "Fold the towel."),
    ("Human", "human", 1, ""),
    ("Human", "human", 1, "Fold the towel."),
    ("YAM", "yam", 1, ""),
])
def test_caption_matches_nvidia_sync_path(nv, procs, monkeypatch, profile, pool, n_views,
                                          instruction):
    videos = make_videos(nv, ROLES3[:n_views], 20.0)
    canned = Canned()
    ref = reference_caption(nv, procs, monkeypatch, videos, profile, canned,
                            instruction=instruction)
    assert ref[0] == "ok", ref
    assert sum(r["status"] == "complete" for r in ref[1]) >= 2
    (out,), engine = batched(nv, procs, monkeypatch,
                             [(episode(videos, pool, profile, instruction), SEGMENTS)], canned,
                             kind="captioner")
    assert out[0] == "ok", out
    assert out[1] == ref[1]
    assert out[2] == ref[2]
    assert len(engine.batches) == 1  # every caption call of the episode in one round


def test_caption_failure_semantics_match(nv, procs, monkeypatch):
    videos = make_videos(nv, ROLES3, 20.0)
    clean = Canned()
    reference_caption(nv, procs, monkeypatch, videos, "YAM", clean)
    canned = Canned()
    canned.table = {d: list(v) for d, v in clean.table.items()}
    canned.table[list(clean.table)[1]] = [None]  # NVIDIA stops there; batched sees all calls
    ref = reference_caption(nv, procs, monkeypatch, videos, "YAM", Canned(), fail_call=1)
    (out,), _ = batched(nv, procs, monkeypatch, [(episode(videos, "yam", "YAM"), SEGMENTS)],
                        canned, kind="captioner")
    assert ref[0] == out[0] == "error"
    assert isinstance(ref[1], nv.models.InputErrorForModel)
    assert isinstance(out[1], nv.models.InputErrorForModel)


def test_engine_video_kwargs_mirror_nvidias_processor(procs):
    for kind, want in NV_VIDEO_KWARGS.items():
        vp = procs[kind].video_processor
        assert {"fps": vp.fps, "max_frames": vp.max_frames} == dict(want)


def test_tempseg_and_captioner_stop_on_im_end(procs):
    for kind in ("tempseg", "captioner"):
        tok = procs[kind].tokenizer
        assert tok.convert_ids_to_tokens(tok.eos_token_id) == "<|im_end|>"



# ---------------------------------------------------------------------------------------------
# N1 (caption pre-registration, 2026-10-23): the stage-isolated caption certificate's premises,
# on NVIDIA's real code and processor with fakes, at $0.
# (i) the restated digest (stages.segment.nvidia_input_digest) equals the input_tensor_digest
#     NVIDIA's own models.generate logs for the same call;
# (ii) captioning from the rows rebuilt from an oracle segment record (spans and
#     segmentation_provenance only: workers_impl.caption.pinned_rows) gives the same per-call
#     digests as captioning from TempSeg's full-metadata rows.

FULL_ROWS = [dict(start_s=2.0, end_s=5.0, metadata={
                 "segmentation_provenance": "fine", "fine_event_id": "fine_0001",
                 "window_index": 0, "label": "reach", "confidence": 0.9}),
             dict(start_s=6.0, end_s=9.5, metadata={
                 "segmentation_provenance": "fine_repaired_by_fine_retry",
                 "fine_event_id": "fine_0002", "window_index": 1, "label": "lift"}),
             dict(start_s=12.0, end_s=16.0, metadata={"window_index": 2})]


class _DigestModel:
    """What NVIDIA's ``models.generate`` needs of a model: a parameter (its device) and a
    ``generate`` that appends a short answer and EOS."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.p])

    def generate(self, input_ids=None, **kw):
        tail = self.tok.encode("the arm lifts", add_special_tokens=False) + [kw["eos_token_id"]]
        return torch.cat([input_ids, torch.tensor([tail], dtype=input_ids.dtype)], dim=1)


class _ModelSession:
    def __init__(self, model, proc):
        self.model, self.proc, self.audit = model, proc, {}

    def load(self, kind):
        assert kind == "captioner"
        return self.model, self.proc

    def close(self):
        pass


def _caption(nv, videos, rows, session, profile):
    if len(videos) == 3:
        return nv.pipeline.caption_three(videos, [dict(r) for r in rows], session, "",
                                         no_progress, profile=profile)
    return nv.pipeline.caption_single(videos, [dict(r) for r in rows], profile, session, "",
                                      no_progress)


def nvidia_logged_digests(nv, procs, monkeypatch, videos, profile, rows):
    """The input_tensor_digest of every Captioner call, as NVIDIA's own generate logs it."""
    logged = []

    def event(name, **values):
        if name == "generation_start":
            logged.append(values["input_tensor_digest"])

    model = _DigestModel(procs["captioner"].tokenizer)
    with monkeypatch.context() as m:
        m.setattr(nv.media, "decode_selected", fake_decode)
        m.setattr(nv.models, "runtime_event", event)
        _caption(nv, videos, rows, _ModelSession(model, procs["captioner"]), profile)
    return logged


def restated_digests(nv, procs, monkeypatch, videos, profile, rows):
    """The digest the balanced stage computes per call (captured request -> real processor)."""
    out = []

    def generate(model, proc, inputs, budget):
        out.append(S.nvidia_input_digest(procs["captioner"], proc.pop_request()))
        return "the arm lifts"

    with monkeypatch.context() as m:
        m.setattr(nv.media, "decode_selected", fake_decode)
        m.setattr(nv.pipeline, "generate", generate)
        session = CachedSession(lambda: FakeNvSession(
            {"captioner": CaptureProcessor(procs["captioner"], budget=512)}))
        _caption(nv, videos, rows, session, profile)
    return out


@pytest.mark.parametrize(("profile", "n_views"), [("Human", 1), ("YAM", 3)])
def test_n1_restated_digest_and_reduced_rows(nv, procs, monkeypatch, profile, n_views):
    import dataclasses

    from flora_deploy.records import Provenance
    from flora_deploy.workers_impl.caption import pinned_rows

    videos = make_videos(nv, ROLES3[:n_views], 20.0, tag="n1")
    logged = nvidia_logged_digests(nv, procs, monkeypatch, videos, profile, FULL_ROWS)
    assert logged and len(logged) == len(FULL_ROWS)
    assert restated_digests(nv, procs, monkeypatch, videos, profile, FULL_ROWS) == logged  # (i)
    oracle = S.segment_record(FULL_ROWS, uid="u", dataset="ds", episode_id="u",
                              provenance=Provenance(models={}, engine="e", preset="exact",
                                                    config_hash="a" * 64, logic_version="l",
                                                    framework="f"))
    oracle = dataclasses.replace(oracle, segments=tuple(  # oracle records carry no metadata
        dataclasses.replace(s, nv_metadata={}) for s in oracle.segments))
    reduced = pinned_rows(oracle)
    assert all(set(r["metadata"]) <= {"segmentation_provenance", "fine_event_id"}
               for r in reduced)
    assert nvidia_logged_digests(nv, procs, monkeypatch, videos, profile, reduced) == logged
