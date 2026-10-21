"""Both NVIDIA segmenter constructions: apredict vs predict parity, retries, failure semantics.

The canned predictor plays a model that "sees" a fixed set of ground-truth events (in global
20 Hz ticks) and renders, for each requested window, exactly the JSON NVIDIA's parser accepts
(``fine_temporal.parse_prediction_with_diagnostics``): ``{"segments": [{s, e, lo, ro}, ...]}`` in
window-local ticks, clipped to the valid envelope with ``lo``/``ro`` marking censored ends, only
segments with emit-core support, in canonical order. Scenarios hide one event from one window
(or return empty / invalid output) to drive NVIDIA's adaptive retry and failure paths.
"""

from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from pathlib import Path

import pytest

from flora_deploy.nv import Family
from flora_deploy.nv import segment_glue as glue
from flora_deploy.nv.linker import load_family

pytestmark = pytest.mark.nvlink


class Layout:
    """Synthetic episode geometry for one window profile (C = core length in 20 Hz ticks)."""

    def __init__(self, nv, pool):
        profile = nv.production.fine_window_profile(nv.production.PROFILES[pool])
        c = profile.core_ticks
        guard = profile.right_guard_us // 50_000
        self.duration_s = (2 * c + 120) / 20
        # starts 1 s before the core boundary between windows 0 and 1 and ends 1 s after
        # window 0's valid end, so window 0 sees it censored (ro=true) and window 1 sees it whole
        self.seam_event = (c - 20, c + guard + 20)
        self.events = [(40, 80), self.seam_event, (c + 120, c + 160), (2 * c + 40, 2 * c + 80)]
        self.window0_valid_end_s = (c + guard) / 20
        # 24 short disjoint events inside window 1's core: above production's hard cap (23),
        # within the 3-view default policy (32)
        self.dense_events = [(c + 22 + 8 * i, c + 26 + 8 * i) for i in range(24)]


@pytest.fixture
def nv(nv_code):
    return load_family(Family.SEGCAP, nv_code("tempseg_code"))


def render(window, events, omit=()) -> str:
    off = window.global_tick_offset
    rows = []
    for gs, ge in events:
        if (gs, ge) in omit:
            continue
        s, e = gs - off, ge - off
        lo, ro = s < window.valid_start_tick, e > window.valid_end_tick
        s, e = max(s, window.valid_start_tick), min(e, window.valid_end_tick)
        if s >= e:
            continue
        cs, ce = window.core_start_tick, window.core_end_tick
        if not ((s < ce and e > cs) or e == cs or s == ce):
            continue
        rows.append(dict(s=s, e=e, lo=lo, ro=ro))
    rows.sort(key=lambda r: (r["s"], r["e"], r["lo"], r["ro"]))
    return json.dumps({"segments": rows})


def _hide_seam_first_round(L, r, w):
    return (L.seam_event,) if r == 0 and w.window_index == 1 else ()


SCENARIOS = {
    # name: (renderer(layout, round, window) -> json, expected requests, seam-event provenance)
    "clean": (lambda L, r, w: render(w, L.events), 1, "fine"),
    # window 1 misses the seam event -> unresolved component -> recentered retry repairs it
    "recentered_retry": (
        lambda L, r, w: render(w, L.events, omit=_hide_seam_first_round(L, r, w)),
        2,
        "fine_repaired_by_fine_retry",
    ),
    # the recentered retry sees other events but not this one -> directional retry repairs it
    "directional_retry": (
        lambda L, r, w: render(
            w, L.events, omit=(L.seam_event,) if r == 1 else _hide_seam_first_round(L, r, w)
        ),
        3,
        "fine_repaired_by_fine_retry",
    ),
    # retries return nothing -> NVIDIA's conservative completion of the unresolved component
    "conservative_completion": (
        lambda L, r, w: (
            render(w, L.events, omit=_hide_seam_first_round(L, r, w))
            if r == 0
            else '{"segments":[]}'
        ),
        2,
        "fine_conservative_completion",
    ),
}


class _Canned:
    def __init__(self, nv, source, binding, renderer):
        self.a = nv.production
        self.source, self.binding, self.renderer = source, binding, renderer
        self.requests: list[dict] = []
        self.calls: list = []  # pipeline.segment() returns predictor.calls

    def _respond(self, request):
        from flora_runtime.video_segmentation.fine_subtask import FINE_RESPONSE_SCHEMA

        if request["source"]["source_id"] != self.source.source_id:
            raise RuntimeError("source changed")
        if request["model_binding"] != self.binding:
            raise RuntimeError("binding changed")
        round_index = len(self.requests)
        self.requests.append(request)
        predictions = []
        for payload in request["windows"]:
            window = self.a.window_from_payload(payload, self.source)
            predictions.append(
                dict(scope_id=window.scope_id, prediction_json=self.renderer(round_index, window))
            )
        return dict(
            schema_version=FINE_RESPONSE_SCHEMA,
            source_id=self.source.source_id,
            model_io_contract=self.binding["model_io_contract"],
            model_binding=self.binding,
            predictions=predictions,
        )


class SyncPredictor(_Canned):
    def predict(self, request):
        return self._respond(request)


class AsyncPredictor(_Canned):
    def __init__(self, *args):
        super().__init__(*args)
        self.awaited = 0

    async def apredict(self, request):
        await asyncio.sleep(0)  # really yield to the loop, as a micro-batcher would
        self.awaited += 1
        return self._respond(request)

    def predict(self, request):  # NVIDIA must prefer apredict
        raise AssertionError("sync predict called although apredict exists")


def _source(nv, duration_s):
    return glue.annotation_source(
        nv, source_id=glue.source_id_for("ab" * 32), location="synthetic.mp4",
        duration_s=duration_s, fps=30.0,
    )


def _rows(segments):
    return [dict(start_s=s.span.start_s, end_s=s.span.end_s, metadata=dict(s.metadata))
            for s in segments]


def _run(nv, cls, pool, n_views, renderer):
    """Run one episode; return (outcome, predictor). outcome is ("ok", rows) or
    ("error", exception type, message, violation codes)."""
    L = Layout(nv, pool)
    source = _source(nv, L.duration_s)
    binding = glue.model_binding(nv, pool)
    predictor = cls(nv, source, binding, lambda r, w: renderer(L, r, w))
    engine = glue.segmenter_for(nv, predictor, binding, pool, n_views)
    try:
        result = ("ok", _rows(asyncio.run(engine.segment_async(source))))
    except Exception as exc:  # compared across predictors by the tests
        codes = [v.get("code") for v in getattr(exc, "diagnostics", {}).get("violations", [])]
        result = ("error", type(exc).__name__, str(exc), codes)
    return result, predictor


CONSTRUCTIONS = [("human", 1), ("yam", 1), ("yam", 3), ("yam32", 3)]


@pytest.mark.parametrize(("pool", "n_views"), CONSTRUCTIONS)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_apredict_matches_predict(nv, pool, n_views, scenario):
    renderer, n_requests, provenance = SCENARIOS[scenario]
    sync_result, sync_pred = _run(nv, SyncPredictor, pool, n_views, renderer)
    async_result, async_pred = _run(nv, AsyncPredictor, pool, n_views, renderer)
    assert sync_result == async_result
    assert sync_result[0] == "ok", sync_result
    assert len(sync_pred.requests) == len(async_pred.requests) == async_pred.awaited == n_requests
    assert [r["windows"] for r in sync_pred.requests] == [r["windows"] for r in async_pred.requests]
    L = Layout(nv, pool)
    rows = sync_result[1]
    seam = [r for r in rows if r["start_s"] == L.seam_event[0] / 20]
    assert len(seam) == 1
    assert seam[0]["metadata"]["segmentation_provenance"] == provenance
    others = [r for r in rows if r is not seam[0]]
    assert len(others) == len(L.events) - 1
    assert {r["metadata"]["segmentation_provenance"] for r in others} == {"fine"}
    if scenario == "conservative_completion":
        # observation-supported extent: closed start .. censored end at window 0's valid end
        assert seam[0]["end_s"] == L.window0_valid_end_s
        assert seam[0]["metadata"]["review_required"] is True
    else:
        assert seam[0]["end_s"] == L.seam_event[1] / 20


@pytest.mark.parametrize(("pool", "n_views"), CONSTRUCTIONS)
def test_retry_round_windows(nv, pool, n_views):
    """Rounds 2 and 3 carry one window each, from NVIDIA's retry scope namespaces."""
    _, pred = _run(nv, AsyncPredictor, pool, n_views, SCENARIOS["directional_retry"][0])
    scopes = [[w["scope_id"] for w in r["windows"]] for r in pred.requests]
    assert len(scopes[0]) >= 2 and all(":fine-window:" in s for s in scopes[0])
    assert len(scopes[1]) == 1 and ":fine-retry:" in scopes[1][0]
    assert len(scopes[2]) == 1 and ":fine-retry-directional:" in scopes[2][0]


@pytest.mark.parametrize(("pool", "n_views"), CONSTRUCTIONS)
def test_parse_error_fails_episode(nv, pool, n_views):
    """Invalid JSON in one window is not eligible for repair or completion: the episode fails
    after a single request, identically for both predictor kinds."""

    def renderer(L, r, w):
        return "not json" if w.window_index == 1 else render(w, L.events)

    sync_result, sync_pred = _run(nv, SyncPredictor, pool, n_views, renderer)
    async_result, async_pred = _run(nv, AsyncPredictor, pool, n_views, renderer)
    assert sync_result == async_result
    assert sync_result[:2] == ("error", "FineSegmentationOutputError")
    assert len(sync_pred.requests) == len(async_pred.requests) == 1


@pytest.mark.parametrize(("pool", "n_views", "fails"), [
    ("human", 1, True), ("yam", 1, True), ("yam", 3, False),
])
def test_quality_policies_differ(nv, pool, n_views, fails):
    """24 segments in one window break production's hard cap (23) but not the 3-view policy;
    a hard-cap violation is ineligible for repair, so the episode fails after one request."""

    def renderer(L, r, w):
        return render(w, L.events[:1] + L.dense_events)

    sync_result, _ = _run(nv, SyncPredictor, pool, n_views, renderer)
    async_result, async_pred = _run(nv, AsyncPredictor, pool, n_views, renderer)
    assert sync_result == async_result
    assert len(async_pred.requests) == 1
    if fails:
        assert sync_result[:2] == ("error", "FineSegmentationOutputError")
        assert sync_result[3] == ["hard_segment_count"]
    else:
        assert sync_result[0] == "ok" and len(sync_result[1]) == 25


@pytest.mark.parametrize(("pool", "n_views"), [("human", 1), ("yam", 1), ("yam", 3), ("dex", 3)])
def test_glue_matches_nvidia_pipeline_segment(nv, monkeypatch, pool, n_views):
    """Our binding, source and constructions equal what NVIDIA's own ``pipeline.segment()``
    builds. Spies on NVIDIA's own classes record the arguments NVIDIA passes to them; we build
    ours purely from ``segment_glue`` under the same spies and compare argument by argument (the
    predictor aside), then run both engines on a 3-round retry scenario."""
    a = nv.production
    L = Layout(nv, pool)
    renderer = SCENARIOS["directional_retry"][0]
    main = nv.media.Video(
        path=Path("synthetic.mp4"), sha256="ab" * 32, rate=Fraction(30),
        pts_us=tuple(Fraction(i * 1_000_000, 30) for i in range(int(L.duration_s * 30))),
        duration_us=int(L.duration_s * 1_000_000), size=(640, 480), cfr=True,
    )
    videos = {role: main for role in ["main", "left_wrist", "right_wrist"][:n_views]}
    fine_cls, hier_cls = a.FineSubtaskSegmenter, a.HierarchicalSubtaskSegmenter

    def norm(value):
        if isinstance(value, _Canned):
            return "<predictor>"
        if isinstance(value, fine_cls):
            return "<fine>"
        if type(value).__name__ == "NoCoarse":
            return "<NoCoarse>"
        return value

    def spying(calls):
        def spy_for(cls):
            def spy(*args, **kwargs):
                calls.append((cls.__name__, [norm(v) for v in args],
                              {k: norm(v) for k, v in kwargs.items()}))
                return cls(*args, **kwargs)
            return spy
        monkeypatch.setattr(a, "FineSubtaskSegmenter", spy_for(fine_cls))
        monkeypatch.setattr(a, "HierarchicalSubtaskSegmenter", spy_for(hier_cls))

    nv_calls, nv_seen = [], []

    def predictor_factory(videos, source, pool_, bound, model, proc, instruction, progress):
        nv_seen.append((source, bound))
        return SyncPredictor(nv, source, bound, lambda r, w: renderer(L, r, w))

    class FakeSession:
        def load(self, kind):
            assert kind == "tempseg"
            return None, None

    spying(nv_calls)
    monkeypatch.setattr(nv.pipeline, "Predictor", predictor_factory)
    nv_segments, _ = nv.pipeline.segment(videos, pool, FakeSession(), "", lambda *a, **k: None)
    monkeypatch.undo()
    ((nv_source, nv_bound),) = nv_seen

    # binding and source: ours from the glue alone
    binding = glue.model_binding(nv, pool)
    assert binding == nv_bound
    source = glue.annotation_source(
        nv, source_id=glue.source_id_for(main.sha256), location=str(main.path),
        duration_s=main.duration_s, fps=float(main.rate),
    )
    assert source == nv_source

    our_calls = []
    spying(our_calls)
    predictor = SyncPredictor(nv, source, binding, lambda r, w: renderer(L, r, w))
    engine = glue.segmenter_for(nv, predictor, binding, pool, n_views)
    monkeypatch.undo()
    assert [c[0] for c in nv_calls] == ["FineSubtaskSegmenter", "HierarchicalSubtaskSegmenter"]
    assert our_calls == nv_calls

    rows = _rows(asyncio.run(engine.segment_async(source)))
    assert rows == nv_segments
    assert len(predictor.requests) == 3


class _NoEOS(RuntimeError):
    pass


@pytest.mark.parametrize("error", ["nvidia_input_error", "context_budget", "other"])
@pytest.mark.parametrize(("pool", "n_views"), [("human", 1), ("yam", 3)])
def test_predictor_error_mid_retry_propagates(nv, pool, n_views, error):
    """L3: an exception from the predictor in retry round 1 (e.g. NVIDIA's InputErrorForModel for
    a missing EOS, or our 32k guard) propagates out of ``segment_async`` unchanged, for predict
    and apredict alike: NVIDIA repairs only ``FineSegmentationOutputError``, so there is no
    repair and no conservative completion."""
    from flora_deploy.nv.budget import check_context_budget

    def raise_it():
        if error == "nvidia_input_error":
            raise nv.models.InputErrorForModel("no complete output")
        if error == "context_budget":
            check_context_budget(40_000, 1024, nv=nv)
        raise _NoEOS("no complete output")

    def renderer(L, r, w):
        if r == 1:
            raise_it()
        return SCENARIOS["recentered_retry"][0](L, r, w)

    L = Layout(nv, pool)
    for cls in (SyncPredictor, AsyncPredictor):
        source = _source(nv, L.duration_s)
        binding = glue.model_binding(nv, pool)
        predictor = cls(nv, source, binding, lambda r, w: renderer(L, r, w))
        engine = glue.segmenter_for(nv, predictor, binding, pool, n_views)
        with pytest.raises(Exception) as info:
            asyncio.run(engine.segment_async(source))
        assert len(predictor.requests) == 2
        if error == "other":
            assert type(info.value) is _NoEOS
        else:
            assert isinstance(info.value, nv.models.InputErrorForModel)
        if error == "context_budget":
            assert info.value.prompt_tokens == 40_000


def test_concurrent_episodes_on_one_loop(nv):
    """L4 (D14): several episodes awaiting ``apredict`` on one event loop give the same segments
    as running them one by one."""
    cases = [("yam", 3, "directional_retry", "ab" * 32),
             ("human", 1, "conservative_completion", "cd" * 32),
             ("yam32", 3, "recentered_retry", "ef" * 32),
             ("yam", 1, "clean", "01" * 32)]

    def build(pool, n_views, scenario, sha):
        L = Layout(nv, pool)
        source = glue.annotation_source(nv, source_id=glue.source_id_for(sha), location="s.mp4",
                                        duration_s=L.duration_s, fps=30.0)
        binding = glue.model_binding(nv, pool)
        predictor = AsyncPredictor(nv, source, binding,
                                   lambda r, w: SCENARIOS[scenario][0](L, r, w))
        return glue.segmenter_for(nv, predictor, binding, pool, n_views), source, predictor

    serial = []
    for case in cases:
        engine, source, _ = build(*case)
        serial.append(_rows(asyncio.run(engine.segment_async(source))))

    async def together():
        built = [build(*case) for case in cases]
        results = await asyncio.gather(*(e.segment_async(s) for e, s, _ in built))
        return results, [p for _, _, p in built]

    results, predictors = asyncio.run(together())
    assert [_rows(r) for r in results] == serial
    assert [p.awaited for p in predictors] == [SCENARIOS[c[2]][1] for c in cases]
