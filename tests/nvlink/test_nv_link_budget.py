"""NVIDIA's 32k context guard: budgets from NVIDIA's call sites, parity with ``models.generate``.

1. The budgets ``nv.budget`` derives from ``pipeline.py`` (and the limit from ``models.py``) are
   the ones NVIDIA's pipeline really passes to ``generate`` for each captured request.
2. With real HF processor token counts (``replay`` of captured requests, the same length vLLM's
   ``prompt_token_ids`` have), ``check_context_budget`` raises exactly when NVIDIA's own
   ``models.generate`` raises ``InputErrorForModel``, and the error is NVIDIA's type.
"""

from __future__ import annotations

import pytest
from test_nv_link_capture import (  # noqa: F401  (module-scoped fixtures)
    ROLES3,
    FakeNvSession,
    make_videos,
    nv,
    nv_dirs,
    procs,
    synthetic_frame,
)

from flora_deploy.nv.budget import (
    CONTEXT_LIMIT,
    ContextBudgetError,
    check_context_budget,
    nvidia_budgets,
    nvidia_context_limit,
)
from flora_deploy.nv.capture import CaptureProcessor, replay
from flora_deploy.nv.session import CachedSession

pytestmark = pytest.mark.nvlink


def test_derived_budgets_and_limit(nv):  # noqa: F811
    assert nvidia_budgets(nv) == {"tempseg": 1024, "captioner": 512}
    assert nvidia_context_limit(nv) == CONTEXT_LIMIT == 32768


class _PastTheGuard(Exception):
    pass


class _Model:
    def parameters(self):
        raise _PastTheGuard  # NVIDIA's generate got past its context check


def nvidia_raises(nv, proc, inputs, budget) -> bool:  # noqa: F811
    try:
        nv.models.generate(_Model(), proc, inputs, budget)
    except nv.models.InputErrorForModel:
        return True
    except _PastTheGuard:
        return False
    raise AssertionError("unreachable")


CASES = [
    ("segment", "yam", 3),
    ("segment", "human", 1),
    ("caption_three", "YAM", 3),
    ("caption_single", "Human", 1),
]


@pytest.mark.parametrize(("path", "profile", "n_views"), CASES)
def test_captured_budget_and_guard_parity(nv, procs, monkeypatch, path, profile, n_views):  # noqa: F811
    kind = "tempseg" if path == "segment" else "captioner"
    caps = {k: CaptureProcessor.for_kind(nv, p, k) for k, p in procs.items()}
    passed = []

    def fake_generate(model, proc, inputs, budget):
        passed.append(budget)
        return '{"segments":[]}' if budget == 1024 else "pick up the cup"

    monkeypatch.setattr(nv.media, "decode_selected",
                        lambda video, wanted, canonical=False: {
                            i: synthetic_frame(video.path.stem, i) for i in wanted})
    monkeypatch.setattr(nv.pipeline, "generate", fake_generate)
    session = CachedSession(lambda: FakeNvSession(caps))
    videos = make_videos(nv, ROLES3[:n_views])
    segments = [dict(start_s=2.0, end_s=5.0, metadata={})]
    progress = lambda *a, **k: None  # noqa: E731
    if path == "segment":
        nv.pipeline.segment(videos, profile, session, "", progress)
    elif path == "caption_three":
        nv.pipeline.caption_three(videos, segments, session, "", progress, profile=profile)
    else:
        nv.pipeline.caption_single(videos, segments, profile, session, "", progress)
    monkeypatch.undo()

    requests = caps[kind].requests
    assert requests and len(passed) == len(requests)
    assert [r.context_budget for r in requests] == passed  # what NVIDIA passed, per request
    req = requests[0]
    out = replay(procs[kind], req)
    n = out["input_ids"].shape[1]
    req.check_context(n, nv=nv)  # real prompts are far from the limit
    for budget in (CONTEXT_LIMIT - n - 1, CONTEXT_LIMIT - n, CONTEXT_LIMIT - n + 1,
                   req.context_budget):
        ours = True
        try:
            check_context_budget(n, budget, nv=nv)
        except ContextBudgetError as exc:
            assert isinstance(exc, nv.models.InputErrorForModel)
        else:
            ours = False
        assert ours == nvidia_raises(nv, procs[kind], out, budget), budget
    assert nvidia_raises(nv, procs[kind], out, CONTEXT_LIMIT - n + 1)
