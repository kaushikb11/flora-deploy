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


def test_runtime_limit_is_the_parsed_one(nv):  # noqa: F811
    check_context_budget(CONTEXT_LIMIT - 512, 512, nv=nv)  # limit parsed from verified bytes
    with pytest.raises(nv.models.InputErrorForModel):
        check_context_budget(CONTEXT_LIMIT - 511, 512, nv=nv)


ISOLATED = r"""
import importlib, importlib.abc, importlib.util, shutil, sys, tempfile
from pathlib import Path
from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.nv import Family, budget, linker

src = Path(sys.argv[1])
code = Path(tempfile.mkdtemp()) / src.name
shutil.copytree(src / "batch_utils", code / "batch_utils",
                ignore=shutil.ignore_patterns("__pycache__"))
nv = linker.load_family(Family.SEGCAP, code,
                        pinned_digests=dict(PINS_BY_KEY["tempseg_code"].code_digests))
assert budget.nvidia_budgets(nv) == {"tempseg": 1024, "captioner": 512}

# 1. budgets come from the verified bytes that run, not from the file on disk
p = code / "batch_utils" / "pipeline.py"
p.write_text(p.read_text().replace("generate(model, proc, inputs, 512)",
                                   "generate(model, proc, inputs, 9999)"))
budget._budgets.cache_clear()
assert budget.nvidia_budgets(nv) == {"tempseg": 1024, "captioner": 512}
print("BUDGETS_FROM_EXECUTED_BYTES")

# 2. a finder inserted ahead of ours after linking: audit() puts ours back in front, so
#    NVIDIA's lazy `from plans import ...` resolves to the verified vendor file again
evil = Path(tempfile.mkdtemp()) / "plans.py"
evil.write_text("def plans_for_episode(ep, clocks):\n    return ['EVIL']\n")

class Foreign(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "plans":
            return importlib.util.spec_from_file_location("plans", evil)

sys.meta_path.insert(0, Foreign())
sys.modules.pop("plans", None)
nv.audit()
mod = importlib.import_module("plans")
assert Path(mod.__file__).resolve().is_relative_to(code.resolve()), mod.__file__
assert "plans" in nv.audit() and nv.audit_report()["finder_reasserted"] == 1
print("FINDER_REASSERTED")

# 3. a data file changed after linking is reported by audit()
m = code / "batch_utils" / "models.json"
m.write_text(m.read_text() + " ")
try:
    nv.audit()
except linker.CodeManifestError as e:
    print("DATA_CHANGE_CAUGHT", "models.json" in str(e))
"""


def test_linker_gaps_closed_on_a_scratch_copy(nv_dirs):  # noqa: F811
    """Review L1/L2 (p7), in a fresh interpreter on a scratch copy of NVIDIA's tree."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", ISOLATED, str(nv_dirs["tempseg_code"])],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.split("\n")
    assert "BUDGETS_FROM_EXECUTED_BYTES" in lines
    assert "FINDER_REASSERTED" in lines
    assert "DATA_CHANGE_CAUGHT True" in lines
