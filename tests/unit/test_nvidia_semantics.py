"""Review 8a R1 / R6: our certified re-implementations raise NVIDIA's rejections as
``nv.errors.NvidiaSemanticsError`` (NVIDIA's verdict, ``nv_failure``) with NVIDIA's exact type
name and message; and an error classifies the same after crossing a process boundary."""

from __future__ import annotations

import ast
import multiprocessing as mp
import pickle
from pathlib import Path

import pytest

import flora_deploy
from flora_deploy.nv import errors as E
from flora_deploy.stages.reference import NvidiaError, transport_error
from tests.fakes import xproc_errors as X

SRC = Path(flora_deploy.__file__).parent

# Messages of NVIDIA's own builtin-type rejections (batch_utils, all three pins). A flora module
# that raises one of these re-implements an NVIDIA rejection and must raise the marker class.
NVIDIA_MESSAGES = (
    "No common caption input geometry fits",  # vendor/robot_caption_geometry.py:41
    "The pipeline returned an invalid caption interval.",  # api.py:55
    "Unowned frame path",  # vendor/flora_unified/unified_input.py:63
    "PNG changed",  # :66
    "Frame dimensions changed",  # :71
    "unexpected terminal answer",  # Rewarder core.py
)
BUILTINS = {"RuntimeError", "ValueError", "AssertionError", "KeyError", "IndexError",
            "TypeError", "LookupError", "Exception"}
# modules that replace NVIDIA functions statement for statement: NVIDIA's asserts in them are
# NVIDIA's verdict, so a bare ``assert`` (ours, ``python``) must not stand in for one
NVIDIA_REPLACEMENTS = ("accel/geometry.py", "accel/pngmem.py")


def _first_str(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and \
            isinstance(node.values[0], ast.Constant):
        return str(node.values[0].value)
    return None


def test_no_builtin_raise_of_an_nvidia_message():
    bad = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                    and node.exc.args):
                continue
            msg = _first_str(node.exc.args[0])
            func = node.exc.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if msg and any(msg.startswith(m) for m in NVIDIA_MESSAGES) and name in BUILTINS:
                bad.append(f"{path.relative_to(SRC)}:{node.lineno}: {name}({msg!r})")
    assert not bad, "raise nv.errors.Nvidia<Builtin> for NVIDIA's rejections:\n" + "\n".join(bad)


def test_no_bare_assert_in_nvidia_replacements():
    bad = []
    for rel in NVIDIA_REPLACEMENTS:
        tree = ast.parse((SRC / rel).read_text())
        bad += [f"{rel}:{n.lineno}" for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    assert not bad, bad


def test_marker_classes_keep_nvidia_type_and_message():
    for marker, base in ((E.NvidiaRuntimeError, RuntimeError), (E.NvidiaValueError, ValueError),
                         (E.NvidiaAssertionError, AssertionError), (E.NvidiaKeyError, KeyError),
                         (E.NvidiaIndexError, IndexError)):
        exc = marker("msg")
        assert isinstance(exc, base) and isinstance(exc, E.NvidiaSemanticsError)
        assert type(exc).__name__ == base.__name__  # f"{type(e).__name__}: {e}" unchanged
        assert str(exc) == str(base("msg"))
        assert E.nvidia_type_name(exc) == f"builtins.{base.__name__}"
        back = pickle.loads(pickle.dumps(exc))
        assert type(back) is marker and str(back) == str(exc)
        assert E.classify(exc) == "nv_failure"
    k = E.as_nvidia(KeyError(3))
    assert type(k) is E.NvidiaKeyError and str(k) == str(KeyError(3))
    other = OSError("x")
    assert E.as_nvidia(other) is other


def test_geometry_pool_rejection_is_nvidias_verdict():
    """R1: the default-on ``geometry_pool`` patch raises NVIDIA's ``RuntimeError`` from our
    frame; it is NVIDIA's verdict, recorded exactly as NVIDIA's own error is."""
    exc = X._geometry()
    assert type(exc).__name__ == "RuntimeError" and isinstance(exc, RuntimeError)
    assert str(exc) == "No common caption input geometry fits"
    assert E.origin(exc) == "ours"
    assert E.classify(exc) == "nv_failure"
    t = transport_error(exc)
    assert f"{t.type_name}: {t.message}" == \
        "builtins.RuntimeError: No common caption input geometry fits"
    assert E.classify(t) == "nv_failure"
    assert E.classify(pickle.loads(pickle.dumps(t))) == "nv_failure"
    from flora_deploy.workers_impl.segcap import classify_nvidia_error

    assert classify_nvidia_error(t)[0] == "nv_failure"
    miss = X._missing_frame()
    assert type(miss).__name__ == "KeyError" and E.classify(miss) == "nv_failure"


def test_local_job_excuses_a_geometry_pool_rejection(tmp_path, monkeypatch):
    """Review 8a p3b end to end: one episode raises from the real ``accel.geometry``
    replacement. It is excused after ONE run (it used to run 12x and fail the job)."""
    import shutil

    from flora_deploy.local import engines as LE
    from flora_deploy.local import run_local, sample_dataset

    template = SRC / "config" / "templates" / "local_demo.yaml"
    runs = {"n": 0}
    orig = LE.FakeSegcapEngine.result_for

    def result_for(self, item):
        if item["episode"].endswith("--ep003"):
            runs["n"] += 1
            raise X._geometry()
        return orig(self, item)

    monkeypatch.setattr(LE.FakeSegcapEngine, "result_for", result_for)
    shutil.copy(template, tmp_path / "flora.yaml")
    sample_dataset(tmp_path / "videos")
    res = run_local(tmp_path / "flora.yaml", store=tmp_path / "store")
    assert res.state == "completed_with_errors", res.state
    assert runs["n"] == 1
    assert not [e for e in res.events if e["event"] == "shard_failed"]


async def test_balanced_caption_interval_rejection_is_nvidias_verdict(monkeypatch):
    """R1 (balanced): ``CaptionStage`` re-checks NVIDIA's caption intervals (api.py:55)."""
    from flora_deploy.stages import caption as C
    from flora_deploy.workers_impl.segment import _classify

    stage = C.CaptionStage.__new__(C.CaptionStage)
    stage.nv, stage.processor, stage.tokenizer, stage.eos = None, None, None, None

    async def fake_tapped(*a, **k):
        return ([{"start_s": 2.0, "end_s": 1.0, "caption": "x"}], []), {}

    class Ep:
        duration_s = 5.0

    monkeypatch.setattr(C, "run_tapped", fake_tapped)
    monkeypatch.setattr(C, "substitute", lambda text, texts: text)
    monkeypatch.setattr(C.CaptureProcessor, "for_kind", staticmethod(lambda *a, **k: None))
    with pytest.raises(RuntimeError) as info:
        await stage.caption(None, Ep(), [{"metadata": {}}])
    exc = info.value
    assert str(exc) == "The pipeline returned an invalid caption interval."
    assert type(exc).__name__ == "RuntimeError"
    assert E.classify(exc) == "nv_failure" and _classify(exc) == "nv_failure"


def test_classification_survives_a_real_process_boundary():
    """R6: the same error classifies the same in the child (in-process, with its traceback),
    in this process after a real ``multiprocessing`` round trip, and when built here."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        child = pool.apply(X.produce)
    assert set(child) == set(X.cases())
    for name, make in X.cases().items():
        local = E.classify(make())
        in_child, blob = child[name]
        crossed = pickle.loads(blob)
        assert isinstance(crossed, NvidiaError) and crossed.__cause__ is None
        assert (in_child, E.classify(crossed)) == (local, local), name
    expect = {"nv_class": "nv_failure", "nv_assert": "nv_failure", "nv_memory": "infra",
              "nv_attribute": "infra", "nv_name": "infra", "geometry": "nv_failure",
              "missing_frame": "nv_failure", "ours_value": "python",
              "ours_runtime": "python", "oserror": "infra"}
    assert {n: c[0] for n, c in child.items()} == expect


def test_string_only_transport_keeps_the_fallback():
    # reference_app builds NvidiaError from strings (no kind, no origin): message fallback
    assert E.classify(NvidiaError("builtins.RuntimeError",
                                  "No common caption input geometry fits")) == "nv_failure"
    assert E.classify(NvidiaError("builtins.ValueError", "x", "ours")) == "python"
    # an unknown kind string is ignored, not trusted
    assert E.classify(NvidiaError("builtins.ValueError", "x", "ours", "bogus")) == "python"
