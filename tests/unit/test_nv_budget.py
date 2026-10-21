"""NVIDIA's 32k context guard, reproduced (``nv.budget``); NVIDIA parity is in nvlink tests."""

from __future__ import annotations

import pickle
import types

import pytest

from flora_deploy.nv import budget, linker
from flora_deploy.nv.budget import CONTEXT_LIMIT, ContextBudgetError, check_context_budget


@pytest.mark.parametrize("b", [512, 1024])
def test_boundary_matches_nvidias_condition(b):
    check_context_budget(CONTEXT_LIMIT - b, b)  # n + budget == limit is allowed
    check_context_budget(0, b)
    with pytest.raises(ContextBudgetError) as info:
        check_context_budget(CONTEXT_LIMIT - b + 1, b)
    err = info.value
    assert (err.prompt_tokens, err.budget, err.limit) == (CONTEXT_LIMIT - b + 1, b, CONTEXT_LIMIT)


def test_custom_limit():
    check_context_budget(90, 10, limit=100)
    with pytest.raises(ContextBudgetError):
        check_context_budget(91, 10, limit=100)


class InputErrorForModel(RuntimeError):  # stands in for NVIDIA's models.InputErrorForModel
    pass


def test_raises_nvidias_error_type_when_given_the_family():
    nv = types.SimpleNamespace(models=types.SimpleNamespace(InputErrorForModel=InputErrorForModel))
    with pytest.raises(InputErrorForModel) as info:
        check_context_budget(CONTEXT_LIMIT, 1, nv=nv)
    assert isinstance(info.value, ContextBudgetError)
    assert budget.error_class(nv) is budget.error_class(nv)


def test_default_class_without_linked_family(monkeypatch):
    monkeypatch.setattr(linker, "_LOADED", None)
    assert budget.error_class() is ContextBudgetError


def test_error_pickles(monkeypatch):
    monkeypatch.setattr(linker, "_LOADED", None)
    with pytest.raises(ContextBudgetError) as info:
        check_context_budget(CONTEXT_LIMIT, 1)
    clone = pickle.loads(pickle.dumps(info.value))
    assert isinstance(clone, ContextBudgetError) and str(clone) == str(info.value)
    assert clone.prompt_tokens == CONTEXT_LIMIT


FAKE_PIPELINE = '''
class Predictor:
    def predict(self, request):
        text = generate(self.model, self.proc, inputs, 1024)

def caption_single(videos):
    caption = generate(model, proc, inputs, 512)

def caption_three(videos):
    caption = generate(model, proc, inputs, 512)
'''

FAKE_MODELS = '''
def generate(model, proc, inputs, budget):
    n = inputs['input_ids'].shape[1]
    if n + budget > 32768:
        raise InputErrorForModel('x')
'''


def _fake_nv(pipeline_src, models_src=FAKE_MODELS):
    def module(name, src):
        # what the nv linker's loader records: the exact bytes it verified and compiled
        mod = types.ModuleType(name)
        loader = types.SimpleNamespace(executed_source=src.encode())
        mod.__spec__ = types.SimpleNamespace(loader=loader)
        return mod

    return types.SimpleNamespace(pipeline=module("p", pipeline_src),
                                 models=module("m", models_src))


@pytest.fixture
def fake_source():
    budget._budgets.cache_clear()
    budget._context_limit.cache_clear()
    yield
    budget._budgets.cache_clear()
    budget._context_limit.cache_clear()


def test_budgets_derived_from_call_sites(fake_source):
    nv = _fake_nv(FAKE_PIPELINE)
    assert budget.nvidia_budgets(nv) == {"tempseg": 1024, "captioner": 512}
    assert budget.budget_for(nv, "captioner") == 512
    assert budget.nvidia_context_limit(nv) == 32768


@pytest.mark.parametrize("src, match", [
    (FAKE_PIPELINE.replace("inputs, 512)\n\ndef caption_three",
                           "inputs, 256)\n\ndef caption_three"), "disagree"),
    (FAKE_PIPELINE.replace("1024", "BUDGET"), "not a literal"),
    (FAKE_PIPELINE + "\ndef other():\n    generate(a, b, c, 1)\n", "unexpected"),
    (FAKE_PIPELINE.replace("caption = generate(model, proc, inputs, 512)", "pass", 1),
     "call sites changed"),
])
def test_budget_derivation_fails_loudly(fake_source, src, match):
    with pytest.raises(ValueError, match=match):
        budget.nvidia_budgets(_fake_nv(src))


IMPORTS = "from .models import Session, generate, runtime_event\nfrom . import models\n"


@pytest.mark.parametrize("extra", [
    "def long_caption(*a):\n    return models.generate(m, p, i, 2048)\n",  # attribute call
    "async def acaption(*a):\n    return generate(m, p, i, 4096)\n",  # async def site
    "g = generate\ndef other(*a):\n    return g(m, p, i, 8192)\n",  # alias
    "WARM = generate(m, p, i, 30000)\n",  # module-level call
    "def f(*a):\n    return getattr(models, 'generate')(m, p, i, 1)\n",  # by name
    "from .models import generate as gen\n",  # re-import under another name
    "def f(generate):\n    return generate(m, p, i, 1)\n",  # shadowing parameter
    "class Other:\n    async def run(self):\n        return generate(m, p, i, 1)\n",
])
def test_budget_derivation_fails_closed_on_new_shapes(fake_source, extra):
    """Review L1 (p4): shapes like these used to return {512, 1024} silently."""
    with pytest.raises(ValueError, match="unexpected"):
        budget.nvidia_budgets(_fake_nv(IMPORTS + FAKE_PIPELINE + "\n" + extra))


def test_async_known_site_is_fine(fake_source):
    src = IMPORTS + FAKE_PIPELINE.replace("def caption_three(videos):",
                                          "async def caption_three(videos):")
    assert budget.nvidia_budgets(_fake_nv(src)) == {"tempseg": 1024, "captioner": 512}


def test_refuses_modules_not_imported_by_the_linker():
    nv = types.SimpleNamespace(pipeline=types.ModuleType("p"), models=types.ModuleType("m"))
    with pytest.raises(ValueError, match="not imported by the nv linker"):
        budget.nvidia_budgets(nv)
    with pytest.raises(ValueError, match="not imported by the nv linker"):
        budget.nvidia_context_limit(nv)


def test_limit_derivation_fails_loudly(fake_source):
    with pytest.raises(ValueError, match="guard not found"):
        budget.nvidia_context_limit(_fake_nv(FAKE_PIPELINE, FAKE_MODELS.replace("> 32768", "")))
