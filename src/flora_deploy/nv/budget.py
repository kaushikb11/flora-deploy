"""NVIDIA's 32k context guard, reproduced for vLLM requests (PLAN.md §6.3).

NVIDIA's ``models.generate(model, proc, inputs, budget)`` refuses a request before generating
when ``input_ids.shape[1] + budget > 32768`` and raises ``models.InputErrorForModel``
(``batch_utils/models.py:168-170`` at nvidia/Flora-TempSeg@78a0e4ac). ``budget`` is the
``max_new_tokens`` of the call. vLLM's ``max_model_len=32768`` only rejects prompts longer than
32768 tokens, so a prompt that leaves fewer than ``budget`` tokens of room would run on vLLM
where NVIDIA errors. ``check_context_budget`` applies NVIDIA's condition to the length of vLLM's
``prompt_token_ids``, which equals HF's ``input_ids`` length for a captured request (Gate C).

The budgets are not guessed. ``nvidia_budgets(nv)`` reads them from NVIDIA's call sites with
``ast``: ``generate(..., 1024)`` in ``pipeline.Predictor.predict`` (TempSeg) and
``generate(..., 512)`` in ``pipeline.caption_single`` / ``pipeline.caption_three`` (Captioner).
``nvidia_context_limit(nv)`` reads the limit from the comparison in ``models.generate``.

LICENSE REVIEW FLAG (PLAN.md §11): ``CONTEXT_LIMIT = 32768`` restates the constant of
``models.py:169``. It is a default only. The nvlink test ``test_nv_link_budget.py`` checks it, and
the derived budgets, against NVIDIA's source and against NVIDIA's own ``generate`` at the
boundary.
"""

from __future__ import annotations

import ast
import functools
import inspect
from typing import Any

CONTEXT_LIMIT = 32768

# Which NVIDIA function calls ``generate`` for which model kind (``pipeline.py``).
_CALL_SITES = {
    "Predictor.predict": "tempseg",
    "caption_single": "captioner",
    "caption_three": "captioner",
}


class ContextBudgetError(RuntimeError):
    """The prompt leaves less than the output budget within NVIDIA's context limit.

    Same meaning as NVIDIA's ``models.InputErrorForModel`` raised by that guard. When a SEGCAP
    family is linked, the raised object is also an instance of NVIDIA's class, so NVIDIA's own
    handling (and anyone catching ``InputErrorForModel``) treats it the same way.
    """

    def __init__(self, message: str, *, prompt_tokens: int, budget: int, limit: int):
        super().__init__(message)
        self.prompt_tokens, self.budget, self.limit = prompt_tokens, budget, limit

    def __reduce__(self):
        # Rebuilt with the receiving process's class (NVIDIA's, if it linked SEGCAP).
        return _rebuild_error, (str(self), self.prompt_tokens, self.budget, self.limit)


def _rebuild_error(message: str, prompt_tokens: int, budget: int, limit: int):
    return error_class()(message, prompt_tokens=prompt_tokens, budget=budget, limit=limit)


@functools.cache
def _nvidia_error_class(nvidia_cls: type) -> type:
    return type("ContextBudgetError", (ContextBudgetError, nvidia_cls), {
        "__module__": __name__, "__qualname__": "ContextBudgetError",
    })


def error_class(nv: Any = None) -> type:
    """``ContextBudgetError``, combined with NVIDIA's ``InputErrorForModel`` when available."""
    if nv is None:
        from flora_deploy.nv import Family
        from flora_deploy.nv.linker import loaded_family

        nv = loaded_family()
        if nv is None or nv.family is not Family.SEGCAP:
            return ContextBudgetError
    return _nvidia_error_class(nv.models.InputErrorForModel)


def check_context_budget(prompt_token_ids_len: int, budget: int, *, limit: int = CONTEXT_LIMIT,
                         nv: Any = None) -> None:
    """Raise like NVIDIA's ``models.generate`` would for a prompt of this many tokens.

    ``nv`` is the linked SEGCAP handle; by default the family linked in this process (if any)
    supplies NVIDIA's exception class.
    """
    n, budget = int(prompt_token_ids_len), int(budget)
    if n + budget > limit:
        raise error_class(nv)(
            f"prompt has {n} tokens; with an output budget of {budget} it exceeds the "
            f"{limit}-token context NVIDIA's generate() allows",
            prompt_tokens=n, budget=budget, limit=limit,
        )


def _source_tree(module) -> ast.Module:
    return ast.parse(inspect.getsource(module))


def _qualified_functions(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    yield f"{node.name}.{item.name}", item


def nvidia_budgets(nv) -> dict[str, int]:
    """``{"tempseg": ..., "captioner": ...}``: the ``budget`` argument of every ``generate(...)``
    call in NVIDIA's ``pipeline.py``, per model kind. Fails if a site is missing, is not a
    literal, or two sites of one kind disagree."""
    return dict(_budgets(nv.pipeline))


@functools.cache
def _budgets(pipeline) -> tuple[tuple[str, int], ...]:
    found: dict[str, set[int]] = {}
    seen = set()
    for qualname, fn in _qualified_functions(_source_tree(pipeline)):
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "generate"):
                continue
            if qualname not in _CALL_SITES:
                raise ValueError(f"unexpected generate() call in pipeline.{qualname}")
            args = node.args
            if len(args) != 4 or not isinstance(args[3], ast.Constant) or not isinstance(
                args[3].value, int
            ):
                raise ValueError(f"pipeline.{qualname}: generate() budget is not a literal")
            found.setdefault(_CALL_SITES[qualname], set()).add(args[3].value)
            seen.add(qualname)
    if seen != set(_CALL_SITES):
        raise ValueError(f"generate() call sites changed: found {sorted(seen)}")
    out = []
    for kind, values in sorted(found.items()):
        if len(values) != 1:
            raise ValueError(f"{kind}: generate() budgets disagree: {sorted(values)}")
        out.append((kind, values.pop()))
    return tuple(out)


def nvidia_context_limit(nv) -> int:
    """The constant in ``models.generate``'s ``n + budget > LIMIT`` guard."""
    tree = _source_tree(nv.models)
    fn = next(f for name, f in _qualified_functions(tree) if name == "generate")
    params = [a.arg for a in fn.args.args]
    budget = params[3]
    limits = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Gt) and isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, ast.Add)
                and isinstance(node.left.right, ast.Name) and node.left.right.id == budget
                and isinstance(node.comparators[0], ast.Constant)):
            limits.append(node.comparators[0].value)
    if len(limits) != 1:
        raise ValueError(f"models.generate context guard not found ({limits})")
    return int(limits[0])


def budget_for(nv, kind: str) -> int:
    """NVIDIA's ``max_new_tokens`` for ``kind`` (``"tempseg"`` or ``"captioner"``)."""
    return nvidia_budgets(nv)[kind]
