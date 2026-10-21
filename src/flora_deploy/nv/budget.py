"""NVIDIA's 32k context guard, reproduced for vLLM requests (PLAN-v3.md §6.3).

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
``nvidia_context_limit(nv)`` reads the limit from the comparison in ``models.generate``, and
``check_context_budget`` uses that parsed limit when a SEGCAP family is linked.

Both parse the exact bytes the nv linker verified and compiled (``executed_source``), never the
file re-read from disk, and fail closed on any shape they do not know: a ``generate`` call in
another function or at module level, ``models.generate(...)``, an alias, a re-import under
another name, a non-literal budget.

LICENSE REVIEW FLAG (PLAN.md §7): ``CONTEXT_LIMIT = 32768`` restates the constant of
``models.py:169``. It is a default only. The nvlink test ``test_nv_link_budget.py`` checks it, and
the derived budgets, against NVIDIA's source and against NVIDIA's own ``generate`` at the
boundary.
"""

from __future__ import annotations

import ast
import functools
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


def check_context_budget(prompt_token_ids_len: int, budget: int, *, limit: int | None = None,
                         nv: Any = None) -> None:
    """Raise like NVIDIA's ``models.generate`` would for a prompt of this many tokens.

    ``nv`` is the linked SEGCAP handle; by default the family linked in this process (if any)
    supplies NVIDIA's exception class. ``limit`` defaults to the constant parsed from the
    linked family's verified ``models.generate`` (``nvidia_context_limit``), or
    ``CONTEXT_LIMIT`` when no real family is linked (fakes, tests).
    """
    n, budget = int(prompt_token_ids_len), int(budget)
    if limit is None:
        limit = _runtime_limit(nv)
    if n + budget > limit:
        raise error_class(nv)(
            f"prompt has {n} tokens; with an output budget of {budget} it exceeds the "
            f"{limit}-token context NVIDIA's generate() allows",
            prompt_tokens=n, budget=budget, limit=limit,
        )


def _runtime_limit(nv: Any) -> int:
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import NvFamily, loaded_family

    family = nv if nv is not None else loaded_family()
    if isinstance(family, NvFamily) and family.family is Family.SEGCAP:
        return nvidia_context_limit(family)  # fails closed if NVIDIA's guard changed shape
    return CONTEXT_LIMIT


def executed_source(module) -> bytes:
    """The exact bytes the nv linker verified and compiled for ``module``.

    Never re-reads the file: a file edited on disk after import would otherwise be parsed
    instead of the code that runs. A module not imported through the linker is refused."""
    loader = getattr(getattr(module, "__spec__", None), "loader", None)
    data = getattr(loader, "executed_source", None)
    if not isinstance(data, bytes):
        raise ValueError(
            f"{getattr(module, '__name__', module)!r} was not imported by the nv linker; "
            "refusing to parse its source from disk"
        )
    return data


def _qualified_functions(tree: ast.Module):
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, defs):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, defs):
                    yield f"{node.name}.{item.name}", item


class _GenerateSites(ast.NodeVisitor):
    """Every ``generate(...)`` call by enclosing top-level function / method; any other use of
    the name ``generate`` (attribute calls such as ``models.generate``, aliases, re-imports,
    parameters, ``getattr(..., "generate")``, module-level calls) is recorded as an error."""

    def __init__(self):
        self.stack: list[tuple[str, str]] = []
        self.calls: dict[str, list[ast.Call]] = {}
        self.errors: list[str] = []

    def _qualname(self) -> str:
        if not self.stack:
            return "<module>"
        if self.stack[0][0] == "class":
            return ".".join(name for _, name in self.stack[:2])
        return self.stack[0][1]

    def _bad(self, node, what: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {what}")

    def _scope(self, kind: str, node) -> None:
        if node.name == "generate":
            self._bad(node, "redefines generate")
        self.stack.append((kind, node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node):
        self._scope("class", node)

    def visit_FunctionDef(self, node):
        self._scope("def", node)

    def visit_AsyncFunctionDef(self, node):
        self._scope("def", node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "generate":
            self.calls.setdefault(self._qualname(), []).append(node)
            for child in (*node.args, *node.keywords):
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "generate":
            self._bad(node, "'generate' used other than as a direct call")

    def visit_Attribute(self, node):
        if node.attr == "generate":
            self._bad(node, "attribute '.generate'")
        self.generic_visit(node)

    def visit_arg(self, node):
        if node.arg == "generate":
            self._bad(node, "parameter named generate")

    def visit_Global(self, node):
        if "generate" in node.names:
            self._bad(node, "global generate")

    visit_Nonlocal = visit_Global

    def visit_Constant(self, node):
        if node.value == "generate":
            self._bad(node, "string 'generate' (getattr?)")

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                self._bad(node, f"star import from {node.module}")
            elif alias.name == "generate" and (
                alias.asname not in (None, "generate") or node.module != "models"
                or node.level != 1
            ):
                self._bad(node, "generate imported under another name or from elsewhere")
            elif alias.asname == "generate":
                self._bad(node, f"{alias.name} imported as generate")

    def visit_Import(self, node):
        for alias in node.names:
            if (alias.asname or alias.name).split(".")[0] == "generate":
                self._bad(node, "module imported as generate")


def nvidia_budgets(nv) -> dict[str, int]:
    """``{"tempseg": ..., "captioner": ...}``: the ``budget`` argument of every ``generate(...)``
    call in NVIDIA's ``pipeline.py`` (the verified bytes that run), per model kind. Fails if a
    site is missing, unexpected (any other function, module level, ``models.generate``, an
    alias, ...), is not a literal, or two sites of one kind disagree."""
    return dict(_budgets(executed_source(nv.pipeline)))


@functools.cache
def _budgets(source: bytes) -> tuple[tuple[str, int], ...]:
    sites = _GenerateSites()
    sites.visit(ast.parse(source))
    if sites.errors:
        raise ValueError("unexpected generate() reference in pipeline: " + "; ".join(sites.errors))
    found: dict[str, set[int]] = {}
    for qualname, calls in sites.calls.items():
        if qualname not in _CALL_SITES:
            raise ValueError(f"unexpected generate() call in pipeline.{qualname}")
        for node in calls:
            args = node.args
            if (len(args) != 4 or node.keywords or not isinstance(args[3], ast.Constant)
                    or type(args[3].value) is not int):
                raise ValueError(f"pipeline.{qualname}: generate() budget is not a literal")
            found.setdefault(_CALL_SITES[qualname], set()).add(args[3].value)
    if set(sites.calls) != set(_CALL_SITES):
        raise ValueError(f"generate() call sites changed: found {sorted(sites.calls)}")
    out = []
    for kind, values in sorted(found.items()):
        if len(values) != 1:
            raise ValueError(f"{kind}: generate() budgets disagree: {sorted(values)}")
        out.append((kind, values.pop()))
    return tuple(out)


def nvidia_context_limit(nv) -> int:
    """The constant in ``models.generate``'s ``n + budget > LIMIT`` guard (verified bytes)."""
    return _context_limit(executed_source(nv.models))


@functools.cache
def _context_limit(source: bytes) -> int:
    tree = ast.parse(source)
    fns = [f for name, f in _qualified_functions(tree) if name == "generate"]
    if len(fns) != 1:
        raise ValueError(f"models.generate not found exactly once ({len(fns)})")
    fn = fns[0]
    params = [a.arg for a in fn.args.args]
    if len(params) < 4:
        raise ValueError(f"models.generate has unexpected parameters {params}")
    budget = params[3]
    limits = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Gt) and isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, ast.Add)
                and isinstance(node.left.right, ast.Name) and node.left.right.id == budget
                and isinstance(node.comparators[0], ast.Constant)
                and type(node.comparators[0].value) is int):
            limits.append(node.comparators[0].value)
    if len(limits) != 1:
        raise ValueError(f"models.generate context guard not found ({limits})")
    return int(limits[0])


def budget_for(nv, kind: str) -> int:
    """NVIDIA's ``max_new_tokens`` for ``kind`` (``"tempseg"`` or ``"captioner"``)."""
    return nvidia_budgets(nv)[kind]
