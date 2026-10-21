"""Opt-in runtime patches of a linked NVIDIA family (PLAN.md D17, form 1).

A ``Patch`` swaps attributes of NVIDIA's *already imported* modules in this process only:

- NVIDIA's files are never touched, so NVIDIA's own SHA checks (``bootstrap`` vendor manifest,
  ``CODE_MANIFEST.json``) still pass and the unmodified code stays importable as the oracle;
- every patch is individually toggleable and reversible (``ActivePatches.restore``);
- every patch names the parity test that proves its replacement output-identical to NVIDIA's
  original, and ``ActivePatches.provenance()`` records what was active for a run.

Targets are dotted attribute paths relative to the family handle:

- ``"batch_utils.<module>.<attr>"``: ``nv.module(<module>)`` (e.g. ``batch_utils.pipeline``);
- ``"vendor.<dotted.module>.<attr>"``: a top-level module NVIDIA's bootstrap made importable,
  via ``nv.vendor`` (which checks it comes from the same snapshot).

A patch factory receives the family handle, the *current* value of each target (which may
already be another active patch's wrapper; patches compose in registry order) and the shared
``AccelState``, and returns the replacement for each target.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from flora_deploy.nv import Family

PATCH_SCHEMA = "flora_accel_patches_v1"


class PatchError(RuntimeError):
    """A patch cannot be applied or restored safely."""


@dataclass(frozen=True)
class Patch:
    name: str
    family: Family
    targets: tuple[str, ...]
    factory: Callable[[Any, Mapping[str, Any], Any], Mapping[str, Any]]
    parity_test: str
    description: str
    version: int = 1

    def provenance(self) -> dict:
        return dict(
            name=self.name,
            version=self.version,
            family=self.family.value,
            targets=list(self.targets),
            parity_test=self.parity_test,
        )


_REGISTRY: dict[str, Patch] = {}
_LOCK = threading.RLock()
# (id(module), attr) -> stack of (ActivePatches, patch name) that replaced it (innermost last)
_OWNERS: dict[tuple[int, str], list[tuple[ActivePatches, str]]] = {}


def register(patch: Patch) -> Patch:
    with _LOCK:
        if patch.name in _REGISTRY and _REGISTRY[patch.name] is not patch:
            raise PatchError(f"patch {patch.name!r} already registered")
        _REGISTRY[patch.name] = patch
    return patch


def registered() -> dict[str, Patch]:
    """All registered patches, in application order."""
    _ensure_builtin()
    return dict(_REGISTRY)


def _ensure_builtin() -> None:
    # The built-in patches register themselves on import.
    from flora_deploy.accel import patches  # noqa: F401


def resolve_target(nv, target: str) -> tuple[Any, str]:
    """Owner (module, or a class inside one) and attribute name a target path designates,
    e.g. ``batch_utils.pipeline.frames_for`` or ``batch_utils.pipeline.Predictor.predict``."""
    head, _, rest = target.partition(".")
    parts = rest.split(".")
    if len(parts) < 2 or not all(parts):
        raise PatchError(f"bad patch target {target!r}")
    if head == "batch_utils":
        load = nv.module
    elif head == "vendor":
        load = nv.vendor
    else:
        raise PatchError(f"bad patch target {target!r}: must start with batch_utils. or vendor.")
    for k in range(len(parts) - 1, 0, -1):
        try:
            owner = load(".".join(parts[:k]))
        except ModuleNotFoundError:
            continue
        for name in parts[k:-1]:
            owner = getattr(owner, name)
        if not hasattr(owner, parts[-1]):
            raise PatchError(f"{target!r}: no attribute {parts[-1]!r}")
        return owner, parts[-1]
    raise PatchError(f"bad patch target {target!r}: no importable module")


@dataclass
class ActivePatches:
    """Patches applied to one family handle; ``restore()`` undoes them (LIFO)."""

    nv: Any
    patches: tuple[Patch, ...]
    state: Any
    _swaps: list[tuple[Any, str, Any, Any]] = field(default_factory=list, repr=False)
    _restored: bool = False

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.patches]

    def provenance(self) -> dict:
        """What ran, for ``Provenance.engine`` / run records."""
        return dict(schema=PATCH_SCHEMA, patches=[p.provenance() for p in self.patches])

    def episode(self):
        """Context manager bounding the per-episode caches (see ``AccelState.episode``)."""
        return self.state.episode()

    def stats(self) -> dict:
        return self.state.stats()

    def restore(self) -> None:
        with _LOCK:
            if self._restored:
                return
            outermost: dict[tuple[int, str], tuple[Any, str, Any]] = {}
            for module, attr, _original, replacement in self._swaps:
                outermost[(id(module), attr)] = (module, attr, replacement)
            for key, (module, attr, replacement) in outermost.items():
                stack = _OWNERS.get(key, [])
                if (
                    not stack
                    or stack[-1][0] is not self
                    or getattr(module, attr) is not replacement
                ):
                    raise PatchError(
                        f"{module.__name__}.{attr} was re-patched after this patch set; restore "
                        "the later patch set first"
                    )
            for module, attr, original, _replacement in reversed(self._swaps):
                setattr(module, attr, original)
                stack = _OWNERS[(id(module), attr)]
                stack.pop()
                if not stack:
                    del _OWNERS[(id(module), attr)]
            self._restored = True
            self.state.close()

    def __enter__(self) -> ActivePatches:
        return self

    def __exit__(self, *exc) -> None:
        self.restore()


def apply_patches(nv, names: Iterable[str], *, state=None, **options) -> ActivePatches:
    """Apply the named patches (registry order) to the linked family ``nv``.

    ``options`` configure the shared ``AccelState`` (e.g. ``cache_bytes``, ``workers``) when no
    ``state`` is given. Returns the handle whose ``restore()`` puts NVIDIA's attributes back.
    """
    from flora_deploy.accel.state import AccelState

    available = registered()
    wanted = list(dict.fromkeys(names))
    unknown = [n for n in wanted if n not in available]
    if unknown:
        raise PatchError(f"unknown patches {unknown}; available: {sorted(available)}")
    chosen = tuple(p for n, p in available.items() if n in wanted)
    for p in chosen:
        if p.family is not nv.family:
            raise PatchError(f"patch {p.name!r} is for {p.family.value}, not {nv.family.value}")
    state = state if state is not None else AccelState(**options)
    active = ActivePatches(nv=nv, patches=chosen, state=state)
    with _LOCK:
        try:
            for patch in chosen:
                resolved = [resolve_target(nv, t) for t in patch.targets]
                for module, attr in resolved:
                    stack = _OWNERS.get((id(module), attr), [])
                    if any(name == patch.name for _, name in stack):
                        raise PatchError(f"patch {patch.name!r} is already active on {attr}")
                current = {
                    t: getattr(m, a) for t, (m, a) in zip(patch.targets, resolved, strict=True)
                }
                replacements = patch.factory(nv, current, state)
                if set(replacements) != set(patch.targets):
                    raise PatchError(f"patch {patch.name!r} factory returned wrong targets")
                for t, (module, attr) in zip(patch.targets, resolved, strict=True):
                    new = replacements[t]
                    setattr(module, attr, new)
                    active._swaps.append((module, attr, current[t], new))
                    _OWNERS.setdefault((id(module), attr), []).append((active, patch.name))
        except BaseException:
            for module, attr, original, _new in reversed(active._swaps):
                setattr(module, attr, original)
                stack = _OWNERS.get((id(module), attr), [])
                if stack and stack[-1][0] is active:
                    stack.pop()
                if not stack:
                    _OWNERS.pop((id(module), attr), None)
            active._restored = True
            state.close()
            raise
    return active
