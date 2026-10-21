"""Runtime call timers and attribute patching (we never edit the wrapped code's files).

``timed(fn, label)`` returns a wrapper that, when a recorder is active, records per call:

- ``wall_s``: ``perf_counter`` wall-clock;
- ``self_s``: wall minus the wall of nested *timed* calls on the same thread (so nested labels,
  e.g. ``media.decode_selected`` inside ``media.frames_for``, are not double counted);
- ``cpu_s``: ``time.thread_time`` of the calling thread. Native worker threads (torch intra-op,
  OpenMP) are not included; the ``Sampler`` measures whole-process CPU.

With no active recorder the wrapper only does one ``ContextVar.get`` and calls through.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any

from flora_deploy.observe.events import current_recorder

WRAPPED_MARK = "__flora_observe_label__"
_stack = threading.local()


def _frames() -> list[list[float]]:
    frames = getattr(_stack, "frames", None)
    if frames is None:
        frames = _stack.frames = []
    return frames


def timed(fn: Callable, label: str | Callable[..., str]) -> Callable:
    """Wrap ``fn``; ``label`` may be a function of the call's ``(*args, **kwargs)``."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rec = current_recorder()
        if rec is None:
            return fn(*args, **kwargs)
        name = label(*args, **kwargs) if callable(label) else label
        frames = _frames()
        frames.append([0.0])  # children wall
        c0, t0 = time.thread_time(), time.perf_counter()
        error = None
        try:
            return fn(*args, **kwargs)
        except BaseException as e:
            error = type(e).__name__
            raise
        finally:
            wall = time.perf_counter() - t0
            cpu = time.thread_time() - c0
            children = frames.pop()[0]
            if frames:
                frames[-1][0] += wall
            rec.call(name, wall_s=wall, self_s=max(0.0, wall - children), cpu_s=cpu, error=error)

    setattr(wrapper, WRAPPED_MARK, label)
    return wrapper


class Patcher:
    """Replace attributes at runtime and restore them (LIFO). Also a context manager.

    ``patch(owner, name, factory)`` sets ``owner.name = factory(original)``. Patching the same
    attribute twice raises, so a wrapper is never stacked on itself by accident."""

    def __init__(self):
        self._undo: list[tuple[Any, str, Any, bool]] = []

    def patch(self, owner: Any, name: str, factory: Callable[[Any], Any]) -> Any:
        had = name in vars(owner) if isinstance(owner, type) else hasattr(owner, name)
        original = getattr(owner, name)
        if any(o is owner and n == name for o, n, _, _ in self._undo):
            raise RuntimeError(f"{owner!r}.{name} is already patched")
        replacement = factory(original)
        setattr(owner, name, replacement)
        self._undo.append((owner, name, original, had))
        return replacement

    def time(self, owner: Any, name: str, label: str | Callable[..., str]) -> Any:
        return self.patch(owner, name, lambda orig: timed(orig, label))

    @property
    def patched(self) -> list[str]:
        return [f"{getattr(o, '__name__', type(o).__name__)}.{n}" for o, n, _, _ in self._undo]

    def restore(self) -> None:
        while self._undo:
            owner, name, original, had = self._undo.pop()
            if had:
                setattr(owner, name, original)
            else:  # an inherited class attribute: drop our override
                delattr(owner, name)

    def __enter__(self) -> Patcher:
        return self

    def __exit__(self, *exc) -> None:
        self.restore()
