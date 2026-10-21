"""Structured measurement events and the per-context active recorder."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

EVENT_PREFIX = "FLORA_OBSERVE "

_current: ContextVar[Recorder | None] = ContextVar("flora_observe_recorder", default=None)


class Recorder:
    """Thread-safe event list for one unit of work (an episode, a batch).

    ``call(label, wall_s, self_s, cpu_s)`` and ``event(kind, **fields)`` append records; ``events``
    returns a snapshot. ``tags`` are copied into every record (e.g. ``episode``)."""

    def __init__(self, **tags: Any):
        self.tags = dict(tags)
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self.t0 = time.perf_counter()

    def event(self, kind: str, **fields: Any) -> dict:
        record = {"kind": kind, "t_s": round(time.perf_counter() - self.t0, 6), **self.tags,
                  **fields}
        with self._lock:
            self._events.append(record)
        return record

    def call(self, label: str, *, wall_s: float, self_s: float, cpu_s: float,
             error: str | None = None) -> None:
        fields: dict[str, Any] = {"label": label, "wall_s": wall_s, "self_s": self_s,
                                  "cpu_s": cpu_s}
        if error:
            fields["error"] = error
        self.event("call", **fields)

    @property
    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)


def current_recorder() -> Recorder | None:
    return _current.get()


@contextmanager
def recording(recorder: Recorder | None) -> Iterator[Recorder | None]:
    """Make ``recorder`` the active recorder for this context (and tasks/threads that copy it)."""
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def emit_json(record: dict, stream=None) -> None:
    """Print one ``FLORA_OBSERVE {json}`` line (sorted keys, NaN rejected)."""
    stream = sys.stdout if stream is None else stream
    stream.write(EVENT_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False,
                                           default=str) + "\n")
    stream.flush()
