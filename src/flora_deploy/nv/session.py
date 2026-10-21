"""``CachedSession``: keep NVIDIA's models resident across episodes (PLAN.md §6.5).

NVIDIA's ``models.Session`` holds one model at a time: ``load(kind)`` first calls ``close()``, and
``pipeline.annotate`` closes it after every video. For a long-lived worker we wrap it:

- one inner NVIDIA ``Session`` per kind (``"tempseg"``, ``"captioner"``), each loaded once with
  NVIDIA's own ``Session.load`` (which runs all of NVIDIA's integrity and GDN checks);
- ``load(kind)`` returns the cached ``(model, proc)``;
- ``close()`` (called by NVIDIA's code per video) does nothing; ``close_all()`` really releases
  every inner session at container exit.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol


class SessionLike(Protocol):
    audit: dict

    def load(self, kind: str) -> tuple[Any, Any]: ...

    def close(self) -> None: ...


class CachedSession:
    """Duck-types NVIDIA ``models.Session`` for ``pipeline.segment`` / ``caption_*``."""

    def __init__(self, session_factory: Callable[[], SessionLike]):
        self._factory = session_factory
        self._sessions: dict[str, SessionLike] = {}
        self._loaded: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.Lock()
        self._kind_locks: dict[str, threading.Lock] = {}

    @classmethod
    def from_family(cls, nv) -> CachedSession:
        """Build from a linked SEGCAP family handle (``nv.models.Session``)."""
        return cls(nv.models.Session)

    def load(self, kind: str) -> tuple[Any, Any]:
        with self._lock:
            if kind in self._loaded:
                return self._loaded[kind]
            kind_lock = self._kind_locks.setdefault(kind, threading.Lock())
        with kind_lock:
            with self._lock:
                if kind in self._loaded:
                    return self._loaded[kind]
            session = self._factory()
            model, proc = session.load(kind)
            with self._lock:
                self._sessions[kind] = session
                self._loaded[kind] = (model, proc)
            return model, proc

    def close(self) -> None:
        """No-op: NVIDIA calls this after every video; models stay resident."""

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._loaded.clear()
        for session in sessions:
            session.close()

    @property
    def loaded_kinds(self) -> tuple[str, ...]:
        return tuple(self._loaded)

    @property
    def audit(self) -> dict:
        """NVIDIA's per-kind load audit records, merged across inner sessions."""
        merged: dict = {}
        for session in self._sessions.values():
            merged.update(getattr(session, "audit", {}) or {})
        return merged

    def __enter__(self) -> CachedSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close_all()
