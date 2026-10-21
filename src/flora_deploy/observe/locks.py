"""``TimedLock``: a mutex that measures contention (used as the exact preset's GPU lock)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from flora_deploy.observe.events import current_recorder


class TimedLock:
    """Non-reentrant lock with global wait/hold totals and per-recorder ``lock`` events.

    ``held(label)`` acquires, records ``wait_s`` and ``hold_s`` to the active recorder, and keeps
    totals (``stats()``). ``busy_fraction(t0, t1)`` is the share of ``[t0, t1]`` (perf_counter
    seconds) during which the lock was held, from the recorded hold intervals."""

    def __init__(self, name: str = "lock", clock=time.perf_counter):
        self.name = name
        self._lock = threading.Lock()
        self._meta = threading.Lock()
        self._clock = clock
        self.acquisitions = 0
        self.contended = 0
        self.wait_s = 0.0
        self.hold_s = 0.0
        self.max_wait_s = 0.0
        self.intervals: list[tuple[float, float]] = []
        self._holder: int | None = None

    @contextmanager
    def held(self, label: str | None = None) -> Iterator[None]:
        if self._holder == threading.get_ident():
            raise RuntimeError(f"{self.name} is not reentrant")
        t_req = self._clock()
        contended = not self._lock.acquire(blocking=False)
        if contended:
            self._lock.acquire()
        t_acq = self._clock()
        self._holder = threading.get_ident()
        try:
            yield
        finally:
            t_rel = self._clock()
            self._holder = None
            self._lock.release()
            wait, hold = t_acq - t_req, t_rel - t_acq
            with self._meta:
                self.acquisitions += 1
                self.contended += contended
                self.wait_s += wait
                self.hold_s += hold
                self.max_wait_s = max(self.max_wait_s, wait)
                self.intervals.append((t_acq, t_rel))
            rec = current_recorder()
            if rec is not None:
                rec.event("lock", lock=self.name, label=label, wait_s=wait, hold_s=hold,
                          contended=contended)

    def locked(self) -> bool:
        return self._lock.locked()

    def busy_fraction(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        with self._meta:
            busy = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in self.intervals)
        return busy / (t1 - t0)

    def stats(self) -> dict:
        with self._meta:
            return {
                "lock": self.name,
                "acquisitions": self.acquisitions,
                "contended": self.contended,
                "wait_s": round(self.wait_s, 3),
                "hold_s": round(self.hold_s, 3),
                "max_wait_s": round(self.max_wait_s, 3),
            }

    def reset(self) -> None:
        with self._meta:
            self.acquisitions = self.contended = 0
            self.wait_s = self.hold_s = self.max_wait_s = 0.0
            self.intervals = []
