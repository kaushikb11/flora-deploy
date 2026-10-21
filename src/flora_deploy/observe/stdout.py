"""``ContextStdout``: a thread-safe, context-aware replacement for ``redirect_stdout``.

``contextlib.redirect_stdout`` swaps the process-global ``sys.stdout``; with several episode
threads each capturing its own lines, captures would steal each other's output and restore the
wrong stream. ``route_stdout(sink)`` instead installs one proxy (reference-counted) that:

- buffers writes per context (a ``ContextVar`` holds the sink and its line buffer, so the routing
  follows ``asyncio.to_thread``);
- hands every *complete* line to the current context's ``sink(line)``;
- forwards complete lines to the real stream under a lock, so lines from concurrent threads are
  never interleaved mid-line.
"""

from __future__ import annotations

import io
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

Sink = Callable[[str], None]


class _Route:
    __slots__ = ("sink", "buf")

    def __init__(self, sink: Sink | None):
        self.sink = sink
        self.buf = ""


_route: ContextVar[_Route | None] = ContextVar("flora_stdout_route", default=None)
_install_lock = threading.Lock()


class ContextStdout(io.TextIOBase):
    def __init__(self, inner):
        self.inner = inner
        self._write_lock = threading.Lock()
        self._loose = threading.local()  # per-thread buffer for writes outside any route
        self.users = 0

    def writable(self) -> bool:
        return True

    @property
    def encoding(self):  # some libraries inspect it
        return getattr(self.inner, "encoding", "utf-8")

    def _buffer_holder(self):
        route = _route.get()
        if route is not None:
            return route
        holder = getattr(self._loose, "route", None)
        if holder is None:
            holder = self._loose.route = _Route(None)
        return holder

    def write(self, s: str) -> int:
        holder = self._buffer_holder()
        holder.buf += s
        if "\n" in holder.buf:
            *lines, holder.buf = holder.buf.split("\n")
            self._emit(holder, lines)
        return len(s)

    def _emit(self, holder: _Route, lines: list[str]) -> None:
        if holder.sink is not None:
            for line in lines:
                holder.sink(line)
        with self._write_lock:
            self.inner.write("".join(line + "\n" for line in lines))

    def flush_route(self, holder: _Route) -> None:
        if holder.buf:
            line, holder.buf = holder.buf, ""
            if holder.sink is not None:
                holder.sink(line)
            with self._write_lock:
                self.inner.write(line)
        self.flush()

    def flush(self) -> None:
        with self._write_lock:
            self.inner.flush()


@contextmanager
def route_stdout(sink: Sink) -> Iterator[None]:
    """Send this context's complete stdout lines to ``sink`` (and still to the real stdout)."""
    with _install_lock:
        proxy = sys.stdout
        if not isinstance(proxy, ContextStdout):
            proxy = ContextStdout(sys.stdout)
            sys.stdout = proxy
        proxy.users += 1
    route = _Route(sink)
    token = _route.set(route)
    try:
        yield
    finally:
        _route.reset(token)
        proxy.flush_route(route)
        with _install_lock:
            proxy.users -= 1
            if proxy.users == 0 and sys.stdout is proxy:
                loose = getattr(proxy._loose, "route", None)
                if loose is not None and loose.buf:
                    proxy.flush_route(loose)
                sys.stdout = proxy.inner
