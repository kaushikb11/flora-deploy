"""Opt-in measurement harness: structured events, call timers, timed locks, resource sampling.

Generic pieces, no NVIDIA imports (the NVIDIA-specific target list lives in
``stages/reference.py``):

- ``events``: ``Recorder`` (thread-safe event list for one unit of work, usually an episode) and
  the ``recording(recorder)`` context. The active recorder is a ``ContextVar``, so it follows
  ``asyncio.to_thread`` (NVIDIA's segmenter calls its predictor that way) but not bare threads.
- ``timers``: ``timed(fn, label)`` wall/thread-CPU timers with self-time accounting, and
  ``Patcher``, which swaps module/class attributes at runtime and restores them.
- ``locks``: ``TimedLock``, a mutex that records wait and hold times (the GPU lock).
- ``sampler``: ``Sampler`` thread for RSS, process CPU, cgroup memory and ``/dev/shm``.
- ``stdout``: ``ContextStdout``, a ``sys.stdout`` proxy that routes complete lines to the sink of
  the current context (thread-safe replacement for ``redirect_stdout``).
- ``aggregate``: per-episode call tables and token distributions.

Every emitted record is a flat JSON-able dict; ``emit_json`` prints it as one
``FLORA_OBSERVE {json}`` line.
"""

from flora_deploy.observe.aggregate import call_table, distribution, merge_tables, token_stats
from flora_deploy.observe.events import (
    EVENT_PREFIX,
    Recorder,
    current_recorder,
    emit_json,
    recording,
)
from flora_deploy.observe.locks import TimedLock
from flora_deploy.observe.sampler import Sampler, cgroup_memory_bytes, rss_bytes, shm_usage
from flora_deploy.observe.stdout import ContextStdout, route_stdout
from flora_deploy.observe.timers import Patcher, timed

__all__ = [
    "EVENT_PREFIX",
    "ContextStdout",
    "Patcher",
    "Recorder",
    "Sampler",
    "TimedLock",
    "call_table",
    "cgroup_memory_bytes",
    "current_recorder",
    "distribution",
    "emit_json",
    "merge_tables",
    "recording",
    "route_stdout",
    "rss_bytes",
    "shm_usage",
    "timed",
    "token_stats",
]
