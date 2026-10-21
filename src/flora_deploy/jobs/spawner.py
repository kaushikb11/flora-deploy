"""``Spawner``: how the orchestrator starts and watches shard calls.

The real implementation (P1a-3) wraps Modal: ``Cls.from_name(...).run_shard.spawn(...)``,
``FunctionCall.from_id(call_id).get(timeout=0)``, ``.cancel()``. Tests use
``tests/fakes/modal.py::FakeSpawner``. The orchestrator must rely only on this protocol.

Status mapping the real implementation must honor (PLAN.md §1.3):
- ``OutputExpiredError`` → ``expired``; ``FunctionTimeoutError`` → ``failed``. Both subclass
  ``TimeoutError``, so catch them **before** treating ``TimeoutError`` as ``pending``.
- Modal retries crashes/OOM/preemptions indefinitely inside one call; the orchestrator never sees
  those as failures. Progress and poison pills are detected from ``AttemptMarker`` files instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from flora_deploy.jobs.types import ShardKind, ShardReceipt

CallState = Literal["pending", "success", "failed", "expired", "cancelled"]


@dataclass(frozen=True)
class CallStatus:
    call_id: str
    state: CallState
    receipt: ShardReceipt | None = None  # set when state == "success"
    error: str | None = None  # set when state == "failed"


class Spawner(Protocol):
    def spawn(self, kind: ShardKind, job_id: str, shard_id: str, config_hash: str) -> str:
        """Start one shard call; returns a call id that survives orchestrator restarts."""
        ...

    def poll(self, call_id: str) -> CallStatus: ...

    def cancel(self, call_id: str) -> None: ...
