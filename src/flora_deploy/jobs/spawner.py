"""``Spawner``: how the orchestrator starts and watches shard calls.

The real implementation (P1a-3) wraps Modal: ``Cls.from_name(...).run_shard.spawn(...)``,
``FunctionCall.from_id(call_id).get(timeout=0)``, ``.cancel()``. Tests use
``tests/fakes/modal.py::FakeSpawner``. The orchestrator must rely only on this protocol.

Status mapping the real implementation must honor (PLAN-v3.md §1.3; the code and the exception
table live in ``modal_app/orchestrator_app.py::status_from_get``):
- a still-running call: ``FunctionCall.get(timeout=0)`` raises the **builtin** ``TimeoutError``
  (modal 1.6.2), *not* ``modal.exception.TimeoutError`` → ``pending``;
- ``OutputExpiredError`` → ``expired``; ``FunctionTimeoutError`` → ``failed`` (both subclass
  ``modal.exception.TimeoutError``); transient client errors (connection / service / internal)
  → ``pending``, never ``failed`` (a respawn without cancel would duplicate the call);
- a call cancelled by the orchestrator's budget hard cap is re-queued by the orchestrator itself
  (``Orchestrator.enforce_budget``) and never consumes a respawn.
- Modal retries crashes/OOM/preemptions indefinitely inside one call; the orchestrator never sees
  those as failures. Progress and poison pills are detected from ``AttemptMarker`` files instead.

Spawn idempotency (``token``): the orchestrator writes a spawn *intent* (ledger status
``spawning`` + a deterministic ``attempts.spawn_token``) before calling ``spawn``, and passes the
token through to ``ShardWorker.run_shard(..., token=...)``. The worker stores it (with its call id)
in its ``AttemptMarker`` and exits without work if the ledger names a different token
(``attempts.superseded``). A restarted orchestrator uses that to adopt a call whose id it never
recorded, or to make a duplicate inert. Implementations that predate ``token`` still work (the
orchestrator detects the older signature), they just lose this protection.
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
    def spawn(self, kind: ShardKind, job_id: str, shard_id: str, config_hash: str, *,
              token: str = "") -> str:
        """Start one shard call; returns a call id that survives orchestrator restarts.
        ``token`` must reach the worker's ``run_shard(..., token=token)``."""
        ...

    def poll(self, call_id: str) -> CallStatus: ...

    def cancel(self, call_id: str) -> None: ...
