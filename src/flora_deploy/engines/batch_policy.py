"""Batch policy of the balanced workers (WP1b): how many episodes feed one micro-batcher.

The policy decides which requests share an ``llm.generate`` round, and vLLM batch grouping
changes numerics (``engines.microbatch``), so it is **code, not a ``flora.yaml`` knob**: this
module is in the ``vllm-video`` logic digest (``config.logic``), so any change here changes the
balanced config key, and with it the certificate key. Making it configurable would mean adding it
to ``kind_modes`` (``config.hashing``); otherwise it would escape the key.

Per shard kind:

- ``infer_episodes``: episodes inside ``MicroBatcher.episode()`` at once (the inference phase).
  Segment 64, caption 32. These bound the requests handed to one ``llm.generate`` (an episode
  sends several windows / segments at once), not vLLM's residency: vLLM runs at most
  ``max_num_seqs`` of them at a time (``engines.offline_llm.GPU_LIMITS``: TempSeg 48 on the
  L40S, the Captioner 11 on the H100) and queues the rest.
- ``prefetch_episodes``: episodes in the CPU preparation phase (input resolve, NVIDIA's
  ``prepare_videos`` probe) or prepared and waiting for an inference slot. Preparation runs
  *outside* the batcher's episode context, so a preparing episode never holds a round back
  (``workers_impl.segment.EpisodeGates``). Host memory of prepared episodes is bounded by this
  count and by accel's shared cache budget (``docs/accel.md``).
- ``max_wait_s`` 0.5: only a safety net once preparation is out of the episode context.
  Arrivals during a running ``generate`` already pile up into the next round.
- ``max_batch`` / ``max_batch_bytes`` / ``max_pending_bytes``: the micro-batcher's bounds.

The runner records the effective policy (``as_json``) in the shard audit row and the workers'
startup facts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

POLICY_SCHEMA = "flora-batch-policy/v1"


@dataclass(frozen=True)
class BatchPolicy:
    kind: str  # balanced shard kind: segment | caption
    infer_episodes: int
    prefetch_episodes: int = 8
    max_batch: int = 128
    max_wait_s: float = 0.5
    max_batch_bytes: int = 3 << 30
    max_pending_bytes: int = 6 << 30

    def __post_init__(self) -> None:
        if self.infer_episodes < 1 or self.prefetch_episodes < 1 or self.max_batch < 1:
            raise ValueError("batch policy bounds must be >= 1")
        if self.max_wait_s <= 0:
            raise ValueError("max_wait_s must be > 0")

    def as_json(self) -> dict[str, Any]:
        body = dataclasses.asdict(self)
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return {"schema": POLICY_SCHEMA, **body,
                "digest": hashlib.sha256(text.encode()).hexdigest()[:16]}


POLICIES: Mapping[str, BatchPolicy] = {
    "segment": BatchPolicy("segment", infer_episodes=64),
    "caption": BatchPolicy("caption", infer_episodes=32),
}


def policy_for(kind: str, **overrides: Any) -> BatchPolicy:
    """The policy of shard kind ``kind``; ``overrides`` (``None`` values ignored) replace
    fields, for tests and the fake workers. A real worker passes none."""
    if kind not in POLICIES:
        raise ValueError(f"no batch policy for shard kind {kind!r}")
    given = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(POLICIES[kind], **given) if given else POLICIES[kind]
