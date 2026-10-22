"""Virtual-clock simulation of a balanced TempSeg shard: batch size and GPU-busy fraction of the
one-phase path (before WP1) against the two-phase path with the batch policy (after).

The **real** ``BalancedShardRunner._episodes`` runs: its ``EpisodeGates``, the real
``MicroBatcher`` round triggers and the policy bounds. Only time is virtual: the event loop's
clock jumps to the next timer whenever nothing is runnable (:class:`VirtualLoop`), so a
150-second shard simulates in milliseconds and the result is deterministic. What is modelled:

- **the job** (:class:`SimSegmentJob`, two-phase like ``SegmentEpisodeJob``): input resolve +
  NVIDIA's probe (``probe_s``), then per ``apredict`` round, frame building (``frames_s`` per
  window) and one ``generate_many`` of its windows. CPU work holds one of ``cores`` slots, so
  probes and frame building contend for the container's cores on both paths.
- **the GPU** (:class:`SimBatcher`): a round of ``n`` requests takes
  ``waves * round_s + per_request_s * n`` with ``waves = ceil(n / max_num_seqs)``; one round at
  a time, as with one ``llm.generate``.

Defaults come from the Wave H L40S run (``reports/wave-h/balanced-tempseg.json``): probes of
3.1 / 3.0 / 3.6 / 29.1 / 41.2 / 2.8 s, 1-2 ``apredict`` rounds, 2-4 windows per round, and
rounds averaging 8.4 s at a mean batch of 4.5 (``round_s`` 7.0 + ``per_request_s`` 0.3). The
run did not time frame building separately: ``frames_s`` 10 per window replays its 6 episodes
(139.6 s wall, GPU busy 41%, 6 of 7 rounds by timeout; measured 139.5 s, 36%, 5 of 6), and
``frames_s`` 0.5 is the accel decode-once case. They are a model, not a measurement: the
design's lab run measures the real numbers.

``python -m tests.fakes.batch_sim`` prints the before/after table (:data:`REGIMES`).
"""

from __future__ import annotations

import asyncio
import math
import selectors
from dataclasses import dataclass, field
from typing import Any

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.engines.microbatch import MicroBatcher
from flora_deploy.jobs.types import EpisodeRef, Shard
from flora_deploy.workers_impl.segment import (
    BalancedShardRunner,
    BalancedWorkerSettings,
    EpisodeOutcome,
    ShardContext,
    gates_of,
)

# Wave H (L40S TempSeg): (probe_s, apredict rounds, windows per round) per episode, cycled. The
# 15-16 s egodex clips have 2 windows of 16 s, the 38-47 s homer clips 4 (28 requests in all;
# the run made 27).
WAVE_H = ((3.12, 1, 2), (2.99, 2, 2), (3.58, 1, 2), (29.06, 2, 4), (41.24, 2, 4), (2.75, 1, 4))


class _Clock:
    now = 0.0


class _VirtualSelector(selectors.BaseSelector):
    """Real selector for the loop's self-pipe; an empty poll advances the virtual clock by the
    loop's timeout instead of sleeping."""

    def __init__(self, clock: _Clock):
        self._real = selectors.DefaultSelector()
        self._clock = clock

    def register(self, fileobj, events, data=None):
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._real.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._real.modify(fileobj, events, data)

    def get_map(self):
        return self._real.get_map()

    def close(self):
        self._real.close()

    def select(self, timeout=None):
        ready = self._real.select(0)
        if ready:
            return ready
        if timeout is None:
            raise RuntimeError("simulation deadlock: nothing runnable and no timer")
        self._clock.now += max(0.0, timeout)
        return []


class VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        self._vclock = _Clock()
        super().__init__(_VirtualSelector(self._vclock))

    def time(self) -> float:
        return self._vclock.now


@dataclass
class GpuModel:
    round_s: float = 7.0
    per_request_s: float = 0.3
    max_num_seqs: int = 32

    def seconds(self, n: int) -> float:
        return math.ceil(n / self.max_num_seqs) * self.round_s + self.per_request_s * n


class SimBatcher(MicroBatcher):
    """``MicroBatcher`` whose ``generate`` takes virtual GPU time (``GpuModel``)."""

    gpu = GpuModel()

    async def _dispatch(self, loop, reqs, sps):
        await asyncio.sleep(self.gpu.seconds(len(reqs)))
        return self._generate(reqs, sps)


class EchoEngine:
    name = "sim"

    def generate_batch(self, requests, samplings):
        return [Completion(key=r.key, token_ids=(1,), finish_reason="stop", prompt_tokens=1850)
                for r in requests]


class SimSegmentJob:
    """``SegmentEpisodeJob``'s shape: probe under ``gates.prep()``; per round frame building
    and one ``generate_many`` under ``gates.infer()``."""

    stage = "segment"
    phased = True
    nv = None

    def __init__(self, cores: int, plan: dict[str, tuple[float, int, int]], frames_s: float):
        self.cpu = asyncio.Semaphore(cores)
        self.plan, self.frames_s = plan, frames_s

    async def _cpu(self, seconds: float) -> None:
        async with self.cpu:
            await asyncio.sleep(seconds)

    async def run(self, ref: EpisodeRef, batcher, ctx) -> EpisodeOutcome:
        probe_s, rounds, windows = self.plan[ref.uid]
        gates = gates_of(ctx)
        async with gates.prep():
            await self._cpu(probe_s)
        async with gates.infer(batcher):
            for r in range(rounds):
                await self._cpu(self.frames_s * windows)
                reqs = [VllmRequest(prompt=f"{ref.uid}/{r}/{w}", videos=(),
                                    key=f"{ref.uid}#{r}.{w}") for w in range(windows)]
                await batcher.generate_many(reqs, [Sampling(max_tokens=1024)] * len(reqs))
        return EpisodeOutcome([], [])


class _Sink:
    def add(self, records=(), audits=(), errors=()):
        if errors:
            raise RuntimeError(f"simulated episode failed: {errors}")

    def flush(self):
        return []


@dataclass
class Scenario:
    name: str
    phased: bool
    infer_episodes: int | None  # None: the batch policy's
    max_wait_s: float | None
    max_num_seqs: int
    prefetch_episodes: int | None = None


BEFORE = Scenario("before (one phase, 32 in flight, max_wait 2.0 s, max_num_seqs 32)",
                  phased=False, infer_episodes=32, max_wait_s=2.0, max_num_seqs=32)
AFTER = Scenario("after (two phases, batch policy, max_num_seqs 64)",
                 phased=True, infer_episodes=None, max_wait_s=None, max_num_seqs=64)


@dataclass
class SimResult:
    scenario: str
    episodes: int
    makespan_s: float
    gpu_busy_s: float
    rounds: int
    mean_batch: float
    triggers: dict
    batch_sizes: list[int] = field(default_factory=list)
    gates: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)

    @property
    def gpu_busy_fraction(self) -> float:
        return self.gpu_busy_s / self.makespan_s if self.makespan_s else 0.0

    def as_json(self) -> dict[str, Any]:
        return {"scenario": self.scenario, "episodes": self.episodes,
                "makespan_s": round(self.makespan_s, 1), "gpu_busy_s": round(self.gpu_busy_s, 1),
                "gpu_busy_fraction": round(self.gpu_busy_fraction, 3), "rounds": self.rounds,
                "mean_batch": self.mean_batch, "triggers": self.triggers,
                "gates": self.gates}


def simulate(sc: Scenario, *, episodes: int = 64, cores: int = 8, frames_s: float = 0.5,
             gpu: GpuModel | None = None,
             mix=WAVE_H) -> SimResult:
    gpu = gpu or GpuModel(max_num_seqs=sc.max_num_seqs)
    refs = {f"sim--ep{i:03d}": EpisodeRef(uid=f"sim--ep{i:03d}", dataset="sim",
                                          episode_id=f"ep{i:03d}", pool="human",
                                          duration_s=30.0, oracle_eligible=True,
                                          envelope_ok=True, index_ref="x")
            for i in range(episodes)}
    plan = {u: mix[i % len(mix)] for i, u in enumerate(refs)}
    settings = BalancedWorkerSettings(phased=sc.phased, max_inflight_episodes=sc.infer_episodes,
                                      prefetch_episodes=sc.prefetch_episodes,
                                      max_wait_s=sc.max_wait_s, flush_every=10**9,
                                      executor_workers=1)
    loop = VirtualLoop()
    try:
        asyncio.set_event_loop(loop)
        job = SimSegmentJob(cores, plan, frames_s)
        runner = BalancedShardRunner(None, EchoEngine(), job, settings)  # type: ignore[arg-type]
        cls = type("Sim", (SimBatcher,), {"gpu": gpu})
        runner.batcher_cls = cls
        shard = Shard("segment-sim", "segment", tuple(refs), 1.0)
        ctx = ShardContext(None, shard, None, "sim")  # type: ignore[arg-type]
        t0 = loop.time()
        loop.run_until_complete(runner._episodes(list(refs), refs, _Sink(), ctx))  # type: ignore[arg-type]
        makespan = loop.time() - t0
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    st = runner.last_stats
    return SimResult(sc.name, episodes, makespan, st["generate_s"], st["rounds"],
                     st["mean_batch"], st["triggers"], st["batch_sizes"], runner.last_gates,
                     runner.policy.as_json())


# (label, episodes in the shard, frames_s per window)
REGIMES = (("wave-h replay", 6, 10.0), ("64 episodes, Wave H frame cost", 64, 10.0),
           ("64 episodes, cheap frames (accel)", 64, 0.5),
           ("256 episodes, cheap frames (accel)", 256, 0.5))


def main() -> None:
    import json

    for label, n, frames_s in REGIMES:
        for sc in (BEFORE, AFTER):
            r = simulate(sc, episodes=n, frames_s=frames_s)
            print(json.dumps({"regime": label, "frames_s": frames_s, **r.as_json()}),
                  flush=True)


if __name__ == "__main__":
    main()
