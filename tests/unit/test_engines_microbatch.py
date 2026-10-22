"""MicroBatcher semantics (D14): round triggers, liveness, ordering, isolation, bounded memory."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.engines.microbatch import BatcherClosed, EngineBatchError, MicroBatcher

SP = Sampling(max_tokens=8)


def req(key: str, nbytes: int = 0) -> VllmRequest:
    videos = ((np.zeros((nbytes,), np.uint8), {}),) if nbytes else ()
    return VllmRequest(prompt=f"<{key}>", videos=videos, key=key)


class EchoEngine:
    """Answers every request with its key encoded as token ids; records each batch."""

    name = "echo"

    def __init__(self, delay: float = 0.0, fail_batch: int | None = None,
                 interrupt_on: str | None = None, error_keys=()):
        self.batches: list[list[str]] = []
        self.delay = delay
        self.fail_batch = fail_batch
        self.interrupt_on = interrupt_on
        self.error_keys = set(error_keys)
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def generate_batch(self, requests, samplings):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            keys = [r.key for r in requests]
            self.batches.append(keys)
            if self.interrupt_on in keys:
                raise KeyboardInterrupt("SIGINT")
            if self.fail_batch is not None and len(self.batches) == self.fail_batch:
                raise RuntimeError("engine fell over")
            if self.delay:
                time.sleep(self.delay)
            return [Completion(key=k, error="bad request") if k in self.error_keys else
                    Completion(key=k, token_ids=tuple(ord(c) for c in k), finish_reason="stop",
                               prompt_tokens=len(k)) for k in keys]
        finally:
            with self._lock:
                self.concurrent -= 1

    def audit(self):
        return {"batches": len(self.batches)}


def decode(c: Completion) -> str:
    return "".join(chr(t) for t in c.token_ids)


async def episode(mb, name: str, n: int, *, before: float = 0.0, rounds: int = 1):
    out = []
    async with mb.episode():
        for r in range(rounds):
            if before:
                await asyncio.sleep(before)
            res = await mb.generate_many([req(f"{name}.{r}.{i}") for i in range(n)], [SP] * n)
            out.append([decode(c) for c in res])
    return out


def test_all_blocked_forms_one_round_across_episodes():
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            res = await asyncio.gather(episode(mb, "a", 2), episode(mb, "b", 3),
                                       episode(mb, "c", 1))
            return res, mb.stats

    res, stats = asyncio.run(main())
    assert eng.batches == [["a.0.0", "a.0.1", "b.0.0", "b.0.1", "b.0.2", "c.0.0"]]
    assert stats.triggers == {"all_blocked": 1}
    assert res == [[["a.0.0", "a.0.1"]], [["b.0.0", "b.0.1", "b.0.2"]], [["c.0.0"]]]


def test_round_waits_for_live_but_busy_episode():
    """An episode still preparing (not blocked) holds the round until it submits."""
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            await asyncio.gather(episode(mb, "fast", 1), episode(mb, "slow", 1, before=0.2))

    t0 = time.perf_counter()
    asyncio.run(main())
    assert time.perf_counter() - t0 >= 0.2
    assert eng.batches == [["fast.0.0", "slow.0.0"]]


def test_timeout_bounds_waiting_for_a_slow_episode():
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=0.05) as mb:
            await asyncio.gather(episode(mb, "fast", 1), episode(mb, "slow", 1, before=0.5))
            return mb.stats

    stats = asyncio.run(main())
    assert eng.batches == [["fast.0.0"], ["slow.0.0"]]
    assert stats.triggers["timeout"] == 1


def test_finished_episode_releases_the_round():
    eng = EchoEngine()

    async def idle(mb):
        async with mb.episode():
            await asyncio.sleep(0.1)  # leaves without submitting

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            await asyncio.gather(episode(mb, "a", 2), idle(mb))
            return mb.stats

    stats = asyncio.run(main())
    assert eng.batches == [["a.0.0", "a.0.1"]]
    assert stats.triggers == {"all_blocked": 1}


def test_size_threshold_and_fifo_order():
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_batch=2, max_wait_s=30) as mb:
            return await episode(mb, "a", 5)

    res = asyncio.run(main())
    assert [len(b) for b in eng.batches] == [2, 2, 1]
    assert sum(eng.batches, []) == [f"a.0.{i}" for i in range(5)]
    assert res == [[f"a.0.{i}" for i in range(5)]]


def test_one_generate_at_a_time_and_next_round_overlaps():
    eng = EchoEngine(delay=0.05)

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            return await asyncio.gather(*(episode(mb, f"e{i}", 2, rounds=3) for i in range(4)))

    res = asyncio.run(main())
    assert eng.max_concurrent == 1
    assert len(eng.batches) == 3  # every round holds all 4 episodes
    assert all(len(b) == 8 for b in eng.batches)
    assert res[2] == [[f"e2.{r}.0", f"e2.{r}.1"] for r in range(3)]


def test_retry_rounds_only_for_episodes_that_need_them():
    """Like NVIDIA's segmenter: some episodes ask again (retry), others finish."""
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            await asyncio.gather(episode(mb, "a", 3, rounds=1), episode(mb, "b", 2, rounds=3),
                                 episode(mb, "c", 1, rounds=2))

    asyncio.run(main())
    assert [sorted(b) for b in eng.batches] == [
        sorted(["a.0.0", "a.0.1", "a.0.2", "b.0.0", "b.0.1", "c.0.0"]),
        sorted(["b.1.0", "b.1.1", "c.1.0"]),
        ["b.2.0", "b.2.1"],
    ]


def test_batch_failure_is_isolated_to_that_batch():
    eng = EchoEngine(fail_batch=1)

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            first = await asyncio.gather(episode(mb, "a", 1), return_exceptions=True)
            second = await episode(mb, "b", 1)
            return first, second, mb.stats

    first, second, stats = asyncio.run(main())
    assert isinstance(first[0], EngineBatchError)
    assert second == [["b.0.0"]]
    assert stats.failed_batches == 1


def test_per_request_errors_pass_through():
    eng = EchoEngine(error_keys={"a.0.1"})

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb, mb.episode():
            return await mb.generate_many([req("a.0.0"), req("a.0.1")], [SP, SP])

    a, b = asyncio.run(main())
    assert a.ok and decode(a) == "a.0.0"
    assert not b.ok and b.error == "bad request"


def test_keyboard_interrupt_ends_the_loop():
    eng = EchoEngine(interrupt_on="b.0.0")

    async def main():
        async with MicroBatcher(eng, max_batch=1, max_wait_s=30) as mb:
            await asyncio.gather(episode(mb, "a", 1), episode(mb, "b", 1))

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(main())


def test_bounded_pending_memory():
    eng = EchoEngine(delay=0.02)
    cap = 3000

    async def main():
        async with MicroBatcher(eng, max_wait_s=30, max_batch_bytes=2000,
                                max_pending_bytes=cap) as mb:

            async def ep(name):
                async with mb.episode():
                    return await mb.generate_many([req(f"{name}.{i}", 1000) for i in range(2)],
                                                  [SP, SP])

            await asyncio.gather(*(ep(f"e{i}") for i in range(6)))
            return mb.stats

    stats = asyncio.run(main())
    assert stats.peak_pending_bytes <= cap
    assert stats.requests == 12
    assert max(len(b) for b in eng.batches) <= 2


def test_oversized_single_request_is_admitted():
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=30, max_batch_bytes=10,
                                max_pending_bytes=10) as mb, mb.episode():
            return await mb.generate(req("big", 100), SP)

    assert decode(asyncio.run(main())) == "big"


def test_calls_outside_an_episode_never_hold_a_round():
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            return await mb.generate(req("x"), SP)

    assert decode(asyncio.run(main())) == "x"


def test_closed_batcher_refuses_and_fails_pending():
    eng = EchoEngine()

    async def main():
        mb = MicroBatcher(eng, max_wait_s=30)
        with pytest.raises(BatcherClosed):
            await mb.generate(req("x"), SP)  # never started
        await mb.start()
        await mb.close()
        with pytest.raises(BatcherClosed):
            await mb.generate(req("y"), SP)

    asyncio.run(main())


def test_cancelled_episode_does_not_block_others():
    eng = EchoEngine()

    async def main():
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            slow = asyncio.create_task(episode(mb, "gone", 1, before=0.05))
            await asyncio.sleep(0.01)
            slow.cancel()
            return await episode(mb, "a", 1)

    assert asyncio.run(main()) == [["a.0.0"]]



class SlowEngine:
    """A long ``llm.generate``: blocks until released (or ``hold`` seconds pass)."""

    name = "slow"

    def __init__(self, hold: float = 5.0, on_start=None):
        self.hold, self.on_start = hold, on_start
        self.release = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()

    def generate_batch(self, requests, samplings):
        self.started.set()
        if self.on_start is not None:
            self.on_start()
        self.release.wait(self.hold)
        self.finished.set()
        return [Completion(key=r.key, token_ids=(1,), finish_reason="stop", prompt_tokens=1)
                for r in requests]


def test_close_on_cancel_does_not_wait_for_the_inflight_generate():
    """Review 6 M4: cancelling (SIGINT -> asyncio cancel) must not block on the in-flight
    ``generate``; the episode's future fails, the loop ends at once."""
    eng = SlowEngine(hold=5.0)

    async def main():
        mb = await MicroBatcher(eng, max_wait_s=0.01).start()
        task = asyncio.ensure_future(mb.generate(req("a"), SP))
        await asyncio.to_thread(eng.started.wait, 2)
        task.cancel()
        t0 = time.perf_counter()
        await mb.close()
        return time.perf_counter() - t0

    took = asyncio.run(main())
    assert took < 1.0, f"close() blocked {took:.1f}s on the in-flight generate"
    eng.release.set()


def test_close_does_not_block_the_event_loop():
    """Even a waiting close leaves the loop free (the wait runs off the loop)."""
    eng = SlowEngine(hold=0.6)
    ticks = []

    async def ticker():
        while True:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.05)

    async def main():
        mb = await MicroBatcher(eng, max_wait_s=0.01).start()
        fut = asyncio.ensure_future(mb.generate(req("a"), SP))
        await asyncio.to_thread(eng.started.wait, 2)
        t = asyncio.ensure_future(ticker())
        await asyncio.sleep(0.1)
        await mb.close()
        await asyncio.sleep(0.2)
        t.cancel()
        fut.cancel()

    asyncio.run(main())
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    assert gaps and max(gaps) < 0.3, f"event loop stalled {max(gaps):.2f}s in close()"


def test_abandoned_generate_never_overlaps_the_next_batcher():
    """An orphaned in-flight generate holds the engine: a later batcher on the same engine
    waits for it instead of calling ``llm.generate`` concurrently (vLLM is not thread-safe)."""
    eng = EchoEngine(delay=0.4)

    async def first():
        mb = await MicroBatcher(eng, max_wait_s=0.01).start()
        task = asyncio.ensure_future(mb.generate(req("a"), SP))
        await asyncio.sleep(0.1)
        task.cancel()
        await mb.close()

    async def second():
        async with MicroBatcher(eng, max_wait_s=0.01) as mb:
            return await mb.generate(req("b"), SP)

    asyncio.run(first())
    asyncio.run(second())
    assert eng.max_concurrent == 1


# --- two-phase episodes (WP1a: workers_impl.segment.EpisodeGates) ----------------------------


async def phased(mb, gates, name: str, n: int, *, prep=None, infer_hold=None, log=None):
    """One episode in two phases: ``prep`` (an awaitable factory) under ``gates.prep()``,
    then ``n`` calls under ``gates.infer(mb)``."""
    with gates.slot():
        async with gates.prep():
            if log is not None:
                log.append(("prep", name, gates.preparing, gates.inferring))
            if prep is not None:
                await prep()
        async with gates.infer(mb):
            if log is not None:
                log.append(("infer", name, gates.preparing, gates.inferring))
            if infer_hold is not None:
                await infer_hold()
            res = await mb.generate_many([req(f"{name}.{i}") for i in range(n)], [SP] * n)
            return [decode(c) for c in res]


def test_preparing_episode_does_not_hold_a_round_back():
    """A long CPU preparation (NVIDIA's probe) runs outside the batcher's episode: the round of
    the episode that is already inferring fires at once (all blocked), not after max_wait_s."""
    from flora_deploy.workers_impl.segment import EpisodeGates

    eng = EchoEngine()

    async def main():
        gates = EpisodeGates(prefetch=4, infer=4)
        release = asyncio.Event()
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            slow = asyncio.ensure_future(phased(mb, gates, "slow", 1, prep=release.wait))
            t0 = time.perf_counter()
            fast = await phased(mb, gates, "fast", 2)
            waited = time.perf_counter() - t0
            assert not slow.done() and mb.live == 0  # slow is preparing, not live
            release.set()
            return fast, await slow, waited, dict(mb.stats.triggers)

    fast, slow, waited, triggers = asyncio.run(main())
    assert fast == ["fast.0", "fast.1"] and slow == ["slow.0"]
    assert waited < 5, waited  # not the 30 s max_wait_s
    assert eng.batches == [["fast.0", "fast.1"], ["slow.0"]]
    assert triggers == {"all_blocked": 2}


def test_prefetch_is_bounded():
    """At most ``prefetch`` episodes hold a prep slot (preparing, or prepared and waiting for
    inference), at most ``infer`` are inferring; all of them finish."""
    from flora_deploy.workers_impl.segment import EpisodeGates

    eng = EchoEngine()
    log: list = []

    async def main():
        gates = EpisodeGates(prefetch=2, infer=1)

        async def prep():
            await asyncio.sleep(0.001)

        async def hold():
            await asyncio.sleep(0.01)

        async with MicroBatcher(eng, max_wait_s=30) as mb:
            outs = await asyncio.gather(*(phased(mb, gates, f"e{i}", 1, prep=prep,
                                                 infer_hold=hold, log=log)
                                          for i in range(8)))
        return outs, gates

    outs, gates = asyncio.run(main())
    assert outs == [[f"e{i}.0"] for i in range(8)]
    assert max(p for _w, _n, p, _i in log) <= 2
    assert max(i for _w, _n, _p, i in log) <= 1
    assert gates.as_json() == {"prefetch": 2, "infer": 1, "peak_preparing": 2,
                               "peak_inferring": 1, "peak_admitted": 3}
    assert gates.preparing == gates.inferring == 0  # everything released


def test_all_blocked_fires_when_every_inferring_episode_is_blocked():
    """Three inferring episodes submit while two more are still preparing: one round of three,
    by the all-blocked trigger; the preparing ones form the next round."""
    from flora_deploy.workers_impl.segment import EpisodeGates

    eng = EchoEngine()

    async def main():
        gates = EpisodeGates(prefetch=8, infer=8)
        release = asyncio.Event()
        async with MicroBatcher(eng, max_wait_s=30) as mb:
            late = [asyncio.ensure_future(phased(mb, gates, f"late{i}", 1, prep=release.wait))
                    for i in range(2)]
            early = await asyncio.gather(*(phased(mb, gates, f"early{i}", 1)
                                           for i in range(3)))
            release.set()
            return early, await asyncio.gather(*late), mb.stats

    early, late, stats = asyncio.run(main())
    assert early == [["early0.0"], ["early1.0"], ["early2.0"]]
    assert late == [["late0.0"], ["late1.0"]]
    assert eng.batches == [["early0.0", "early1.0", "early2.0"], ["late0.0", "late1.0"]]
    assert stats.triggers == {"all_blocked": 2}


def test_failed_preparation_releases_its_slot():
    from flora_deploy.workers_impl.segment import EpisodeGates

    eng = EchoEngine()

    async def main():
        gates = EpisodeGates(prefetch=1, infer=1)

        async def boom():
            raise ValueError("probe failed")

        async with MicroBatcher(eng, max_wait_s=30) as mb:
            with pytest.raises(ValueError):
                await phased(mb, gates, "bad", 1, prep=boom)
            ok = await phased(mb, gates, "ok", 1)
        return ok, gates

    ok, gates = asyncio.run(main())
    assert ok == ["ok.0"] and gates.preparing == 0
