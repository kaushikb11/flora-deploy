"""observe/: timers, patcher, timed lock, sampler, aggregation, event emission."""

import asyncio
import io
import json
import threading
import time
from types import SimpleNamespace

import pytest

from flora_deploy import observe
from flora_deploy.observe import (
    Patcher,
    Recorder,
    Sampler,
    TimedLock,
    call_table,
    distribution,
    merge_tables,
    recording,
    timed,
    token_stats,
)


def busy(seconds):
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


def test_timed_records_wall_self_cpu_and_nesting():
    ns = SimpleNamespace()
    ns.inner = timed(lambda: busy(0.02), "inner")

    def outer():
        ns.inner()
        time.sleep(0.03)
        return 7

    ns.outer = timed(outer, "outer")
    rec = Recorder(episode="e")
    with recording(rec):
        assert ns.outer() == 7
    table = call_table(rec.events)
    assert table["inner"]["calls"] == table["outer"]["calls"] == 1
    assert table["inner"]["cpu_s"] >= 0.015
    # outer's self time excludes inner's wall; outer's CPU is its own thread's CPU
    assert table["outer"]["wall_s"] >= table["inner"]["wall_s"] + 0.025
    assert abs(table["outer"]["self_s"] - (table["outer"]["wall_s"] - table["inner"]["wall_s"])
               ) < 1e-3
    assert all(e["episode"] == "e" for e in rec.events)


def test_timed_without_recorder_is_transparent_and_records_errors():
    calls = []
    f = timed(lambda x: calls.append(x) or x * 2, "f")
    assert f(3) == 6 and calls == [3]

    def boom():
        raise KeyError("x")

    g = timed(boom, lambda: "dyn")
    rec = Recorder()
    with recording(rec), pytest.raises(KeyError):
        g()
    assert call_table(rec.events)["dyn"]["errors"] == 1


def test_recorder_follows_to_thread_but_not_bare_threads():
    rec = Recorder()
    f = timed(lambda: None, "f")

    async def main():
        await asyncio.to_thread(f)

    with recording(rec):
        asyncio.run(main())
        t = threading.Thread(target=f)
        t.start()
        t.join()
    assert call_table(rec.events)["f"]["calls"] == 1


def test_patcher_restores_modules_and_classes():
    class Base:
        def m(self):
            return "base"

    class Child(Base):
        pass

    mod = SimpleNamespace(fn=lambda: 1)
    with Patcher() as p:
        p.time(mod, "fn", "mod.fn")
        p.time(Child, "m", "child.m")
        assert mod.fn() == 1 and Child().m() == "base"
        assert "m" in vars(Child)
        with pytest.raises(RuntimeError, match="already patched"):
            p.time(mod, "fn", "again")
        assert p.patched == ["SimpleNamespace.fn", "Child.m"] or len(p.patched) == 2
    assert "m" not in vars(Child)  # inherited attribute: override removed, not copied
    assert not hasattr(mod.fn, "__wrapped__")


def test_timed_lock_serializes_and_measures():
    lock = TimedLock("gpu")
    active, peak = [0], [0]
    guard = threading.Lock()
    rec = Recorder()

    def work():
        with recording(rec):
            for _ in range(5):
                with lock.held("gen"):
                    with guard:
                        active[0] += 1
                        peak[0] = max(peak[0], active[0])
                    time.sleep(0.002)
                    with guard:
                        active[0] -= 1

    t0 = time.perf_counter()
    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()
    assert peak[0] == 1
    s = lock.stats()
    assert s["acquisitions"] == 20 and s["contended"] > 0 and s["wait_s"] > 0
    assert 0.3 < lock.busy_fraction(t0, t1) <= 1.0
    assert call_table(rec.events)["lock:gpu"]["calls"] == 20
    with lock.held(), pytest.raises(RuntimeError, match="reentrant"):
        with lock.held():
            pass
    lock.reset()
    assert lock.stats()["acquisitions"] == 0 and lock.busy_fraction(t0, t0) == 0.0


def test_sampler_summary_fields():
    s = Sampler(interval_s=0.01)
    s.start()
    busy(0.05)
    out = s.stop()
    assert out["samples"] >= 2 and out["rss_max_gib"] > 0
    assert out["cpu_s"] > 0 and out["cores_busy"] > 0
    assert observe.rss_bytes()["rss"] > 0
    assert observe.shm_usage("/definitely/missing") is None
    assert observe.cgroup_memory_bytes(files=("/definitely/missing",)) is None


def test_distribution_and_tokens():
    d = distribution([5, 1, 3, 2, 4])
    assert d == {"count": 5, "sum": 15.0, "mean": 3.0, "p50": 3.0, "p90": 5.0, "max": 5.0}
    assert distribution([])["count"] == 0
    events = [{"event": "generation_end", "input_tokens": 100, "generated_tokens": 10,
               "output_limit": 512, "elapsed_s": 1.0},
              {"event": "generation_start", "input_tokens": 999},
              {"event": "generation_end", "input_tokens": 300, "generated_tokens": 30,
               "output_limit": 1024, "elapsed_s": 2.0}]
    t = token_stats(events)
    assert t["prompt_tokens"]["sum"] == 400 and t["output_tokens"]["max"] == 30
    assert t["output_limit"] == [512, 1024]


def test_merge_tables():
    a = {"x": {"calls": 1, "wall_s": 1.0, "max_wall_s": 1.0}}
    b = {"x": {"calls": 2, "wall_s": 0.5, "max_wall_s": 0.4}, "y": {"calls": 1}}
    m = merge_tables([a, b])
    assert m["x"] == {"calls": 3, "wall_s": 1.5, "max_wall_s": 1.0} and m["y"]["calls"] == 1


def test_emit_json():
    buf = io.StringIO()
    observe.emit_json({"kind": "batch", "b": 1, "a": [1]}, stream=buf)
    line = buf.getvalue()
    assert line.startswith(observe.EVENT_PREFIX) and line.endswith("\n")
    assert json.loads(line[len(observe.EVENT_PREFIX):]) == {"a": [1], "b": 1, "kind": "batch"}
    with pytest.raises(ValueError):
        observe.emit_json({"x": float("nan")}, stream=buf)
