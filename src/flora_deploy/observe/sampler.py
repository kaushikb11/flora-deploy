"""Resource sampling: RSS, process CPU, cgroup memory, ``/dev/shm`` (no psutil; reads /proc)."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

CGROUP_MEMORY_FILES = (
    Path("/sys/fs/cgroup/memory.current"),  # cgroup v2
    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),  # cgroup v1
)


def _proc_status(path: Path = Path("/proc/self/status")) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if parts and parts[0].isdigit():
            scale = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
            out[key] = int(parts[0]) * scale
    return out


def rss_bytes() -> dict[str, int]:
    """Current and peak RSS of this process (``VmRSS``/``VmHWM``; ru_maxrss fallback)."""
    status = _proc_status()
    if "VmRSS" in status:
        return {"rss": status["VmRSS"], "rss_peak": status.get("VmHWM", status["VmRSS"]),
                "threads": status.get("Threads", 0)}
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    return {"rss": peak, "rss_peak": peak, "threads": threading.active_count()}


def cgroup_memory_bytes(files=CGROUP_MEMORY_FILES) -> int | None:
    """Container memory incl. page cache (what the memory limit is enforced against)."""
    for f in files:
        try:
            return int(Path(f).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def shm_usage(path: str | Path = "/dev/shm") -> dict | None:
    try:
        du = shutil.disk_usage(path)
    except OSError:
        return None
    return {"total": du.total, "used": du.used}


def process_cpu_s() -> float:
    t = os.times()
    return t.user + t.system


class Sampler:
    """Background thread sampling every ``interval_s``; ``stop()`` returns ``summary()``.

    The summary reports RSS (start/end/max), cgroup memory max, ``/dev/shm`` max used, and the
    process CPU used over the window as ``cpu_s`` and ``cores_busy`` (= cpu_s / wall_s)."""

    def __init__(self, interval_s: float = 1.0, shm_path: str | Path = "/dev/shm",
                 clock=time.perf_counter):
        self.interval_s = interval_s
        self.shm_path = shm_path
        self.clock = clock
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> dict:
        s = {"t": self.clock(), "cpu_s": process_cpu_s(), **rss_bytes()}
        cg = cgroup_memory_bytes()
        if cg is not None:
            s["cgroup_mem"] = cg
        shm = shm_usage(self.shm_path)
        if shm is not None:
            s["shm_used"], s["shm_total"] = shm["used"], shm["total"]
        try:
            s["load1"] = os.getloadavg()[0]
        except OSError:
            pass
        self.samples.append(s)
        return s

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def start(self) -> Sampler:
        self.sample()
        self._thread = threading.Thread(target=self._run, name="flora-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self.interval_s + 1)
        self.sample()
        return self.summary()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        first, last = self.samples[0], self.samples[-1]
        wall = last["t"] - first["t"]
        cpu = last["cpu_s"] - first["cpu_s"]

        def mx(key):
            vals = [s[key] for s in self.samples if key in s]
            return max(vals) if vals else None

        gib = 2**30
        out = {
            "samples": len(self.samples),
            "wall_s": round(wall, 3),
            "cpu_s": round(cpu, 3),
            "cores_busy": round(cpu / wall, 3) if wall > 0 else None,
            "rss_start_gib": round(first["rss"] / gib, 3),
            "rss_end_gib": round(last["rss"] / gib, 3),
            "rss_max_gib": round(mx("rss") / gib, 3),
            "rss_peak_gib": round(mx("rss_peak") / gib, 3),
            "threads_max": mx("threads"),
        }
        if mx("cgroup_mem") is not None:
            out["cgroup_mem_max_gib"] = round(mx("cgroup_mem") / gib, 3)
        if mx("shm_used") is not None:
            out["shm_used_max_mib"] = round(mx("shm_used") / 2**20, 3)
            out["shm_total_gib"] = round(last.get("shm_total", 0) / gib, 3)
        if mx("load1") is not None:
            out["load1_max"] = round(mx("load1"), 2)
        return out
