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


# ---------------------------------------------------------------------------------------------
# Container resources of the balanced workers: what the pools' CPU / memory requests can be cut
# to. Read-only and output-neutral (/proc, cgroup files, ``nvidia-smi``); sampled at most every
# ``CONTAINER_MIN_INTERVAL_S``.
# ---------------------------------------------------------------------------------------------

CONTAINER_MIN_INTERVAL_S = 10.0
CGROUP_CPU_FILES = (
    (Path("/sys/fs/cgroup/cpu.stat"), "usage_usec", 1e-6),  # cgroup v2
    (Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"), None, 1e-9),  # cgroup v1 (ns)
)


def available_cores() -> int:
    """Cores this process may run on (affinity, then the cgroup quota; ``os.cpu_count``
    fallback)."""
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return max(1, int(n))


def _children(pid: int) -> list[int]:
    out: list[int] = []
    try:
        for task in Path(f"/proc/{pid}/task").iterdir():
            out.extend(int(c) for c in (task / "children").read_text().split())
    except (OSError, ValueError):
        pass
    return out


def tree_rss_bytes(root: int | None = None) -> dict[str, int] | None:
    """RSS of this process and all its descendants (vLLM's engine-core process included):
    ``tree_rss`` (sum of VmRSS) and ``tree_rss_hwm`` (sum of each process's VmHWM, an upper
    bound of the tree's peak). ``None`` without ``/proc``."""
    root = os.getpid() if root is None else root
    if not Path(f"/proc/{root}/status").is_file():
        return None
    rss = hwm = procs = 0
    todo, seen = [root], set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        st = _proc_status(Path(f"/proc/{pid}/status"))
        if "VmRSS" in st:
            rss += st["VmRSS"]
            hwm += st.get("VmHWM", st["VmRSS"])
            procs += 1
        todo.extend(_children(pid))
    return {"tree_rss": rss, "tree_rss_hwm": hwm, "tree_procs": procs}


def cgroup_cpu_s(files=CGROUP_CPU_FILES) -> float | None:
    """CPU seconds used by the whole container (every process), from the cgroup."""
    for path, key, scale in files:
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        try:
            if key is None:
                return int(text.strip()) * scale
            for line in text.splitlines():
                k, _, v = line.partition(" ")
                if k == key:
                    return int(v) * scale
        except ValueError:
            continue
    return None


def gpu_memory(timeout_s: float = 5.0, run=None) -> list[dict] | None:
    """Per GPU: ``memory.used`` / ``memory.total`` (MiB) and ``utilization.gpu`` (%), from
    ``nvidia-smi``. It counts every process on the GPU (vLLM's engine core included), which
    ``torch.cuda.max_memory_allocated`` in this process would not. ``None`` without it."""
    import subprocess

    if run is None:
        if shutil.which("nvidia-smi") is None:
            return None
        run = subprocess.run
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                   "--format=csv,noheader,nounits"], capture_output=True, text=True,
                  timeout=timeout_s, check=True).stdout
    except Exception:  # noqa: BLE001 - diagnostics only
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            used, total, util = (float(p) for p in parts[:3])
        except ValueError:
            continue
        gpus.append({"mem_used_mib": used, "mem_total_mib": total, "util_pct": util})
    return gpus


class ContainerSampler(Sampler):
    """:class:`Sampler` plus the container view: process-tree RSS, container CPU from the
    cgroup, GPU memory and utilization from ``nvidia-smi``; never more often than every
    ``CONTAINER_MIN_INTERVAL_S`` seconds in its thread. ``summary()`` adds ``cores``,
    ``tree_rss_max_gib``, ``tree_rss_hwm_gib``, ``container_cores_busy``,
    ``container_core_util``, ``gpu_mem_used_max_mib``, ``gpu_mem_total_mib`` and
    ``gpu_util_mean_pct`` / ``gpu_util_max_pct``. Samples are capped (``max_samples``): the
    oldest are thinned, the first is kept."""

    def __init__(self, interval_s: float = CONTAINER_MIN_INTERVAL_S, *, gpu=gpu_memory,
                 tree=tree_rss_bytes, cgroup_cpu=cgroup_cpu_s, max_samples: int = 4096,
                 **kw):
        super().__init__(max(float(interval_s), CONTAINER_MIN_INTERVAL_S), **kw)
        self._gpu, self._tree, self._cgroup_cpu = gpu, tree, cgroup_cpu
        self.max_samples = max(4, int(max_samples))
        self.cores = available_cores()

    def sample(self) -> dict:
        s = super().sample()
        tree = self._tree()
        if tree:
            s.update(tree)
        cpu = self._cgroup_cpu()
        if cpu is not None:
            s["cgroup_cpu_s"] = cpu
        gpus = self._gpu() if self._gpu is not None else None
        if gpus:
            s["gpus"] = gpus
        if len(self.samples) > self.max_samples:  # keep the first, thin the rest
            self.samples[1:] = self.samples[1::2]
        return s

    def summary(self) -> dict:
        out = super().summary()
        if not self.samples:
            return out
        gib = 2**30
        out["cores"] = self.cores
        out["interval_s"] = self.interval_s
        trees = [s for s in self.samples if "tree_rss" in s]
        if trees:
            out["tree_rss_max_gib"] = round(max(s["tree_rss"] for s in trees) / gib, 3)
            out["tree_rss_hwm_gib"] = round(max(s["tree_rss_hwm"] for s in trees) / gib, 3)
            out["tree_procs_max"] = max(s["tree_procs"] for s in trees)
        cpus = [s for s in self.samples if "cgroup_cpu_s" in s]
        if len(cpus) >= 2 and cpus[-1]["t"] > cpus[0]["t"]:
            busy = (cpus[-1]["cgroup_cpu_s"] - cpus[0]["cgroup_cpu_s"]) / (
                cpus[-1]["t"] - cpus[0]["t"])
            out["container_cores_busy"] = round(busy, 3)
            out["container_core_util"] = round(busy / self.cores, 4)
        gpus = [g for s in self.samples for g in s.get("gpus", ())]
        if gpus:
            out["gpu_mem_used_max_mib"] = max(g["mem_used_mib"] for g in gpus)
            out["gpu_mem_total_mib"] = max(g["mem_total_mib"] for g in gpus)
            out["gpu_util_max_pct"] = max(g["util_pct"] for g in gpus)
            out["gpu_util_mean_pct"] = round(sum(g["util_pct"] for g in gpus) / len(gpus), 1)
            out["gpu_samples"] = len(gpus)
        return out

    def summary_since(self, t: float) -> dict:
        """``summary()`` over the samples taken at or after clock time ``t`` (one shard)."""
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view.samples = [s for s in list(self.samples) if s["t"] >= t]
        return view.summary()

    def snapshot(self) -> dict:
        """Take one sample now and summarize. ``rss_peak_gib`` / ``tree_rss_hwm_gib`` cover
        the time since each process started (the engine build included)."""
        self.sample()
        return self.summary()
