"""Container resource sampling for the balanced workers (``observe.sampler.ContainerSampler``):
peak RSS of the process tree, container core utilization and GPU memory, at most every 10 s,
into the startup facts and the shard audit row."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

from flora_deploy.observe import sampler as SM


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fakes(clock):
    cpu = {"s": 100.0}
    gpu_used = {"mib": 30000.0}

    def cgroup_cpu():
        return cpu["s"]

    def gpu():
        return [{"mem_used_mib": gpu_used["mib"], "mem_total_mib": 46068.0, "util_pct": 90.0}]

    def tree():
        return {"tree_rss": 3 << 30, "tree_rss_hwm": 5 << 30, "tree_procs": 2}

    return cpu, gpu_used, dict(gpu=gpu, tree=tree, cgroup_cpu=cgroup_cpu)


def test_interval_is_never_below_ten_seconds():
    assert SM.ContainerSampler(0.01).interval_s == SM.CONTAINER_MIN_INTERVAL_S
    assert SM.ContainerSampler(30).interval_s == 30


def test_summary_reports_peaks_core_use_and_gpu_memory():
    clock = Clock()
    cpu, gpu_used, kw = fakes(clock)
    s = SM.ContainerSampler(clock=clock, **kw)
    s.sample()
    clock.t, cpu["s"], gpu_used["mib"] = 10.0, 140.0, 41000.0  # 4 cores busy over 10 s
    s.sample()
    out = s.summary()
    assert out["container_cores_busy"] == 4.0
    assert out["container_core_util"] == round(4.0 / s.cores, 4)
    assert out["gpu_mem_used_max_mib"] == 41000.0 and out["gpu_mem_total_mib"] == 46068.0
    assert out["gpu_util_mean_pct"] == 90.0
    assert out["tree_rss_max_gib"] == 3.0 and out["tree_rss_hwm_gib"] == 5.0
    assert out["rss_peak_gib"] > 0 and out["cores"] == SM.available_cores()


def test_summary_since_covers_one_shard():
    clock = Clock()
    cpu, gpu_used, kw = fakes(clock)
    s = SM.ContainerSampler(clock=clock, **kw)
    s.sample()  # startup, GPU at 30 GB
    gpu_used["mib"] = 44000.0
    clock.t = 10.0
    s.sample()
    gpu_used["mib"] = 35000.0
    clock.t, cpu["s"] = 20.0, 120.0
    s.sample()
    shard = s.summary_since(20.0)
    assert shard["samples"] == 1 and shard["gpu_mem_used_max_mib"] == 35000.0
    assert s.summary()["gpu_mem_used_max_mib"] == 44000.0


def test_samples_are_capped():
    clock = Clock()
    _cpu, _gpu, kw = fakes(clock)
    s = SM.ContainerSampler(clock=clock, max_samples=8, **kw)
    for i in range(100):
        clock.t = float(i)
        s.sample()
    assert len(s.samples) <= 9 and s.samples[0]["t"] == 0.0


def test_gpu_memory_parses_nvidia_smi_and_tolerates_failure():
    def run(cmd, **kw):
        assert cmd[0] == "nvidia-smi"
        return SimpleNamespace(stdout="41000, 46068, 87\n")

    assert SM.gpu_memory(run=run) == [
        {"mem_used_mib": 41000.0, "mem_total_mib": 46068.0, "util_pct": 87.0}]

    def broken(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 5)

    assert SM.gpu_memory(run=broken) is None


def test_tree_rss_and_cgroup_cpu_are_best_effort():
    tree = SM.tree_rss_bytes()
    if os.path.isdir("/proc/self"):
        assert tree is not None and tree["tree_procs"] >= 1
    else:
        assert tree is None
    assert SM.cgroup_cpu_s(files=()) is None


def test_runner_records_resources_in_startup_facts_and_the_shard_audit(tmp_path, monkeypatch):
    from flora_deploy.workers_impl.segment import BalancedShardRunner, BalancedWorkerSettings

    clock = Clock()
    _cpu, _gpu, kw = fakes(clock)
    real = SM.ContainerSampler
    monkeypatch.setattr(SM, "ContainerSampler", lambda: real(clock=clock, **kw))
    job = SimpleNamespace(stage="caption", nv=None)
    runner = BalancedShardRunner(None, object(), job,  # type: ignore[arg-type]
                                 BalancedWorkerSettings(sample_resources=True))
    try:
        facts = runner.startup_facts()
        assert facts["batch_policy"]["kind"] == "caption"
        assert facts["batch_policy"]["infer_episodes"] == 32
        assert facts["resources"]["gpu_mem_used_max_mib"] == 30000.0
        row = runner._shard_audit(SimpleNamespace(shard_id="caption-0"), 0, 0.0)
        assert row["shard_audit"] and row["resources"]["tree_rss_hwm_gib"] == 5.0
        assert row["batch_policy"]["digest"] == facts["batch_policy"]["digest"]
    finally:
        runner.sampler._stop.set()
