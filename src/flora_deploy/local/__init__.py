"""Local mode (G2): the real job system end to end in one process, no Modal, no GPU.

* :func:`run_local` / :class:`LocalRunner` (``local.runner``): index → plan → orchestrate →
  compose into a ``LocalStore`` directory (``flora run --local --fake-models``);
* :class:`LocalSpawner` (``local.spawner``): the orchestrator's ``Spawner`` on a thread pool with
  Modal's call semantics (pending/success/failed/expired/cancelled, retries, container ids);
* fake engines (``local.engines``) plugged into the real ``SegcapShardRunner`` /
  ``RewardShardRunner`` at their model boundary;
* fault injection (``local.faults``): preempt, OOM poison, Python failures, infra errors;
* :func:`sample_dataset` (``local.sample``): a synthetic ``mp4_dir`` dataset for the demo.

Nothing here changes outputs of real runs (``config/logic.py``: non-output modules).
"""

from flora_deploy.local.faults import Fault, FaultPlan, parse_faults
from flora_deploy.local.runner import (
    LocalRunError,
    LocalRunner,
    LocalRunResult,
    completion_state,
    run_local,
    summary_lines,
)
from flora_deploy.local.sample import sample_dataset
from flora_deploy.local.spawner import FileKV, LocalSpawner

__all__ = [
    "Fault",
    "FaultPlan",
    "FileKV",
    "LocalRunError",
    "LocalRunResult",
    "LocalRunner",
    "LocalSpawner",
    "completion_state",
    "parse_faults",
    "run_local",
    "sample_dataset",
    "summary_lines",
]
