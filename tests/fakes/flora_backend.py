"""A fake ``sdk.Backend`` (no Modal) and a "deployed app" harness for SDK / CLI tests.

``FakeBackend`` maps every store URI to a ``LocalStore`` under one directory, records spawns,
cancels and log requests, and keeps the orchestrator lease Dict in memory (``FakeDict``).
When given ``orchestrate``, a spawn of ``orchestrator_run`` runs it synchronously: the
:class:`DeployedApp` harness does that with the *real* deployed-side code path:
``sdk.deployed_deps`` (the ``FACTORY``) builds the deps from the run spec in the store, and the
real ``Orchestrator`` drives G2's ``LocalSpawner`` with the real worker runners and fake engines
(``flora_deploy.local``), so a submitted job really runs, end to end, with zero Modal calls.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from flora_deploy.jobs.stores import LocalStore
from tests.fakes.jobs import FakeDict

TEMPLATE = Path(__file__).resolve().parents[2] / "tests/fixtures/local_demo_exact.yaml"
RECEIPTS = {"captioner_bf16": "0" * 64}
JOB = "local-demo"


class FakeBackend:
    def __init__(self, root: Path, orchestrate: Callable[..., None] | None = None):
        self.root = Path(root)
        self.orchestrate = orchestrate
        self.stores: dict[str, LocalStore] = {}
        self.spawned: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.log_calls: list[dict[str, Any]] = []
        self.log_lines = ["line 1", "line 2"]
        self.kv = FakeDict()

    def store(self, uri: str, *, environment: str | None = None) -> LocalStore:
        if uri not in self.stores:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", uri)
            self.stores[uri] = LocalStore(self.root / slug)
        return self.stores[uri]

    def spawn(self, app_name: str, function: str, *args: Any, environment: str | None = None,
              **kwargs: Any) -> str:
        cid = f"fc-{len(self.spawned) + 1:04d}"
        self.spawned.append({"app": app_name, "function": function, "args": args,
                             "environment": environment, "call_id": cid})
        if self.orchestrate is not None and function == "orchestrator_run":
            self.orchestrate(app_name, *args)
        return cid

    def cancel_call(self, call_id: str, *, terminate: bool = True) -> None:
        self.cancelled.append(call_id)

    def lease_kv(self, environment: str | None = None) -> FakeDict:
        return self.kv

    def logs(self, app_name: str, *, environment: str | None = None, follow: bool = False,
             tail: int | None = None, function_call: str | None = None) -> Iterator[str]:
        self.log_calls.append({"app": app_name, "follow": follow, "tail": tail,
                               "function_call": function_call})
        yield from self.log_lines


def write_config(d: Path, *, stage_to: str = "volume://flora-data/demo", **pipeline: Any
                 ) -> Path:
    """The local demo template, staged to a Volume (so the Modal factory can plan it)."""
    data = yaml.safe_load(TEMPLATE.read_text())
    data["input"]["stage_to"] = stage_to
    data["pipeline"].update(pipeline)
    p = d / "flora.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


class DeployedApp:
    """One "deployed" config: plans the app (``plan_app``), writes its deployment record, indexes
    a synthetic dataset into the job store, and (``run=True``) runs spawned orchestrators."""

    def __init__(self, tmp: Path, *, run: bool = True, faults: str | None = None,
                 deploy: bool = True, index: bool = True, max_ticks: int = 20000):
        from flora_deploy.config.schema import load_config
        from flora_deploy.local import sample_dataset
        from flora_deploy.modal_app.factory import deployment_path, deployment_record, plan_app

        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.config_path = write_config(self.tmp)
        self.cfg = load_config(self.config_path)
        self.videos = sample_dataset(self.tmp / "videos")
        self.faults = faults
        self.max_ticks = max_ticks
        self.outcomes: list[str] = []
        self.backend = FakeBackend(self.tmp / "stores", self._orchestrate if run else None)
        self.plan = plan_app(self.cfg, derived_receipts=RECEIPTS)
        self.store_uri = self.plan.store.uri
        self.store = self.backend.store(self.store_uri)
        if deploy:
            rec = deployment_record(self.plan, app_id="ap-fake",
                                    deployed_utc="2026-10-21T00:00:00Z")
            self.store.put_bytes(deployment_path(self.plan.app_name), json.dumps(rec).encode())
        if index:
            self.index()

    def index(self) -> dict:
        from flora_deploy.jobs.indexing import IndexJob, open_dataset, run_index

        ijob = IndexJob.from_config(self.cfg, JOB)
        files = open_dataset(ijob.dataset, local_root=self.videos)
        self.media_root = Path(files.root).resolve()
        return run_index(ijob, self.store, files, log=lambda m: None)

    def env(self) -> dict[str, str]:
        return {"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(self.store.root),
                "FLORA_CONFIG_HASH": self.plan.config_hash, "FLORA_APP_NAME": self.plan.app_name}

    def _orchestrate(self, app_name: str, job_id: str, factory: str, generation: int = 0,
                     watch: bool = False) -> None:
        from flora_deploy.jobs.orchestrator import DictLease, Orchestrator
        from flora_deploy.local import LocalRunner
        from flora_deploy.modal_app.orchestrator_app import load_factory

        assert app_name == self.plan.app_name
        fn = load_factory(factory)
        deps = fn(job_id, self.env())  # sdk.deployed_deps, as the deployed app calls it
        work = self.tmp / f"work-{uuid.uuid4().hex[:6]}"
        runner = LocalRunner(self.cfg, config_path=self.config_path, store=deps.store,
                             job_id=job_id, faults=self.faults, work_dir=work)
        runner.settings = deps.settings
        runner.media_root = self.media_root
        orch = Orchestrator(settings=deps.settings, store=deps.store, spawner=runner.spawner,
                            lease=DictLease(self.backend.kv, f"orchestrator/{job_id}"),
                            manifests=deps.manifests, attempts=deps.attempts,
                            progress=deps.progress, budget=runner._budget(deps),
                            spawn_grace_s=5.0, holder=f"fake-orch-{generation}")
        assert orch.acquire()
        result = "running"
        try:
            import time

            for _ in range(self.max_ticks):
                result = orch.tick()
                if result != "running":
                    break
                time.sleep(0.01)
        finally:
            orch.release()
            runner.close()
            shutil.rmtree(work, ignore_errors=True)
        self.outcomes.append(result)
