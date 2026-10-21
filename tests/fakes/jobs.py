"""Small fakes for the orchestrator's protocols (manifest, attempts, progress, budget, lease KV)
and a fake shard worker that behaves like ``ShardWorker`` (marker -> skip done -> parts -> receipt).

The P1a-2 implementations replace ``FakeManifests``/``FakeAttempts``/``FakeProgress`` via adapters;
these keep only what the orchestrator needs. Store-backed parts (markers, parts, receipts, errors,
quarantine records) go through ``FakeStore`` so its listing lag applies.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from flora_deploy.jobs.orchestrator import (
    BudgetDecision,
    DictLease,
    Orchestrator,
    RunOutcome,
    utc,
    write_control,
)
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.types import (
    AttemptMarker,
    ControlRecord,
    EpisodeError,
    Manifest,
    Shard,
    ShardReceipt,
)
from tests.fakes.modal import FakeSpawner, FakeStore, Preempted

JOB = "job1"
CFG = "cfg1"


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeDict:
    """``modal.Dict`` subset: atomic ``put(skip_if_exists=True)``, ``get``, ``contains``."""

    def __init__(self):
        self.data: dict[Any, Any] = {}

    def put(self, key, value, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = value
        return True

    def get(self, key, default=None):
        return self.data.get(key, default)

    def contains(self, key) -> bool:
        return key in self.data


class FakeManifests:
    """In-memory (durable for the test's lifetime) manifest + replans."""

    def __init__(self, manifest: Manifest):
        self.base = manifest
        self.added: dict[str, Shard] = {}

    def load(self) -> Manifest:
        return Manifest(self.base.job_id, self.base.config_hash, self.base.episodes,
                        (*self.base.shards, *self.added.values()), self.base.created_utc)

    def replan(self, parent: Shard, children: Sequence[Shard]) -> None:
        for c in children:
            assert c.bisected_from == parent.shard_id
            self.added.setdefault(c.shard_id, c)

    def shard(self, shard_id: str) -> Shard:
        return next(s for s in self.load().shards if s.shard_id == shard_id)


class FakeAttempts:
    def __init__(self, store: FakeStore, cfg: str = CFG, job: str = JOB):
        self.store, self.cfg, self.paths = store, cfg, JobPaths(job)
        self._q: dict[str, ControlRecord] = {}
        self.poison_records: list[str] = []

    def _prefix(self, s: Shard) -> str:
        return self.paths.attempt(self.cfg, s.kind, s.shard_id, "x").rsplit("/", 1)[0] + "/"

    def starts(self, shard: Shard) -> int:
        return sum(1 for _ in self.store.list(self._prefix(shard)))

    def quarantined(self) -> Mapping[str, ControlRecord]:
        return dict(self._q)

    def quarantine(self, shard: Shard, reason: str, detail: Mapping[str, Any]) -> None:
        rec = ControlRecord("quarantine", shard.shard_id, reason, f"q-{len(self._q)}",
                            dict(detail))
        self._q.setdefault(shard.shard_id, rec)
        self.store.put_bytes(f"{self.paths.root}/control/quarantine/{shard.shard_id}.json",
                             json.dumps(asdict(rec)).encode())

    def bisect(self, shard: Shard, done: set[str]) -> tuple[Shard, ...]:
        todo = [u for u in shard.episodes if u not in done]
        per = shard.est_gpu_seconds / max(len(shard.episodes), 1)
        return tuple(Shard(f"{shard.kind}-ep-{u}", shard.kind, (u,), per,
                           depends_on=shard.depends_on, bisected_from=shard.shard_id)
                     for u in todo)

    def record_poison(self, shard: Shard, diagnostics: Mapping[str, Any]) -> None:
        (uid,) = shard.episodes
        err = EpisodeError(uid, "segment", shard.shard_id, "orchestrator", "poison_pill",
                           "quarantined after repeated container starts", dict(diagnostics))
        path = self.paths.errors(self.cfg, shard.kind, shard.shard_id, "orchestrator", 0)
        self.store.put_bytes(path, (json.dumps(asdict(err)) + "\n").encode())
        self.poison_records.append(uid)


def parts_prefix(paths: JobPaths, cfg: str, kind: str, shard_id: str) -> str:
    return paths.part(cfg, kind, shard_id, "x", 0).rsplit("/", 1)[0] + "/"


class FakeProgress:
    def __init__(self, store: FakeStore, cfg: str = CFG, job: str = JOB):
        self.store, self.cfg, self.paths = store, cfg, JobPaths(job)

    def episodes_done(self, shard: Shard) -> set[str]:
        done: set[str] = set()
        for p in self.store.list(parts_prefix(self.paths, self.cfg, shard.kind, shard.shard_id)):
            for line in self.store.get_bytes(p).decode().splitlines():
                done.add(json.loads(line)["uid"])
        return done


class FakeBudget:
    """Allows ``limit`` spawns, then denies until ``limit`` is raised."""

    def __init__(self, limit: int | None = None):
        self.limit = limit
        self.admitted = 0

    def admit(self, shard: Shard, in_flight: Mapping[str, int]) -> BudgetDecision:
        if self.limit is not None and self.admitted >= self.limit:
            return BudgetDecision(False, "projected spend over budget", {"admitted": self.admitted})
        self.admitted += 1
        return BudgetDecision(True)


@dataclass
class WorkerBehavior:
    """What the fake worker does. ``oom`` episodes crash the container every time they are
    reached (after the marker and the episodes before them are written). ``init_fail[shard]``:
    raise before writing the marker for the first n starts (init / import failure)."""

    oom: set[str] = field(default_factory=set)
    init_fail: dict[str, int] = field(default_factory=dict)
    crash_after_marker: dict[str, int] = field(default_factory=dict)  # preemption with a marker


def make_worker(manifests: FakeManifests, behavior: WorkerBehavior, kind: str,
                processed: list[tuple[str, str]] | None = None) -> Callable:
    paths = JobPaths(JOB)

    def worker(job_id: str, shard_id: str, cfg: str, container: str,
               store: FakeStore) -> ShardReceipt:
        if behavior.init_fail.get(shard_id, 0) > 0:
            behavior.init_fail[shard_id] -= 1
            raise RuntimeError(f"init failure in {shard_id}")
        shard = manifests.shard(shard_id)
        marker = AttemptMarker(shard_id, container, "t", 0)
        store.put_bytes(paths.attempt(cfg, kind, shard_id, container),
                        json.dumps(asdict(marker)).encode())
        if behavior.crash_after_marker.get(shard_id, 0) > 0:
            behavior.crash_after_marker[shard_id] -= 1
            raise Preempted("preempt")
        done: set[str] = set()
        for p in store.list(parts_prefix(paths, cfg, kind, shard_id)):
            done |= {json.loads(x)["uid"] for x in store.get_bytes(p).decode().splitlines()}
        parts = []
        for k, uid in enumerate(shard.episodes):
            if uid in done:
                continue
            if uid in behavior.oom:
                raise Preempted("oom")
            part = paths.part(cfg, kind, shard_id, container, k)
            store.put_bytes(part, (json.dumps({"uid": uid}) + "\n").encode())
            parts.append(part)
            if processed is not None:
                processed.append((shard_id, uid))
        receipt = ShardReceipt(shard_id, kind, cfg, container,  # type: ignore[arg-type]
                               tuple(shard.episodes), (), tuple(parts), "H200", 1.0, "t")
        store.put_bytes(paths.receipt(cfg, kind, shard_id),
                        json.dumps(asdict(receipt)).encode())
        return receipt

    return worker


class RecordingSpawner(FakeSpawner):
    """FakeSpawner that records, at every spawn, how many calls are really in flight per kind."""

    def __init__(self, *a, pools: Mapping[str, PoolSettings], **kw):
        super().__init__(*a, **kw)
        self.pools = pools
        self.peak_gpus = 0
        self.peak_by_kind: dict[str, int] = {}
        self.spawns: list[str] = []

    def in_flight(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls.values():
            if not c.done:
                out[c.kind] = out.get(c.kind, 0) + 1
        return out

    def spawn(self, kind, job_id, shard_id, config_hash):
        call_id = super().spawn(kind, job_id, shard_id, config_hash)
        self.spawns.append(shard_id)
        inf = self.in_flight()
        gpus = sum(-(-n // self.pools[k].max_inputs) * self.pools[k].gpu_count
                   for k, n in inf.items())
        self.peak_gpus = max(self.peak_gpus, gpus)
        for k, n in inf.items():
            self.peak_by_kind[k] = max(self.peak_by_kind.get(k, 0), n)
        return call_id


def make_manifest(spec: Mapping[str, Sequence[str]], kind: str = "segcap",
                  deps: Mapping[str, Sequence[str]] | None = None,
                  kinds: Mapping[str, str] | None = None) -> Manifest:
    shards = tuple(
        Shard(sid, (kinds or {}).get(sid, kind), tuple(eps), 60.0 * len(eps),  # type: ignore[arg-type]
              depends_on=tuple((deps or {}).get(sid, ())))
        for sid, eps in spec.items())
    return Manifest(JOB, CFG, (), shards, "t")


def make_settings(pools: Mapping[str, PoolSettings], max_gpus: int = 8, **kw) -> RunSettings:
    return RunSettings(job_id=JOB, preset="exact", config_hash=CFG, budget_usd=10.0,
                       max_gpus=max_gpus, pools=dict(pools), **kw)  # type: ignore[arg-type]


def control(store: FakeStore, kind: str, t: float, reason: str = "operator") -> None:
    write_control(store, JobPaths(JOB),
                  ControlRecord(kind, JOB, reason, utc(t)))  # type: ignore[arg-type]


DEFAULT_POOLS = {"segcap": PoolSettings(gpu="H200", max_containers=2, max_inputs=1)}


class Harness:
    """Everything one job needs, wired with fakes. ``orchestrator()`` builds a fresh instance over
    the same durable state (a restart); ``run()`` drives it with the spawner ticking in between."""

    def __init__(self, spec: Mapping[str, Sequence[str]], *, pools=None, max_gpus: int = 8,
                 lag: int = 1, behavior: WorkerBehavior | None = None, deps=None, kinds=None,
                 python_retries: int = 0, max_restarts: int = 50, budget: FakeBudget | None = None,
                 **settings_kw):
        self.pools = dict(pools or DEFAULT_POOLS)
        self.settings = make_settings(self.pools, max_gpus=max_gpus, **settings_kw)
        self.store = FakeStore(lag=lag)
        self.manifests = FakeManifests(make_manifest(spec, deps=deps, kinds=kinds))
        self.attempts = FakeAttempts(self.store)
        self.progress = FakeProgress(self.store)
        self.behavior = behavior or WorkerBehavior()
        self.processed: list[tuple[str, str]] = []
        workers = {k: make_worker(self.manifests, self.behavior, k, self.processed)
                   for k in self.pools}
        self.spawner = RecordingSpawner(self.store, workers, pools=self.pools,
                                        python_retries=python_retries, max_restarts=max_restarts)
        self.clock = FakeClock()
        self.kv = FakeDict()
        self.budget = budget or FakeBudget()
        self.events: list[dict] = []

    def lease(self) -> DictLease:
        return DictLease(self.kv, f"orchestrator/{JOB}", clock=self.clock)

    def orchestrator(self, **kw) -> Orchestrator:
        args = dict(settings=self.settings, store=self.store, spawner=self.spawner,
                    lease=self.lease(), manifests=self.manifests, attempts=self.attempts,
                    progress=self.progress, budget=self.budget, events=self.events.append,
                    clock=self.clock)
        args.update(kw)
        return Orchestrator(**args)

    def sleep(self, dt: float) -> None:
        self.clock.advance(dt)
        self.spawner.tick()

    def run(self, orch: Orchestrator | None = None, max_ticks: int = 500) -> RunOutcome:
        orch = orch or self.orchestrator()
        return orch.run_until_idle(sleep=self.sleep, poll_interval_s=10.0, max_ticks=max_ticks)

    def receipt(self, shard_id: str, kind: str = "segcap") -> bool:
        return self.store.exists(JobPaths(JOB).receipt(CFG, kind, shard_id))

    def done_uids(self) -> set[str]:
        return {u for _, u in self.processed}

    def spawn_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for sid in self.spawner.spawns:
            out[sid] = out.get(sid, 0) + 1
        return out

    def names(self) -> list[str]:
        return [e["event"] for e in self.events]
