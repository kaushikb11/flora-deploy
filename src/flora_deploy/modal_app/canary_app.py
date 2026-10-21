"""``flora-canary``: a real-Modal, CPU-only canary of the job system (C2 in docs/NEXT_EXECUTION.md).

It runs the real pieces against each other on Modal, at cents of cost:

* the real :class:`~flora_deploy.jobs.orchestrator.Orchestrator` driven by ``orchestrator_app``'s
  :class:`~flora_deploy.modal_app.orchestrator_app.TickDriver` (store refresh each tick, budget
  hard cap, transient-error retry), with the real
  :class:`~flora_deploy.modal_app.orchestrator_app.ModalSpawner` (``spawn`` / real
  ``FunctionCall.from_id(...).get(timeout=0)`` polling through ``status_from_get`` / ``cancel`` /
  ``terminate``) and a real ``modal.Dict`` lease (:class:`DictLease`);
* the real store adapters (``jobs.deps.build_deps``) over a real **Volume v2** (``flora-canary``)
  through :class:`~flora_deploy.jobs.stores.VolumeStore` (commit per write, reload per tick /
  per input);
* :class:`CanaryWorker`, a trivial CPU worker class whose core (:class:`ContractWorker`)
  implements the full ``ShardWorker`` contract (``flora_deploy.workers`` steps 0-6) and writes
  fake ``flora_labels_v1`` records. It also passes ``tests/conformance``.

The only difference from a deployment: the app runs ephemerally (``app.run()``, never
``modal deploy``), so ``ModalSpawner`` resolves the worker class through :class:`_SameApp`
instead of ``modal.Cls.from_name`` (an ephemeral app cannot be looked up by name).

Scenarios (``SCENARIOS``) and their pass criteria (:func:`evaluate`) are driven by
``scripts/canary.py``. Each job writes only under ``flora/<job_id>/`` on ``flora-canary``.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import modal

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.manifest import load_manifest, save_manifest, stages_of
from flora_deploy.jobs.orchestrator import (
    DictLease,
    Orchestrator,
    ShardLedger,
    call_path,
    utc,
)
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.stores import VolumeStore
from flora_deploy.jobs.types import (
    EpisodeError,
    EpisodeRef,
    Manifest,
    Shard,
    ShardKind,
    ShardReceipt,
    episode_uid,
)
from flora_deploy.modal_app.images import cpu_image
from flora_deploy.records import Provenance, Segment, StageRecord

APP_NAME = "flora-canary"
VOLUME_NAME = "flora-canary"
LEASE_DICT = "flora-canary-leases"
MOUNT = "/canary"
KIND: ShardKind = "segcap"
CFG = "canarycfg1"
WORKER_CLASS = "CanaryWorker"
CPU = 0.25
MEMORY_MIB = 512
TIMEOUT_S = 600  # hard cap per container (task rule: <= 10 min)
LEASE_TTL_S = 45.0  # short, so a crashed generation's lease expires inside the canary


# ----------------------------------------------------------------------------------------------
# Worker core (no Modal): the ShardWorker contract, steps 0-6
# ----------------------------------------------------------------------------------------------


class NvFailure(Exception):
    """Raised by an engine where NVIDIA's own pipeline reports a failure for an episode."""


Engine = Callable[[EpisodeRef, str, str], StageRecord]  # (ref, stage, config_hash) -> record
Hook = Callable[[Shard, str], None]  # (shard, uid) before each episode


def fake_record(ref: EpisodeRef, stage: str, cfg: str) -> StageRecord:
    return StageRecord(
        uid=ref.uid, dataset=ref.dataset, episode_id=ref.episode_id,
        stage=stage,  # type: ignore[arg-type]
        provenance=Provenance(models={"canary": "0"}, engine="canary-cpu", preset="exact",
                              config_hash=cfg, logic_version="canary-1",
                              framework="flora-deploy canary"),
        certified=True,
        segments=(Segment(f"{ref.uid}:0", 0.0, ref.duration_s or 1.0,
                          caption="canary" if stage == "caption" else None),))


class ContractWorker:
    """One shard per ``run_shard`` call, idempotently (``flora_deploy.workers``):

    0. superseded spawn -> ``None``, nothing written; 1. marker (token + call id) first;
    2. skip episodes in earlier containers' parts; 3. upstream-poisoned episodes ->
    ``EpisodeError(poison_pill)``, not run; 4. records to a local buffer, flushed whole every
    ``flush_every`` episodes; :class:`NvFailure` -> ``EpisodeError(nv_failure)``;
    5. SIGINT (``KeyboardInterrupt``) -> flush, ``mark_preempted``, re-raise; another exception
    -> ``mark_exception``, re-raise; 6. receipt LAST naming every part.
    ``on_exit`` (``@modal.exit``) repeats step 5 if a shard is still in progress."""

    def __init__(self, store: Store, container_id: str, call_id: str, *, kind: ShardKind = KIND,
                 clock: Callable[[], float] = time.time, engine: Engine = fake_record,
                 before_episode: Hook | None = None, after_marker: Hook | None = None,
                 flush_every: int = 2):
        self.store, self.container_id, self.call_id, self.kind = store, container_id, call_id, kind
        self.clock, self.engine = clock, engine
        self.before_episode, self.after_marker = before_episode, after_marker
        self.flush_every = max(1, flush_every)
        self._cur: dict[str, Any] | None = None

    def _flush(self) -> None:
        c = self._cur
        if c is None or not c["buf"]:
            return
        c["own"].append(P.write_part(c["jf"], self.kind, c["sid"], self.container_id, c["k"],
                                     c["buf"]))
        c["k"] += 1
        c["buf"] = []

    def _preempted(self) -> None:
        c = self._cur
        if c is None:
            return
        self._flush()
        A.mark_preempted(c["jf"], self.kind, c["sid"], self.container_id,
                         ended_utc=utc(self.clock()), marker=c["marker"])
        self._cur = None

    def on_exit(self) -> None:
        self._preempted()

    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        from flora_deploy.jobs.deps import StoreManifestSource

        jf = A.JobFiles(self.store, job_id, config_hash)
        if A.superseded(jf, self.kind, shard_id, token):  # 0
            return None
        src = StoreManifestSource(jf)
        shard = src.shard(shard_id)
        refs = {r.uid: r for r in src.base().episodes}
        marker = A.write_attempt(jf, self.kind, shard_id, self.container_id,  # 1
                                 started_utc=utc(self.clock()),
                                 orchestrator_attempt=A.token_attempt(token), token=token,
                                 call_id=self.call_id)
        self._cur = c = {"jf": jf, "sid": shard_id, "marker": marker, "buf": [], "k": 0,
                         "own": []}
        try:
            if self.after_marker is not None:
                self.after_marker(shard, "")
            prev = P.shard_parts(jf, shard)  # 2
            done = set(prev.done)
            skip: set[str] = set()
            if shard.depends_on:  # 3
                ups = [s for s in src.load().shards if s.shard_id in shard.depends_on]
                skip = A.upstream_poisoned(jf, shard, ups) - done
                A.write_upstream_failures(jf, shard, skip)
            failed: list[str] = []
            errors: list[EpisodeError] = []
            n = 0
            for uid in shard.episodes:  # 4
                if uid in done or uid in skip:
                    continue
                if self.before_episode is not None:
                    self.before_episode(shard, uid)
                try:
                    recs = [self.engine(refs[uid], st, config_hash) for st in stages_of(self.kind)]
                except NvFailure as exc:
                    failed.append(uid)
                    errors += [EpisodeError(uid, st, shard_id, self.container_id, "nv_failure",
                                            str(exc)) for st in stages_of(self.kind)]
                    continue
                c["buf"].extend(recs)
                n += 1
                if n % self.flush_every == 0:
                    self._flush()
            self._flush()
            if errors:
                P.write_errors(jf, self.kind, shard_id, self.container_id, 0, errors)
        except KeyboardInterrupt:  # 5: SIGINT (preemption)
            self._preempted()
            raise
        except Exception as exc:
            A.mark_exception(jf, self.kind, shard_id, self.container_id,
                             ended_utc=utc(self.clock()), error=repr(exc), marker=marker)
            self._cur = None
            raise
        bad = skip | set(failed)
        rec = ShardReceipt(shard_id, self.kind, config_hash, self.container_id,
                           tuple(u for u in shard.episodes if u not in bad),
                           tuple(u for u in shard.episodes if u in bad),
                           tuple(sorted(set(prev.parts) | set(c["own"]))), None,
                           0.0, utc(self.clock()))
        P.write_receipt(jf, rec)  # 6: LAST
        self._cur = None
        return rec


def conformance_spec():
    """``tests/conformance`` registration of :class:`ContractWorker` (fake engine + script)."""
    from tests.conformance.worker_suite import WorkerEnv, WorkerSpec

    def factory(env: WorkerEnv) -> ContractWorker:
        def engine(ref: EpisodeRef, stage: str, cfg: str) -> StageRecord:
            if ref.uid == env.script.interrupt_before:
                raise KeyboardInterrupt("SIGINT")
            if ref.uid in env.script.nv_fail:
                raise NvFailure(f"NVIDIA reported a failure for {ref.uid}")
            return fake_record(ref, stage, cfg)

        return ContractWorker(env.store, env.container_id, env.call_id, clock=env.clock,
                              engine=engine)

    return WorkerSpec("canary-contract-worker", KIND, factory)


# ----------------------------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Behavior:
    """What :class:`CanaryWorker` does, per job (``flora/<job>/canary/behavior.json``)."""

    episode_s: float = 1.0
    flush_every: int = 2
    preempt: dict[str, int] = field(default_factory=dict)  # shard -> simulate_preemption(s)
    poison: tuple[str, ...] = ()  # uids that os._exit(137) the container every time
    slow: dict[str, float] = field(default_factory=dict)  # shard -> episode seconds

    @staticmethod
    def path(job_id: str) -> str:
        return f"{JobPaths(job_id).root}/canary/behavior.json"

    @classmethod
    def load(cls, store: Store, job_id: str) -> Behavior:
        p = cls.path(job_id)
        if not store.exists(p):
            return cls()
        d = json.loads(store.get_bytes(p))
        d["poison"] = tuple(d.get("poison", ()))
        return cls(**d)


@dataclass(frozen=True)
class Scenario:
    name: str
    shards: dict[str, tuple[str, ...]]  # shard id -> episode ids
    behavior: Behavior = Behavior()
    max_containers: int = 4
    budget_container_s: float | None = None  # hard-cap budget in container-seconds of the pool
    est_s: float = 30.0  # per-shard estimate (ManifestEstimator)
    crash_orchestrator_after_ticks: int | None = None
    poll_s: float = 3.0
    max_ticks: int = 160

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["behavior"]["poison"] = list(self.behavior.poison)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scenario:
        d = dict(d)
        b = dict(d.pop("behavior"))
        b["poison"] = tuple(b.get("poison", ()))
        d["shards"] = {k: tuple(v) for k, v in d["shards"].items()}
        return cls(behavior=Behavior(**b), **d)


DATASET = "canary/ds"


def uid(ep: str) -> str:
    return episode_uid(DATASET, ep)


def _grid(n_shards: int, eps: int, tag: str) -> dict[str, tuple[str, ...]]:
    return {f"{KIND}-{i:05d}": tuple(f"{tag}{i}-{j}" for j in range(eps)) for i in range(n_shards)}


def scenarios() -> dict[str, Scenario]:
    poison_shards = _grid(5, 2, "q")
    poison_shards[f"{KIND}-00005"] = ("pa", "poison", "pb", "pc")
    return {s.name: s for s in [
        Scenario("happy", _grid(8, 2, "h"), Behavior(episode_s=1.0)),
        Scenario("preempt", _grid(6, 3, "p"),
                 Behavior(episode_s=2.0, preempt={f"{KIND}-00002": 3})),
        Scenario("poison", poison_shards, Behavior(episode_s=0.5, poison=(uid("poison"),))),
        Scenario("restart", _grid(8, 2, "r"), Behavior(episode_s=3.0),
                 crash_orchestrator_after_ticks=3),
        Scenario("budget", _grid(2, 4, "b"), Behavior(episode_s=1.0, flush_every=1,
                                                      slow={f"{KIND}-00000": 25.0,
                                                            f"{KIND}-00001": 25.0}),
                 max_containers=2, budget_container_s=80.0, est_s=20.0),
    ]}


def build_manifest(job_id: str, sc: Scenario) -> Manifest:
    eps = sorted({e for es in sc.shards.values() for e in es})
    refs = tuple(EpisodeRef(uid=uid(e), dataset=DATASET, episode_id=e, pool="human",
                            duration_s=2.0, oracle_eligible=True, envelope_ok=True,
                            index_ref=JobPaths(job_id).index(uid(e))) for e in eps)
    shards = tuple(Shard(sid, KIND, tuple(uid(e) for e in es), sc.est_s)
                   for sid, es in sc.shards.items())
    return Manifest(job_id, CFG, refs, shards, utc(time.time()))


def pool_settings(sc: Scenario):
    from flora_deploy.jobs.settings import PoolSettings

    return PoolSettings(gpu="CPU", gpu_count=0, max_containers=sc.max_containers, max_inputs=1,
                        cpu_cores=CPU, memory_gib=MEMORY_MIB / 1024)


def canary_rates():
    from flora_deploy.jobs.budget import pinned_rates

    r = pinned_rates()
    return replace(r, gpu_per_s={**r.gpu_per_s, "CPU": 0.0})


def settings_for(job_id: str, sc: Scenario):
    from flora_deploy.jobs.settings import RunSettings

    pool = pool_settings(sc)
    budget = 1.0
    if sc.budget_container_s is not None:
        budget = canary_rates().pool_per_second(pool) * sc.budget_container_s
    return RunSettings(job_id=job_id, preset="exact", config_hash=CFG, budget_usd=budget,
                       max_gpus=1, pools={KIND: pool})


# ----------------------------------------------------------------------------------------------
# Modal app
# ----------------------------------------------------------------------------------------------

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
leases = modal.Dict.from_name(LEASE_DICT, create_if_missing=True)


def _container_id() -> str:
    return os.environ.get("MODAL_TASK_ID") or f"local-{os.getpid()}"


@app.cls(image=cpu_image(), cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S,
         volumes={MOUNT: volume}, max_containers=24, scaledown_window=10)
class CanaryWorker:
    @modal.enter()
    def _enter(self) -> None:
        self.store = VolumeStore(MOUNT, volume)
        self.core: ContractWorker | None = None

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> dict[str, Any] | None:
        # A warm container sees the Volume as of its start / last reload: reload before reading
        # the ledger (step 0) and earlier parts (step 2). One input per container, nothing open.
        self.store.refresh()
        beh = Behavior.load(self.store, job_id)
        cid = _container_id()

        def after_marker(shard: Shard, _uid: str) -> None:
            wait = beh.preempt.get(shard.shard_id)
            if wait and A.count_starts(A.JobFiles(self.store, job_id, config_hash), KIND,
                                       shard.shard_id, known={cid}) == 1:
                from modal.exception import simulate_preemption

                print(f"canary: simulate_preemption({wait}) on {shard.shard_id} in {cid}",
                      flush=True)
                simulate_preemption(wait)

        def before_episode(shard: Shard, u: str) -> None:
            if u in beh.poison:
                print(f"canary: poison {u} -> os._exit(137) in {cid}", flush=True)
                self.store.commit()
                os._exit(137)
            time.sleep(beh.slow.get(shard.shard_id, beh.episode_s))

        self.core = ContractWorker(self.store, cid, modal.current_function_call_id() or "",
                                   before_episode=before_episode, after_marker=after_marker,
                                   flush_every=beh.flush_every)
        try:
            rec = self.core.run_shard(job_id, shard_id, config_hash, token)
        finally:
            self.store.commit()
        return asdict(rec) if rec is not None else None

    @modal.exit()
    def _exit(self) -> None:
        if getattr(self, "core", None) is not None:
            self.core.on_exit()  # type: ignore[union-attr]
            self.store.commit()


class _SameApp:
    """``modal.Cls`` stand-in for :class:`ModalSpawner`: an ephemeral app's classes cannot be
    looked up with ``Cls.from_name``, so return this app's own (hydrated) class."""

    @staticmethod
    def from_name(app_name: str, cls_name: str) -> Any:
        assert (app_name, cls_name) == (APP_NAME, WORKER_CLASS), (app_name, cls_name)
        return CanaryWorker


def _crash_flag(job_id: str) -> str:
    return f"{JobPaths(job_id).root}/canary/orchestrator_crashed"


@app.function(image=cpu_image(), cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S,
              volumes={MOUNT: volume})
def orchestrate(job_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """One orchestrator generation for ``job_id`` (the ``orchestrator_app.run`` code path)."""
    from flora_deploy.jobs.budget import ManifestEstimator, ProjectionBudgetGate
    from flora_deploy.jobs.deps import build_deps
    from flora_deploy.modal_app.orchestrator_app import ModalSpawner, TickDriver, _holder

    sc = Scenario.from_dict(scenario)
    store = VolumeStore(MOUNT, volume)
    store.refresh()
    paths = JobPaths(job_id)
    if not store.exists(paths.manifest):
        save_manifest(store, build_manifest(job_id, sc))
        store.put_bytes(Behavior.path(job_id), json.dumps(
            {**asdict(sc.behavior), "poison": list(sc.behavior.poison)}).encode())
    settings = settings_for(job_id, sc)
    deps = build_deps(store, settings, worker_app=APP_NAME, worker_classes={KIND: WORKER_CLASS})
    if sc.budget_container_s is not None:
        deps.budget = ProjectionBudgetGate(
            settings=settings, store=store, shards=lambda: deps.manifests.load().shards,
            rates=canary_rates(), policy="next_shard", estimator=ManifestEstimator(),
            startup_s={KIND: 0.0}, scaledown_s=0.0, tick_reserve_s=3.0, refresh_s=0.0,
            learn_calibration=False)
    events: list[dict[str, Any]] = []

    def sink(e: dict[str, Any]) -> None:
        events.append(e)
        print(json.dumps(e, sort_keys=True, default=str), flush=True)

    holder = _holder()
    spawner = ModalSpawner(APP_NAME, {KIND: WORKER_CLASS}, cls_api=_SameApp)
    orch = Orchestrator(settings=settings, store=store, spawner=spawner,
                        lease=DictLease(leases, f"orchestrator/{job_id}"),
                        manifests=deps.manifests, attempts=deps.attempts, progress=deps.progress,
                        budget=deps.budget, events=sink, holder=holder,
                        lease_ttl_s=LEASE_TTL_S, spawn_grace_s=30.0)
    ticks = 0

    def sleep(dt: float) -> None:
        nonlocal ticks
        ticks += 1
        n = sc.crash_orchestrator_after_ticks
        if n is not None and ticks >= n and not store.exists(_crash_flag(job_id)):
            store.put_bytes(_crash_flag(job_id), holder.encode() if holder else b"?")
            store.commit()
            print(f"canary: orchestrator {holder} crashing (os._exit) after {ticks} ticks",
                  flush=True)
            os._exit(1)  # no finally: the lease stays held, like a dead container
        time.sleep(dt)

    t0 = time.time()
    driver = TickDriver(orch)
    while True:
        outcome = driver.run_until_idle(sleep=sleep, poll_interval_s=sc.poll_s,
                                        max_ticks=sc.max_ticks)
        if outcome.result != "lease_held" or time.time() - t0 > LEASE_TTL_S + 60:
            break
        time.sleep(5.0)  # a crashed generation still holds the lease: wait out its TTL
    store.commit()

    # Seam probe: what does polling a call the budget cap terminated return?
    probes: dict[str, Any] = {}
    for led in orch._ledgers.values():
        for e in led.calls:
            if e.get("outcome") == "cancelled_budget" and e.get("call_id"):
                # Termination is asynchronous: a call reads "pending" until Modal finishes
                # tearing it down. Poll until it leaves "pending" (bounded), then record it.
                deadline = time.time() + 90.0
                st = spawner.poll(e["call_id"])
                while st.state == "pending" and time.time() < deadline:
                    time.sleep(3.0)
                    st = spawner.poll(e["call_id"])
                probes[e["call_id"]] = {"state": st.state, "error": st.error}
    return {"job_id": job_id, "holder": holder, "outcome": asdict(outcome),
            "wall_s": round(time.time() - t0, 1), "events": dict(Counter(e["event"]
                                                                       for e in events)),
            "poll_after_terminate": probes,
            "crashed_holder": store.get_bytes(_crash_flag(job_id)).decode()
            if store.exists(_crash_flag(job_id)) else None}


def summarize(store: Store, job_id: str, sc: Scenario) -> dict[str, Any]:
    """Final state of one job, from store files alone."""
    from flora_deploy.jobs.deps import build_deps
    from flora_deploy.jobs.reconciler import reconcile

    jf = A.JobFiles(store, job_id, CFG)
    manifest = load_manifest(store, job_id)
    prog = P.derive(jf, manifest, limit=3)
    ci = P.compose_input(jf, prog)
    raw: Counter[tuple[str, str]] = Counter()
    for p in store.list(f"{jf.paths.root}/{CFG}/parts/"):
        if p.endswith(".jsonl"):
            for r in P.read_part(jf, p):
                raw[(r.uid, r.stage)] += 1
    shards: dict[str, Any] = {}
    for sid, st in prog.shards.items():
        cp = call_path(jf.paths, CFG, KIND, sid)
        led = ShardLedger.from_bytes(store.get_bytes(cp)) if store.exists(cp) else None
        ms = A.markers(jf, KIND, sid)
        shards[sid] = {"state": st.state, "ledger": led.status if led else None,
                       "spawn_n": led.spawn_n if led else 0,
                       "respawns": led.respawns if led else 0,
                       "outcomes": [c.get("outcome") for c in led.calls] if led else [],
                       "starts": len(ms), "preempted": sum(m.preempted for m in ms),
                       "errored": sum(bool(m.error) for m in ms),
                       "receipt": st.receipt is not None,
                       "bisected_from": st.shard.bisected_from}
    errs: Counter[str] = Counter()
    for p in store.list(f"{jf.paths.root}/{CFG}/errors/"):
        for e in P.read_errors(jf, p):
            errs[f"{e.kind}:{e.uid}:{e.stage}"] += 1
    controls = sorted(p.split("/control/", 1)[1].split("/", 1)[0]
                      for p in store.list(f"{jf.paths.root}/control/"))
    d = build_deps(store, settings_for(job_id, sc))
    status = reconcile(job_id=job_id, config_hash=CFG, store=store, manifests=d.manifests,
                       attempts=d.attempts, max_starts=3)
    return {
        "complete": prog.complete, "reconciled_state": status.state,
        "expected": len(ci.expected), "records": len(ci.records),
        "missing": sorted(map(list, ci.missing)), "excused": sorted(map(list, ci.excused)),
        "compose_duplicates": ci.duplicates, "conflicts": [list(k) for k in ci.conflicts],
        "foreign": ci.foreign,
        "raw_duplicates": sorted([list(k) + [n] for k, n in raw.items() if n > 1]),
        "shards": shards, "errors": dict(errs), "controls": dict(Counter(controls)),
    }


@app.function(image=cpu_image(), cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S,
              volumes={MOUNT: volume})
def verify(jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``{job_id: scenario dict}`` -> summaries (after a reload: every container's commits)."""
    store = VolumeStore(MOUNT, volume)
    store.refresh()
    out: dict[str, Any] = {}
    for job_id, sc in jobs.items():
        try:
            out[job_id] = summarize(store, job_id, Scenario.from_dict(sc))
        except Exception as exc:  # report, never hide the others
            out[job_id] = {"error": repr(exc)}
    return out


# ----------------------------------------------------------------------------------------------
# Pass criteria
# ----------------------------------------------------------------------------------------------


def evaluate(sc: Scenario, run: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    """Problems with one scenario's result (empty = pass)."""
    bad: list[str] = []
    if "error" in run:
        return [f"orchestrator call failed: {run['error']}"]
    if "error" in summary:
        return [f"verify failed: {summary['error']}"]
    result = run["outcome"]["result"]
    shards = summary["shards"]
    planned = [s for s in shards if not shards[s]["bisected_from"]]

    def want(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(msg)

    want(not summary["raw_duplicates"], f"duplicate records {summary['raw_duplicates']}")
    want(summary["compose_duplicates"] == 0 and not summary["conflicts"]
         and summary["foreign"] == 0, "compose saw duplicates / conflicts / foreign records")
    if sc.name == "budget":
        want(result == "budget_stopped", f"result {result!r}, expected budget_stopped")
        want(summary["controls"].get("budget_stop", 0) >= 1, "no budget_stop record")
        want(not any(s["receipt"] for s in shards.values()), "a capped shard finished")
        want(all("cancelled_budget" in s["outcomes"] for s in shards.values()),
             "not every live call was cancelled by the hard cap")
        want(all(s["respawns"] == 0 for s in shards.values()), "budget cancel consumed a respawn")
        for cid, p in run.get("poll_after_terminate", {}).items():
            want(p["state"] == "cancelled",
                 f"poll of terminated call {cid} -> {p['state']} ({p['error']})")
        return bad
    want(result == "done", f"result {result!r}, expected done")
    want(summary["complete"] and summary["reconciled_state"] == "done",
         f"files say complete={summary['complete']} state={summary['reconciled_state']}")
    want(not summary["missing"], f"missing records {summary['missing']}")
    if sc.name == "poison":
        poison = uid("poison")
        want(summary["excused"] == [[poison, "caption"], [poison, "segment"]],
             f"excused {summary['excused']}")
        parent = f"{KIND}-00005"
        want(shards[parent]["state"] == "quarantined", f"{parent} not quarantined")
        kids = sorted(s for s, v in shards.items() if v["bisected_from"] == parent)
        want(len(kids) >= 2, f"bisection children {kids}")
        child = A.child_shard_id(KIND, poison)
        want(shards.get(child, {}).get("ledger") == "poisoned", f"{child} not poisoned")
        want(f"poison_pill:{poison}:segment" in summary["errors"], "no poison_pill error")
        want(all(shards[s]["receipt"] for s in kids if s != child), "a healthy child unfinished")
    else:
        want(not summary["excused"], f"unexpected excused {summary['excused']}")
        want(all(shards[s]["state"] == "done" for s in shards), "a shard is not done")
    healthy = [s for s in planned if not (sc.name == "poison" and s == f"{KIND}-00005")]
    want(all(shards[s]["spawn_n"] == 1 for s in healthy),
         f"respawned shards {[s for s in healthy if shards[s]['spawn_n'] != 1]}")
    if sc.name == "preempt":
        sid = next(iter(sc.behavior.preempt))
        want(shards[sid]["preempted"] >= 1, f"{sid}: no preempted marker")
        want(shards[sid]["starts"] >= 2, f"{sid}: Modal did not restart the preempted input")
        want(summary["controls"].get("quarantine", 0) == 0, "preemption caused a quarantine")
    if sc.name == "restart":
        want(run.get("crashed_holder") is not None, "the orchestrator never crashed")
        want(run["events"].get("reattached", 0) >= 1,
             f"no re-attach after the orchestrator crash: {run['events']}")
    return bad
