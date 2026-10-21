"""``run_local``: one whole job in this process, with the real job system and fake models.

``index (local) → plan → orchestrate → compose``, every step the production code:

* **index**: ``jobs.indexing.run_index`` over the dataset (``flora index --local``), receipts in
  the job store; resumable;
* **plan**: ``jobs.manifest.build_manifest`` from the receipts, saved once (a re-run reuses the
  stored manifest; a different config hash for the same job id is refused);
* **orchestrate**: the real ``jobs.orchestrator.Orchestrator`` with the real store adapters
  (``jobs.deps.build_deps``), a ``DictLease`` over a file KV (:class:`~flora_deploy.local.spawner.
  FileKV`), the real ``jobs.budget.ProjectionBudgetGate`` (pinned Modal prices, manifest
  estimates: what the job *would* cost) and a :class:`~flora_deploy.local.spawner.LocalSpawner`
  running the real ``SegcapShardRunner`` / ``RewardShardRunner`` with fake engines
  (``local.engines``);
* **compose**: ``export.compose.compose`` (``partial`` unless the orchestrator said ``done``).

Completion (``LocalRunResult.state``), derived from the orchestrator's verdict and the composed
summary only:

* ``done``: every expected record exists; nothing excused, no poison pill;
* ``completed_with_errors``: the job finished, but some episodes are excused (``EpisodeError``
  files: NVIDIA-semantics rejections, poison pills isolated by D15); their records are absent;
* ``failed``: a shard failed terminally (respawns exhausted) or depends on one;
* ``budget_stopped`` (``flora run --local --resume [--budget N]`` continues), ``cancelled``,
  ``interrupted`` (``crash_after_ticks``: a simulated orchestrator crash, lease not released),
  ``lease_held`` (another orchestrator holds the job's lease).

Config hash: the real D11 document (``config.hashing.hash_document``) with fake derived-weight
receipts and image fingerprints (:data:`FAKE_RECEIPT`) and the local dataset revision as input
content, so fake outputs never share a hash (hence store paths) with real ones.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flora_deploy.jobs.manifest import build_manifest, load_manifest, save_manifest
from flora_deploy.jobs.orchestrator import DictLease, Orchestrator, utc, write_control
from flora_deploy.jobs.settings import RunSettings
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import ControlRecord, Manifest
from flora_deploy.local.engines import (
    FAKE_CERTIFICATE,
    FAKE_MODELS,
    EpisodeRegistry,
    FakeLLM,
    FakeSegcapEngine,
    fake_rewarder_family,
    vllm_shim,
)
from flora_deploy.local.faults import FaultPlan, maybe_fail, note_episodes
from flora_deploy.local.spawner import FileKV, LocalSpawner

FAKE_RECEIPT = "flora-local-fake-models"
SUPPORTED_KINDS = ("segcap", "reward.terminal", "reward.dense")
LOCAL_DIR = ".flora-local"  # under the store root, outside the job tree (``flora/``)
STATES = ("done", "completed_with_errors", "failed", "budget_stopped", "cancelled",
          "interrupted", "lease_held")

Log = Callable[[str], None]


class LocalRunError(RuntimeError):
    """The local run cannot start (bad config for local mode, manifest mismatch, ...)."""


class FaultStore:
    """``Store`` wrapper given to every local worker runner: delivers the ``fail`` fault at a
    container start's first part / errors / audit / receipt write (``faults.maybe_fail``). Reads and
    every other write pass straight through."""

    _AREAS = ("/parts/", "/errors/", "/audit/", "/receipts/")

    def __init__(self, store: Store):
        self.store = store

    def put_bytes(self, path: str, data: bytes) -> None:
        if any(a in path for a in self._AREAS):
            maybe_fail(path.rsplit("/", 1)[-1])
        self.store.put_bytes(path, data)

    def get_bytes(self, path: str) -> bytes:
        return self.store.get_bytes(path)

    def exists(self, path: str) -> bool:
        return self.store.exists(path)

    def list(self, prefix: str):
        return self.store.list(prefix)


def fake_config_hash(cfg: Any, *, input_digest: str | None = None) -> str:
    """D11 job hash of ``cfg`` with fake derived-weight receipts and image fingerprints."""
    from flora_deploy.config.hashing import config_hash

    receipts = {"captioner_bf16": FAKE_RECEIPT, "tempseg_merged": FAKE_RECEIPT}
    return config_hash(cfg, derived_receipts=receipts, input_digest=input_digest,
                       image_fingerprint=lambda name: f"{FAKE_RECEIPT}:{name}").hash


def completion_state(outcome: str, summary: Mapping[str, Any] | None) -> str:
    """``done`` / ``completed_with_errors`` / ``failed`` / ... (module docstring)."""
    if outcome != "done":
        return outcome if outcome in STATES else "failed"
    if summary is None or not summary.get("complete"):
        return "failed"
    excused = sum(int(s.get("excused", 0)) for s in summary.get("stages", {}).values())
    if excused or summary.get("poison_pills"):
        return "completed_with_errors"
    return "done"


@dataclass
class LocalRunResult:
    job_id: str
    config_hash: str
    store_root: str | None
    state: str
    outcome: str  # the orchestrator's last RunResult
    summary: dict[str, Any] | None  # compose summary (None if compose was impossible)
    index: dict[str, Any] | None = None
    spawner: dict[str, Any] = field(default_factory=dict)
    faults: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.state in ("done", "completed_with_errors")

    def counts(self, kind: str) -> int:
        return sum(1 for e in self.events if e.get("event") == kind)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("events")
        d["event_counts"] = dict(sorted(_count(e["event"] for e in self.events).items()))
        return d


def _container_exit(exc: BaseException) -> None:
    """``workers_app._die`` (``os._exit`` after a CUDA fault / GPU OOM) in the local model: the
    container dies and the spawner restarts the call in a new one, as Modal does."""
    from flora_deploy.local.faults import ContainerCrash

    raise ContainerCrash(f"container exit after {type(exc).__name__}: {exc}"[:200])


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


NOTABLE = {"spawn_error", "call_failed", "call_expired", "call_cancelled", "reattached",
           "spawn_adopted", "poison_detected", "quarantined", "bisected", "poisoned",
           "shard_failed", "budget_stop", "budget_overspend_cancel", "call_cancelled_budget",
           "budget_stopped", "cancelled", "done", "failed", "lease_lost", "lease_held"}


class LocalRunner:
    """A local job over one store (see the module docstring). Use :func:`run_local` for the
    one-shot path; the steps are separate so tests can crash and restart the orchestrator while
    the spawner (the "Modal" side) keeps running.

    ``cfg``: a ``FloraConfig`` or a path to ``flora.yaml``. ``store``: a directory (a
    ``LocalStore`` is opened on it) or any ``Store``; default ``<config dir>/flora-local-store``.
    A relative ``input.path`` resolves against the config file's directory (else the cwd)."""

    def __init__(self, cfg: Any, *, config_path: str | Path | None = None,
                 store: str | Path | Store | None = None, job_id: str | None = None,
                 fake_models: bool = True, faults: FaultPlan | str | Iterable[str] | None = None,
                 budget_usd: float | None = None, poll_interval_s: float = 0.02,
                 lease_ttl_s: float = 30.0, spawn_grace_s: float = 5.0,
                 max_workers: int | None = None, work_dir: str | Path | None = None,
                 log: Log | None = None, verbose: bool = False):
        from flora_deploy.config.schema import load_config
        from flora_deploy.jobs.stores import LocalStore

        if not fake_models:
            raise LocalRunError("local runs need --fake-models: the real NVIDIA models need "
                                "GPUs (deploy to Modal for real outputs)")
        if isinstance(cfg, str | Path):
            config_path = Path(cfg)
            cfg = load_config(config_path)
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path is not None else None
        base = self.config_path.parent if self.config_path is not None else Path.cwd()
        self.job_id = job_id or cfg.project
        self.faults = faults if isinstance(faults, FaultPlan) else FaultPlan.parse(faults)
        self.poll_interval_s, self.lease_ttl_s = poll_interval_s, lease_ttl_s
        self.spawn_grace_s = spawn_grace_s
        self.log: Log = log or (lambda m: None)
        self.verbose = verbose
        self.events: list[dict] = []

        if store is None:
            store = base / "flora-local-store"
        if isinstance(store, str | Path):
            self.store_root: Path | None = Path(store).expanduser().resolve()
            self.store: Store = LocalStore(self.store_root)
        else:
            self.store_root = None
            self.store = store
        if work_dir is not None:
            self.local_dir = Path(work_dir)
        elif self.store_root is not None:
            self.local_dir = self.store_root / LOCAL_DIR
        else:
            self.local_dir = Path(tempfile.mkdtemp(prefix="flora-local-"))
        self.local_dir.mkdir(parents=True, exist_ok=True)

        if cfg.input.repo is None:
            p = Path(str(cfg.input.path)).expanduser()
            self.dataset_root: Path | None = (p if p.is_absolute() else base / p).resolve()
        else:
            self.dataset_root = None  # resolved by index() (Hub mirror)

        from flora_deploy.config.presets import get_preset, shard_kinds_for

        kinds = shard_kinds_for(get_preset(cfg.preset), cfg.pipeline.stages)
        unsupported = [k for k in kinds if k not in SUPPORTED_KINDS]
        if unsupported:
            raise LocalRunError(f"the local fake runner has no engine for shard kind(s) "
                                f"{unsupported} (supported: {', '.join(SUPPORTED_KINDS)}; use "
                                "preset exact and stages segment/caption/reward.terminal/"
                                "reward.dense)")
        self.budget_usd = budget_usd
        n = sum(p.max_containers * p.max_inputs for p in cfg.pools().values())
        self.max_workers = max_workers or 2 * n + 2
        self.registry = EpisodeRegistry()
        self.nv = fake_rewarder_family(self.registry)
        self.settings: RunSettings | None = None
        self.index_summary: dict[str, Any] | None = None
        self.manifest: Manifest | None = None
        self.media_root: Path | None = self.dataset_root
        self._spawner: LocalSpawner | None = None
        self.last_outcome = "running"

    # -- events -----------------------------------------------------------------------------
    def _event(self, e: dict) -> None:
        self.events.append(e)
        name = e.get("event")
        if self.verbose or name in NOTABLE:
            detail = {k: v for k, v in e.items() if k not in ("event", "job_id", "t", "holder")}
            self.log(f"  [{name}] " + " ".join(f"{k}={v}" for k, v in sorted(detail.items())))

    # -- steps -----------------------------------------------------------------------------
    def index(self) -> dict[str, Any]:
        from flora_deploy.jobs.indexing import IndexJob, open_dataset, run_index

        ijob = IndexJob.from_config(self.cfg, self.job_id)
        if self.dataset_root is not None:
            if not self.dataset_root.is_dir():
                raise LocalRunError(f"input.path {self.dataset_root} is not a directory (make "
                                    "the demo data with `flora run --local --sample-data`)")
            files = open_dataset(ijob.dataset, local_root=self.dataset_root)
        else:
            stage = self.cfg.input.stage_to
            work = (Path(stage).expanduser() if stage and "://" not in stage
                    else Path("~/.cache/flora-deploy/datasets").expanduser())
            files = open_dataset(ijob.dataset, work=work, staged=True)
        self.media_root = Path(files.root).resolve()  # NVIDIA wants absolute video paths
        quiet = (lambda m: None) if not self.verbose else self.log
        summary = run_index(ijob, self.store, files, log=quiet)
        c = summary["counts"]
        self.log(f"index: {c['episodes']} episodes, {c['indexed']} indexed, "
                 f"{c['source_error']} source errors, {c['failed']} failed "
                 f"({c['resumed']} resumed)")
        self.index_summary = summary
        return summary

    def plan(self) -> Manifest:
        from flora_deploy.config.schema import to_run_settings
        from flora_deploy.jobs.indexing import load_indexes

        if self.index_summary is None:
            self.index()
        assert self.index_summary is not None
        digest = str(self.index_summary["dataset"].get("revision") or "") or None
        cfg_hash = fake_config_hash(self.cfg, input_digest=digest)
        settings = to_run_settings(self.cfg, self.job_id, config_hash=cfg_hash)
        if self.budget_usd is not None:
            settings = dataclasses.replace(settings, budget_usd=float(self.budget_usd))
        self.settings = settings
        paths = JobPaths(self.job_id)
        if self.store.exists(paths.manifest):
            m = load_manifest(self.store, self.job_id)
            if m.config_hash != cfg_hash:
                raise LocalRunError(
                    f"job {self.job_id!r} in this store was planned for config {m.config_hash}"
                    f", not {cfg_hash} (the config or the data changed); use another --job or "
                    "a fresh --store")
            self.log(f"plan: reusing the stored manifest ({len(m.shards)} shards)")
        else:
            loaded = load_indexes(self.store, self.job_id, policy=self.cfg.pipeline.envelope)
            if not loaded.indexes:
                raise LocalRunError("no indexed episodes to plan (see the index summary)")
            m = build_manifest(loaded.indexes, settings, created_utc=utc(time.time()),
                               pool=loaded.pools)
            save_manifest(self.store, m)
            by_kind = _count(s.kind for s in m.shards)
            self.log(f"plan: {len(m.episodes)} episodes, {len(m.shards)} shards ("
                     + ", ".join(f"{k} {n}" for k, n in by_kind.items()) + ")")
        self.manifest = m
        return m

    # -- workers ---------------------------------------------------------------------------
    def _runner(self, kind: str, container_id: str, call_id: Callable[[], str]):
        from flora_deploy.config.schema import output_settings
        from flora_deploy.workers_impl.reward import (
            RewardEngine,
            RewardShardRunner,
            RewardWorkerSettings,
        )
        from flora_deploy.workers_impl.segcap import SegcapShardRunner, SegcapWorkerSettings

        assert self.settings is not None
        cfg, pool = self.cfg, self.settings.pools[kind]  # type: ignore[index]
        instr = cfg.pipeline.episode_instruction
        local_dir = str(self.local_dir / "containers" / container_id)
        media = str(self.media_root) if self.media_root is not None else None
        store = FaultStore(self.store)
        if kind == "segcap":
            s = SegcapWorkerSettings(
                preset=cfg.preset, models=FAKE_MODELS["segcap"], gpu=pool.gpu,
                certificate=FAKE_CERTIFICATE, envelope=cfg.pipeline.envelope,
                instruction_text=instr.text, media_root=media, local_dir=local_dir,
                container_id=container_id)
            return SegcapShardRunner(store, FakeSegcapEngine(), s, call_id=call_id,
                                     fatal=_container_exit)
        s = RewardWorkerSettings(
            terminal_mode=output_settings(cfg)["terminal_mode"], preset=cfg.preset,
            models=FAKE_MODELS["reward"], gpu=pool.gpu, certificate=FAKE_CERTIFICATE,
            config_extra={"fake_models": FAKE_RECEIPT}, decoder="inline", workers=1,
            envelope=cfg.pipeline.envelope, default_task=instr.text, media_root=media,
            local_dir=local_dir, container_id=container_id)
        runner = RewardShardRunner(store, RewardEngine(FakeLLM(), "fake-processor"), self.nv, s,
                                   call_id=call_id)
        from_index = runner.episode_from_index

        def episode(ref):  # the runner's own reader, plus fault targeting by video paths
            ep = from_index(ref)
            self.registry.add(ep.videos, ref.uid)
            note_episodes(1)
            return ep

        runner.episode_for = episode
        return runner

    @property
    def spawner(self) -> LocalSpawner:
        if self._spawner is None:
            # reward runners import ``vllm`` (stages.reward.sampling_params): keep the shim up
            # for as long as calls can run, i.e. until close()
            self._shim = vllm_shim()
            self._shim.__enter__()
            self._spawner = LocalSpawner(self._runner, faults=self.faults,
                                         max_workers=self.max_workers)
        return self._spawner

    def _budget(self, deps):
        from flora_deploy.jobs.budget import ManifestEstimator, ProjectionBudgetGate, pinned_rates

        assert self.settings is not None
        return ProjectionBudgetGate(
            settings=self.settings, store=self.store,
            shards=lambda: deps.manifests.load().shards, rates=pinned_rates(),
            policy="next_shard", estimator=ManifestEstimator(),
            startup_s={k: 0.0 for k in self.settings.pools}, scaledown_s=0.0,
            tick_reserve_s=max(4 * self.poll_interval_s, 0.5), refresh_s=0.0,
            learn_calibration=False)

    def resume(self, reason: str = "flora run --local --resume") -> None:
        """Write a ``resume`` record: lifts a ``budget_stop`` (raise the budget with
        ``budget_usd=``)."""
        assert self.settings is not None
        write_control(self.store, JobPaths(self.job_id), ControlRecord(
            "resume", self.job_id, reason, utc(time.time()),
            {"config_hash": self.settings.config_hash}))
        self.log("resume: budget stop lifted")

    def orchestrate(self, *, max_ticks: int | None = None,
                    crash_after_ticks: int | None = None) -> str:
        """Run the real orchestrator until the job is idle. ``crash_after_ticks``: stop after
        that many ticks *without* releasing the lease (an orchestrator crash; returns
        ``interrupted``). Returns the orchestrator's result."""
        from flora_deploy.jobs.deps import build_deps

        if self.manifest is None:
            self.plan()
        assert self.settings is not None
        deps = build_deps(self.store, self.settings)
        lease = DictLease(FileKV(self.local_dir / "lease"),
                          f"orchestrator/{self.job_id}/{self.settings.config_hash}")
        orch = Orchestrator(settings=self.settings, store=self.store, spawner=self.spawner,
                            lease=lease, manifests=deps.manifests, attempts=deps.attempts,
                            progress=deps.progress, budget=self._budget(deps),
                            events=self._event, lease_ttl_s=self.lease_ttl_s,
                            spawn_grace_s=self.spawn_grace_s,
                            holder=f"local-orch-{uuid.uuid4().hex[:10]}")
        deadline = time.monotonic() + self.lease_ttl_s + 5.0
        waited = False
        while not orch.acquire():
            if time.monotonic() > deadline:
                self.last_outcome = "lease_held"
                return "lease_held"
            if not waited:
                self.log(f"orchestrate: lease held by a previous orchestrator; waiting up to "
                         f"{self.lease_ttl_s:.0f} s for it to expire")
                waited = True
            time.sleep(min(0.2, self.lease_ttl_s / 5))
        ticks, crashed = 0, False
        try:
            while True:
                result = orch.tick()
                ticks += 1
                if result != "running":
                    break
                if crash_after_ticks is not None and ticks >= crash_after_ticks:
                    crashed = True
                    result = "interrupted"
                    break
                if max_ticks is not None and ticks >= max_ticks:
                    result = "max_ticks"
                    break
                time.sleep(self.poll_interval_s)
        finally:
            if not crashed:
                orch.release()
        self.log(f"orchestrate: {result} after {ticks} ticks, {orch.spawned} calls spawned")
        self.last_outcome = result
        return result

    def compose(self, outcome: str | None = None) -> dict[str, Any] | None:
        from flora_deploy.export.compose import ComposeError, compose

        outcome = outcome or self.last_outcome
        partial = outcome != "done"
        try:
            res = compose(self.store, self.job_id, partial=partial)
        except ComposeError as e:
            if partial:
                self.log(f"compose: refused ({e})")
                return None
            self.log(f"compose: incomplete job ({e}); composing what exists")
            res = compose(self.store, self.job_id, partial=True)
        s = res.summary
        for stage, c in s["stages"].items():
            self.log(f"compose: {stage}: {c['records']} records, missing {c['missing']}, "
                     f"excused {c['excused']}")
        return s

    def close(self) -> None:
        if self._spawner is not None:
            self._spawner.close()
            self._spawner = None
            self._shim.__exit__(None, None, None)

    def result(self, outcome: str, summary: dict[str, Any] | None, t0: float) -> LocalRunResult:
        assert self.settings is not None
        return LocalRunResult(
            job_id=self.job_id, config_hash=self.settings.config_hash,
            store_root=str(self.store_root) if self.store_root is not None else None,
            state=completion_state(outcome, summary), outcome=outcome, summary=summary,
            index=self.index_summary,
            spawner=self._spawner.stats() if self._spawner is not None else {},
            faults=list(self.faults.log), events=list(self.events),
            wall_s=round(time.monotonic() - t0, 3))

    def run(self, *, resume: bool = False, crash_after_ticks: int | None = None
            ) -> LocalRunResult:
        """index → plan → orchestrate → compose (the spawner stays open; :meth:`close` it)."""
        t0 = time.monotonic()
        self.index()
        self.plan()
        if resume:
            self.resume()
        outcome = self.orchestrate(crash_after_ticks=crash_after_ticks)
        summary = self.compose(outcome) if outcome != "lease_held" else None
        return self.result(outcome, summary, t0)


def run_local(cfg: Any, *, config_path: str | Path | None = None,
              store: str | Path | Store | None = None, job_id: str | None = None,
              fake_models: bool = True, faults: FaultPlan | str | Iterable[str] | None = None,
              resume: bool = False, budget_usd: float | None = None, log: Log | None = None,
              verbose: bool = False, **kw: Any) -> LocalRunResult:
    """Run one job end to end in this process (module docstring). ``cfg``: ``FloraConfig`` or a
    ``flora.yaml`` path. ``faults``: e.g. ``"preempt:segcap-00000,oom:ep003"``
    (``local.faults``). Other keywords go to :class:`LocalRunner`."""
    runner = LocalRunner(cfg, config_path=config_path, store=store, job_id=job_id,
                         fake_models=fake_models, faults=faults, budget_usd=budget_usd,
                         log=log, verbose=verbose, **kw)
    try:
        return runner.run(resume=resume)
    finally:
        runner.close()


def summary_lines(res: LocalRunResult) -> list[str]:
    """Human-readable result summary (the CLI prints it)."""
    lines = [f"job {res.job_id} (config {res.config_hash[:12]}): {res.state.upper()} "
             f"in {res.wall_s:.1f} s (orchestrator: {res.outcome})"]
    sp = res.spawner
    if sp:
        lines.append(f"calls {sp['calls']} {sp['states']}, containers {sp['containers']}, "
                     f"preemptions {sp['preemptions']}, crashes {sp['crashes']}, python "
                     f"failures {sp['python_failures']}")
    if res.faults:
        fired = _count(f"{f['fault']}:{f['target']}" for f in res.faults)
        lines.append("faults fired: " + ", ".join(f"{k} x{v}" for k, v in fired.items()))
    s = res.summary
    if s:
        for stage, c in s["stages"].items():
            share = "-" if c["certified_share"] is None else f"{c['certified_share']:.0%}"
            lines.append(f"  {stage}: {c['records']}/{c['expected']} records, missing "
                         f"{c['missing']}, excused {c['excused']}, certified {share}")
        lines.append(f"  shards {s['shards']}, poison pills {len(s['poison_pills'])}, errors "
                     f"{s['errors']['by_kind'] or 0}, duplicates resolved "
                     f"{s['duplicates_resolved']}, conflicts {len(s['conflicts'])}")
    if res.state == "budget_stopped":
        lines.append("budget stop: continue with `flora run --local --fake-models --resume "
                     "[--budget USD]`")
    if res.store_root:
        lines.append(f"store: {res.store_root}; read results with `flora results --job "
                     f"{res.job_id} --stage caption --store {res.store_root}`")
    return lines


def digest_outputs(summary: Mapping[str, Any] | None) -> str:
    """One digest over the composed outputs (determinism checks)."""
    outs = (summary or {}).get("outputs", {})
    return hashlib.sha256(json.dumps({k.rsplit("/", 1)[-1]: v["sha256"]
                                      for k, v in outs.items()}, sort_keys=True).encode()
                          ).hexdigest()
