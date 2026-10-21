"""``run_local``: one whole job in this process, with the real job system and fake models.

``index (local) → plan → orchestrate → compose``, every step the production code:

* **index**: ``jobs.indexing.run_index`` over the dataset (``flora index --local``), receipts in
  the job store; resumable;
* **plan**: ``jobs.planning.plan_job``, the planning path ``sdk.submit`` uses: the config hash
  *and its stage hashes*, one manifest per config (``JobPaths.config_manifest``; a re-run of the
  same config reuses it, a changed config re-plans under its own hash and re-runs only the
  kinds whose stage hash changed), exclusions recorded in the manifest. It then writes the
  job's **run spec** (``sdk.write_run_spec``, ``app_name: local``, a ``local`` section with the
  config path and the lease directory), so ``flora status / report / cancel / resume`` and
  ``sdk.open_job`` work on local jobs (review 6 H3/H4);
* **orchestrate**: the real ``jobs.orchestrator.Orchestrator`` driven by
  ``jobs.driver.TickDriver`` (store refresh, transient-error backoff), as on Modal, with the real
  store adapters and the production budget gate (``jobs.deps.build_deps``: rate-table
  estimates, container startup and scale-down tail, pinned Modal prices: what the job *would*
  cost on Modal), the Modal lease name ``orchestrator/<job>`` over a file KV
  (:class:`~flora_deploy.local.spawner.FileKV`) and a :class:`~flora_deploy.local.spawner.
  LocalSpawner` running the real ``SegcapShardRunner`` / ``RewardShardRunner`` with fake engines
  (``local.engines``);
* **compose**: ``export.compose.compose`` for this config hash, with the config's
  ``max_episode_error_rate`` and ``output.formats`` (``partial`` unless the orchestrator said
  ``done`` or ``completed_with_errors``).

Completion (``LocalRunResult.state``): the orchestrator's verdict, combined with the composed
summary by ``export.compose.job_state``, the function ``sdk.Job.status`` uses, so a local run
ends in the state the same job would on Modal (review 6 H2):

* ``done``: resolved, and per kind the excluded episodes stay within
  ``retries.max_episode_error_rate``;
* ``completed_with_errors``: resolved, but more episodes were excluded than that (NVIDIA-semantics
  rejections, D15 poison pills, episodes left out of the plan);
* ``failed``: a shard failed terminally (respawns exhausted) or depends on one;
* ``budget_stopped`` (``flora run --local --resume [--budget N]`` or ``flora resume`` continues),
  ``cancelled``, ``interrupted`` (``crash_after_ticks``: a simulated orchestrator crash, lease not
  released), ``lease_held`` (another orchestrator holds the job's lease).

**Differences from Modal that remain** (review 6 M5): calls run on threads of this process, so
they die with it (a restarted orchestrator in a new process sees them ``expired`` and respawns,
where Modal calls outlive the orchestrator); ``terminate`` is a graceful cancel (the worker
flushes and marks the start preempted, while Modal may kill the container before
``@modal.exit``); the CUDA-fatal container exit is not modelled; spend comes from wall time of
fake calls (seconds), so budget tests exercise the projection (startup, scale-down, rate
table), not realistic spend. The ``timeout`` fault models ``FunctionTimeoutError``.

Fake records are ``certified: false`` with ``uncertified_reason: fake_models`` (review 6 M7).

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

from flora_deploy.jobs.orchestrator import DictLease, Orchestrator, utc, write_control
from flora_deploy.jobs.settings import RunSettings
from flora_deploy.jobs.store import JobPaths, Store
from flora_deploy.jobs.types import ControlRecord, Manifest, safe_key
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


FAKE_UNCERTIFIED = "fake_models"


def uncertify(data: bytes) -> bytes:
    """A part file's records with ``certified: false`` / ``uncertified_reason: fake_models``
    (an uncertified record keeps its own reason)."""
    out = []
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("certified", False):
            rec["certified"], rec["uncertified_reason"] = False, FAKE_UNCERTIFIED
        out.append(json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return "".join(x + "\n" for x in out).encode()


class FaultStore:
    """``Store`` wrapper given to every local worker runner: delivers the ``fail`` fault at a
    container start's first part / errors / audit / receipt write (``faults.maybe_fail``), and
    marks every record a fake engine produced uncertified (:func:`uncertify`: fake labels must
    never pass a ``certified`` filter). Reads and every other write pass straight through."""

    _AREAS = ("/parts/", "/errors/", "/audit/", "/receipts/")

    def __init__(self, store: Store):
        self.store = store

    def put_bytes(self, path: str, data: bytes) -> None:
        if any(a in path for a in self._AREAS):
            maybe_fail(path.rsplit("/", 1)[-1])
        if "/parts/" in path and path.endswith(".jsonl"):
            data = uncertify(data)
        self.store.put_bytes(path, data)

    def get_bytes(self, path: str) -> bytes:
        return self.store.get_bytes(path)

    def exists(self, path: str) -> bool:
        return self.store.exists(path)

    def list(self, prefix: str):
        return self.store.list(prefix)


def fake_config(cfg: Any, *, input_digest: str | None = None) -> Any:
    """D11 ``ConfigHash`` (job hash + stage hashes) of ``cfg`` with fake derived-weight receipts
    and image fingerprints."""
    from flora_deploy.config.hashing import config_hash

    receipts = {"captioner_bf16": FAKE_RECEIPT, "tempseg_merged": FAKE_RECEIPT}
    return config_hash(cfg, derived_receipts=receipts, input_digest=input_digest,
                       image_fingerprint=lambda name: f"{FAKE_RECEIPT}:{name}")


def fake_config_hash(cfg: Any, *, input_digest: str | None = None) -> str:
    """D11 job hash of ``cfg`` with fake derived-weight receipts and image fingerprints."""
    return fake_config(cfg, input_digest=input_digest).hash


def completion_state(outcome: str, summary: Mapping[str, Any] | None) -> str:
    """``done`` / ``completed_with_errors`` / ``failed`` / ... (module docstring): the
    orchestrator's outcome through ``export.compose.job_state``, as ``sdk.Job.status``. A
    resolved job whose compose found it incomplete (or could not compose) is ``failed``."""
    from flora_deploy.export.compose import job_state

    if outcome not in STATES:
        return "failed"
    state = job_state(outcome, dict(summary) if summary is not None else None)
    if state in ("done", "completed_with_errors") and (
            summary is None or summary.get("status") not in ("complete",
                                                             "completed_with_errors")):
        return "failed"
    return state


def store_path(store: str | Path) -> Path:
    """A local store directory from ``--store``: a path or a ``file://`` URI (``Path`` folds
    ``file:///abs`` to ``file:/abs``; both are accepted)."""
    from urllib.parse import unquote, urlparse

    text = str(store)
    if text.startswith("file:"):
        u = urlparse(text)
        if u.netloc not in ("", "localhost") or not u.path:
            raise LocalRunError(f"--store {text!r}: file:// URIs take an absolute path "
                                "(file:///dir)")
        return Path(unquote(u.path))
    if "://" in text:
        raise LocalRunError(f"--store {text!r}: local runs need a directory or file:///dir")
    return Path(text).expanduser()


class _CrashableLease:
    """A lease whose ``release`` can be withheld (``crash_after_ticks``: an orchestrator that
    dies keeps its lease until the TTL runs out)."""

    def __init__(self, inner: Any, crash: bool):
        self.inner, self.crash = inner, crash

    def acquire(self, holder: str, ttl_s: float) -> bool:
        return self.inner.acquire(holder, ttl_s)

    def renew(self, holder: str, ttl_s: float) -> bool:
        return self.inner.renew(holder, ttl_s)

    def verify(self, holder: str) -> bool:
        return self.inner.verify(holder)

    def release(self, holder: str) -> None:
        if not self.crash:
            self.inner.release(holder)

    def current(self) -> Any:
        return self.inner.current()


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
    unfired: list[str] = field(default_factory=list)  # injected faults that never fired

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
        try:
            safe_key(self.job_id)
        except ValueError as e:
            raise LocalRunError(f"--job {self.job_id!r}: {e}") from None
        if budget_usd is not None and not budget_usd > 0:
            raise LocalRunError(f"--budget {budget_usd:g}: the budget must be positive")
        self.faults = faults if isinstance(faults, FaultPlan) else FaultPlan.parse(faults)
        self.poll_interval_s, self.lease_ttl_s = poll_interval_s, lease_ttl_s
        self.spawn_grace_s = spawn_grace_s
        self.log: Log = log or (lambda m: None)
        self.verbose = verbose
        self.events: list[dict] = []

        if store is None:
            store = base / "flora-local-store"
        if isinstance(store, str | Path):
            self.store_root: Path | None = store_path(store).resolve()
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
        if name == "lease_held" and getattr(self, "_lease_logged", False):
            return  # one line per wait, not one per poll
        if name == "lease_held":
            self._lease_logged = True
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
        """Plan as ``sdk.submit`` does (``jobs.planning.plan_job``), then write the run spec."""
        from flora_deploy import sdk
        from flora_deploy.config.schema import to_run_settings
        from flora_deploy.jobs.planning import NothingToPlan, plan_job

        if self.index_summary is None:
            self.index()
        assert self.index_summary is not None
        digest = str(self.index_summary["dataset"].get("revision") or "") or None
        ch = fake_config(self.cfg, input_digest=digest)
        settings = to_run_settings(self.cfg, self.job_id, config_hash=ch.hash,
                                   stage_hashes=dict(ch.stage_hashes))
        spec_path = sdk.run_spec_path(self.job_id, ch.hash)
        old = json.loads(self.store.get_bytes(spec_path)) if self.store.exists(spec_path) \
            else None
        if self.budget_usd is not None:
            settings = dataclasses.replace(settings, budget_usd=float(self.budget_usd))
        elif old is not None:  # the budget sticks, as a submitted job's does
            settings = dataclasses.replace(settings,
                                           budget_usd=float(old["settings"]["budget_usd"]))
        self.settings = settings
        try:
            m = plan_job(self.store, settings, envelope=self.cfg.pipeline.envelope,
                         created_utc=utc(time.time()),
                         default_task=self.cfg.pipeline.episode_instruction.text, log=self.log)
        except NothingToPlan as e:
            raise LocalRunError(f"{e} (see the index summary)") from None
        self.manifest = m
        self.write_spec(old)
        return m

    def write_spec(self, old: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The run spec ``sdk.open_job`` reads (``app_name: local``; see the module docstring)."""
        from flora_deploy import sdk

        assert self.settings is not None
        s = self.settings
        spec = {
            "schema": sdk.RUN_SPEC_SCHEMA, "job_id": self.job_id, "config_hash": s.config_hash,
            "stage_hashes": dict(s.stage_hashes), "app_name": "local", "environment": None,
            "store": self.store_root.as_uri() if self.store_root is not None else None,
            "preset": s.preset, "worker_classes": {k: "local" for k in s.pools},
            "factory": None, "deployment": None, "settings": sdk.settings_to_dict(s),
            "submitted_utc": (old or {}).get("submitted_utc") or utc(time.time()),
            "launches": list((old or {}).get("launches", [])),
            "local": {"config": (str(self.config_path.resolve()) if self.config_path is not None
                                 else None),
                      "lease": str((self.local_dir / "lease").resolve()),
                      "fake_models": True},
        }
        sdk.write_run_spec(self.store, spec)
        return spec

    def job(self) -> Any:
        """This job as an ``sdk.Job`` (status, report, cancel, the resume checks)."""
        from flora_deploy import sdk

        assert self.settings is not None
        spec = sdk.read_run_spec(self.store, self.job_id, self.settings.config_hash)
        return sdk.Job(self.job_id, self.settings.config_hash, spec.get("store") or "",
                       self.store, spec)

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
            return SegcapShardRunner(store, FakeSegcapEngine(), s, call_id=call_id)
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
        """The production gate (``jobs.deps.default_budget_gate``, which ``build_deps`` and
        ``sdk.Job`` use): rate-table estimates with container startup and the scale-down tail
        (review 6 M4)."""
        return deps.budget

    def resume(self, reason: str = "flora run --local --resume") -> None:
        """Lift a ``budget_stop`` (raise the budget with ``budget_usd=``), refusing what
        ``flora resume`` refuses (``sdk.Job.check_resume``: e.g. the same budget that stopped
        it, while the projection is still over it)."""
        from flora_deploy import sdk

        assert self.settings is not None
        job = self.job()
        try:
            v = job.status()
            job.check_resume(v, self.budget_usd)
        except sdk.FloraError as e:
            raise LocalRunError(str(e)) from None
        if v.state != "budget_stopped":
            self.log(f"resume: job is {v.state}; nothing to lift")
            return
        write_control(self.store, JobPaths(self.job_id), ControlRecord(
            "resume", self.job_id, reason, utc(time.time()),
            {"config_hash": self.settings.config_hash, "budget_usd": self.settings.budget_usd}))
        self.log("resume: budget stop lifted")

    def orchestrate(self, *, max_ticks: int | None = None,
                    crash_after_ticks: int | None = None) -> str:
        """Run the real orchestrator until the job is idle. ``crash_after_ticks``: stop after
        that many ticks *without* releasing the lease (an orchestrator crash; returns
        ``interrupted``). Returns the orchestrator's result."""
        from flora_deploy.jobs.deps import build_deps
        from flora_deploy.jobs.driver import TickDriver

        if self.manifest is None:
            self.plan()
        assert self.settings is not None
        deps = build_deps(self.store, self.settings)
        crash = crash_after_ticks is not None
        lease = _CrashableLease(DictLease(FileKV(self.local_dir / "lease"),
                                          f"orchestrator/{self.job_id}"), crash)
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
        limit = crash_after_ticks if crash else max_ticks
        out = TickDriver(orch).run_until_idle(sleep=time.sleep,
                                              poll_interval_s=self.poll_interval_s,
                                              max_ticks=limit)
        result: str = out.result
        if crash and result == "max_ticks":
            result = "interrupted"  # the lease stays held until its TTL runs out
        elif crash:
            lease.inner.release(orch.holder)  # finished before the simulated crash
        self.log(f"orchestrate: {result} after {out.ticks} ticks, {orch.spawned} calls spawned")
        self.last_outcome = result
        return result

    def compose(self, outcome: str | None = None) -> dict[str, Any] | None:
        from flora_deploy.export.compose import ComposeError, compose

        assert self.settings is not None
        outcome = outcome or self.last_outcome
        partial = outcome not in ("done", "completed_with_errors")
        kw = dict(config_hash=self.settings.config_hash,
                  max_error_rate=self.settings.max_episode_error_rate,
                  formats=self.settings.output_formats)
        try:
            res = compose(self.store, self.job_id, partial=partial, **kw)
        except ComposeError as e:
            if partial:
                self.log(f"compose: refused ({e})")
                return None
            self.log(f"compose: incomplete job ({e}); composing what exists")
            res = compose(self.store, self.job_id, partial=True, **kw)
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
            wall_s=round(time.monotonic() - t0, 3),
            unfired=[f.spec() for f in self.faults.faults if f.fired == 0])

    def run(self, *, resume: bool = False, crash_after_ticks: int | None = None
            ) -> LocalRunResult:
        """index → plan → orchestrate → compose (the spawner stays open; :meth:`close` it)."""
        t0 = time.monotonic()
        self.index()
        self.plan()
        if resume:
            self.resume()
        outcome = self.orchestrate(crash_after_ticks=crash_after_ticks)
        for spec in (f.spec() for f in self.faults.faults if f.fired == 0):
            self.log(f"warning: injected fault {spec} never fired (no shard or episode "
                     "matched it: shard ids are content hashes, see the manifest; episode "
                     "targets are uids or episode ids, e.g. episode_000001)")
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


def launch_local(job: Any, *, reason: str = "resume") -> str:
    """``sdk.Job.launch`` of a local job (``flora resume`` on a ``flora run --local`` job): run
    its orchestrator in this process until the job stops, then compose. The config file the
    run spec names must still hash to the job's config."""
    from flora_deploy import sdk

    local = job.spec.get("local") or {}
    cfg_path = local.get("config")
    if not cfg_path or not Path(cfg_path).is_file():
        where = cfg_path or "an in-memory config"
        raise sdk.FloraError(f"job {job.job_id} ran locally from {where}, which is gone; rerun "
                             "it with `flora run --local --fake-models --resume --config "
                             "<flora.yaml>`")
    root = getattr(job.store, "root", None)
    runner = LocalRunner(cfg_path, store=Path(root) if root is not None else job.store,
                         job_id=job.job_id)
    try:
        runner.index()
        runner.plan()
        assert runner.settings is not None
        if runner.settings.config_hash != job.config_hash:
            raise sdk.FloraError(f"{cfg_path} now hashes to {runner.settings.config_hash}, not "
                                 f"the job's {job.config_hash}; `flora run --local` runs the "
                                 "changed config")
        outcome = runner.orchestrate()
        if outcome != "lease_held":
            runner.compose(outcome)
    finally:
        runner.close()
    call_id = f"local-{uuid.uuid4().hex[:10]}"
    spec = sdk.read_run_spec(job.store, job.job_id, job.config_hash)
    spec.setdefault("launches", []).append({"call_id": call_id, "reason": reason,
                                            "utc": utc(time.time()), "outcome": outcome})
    sdk.write_run_spec(job.store, spec)
    job.spec = spec
    return call_id


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
    if res.unfired:
        lines.append("warning: faults that never fired: " + ", ".join(res.unfired))
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
                     "--budget USD` (or `flora resume <job> --budget-usd USD --store <store>`)")
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
