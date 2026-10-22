"""Python SDK: submit and watch jobs on a deployed ``flora-deploy`` app (the CLI's engine).

Mirrors the CLI one to one::

    import flora_deploy as fd
    from flora_deploy import sdk

    options = sdk.plan("flora.yaml", video_hours=2.0)        # flora plan
    job = sdk.submit("flora.yaml")                           # flora run --detach
    job.status()                                            # flora status <job>
    job.wait(poll_s=30)                                     # flora run (streaming) + compose
    records = job.results("caption")                        # flora results
    job.report()                                            # flora report <job>
    job.resume(budget_usd=20)                               # flora resume <job>
    job.cancel()                                            # flora cancel <job>

**What ``submit`` does.** It plans the config's app offline (``modal_app.factory.plan_app``: the
same hashes ``flora deploy`` used), reads the deployment record ``flora deploy`` wrote
(``flora/deployments/<app>.json``) and refuses when there is none (:class:`NotDeployed`) or its
config hash differs (:class:`DeploymentMismatch`: redeploy, or pass the same ``--receipt`` /
``--determinism``). It then plans the job from the episode index ``flora index`` wrote (a stored
manifest of the same config is reused), writes the **run spec** (``flora/<job>/runs/<config
hash>.json``: the job's ``RunSettings``, worker classes, launches) and spawns the deployed app's
``orchestrator_run(job, FACTORY)``. Inside the app, :func:`deployed_deps` (the ``FACTORY``) reads
that spec back, so the deployed orchestrator runs exactly the settings the client planned.

**Status** (:meth:`Job.status`) is derived from the store's files only (``jobs.reconciler``),
plus the orchestrator lease (alive or not), spend from the real budget gate
(``jobs.deps.default_budget_gate``: container receipts, the call history, pinned prices) and an
ETA from ``planner/rates.json`` calibrated by finished calls. A resolved job is composed by the
client (:meth:`Job.compose`, also run by :meth:`Job.wait`): the deployed app does not compose.

**Backends.** Every Modal call goes through a :class:`Backend` (default :class:`ModalBackend`),
so the SDK and the CLI are testable without Modal (``tests/fakes/flora_backend.py``). A
``modal-volume://`` store is read and written from the client through the Volume API
(:class:`RemoteVolumeStore`); inside a container that mounts it, through the mount.

Exit codes the CLI derives from a state (:data:`EXIT_CODES`): ``done`` 0, ``failed`` 1,
``completed_with_errors`` 3, ``budget_stopped`` 4, ``cancelled`` (and other stops) 5; 2 is a usage
error, a refusal or an internal error. Same scheme as ``flora run --local`` and ``flora compose``.

**Local jobs.** ``flora run --local`` writes a run spec too (``app_name: local`` and a ``local``
section: config path, lease directory). :func:`open_job` opens it like any other; the lease is
the local file KV, :meth:`Job.resume` runs the orchestrator in this process
(``local.runner.launch_local``), :meth:`Job.cancel` writes the record (the calls of a local run
live in its process) and :meth:`Job.logs` refuses (no Modal logs).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flora_deploy.jobs.store import (
    JobPaths,
    Store,
    fake_job_id,
    is_fake_job_id,
    prefetch,
    prefetch_areas,
)
from flora_deploy.jobs.stores.client_volume import ClientVolumeStore

RUN_SPEC_SCHEMA = "flora_run_v1"
FACTORY = "flora_deploy.sdk:deployed_deps"  # the deps factory the deployed orchestrator loads
LEASE_DICT = "flora-orchestrator-leases"  # modal_app.orchestrator_app.LEASE_DICT
ORCHESTRATOR_FUNCTION = "orchestrator_run"
TERMINAL = ("done", "completed_with_errors", "failed", "cancelled")
STOPPED = (*TERMINAL, "budget_stopped")
EXIT_CODES = {"done": 0, "failed": 1, "completed_with_errors": 3, "budget_stopped": 4,
              "cancelled": 5}


def exit_code(state: str) -> int:
    """CLI exit code of a job state (non-final states: 0)."""
    if state in EXIT_CODES:
        return EXIT_CODES[state]
    return 0 if state in ("planned", "running", "composing", "indexed") else 5


# ----------------------------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------------------------


class FloraError(RuntimeError):
    """A refusal with a message that says what to do (CLI exit code 2)."""


class NotDeployed(FloraError):
    pass


class DeploymentMismatch(FloraError):
    pass


class JobNotFound(FloraError):
    pass


# ----------------------------------------------------------------------------------------------
# Backend: every Modal call, behind one narrow interface
# ----------------------------------------------------------------------------------------------


class Backend(Protocol):
    def store(self, uri: str, *, environment: str | None = None) -> Store:
        """The job store at ``uri`` as this process can reach it."""
        ...

    def spawn(self, app_name: str, function: str, *args: Any, environment: str | None = None,
              **kwargs: Any) -> str:
        """``modal.Function.from_name(app, function).spawn(*args, **kwargs)``; the call id."""
        ...

    def cancel_call(self, call_id: str, *, terminate: bool = True) -> None: ...

    def lease_kv(self, environment: str | None = None) -> Any:
        """The orchestrators' lease Dict (``flora-orchestrator-leases``), or ``None``."""
        ...

    def logs(self, app_name: str, *, environment: str | None = None, follow: bool = False,
             tail: int | None = None, function_call: str | None = None) -> Iterator[str]: ...


# ``Store`` over a Volume through the Modal client (no mount): ``jobs.stores.ClientVolumeStore``.
RemoteVolumeStore = ClientVolumeStore


class ModalBackend:
    """The real backend (modal 1.6.2). Never used by tests."""

    def store(self, uri: str, *, environment: str | None = None) -> Store:
        """The mount inside a container that has it, else the Modal client's Volume API
        (``export.results.store_from_uri``)."""
        from flora_deploy.export.results import store_from_uri

        if uri.startswith("volume://"):
            uri = "modal-volume://" + uri[len("volume://"):]
        return store_from_uri(uri, environment=environment)

    def spawn(self, app_name: str, function: str, *args: Any, environment: str | None = None,
              **kwargs: Any) -> str:
        import modal

        fn = modal.Function.from_name(app_name, function, environment_name=environment)
        return fn.spawn(*args, **kwargs).object_id

    def cancel_call(self, call_id: str, *, terminate: bool = True) -> None:
        import modal

        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=terminate)

    def lease_kv(self, environment: str | None = None) -> Any:
        import modal

        return modal.Dict.from_name(LEASE_DICT, environment_name=environment,
                                    create_if_missing=True)

    def logs(self, app_name: str, *, environment: str | None = None, follow: bool = False,
             tail: int | None = None, function_call: str | None = None) -> Iterator[str]:
        cmd = ["modal", "app", "logs", app_name, "--timestamps", "--show-function-call-id"]
        if environment:
            cmd += ["--env", environment]
        if follow:
            cmd.append("--follow")
        elif tail:
            cmd += ["--tail", str(tail)]
        if function_call:
            cmd += ["--function-call", function_call]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.rstrip("\n")


_backend: Backend | None = None


def get_backend() -> Backend:
    return _backend if _backend is not None else ModalBackend()


def set_backend(backend: Backend | None) -> None:
    """Replace the process-wide default backend (tests); ``None`` restores Modal."""
    global _backend
    _backend = backend


# ----------------------------------------------------------------------------------------------
# Run spec (what the client planned; the deployed orchestrator reads it back)
# ----------------------------------------------------------------------------------------------


def run_spec_path(job_id: str, config_hash: str) -> str:
    from flora_deploy.jobs.types import safe_key

    return f"{JobPaths(job_id).root}/runs/{safe_key(config_hash)}.json"


def run_pointer_path(job_id: str) -> str:
    """The job's latest run spec (a copy): ``flora status <job>`` finds the config from it."""
    return f"{JobPaths(job_id).root}/run.json"


def settings_to_dict(settings: Any) -> dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(settings)))


def settings_from_dict(d: Mapping[str, Any]) -> Any:
    from flora_deploy.jobs.settings import PoolSettings, RunSettings

    kw = dict(d)
    kw["pools"] = {k: PoolSettings(**v) for k, v in dict(kw.get("pools", {})).items()}
    if "output_formats" in kw:
        kw["output_formats"] = tuple(kw["output_formats"])
    known = {f.name for f in dataclasses.fields(RunSettings)}
    return RunSettings(**{k: v for k, v in kw.items() if k in known})


def write_run_spec(store: Store, spec: Mapping[str, Any]) -> None:
    data = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    store.put_bytes(run_spec_path(spec["job_id"], spec["config_hash"]), data)
    store.put_bytes(run_pointer_path(spec["job_id"]), data)


def read_run_spec(store: Store, job_id: str, config_hash: str | None = None) -> dict[str, Any]:
    path = run_spec_path(job_id, config_hash) if config_hash else run_pointer_path(job_id)
    if not store.exists(path):
        what = f"config {config_hash}" if config_hash else "any config"
        raise JobNotFound(f"job {job_id!r} has no run spec for {what} in this store (was it "
                          "submitted with `flora run`? check --store / --config)")
    spec = json.loads(store.get_bytes(path))
    if spec.get("schema") != RUN_SPEC_SCHEMA:
        raise FloraError(f"{path}: unknown run spec schema {spec.get('schema')!r}")
    return spec


def spec_is_fake(spec: Mapping[str, Any]) -> bool:
    """Whether a run spec is a fake-models run: ``fake_models`` (``submit``), or a local run
    (``flora run --local`` always runs the fake models)."""
    return bool(spec.get("fake_models") or (spec.get("local") or {}).get("fake_models"))


def check_job_namespace(job_id: str, spec: Mapping[str, Any]) -> None:
    """Refuse a run spec whose fake-ness disagrees with its job id (review 8b M5): fake-models
    runs live under ``<job>.fake`` (``jobs.store.fake_job_id``), real ones never do. A spec
    written before the namespace existed can be a fake run under the real id."""
    fake = spec_is_fake(spec)
    if fake == is_fake_job_id(job_id):
        return
    if fake:
        raise FloraError(f"job {job_id!r} holds a fake-models run (config "
                         f"{spec.get('config_hash')}), not real outputs; fake-models runs now "
                         f"use the job id {fake_job_id(job_id)!r}. Rerun it there, or submit "
                         "the real job under another --job")
    raise FloraError(f"job {job_id!r} is a fake-models job id but holds a real run (config "
                     f"{spec.get('config_hash')}); refusing to mix them")


def check_results_namespace(store: Store, job_id: str, config_hash: str | None = None) -> None:
    """``flora results`` / ``flora compose``: refuse a job whose run spec is the other kind
    (:func:`check_job_namespace`), and point a real job id with no run at its fake-models run.
    A job with no run spec at all (composed by hand) passes."""
    path = run_spec_path(job_id, config_hash) if config_hash else run_pointer_path(job_id)
    if store.exists(path):
        check_job_namespace(job_id, json.loads(store.get_bytes(path)))
        return
    if not is_fake_job_id(job_id) and store.exists(run_pointer_path(fake_job_id(job_id))):
        raise JobNotFound(f"job {job_id!r} has no run in this store; its fake-models run is "
                          f"job {fake_job_id(job_id)!r} (pass --job {fake_job_id(job_id)})")


def _container_store(env: Mapping[str, str]) -> Store:
    """The job store inside a deployed container (the env ``modal_app.factory`` sets)."""
    kind = env.get("FLORA_STORE_KIND", "volume")
    if kind == "bucket":
        from flora_deploy.jobs.stores import BucketStore

        return BucketStore(env["FLORA_STORE_BUCKET"], prefix=env.get("FLORA_STORE_PREFIX", ""),
                           endpoint_url=env.get("AWS_ENDPOINT_URL") or None)
    root = env.get("FLORA_STORE_ROOT")
    if not root:
        raise RuntimeError("FLORA_STORE_ROOT is not set (the app factory sets the store)")
    if kind == "local":
        from flora_deploy.jobs.stores import LocalStore

        return LocalStore(root)
    from flora_deploy.jobs.stores import VolumeStore

    name = env.get("FLORA_STORE_VOLUME")
    volume = None
    if name:
        import modal

        volume = modal.Volume.from_name(name, version=2)
    return VolumeStore(root, volume)


def deployed_deps(job_id: str, env: Mapping[str, str] | None = None) -> Any:
    """``FACTORY``: the ``OrchestratorDeps`` of ``job_id`` inside the deployed app.

    Reads the run spec of the app's config (``FLORA_CONFIG_HASH``) from the store, applies the
    deployment's warm pool / scaledown (``jobs.deps.deployment_settings_kw``) and builds the real
    adapters and budget gate (``jobs.deps.build_deps``) for the app's worker classes."""
    from flora_deploy.jobs.deps import build_deps, deployment_settings_kw

    env = os.environ if env is None else env
    store = _container_store(env)
    refresh = getattr(store, "refresh", None)
    if callable(refresh):
        refresh()
    spec = read_run_spec(store, job_id, env.get("FLORA_CONFIG_HASH") or None)
    settings = dataclasses.replace(settings_from_dict(spec["settings"]),
                                   **deployment_settings_kw(env))
    return build_deps(store, settings, worker_app=env.get("FLORA_APP_NAME") or spec["app_name"],
                      worker_classes=spec["worker_classes"])


# ----------------------------------------------------------------------------------------------
# Status, report
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JobView:
    """``flora status``: the reconciler's view plus spend, budget and ETA."""

    job_id: str
    config_hash: str
    app_name: str
    state: str
    counts: dict[str, dict[str, int]]  # kind -> shard status -> n
    active_calls: int
    orchestrator_alive: bool
    errors: tuple[str, ...]
    notes: tuple[str, ...]
    stuck: tuple[dict[str, Any], ...]
    spend_usd: float
    spend_source: str
    projected_usd: float
    budget_usd: float
    eta_s: float | None
    composed: bool
    summary_status: str | None = None
    episodes: int = 0
    video_s: float = 0.0
    # A `flora run --local` job: stand-in models in this process, nothing charged. Its spend
    # figures are the budget gate's simulation, never money.
    simulated: bool = False

    @property
    def exit_code(self) -> int:
        return exit_code(self.state)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _ledgers(store: Store, job_id: str, config_hash: str, manifest: Any) -> list[Any]:
    from flora_deploy.jobs.orchestrator import ShardLedger

    paths = JobPaths(job_id, config_hash=config_hash, stage_hashes=dict(manifest.stage_hashes))
    out = []
    for s in manifest.shards:
        p = paths.call(config_hash, s.kind, s.shard_id)
        if store.exists(p):
            try:
                out.append(ShardLedger.from_bytes(store.get_bytes(p)))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _eta_s(gate: Any, shards: Iterable[Any], done: set[str], settings: Any) -> float | None:
    """Rough wall time left: per kind, remaining container-seconds over its containers (plus one
    startup), from the gate's rate-table estimates and online calibration; kinds overlap, so the
    slowest kind bounds it."""
    cal = gate.effective_calibration()
    est = gate.estimator
    per_kind: dict[str, float] = {}
    for s in shards:
        if s.shard_id in done:
            continue
        pool = settings.pools[s.kind]
        per_kind[s.kind] = per_kind.get(s.kind, 0.0) + (
            est.container_seconds(s, pool) * cal.get(s.kind, 1.0))
    if not per_kind:
        return 0.0
    out = 0.0
    for kind, cs in per_kind.items():
        pool = settings.pools[kind]
        out = max(out, cs / max(pool.max_containers, 1) + est.startup_s(kind))
    return out if math.isfinite(out) else None


@dataclass
class Job:
    """A submitted job. Open one with :func:`open_job` (or get it from :func:`submit`)."""

    job_id: str
    config_hash: str
    store_uri: str
    store: Store
    spec: dict[str, Any]
    backend: Backend = field(default_factory=get_backend)
    clock: Callable[[], float] = time.time

    # -- derived pieces --------------------------------------------------------------------
    @property
    def app_name(self) -> str:
        return self.spec["app_name"]

    @property
    def environment(self) -> str | None:
        return self.spec.get("environment")

    @property
    def settings(self) -> Any:
        return settings_from_dict(self.spec["settings"])

    @property
    def is_local(self) -> bool:
        """A ``flora run --local`` job (its run spec has a ``local`` section): its orchestrator
        runs in a local process and its lease is a file KV next to the store."""
        return bool(self.spec.get("local"))

    def _refresh(self) -> None:
        fn = getattr(self.store, "refresh", None)
        if callable(fn):
            fn()

    def _prefetch(self, manifest: Any, areas: Iterable[str]) -> None:
        """Read the job's ``areas`` ahead, in parallel (``jobs.store.prefetch_areas``: one
        Volume round trip per shard file otherwise; a no-op on a mounted or local store)."""
        hashes = [self.config_hash, *dict(getattr(manifest, "stage_hashes", None) or {}).values()]
        prefetch_areas(self.store, JobPaths(self.job_id).root, hashes, areas)

    def _deps(self) -> Any:
        from flora_deploy.jobs.deps import build_deps

        return build_deps(self.store, self.settings, worker_app=self.app_name,
                          worker_classes=self.spec["worker_classes"], clock=self.clock)

    def _lease(self) -> Any:
        from flora_deploy.jobs.orchestrator import DictLease

        if self.is_local:
            from flora_deploy.local.spawner import FileKV

            return DictLease(FileKV(self.spec["local"]["lease"]), f"orchestrator/{self.job_id}",
                             clock=self.clock)
        kv = self.backend.lease_kv(self.environment)
        return None if kv is None else DictLease(kv, f"orchestrator/{self.job_id}",
                                                 clock=self.clock)

    def _summary_path(self) -> str:
        return JobPaths(self.job_id).output(self.config_hash, "summary", "json")

    def summary(self) -> dict[str, Any] | None:
        """The composed ``summary.json``, or ``None`` before compose."""
        p = self._summary_path()
        return json.loads(self.store.get_bytes(p)) if self.store.exists(p) else None

    # -- status ----------------------------------------------------------------------------
    def status(self) -> JobView:
        from flora_deploy.jobs.driver import budget_state_from_ledgers
        from flora_deploy.jobs.reconciler import reconcile

        self._refresh()
        deps = self._deps()
        settings = deps.settings
        self._prefetch(deps.manifests.load(), ("calls", "receipts"))
        st = reconcile(job_id=self.job_id, config_hash=self.config_hash, store=self.store,
                       manifests=deps.manifests, attempts=deps.attempts, lease=self._lease(),
                       clock=self.clock, max_starts=settings.max_starts_before_quarantine,
                       compose_done=lambda: self.store.exists(self._summary_path()),
                       max_error_rate=settings.max_episode_error_rate)
        manifest = deps.manifests.load()
        now = self.clock()
        gate = deps.budget
        ledgers = _ledgers(self.store, self.job_id, self.config_hash, manifest)
        spend_usd, source, projected, eta = 0.0, "none", 0.0, None
        if hasattr(gate, "spend_report"):
            gate.observe(budget_state_from_ledgers(ledgers, now))
            rep = gate.spend_report(now)
            spend_usd, source = rep.total_usd, rep.source
            in_flight: dict[str, int] = {}
            for led in ledgers:
                if led.status in ("spawning", "active"):
                    in_flight[led.kind] = in_flight.get(led.kind, 0) + 1
            projected = gate.projection(in_flight).total_usd
            done = _done_shards(self, manifest)
            if st.state not in STOPPED:
                eta = _eta_s(gate, manifest.shards, done, settings)
        from flora_deploy.export.compose import job_state

        summary = self.summary()
        state = job_state(st.state, summary)
        return JobView(
            job_id=self.job_id, config_hash=self.config_hash, app_name=self.app_name,
            state=state, counts=st.counts, active_calls=st.active_calls,
            orchestrator_alive=st.orchestrator_alive, errors=st.errors, notes=st.notes,
            stuck=tuple(dataclasses.asdict(s) for s in st.stuck_calls),
            spend_usd=round(spend_usd, 4), spend_source=source,
            projected_usd=round(max(projected, spend_usd), 4),
            budget_usd=float(settings.budget_usd), eta_s=eta,
            composed=summary is not None,
            summary_status=summary.get("status") if summary else None,
            episodes=len(manifest.episodes),
            video_s=round(sum(e.duration_s for e in manifest.episodes), 3),
            simulated=self.is_local)

    # -- lifecycle -------------------------------------------------------------------------
    def launch(self, reason: str = "submit") -> str:
        """Spawn the deployed orchestrator for this job (its lease keeps it single). A local job
        runs its orchestrator in this process instead, until the job stops
        (``local.runner.launch_local``)."""
        if self.is_local:
            from flora_deploy.local.runner import launch_local

            return launch_local(self, reason=reason)
        call_id = self.backend.spawn(self.app_name, ORCHESTRATOR_FUNCTION, self.job_id,
                                     self.spec.get("factory", FACTORY), 0, True,
                                     environment=self.environment)
        from flora_deploy.jobs.orchestrator import utc

        self.spec.setdefault("launches", []).append(
            {"call_id": call_id, "reason": reason, "utc": utc(self.clock())})
        write_run_spec(self.store, self.spec)
        return call_id

    def compose(self, *, partial: bool = False) -> dict[str, Any]:
        """Compose the job's parts into outputs (``flora compose``); returns the summary."""
        from flora_deploy.export.compose import compose

        settings = self.settings
        res = compose(self.store, self.job_id, config_hash=self.config_hash, partial=partial,
                      max_error_rate=settings.max_episode_error_rate,
                      formats=settings.output_formats)
        return res.summary

    def wait(self, poll_s: float = 30.0, *, timeout_s: float | None = None,
             on_status: Callable[[JobView], None] | None = None, compose: bool = True,
             dead_after_s: float = 300.0,
             sleep: Callable[[float], None] = time.sleep) -> JobView:
        """Poll :meth:`status` until the job stops (done, completed_with_errors, failed,
        cancelled, budget_stopped), composing a resolved job (``compose=True``). A dead
        orchestrator with work left is reported, never silently waited on forever: once no
        orchestrator has held the lease for ``dead_after_s`` (cold start included), it raises
        :class:`FloraError` (``flora resume`` restarts it)."""
        t0 = self.clock()
        dead_since: float | None = None
        while True:
            v = self.status()
            if compose and not v.composed and v.state in ("composing", "completed_with_errors"):
                self.compose()
                v = self.status()
            if on_status is not None:
                on_status(v)
            if v.state in STOPPED:
                return v
            if v.state in ("planned", "running") and not v.orchestrator_alive:
                dead_since = self.clock() if dead_since is None else dead_since
                if self.clock() - dead_since >= dead_after_s:
                    raise FloraError(f"job {self.job_id}: no orchestrator has held the lease "
                                     f"for {dead_after_s:.0f} s and work remains; `flora "
                                     "resume` restarts it")
            else:
                dead_since = None
            if timeout_s is not None and self.clock() - t0 >= timeout_s:
                return v
            sleep(poll_s)

    def results(self, stage: str) -> list[Any]:
        """Composed ``StageRecord``s of one stage (``flora results``)."""
        from flora_deploy.export.results import iter_records

        return list(iter_records(self.store, self.job_id, stage, config_hash=self.config_hash))

    def export(self, out: str | os.PathLike[str], *, to: str = "lerobot",
               **kw: Any) -> Any:
        """Write the labels into a copy of the job's dataset (``flora export``; keywords as
        :func:`export_job`). Returns the ``export.lerobot.ExportReport``."""
        if to != "lerobot":
            raise FloraError(f"unknown export target {to!r}; only lerobot is supported")
        check_results_namespace(self.store, self.job_id, self.config_hash)
        return export_job(self.store, self.job_id, out, config_hash=self.config_hash, **kw)

    def cancel(self, reason: str = "operator cancel") -> list[str]:
        """Write a ``cancel`` record (the orchestrator cancels its calls on its next tick). With
        no live orchestrator, cancels the recorded live calls directly. Returns the call ids
        cancelled here. A job that already ended (done, completed_with_errors, failed,
        cancelled) is left alone: cancelling it is a no-op (review 6 M1; the reconciler also
        ignores a cancel that arrives after every shard was satisfied)."""
        from flora_deploy.jobs.orchestrator import utc, write_control
        from flora_deploy.jobs.types import ControlRecord

        if self.status().state in TERMINAL:
            return []
        write_control(self.store, JobPaths(self.job_id), ControlRecord(
            "cancel", self.job_id, reason, utc(self.clock()), {"config_hash": self.config_hash}))
        lease = self._lease()
        if lease is not None and lease.current() is not None:
            return []
        if self.is_local:  # a local job's calls died with the process that ran them
            return []
        deps = self._deps()
        cancelled = []
        for led in _ledgers(self.store, self.job_id, self.config_hash, deps.manifests.load()):
            if led.status in ("spawning", "active") and led.call_id:
                self.backend.cancel_call(led.call_id, terminate=True)
                cancelled.append(led.call_id)
        return cancelled

    def resume(self, *, budget_usd: float | None = None,
               reason: str = "flora resume") -> str | None:
        """Lift a budget stop (optionally raising the budget) and restart a dead orchestrator.
        Refuses finished, failed and cancelled jobs. Returns the new orchestrator call id, or
        ``None`` when one is still alive."""
        from flora_deploy.jobs.orchestrator import utc, write_control
        from flora_deploy.jobs.types import ControlRecord

        v = self.status()
        self.check_resume(v, budget_usd)
        if budget_usd is not None:
            self.spec["settings"]["budget_usd"] = float(budget_usd)
            write_run_spec(self.store, self.spec)
        if v.state == "budget_stopped":
            write_control(self.store, JobPaths(self.job_id), ControlRecord(
                "resume", self.job_id, reason, utc(self.clock()),
                {"config_hash": self.config_hash, "budget_usd": self.settings.budget_usd}))
        if v.orchestrator_alive:
            return None
        return self.launch(reason="resume")

    def check_resume(self, v: JobView, budget_usd: float | None = None) -> None:
        """Raise :class:`FloraError` when :meth:`resume` would refuse (``flora resume`` and
        ``flora run --local --resume`` share it): a finished, failed or cancelled job; a budget
        that is not positive or below the spend; a budget stop whose projection is still over
        the budget without a new one."""
        if v.state in TERMINAL:
            hint = {"cancelled": "a cancelled job stays cancelled; submit it with a new --job",
                    "failed": "shards failed for good (see `flora report`); fix the cause and "
                              "submit with a new --job"}.get(v.state, "nothing left to run")
            raise FloraError(f"job {self.job_id} is {v.state}: {hint}")
        if budget_usd is not None:
            if budget_usd <= 0:
                raise FloraError("--budget-usd must be positive")
            if budget_usd < v.spend_usd:
                raise FloraError(f"--budget-usd {budget_usd:g} is below the spend so far "
                                 f"(${v.spend_usd:,.2f})")
        elif v.state == "budget_stopped" and v.projected_usd > v.budget_usd:
            raise FloraError(f"job {self.job_id} stopped at its budget (${v.budget_usd:,.2f}; "
                             f"projected ${v.projected_usd:,.2f}); pass --budget-usd to raise it")

    def report(self) -> dict[str, Any]:
        """``flora report``: spend by kind and container, $/video-hour, errors, certification and
        where the outputs are."""
        from flora_deploy.config.presets import needs_certificate
        from flora_deploy.jobs.budget import (
            calls_cost,
            container_cost,
            pinned_rates,
            read_container_receipts,
            region_multiplier_for,
        )
        from flora_deploy.jobs.driver import budget_state_from_ledgers

        v = self.status()
        settings = self.settings
        deps = self._deps()
        manifest = deps.manifests.load()
        self._prefetch(manifest, ("attempts",))  # the per-container rows read every marker
        now = self.clock()
        rates = pinned_rates(region_multiplier=region_multiplier_for(settings.region))
        calls = budget_state_from_ledgers(
            _ledgers(self.store, self.job_id, self.config_hash, manifest), now).calls
        from flora_deploy.jobs.pools import read_drains

        drains, prewarms = read_drains(self.store, JobPaths(self.job_id).root, self.config_hash)
        by_kind = {}
        for kind in sorted({c.kind for c in calls}):
            mine = [c for c in calls if c.kind == kind]
            by_kind[kind] = {"calls": len(mine), "usd": round(calls_cost(
                mine, settings, rates, now, scaledown_s=float(settings.scaledown_s),
                warm=settings.min_containers, drains=drains,
                prewarms={k: v for k, v in prewarms.items() if k == kind}), 4)}
        containers = [{"container_id": r.container_id, "gpu": r.gpu, "gpu_count": r.gpu_count,
                       "usd": round(container_cost(r, rates, now), 4),
                       "stopped": r.stopped_utc is not None, "source": "container_receipt"}
                      for r in read_container_receipts(self.store, self.job_id)]
        if not containers:  # no worker writes ContainerReceipts: derive them (review 11a H1)
            from flora_deploy.jobs.containers import job_container_rows

            containers = job_container_rows(
                self.store, self.job_id, self.config_hash, manifest, pools=settings.pools,
                scaledown_s=float(settings.scaledown_s), now=now, rates=rates,
                settings=settings)
        if self.is_local and not v.orchestrator_alive:  # its containers died with the process
            containers = [{**c, "stopped": True} for c in containers]
        done_ids = _done_shards(self, manifest)
        done_eps = {u for s in manifest.shards if s.shard_id in done_ids for u in s.episodes}
        dur = {e.uid: e.duration_s for e in manifest.episodes}
        done_video_h = sum(dur.get(u, 0.0) for u in done_eps) / 3600.0
        summary = self.summary()
        paths = JobPaths(self.job_id)
        outputs = {}
        certification: dict[str, Any] = {"preset": settings.preset,
                                         "needs_certificate": needs_certificate(settings.preset),
                                         "stages": {}}
        errors: dict[str, Any] = {"files": len(v.errors), "notes": list(v.notes)}
        if summary is not None:
            files = summary.get("files") or {
                f"{st}.{fmt}": paths.output(self.config_hash, st, fmt)
                for st in summary.get("stages", {}) for fmt in ("jsonl", "parquet")}
            for stage, c in summary.get("stages", {}).items():
                outputs[stage] = {key.rsplit(".", 1)[1]: f"{self.store_uri.rstrip('/')}/{p}"
                                  for key, p in sorted(files.items())
                                  if key.rsplit(".", 1)[0] == stage}
                certification["stages"][stage] = {
                    "certified_share": c.get("certified_share"),
                    "uncertified_reasons": c.get("uncertified_reasons", {})}
            errors.update({"by_kind": summary.get("errors", {}).get("by_kind", {}),
                           "poison_pills": summary.get("poison_pills", []),
                           "problems": summary.get("problems", []),
                           "excused": {s: c.get("excused", 0)
                                       for s, c in summary.get("stages", {}).items()},
                           "missing": {s: c.get("missing", 0)
                                       for s, c in summary.get("stages", {}).items()}})
        return {
            "job_id": self.job_id, "config_hash": self.config_hash, "app_name": self.app_name,
            "state": v.state, "store": self.store_uri,
            "spend": {"total_usd": v.spend_usd, "source": v.spend_source,
                      "simulated": v.simulated,
                      "budget_usd": v.budget_usd, "projected_usd": v.projected_usd,
                      "by_kind": by_kind, "containers": containers, "prices": rates.source},
            "video_hours": {"planned": round(v.video_s / 3600, 4),
                            "done": round(done_video_h, 4)},
            "usd_per_video_hour": (round(v.spend_usd / done_video_h, 2)
                                   if done_video_h > 0 and not v.simulated else None),
            "errors": errors, "certification": certification,
            "outputs": outputs,
            "summary": (f"{self.store_uri.rstrip('/')}/{self._summary_path()}"
                        if summary is not None else None),
            "composed": summary is not None,
        }

    def logs(self, *, kind: str | None = None, follow: bool = False,
             tail: int | None = 100) -> Iterator[str]:
        """App logs (``modal app logs``). ``kind``: ``orchestrator`` (the job's orchestrator
        launches) or a shard kind (its recorded calls, live ones first); default: the whole
        app (shared by every job of this config)."""
        if self.is_local:
            raise FloraError(f"job {self.job_id} ran locally (`flora run --local`): it has no "
                             "Modal logs; its events were printed by that run (use --verbose)")
        if kind is None:
            yield from self.backend.logs(self.app_name, environment=self.environment,
                                         follow=follow, tail=tail)
            return
        if kind == "orchestrator":
            ids = [x["call_id"] for x in reversed(self.spec.get("launches", []))]
        else:
            if kind not in self.spec["worker_classes"]:
                raise FloraError(f"--kind {kind!r}: this job runs "
                                 f"{', '.join(['orchestrator', *self.spec['worker_classes']])}")
            leds = [led for led in _ledgers(self.store, self.job_id, self.config_hash,
                                            self._deps().manifests.load()) if led.kind == kind]
            live = [led.call_id for led in leds if led.status in ("spawning", "active")
                    and led.call_id]
            old = [c["call_id"] for led in leds for c in reversed(led.calls)
                   if c.get("outcome") == "spawned" and c.get("call_id") not in live]
            ids = list(dict.fromkeys(live + old))
        if not ids:
            raise FloraError(f"job {self.job_id}: no {kind} calls recorded yet")
        for cid in ids[:1] if follow else ids[:5]:
            yield f"--- {kind} call {cid} ---"
            yield from self.backend.logs(self.app_name, environment=self.environment,
                                         follow=follow, tail=tail, function_call=cid)


def export_job(store: Store, job_id: str, out: str | os.PathLike[str], *,
               config_hash: str | None = None, source: str | os.PathLike[str] | None = None,
               stages: Sequence[str] | None = None, allow_uncertified: bool = False,
               allow_fake: bool = False, include_videos: bool = False,
               push_to_hub: str | None = None, private: bool = True, hub_api: Any = None,
               download: Any = None, keep_local_paths: bool = False) -> Any:
    """``flora export --to lerobot`` (docs/export.md): ``export.lerobot.export_job``, then an
    upload to the dataset repo ``push_to_hub`` (private unless ``private=False``; refused when
    the export names a local path). ``keep_local_paths`` records local paths in the export
    (local use only: it cannot be combined with ``push_to_hub``)."""
    from flora_deploy.export.hub import push_to_hub as push
    from flora_deploy.export.lerobot import ExportError
    from flora_deploy.export.lerobot import export_job as do_export

    if keep_local_paths and push_to_hub:
        raise ExportError("--keep-local-paths records this machine's paths in the export; it "
                          "cannot be combined with --push-to-hub")
    report = do_export(store, job_id, out, config_hash=config_hash, source=source,
                       stages=stages, allow_uncertified=allow_uncertified,
                       allow_fake=allow_fake, include_videos=include_videos, download=download,
                       keep_local_paths=keep_local_paths)
    if push_to_hub:
        report.pushed = push(out, push_to_hub, private=private, api=hub_api)
    return report


def _done_shards(job: Job, manifest: Any) -> set[str]:
    """Shards with a receipt (the reconciler only returns counts)."""
    paths = JobPaths(job.job_id, config_hash=job.config_hash,
                     stage_hashes=dict(manifest.stage_hashes))
    return {s.shard_id for s in manifest.shards
            if job.store.exists(paths.receipt(job.config_hash, s.kind, s.shard_id))}


# ----------------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------------


def _load_cfg(cfg: Any, *, allow_uncertified: bool = False,
              receipts: Mapping[str, str] | None = None) -> Any:
    """A ``FloraConfig`` (a path is loaded with its derived content digests,
    ``config.receipts.load_certified``, so a certified preset validates)."""
    if isinstance(cfg, str | Path):
        from flora_deploy.config.receipts import load_certified

        return load_certified(cfg, allow_uncertified=allow_uncertified, overrides=receipts)[0]
    return cfg


def store_uri_for(cfg: Any) -> str:
    """The job store of a config (``output.sink``; ``volume://`` -> ``modal-volume://``)."""
    sink = cfg.output.sink
    return "modal-volume://" + sink[len("volume://"):] if sink.startswith("volume://") else sink


def open_job(job_id: str, *, store: str | None = None, config: Any = None,
             config_hash: str | None = None, environment: str | None = None,
             backend: Backend | None = None) -> Job:
    """A submitted job, from its run spec. ``store``: the job store URI (default: the config's
    ``output.sink``); ``environment``: the Modal environment of a ``modal-volume://`` store
    (default: the config's, else Modal's default)."""
    be = backend or get_backend()
    cfg = _load_cfg(config) if config is not None else None
    if store is None:
        if cfg is None:
            raise FloraError("give --store (or FLORA_STORE) or --config to find the job store")
        store = store_uri_for(cfg)
    env = environment or (cfg.modal.environment if cfg is not None else None)
    return open_job_in(be.store(store, environment=env), job_id, store_uri=store,
                       config_hash=config_hash, backend=be)


def open_job_in(st: Store, job_id: str, *, store_uri: str, config_hash: str | None = None,
                backend: Backend | None = None) -> Job:
    """:func:`open_job` on an already opened job store (the CLI resolves which store)."""
    be = backend or get_backend()
    check_results_namespace(st, job_id, config_hash)
    spec = read_run_spec(st, job_id, config_hash)
    check_job_namespace(job_id, spec)
    return Job(job_id, spec["config_hash"], spec.get("store", store_uri), st, spec, be)


def _plan_deployment(cfg: Any, **kw: Any) -> Any:
    from flora_deploy.modal_app.factory import plan_app

    return plan_app(cfg, **kw)


def submit(cfg: Any, *, job_id: str | None = None, store: str | None = None,
           receipts: Mapping[str, str] | None = None, determinism: str | None = None,
           allow_uncertified: bool = False, budget_usd: float | None = None,
           bucket_secret: str = "flora-bucket", backend: Backend | None = None,
           log: Callable[[str], None] | None = None,
           planner: Callable[..., Any] | None = None, fake_models: bool = False) -> Job:
    """Start a job on the deployed app (``flora run --detach``); see the module docstring.

    ``cfg``: a ``FloraConfig`` or a path. ``receipts`` / ``determinism`` / ``allow_uncertified``
    / ``store`` / ``bucket_secret``: as passed to ``flora deploy`` (they decide the config hash
    and so which deployment runs it). ``receipts`` overrides the config's ``weights.derived``
    (recorded by ``flora prepare``; ``config.receipts.resolve``). ``budget_usd`` overrides
    ``modal.budget_usd``."""
    from flora_deploy.config.hashing import MissingReceiptError
    from flora_deploy.config.receipts import derived_content
    from flora_deploy.config.receipts import resolve as resolve_receipts
    from flora_deploy.config.schema import to_run_settings
    from flora_deploy.jobs.orchestrator import utc
    from flora_deploy.jobs.planning import NothingToPlan, plan_job
    from flora_deploy.modal_app.factory import DeployError, deployment_path

    say = log or (lambda m: None)
    be = backend or get_backend()
    cfg = _load_cfg(cfg, allow_uncertified=allow_uncertified, receipts=receipts)
    try:
        plan = (planner or _plan_deployment)(
            cfg, store=store, bucket_secret=bucket_secret,
            derived_receipts=resolve_receipts(cfg, receipts, fake_models=fake_models),
            derived_content=derived_content(cfg, receipts),
            allow_uncertified=allow_uncertified, determinism=determinism)
    except (DeployError, MissingReceiptError, ValueError) as e:
        raise FloraError(str(e)) from None
    job_id = job_id or cfg.project
    fake = bool(getattr(plan, "fake_models", False))
    if fake:  # review 8b M5: never the real job's manifest, run spec, index or lease
        job_id = fake_job_id(job_id)
    elif is_fake_job_id(job_id):
        raise FloraError(f"job id {job_id!r} is reserved for fake-models runs (`flora run "
                         "--fake-models`); pick another --job for a real run")
    st = be.store(plan.store.uri, environment=plan.environment)

    rec_path = deployment_path(plan.app_name)
    if not st.exists(rec_path):
        raise NotDeployed(f"{plan.app_name} is not deployed for this config (no {rec_path} in "
                          f"{plan.store.uri}); run `flora deploy` with the same --receipt / "
                          "--determinism first")
    record = json.loads(st.get_bytes(rec_path))
    if record.get("config_hash") != plan.config_hash or \
            dict(record.get("stage_hashes", {})) != dict(plan.stage_hashes):
        raise DeploymentMismatch(
            f"{plan.app_name} was deployed for config {record.get('config_hash')}, this config "
            f"hashes to {plan.config_hash}; redeploy (`flora deploy`) or pass the receipts and "
            "determinism it was deployed with")
    missing = sorted(set(plan.worker_classes) - set(record.get("worker_classes", {})))
    if missing:
        raise DeploymentMismatch(f"the deployment has no worker for {missing}; redeploy")

    settings = to_run_settings(cfg, job_id, config_hash=plan.config_hash,
                               stage_hashes=dict(plan.stage_hashes))
    if getattr(plan, "fake_models", False):  # CPU containers: priced as such (jobs.budget)
        settings = dataclasses.replace(settings, fake_models=True)
    if budget_usd is not None:
        settings = dataclasses.replace(settings, budget_usd=float(budget_usd))
    now = utc(time.time())
    if not st.exists(JobPaths(job_id).config_manifest(plan.config_hash)):
        # planning reads every index receipt, one per episode: read them ahead in parallel
        # (on a Volume through the client, one round trip each made submit take minutes)
        prefetch(st, st.list(f"{JobPaths(job_id).root}/index/"))
    try:
        plan_job(st, settings, envelope=cfg.pipeline.envelope, created_utc=now,
                 default_task=cfg.pipeline.episode_instruction.text, log=say)
    except NothingToPlan as e:
        raise FloraError(f"{e} in {plan.store.uri}; run `flora index --job {job_id}` first"
                         ) from None

    spec_path = run_spec_path(job_id, plan.config_hash)
    if st.exists(spec_path):
        spec = read_run_spec(st, job_id, plan.config_hash)
        check_job_namespace(job_id, spec)
        job = Job(job_id, plan.config_hash, plan.store.uri, st, spec, be)
        v = job.status()
        if v.state in STOPPED:
            hint = ("`flora resume` continues it" if v.state == "budget_stopped"
                    else "see `flora report`; submit with a new --job to run it again")
            raise FloraError(f"job {job_id} was already submitted for this config and is "
                             f"{v.state}: {hint}")
        if budget_usd is not None and float(budget_usd) != spec["settings"]["budget_usd"]:
            raise FloraError(f"job {job_id} is already running with budget "
                             f"${spec['settings']['budget_usd']:,.2f}; change it with "
                             "`flora resume --budget-usd`")
        if v.orchestrator_alive:
            say(f"job {job_id} is already running; attaching")
            return job
        say(f"job {job_id}: restarting its orchestrator")
        job.launch(reason="resubmit")
        return job

    spec = {
        "schema": RUN_SPEC_SCHEMA, "job_id": job_id, "config_hash": plan.config_hash,
        "stage_hashes": dict(plan.stage_hashes), "app_name": plan.app_name,
        "environment": plan.environment, "store": plan.store.uri, "preset": plan.preset,
        "worker_classes": dict(plan.worker_classes), "factory": FACTORY,
        "deployment": {"app_id": record.get("app_id"),
                       "deployed_utc": record.get("deployed_utc")},
        "settings": settings_to_dict(settings), "submitted_utc": now, "launches": [],
        "fake_models": fake,
    }
    write_run_spec(st, spec)
    job = Job(job_id, plan.config_hash, plan.store.uri, st, spec, be)
    call_id = job.launch()
    say(f"job {job_id}: orchestrator {call_id} on {plan.app_name}")
    return job


def plan(cfg: Any, *, video_hours: float | None = None, episodes: int | None = None,
         index: Mapping[str, Any] | str | Path | None = None, budget_usd: float | None = None,
         deadline_s: float | None = None, presets: Iterable[str] | None = None,
         max_gpus: int | None = None, rates_file: str | Path | None = None,
         live_prices: bool = False) -> list[Any]:
    """``flora plan``: 2-4 ``planner.options.PlanOption``s for a workload. ``index``: the
    ``summary.json`` ``flora index`` wrote (dict or path); else ``video_hours`` [``episodes``]."""
    from flora_deploy.config.presets import AVAILABLE_PRESETS
    from flora_deploy.jobs.budget import load_rates, modal_rates_provider, region_multiplier_for
    from flora_deploy.planner.options import (
        PlanRequest,
        WorkloadSummary,
        plan_options,
        with_pool_overrides,
    )
    from flora_deploy.planner.rates import RateTable

    cfg = _load_cfg(cfg)
    if index is not None:
        data = (json.loads(Path(index).read_text(encoding="utf-8"))
                if isinstance(index, str | Path) else index)
        workload = WorkloadSummary.from_dict(data)
    elif video_hours is not None:
        cams = len(cfg.input.cameras) if cfg.input.cameras else 1
        workload = WorkloadSummary.from_hours(video_hours, cameras=cams,
                                              pool=cfg.pipeline.embodiment, episodes=episodes)
    else:
        raise FloraError("give the workload: video_hours=H [episodes=N] or index=summary.json")
    table = RateTable.load(rates_file)
    rates = load_rates(modal_rates_provider if live_prices else None,
                       region_multiplier=region_multiplier_for(cfg.modal.region))
    req = PlanRequest(workload=workload, stages=cfg.pipeline.stages,
                      presets=tuple(presets or AVAILABLE_PRESETS), deadline_s=deadline_s,
                      budget_usd=budget_usd if budget_usd is not None else cfg.modal.budget_usd,
                      max_gpus=max_gpus or cfg.modal.max_gpus)
    req = with_pool_overrides(req, cfg.preset, cfg.pools())
    return plan_options(req, table=table, rates=rates)


__all__ = [
    "EXIT_CODES", "FACTORY", "Backend", "DeploymentMismatch", "FloraError", "Job", "JobNotFound",
    "JobView", "ModalBackend", "NotDeployed", "RemoteVolumeStore", "deployed_deps", "exit_code",
    "get_backend", "open_job", "plan", "set_backend", "submit",
]
