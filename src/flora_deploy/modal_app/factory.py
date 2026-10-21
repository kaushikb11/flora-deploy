"""App factory (PLAN.md D10, §7): one deployed Modal app per config, built by ``build_app(cfg)``.

Modal finds a container's code by module + qualname, so every worker class and function is
defined **undecorated at module scope** (``reference_app.ReferenceWorker``,
``workers_app.RewardWorker``, the orchestrator wrappers below) and decorated here, from the
config's pools; never ``serialized=True``. In the container, Modal imports the module, finds the
raw object by name and, since no module-level app has the deployed app's name, uses a fresh app
(``modal/_runtime/user_code_imports.py`` ``get_active_app_fallback``).

Two steps, so the same truth drives ``build_app``, ``flora deploy --dry-run`` and ``--emit``:

1. :func:`plan_app` turns a validated ``FloraConfig`` into an :class:`AppPlan`: plain, JSON-able
   option sets per Modal object (GPU, cpu, ``memory=(request, limit)``, containers,
   ``@modal.concurrent`` inputs, timeouts, retries, scaledown, Volumes, secret *names*, env).
   It contacts nothing and refuses what cannot run: shard kinds without a worker class (balanced
   ``segment``/``caption`` until P1b-1, ``reward.rubric``), uncertified presets without
   ``allow_uncertified`` (D9/D16), local stores, decorated (module-scope) workers.
2. :func:`materialize` (``build_app``) applies the decorators to a fresh ``modal.App``. Still
   offline: images, Volumes and secrets are lazy until ``app.deploy()``.

Objects, by stable Modal name (``ModalSpawner`` looks classes up by ``AppPlan.worker_classes``):

- one worker class per worker *class* used: ``segcap`` -> ``ReferenceWorker`` (img_hfref, the
  exact worker), ``reward.terminal``/``reward.dense`` -> ``RewardWorker`` (img_vllm). Kinds that
  share a class share its containers: their pools must agree on GPU and ``max_inputs``; cpu and
  memory take the larger, ``max_containers`` the sum (the orchestrator caps each kind itself);
- ``orchestrator_run`` (CPU, nonpreemptible, 24 h, ``modal.Retries``) and ``reconcile``; with
  ``reconcile_cron`` also ``reconcile_watched`` on a ``modal.Cron``. Their continuation/revive
  spawns go to this app (``FLORA_APP_NAME``), and the watch list is per app;
- ``index_run_shard`` when ``INDEX_FUNCTION`` resolves to an undecorated function (P1a-6b);
  otherwise it is listed in ``AppPlan.skipped``.

The app name is ``flora-<project>-<digest of the stage hashes>`` (D11), so two configs that
produce different outputs never share an app, and resource-only edits (GPU, containers) redeploy
the same app in place.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlparse

from flora_deploy.jobs.settings import PoolSettings

APP_ENV = "FLORA_APP_NAME"
WATCH_ENV = "FLORA_WATCH_DICT"
STORE_MOUNT_ROOT = "/vol"
DEFAULT_BUCKET_SECRET = "flora-bucket"
MEMORY_LIMIT_FACTOR = 1.25  # memory=(request, limit): a deterministic OOM at the limit (D15)
WORKER_RETRIES = {"max_retries": 2, "initial_delay": 10.0, "backoff_coefficient": 2.0}
ORCH_RETRIES = {"max_retries": 3, "initial_delay": 30.0, "backoff_coefficient": 2.0}
MIN_WORKER_TIMEOUT_S = 30 * 60
MAX_TIMEOUT_S = 24 * 3600
RECONCILE_EVERY = "*/10 * * * *"

IMAGE_BUILDERS = {"img_cpu": "cpu_image", "img_hfref": "hfref_image", "img_vllm": "vllm_image"}


class DeployError(ValueError):
    """The config cannot be deployed as is (the message says why and what to do)."""


class WorkerUnavailable(DeployError):
    """A shard kind's worker class does not exist (yet) in this checkout."""


# ---------------------------------------------------------------------------------------------
# Registry: shard kind -> undecorated module-scope worker class (by name, imported lazily so the
# factory works whether or not a worker module has landed).


@dataclass(frozen=True)
class WorkerRef:
    module: str
    name: str  # class name == Modal class name (ModalSpawner's lookup key)
    image: str  # modal_app.images key; must match the preset's EngineIdentity.image
    startup_timeout_s: int
    scaledown_window_s: int = 300
    cache: bool = False  # mount flora-cache (compile/Triton caches)
    recipe: Literal["segcap", "reward"] = "reward"  # container env recipe


WORKERS: dict[str, WorkerRef] = {
    # The fast exact worker (S1); its run_shard lands with P1a-5s.
    "segcap": WorkerRef("flora_deploy.modal_app.reference_app", "ReferenceWorker", "img_hfref",
                        startup_timeout_s=8 * 60, recipe="segcap"),
    "reward.terminal": WorkerRef("flora_deploy.modal_app.workers_app", "RewardWorker",
                                 "img_vllm", startup_timeout_s=15 * 60, cache=True),
    "reward.dense": WorkerRef("flora_deploy.modal_app.workers_app", "RewardWorker", "img_vllm",
                              startup_timeout_s=15 * 60, cache=True),
}
PENDING_WORKERS: dict[str, str] = {
    "segment": "the balanced SegmentWorker lands with P1b-1",
    "caption": "the balanced CaptionWorker lands with P1b-1",
    "reward.rubric": "there is no RubricWorker yet",
}
INDEX_FUNCTION: tuple[str, str] = ("flora_deploy.modal_app.index_app", "index_run_shard")


def resolve(module: str, attr: str, *, want: Literal["class", "function"]) -> Any:
    """The raw module-scope object, or ``WorkerUnavailable`` / ``DeployError``."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise WorkerUnavailable(f"{module} is not importable ({e})") from None
    obj = getattr(mod, attr, None)
    if obj is None:
        raise WorkerUnavailable(f"{module} defines no {attr!r}")
    import modal

    if isinstance(obj, modal.Cls | modal.Function):
        raise DeployError(f"{module}.{attr} is decorated at module scope; the app factory needs it "
                          "undecorated (PLAN.md D10) so it can decorate it for this app")
    if want == "class" and not isinstance(obj, type):
        raise DeployError(f"{module}.{attr} is not a class")
    if want == "function" and not callable(obj):
        raise DeployError(f"{module}.{attr} is not a function")
    if getattr(obj, "__qualname__", attr) != attr:
        raise DeployError(f"{module}.{attr} is not defined at module scope")
    return obj


# ---------------------------------------------------------------------------------------------
# Job store as seen from containers


@dataclass(frozen=True)
class StoreSpec:
    uri: str  # normalized: modal-volume://<name>[/sub] or s3|r2|gs://bucket[/prefix]
    kind: Literal["volume", "bucket"]
    volume: str | None = None
    mount: str | None = None  # where the store Volume is mounted
    root: str = ""  # volume: path under the mount; bucket: key prefix
    bucket: str | None = None
    secret: str | None = None  # Modal Secret *name* with the bucket credentials

    def env(self) -> dict[str, str]:
        if self.kind == "volume":
            root = f"{self.mount}/{self.root}" if self.root else str(self.mount)
            return {"FLORA_STORE": self.uri, "FLORA_STORE_KIND": "volume",
                    "FLORA_STORE_ROOT": root, "FLORA_STORE_VOLUME": str(self.volume),
                    "FLORA_VOLUME_MOUNT": str(self.mount)}
        return {"FLORA_STORE": self.uri, "FLORA_STORE_KIND": "bucket",
                "FLORA_STORE_BUCKET": str(self.bucket), "FLORA_STORE_PREFIX": self.root}


_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def parse_store(uri: str, *, bucket_secret: str = DEFAULT_BUCKET_SECRET) -> StoreSpec:
    """``modal-volume://name[/sub]`` (or ``volume://``) or ``s3|r2|gs://bucket[/prefix]``."""
    u = urlparse(uri)
    sub = u.path.strip("/")
    if u.scheme in ("modal-volume", "volume"):
        if not _VOLUME_NAME.match(u.netloc or ""):
            raise DeployError(f"store {uri!r}: modal-volume://<volume name>[/path]")
        norm = f"modal-volume://{u.netloc}" + (f"/{sub}" if sub else "")
        return StoreSpec(norm, "volume", volume=u.netloc, mount=f"{STORE_MOUNT_ROOT}/{u.netloc}",
                         root=sub)
    if u.scheme in ("s3", "r2", "gs"):
        if not u.netloc:
            raise DeployError(f"store {uri!r}: {u.scheme}://<bucket>[/prefix]")
        if not bucket_secret:
            raise DeployError("a bucket store needs a Modal Secret with its credentials")
        return StoreSpec(f"{u.scheme}://{u.netloc}" + (f"/{sub}" if sub else ""), "bucket",
                         root=sub, bucket=u.netloc, secret=bucket_secret)
    raise DeployError(f"store {uri!r}: Modal containers cannot reach a local store; use "
                      "modal-volume://<name> or a bucket (s3://, r2://, gs://)")


# ---------------------------------------------------------------------------------------------
# The plan (pure data)


@dataclass(frozen=True)
class VolumeMount:
    name: str
    read_only: bool = False
    create: bool = False  # create_if_missing


@dataclass(frozen=True)
class ObjectSpec:
    """One Modal object and the exact decorator options it gets."""

    kind: Literal["cls", "function"]
    name: str  # Modal name (class name / function name)
    module: str  # where the undecorated object lives
    attr: str
    image: str
    cpu: float
    memory: tuple[int, int] | int  # MiB: (request, limit) for workers
    timeout: int
    gpu: str | None = None
    startup_timeout: int | None = None
    min_containers: int | None = None
    max_containers: int | None = None
    scaledown_window: int | None = None
    max_inputs: int | None = None  # @modal.concurrent(max_inputs=...) when > 1
    retries: dict[str, float] | None = None
    volumes: dict[str, VolumeMount] = field(default_factory=dict)
    secrets: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    nonpreemptible: bool = False
    schedule: str | None = None  # cron
    region: str | None = None
    shard_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppPlan:
    app_name: str
    project: str
    environment: str
    preset: str
    config_hash: str
    stage_hashes: dict[str, str]
    image_fingerprints: dict[str, str]
    worker_classes: dict[str, str]  # shard kind -> Modal class name (for ModalSpawner)
    store: StoreSpec
    objects: tuple[ObjectSpec, ...]
    tags: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # what is not deployed, and why
    watch_dict: str = ""

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)


def _digest(stage_hashes: Mapping[str, str]) -> str:
    blob = json.dumps(dict(sorted(stage_hashes.items())), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def app_name_for(project: str, stage_hashes: Mapping[str, str]) -> str:
    """``flora-<project slug>-<12 hex of the stage hashes>`` (<= 60 chars, Modal-safe)."""
    slug = re.sub(r"[^a-z0-9-]+", "-", project.lower()).strip("-")[:40].strip("-") or "job"
    return f"flora-{slug}-{_digest(stage_hashes)}"


def worker_timeout_s(shard_target_gpu_minutes: float) -> int:
    """Per-call timeout: 3x the shard target, at least 30 min, at most Modal's 24 h."""
    return int(min(MAX_TIMEOUT_S, max(MIN_WORKER_TIMEOUT_S,
                                      math.ceil(shard_target_gpu_minutes * 60 * 3))))


def memory_mib(gib: float) -> tuple[int, int]:
    request = int(math.ceil(gib * 1024))
    return request, int(math.ceil(request * MEMORY_LIMIT_FACTOR))


def gpu_string(pool: PoolSettings) -> str:
    return pool.gpu if pool.gpu_count == 1 else f"{pool.gpu}:{pool.gpu_count}"


def merge_pools(kinds: list[str], pools: Mapping[str, PoolSettings]) -> PoolSettings:
    """One class serving several kinds: same GPU and inputs; larger cpu/memory; summed
    containers (each kind still stays within its own pool, the orchestrator caps it)."""
    ps = [pools[k] for k in kinds]
    first = ps[0]
    for k, p in zip(kinds[1:], ps[1:], strict=True):
        if (p.gpu, p.gpu_count, p.max_inputs) != (first.gpu, first.gpu_count, first.max_inputs):
            raise DeployError(
                f"{kinds[0]} and {k} run in one worker class, so their resources must agree on "
                f"gpu, gpu_count and max_inputs (got {first.gpu}x{first.gpu_count}/"
                f"{first.max_inputs} vs {p.gpu}x{p.gpu_count}/{p.max_inputs})")
    return replace(first, max_containers=sum(p.max_containers for p in ps),
                   cpu_cores=max(p.cpu_cores for p in ps),
                   memory_gib=max(p.memory_gib for p in ps))


def _recipe_env(recipe: str, determinism: str) -> dict[str, str]:
    if recipe == "segcap":
        from flora_deploy.accel import determinism as det
        from flora_deploy.modal_app.warm import linked_offline_env

        return {**linked_offline_env(), **det.container_env(determinism)}
    from flora_deploy.modal_app import rewarder_app as ra

    return {ra.IMAGE_FP_ENV: ra.IMAGE_FP, **ra.cache_env()}


def plan_app(
    cfg: Any,
    *,
    store: str | StoreSpec | None = None,
    bucket_secret: str = DEFAULT_BUCKET_SECRET,
    derived_receipts: Mapping[str, str] | None = None,
    allow_uncertified: bool = False,
    min_containers: int = 0,
    reconcile_cron: str | None = None,
    determinism: str = "nvidia",
    registry: Mapping[str, WorkerRef] | None = None,
    index_function: tuple[str, str] | None = INDEX_FUNCTION,
    missing_workers: Literal["error", "skip"] = "error",
    hash_kwargs: Mapping[str, Any] | None = None,
) -> AppPlan:
    """Everything ``build_app`` will register, as data. Contacts nothing.

    ``cfg``: a validated ``FloraConfig``. ``store``: the job store URI (default: the config's
    ``output.sink``). ``derived_receipts``: as for ``config.hashing.config_hash`` (exact segcap
    needs ``captioner_bf16``). ``missing_workers="skip"`` leaves out kinds whose worker class is
    not importable yet (listed in ``skipped``) instead of refusing; ``flora deploy`` never skips.
    ``hash_kwargs``: extra ``config_hash`` arguments (tests)."""
    from flora_deploy.accel import determinism as det
    from flora_deploy.config.hashing import config_hash
    from flora_deploy.config.presets import find_certificates, get_preset, needs_certificate
    from flora_deploy.config.schema import to_run_settings
    from flora_deploy.modal_app import orchestrator_app as oa
    from flora_deploy.modal_app.images import image_fingerprint

    if determinism not in det.MODES:
        raise DeployError(f"determinism {determinism!r}: use one of {', '.join(det.MODES)}")
    if min_containers < 0:
        raise DeployError("min_containers must be >= 0")
    reg = dict(WORKERS if registry is None else registry)
    ch = config_hash(cfg, derived_receipts=derived_receipts, **dict(hash_kwargs or {}))
    rs = to_run_settings(cfg, "deploy", config_hash=ch.hash)
    preset = get_preset(cfg.preset)
    st = store if isinstance(store, StoreSpec) else parse_store(store or cfg.output.sink,
                                                                bucket_secret=bucket_secret)

    if needs_certificate(cfg.preset) and not (cfg.allow_uncertified or allow_uncertified):
        certs = find_certificates(cfg.preset, rs.pools)
        missing = [f"{k} on {p.gpu}" for k, p in rs.pools.items() if k not in certs]
        if missing:
            raise DeployError(f"preset {cfg.preset!r} has no Gate certificate for "
                              f"{', '.join(missing)}; deploy 'exact' or allow uncertified")

    name = app_name_for(cfg.project, ch.stage_hashes)
    watch_dict = f"flora-watch-{_digest(ch.stage_hashes)}"
    skipped: dict[str, str] = {}
    by_class: dict[str, list[str]] = {}
    refs: dict[str, WorkerRef] = {}
    for kind in rs.pools:
        ref = reg.get(kind)
        if ref is None:
            why = PENDING_WORKERS.get(kind, "no worker class is registered for it")
            if missing_workers == "skip":
                skipped[kind] = why
                continue
            raise WorkerUnavailable(f"shard kind {kind!r} cannot be deployed: {why}")
        if ref.image != preset.engines[kind].image:
            raise DeployError(f"{kind}: worker {ref.name} uses {ref.image} but preset "
                              f"{cfg.preset!r} runs it on {preset.engines[kind].image}")
        try:
            resolve(ref.module, ref.name, want="class")
        except WorkerUnavailable as e:
            if missing_workers == "skip":
                skipped[kind] = str(e)
                continue
            raise WorkerUnavailable(f"shard kind {kind!r} cannot be deployed: {e}") from None
        prev = refs.setdefault(ref.name, ref)
        if (prev.module, prev.image, prev.recipe) != (ref.module, ref.image, ref.recipe):
            raise DeployError(f"worker class {ref.name} is registered inconsistently")
        by_class.setdefault(ref.name, []).append(kind)

    base_env = {
        APP_ENV: name, "FLORA_PROJECT": cfg.project, "FLORA_PRESET": cfg.preset,
        "FLORA_ENVELOPE": cfg.pipeline.envelope, "FLORA_CONFIG_HASH": ch.hash,
        "FLORA_LOG_LEVEL": cfg.observability.log_level,
        "FLORA_HEARTBEAT_S": f"{cfg.observability.heartbeat_s:g}",
        **st.env(),
    }
    store_volumes = ({str(st.mount): VolumeMount(str(st.volume), create=True)}
                     if st.kind == "volume" else {})
    secrets = (st.secret,) if st.secret else ()

    from flora_deploy.layout import CACHE_MOUNT, CACHE_VOLUME, WEIGHTS_MOUNT, WEIGHTS_VOLUME

    objects: list[ObjectSpec] = []
    timeout = worker_timeout_s(rs.shard_target_gpu_minutes)
    for cls_name, kinds in by_class.items():
        ref = refs[cls_name]
        pool = merge_pools(kinds, rs.pools)
        volumes = {str(WEIGHTS_MOUNT): VolumeMount(WEIGHTS_VOLUME, read_only=True)}
        if ref.cache:
            volumes[str(CACHE_MOUNT)] = VolumeMount(CACHE_VOLUME, create=True)
        volumes.update(store_volumes)
        env = {**_recipe_env(ref.recipe, determinism), **base_env,
               "FLORA_DETERMINISM": determinism, "FLORA_SHARD_KINDS": ",".join(kinds),
               "FLORA_STAGE_HASHES": json.dumps({k: ch.stage_hashes[k] for k in kinds},
                                                sort_keys=True)}
        objects.append(ObjectSpec(
            kind="cls", name=cls_name, module=ref.module, attr=ref.name, image=ref.image,
            gpu=gpu_string(pool), cpu=float(pool.cpu_cores), memory=memory_mib(pool.memory_gib),
            timeout=timeout, startup_timeout=ref.startup_timeout_s,
            min_containers=min_containers, max_containers=pool.max_containers,
            scaledown_window=ref.scaledown_window_s,
            max_inputs=pool.max_inputs if pool.max_inputs > 1 else None,
            retries=dict(WORKER_RETRIES), volumes=volumes, secrets=secrets,
            env=dict(sorted(env.items())), region=cfg.modal.region, shard_kinds=tuple(kinds)))

    orch_env = dict(sorted({**base_env, WATCH_ENV: watch_dict}.items()))
    here = __name__
    common = {"module": here, "image": "img_cpu", "cpu": float(oa.CPU_CORES),
              "memory": int(oa.MEMORY_MIB), "volumes": dict(store_volumes), "secrets": secrets,
              "env": orch_env, "region": cfg.modal.region}
    objects.append(ObjectSpec(kind="function", name="orchestrator_run", attr="orchestrator_run",
                              timeout=int(oa.TIMEOUT_S), nonpreemptible=True,
                              retries=dict(ORCH_RETRIES), **common))
    objects.append(ObjectSpec(kind="function", name="reconcile", attr="reconcile",
                              timeout=15 * 60, **common))
    if reconcile_cron:
        cron = RECONCILE_EVERY if reconcile_cron.strip().lower() in ("1", "on", "true",
                                                                       "yes") else reconcile_cron
        objects.append(ObjectSpec(kind="function", name="reconcile_watched",
                                  attr="reconcile_watched", timeout=15 * 60,
                                  schedule=cron.strip(), **common))
    else:
        skipped["reconcile_watched"] = "no reconcile cron requested"

    if index_function is not None:
        try:
            resolve(*index_function, want="function")
        except DeployError as e:
            skipped["index"] = f"index function not deployed: {e}"
        else:
            objects.append(ObjectSpec(
                kind="function", name="index_run_shard", module=index_function[0],
                attr=index_function[1], image="img_cpu", cpu=2.0, memory=(4096, 8192),
                timeout=60 * 60, retries=dict(WORKER_RETRIES), max_containers=16,
                volumes=dict(store_volumes), secrets=secrets, env=dict(sorted(base_env.items())),
                region=cfg.modal.region))

    images_used = sorted({o.image for o in objects})
    tags = {**cfg.modal.tags, "flora-project": cfg.project, "flora-preset": cfg.preset}
    return AppPlan(
        app_name=name, project=cfg.project, environment=cfg.modal.environment,
        preset=cfg.preset, config_hash=ch.hash, stage_hashes=dict(ch.stage_hashes),
        image_fingerprints={i: image_fingerprint(i) for i in images_used},
        worker_classes={k: n for n, ks in by_class.items() for k in ks},
        store=st, objects=tuple(objects), tags=dict(sorted(tags.items())), skipped=skipped,
        watch_dict=watch_dict)


# ---------------------------------------------------------------------------------------------
# Materialize


def _volume(m: VolumeMount) -> Any:
    import modal

    v = modal.Volume.from_name(m.name, version=2, create_if_missing=m.create)
    return v.read_only() if m.read_only else v


def modal_kwargs(o: ObjectSpec) -> dict[str, Any]:
    """The ``app.cls`` / ``app.function`` keyword arguments of one object."""
    import modal

    from flora_deploy.modal_app import images

    kw: dict[str, Any] = {
        "image": getattr(images, IMAGE_BUILDERS[o.image])(),
        "cpu": o.cpu,
        "memory": o.memory,
        "timeout": o.timeout,
        "volumes": {p: _volume(m) for p, m in o.volumes.items()},
        "secrets": [modal.Secret.from_name(s) for s in o.secrets],
        "env": dict(o.env),
    }
    optional = {"gpu": o.gpu, "startup_timeout": o.startup_timeout,
                "min_containers": o.min_containers, "max_containers": o.max_containers,
                "scaledown_window": o.scaledown_window, "region": o.region}
    kw.update({k: v for k, v in optional.items() if v is not None})
    if o.retries:
        kw["retries"] = modal.Retries(max_retries=int(o.retries["max_retries"]),
                                      initial_delay=o.retries["initial_delay"],
                                      backoff_coefficient=o.retries["backoff_coefficient"])
    if o.nonpreemptible:
        kw["nonpreemptible"] = True
    if o.schedule:
        kw["schedule"] = modal.Cron(o.schedule)
    return kw


def materialize(plan: AppPlan) -> Any:
    """A fresh ``modal.App`` with every object of ``plan`` registered (offline)."""
    import modal

    app = modal.App(plan.app_name, tags=dict(plan.tags))
    for o in plan.objects:
        target = resolve(o.module, o.attr, want="class" if o.kind == "cls" else "function")
        kw = modal_kwargs(o)
        if o.kind == "cls":
            if o.name != o.attr:
                raise DeployError(f"class {o.attr} would register as {o.attr}, not {o.name}")
            wrapped = modal.concurrent(max_inputs=o.max_inputs)(target) if o.max_inputs else target
            app.cls(**kw)(wrapped)
        else:
            app.function(name=o.name, **kw)(target)
    return app


def build_app(cfg: Any, **kwargs: Any) -> Any:
    """``modal.App`` for ``cfg`` (a ``FloraConfig``, or an ``AppPlan``); see :func:`plan_app`."""
    plan = cfg if isinstance(cfg, AppPlan) else plan_app(cfg, **kwargs)
    return materialize(plan)


def registered(app: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(functions, classes)`` registered on ``app`` by Modal name (offline inspection)."""
    from modal._utils.async_utils import synchronizer

    state = synchronizer._translate_in(app)._local_state
    return dict(state.functions), dict(state.classes)


def worker_classes(cfg: Any, **kwargs: Any) -> dict[str, str]:
    """Shard kind -> Modal class name, as ``ModalSpawner(app_name, worker_classes)`` wants."""
    return plan_app(cfg, **kwargs).worker_classes


def deployment_record(plan: AppPlan, **extra: Any) -> dict[str, Any]:
    """What ``flora deploy`` stores at :func:`deployment_path` (and ``--dry-run`` prints)."""
    from flora_deploy import LOGIC_VERSION, __version__
    from flora_deploy.config.pins import RUNTIME

    return {
        "schema": "flora_deployment_v1",
        "app_name": plan.app_name,
        "project": plan.project,
        "environment": plan.environment,
        "preset": plan.preset,
        "config_hash": plan.config_hash,
        "stage_hashes": plan.stage_hashes,
        "image_fingerprints": plan.image_fingerprints,
        "worker_classes": plan.worker_classes,
        "functions": [o.name for o in plan.objects if o.kind == "function"],
        "classes": [o.name for o in plan.objects if o.kind == "cls"],
        "skipped": plan.skipped,
        "store": plan.store.uri,
        "watch_dict": plan.watch_dict,
        "flora_version": __version__,
        "logic_version": LOGIC_VERSION,
        "modal_client": RUNTIME["modal_client"],
        **extra,
    }


def deployment_path(app_name: str) -> str:
    from flora_deploy.jobs.types import safe_key

    return f"flora/deployments/{safe_key(app_name)}.json"


# ---------------------------------------------------------------------------------------------
# Orchestrator functions (undecorated, module scope; decorated by ``materialize``). They spawn
# continuations / revivals in their own deployed app and use that app's watch list.


def _own(name: str) -> Any:
    import modal

    return modal.Function.from_name(os.environ[APP_ENV], name)


def _watch_kv() -> Any:
    import modal

    return modal.Dict.from_name(os.environ[WATCH_ENV], create_if_missing=True)


def orchestrator_run(job_id: str, factory: str, generation: int = 0,
                     watch: bool = False) -> dict[str, Any]:
    """One orchestrator generation (``orchestrator_app.run_generation``) in this app."""
    from flora_deploy.modal_app import orchestrator_app as oa

    return oa.run_generation(job_id, factory, generation, watch,
                             spawn_successor=_own("orchestrator_run").spawn,
                             watch_kv=_watch_kv())


def reconcile(job_id: str, factory: str, revive: bool = True) -> dict[str, Any]:
    """Stateless status from files; revives this app's orchestrator if it died."""
    from flora_deploy.modal_app import orchestrator_app as oa

    return oa._reconcile_job(job_id, factory, revive,
                             spawn_run=_own("orchestrator_run").spawn)


def reconcile_watched() -> dict[str, Any]:
    """Scheduled reconciler over this app's watch list."""
    from flora_deploy.modal_app import orchestrator_app as oa

    kv = _watch_kv()
    spawn = _own("orchestrator_run").spawn

    def one(job_id: str, factory: str, revive: bool) -> dict[str, Any]:
        return oa._reconcile_job(job_id, factory, revive, spawn_run=spawn)

    result = oa.reconcile_all(list(kv.items()), one, lambda job_id: kv.pop(job_id, None))
    print(json.dumps({"event": "reconcile_watched", "jobs": result}, sort_keys=True,
                     default=str), flush=True)
    return result
