"""App factory (PLAN.md D10, §2): one deployed Modal app per config, built by ``build_app(cfg)``.

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
   It contacts nothing and refuses what cannot run: shard kinds without a worker class
   (``PENDING_WORKERS``), uncertified presets without
   ``allow_uncertified`` (D9/D16), local stores, decorated (module-scope) workers, and inputs the
   containers cannot read (a local ``input.path`` that was not staged to a Volume).
2. :func:`materialize` (``build_app``) applies the decorators to a fresh ``modal.App``. Still
   offline: images, Volumes and secrets are lazy until ``app.deploy()``.

Objects, by stable Modal name (``ModalSpawner`` looks classes up by ``AppPlan.worker_classes``):

- one worker class per worker *class* used: ``segcap`` -> ``ReferenceWorker`` (img_hfref, the
  exact worker), ``reward.terminal``/``reward.dense`` -> ``RewardWorker`` (img_vllm),
  ``reward.rubric`` -> ``RubricWorker`` (img_vllm, TP=2; the rubric spec arrives with the
  kinds' modes in the certification env), balanced
  ``segment`` -> ``SegmentWorker`` and ``caption`` -> ``CaptionWorker`` (img_vllm,
  ``balanced_app``: one pool each, L40S / H100! by default; their env adds the recorded
  derived-weight receipts ``FLORA_DERIVED_RECEIPTS``, the pool GPUs, the kinds' modes and the
  embodiment, which the workers verify and stamp). Kinds that
  share a class share its containers: their pools must agree on GPU and ``max_inputs``; cpu and
  memory take the larger, ``max_containers`` the sum (the orchestrator caps each kind itself);
- ``orchestrator_run`` (small preemptible CPU, 2 s idle, 24 h, ``modal.Retries``; sizes in
  ``jobs.pools``) and ``reconcile``; with ``reconcile_cron`` also ``reconcile_watched`` on a
  ``modal.Cron``. Their continuation/revive spawns go to this app (``FLORA_APP_NAME``), and the
  watch list is per app, as is the pool Dict (``FLORA_POOL_DICT``, ``<watch dict>-pools``)
  through which the app's jobs agree when a worker class may be drained (``jobs.pools``);
- ``index_run_shard`` when ``INDEX_FUNCTION`` resolves to an undecorated function (P1a-6b);
  otherwise it is listed in ``AppPlan.skipped``.

The app name is ``flora-<project>-<digest of the stage hashes>`` (D11), so two configs that
produce different outputs never share an app, and resource-only edits (GPU, containers) redeploy
the same app in place.

**Inputs (C1).** Index receipts hold dataset-relative video paths, so every worker gets
``FLORA_MEDIA_ROOT`` (:func:`media_for`): with ``input.stage_to: volume://<vol>/<dir>`` the
staged dataset on that Volume (mounted read-only at ``/media/<vol>``, or through the store mount
when it is the store's Volume), laid out as ``flora index`` stages it (``<dir>/<dataset key>
[/<subdir>]`` for a Hub dataset, ``<dir>`` itself for a local dataset uploaded there); without
``stage_to`` a Hub dataset is read from the Hub at its pinned revision
(``hf://datasets/<repo>@<rev>[/<subdir>]``). Workers check it in a preflight before building
their engine. The configured instruction travels as ``FLORA_INSTRUCTION_SOURCE`` /
``FLORA_INSTRUCTION_TEXT`` (C2).

**Caches (M7).** ``flora-cache`` is mounted read-only on workers (``FLORA_CACHE_MODE=ro``: a
local copy, never committed); ``warm_reward_cache`` (one container, read-write) is its only
writer.

**Timeouts (M5).** Per worker class, from ``planner/rates.json``: the wall time of a full shard
(``shard_target_gpu_minutes`` of placeholder-rate work, converted to video seconds and priced at
the measured container-s per video-s, upper quantile, times ``max_inputs``) plus startup, times
:data:`TIMEOUT_SAFETY`, at least 30 min.

**Fake models (H0, ``flora deploy --fake-models``).** ``plan_app(fake_models=True)`` plans the
same app (orchestrator, reconciler, index function, store and media Volumes, secrets, env,
timeouts, retries, container caps) but registers :data:`FAKE_WORKERS`: ``FakeSegcapWorker`` /
``FakeRewardWorker`` / ``FakeSegmentWorker`` / ``FakeCaptionWorker`` (``workers_app``: the real
shard runners with G2's fake engines, ``local.balanced`` for the balanced kinds) on
``img_cpu`` with no GPU and :data:`FAKE_WORKER_CPU` / :data:`FAKE_WORKER_MEMORY_GIB`. It
hashes ``fake_models`` into every stage hash (``config.hashing``), fills missing derived-weight
receipts with :data:`FAKE_RECEIPT`, names the app ``flora-<project>-fake-<digest>``, tags it
``flora-fake-models``, sets ``FLORA_FAKE_MODELS=1`` on workers and records
``certificate: fake-models`` in the deployment record. No weights are loaded;
``warm_reward_cache`` is not deployed.
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

from flora_deploy.jobs.settings import WORKER_SCALEDOWN_S, PoolSettings

APP_ENV = "FLORA_APP_NAME"
WATCH_ENV = "FLORA_WATCH_DICT"
STORE_MOUNT_ROOT = "/vol"
DEFAULT_BUCKET_SECRET = "flora-bucket"
MEMORY_LIMIT_FACTOR = 1.25  # memory=(request, limit): a deterministic OOM at the limit (D15)
WORKER_RETRIES = {"max_retries": 2, "initial_delay": 10.0, "backoff_coefficient": 2.0}
ORCH_RETRIES = {"max_retries": 3, "initial_delay": 30.0, "backoff_coefficient": 2.0}
MIN_WORKER_TIMEOUT_S = 30 * 60
MAX_TIMEOUT_S = 24 * 3600
TIMEOUT_SAFETY = 2.0  # worker timeout = safety x (rate-table wall time of a full shard + startup)
MEDIA_MOUNT_ROOT = "/media"
CACHE_MODE_ENV = "FLORA_CACHE_MODE"  # ro (workers: local copy, no commit) | rw (warm-up only)
LOCAL_CACHE_ROOT = "/tmp/flora-cache"
WARM_CACHE_FUNCTION = "warm_reward_cache"
RECONCILE_EVERY = "*/10 * * * *"
DERIVED_RECEIPTS_ENV = "FLORA_DERIVED_RECEIPTS"  # balanced workers: weights.derived they verify

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
    scaledown_window_s: int = WORKER_SCALEDOWN_S
    cache: bool = False  # mount flora-cache (compile/Triton caches)
    recipe: Literal["segcap", "reward", "balanced"] = "reward"  # container env recipe


WORKERS: dict[str, WorkerRef] = {
    # The fast exact worker (S1); its run_shard lands with P1a-5s.
    "segcap": WorkerRef("flora_deploy.modal_app.reference_app", "ReferenceWorker", "img_hfref",
                        startup_timeout_s=8 * 60, recipe="segcap"),
    "reward.terminal": WorkerRef("flora_deploy.modal_app.workers_app", "RewardWorker",
                                 "img_vllm", startup_timeout_s=15 * 60, cache=True),
    "reward.dense": WorkerRef("flora_deploy.modal_app.workers_app", "RewardWorker", "img_vllm",
                              startup_timeout_s=15 * 60, cache=True),
    # balanced (P1b-1): vLLM TempSeg / Captioner, one pool each (D8)
    "segment": WorkerRef("flora_deploy.modal_app.balanced_app", "SegmentWorker", "img_vllm",
                         startup_timeout_s=15 * 60, cache=True, recipe="balanced"),
    "caption": WorkerRef("flora_deploy.modal_app.balanced_app", "CaptionWorker", "img_vllm",
                         startup_timeout_s=15 * 60, cache=True, recipe="balanced"),
    # NVIDIA's rubric recipe: TP=2 on its own pool (H100!:2); no flora-cache (see workers_app)
    "reward.rubric": WorkerRef("flora_deploy.modal_app.workers_app", "RubricWorker", "img_vllm",
                               startup_timeout_s=20 * 60),
}
PENDING_WORKERS: dict[str, str] = {}
INDEX_FUNCTION: tuple[str, str] = ("flora_deploy.modal_app.index_app", "index_run_shard")

# --fake-models (H0): CPU workers with G2's fake engines behind the real shard runners.
FAKE_WORKERS: dict[str, WorkerRef] = {
    "segcap": WorkerRef("flora_deploy.modal_app.workers_app", "FakeSegcapWorker", "img_cpu",
                        startup_timeout_s=8 * 60, recipe="segcap"),
    "reward.terminal": WorkerRef("flora_deploy.modal_app.workers_app", "FakeRewardWorker",
                                 "img_cpu", startup_timeout_s=15 * 60, cache=True),
    "reward.dense": WorkerRef("flora_deploy.modal_app.workers_app", "FakeRewardWorker",
                              "img_cpu", startup_timeout_s=15 * 60, cache=True),
    "reward.rubric": WorkerRef("flora_deploy.modal_app.workers_app", "FakeRubricWorker",
                               "img_cpu", startup_timeout_s=15 * 60),
    "segment": WorkerRef("flora_deploy.modal_app.workers_app", "FakeSegmentWorker", "img_cpu",
                         startup_timeout_s=8 * 60, recipe="balanced"),
    "caption": WorkerRef("flora_deploy.modal_app.workers_app", "FakeCaptionWorker", "img_cpu",
                         startup_timeout_s=8 * 60, recipe="balanced"),
}
FAKE_RECEIPT = "flora-fake-models"
FAKE_CERTIFICATE = "fake-models"
FAKE_MODELS_ENV = "FLORA_FAKE_MODELS"
FAKE_WORKER_CPU = 2.0
FAKE_WORKER_MEMORY_GIB = 12.0  # fake decode keeps every frame in RAM (1080p ~ 6 MiB/frame)


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


_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


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
    fake_models: bool = False  # H0 rehearsal app (module docstring)

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


def app_name_for(project: str, stage_hashes: Mapping[str, str], *,
                 fake_models: bool = False) -> str:
    """``flora-<project slug>-<12 hex of the stage hashes>`` (<= 60 chars, Modal-safe);
    ``flora-<slug>-fake-<12 hex>`` for a ``--fake-models`` app."""
    width = 35 if fake_models else 40
    slug = re.sub(r"[^a-z0-9-]+", "-", project.lower()).strip("-")[:width].strip("-") or "job"
    return f"flora-{slug}-{'fake-' if fake_models else ''}{_digest(stage_hashes)}"


def worker_timeout_s(shard_target_gpu_minutes: float, *, preset: str = "exact",
                     kinds: tuple[str, ...] = (), max_inputs: int = 1,
                     startup_s: float = 0.0, table: Any = None) -> int:
    """Per-call timeout (M5): :data:`TIMEOUT_SAFETY` x (the wall time of a full shard of each
    kind, from ``planner/rates.json`` at the upper quantile and the pool's concurrency, plus
    startup), at least 30 min, at most Modal's 24 h.

    Manifests pack ``shard_target_gpu_minutes`` of *placeholder-rate* work per shard
    (``jobs.manifest.DEFAULT_RATES``), so a full shard holds ``target / DEFAULT_RATES[kind]``
    video seconds; the rate table says what those really cost (exact segcap: 8.62 container-s per
    video-s, not 3.0). Without kinds: 3x the target, as before."""
    target_s = shard_target_gpu_minutes * 60
    if not kinds:
        return int(min(MAX_TIMEOUT_S, max(MIN_WORKER_TIMEOUT_S, math.ceil(target_s * 3))))
    from flora_deploy.jobs.manifest import DEFAULT_RATES
    from flora_deploy.planner.rates import RateTable

    table = table or RateTable.load()
    wall = 0.0
    for kind in kinds:
        video_s = target_s / DEFAULT_RATES[kind]  # type: ignore[index]
        rates = []
        for cams in (1, 3):  # the slowest camera layout the table knows
            try:
                e = table.rate(preset, kind, cams, max_inputs)
            except KeyError:
                continue
            rates.append(e.container_s_per_video_s * (1.0 + e.uncertainty))
        per = max(rates) if rates else DEFAULT_RATES[kind]  # type: ignore[index]
        try:
            st = table.startup_for(preset, kind)
            start = st.startup_s * (1.0 + st.uncertainty)
        except KeyError:
            start = startup_s
        wall = max(wall, video_s * per * max(max_inputs, 1) + max(start, startup_s))
    return int(min(MAX_TIMEOUT_S, max(MIN_WORKER_TIMEOUT_S, math.ceil(wall * TIMEOUT_SAFETY))))


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


def _recipe_env(recipe: str, determinism: str, cache_mode: str = "ro") -> dict[str, str]:
    if recipe == "segcap":
        from flora_deploy.accel import determinism as det
        from flora_deploy.modal_app.warm import linked_offline_env

        return {**linked_offline_env(), **det.container_env(determinism)}
    if recipe == "balanced":  # compile caches: balanced_app.cache_env (read-only mount, M7)
        # No VLLM_ALLOW_INSECURE_SERIALIZATION: the GDN dtype check calls the worker extension's
        # method by name (engines/worker_ext.py), which vLLM's msgpack encoder accepts as is.
        from flora_deploy.modal_app.balanced_app import COMPILE_CACHE_ENV, PUBLISH_ENV

        # the compile-cache switches, from the deploying process (a certified run forces one
        # job cold; review 13 W1); not hashed: the cache is recorded per record instead
        switches = {k: os.environ[k] for k in (COMPILE_CACHE_ENV, PUBLISH_ENV)
                    if os.environ.get(k)}
        return {CACHE_MODE_ENV: cache_mode, "TOKENIZERS_PARALLELISM": "false", **switches}
    from flora_deploy.modal_app import rewarder_app as ra

    root = LOCAL_CACHE_ROOT if cache_mode == "ro" else None
    cache = ra.cache_env(root) if root is not None else ra.cache_env()
    return {ra.IMAGE_FP_ENV: ra.IMAGE_FP, **cache, CACHE_MODE_ENV: cache_mode}


def _has_function(module: str, attr: str, skipped: dict[str, str]) -> bool:
    try:
        resolve(module, attr, want="function")
    except DeployError as e:
        skipped[attr] = str(e)
        return False
    return True


@dataclass(frozen=True)
class MediaSpec:
    """Where containers read the dataset's videos (C1): ``root`` is ``FLORA_MEDIA_ROOT``."""

    root: str
    volumes: dict[str, VolumeMount] = field(default_factory=dict)
    how: str = ""


def media_for(cfg: Any, store: StoreSpec) -> MediaSpec:
    """The media root the workers resolve index receipts' relative paths against (module
    docstring). Raises :class:`DeployError` for inputs containers cannot reach."""
    from flora_deploy.jobs.indexing import DatasetSpec

    inp = cfg.input
    spec = DatasetSpec.from_config(inp)
    stage = (inp.stage_to or "").strip()
    if stage:
        u = urlparse(stage)
        if u.scheme not in ("volume", "modal-volume"):
            raise DeployError(
                f"input.stage_to {stage!r}: workers read staged inputs from a Modal Volume; use "
                "volume://<volume>/<dir> (bucket staging is not supported yet)")
        if not _VOLUME_NAME.match(u.netloc or ""):
            raise DeployError(f"input.stage_to {stage!r}: volume://<volume name>/<dir>")
        sub = u.path.strip("/")
        if store.kind == "volume" and store.volume == u.netloc:
            base, vols = str(store.mount), {}
        else:
            base = f"{MEDIA_MOUNT_ROOT}/{u.netloc}"
            vols = {base: VolumeMount(u.netloc, read_only=True)}
        root = f"{base}/{sub}" if sub else base
        if spec.repo is not None:  # as ``flora index`` stages a Hub dataset (DatasetMirror)
            root = f"{root}/{spec.key}" + (f"/{spec.subdir}" if spec.subdir else "")
        return MediaSpec(root, vols, f"staged on Volume {u.netloc}")
    if spec.repo is not None:
        root = f"hf://datasets/{spec.repo}@{spec.revision}" + (
            f"/{spec.subdir}" if spec.subdir else "")
        return MediaSpec(root, {}, "Hugging Face Hub at the pinned revision")
    raise DeployError(
        f"input.path {inp.path!r} is a local directory, which Modal containers cannot read: "
        "upload it to a Volume (`modal volume put <vol> <dir> <dest>`) and set "
        "input.stage_to: volume://<vol>/<dest>")


def instruction_env(cfg: Any) -> dict[str, str]:
    """C2: the configured instruction for workers (the fixed text when ``source: fixed``)."""
    ins = cfg.pipeline.episode_instruction
    env = {"FLORA_INSTRUCTION_SOURCE": ins.source}
    if ins.source == "fixed" and ins.text:
        env["FLORA_INSTRUCTION_TEXT"] = ins.text
    return env


def balanced_env(cfg: Any, kinds: list[str], pools: Mapping[str, PoolSettings],
                 derived_receipts: Mapping[str, str], *, fake_models: bool = False
                 ) -> dict[str, str]:
    """What a balanced worker verifies (``modal_app.balanced_app``): the recorded receipts of
    the derived weights its kinds need (``weights.derived``). The certification env (pool GPU,
    modes, embodiment: :func:`workers_impl.certstamp.cert_env`) is set on every worker class
    by :func:`plan_app`.

    A derived weight without a pinned content digest (``pins.DERIVED_PINS``) is refused here,
    at plan time (``flora plan`` / ``deploy`` / ``run``, ``sdk.submit``), with the steps that
    fix it: the workers refuse it at start, which would be a billed GPU start per shard (review
    10 M1). ``fake_models`` reads no weights, so it skips that check."""
    from flora_deploy.config.presets import get_preset
    from flora_deploy.config.receipts import UnpinnedDerivedWeight, require_pinned

    preset = get_preset(cfg.preset)
    names = sorted({n for k in kinds for n in preset.derived_weights.get(k, ())})  # type: ignore[call-overload]
    missing = [n for n in names if not derived_receipts.get(n)]
    if missing:
        raise DeployError(f"{', '.join(kinds)}: no recorded receipt for derived weight(s) "
                          f"{missing} (run `flora prepare`, or pass --receipt NAME=<sha256>)")
    if not fake_models:
        kinds_of = {n: [k for k in kinds if n in preset.derived_weights.get(k, ())]  # type: ignore[call-overload]
                    for n in names}
        try:
            require_pinned(names, kinds_of=kinds_of, preset=cfg.preset)
        except UnpinnedDerivedWeight as e:
            raise DeployError(str(e)) from None
    return {DERIVED_RECEIPTS_ENV: json.dumps({n: derived_receipts[n] for n in names},
                                             sort_keys=True)}


def caption_segments_env(cfg: Any, st: StoreSpec) -> dict[str, str]:
    """``workers_impl.caption.CAPTION_SEGMENTS_ENV`` of a stage-isolated caption job
    (``pipeline.caption_segments``): the staged oracle segment file's key in the job store, its
    sha256 and the oracle stage hash. The file must be inside the job store (same Volume or
    bucket, under its root). Empty for every other job."""
    from flora_deploy.workers_impl.caption import CAPTION_SEGMENTS_ENV

    cs = getattr(cfg.pipeline, "caption_segments", None)
    if cs is None:
        return {}
    try:
        rec = parse_store(cs.records, bucket_secret=st.secret or DEFAULT_BUCKET_SECRET)
    except DeployError as e:
        raise DeployError(f"pipeline.caption_segments.records: {e}") from None
    same = (rec.kind, rec.volume, rec.bucket) == (st.kind, st.volume, st.bucket)
    prefix = f"{st.root}/" if st.root else ""
    if not same or not rec.root.startswith(prefix) or rec.root == st.root:
        raise DeployError(f"pipeline.caption_segments.records {cs.records!r} is not a file in "
                          f"the job store {st.uri!r}: stage it there")
    return {CAPTION_SEGMENTS_ENV: json.dumps({"key": rec.root[len(prefix):], "sha256": cs.sha256,
                                              "oracle_stage_hash": cs.oracle_stage_hash},
                                             sort_keys=True)}


def plan_app(
    cfg: Any,
    *,
    store: str | StoreSpec | None = None,
    bucket_secret: str = DEFAULT_BUCKET_SECRET,
    derived_receipts: Mapping[str, str] | None = None,
    derived_content: Mapping[str, str] | None = None,
    allow_uncertified: bool = False,
    min_containers: int = 0,
    reconcile_cron: str | None = None,
    determinism: str | None = None,
    registry: Mapping[str, WorkerRef] | None = None,
    index_function: tuple[str, str] | None = INDEX_FUNCTION,
    missing_workers: Literal["error", "skip"] = "error",
    hash_kwargs: Mapping[str, Any] | None = None,
    fake_models: bool = False,
) -> AppPlan:
    """Everything ``build_app`` will register, as data. Contacts nothing.

    ``cfg``: a validated ``FloraConfig``. ``store``: the job store URI (default: the config's
    ``output.sink``). ``derived_receipts``: as for ``config.hashing.config_hash`` (exact segcap
    needs ``captioner_bf16``). ``derived_content``: the derived weights' content digests, for
    certificate matching (none: a kind with derived weights matches no certificate).
    ``missing_workers="skip"`` leaves out kinds whose worker class is
    not importable yet (listed in ``skipped``) instead of refusing; ``flora deploy`` never skips.
    ``hash_kwargs``: extra ``config_hash`` arguments (tests). ``determinism`` (``flora deploy
    --determinism``) overrides ``pipeline.determinism`` through ``config.schema.with_pipeline``,
    so it changes the config hash, the app name and the store paths (review 4 H3); workers get
    ``config.schema.output_settings`` (``FLORA_DETERMINISM``, ``FLORA_TERMINAL_MODE``).
    ``fake_models``: the H0 rehearsal app (module docstring)."""
    from flora_deploy.accel import determinism as det
    from flora_deploy.config.hashing import config_hash
    from flora_deploy.config.presets import find_certificates, get_preset, needs_certificate
    from flora_deploy.config.schema import (
        ConfigError,
        output_settings,
        to_run_settings,
        with_pipeline,
    )
    from flora_deploy.modal_app import orchestrator_app as oa
    from flora_deploy.modal_app.images import image_fingerprint
    from flora_deploy.workers_impl.certstamp import cert_env

    if determinism is not None and determinism not in det.MODES:
        raise DeployError(f"determinism {determinism!r}: use one of {', '.join(det.MODES)}")
    if determinism is not None and determinism != cfg.pipeline.determinism:
        try:
            cfg = with_pipeline(cfg, determinism=determinism)
        except ConfigError as e:
            raise DeployError(str(e)) from None
    outputs = output_settings(cfg)
    determinism = str(outputs["determinism"])
    if min_containers < 0:
        raise DeployError("min_containers must be >= 0")
    reg = dict((FAKE_WORKERS if fake_models else WORKERS) if registry is None else registry)
    hkw = dict(hash_kwargs or {})
    if fake_models:
        from flora_deploy.config.presets import shard_kinds_for

        p = get_preset(cfg.preset)
        needed = {r for k in shard_kinds_for(p, cfg.pipeline.stages)
                  for r in p.derived_weights.get(k, ())}
        derived_receipts = {**{r: FAKE_RECEIPT for r in sorted(needed)},
                            **dict(derived_receipts or {})}
        hkw["fake_models"] = True
    ch = config_hash(cfg, derived_receipts=derived_receipts, **hkw)
    rs = to_run_settings(cfg, "deploy", config_hash=ch.hash, stage_hashes=dict(ch.stage_hashes),
                         min_containers=min_containers)
    preset = get_preset(cfg.preset)
    st = store if isinstance(store, StoreSpec) else parse_store(store or cfg.output.sink,
                                                                bucket_secret=bucket_secret)

    if needs_certificate(cfg.preset) and not (cfg.allow_uncertified or allow_uncertified):
        from flora_deploy.validate.certkey import lookup_context  # Fix-Cert: H6/H7

        certs = find_certificates(cfg.preset, rs.pools, **lookup_context(
            cfg, rs.pools, derived_content=derived_content))
        missing = [f"{k} on {p.gpu}" for k, p in rs.pools.items() if k not in certs]
        if missing:
            raise DeployError(f"preset {cfg.preset!r} has no Gate certificate for "
                              f"{', '.join(missing)}; deploy 'exact' or allow uncertified")

    name = app_name_for(cfg.project, ch.stage_hashes, fake_models=fake_models)
    watch_dict = f"flora-watch-{_digest(ch.stage_hashes)}"
    if fake_models:  # per app (so per project): concurrent H0 runs share stage hashes, and each
        # deletes its own watch Dict on cleanup (review 9a M2)
        watch_dict = f"flora-watch-fake-{_digest({'app': name})}"
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
        if ref.image != preset.engines[kind].image and not fake_models:
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
        **({FAKE_MODELS_ENV: "1"} if fake_models else {}),
    }
    store_volumes = ({str(st.mount): VolumeMount(str(st.volume), create=True)}
                     if st.kind == "volume" else {})
    secrets = (st.secret,) if st.secret else ()
    media = media_for(cfg, st) if by_class else None

    from flora_deploy.layout import CACHE_MOUNT, CACHE_VOLUME, WEIGHTS_MOUNT, WEIGHTS_VOLUME

    objects: list[ObjectSpec] = []
    scaledown = WORKER_SCALEDOWN_S
    for cls_name, kinds in by_class.items():
        ref = refs[cls_name]
        pool = merge_pools(kinds, rs.pools)
        scaledown = max(scaledown, ref.scaledown_window_s)
        volumes = {str(WEIGHTS_MOUNT): VolumeMount(WEIGHTS_VOLUME, read_only=True)}
        if ref.cache:  # M7: read-only on workers; warm_reward_cache is the one writer
            volumes[str(CACHE_MOUNT)] = VolumeMount(CACHE_VOLUME, read_only=True)
        volumes.update(store_volumes)
        assert media is not None
        volumes.update(media.volumes)
        env = {**_recipe_env(ref.recipe, determinism), **base_env, **instruction_env(cfg),
               "FLORA_MEDIA_ROOT": media.root,
               "FLORA_DETERMINISM": determinism,
               "FLORA_TERMINAL_MODE": str(outputs["terminal_mode"]),
               "FLORA_OUTPUT_SETTINGS": json.dumps(outputs, sort_keys=True),
               "FLORA_SHARD_KINDS": ",".join(kinds),
               "FLORA_STAGE_HASHES": json.dumps({k: ch.stage_hashes[k] for k in kinds},
                                                sort_keys=True),
               # provenance every record carries (embodiment, cert_key, gpu), every kind
               **cert_env(cfg, {k: rs.pools[k] for k in kinds})}
        if ref.recipe == "segcap":  # review 6 H2: one input per container, c episodes inside
            from flora_deploy.jobs.settings import EPISODE_CONCURRENCY_ENV, episode_concurrency

            if pool.max_inputs > 1:
                raise DeployError(f"{cls_name}: max_inputs {pool.max_inputs} > 1, but the segcap "
                                  "worker runs one shard per container; use episode_concurrency")
            env[EPISODE_CONCURRENCY_ENV] = str(episode_concurrency(kinds[0], pool))
            from flora_deploy.config.schema import exact_scheduling
            from flora_deploy.workers_impl.segcap import SCHEDULING_ENV

            env[SCHEDULING_ENV] = json.dumps(exact_scheduling(cfg), sort_keys=True)  # G3 order
        if ref.recipe == "balanced":
            env.update(balanced_env(cfg, kinds, rs.pools, derived_receipts or {},
                                    fake_models=fake_models))
            if "caption" in kinds:
                env.update(caption_segments_env(cfg, st))
            if pool.max_inputs > 1:
                raise DeployError(f"{cls_name}: max_inputs {pool.max_inputs} > 1, but a balanced "
                                  "worker runs one shard per container (its episodes run "
                                  "concurrently inside it)")
        timeout = worker_timeout_s(rs.shard_target_gpu_minutes, preset=cfg.preset,
                                   kinds=tuple(kinds), max_inputs=pool.max_inputs,
                                   startup_s=float(ref.startup_timeout_s) / 2)
        gpu = gpu_string(pool)
        if fake_models:  # CPU containers sized for the fake engines, not the models
            gpu = None
            pool = replace(pool, cpu_cores=FAKE_WORKER_CPU, memory_gib=FAKE_WORKER_MEMORY_GIB)
        objects.append(ObjectSpec(
            kind="cls", name=cls_name, module=ref.module, attr=ref.name, image=ref.image,
            gpu=gpu, cpu=float(pool.cpu_cores), memory=memory_mib(pool.memory_gib),
            timeout=timeout, startup_timeout=ref.startup_timeout_s,
            min_containers=min_containers, max_containers=pool.max_containers,
            scaledown_window=ref.scaledown_window_s,
            max_inputs=pool.max_inputs if pool.max_inputs > 1 else None,
            retries=dict(WORKER_RETRIES), volumes=volumes, secrets=secrets,
            env=dict(sorted(env.items())), region=cfg.modal.region, shard_kinds=tuple(kinds)))
        if fake_models and ref.cache:
            skipped[WARM_CACHE_FUNCTION] = "fake models: no engine cache to warm"
        elif ref.cache and ref.recipe == "reward" and _has_function(ref.module,
                                                                   WARM_CACHE_FUNCTION, skipped):
            warm_volumes = {**volumes, str(CACHE_MOUNT): VolumeMount(CACHE_VOLUME, create=True)}
            objects.append(ObjectSpec(
                kind="function", name=WARM_CACHE_FUNCTION, module=ref.module,
                attr=WARM_CACHE_FUNCTION, image=ref.image, gpu=gpu_string(pool),
                cpu=float(pool.cpu_cores), memory=memory_mib(pool.memory_gib),
                timeout=2 * ref.startup_timeout_s, max_containers=1, volumes=warm_volumes,
                secrets=secrets, region=cfg.modal.region,
                env=dict(sorted({**env, **_recipe_env(ref.recipe, determinism, "rw")}.items()))))

    from flora_deploy.jobs import pools as jp

    orch_env = dict(sorted({**base_env, WATCH_ENV: watch_dict,
                            jp.POOL_DICT_ENV: jp.pool_dict_name(watch_dict),
                            "FLORA_SCALEDOWN_S": str(scaledown),
                            "FLORA_MIN_CONTAINERS": str(min_containers),
                            **({"FLORA_REGION": cfg.modal.region} if cfg.modal.region else {}),
                            }.items()))
    here = __name__
    common = {"module": here, "image": "img_cpu", "cpu": float(jp.ORCHESTRATOR_CPU),
              "memory": (int(jp.ORCHESTRATOR_MEMORY_MIB), int(jp.ORCHESTRATOR_MEMORY_LIMIT_MIB)),
              "scaledown_window": int(jp.ORCHESTRATOR_SCALEDOWN_S),
              "volumes": dict(store_volumes), "secrets": secrets,
              "env": orch_env, "region": cfg.modal.region}
    objects.append(ObjectSpec(kind="function", name="orchestrator_run", attr="orchestrator_run",
                              timeout=int(oa.TIMEOUT_S),
                              nonpreemptible=not jp.ORCHESTRATOR_PREEMPTIBLE,
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
    tags = {**cfg.modal.tags, "flora-project": cfg.project, "flora-preset": cfg.preset,
            **({"flora-fake-models": "true"} if fake_models else {})}
    return AppPlan(
        app_name=name, project=cfg.project, environment=cfg.modal.environment,
        preset=cfg.preset, config_hash=ch.hash, stage_hashes=dict(ch.stage_hashes),
        image_fingerprints={i: image_fingerprint(i) for i in images_used},
        worker_classes={k: n for n, ks in by_class.items() for k in ks},
        store=st, objects=tuple(objects), tags=dict(sorted(tags.items())), skipped=skipped,
        watch_dict=watch_dict, fake_models=fake_models)


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
        **({"fake_models": True, "certificate": FAKE_CERTIFICATE} if plan.fake_models else {}),
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
    import modal

    from flora_deploy.jobs.pools import POOL_DICT_ENV
    from flora_deploy.modal_app import orchestrator_app as oa

    pool_kv = (modal.Dict.from_name(os.environ[POOL_DICT_ENV], create_if_missing=True)
               if os.environ.get(POOL_DICT_ENV) else None)
    return oa.run_generation(job_id, factory, generation, watch,
                             spawn_successor=_own("orchestrator_run").spawn,
                             watch_kv=_watch_kv(), pool_kv=pool_kv)


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
