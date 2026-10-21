"""``flora.yaml``: the user's job config (pydantic v2), and its mapping onto ``RunSettings``.

The job system only ever sees ``RunSettings`` (``jobs/settings.py``); this module is the one
place that knows the YAML layout. Every model forbids unknown keys so typos fail loudly.

Layout (see ``config/templates/*.yaml`` for commented templates)::

    project: my-labels
    license_ack: research-only          # required (NVIDIA's models are non-commercial)
    modal: {environment, max_gpus, budget_usd, region, tags}
    preset: exact | balanced             # budget / interactive: not available yet
    allow_uncertified: false             # required for balanced until its certificates exist
    input: {format, repo, revision, path, cameras, stage_to}
    pipeline: {stages, embodiment, episode_instruction, envelope, shard_target_gpu_minutes,
               determinism, terminal_rationale}
    resources: {<shard kind>: {gpu, gpu_count, max_containers, max_inputs, cpu, memory}}
    output: {sink, formats}
    retries: {max_starts_before_quarantine, max_episode_error_rate}
    observability: {log_level, heartbeat_s, progress}
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from flora_deploy.config.presets import (
    GPU_MEMORY_GIB,
    PresetUnavailableError,
    find_certificates,
    get_preset,
    gpu_problem,
    needs_certificate,
    shard_kinds_for,
)
from flora_deploy.jobs.settings import (
    MAX_EPISODE_ERROR_RATE,
    WORKER_SCALEDOWN_S,
    EnvelopePolicy,
    PoolSettings,
    Preset,
    RunSettings,
)
from flora_deploy.jobs.types import MAX_STARTS_BEFORE_QUARANTINE, ShardKind
from flora_deploy.media.types import CameraRole
from flora_deploy.nv.capture import FORBIDDEN_PROMPT_TOKENS, VIDEO_TOKEN
from flora_deploy.records import Stage

InputFormat = Literal["lerobot_v21", "lerobot_v3", "mp4_dir"]
# Embodiment pools, as in ``jobs.types.EpisodeRef.pool``.
Embodiment = Literal["human", "yam", "yam32", "dex", "tianji"]
OutputFormat = Literal["parquet", "jsonl", "lerobot"]
Determinism = Literal["nvidia", "deterministic"]  # accel.determinism.MODES

_SHA = re.compile(r"^[0-9a-f]{40}$")
# Tokens an instruction must not contain: NVIDIA's processor and our vLLM wrap treat them as
# structure, not text (nv.capture.vllm_prompt refuses them per episode; refuse them up front).
INSTRUCTION_FORBIDDEN_TOKENS = (*FORBIDDEN_PROMPT_TOKENS, VIDEO_TOKEN)
# Sane upper bounds: a typo (or ``.inf``) must not disable a hard cap.
MAX_BUDGET_USD = 1_000_000.0
MAX_GPUS = 4096
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class ConfigError(ValueError):
    """``flora.yaml`` is invalid. The message lists every problem with its YAML path."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModalSection(_Model):
    environment: str = "main"
    max_gpus: int = Field(ge=1, le=MAX_GPUS, strict=True,
                          description="workspace GPU cap the job must stay under")
    budget_usd: float = Field(gt=0, le=MAX_BUDGET_USD, allow_inf_nan=False,
                              description="hard spend cap for the job")
    region: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    environment_budget_factor: float = Field(1.2, ge=1.0, le=10.0, allow_inf_nan=False)

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: dict[str, str]) -> dict[str, str]:
        bad = [k for k in v if not _TAG.match(k)]
        if bad:
            raise ValueError(f"tag keys {bad} must match [A-Za-z0-9._-], max 63 chars")
        return v


class InputSection(_Model):
    format: InputFormat
    repo: str | None = Field(None, description="Hugging Face dataset repo id")
    revision: str | None = Field(None, description="full 40-hex commit of `repo`")
    path: str | None = Field(None, description="subdirectory of `repo`, or a local directory")
    cameras: dict[str, CameraRole] | None = Field(
        None, description="dataset camera key -> main | left_wrist | right_wrist")
    stage_to: str | None = Field(None, description="where to stage downloaded inputs")

    @field_validator("path")
    @classmethod
    def _path(cls, v: str | None) -> str | None:
        """Normalize, so ``sub``, ``./sub`` and ``sub/`` are one input (and one hash)."""
        if v is None:
            return v
        if not v.strip():
            raise ValueError("`path` is empty; leave it out instead")
        norm = posixpath.normpath(v.strip())
        return "/" + norm.lstrip("/") if norm.startswith("/") else norm

    @model_validator(mode="after")
    def _where(self) -> InputSection:
        if self.repo is None and self.path is None:
            raise ValueError("set `repo` (a Hugging Face dataset) or `path` (a local directory)")
        if self.repo is not None:
            if not self.revision or not _SHA.match(self.revision):
                raise ValueError(
                    f"`revision` must be the full 40-hex commit of {self.repo!r} (got "
                    f"{self.revision!r}); branch names would make results non-reproducible")
            if self.path is not None and (self.path.startswith("/") or self.path == ".."
                                          or self.path.startswith("../")):
                raise ValueError(f"`path` {self.path!r} must be a subdirectory inside the repo")
        elif self.revision is not None:
            raise ValueError("`revision` only applies with `repo`")
        if self.format == "mp4_dir" and self.repo is None and not self.path:
            raise ValueError("mp4_dir needs `path`")
        if self.cameras is not None:
            roles = list(self.cameras.values())
            dupes = sorted({r for r in roles if roles.count(r) > 1})
            if dupes:
                raise ValueError(f"camera roles mapped more than once: {dupes}")
            if "main" not in roles:
                raise ValueError("the camera map needs a 'main' camera")
            if len(roles) not in (1, 3):
                raise ValueError("NVIDIA's pipeline takes 1 view (main) or 3 views "
                                 "(main, left_wrist, right_wrist)")
        return self

    @property
    def dataset(self) -> str:
        return self.repo if self.repo is not None else str(self.path)

    @property
    def subdir(self) -> str | None:
        """Subdirectory inside ``repo`` (None: the repo root, or a local input)."""
        if self.repo is None or self.path in (None, "."):
            return None
        return self.path


class EpisodeInstruction(_Model):
    source: Literal["dataset_task", "fixed", "none"] = "dataset_task"
    text: str | None = None

    @model_validator(mode="after")
    def _text(self) -> EpisodeInstruction:
        if self.source == "fixed" and not (self.text and self.text.strip()):
            raise ValueError("episode_instruction.source 'fixed' needs `text`")
        if self.source != "fixed" and self.text is not None:
            raise ValueError("episode_instruction.text only applies with source 'fixed'")
        if self.text is not None:
            bad = [t for t in INSTRUCTION_FORBIDDEN_TOKENS if t in self.text]
            if bad:
                raise ValueError(f"episode_instruction.text contains special token(s) {bad}, "
                                 "which NVIDIA's processor treats as video structure")
        return self


class PipelineSection(_Model):
    stages: tuple[Stage, ...] = Field(min_length=1)
    embodiment: Embodiment
    episode_instruction: EpisodeInstruction = Field(default_factory=EpisodeInstruction)
    envelope: EnvelopePolicy = "warn"  # D13
    shard_target_gpu_minutes: float = Field(10.0, gt=0, le=24 * 60, allow_inf_nan=False)
    # D-1 (docs/determinism.md): GPU kernel mode of the exact `segcap` worker. "nvidia" is
    # NVIDIA's default behaviour; "deterministic" changes logits (and sometimes text): a
    # different reproducible reference, so it is hashed into the segcap stage hash (review 4 H3).
    determinism: Determinism = "nvidia"
    # NVIDIA's terminal_reward.py `--with-rationale` (default off: score only). Different
    # prompt, max_tokens and regex, so it is hashed into the reward.terminal stage hash (M4).
    terminal_rationale: bool = Field(False, strict=True)

    @field_validator("stages")
    @classmethod
    def _stages(cls, v: tuple[Stage, ...]) -> tuple[Stage, ...]:
        if len(set(v)) != len(v):
            raise ValueError("stages are listed more than once")
        if "caption" in v and "segment" not in v:
            raise ValueError("`caption` needs `segment` (captions are per segment)")
        order = get_args(Stage)
        return tuple(sorted(v, key=order.index))

    @model_validator(mode="after")
    def _instruction_for_rewards(self) -> PipelineSection:
        """NVIDIA's reward prompts are built around the task instruction (C2): ``none`` would
        give every reward episode an empty task, so it is refused up front."""
        rewards = [s for s in self.stages if s.startswith("reward.")]
        if rewards and self.episode_instruction.source == "none":
            raise ValueError(f"stages {rewards} need a task instruction, but "
                             "episode_instruction.source is 'none'; use dataset_task, or fixed "
                             "with `text`")
        return self


class ResourceOverride(_Model):
    gpu: str | None = None
    gpu_count: int | None = Field(None, ge=1, le=8)
    max_containers: int | None = Field(None, ge=1)
    max_inputs: int | None = Field(None, ge=1)
    cpu: float | None = Field(None, gt=0, le=512, allow_inf_nan=False,
                              description="physical cores")
    memory: float | None = Field(None, gt=0, le=4096, allow_inf_nan=False, description="GiB")

    @field_validator("gpu")
    @classmethod
    def _gpu(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if ":" in v:
            raise ValueError(f"gpu {v!r}: give the count as `gpu_count`, not 'TYPE:N'")
        if v not in GPU_MEMORY_GIB:
            raise ValueError(f"gpu {v!r} is not a Modal GPU type; use one of "
                             f"{', '.join(GPU_MEMORY_GIB)}")
        return v

    def apply(self, pool: PoolSettings) -> PoolSettings:
        fields = {
            "gpu": self.gpu, "gpu_count": self.gpu_count, "max_containers": self.max_containers,
            "max_inputs": self.max_inputs, "cpu_cores": self.cpu, "memory_gib": self.memory,
        }
        return replace(pool, **{k: v for k, v in fields.items() if v is not None})


class OutputSection(_Model):
    sink: str = Field(description="bucket URI (s3://, r2://, gs://) or volume://<name>/<path>")
    formats: tuple[OutputFormat, ...] = ("parquet",)

    @field_validator("sink")
    @classmethod
    def _sink(cls, v: str) -> str:
        if not re.match(r"^(s3|r2|gs|volume|file)://.+", v):
            raise ValueError(f"sink {v!r}: use s3://, r2://, gs://, volume:// or file://")
        return v

    @field_validator("formats")
    @classmethod
    def _formats(cls, v: tuple[OutputFormat, ...]) -> tuple[OutputFormat, ...]:
        if not v or len(set(v)) != len(v):
            raise ValueError("formats must be a non-empty list without repeats")
        return v


class RetriesSection(_Model):
    max_starts_before_quarantine: int = Field(MAX_STARTS_BEFORE_QUARANTINE, ge=1, le=10)  # D15
    # Job-level error-rate guard: a job whose excluded episodes (NVIDIA-semantics failures, poison
    # pills) exceed this share of a stage ends ``completed_with_errors``, not ``done``.
    max_episode_error_rate: float = Field(MAX_EPISODE_ERROR_RATE, ge=0.0, le=1.0,
                                          allow_inf_nan=False)


class ObservabilitySection(_Model):
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    heartbeat_s: float = Field(60.0, gt=0, le=3600, allow_inf_nan=False)
    progress: bool = True


class FloraConfig(_Model):
    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
    license_ack: Literal["research-only"]
    modal: ModalSection
    preset: Preset = "exact"
    # Run a preset whose Gate certificates do not exist (yet) for its (shard kind, GPU) pools.
    # Also settable per invocation (``parse_config(..., allow_uncertified=True)``, the CLI's
    # ``--allow-uncertified``). Not hashed: it changes what may run, not what a run produces.
    allow_uncertified: bool = Field(False, strict=True)
    input: InputSection
    pipeline: PipelineSection
    resources: dict[ShardKind, ResourceOverride] = Field(default_factory=dict)
    output: OutputSection
    retries: RetriesSection = Field(default_factory=RetriesSection)
    observability: ObservabilitySection = Field(default_factory=ObservabilitySection)

    @model_validator(mode="after")
    def _resolve(self, info: ValidationInfo) -> FloraConfig:
        try:
            preset = get_preset(self.preset)
        except PresetUnavailableError as e:
            raise ValueError(str(e)) from None
        kinds = shard_kinds_for(preset, self.pipeline.stages)
        extra = sorted(set(self.resources) - set(kinds))
        if extra:
            hint = (" (the exact preset runs segment and caption together as `segcap`)"
                    if self.preset == "exact" and {"segment", "caption"} & set(extra) else "")
            raise ValueError(f"resources {extra} are not used by preset {self.preset!r} with "
                             f"stages {list(self.pipeline.stages)}; used: {kinds}{hint}")
        if self.preset == "exact" and ("segment" in self.pipeline.stages) != (
                "caption" in self.pipeline.stages):
            raise ValueError("the exact preset runs NVIDIA's pipeline, which does segment and "
                             "caption together: list both or neither")
        pools = self.pools()
        bad_gpu = [p for k, pool in pools.items()
                   if (p := gpu_problem(self.preset, k, pool.gpu, pool.gpu_count))]
        if bad_gpu:
            raise ValueError("; ".join(bad_gpu))
        allow = self.allow_uncertified or bool((info.context or {}).get("allow_uncertified"))
        if needs_certificate(self.preset) and not allow:
            from flora_deploy.config.hashing import kind_modes

            certs = find_certificates(self.preset, pools,
                                      modes={k: kind_modes(self, k) for k in pools})
            missing = [f"{k} on {p.gpu}" for k, p in pools.items() if k not in certs]
            if missing:
                raise ValueError(
                    f"preset {self.preset!r} has no Gate certificate for {', '.join(missing)} "
                    "(a provisional certificate does not count; config/certificates/), so its "
                    "outputs are not certified faithful to NVIDIA's. "
                    "Use preset 'exact', or set `allow_uncertified: true` (or pass "
                    "--allow-uncertified) to run it as an uncertified stack")
        need = sum(p.gpu_count * p.max_containers for p in pools.values())
        if need > self.modal.max_gpus:
            raise ValueError(
                f"resources can use {need} GPUs at once but modal.max_gpus is "
                f"{self.modal.max_gpus}; lower max_containers or raise max_gpus")
        return self

    @model_validator(mode="after")
    def _review4(self) -> FloraConfig:
        """Review 4 checks: determinism scope, per-device GPU memory (M7), embodiment vs camera
        count, sink and repo id syntax (review 3 A10)."""
        problems = review4_problems(self)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def pools(self) -> dict[ShardKind, PoolSettings]:
        """Preset defaults for the shard kinds this pipeline needs, with overrides applied."""
        preset = get_preset(self.preset)
        out: dict[ShardKind, PoolSettings] = {}
        for kind in shard_kinds_for(preset, self.pipeline.stages):
            pool = preset.pools[kind]
            if kind in self.resources:
                pool = self.resources[kind].apply(pool)
            out[kind] = pool
        return out


# ---------------------------------------------------------------------------------------------
# Review 4 additions (additive; kept apart from the validators above)

# Shard kinds whose engine uses more than one GPU, and how many (tensor parallel). Every other
# kind loads its model on one device: NVIDIA's reward `load_llm` keeps `tensor_parallel_size` 1
# and our engine spec passes it unchanged; NVIDIA's HF loader would shard `segcap` over several
# GPUs with `device_map="balanced"`, which is not the validated single-H200 reference.
TP_GPUS: dict[str, int] = {"reward.rubric": 2}  # NVIDIA's rubric settings: TP=2 (PLAN.md §7)
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
# A bucket/volume name right after the scheme; file:// takes an absolute path (file:///x).
_SINK = re.compile(r"^((s3|r2|gs|volume)://[^\s/]|file:///?[^\s/])[^\s]*$")


def per_device_gpu_problem(preset: str, kind: str, gpu: str, gpu_count: int) -> str | None:
    """M7: memory is checked **per device** (an engine that loads on one GPU cannot use a second
    one's memory), and ``gpu_count`` must be what the kind's engine actually uses."""
    from flora_deploy.config.presets import min_gpu_memory_gib

    want = TP_GPUS.get(kind, 1)
    if gpu_count != want:
        engine = "tensor-parallel" if want > 1 else "single-device"
        return (f"{kind} runs on {want} GPU(s) per container (its engine is {engine}); "
                f"gpu_count {gpu_count} is not supported")
    mem = GPU_MEMORY_GIB.get(gpu)
    if mem is None:
        return None  # reported by gpu_problem
    # The minimums are GPU classes (exact segcap: an H200/B200; rewards: an 80 GB GPU, which
    # NVIDIA's recipes are sized for, also per device of the TP=2 rubric engine).
    need = min_gpu_memory_gib(preset, kind)
    if mem < need:
        fits = [g for g, m in GPU_MEMORY_GIB.items() if m >= need]
        return (f"{kind} needs >= {need:g} GiB of memory on each GPU, but {gpu} has {mem:g} "
                f"GiB (use one of {', '.join(fits)})")
    return None


def review4_problems(cfg: FloraConfig) -> list[str]:
    problems: list[str] = []
    kinds = shard_kinds_for(get_preset(cfg.preset), cfg.pipeline.stages)
    if cfg.pipeline.determinism != "nvidia" and "segcap" not in kinds:
        problems.append(f"pipeline.determinism {cfg.pipeline.determinism!r} applies to the "
                        "exact preset's segcap worker (segment + caption) only")
    for kind, pool in cfg.pools().items():
        p = per_device_gpu_problem(cfg.preset, kind, pool.gpu, pool.gpu_count)
        if p:
            problems.append(p)
    cams = cfg.input.cameras
    if cams is not None:
        if cfg.pipeline.embodiment == "human" and len(cams) == 3:
            # NVIDIA's pipeline.annotate: "Choose YAM, Dex or Tianji when uploading wrist cameras."
            problems.append("embodiment 'human' takes 1 camera view; NVIDIA's pipeline rejects "
                            "wrist cameras for human videos (use yam, dex or tianji)")
        if cfg.pipeline.embodiment == "yam32" and len(cams) != 3:
            # NVIDIA: "The 32-second profile requires three synchronized camera views."
            problems.append("embodiment 'yam32' needs 3 camera views (main, left_wrist, "
                            "right_wrist)")
    if not _SINK.match(cfg.output.sink):
        problems.append(f"output.sink {cfg.output.sink!r}: needs a bucket or path after the "
                        "scheme, and no whitespace")
    if cfg.input.repo is not None and not _REPO_ID.match(cfg.input.repo):
        problems.append(f"input.repo {cfg.input.repo!r} is not a Hugging Face repo id "
                        "('namespace/name')")
    return problems


def output_settings(cfg: FloraConfig) -> dict[str, Any]:
    """The hashed, output-affecting run modes a worker must apply, taken from the config (never
    from a side channel): ``determinism`` (exact segcap) and ``terminal_mode``
    (reward.terminal). Deploy code forwards exactly these; workers record what they applied."""
    return {"determinism": cfg.pipeline.determinism,
            "terminal_mode": ("terminal_rationale" if cfg.pipeline.terminal_rationale
                              else "terminal")}


def with_pipeline(cfg: FloraConfig, **fields: Any) -> FloraConfig:
    """``cfg`` with ``pipeline`` fields replaced, re-validated. ``flora deploy --determinism
    deterministic`` is ``with_pipeline(cfg, determinism="deterministic")``: the override changes
    the config, and with it the hash, the app name and the store paths."""
    data = cfg.model_dump(mode="python")
    data["pipeline"] = {**data["pipeline"], **fields}
    return parse_config(data, allow_uncertified=cfg.allow_uncertified)


def _format_errors(err: ValidationError, source: str) -> str:
    lines = [f"{source}: {err.error_count()} problem(s)"]
    for e in err.errors():
        where = ".".join(str(p) for p in e["loc"]) or "(top level)"
        msg = e["msg"].removeprefix("Value error, ")
        if e["type"] == "extra_forbidden":
            msg = "unknown key (typo?)"
        elif e["type"] == "missing":
            msg = "required"
        lines.append(f"  - {where}: {msg}")
    return "\n".join(lines)


def parse_config(data: Any, *, source: str = "flora.yaml",
                 allow_uncertified: bool = False) -> FloraConfig:
    """Validate a parsed YAML/JSON document. Raises ``ConfigError`` listing every problem.

    ``allow_uncertified`` (the CLI's ``--allow-uncertified``) has the effect of the config's
    ``allow_uncertified: true``."""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    try:
        return FloraConfig.model_validate(data, context={"allow_uncertified": allow_uncertified})
    except ValidationError as e:
        raise ConfigError(_format_errors(e, source)) from None


def _yaml_loader():
    """``yaml.SafeLoader`` that rejects duplicate mapping keys (PyYAML keeps the last one)."""
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        seen: dict[Any, Any] = {}
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                first = seen.get(key)
            except TypeError:  # unhashable key: PyYAML raises its own error below
                continue
            if first is not None:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r} (first at line {first.line + 1})",
                    key_node.start_mark)
            seen[key] = key_node.start_mark
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                    construct_mapping)
    return UniqueKeyLoader


def loads_yaml(text: str, *, source: str = "flora.yaml") -> Any:
    """Parse YAML safely and strictly (a duplicate key is an error). Raises ``ConfigError``."""
    import yaml

    try:
        return yaml.load(text, Loader=_yaml_loader())  # a SafeLoader subclass
    except yaml.YAMLError as e:
        raise ConfigError(f"{source}: not valid YAML: {e}") from None


def loads_config(text: str, *, source: str = "flora.yaml",
                 allow_uncertified: bool = False) -> FloraConfig:
    return parse_config(loads_yaml(text, source=source), source=source,
                        allow_uncertified=allow_uncertified)


def load_config(path: str | Path, *, allow_uncertified: bool = False) -> FloraConfig:
    p = Path(path)
    return loads_config(p.read_text(encoding="utf-8"), source=str(p),
                        allow_uncertified=allow_uncertified)


def to_run_settings(
    cfg: FloraConfig,
    job_id: str,
    *,
    config_hash: str | None = None,
    derived_receipts: dict[str, str] | None = None,
    stage_hashes: dict[str, str] | None = None,
    min_containers: int = 0,
    scaledown_s: float = WORKER_SCALEDOWN_S,
) -> RunSettings:
    """Map a validated config onto ``RunSettings``.

    ``config_hash`` defaults to ``hashing.config_hash(cfg, derived_receipts=...)`` (D11), and then
    ``stage_hashes`` to its per-kind hashes (shard files are keyed by them). Passing a
    ``config_hash`` without ``stage_hashes`` keys everything by the job hash.
    ``min_containers`` / ``scaledown_s``: the deployment's warm pool and ``scaledown_window``
    (the budget gate charges both)."""
    if config_hash is None:
        from flora_deploy.config.hashing import config_hash as compute

        ch = compute(cfg, derived_receipts=derived_receipts)
        config_hash = ch.hash
        if stage_hashes is None:
            stage_hashes = dict(ch.stage_hashes)
    return RunSettings(
        job_id=job_id,
        preset=cfg.preset,
        config_hash=config_hash,
        budget_usd=cfg.modal.budget_usd,
        max_gpus=cfg.modal.max_gpus,
        pools=cfg.pools(),
        shard_target_gpu_minutes=cfg.pipeline.shard_target_gpu_minutes,
        max_starts_before_quarantine=cfg.retries.max_starts_before_quarantine,
        envelope=cfg.pipeline.envelope,
        environment_budget_factor=cfg.modal.environment_budget_factor,
        tags=dict(cfg.modal.tags),
        stage_hashes=dict(stage_hashes or {}),
        region=cfg.modal.region,
        scaledown_s=scaledown_s,
        min_containers=min_containers,
        max_episode_error_rate=cfg.retries.max_episode_error_rate,
        **output_settings(cfg),
    )
