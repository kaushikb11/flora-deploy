"""``flora.yaml``: the user's job config (pydantic v2), and its mapping onto ``RunSettings``.

The job system only ever sees ``RunSettings`` (``jobs/settings.py``); this module is the one
place that knows the YAML layout. Every model forbids unknown keys so typos fail loudly.

Layout (see ``examples/*.yaml`` for commented templates)::

    project: my-labels
    license_ack: research-only          # required (NVIDIA's models are non-commercial)
    modal: {environment, max_gpus, budget_usd, region, tags}
    preset: exact | balanced             # budget / interactive: not available yet
    input: {format, repo, revision, path, cameras, stage_to}
    pipeline: {stages, embodiment, episode_instruction, envelope, shard_target_gpu_minutes}
    resources: {<shard kind>: {gpu, gpu_count, max_containers, max_inputs, cpu, memory}}
    output: {sink, formats}
    retries: {max_starts_before_quarantine}
    observability: {log_level, heartbeat_s, progress}
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from flora_deploy.config.presets import PresetUnavailableError, get_preset, shard_kinds_for
from flora_deploy.jobs.settings import EnvelopePolicy, PoolSettings, Preset, RunSettings
from flora_deploy.jobs.types import MAX_STARTS_BEFORE_QUARANTINE, ShardKind
from flora_deploy.media.types import CameraRole
from flora_deploy.records import Stage

InputFormat = Literal["lerobot_v21", "lerobot_v3", "mp4_dir"]
# Embodiment pools, as in ``jobs.types.EpisodeRef.pool``.
Embodiment = Literal["human", "yam", "yam32", "dex", "tianji"]
OutputFormat = Literal["parquet", "jsonl", "lerobot"]

_SHA = re.compile(r"^[0-9a-f]{40}$")
_GPU = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*!?$")  # Modal GPU string without ":count"
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class ConfigError(ValueError):
    """``flora.yaml`` is invalid. The message lists every problem with its YAML path."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModalSection(_Model):
    environment: str = "main"
    max_gpus: int = Field(ge=1, description="workspace GPU cap the job must stay under")
    budget_usd: float = Field(gt=0, description="hard spend cap for the job")
    region: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    environment_budget_factor: float = Field(1.2, ge=1.0)

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

    @model_validator(mode="after")
    def _where(self) -> InputSection:
        if self.repo is None and self.path is None:
            raise ValueError("set `repo` (a Hugging Face dataset) or `path` (a local directory)")
        if self.repo is not None:
            if not self.revision or not _SHA.match(self.revision):
                raise ValueError(
                    f"`revision` must be the full 40-hex commit of {self.repo!r} (got "
                    f"{self.revision!r}); branch names would make results non-reproducible")
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


class EpisodeInstruction(_Model):
    source: Literal["dataset_task", "fixed", "none"] = "dataset_task"
    text: str | None = None

    @model_validator(mode="after")
    def _text(self) -> EpisodeInstruction:
        if self.source == "fixed" and not (self.text and self.text.strip()):
            raise ValueError("episode_instruction.source 'fixed' needs `text`")
        if self.source != "fixed" and self.text is not None:
            raise ValueError("episode_instruction.text only applies with source 'fixed'")
        return self


class PipelineSection(_Model):
    stages: tuple[Stage, ...] = Field(min_length=1)
    embodiment: Embodiment
    episode_instruction: EpisodeInstruction = Field(default_factory=EpisodeInstruction)
    envelope: EnvelopePolicy = "warn"  # D13
    shard_target_gpu_minutes: float = Field(10.0, gt=0)

    @field_validator("stages")
    @classmethod
    def _stages(cls, v: tuple[Stage, ...]) -> tuple[Stage, ...]:
        if len(set(v)) != len(v):
            raise ValueError("stages are listed more than once")
        if "caption" in v and "segment" not in v:
            raise ValueError("`caption` needs `segment` (captions are per segment)")
        order = get_args(Stage)
        return tuple(sorted(v, key=order.index))


class ResourceOverride(_Model):
    gpu: str | None = None
    gpu_count: int | None = Field(None, ge=1, le=8)
    max_containers: int | None = Field(None, ge=1)
    max_inputs: int | None = Field(None, ge=1)
    cpu: float | None = Field(None, gt=0, description="physical cores")
    memory: float | None = Field(None, gt=0, description="GiB")

    @field_validator("gpu")
    @classmethod
    def _gpu(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if ":" in v:
            raise ValueError(f"gpu {v!r}: give the count as `gpu_count`, not 'TYPE:N'")
        if not _GPU.match(v):
            raise ValueError(f"gpu {v!r} is not a Modal GPU type (e.g. H200, H100!, L40S)")
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


class ObservabilitySection(_Model):
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    heartbeat_s: float = Field(60.0, gt=0)
    progress: bool = True


class FloraConfig(_Model):
    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
    license_ack: Literal["research-only"]
    modal: ModalSection
    preset: Preset = "exact"
    input: InputSection
    pipeline: PipelineSection
    resources: dict[ShardKind, ResourceOverride] = Field(default_factory=dict)
    output: OutputSection
    retries: RetriesSection = Field(default_factory=RetriesSection)
    observability: ObservabilitySection = Field(default_factory=ObservabilitySection)

    @model_validator(mode="after")
    def _resolve(self) -> FloraConfig:
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
        need = sum(p.gpu_count * p.max_containers for p in pools.values())
        if need > self.modal.max_gpus:
            raise ValueError(
                f"resources can use {need} GPUs at once but modal.max_gpus is "
                f"{self.modal.max_gpus}; lower max_containers or raise max_gpus")
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


def parse_config(data: Any, *, source: str = "flora.yaml") -> FloraConfig:
    """Validate a parsed YAML/JSON document. Raises ``ConfigError`` listing every problem."""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    try:
        return FloraConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_errors(e, source)) from None


def loads_config(text: str, *, source: str = "flora.yaml") -> FloraConfig:
    import yaml

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{source}: not valid YAML: {e}") from None
    return parse_config(data, source=source)


def load_config(path: str | Path) -> FloraConfig:
    p = Path(path)
    return loads_config(p.read_text(encoding="utf-8"), source=str(p))


def to_run_settings(
    cfg: FloraConfig,
    job_id: str,
    *,
    config_hash: str | None = None,
    derived_receipts: dict[str, str] | None = None,
) -> RunSettings:
    """Map a validated config onto ``RunSettings``.

    ``config_hash`` defaults to ``hashing.config_hash(cfg, derived_receipts=...)`` (D11)."""
    if config_hash is None:
        from flora_deploy.config.hashing import config_hash as compute

        config_hash = compute(cfg, derived_receipts=derived_receipts).hash
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
    )
