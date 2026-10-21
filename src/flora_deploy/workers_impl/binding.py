"""Certification binding stamped into every record a worker writes (``docs/certification.md``,
"Provenance a record must carry"): ``provenance.embodiment``, ``provenance.cert_key`` and
``provenance.gpu``. ``flora certify`` refuses records without them.

One helper for every worker (exact segcap, reward, and the balanced segment / caption workers):

- :func:`cert_binding` computes the binding of one shard kind once per container:
  ``validate.certkey.key_document(preset, kind, pool_gpu, modes=..., derived_content=...,
  dtbias_fp32=...)``, the job's embodiment and the GPU the container runs on. A key that cannot
  be computed (a derived weight without a content digest, an engine the GPU cannot run) leaves
  ``cert_key`` unset and says why (``problem``): such records stay uncertifiable (fail closed);
  the worker still runs.
- :func:`binding_from_env` builds the bindings of a container's shard kinds from the
  environment the app factory sets (:data:`ENV_EMBODIMENT`, :data:`ENV_POOL_GPUS`,
  :data:`ENV_DERIVED_CONTENT`, plus ``FLORA_PRESET`` and the output settings).
- :func:`stamp` applies a binding to a record; ``workers_impl.reward.ShardPartSink`` does it for
  every record it is given (``binding=``), next to the stage-hash stamp of ``config_hash``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from flora_deploy.records import StageRecord

ENV_EMBODIMENT = "FLORA_EMBODIMENT"  # cfg.pipeline.embodiment
ENV_POOL_GPUS = "FLORA_POOL_GPUS"  # JSON {shard kind: pool GPU} (RunSettings.pools)
ENV_DERIVED_CONTENT = "FLORA_DERIVED_CONTENT"  # JSON {name: content digest}
ENV_DTBIAS_FP32 = "FLORA_DTBIAS_FP32"  # "1": the balanced vLLM engines load the FP32 dt_bias


@dataclass(frozen=True)
class CertBinding:
    embodiment: str | None
    cert_key: dict[str, Any] | None
    gpu: str | None = None  # the container's GPU (nvidia-smi name); stamped when a record has none
    problem: str | None = None  # why ``cert_key`` is None


def kind_modes_from(kind: str, *, determinism: str = "nvidia",
                    terminal_mode: str = "terminal") -> dict[str, Any]:
    """``config.hashing.kind_modes`` from the applied run modes (what a worker knows)."""
    from flora_deploy.config.hashing import kind_modes

    pipeline = SimpleNamespace(determinism=determinism,
                               terminal_rationale=terminal_mode == "terminal_rationale")
    return kind_modes(SimpleNamespace(pipeline=pipeline), kind)


def cert_binding(preset: str, kind: str, pool_gpu: str | None, *, embodiment: str | None,
                 modes: Mapping[str, Any] | None = None,
                 derived_content: Mapping[str, str] | None = None,
                 dtbias_fp32: bool = False, gpu: str | None = None,
                 **key_kwargs: Any) -> CertBinding:
    """The binding of ``kind`` (module docstring). ``pool_gpu``: the pool's GPU type (what the
    certificate is for); ``gpu``: the device this container runs on. ``key_kwargs`` go to
    ``key_document`` (tests: ``image_fingerprint``, ``logic_root``)."""
    from flora_deploy.validate.certkey import CertKeyError, key_document

    if not pool_gpu:
        return CertBinding(embodiment, None, gpu, f"{kind}: no pool GPU known")
    try:
        key = key_document(preset, kind, pool_gpu, modes=modes, derived_content=derived_content,
                           dtbias_fp32=dtbias_fp32, **key_kwargs)
    except (CertKeyError, KeyError) as e:
        return CertBinding(embodiment, None, gpu, str(e))
    return CertBinding(embodiment, key, gpu)


def binding_from_env(kinds: Iterable[str], env: Mapping[str, str] | None = None, *,
                     gpu: str | None = None, **key_kwargs: Any) -> dict[str, CertBinding]:
    """Kind -> :class:`CertBinding` for a container serving ``kinds``, from the factory's
    environment (module docstring). Missing variables give bindings without ``cert_key``
    (``problem`` says which), never an error."""
    env = os.environ if env is None else env
    preset = env.get("FLORA_PRESET", "exact")
    embodiment = env.get(ENV_EMBODIMENT) or None
    try:
        pools = json.loads(env.get(ENV_POOL_GPUS) or "{}")
        content = json.loads(env.get(ENV_DERIVED_CONTENT) or "{}")
        outputs = json.loads(env.get("FLORA_OUTPUT_SETTINGS") or "{}")
    except ValueError as e:
        return {k: CertBinding(embodiment, None, gpu, f"bad binding environment: {e}")
                for k in kinds}
    determinism = env.get("FLORA_DETERMINISM") or outputs.get("determinism") or "nvidia"
    terminal = (env.get("FLORA_TERMINAL_MODE") or env.get("FLORA_REWARD_TERMINAL_MODE")
                or outputs.get("terminal_mode") or "terminal")
    out = {}
    for k in kinds:
        out[k] = cert_binding(preset, k, pools.get(k), embodiment=embodiment,
                              modes=kind_modes_from(k, determinism=determinism,
                                                    terminal_mode=terminal),
                              derived_content=content, gpu=gpu,
                              dtbias_fp32=env.get(ENV_DTBIAS_FP32) == "1", **key_kwargs)
    return out


def binding_env(cfg: Any, pools: Mapping[str, Any], *,
                derived_content: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment :func:`binding_from_env` reads, from a config (for the app factory and
    local runs). ``pools``: shard kind -> pool (``.gpu``) or GPU name."""
    gpus = {k: (p if isinstance(p, str) else getattr(p, "gpu", None)) for k, p in pools.items()}
    return {ENV_EMBODIMENT: str(cfg.pipeline.embodiment or ""),
            ENV_POOL_GPUS: json.dumps(gpus, sort_keys=True),
            ENV_DERIVED_CONTENT: json.dumps(dict(derived_content or {}), sort_keys=True)}


def stamp(rec: StageRecord, binding: CertBinding | None) -> StageRecord:
    """``rec`` with ``binding``'s embodiment, cert_key and (when unset) GPU in its provenance."""
    if binding is None:
        return rec
    p = rec.provenance
    return dataclasses.replace(rec, provenance=dataclasses.replace(
        p, embodiment=binding.embodiment if binding.embodiment is not None else p.embodiment,
        cert_key=binding.cert_key if binding.cert_key is not None else p.cert_key,
        gpu=p.gpu or binding.gpu))
