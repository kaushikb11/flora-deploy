"""Certification provenance every record must carry (``docs/certification.md``, "Provenance a
record must carry"): ``embodiment``, ``cert_key`` and ``gpu`` (``config_hash``, the 64-hex stage
hash, is stamped by the part sink). The **one** helper every worker kind uses: exact
``segcap`` and ``reward.terminal`` / ``reward.dense`` (``workers_impl.reward.ShardPartSink``,
``stamp=``, from ``cert_stamps`` of the runner settings) and the balanced ``segment`` /
``caption`` workers (``workers_impl.balanced.StampedJob``). Compute the stamps once per
container (:func:`stamps_from_env` / :func:`stamp_from_env`; :func:`fake_stamps` for fake-model
workers), then ``stamp.apply(record)`` per record.

A new worker kind (e.g. ``reward.rubric``) needs one line: the factory already sets the env on
every worker class, so ``settings = with_stamps(reward_settings_from_env(env, gpu=gpu_name()),
env)`` (``fake=True`` in a fake container) is all it takes, exactly as ``modal_app.
workers_app.RewardWorker`` / ``fake_reward_runner`` do for the other reward kinds;
``tests/unit/test_certstamp.py`` then covers it once the kind is in a preset.

``cert_key`` is ``validate.certkey.key_document`` of the stack the container actually runs:
preset, shard kind, the **pool** GPU (it selects the vLLM per-GPU limits), the kind's
output-affecting modes (``config.hashing.kind_modes``), the content digests of the derived
weights the container verified at start, and the ``dt_bias`` plugin flag. A worker whose key
cannot be computed (missing env, a derived weight without a content digest) fails closed at
start (:class:`CertStampError`).

Container env (:func:`cert_env`, set by ``modal_app.factory.plan_app`` on **every** worker
class and by ``local.runner`` for local runs): ``FLORA_EMBODIMENT``, ``FLORA_POOL_GPUS`` (JSON
kind -> pool GPU) and ``FLORA_KIND_MODES`` (JSON kind -> modes), next to ``FLORA_PRESET``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from flora_deploy.records import StageRecord

EMBODIMENT_ENV = "FLORA_EMBODIMENT"
POOL_GPUS_ENV = "FLORA_POOL_GPUS"
KIND_MODES_ENV = "FLORA_KIND_MODES"
PRESET_ENV = "FLORA_PRESET"
SHARD_KINDS_ENV = "FLORA_SHARD_KINDS"


class CertStampError(RuntimeError):
    """The container cannot say which stack produced its records: refuse to start."""


@dataclass(frozen=True)
class CertStamp:
    embodiment: str | None
    cert_key: Mapping[str, Any] | None = None
    gpu: str | None = None  # the nvidia-smi name; a record's own gpu wins
    extra: Mapping[str, Any] = field(default_factory=dict)

    def apply(self, rec: StageRecord) -> StageRecord:
        p = rec.provenance
        return dataclasses.replace(rec, provenance=dataclasses.replace(
            p, embodiment=p.embodiment or self.embodiment,
            cert_key=p.cert_key if p.cert_key is not None else (
                dict(self.cert_key) if self.cert_key is not None else None),
            gpu=p.gpu or self.gpu))


def cert_env(cfg: Any, pools: Mapping[str, Any]) -> dict[str, str]:
    """The env :func:`stamp_from_env` reads, from a config, for the shard kinds of ``pools``
    (kind -> ``PoolSettings`` or GPU name). ``FLORA_PRESET`` is set alongside by the caller."""
    from flora_deploy.config.hashing import kind_modes

    gpus = {k: (p if isinstance(p, str) else p.gpu) for k, p in pools.items()}
    return {POOL_GPUS_ENV: json.dumps(gpus, sort_keys=True),
            KIND_MODES_ENV: json.dumps({k: kind_modes(cfg, k) for k in pools}, sort_keys=True),
            EMBODIMENT_ENV: str(cfg.pipeline.embodiment or "")}


def pinned_content(preset: str, kind: str) -> dict[str, str]:
    """Content digests of the derived weights ``kind`` loads, for workers whose loader refuses
    anything but the pinned content (exact segcap: ``accel.captioner_load``). A name whose pin
    has no digest is left out, so the key fails closed."""
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.config.receipts import pin_key
    from flora_deploy.validate.certkey import needed_derived

    out = {}
    for n in needed_derived(preset, kind):
        pin = DERIVED_PINS_BY_KEY.get(pin_key(n))
        if pin is not None and pin.content_sha256:
            out[n] = pin.content_sha256
    return out


def cert_stamp(preset: str, kind: str, pool_gpu: str, *, embodiment: str | None,
               modes: Mapping[str, Any] | None = None,
               derived_content: Mapping[str, str] | None = None, gpu: str | None = None,
               dtbias_fp32: bool = False, **key_kw: Any) -> CertStamp:
    """The stamp of a real stack; raises :class:`CertStampError` when the key cannot be built."""
    from flora_deploy.validate.certkey import CertKeyError, key_document

    if not embodiment:
        raise CertStampError(f"no embodiment configured ({EMBODIMENT_ENV})")
    try:
        doc = key_document(preset, kind, pool_gpu, modes=modes, derived_content=derived_content,
                           dtbias_fp32=dtbias_fp32, **key_kw)
    except (CertKeyError, KeyError) as e:
        raise CertStampError(f"{kind}: cannot compute the certification key: {e}") from None
    return CertStamp(embodiment, doc, gpu)


def _read_env(kind: str, env: Mapping[str, str]) -> tuple[str, str, dict[str, Any]]:
    try:
        pools = json.loads(env.get(POOL_GPUS_ENV) or "{}")
        modes = json.loads(env.get(KIND_MODES_ENV) or "{}")
    except ValueError as e:
        raise CertStampError(f"{POOL_GPUS_ENV} / {KIND_MODES_ENV}: not JSON ({e})") from None
    preset = env.get(PRESET_ENV) or ""
    if not preset:
        raise CertStampError(f"{PRESET_ENV} is not set")
    pool_gpu = pools.get(kind)
    if not pool_gpu:
        raise CertStampError(f"{POOL_GPUS_ENV} names no pool GPU for {kind!r}")
    if kind not in modes:
        raise CertStampError(f"{KIND_MODES_ENV} names no modes for {kind!r}")
    return preset, str(pool_gpu), dict(modes[kind] or {})


def stamp_from_env(kind: str, env: Mapping[str, str] = os.environ, *,
                   derived_content: Mapping[str, str] | None = None, gpu: str | None = None,
                   dtbias_fp32: bool = False, **key_kw: Any) -> CertStamp:
    """:func:`cert_stamp` from the container env the factory sets (module docstring)."""
    preset, pool_gpu, modes = _read_env(kind, env)
    return cert_stamp(preset, kind, pool_gpu, embodiment=env.get(EMBODIMENT_ENV) or None,
                      modes=modes, derived_content=derived_content, gpu=gpu,
                      dtbias_fp32=dtbias_fp32, **key_kw)


def stamps_from_env(kinds: Iterable[str], env: Mapping[str, str] = os.environ, *,
                    gpu: str | None = None, derived_content: Mapping[str, str] | None = None,
                    **key_kw: Any) -> dict[str, CertStamp]:
    """Kind -> :func:`stamp_from_env` for a container serving ``kinds`` (fails closed)."""
    return {k: stamp_from_env(k, env, gpu=gpu, derived_content=derived_content, **key_kw)
            for k in kinds}


def container_kinds(env: Mapping[str, str] = os.environ) -> list[str]:
    """The shard kinds this container serves (``FLORA_SHARD_KINDS``, set by the factory)."""
    kinds = [k.strip() for k in (env.get(SHARD_KINDS_ENV) or "").split(",") if k.strip()]
    if not kinds:
        raise CertStampError(f"{SHARD_KINDS_ENV} is not set")
    return kinds


def container_stamps(env: Mapping[str, str] = os.environ, *, gpu: str | None = None,
                     fake: bool = False, derived_content: Mapping[str, str] | None = None,
                     **key_kw: Any) -> dict[str, CertStamp]:
    """Kind -> stamp for every kind of this container (:func:`container_kinds`):
    :func:`stamps_from_env` or, with ``fake``, :func:`fake_stamps`."""
    kinds = container_kinds(env)
    if fake:
        return fake_stamps(kinds, env, gpu=gpu, **key_kw)
    return stamps_from_env(kinds, env, gpu=gpu, derived_content=derived_content, **key_kw)


def with_stamps(settings: Any, env: Mapping[str, str] = os.environ, *, fake: bool = False,
                derived_content: Mapping[str, str] | None = None, **key_kw: Any) -> Any:
    """``settings`` (``SegcapWorkerSettings`` / ``RewardWorkerSettings``: any frozen dataclass
    with ``gpu`` and ``cert_stamps``) with :func:`container_stamps` for its container."""
    return dataclasses.replace(settings, cert_stamps=container_stamps(
        env, gpu=settings.gpu, fake=fake, derived_content=derived_content, **key_kw))


def fake_content(preset: str, kind: str) -> dict[str, str]:
    """64-hex stand-ins for the content digests of a fake container's derived weights (fake
    weights have none); the records are fake, so ``flora certify`` refuses them anyway."""
    from flora_deploy.validate.certkey import needed_derived

    return {n: hashlib.sha256(f"flora-fake-models:{n}".encode()).hexdigest()
            for n in needed_derived(preset, kind)}


def fake_stamp(kind: str, env: Mapping[str, str] = os.environ, *, gpu: str | None = None,
               **key_kw: Any) -> CertStamp:
    """Fake-model workers: the same fields from the same env as :func:`stamp_from_env`, with
    stand-in derived digests (:func:`fake_content`) and the pool GPU when the container has
    none (CPU containers), and a stand-in image fingerprint (fake workers do not run the
    kind's image; this also keeps ``modal_app`` unimported in local runs). ``flora certify``
    refuses fake records outright."""
    preset, pool_gpu, _ = _read_env(kind, env)
    key_kw.setdefault("image_fingerprint", lambda name: f"flora-fake-models:{name}")
    return stamp_from_env(kind, env, derived_content=fake_content(preset, kind),
                          gpu=gpu or pool_gpu, **key_kw)


def fake_stamps(kinds: Iterable[str], env: Mapping[str, str] = os.environ, *,
                gpu: str | None = None, **key_kw: Any) -> dict[str, CertStamp]:
    """Kind -> :func:`fake_stamp`."""
    return {k: fake_stamp(k, env, gpu=gpu, **key_kw) for k in kinds}
