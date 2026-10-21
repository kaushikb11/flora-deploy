"""Certification provenance every record must carry (``docs/certification.md``, "Provenance a
record must carry"): ``embodiment``, ``cert_key`` and ``gpu`` (``config_hash`` is stamped by the
part sink). Shared by every worker: compute a :class:`CertStamp` once per container, then
``stamp.apply(record)`` per record.

``cert_key`` is ``validate.certkey.key_document`` of the stack the container actually runs:
preset, shard kind, the **pool** GPU (it selects the vLLM per-GPU limits), the kind's
output-affecting modes (``config.hashing.kind_modes``), the content digests of the derived
weights the container verified at start, and the ``dt_bias`` plugin flag. A worker whose key
cannot be computed (e.g. a derived weight without a content digest) fails closed at start.

Container env (set by ``modal_app.factory``): ``FLORA_EMBODIMENT``, ``FLORA_POOL_GPUS`` (JSON
kind -> pool GPU) and ``FLORA_KIND_MODES`` (JSON kind -> modes).
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from flora_deploy.records import StageRecord

EMBODIMENT_ENV = "FLORA_EMBODIMENT"
POOL_GPUS_ENV = "FLORA_POOL_GPUS"
KIND_MODES_ENV = "FLORA_KIND_MODES"


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


def cert_stamp(preset: str, kind: str, pool_gpu: str, *, embodiment: str | None,
               modes: Mapping[str, Any] | None = None,
               derived_content: Mapping[str, str] | None = None, gpu: str | None = None,
               dtbias_fp32: bool = False, **key_kw: Any) -> CertStamp:
    """The stamp of a real stack; raises :class:`CertStampError` when the key cannot be built."""
    from flora_deploy.validate.certkey import CertKeyError, key_document

    if not embodiment:
        raise CertStampError("no embodiment configured (FLORA_EMBODIMENT)")
    try:
        doc = key_document(preset, kind, pool_gpu, modes=modes, derived_content=derived_content,
                           dtbias_fp32=dtbias_fp32, **key_kw)
    except CertKeyError as e:
        raise CertStampError(f"{kind}: cannot compute the certification key: {e}") from None
    return CertStamp(embodiment, doc, gpu)


def stamp_from_env(kind: str, env: Mapping[str, str] = os.environ, *,
                   derived_content: Mapping[str, str] | None = None, gpu: str | None = None,
                   dtbias_fp32: bool = False, **key_kw: Any) -> CertStamp:
    """:func:`cert_stamp` from the container env the factory sets (module docstring)."""
    try:
        pools = json.loads(env.get(POOL_GPUS_ENV) or "{}")
        modes = json.loads(env.get(KIND_MODES_ENV) or "{}")
    except ValueError as e:
        raise CertStampError(f"{POOL_GPUS_ENV} / {KIND_MODES_ENV}: not JSON ({e})") from None
    pool_gpu = pools.get(kind)
    if not pool_gpu:
        raise CertStampError(f"{POOL_GPUS_ENV} names no pool GPU for {kind!r}")
    return cert_stamp(env.get("FLORA_PRESET", ""), kind, pool_gpu,
                      embodiment=env.get(EMBODIMENT_ENV) or None, modes=modes.get(kind) or {},
                      derived_content=derived_content, gpu=gpu, dtbias_fp32=dtbias_fp32,
                      **key_kw)


def fake_stamp(env: Mapping[str, str] = os.environ, *, gpu: str | None = None) -> CertStamp:
    """Fake-model workers: embodiment and GPU only (``flora certify`` refuses fake records
    outright, and fake derived weights have no content digest to key)."""
    return CertStamp(env.get(EMBODIMENT_ENV) or None, None, gpu)
