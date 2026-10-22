"""Per-episode job wrappers for the balanced workers (``workers_impl.segment`` /
``workers_impl.caption``), applied by the Modal worker (``modal_app.balanced_app``) and the
fake workers alike:

* :class:`AccelJob` runs each episode coroutine inside ``active.episode()`` of the OPT-1
  ``accel`` patches (``docs/accel.md``: decode-once, in-memory PNG, the transport and geometry
  pools). The patches swap NVIDIA attributes the balanced stages call (``frames_for``,
  ``Predictor.predict``, ``caption_*``, ``human_transport``, ...), each proven output-identical
  by its registered parity test; :func:`apply_accel` applies only registered patches.
  ``active.episode()`` binds a per-episode cache / decoder scope to the coroutine's context
  (``asyncio.to_thread`` copies it), so overlapping episodes each get their own scope, released
  when that episode ends. The scope wraps the whole ``job.run``, so it covers both phases of
  a two-phase episode (``workers_impl.segment.EpisodeGates``: the probe before the batcher's
  episode, inference inside it). ``tests/nvlink/test_balanced_accel.py`` checks that every
  model request is byte-identical with and without the patches, and on both paths.
* :class:`StampedJob` stamps each record with the certification provenance
  (``workers_impl.certstamp``) and its ``certified`` flag: a committed certificate matching the
  stack's key (``CertStamp.certify``) and the episode's measured envelope verdict.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Mapping
from typing import Any

ACCEL_ENV = "FLORA_BALANCED_ACCEL"  # comma-separated patch names, "all" (default) or "none"


class _Wrapped:
    def __init__(self, job: Any):
        self.job = job

    @property
    def stage(self) -> str:
        return self.job.stage

    @property
    def nv(self) -> Any:
        return self.job.nv

    def __getattr__(self, name: str) -> Any:  # e.g. ``segments`` of a caption job
        return getattr(self.job, name)


class AccelJob(_Wrapped):
    """``EpisodeJob`` whose episodes each run inside ``active.episode()``."""

    def __init__(self, job: Any, active: Any):
        super().__init__(job)
        self.active = active

    async def run(self, ref, batcher, ctx):
        with self.active.episode():
            out = await self.job.run(ref, batcher, ctx)
        prov = self.active.provenance()
        out.audits = [{**a, "accel": prov} for a in out.audits]
        return out


class StampedJob(_Wrapped):
    """``EpisodeJob`` whose records carry ``stamp`` (a ``certstamp.CertStamp``)."""

    def __init__(self, job: Any, stamp: Any):
        super().__init__(job)
        self.stamp = stamp

    async def run(self, ref, batcher, ctx):
        out = await self.job.run(ref, batcher, ctx)
        certify = getattr(self.stamp, "certify", None)
        recs = [self.stamp.apply(r) for r in out.records]
        if callable(certify):
            recs = [certify(r, getattr(ref, "envelope_ok", None)) for r in recs]
        return dataclasses.replace(out, records=recs)


def accel_names(env: Mapping[str, str] = os.environ) -> tuple[str, ...]:
    """The patches to apply: ``FLORA_BALANCED_ACCEL`` (default all registered patches)."""
    from flora_deploy.accel import ALL_PATCHES

    raw = (env.get(ACCEL_ENV) or "all").strip().lower()
    if raw in ("none", "off", "0", ""):
        return ()
    if raw == "all":
        return tuple(ALL_PATCHES)
    return tuple(n.strip() for n in raw.split(",") if n.strip())


def apply_accel(nv: Any, names: Iterable[str] | None = None, **options: Any) -> Any:
    """``accel.apply_patches(nv, names)`` or ``None`` when no patch is wanted. Refuses a patch
    without a parity test (every registered one has one)."""
    from flora_deploy.accel import apply_patches, registered

    names = tuple(accel_names() if names is None else names)
    if not names:
        return None
    reg = registered()
    unproven = [n for n in names if n in reg and not reg[n].parity_test]
    if unproven:
        raise ValueError(f"accel patches without an identity proof: {unproven}")
    return apply_patches(nv, names, **options)


def wrap_job(job: Any, *, active: Any = None, stamp: Any = None) -> Any:
    """The job the runner gets: accel scope innermost, the stamp outermost."""
    if active is not None:
        job = AccelJob(job, active)
    if stamp is not None:
        job = StampedJob(job, stamp)
    return job
