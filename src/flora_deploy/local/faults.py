"""Fault injection for the local runner (``flora run --local --inject ...``).

A :class:`FaultPlan` is a list of :class:`Fault`\\ s parsed from ``KIND:TARGET[xN]`` items
(comma-separated on the command line). Shard targets are ``fnmatch`` patterns on shard ids
(``segcap-00000``, ``reward.*``); episode targets match a uid, or an episode id (the part of the
uid after ``--``), optionally restricted to one shard kind with ``@kind`` (``ep003@segcap``).

=============  =======  ===========================================================================
kind           target   effect (the Modal behaviour it models)
=============  =======  ===========================================================================
``preempt``    shard    the container is preempted (SIGINT) after its first episode; the worker
                        flushes and marks the start ``preempted``; Modal restarts the same call
                        in a new container. Default ``x1`` (the first start of each match).
``oom``        episode  the container dies (OOM / SIGKILL: no mark, so the start counts toward
                        D15) whenever it reaches the episode; Modal restarts the call forever, so
                        the orchestrator quarantines, bisects and finally poison-pills it.
                        Default: every time.
``fail``       shard    a Python exception in the worker (``mark_exception``); counts against
                        ``modal.Retries`` (2 retries), then the call fails and the orchestrator
                        respawns it (``max_respawns``). Default ``x1``.
``infra``      shard    the call's input is lost before a container runs it (Modal reports the
                        call ``expired``); the orchestrator respawns. Default ``x1``.
``spawnerr``   shard    ``spawn`` itself raises a transient client error; the orchestrator
                        retries on its next tick. Default ``x1``.
``nverr``      episode  the fake model rejects the episode with NVIDIA's ``ValueError`` semantics:
                        an ``EpisodeError(kind="nv_failure")``, excused in compose (the job then
                        ends ``completed_with_errors``). Default: every time.
=============  =======  ===========================================================================

``preempt`` and ``oom`` fire at *checkpoints* the fake engines call before each episode
(:func:`checkpoint`), ``nverr`` in the engines (:func:`nv_rejects`), ``fail`` at a start's first
part / errors / audit / receipt write (:func:`maybe_fail`, via ``local.runner.FaultStore``),
``infra`` / ``spawnerr`` in the spawner. All of them read the container context the
:class:`~flora_deploy.local.spawner.LocalSpawner` sets for the thread running a call. A checkpoint
also delivers a ``cancel`` of the running call (as SIGINT, like ``@modal.exit`` after
``FunctionCall.cancel``).
"""

from __future__ import annotations

import fnmatch
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

SHARD_FAULTS = ("preempt", "fail", "infra", "spawnerr")
EPISODE_FAULTS = ("oom", "nverr")
FAULT_KINDS = SHARD_FAULTS + EPISODE_FAULTS
FOREVER = -1

_ITEM = re.compile(r"^(?P<kind>[a-z]+):(?P<target>[^\s,]+?)(?:x(?P<n>\d+))?$")


class ContainerCrash(BaseException):  # noqa: N818 - models a dead container, not an error
    """The container died (OOM / SIGKILL). A ``BaseException`` that is neither ``Exception`` nor
    ``KeyboardInterrupt``, so the worker runners' handlers never see it and leave the attempt
    marker unmarked, exactly like a killed process."""


class InjectedFailure(RuntimeError):
    """The ``fail`` fault: a Python exception in user code."""


class FakeNvidiaInputError(ValueError):
    """The ``nverr`` fault: NVIDIA's ``ValueError`` semantics for a bad episode."""


@dataclass
class Fault:
    kind: str
    target: str
    times: int  # remaining firings; FOREVER = unlimited
    fired: int = 0

    @property
    def episode_kind(self) -> str | None:
        return self.target.split("@", 1)[1] if "@" in self.target else None

    def matches_shard(self, shard_id: str) -> bool:
        return fnmatch.fnmatchcase(shard_id, self.target)

    def matches_episode(self, uid: str, shard_kind: str) -> bool:
        name, _, kind = self.target.partition("@")
        if kind and kind != shard_kind:
            return False
        return uid == name or uid.rsplit("--", 1)[-1] == name

    def spec(self) -> str:
        n = "" if self.times == FOREVER and self.kind in EPISODE_FAULTS else f"x{self.times}"
        return f"{self.kind}:{self.target}{n}"


def parse_fault(item: str) -> Fault:
    m = _ITEM.match(item.strip())
    if not m or m["kind"] not in FAULT_KINDS:
        raise ValueError(f"bad fault {item!r}: use KIND:TARGET[xN] with KIND one of "
                         f"{', '.join(FAULT_KINDS)} (e.g. preempt:segcap-00000, oom:ep003x5)")
    kind, target = m["kind"], m["target"]
    if kind in SHARD_FAULTS and "@" in target:
        raise ValueError(f"bad fault {item!r}: '@kind' only applies to episode faults")
    default = FOREVER if kind in EPISODE_FAULTS else 1
    times = int(m["n"]) if m["n"] is not None else default
    if times < 1 and times != FOREVER:
        raise ValueError(f"bad fault {item!r}: xN needs N >= 1")
    return Fault(kind, target, times)


def parse_faults(items: str | Iterable[str] | None) -> list[Fault]:
    """``"preempt:segcap-00000,oom:ep003"`` (or an iterable of such strings) -> faults."""
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    out = []
    for chunk in items:
        out += [parse_fault(x) for x in chunk.split(",") if x.strip()]
    return out


@dataclass
class FaultPlan:
    """Thread-safe fault bookkeeping shared by the spawner and the fake engines."""

    faults: list[Fault] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)  # every firing, for tests and the summary
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def parse(cls, items: str | Iterable[str] | None) -> FaultPlan:
        return cls(parse_faults(items))

    def __bool__(self) -> bool:
        return bool(self.faults)

    def take(self, kind: str, *, shard_id: str, shard_kind: str = "",
             uid: str | None = None) -> bool:
        """Consume one firing of the first matching fault of ``kind``; True if one fired."""
        with self._lock:
            for f in self.faults:
                if f.kind != kind or f.times == 0:
                    continue
                hit = (f.matches_episode(uid, shard_kind) if uid is not None
                       else f.matches_shard(shard_id))
                if not hit:
                    continue
                if f.times != FOREVER:
                    f.times -= 1
                f.fired += 1
                self.log.append({"fault": kind, "target": f.target, "shard_id": shard_id,
                                 "uid": uid})
                return True
        return False

    def fired(self, kind: str | None = None) -> int:
        with self._lock:
            return sum(1 for e in self.log if kind is None or e["fault"] == kind)


# --- container context -----------------------------------------------------------------------


@dataclass
class ContainerContext:
    """What a fake engine knows about the container running it (set per thread by the
    spawner): the call, the shard, the fault plan and how far this start got."""

    call_id: str
    container_id: str
    shard_id: str
    shard_kind: str
    faults: FaultPlan
    cancelled: threading.Event
    episodes: int = 0  # episodes this start has to do (set by the engine / episode hook)
    checkpoints: int = 0
    preempt_at: int | None = None  # checkpoint number at which this start is preempted
    fail_checked: bool = False


_CURRENT = threading.local()


def current() -> ContainerContext | None:
    return getattr(_CURRENT, "ctx", None)


def set_current(ctx: ContainerContext | None) -> None:
    _CURRENT.ctx = ctx


def checkpoint(uid: str) -> None:
    """Called by a fake engine before it works on episode ``uid``. Raises what the container
    would experience at that point: ``KeyboardInterrupt`` (cancel / preemption, i.e. SIGINT) or
    :class:`ContainerCrash` (OOM). Both are ``BaseException``\\ s, so they pass the runners' and
    ``stages.reward``'s per-episode ``except Exception`` isolation, as a signal would. A start
    with several episodes is preempted at its second checkpoint (after one episode), a
    single-episode start at its only one. Outside a spawner context it does nothing."""
    ctx = current()
    if ctx is None:
        return
    ctx.checkpoints += 1
    if ctx.cancelled.is_set():
        raise KeyboardInterrupt(f"call {ctx.call_id} cancelled")
    if ctx.preempt_at is None:
        hit = ctx.faults.take("preempt", shard_id=ctx.shard_id)
        ctx.preempt_at = min(2, max(ctx.episodes, 1)) if hit else 0
    if ctx.preempt_at and ctx.checkpoints == ctx.preempt_at:
        raise KeyboardInterrupt(f"container {ctx.container_id} preempted")
    if ctx.faults.take("oom", shard_id=ctx.shard_id, shard_kind=ctx.shard_kind, uid=uid):
        raise ContainerCrash(f"container {ctx.container_id} OOM-killed on {uid}")


def maybe_fail(what: str) -> None:
    """The ``fail`` fault, checked once per container start at its first store write of a part,
    errors or audit file (``local.runner.FaultStore``): a Python exception in the middle of the
    shard's user code, which the runner marks (``mark_exception``) and re-raises."""
    ctx = current()
    if ctx is None or ctx.fail_checked:
        return
    ctx.fail_checked = True
    if ctx.faults.take("fail", shard_id=ctx.shard_id):
        raise InjectedFailure(f"injected python failure in {ctx.shard_id} (writing {what})")


def note_episodes(n: int) -> None:
    """Tell the current start how many episodes it will process (preemption placement)."""
    ctx = current()
    if ctx is not None:
        ctx.episodes += n


def nv_rejects(uid: str) -> bool:
    """The ``nverr`` fault for ``uid`` in the current container (engines call it per episode)."""
    ctx = current()
    if ctx is None:
        return False
    return ctx.faults.take("nverr", shard_id=ctx.shard_id, shard_kind=ctx.shard_kind, uid=uid)
