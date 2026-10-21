"""Conformance suite for ``ShardWorker`` implementations (``flora_deploy.workers``, steps 0-6).

Every worker registers a :class:`WorkerSpec` (``tests/conformance/registry.py``): a name, the
shard kind it serves and a **factory** ``factory(env) -> ConformanceWorker``. The factory builds
the worker over ``env.store`` with ``env.container_id`` / ``env.call_id`` standing in for
``MODAL_TASK_ID`` / ``modal.current_function_call_id()``, and with its stage engine replaced by
a fake that follows ``env.script``:

* ``script.nv_fail``: uids whose stage reports an NVIDIA-semantics failure (the real engine's
  own failure path, e.g. NVIDIA returning "no segments"), which must become an
  ``EpisodeError(kind="nv_failure")`` without crashing the container;
* ``script.interrupt_before``: the uid during which SIGINT arrives (Modal preemption). The engine
  raises ``KeyboardInterrupt`` there, exactly what Python raises on SIGINT; ``run_shard`` must
  let it propagate (Modal must see the interruption) after flushing, and ``on_exit`` (the
  ``@modal.exit`` hook) runs afterwards.

Each check is a plain function ``check_x(spec, root)`` that raises ``AssertionError`` on a
violation, so ``test_workers.py`` runs every check for every registered worker *and* proves each
check catches the deliberately broken variants in ``reference_worker.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.manifest import from_json, load_manifest, save_manifest, stages_of
from flora_deploy.jobs.orchestrator import ShardLedger, call_path, utc
from flora_deploy.jobs.store import Store
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import (
    AttemptMarker,
    EpisodeError,
    EpisodeRef,
    Manifest,
    Shard,
    ShardKind,
    ShardReceipt,
    episode_uid,
)
from flora_deploy.records import SCHEMA, StageRecord

JOB, CFG = "conf-job", "cfgconf1"


# ----------------------------------------------------------------------------------------------
# What a worker factory receives / returns
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Script:
    nv_fail: frozenset[str] = frozenset()
    interrupt_before: str | None = None


@dataclass
class WorkerEnv:
    store: Store
    container_id: str
    call_id: str
    clock: Callable[[], float]
    script: Script = field(default_factory=Script)


class ConformanceWorker(Protocol):
    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None: ...

    def on_exit(self) -> None:
        """The ``@modal.exit`` path (runs after SIGINT, and on normal scale-down)."""
        ...


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    kind: ShardKind
    factory: Callable[[WorkerEnv], ConformanceWorker]

    def __str__(self) -> str:
        return self.name


# ----------------------------------------------------------------------------------------------
# World: a store with a manifest, and a write log
# ----------------------------------------------------------------------------------------------


class RecordingStore:
    """Wraps a Store and logs every ``put_bytes`` in order."""

    def __init__(self, inner: Store):
        self.inner = inner
        self.puts: list[tuple[str, bytes]] = []

    def put_bytes(self, path: str, data: bytes) -> None:
        self.inner.put_bytes(path, data)
        self.puts.append((path, bytes(data)))

    def get_bytes(self, path: str) -> bytes:
        return self.inner.get_bytes(path)

    def exists(self, path: str) -> bool:
        return self.inner.exists(path)

    def list(self, prefix: str) -> Iterator[str]:
        return self.inner.list(prefix)


class World:
    """One shard of ``n`` episodes (``SHARD``), optionally depending on an upstream ``segment``
    shard over the same episodes, saved as the job manifest."""

    SHARD_SEQ = "00000"

    def __init__(self, spec: WorkerSpec, root: Path, *, n: int = 4, upstream: bool = False):
        self.spec, self.kind = spec, spec.kind
        self.store = RecordingStore(LocalStore(root))
        self.refs = tuple(EpisodeRef(uid=episode_uid("conf/ds", f"ep{i}"), dataset="conf/ds",
                                     episode_id=f"ep{i}", pool="human", duration_s=5.0,
                                     oracle_eligible=True, envelope_ok=True, index_ref="x")
                          for i in range(n))
        self.uids = tuple(r.uid for r in self.refs)
        self.shard_id = f"{self.kind}-{self.SHARD_SEQ}"
        shards = []
        self.upstream: Shard | None = None
        if upstream:
            self.upstream = Shard("segment-up-00000", "segment", self.uids, 10.0)
            shards.append(self.upstream)
        self.shard = Shard(self.shard_id, self.kind, self.uids, 10.0,
                           depends_on=(self.upstream.shard_id,) if upstream else ())
        shards.append(self.shard)
        save_manifest(self.store, Manifest(JOB, CFG, self.refs, tuple(shards), "t"))
        self.jf = A.JobFiles(self.store, JOB, CFG)
        self._t = itertools.count(1_800_000_000)
        self._mark = 0

    def clock(self) -> float:
        return float(next(self._t))

    def token(self, n: int = 1) -> str:
        return A.spawn_token(JOB, CFG, self.shard_id, n)

    def ledger(self, n: int = 1, status: str = "active") -> None:
        """What the orchestrator writes before / after spawning call ``n``."""
        led = ShardLedger(self.shard_id, self.kind, status=status,  # type: ignore[arg-type]
                          call_id=f"fc-{n}", spawn_n=n, token=self.token(n),
                          updated_utc=utc(self.clock()))
        self.store.put_bytes(call_path(self.jf.paths, CFG, self.kind, self.shard_id),
                             led.to_bytes())

    def worker(self, container: str, script: Script | None = None) -> ConformanceWorker:
        return self.spec.factory(WorkerEnv(self.store, container, f"fc-{container}", self.clock,
                                           script or Script()))

    def mark(self) -> int:
        """Start a new section of the write log; returns its index."""
        self._mark = len(self.store.puts)
        return self._mark

    def puts_since(self, i: int | None = None) -> list[tuple[str, bytes]]:
        return self.store.puts[self._mark if i is None else i:]

    # file views -------------------------------------------------------------------------------
    def part_paths(self) -> list[str]:
        return sorted(p for p in self.store.list(P.part_prefix(self.jf, self.shard))
                      if p.endswith(".jsonl"))

    def records(self, paths: list[str] | None = None) -> list[StageRecord]:
        return [r for p in (paths if paths is not None else self.part_paths())
                for r in P.read_part(self.jf, p)]

    def own_parts(self, container: str) -> list[str]:
        return [p for p in self.part_paths() if p.rsplit("/", 1)[-1].startswith(f"{container}-")]

    def receipt(self) -> ShardReceipt | None:
        return P.read_receipt(self.jf, self.kind, self.shard_id)

    def marker(self, container: str) -> AttemptMarker | None:
        return A.read_attempt(self.jf, self.kind, self.shard_id, container)

    def errors(self) -> list[EpisodeError]:
        prefix = self.jf.paths.errors(CFG, self.kind, self.shard_id, "x", 0)[:-len("x-0.jsonl")]
        out: list[EpisodeError] = []
        for p in self.store.list(prefix):
            out += P.read_errors(self.jf, p)
        return out

    def seed_previous(self, uids: tuple[str, ...], container: str = "ta-prev") -> str:
        """Parts an earlier container of this shard left behind (a valid previous attempt)."""
        A.write_attempt(self.jf, self.kind, self.shard_id, container, started_utc=utc(1.0),
                        orchestrator_attempt=1, token=self.token(1), call_id="fc-1")
        return P.write_part(self.jf, self.kind, self.shard_id, container, 0,
                            [reference_record(r, st) for r in self.refs if r.uid in uids
                             for st in stages_of(self.kind)])

    def poison_upstream(self, uid: str) -> None:
        assert self.upstream is not None
        child = Shard(A.child_shard_id("segment", uid), "segment", (uid,), 1.0,
                      bisected_from=self.upstream.shard_id)
        A.write_poison(self.jf, child, message="poisoned upstream (test)")

    def run(self, container: str, *, token_n: int | None = 1, script: Script | None = None,
            ledger: bool = True) -> ShardReceipt | None:
        if ledger and token_n is not None:
            self.ledger(token_n)
        tok = self.token(token_n) if token_n is not None else ""
        return self.worker(container, script).run_shard(JOB, self.shard_id, CFG, tok)


def reference_record(ref: EpisodeRef, stage: str) -> StageRecord:
    """A valid record (what a previous container of a conformant worker could have written)."""
    from flora_deploy.records import Provenance, Segment

    return StageRecord(uid=ref.uid, dataset=ref.dataset, episode_id=ref.episode_id,
                       stage=stage,  # type: ignore[arg-type]
                       provenance=Provenance(models={"seed": "0"}, engine="seed", preset="exact",
                                             config_hash=CFG, logic_version="seed",
                                             framework="flora-deploy test"),
                       certified=True, segments=(Segment(f"{ref.uid}:0", 0.0, 1.0),))


def _interrupted(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except KeyboardInterrupt:
        return True
    return False


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------


def check_superseded_returns_none(spec: WorkerSpec, root: Path) -> None:
    """Step 0: a spawn the ledger no longer wants returns None and writes nothing."""
    w = World(spec, root)
    w.ledger(2)  # the orchestrator has moved on to spawn 2
    w.mark()
    out = w.worker("ta-stale").run_shard(JOB, w.shard_id, CFG, w.token(1))
    assert out is None, f"superseded spawn returned {out!r}"
    assert w.puts_since() == [], f"superseded spawn wrote {[p for p, _ in w.puts_since()]}"
    # same token, but the shard is already done: also a stale duplicate
    w.ledger(2, status="done")
    w.mark()
    out = w.worker("ta-dup").run_shard(JOB, w.shard_id, CFG, w.token(2))
    assert out is None and w.puts_since() == [], "a call for a finished shard did work"


def check_marker_first(spec: WorkerSpec, root: Path) -> None:
    """Step 1: the first write is this container's AttemptMarker, carrying token + call id."""
    w = World(spec, root)
    w.ledger(1)
    i = w.mark()
    w.worker("ta-1").run_shard(JOB, w.shard_id, CFG, w.token(1))
    puts = w.puts_since(i)
    assert puts, "worker wrote nothing"
    first = puts[0][0]
    want = w.jf.paths.attempt(CFG, w.kind, w.shard_id, "ta-1")
    assert first == want, f"first write was {first!r}, expected the attempt marker {want!r}"
    m = from_json(AttemptMarker, puts[0][1])
    assert m.token == w.token(1), f"marker token {m.token!r} != spawn token {w.token(1)!r}"
    assert m.call_id == "fc-ta-1", f"marker call_id {m.call_id!r} != the Modal call id"
    assert m.shard_id == w.shard_id and m.container_id == "ta-1"
    assert m.orchestrator_attempt == 1, "orchestrator_attempt must be the token's spawn number"


def check_skip_done(spec: WorkerSpec, root: Path) -> None:
    """Step 2: episodes already in a previous container's parts are not run again."""
    w = World(spec, root)
    done = w.uids[:2]
    w.seed_previous(done)
    rec = w.run("ta-2")
    redone = {r.uid for r in w.records(w.own_parts("ta-2"))} & set(done)
    assert not redone, f"re-ran episodes a previous container already finished: {sorted(redone)}"
    assert rec is not None and set(rec.episodes_done) == set(w.uids), \
        "skipped episodes must still be reported done"


def check_upstream_poison(spec: WorkerSpec, root: Path) -> None:
    """Step 3: an episode whose upstream stage was poison-pilled is not run; it gets an
    ``EpisodeError(poison_pill)`` and is reported in ``episodes_failed``."""
    w = World(spec, root, upstream=True)
    bad = w.uids[1]
    w.poison_upstream(bad)
    rec = w.run("ta-1")
    assert bad not in {r.uid for r in w.records()}, "ran an upstream-poisoned episode"
    assert rec is not None, "worker returned no receipt"
    assert bad in rec.episodes_failed and bad not in rec.episodes_done, \
        f"poisoned {bad} must be in episodes_failed only: {rec}"
    errs = {(e.uid, e.stage, e.kind) for e in w.errors()}
    for st in stages_of(w.kind):
        assert (bad, st, "poison_pill") in errs, f"no poison_pill EpisodeError for {bad}/{st}"


def check_nv_failure(spec: WorkerSpec, root: Path) -> None:
    """Step 4: an NVIDIA-semantics failure is an EpisodeError(nv_failure); the shard finishes."""
    w = World(spec, root)
    bad = w.uids[2]
    try:
        rec = w.run("ta-1", script=Script(nv_fail=frozenset({bad})))
    except Exception as exc:
        raise AssertionError(f"an NVIDIA failure crashed the worker: {exc!r}") from exc
    assert rec is not None
    assert bad in rec.episodes_failed and bad not in rec.episodes_done
    assert set(rec.episodes_done) == set(w.uids) - {bad}
    errs = {(e.uid, e.kind) for e in w.errors()}
    assert (bad, "nv_failure") in errs, f"no nv_failure EpisodeError for {bad}: {errs}"
    assert bad not in {r.uid for r in w.records()}, "a failed episode must have no record"
    m = w.marker("ta-1")
    assert m is not None and m.error is None, "an NVIDIA failure is not a python exception"


def check_records_valid(spec: WorkerSpec, root: Path) -> None:
    """Records validate as ``flora_labels_v1`` with the right uid / stage / provenance."""
    w = World(spec, root)
    w.run("ta-1")
    by_uid = {r.uid: r for r in w.refs}
    keys = set()
    for p in w.part_paths():
        for line in w.store.get_bytes(p).decode().splitlines():
            if not line.strip():
                continue
            rec = StageRecord.from_json(line)  # raises on a bad schema / shape
            assert rec.schema == SCHEMA
            assert rec.to_json() == line, "part lines must be canonical StageRecord.to_json()"
            assert rec.uid in by_uid, f"record for unknown uid {rec.uid!r}"
            assert rec.stage in stages_of(w.kind), f"stage {rec.stage!r} for kind {w.kind!r}"
            ref = by_uid[rec.uid]
            assert (rec.dataset, rec.episode_id) == (ref.dataset, ref.episode_id), \
                f"{rec.uid}: dataset/episode_id must come from the EpisodeRef"
            pv = rec.provenance
            assert pv.config_hash == CFG, f"provenance.config_hash {pv.config_hash!r} != {CFG!r}"
            assert pv.engine and pv.logic_version and pv.framework and pv.preset, \
                "provenance must name engine, preset, logic_version and framework"
            keys.add((rec.uid, rec.stage))
    want = {(u, st) for u in w.uids for st in stages_of(w.kind)}
    assert keys == want, f"missing records: {sorted(want - keys)}"


def check_parts_whole(spec: WorkerSpec, root: Path) -> None:
    """Parts are written whole, once per (shard, container, k), and never extended."""
    w = World(spec, root, n=5)
    i = w.mark()
    w.run("ta-1")
    prefix = P.part_prefix(w.jf, w.shard)
    seen: dict[str, int] = {}
    for path, data in w.puts_since(i):
        if not path.startswith(prefix):
            continue
        seen[path] = seen.get(path, 0) + 1
        assert path.rsplit("/", 1)[-1].startswith("ta-1-"), f"{path}: not this container's part"
        text = data.decode()
        assert text and text.endswith("\n"), f"{path}: not whole JSONL lines"
        for line in text.splitlines():
            StageRecord.from_json(line)
    again = sorted(p for p, n in seen.items() if n > 1)
    assert not again, f"part files rewritten (appended): {again}"
    assert seen, "no part written"


def check_preempt_flush_then_mark(spec: WorkerSpec, root: Path) -> None:
    """Step 5: on SIGINT the finished episodes are flushed, then the marker gets
    ``preempted=True``; no receipt. Modal's retry in a new container skips them."""
    w = World(spec, root, n=5)
    w.ledger(1)
    wk = w.worker("ta-1", Script(interrupt_before=w.uids[3]))
    assert _interrupted(lambda: wk.run_shard(JOB, w.shard_id, CFG, w.token(1))), \
        "run_shard must let the KeyboardInterrupt (SIGINT) propagate"
    wk.on_exit()
    have = {r.uid for r in w.records()}
    assert set(w.uids[:3]) <= have, f"finished episodes lost on preemption: " \
        f"{sorted(set(w.uids[:3]) - have)}"
    m = w.marker("ta-1")
    assert m is not None and m.preempted and m.ended_utc, "marker not flagged preempted"
    assert not m.crashed and not m.error, \
        f"a preemption recorded as a crash / error (crashed={m.crashed}, error={m.error!r})"
    assert w.receipt() is None, "a preempted start must not write a receipt"
    assert A.counted_starts(w.jf, w.kind, w.shard_id) == 0, "a preempted start counted for D15"
    rec = w.worker("ta-2").run_shard(JOB, w.shard_id, CFG, w.token(1))
    redone = {r.uid for r in w.records(w.own_parts("ta-2"))} & set(w.uids[:3])
    assert rec is not None and not redone, f"retry re-ran flushed episodes {sorted(redone)}"


def check_receipt_last_with_all_parts(spec: WorkerSpec, root: Path) -> None:
    """Step 6: the receipt is the last write and names every part of the shard."""
    w = World(spec, root)
    old = w.seed_previous(w.uids[:1])
    i = w.mark()
    rec = w.run("ta-2")
    puts = w.puts_since(i)
    rpath = w.jf.paths.receipt(CFG, w.kind, w.shard_id)
    assert puts and puts[-1][0] == rpath, \
        f"last write was {puts[-1][0] if puts else None!r}, not the receipt"
    assert sum(1 for p, _ in puts if p == rpath) == 1, "receipt written more than once"
    stored = w.receipt()
    assert stored is not None and rec is not None and stored == rec, \
        "returned receipt differs from the stored one"
    assert set(stored.parts) == set(w.part_paths()), \
        f"receipt.parts {stored.parts} != every part of the shard {w.part_paths()}"
    assert old in stored.parts, "receipt must name the earlier containers' parts"
    assert (stored.shard_id, stored.kind, stored.config_hash, stored.container_id) == \
        (w.shard_id, w.kind, CFG, "ta-2")
    assert set(stored.episodes_done) == set(w.uids) and not stored.episodes_failed


def check_idempotent_rerun(spec: WorkerSpec, root: Path) -> None:
    """Preempted start + retry + a manual re-run: compose sees every (uid, stage) exactly once."""
    w = World(spec, root, n=5)
    w.ledger(1)
    wk = w.worker("ta-1", Script(interrupt_before=w.uids[2]))
    _interrupted(lambda: wk.run_shard(JOB, w.shard_id, CFG, w.token(1)))
    wk.on_exit()
    w.worker("ta-2").run_shard(JOB, w.shard_id, CFG, w.token(1))
    w.worker("ta-3").run_shard(JOB, w.shard_id, CFG, "")  # manual re-run (no token)
    prog = P.derive(w.jf, load_manifest(w.store, JOB), limit=3)
    assert prog.shards[w.shard_id].state == "done"
    ci = P.compose_input(w.jf, prog)
    assert not ci.missing, f"missing after compose: {sorted(ci.missing)}"
    assert not ci.conflicts and ci.foreign == 0
    assert ci.duplicates == 0, f"{ci.duplicates} duplicate (uid, stage) records across parts"
    assert set(ci.records) == {(u, st) for u in w.uids for st in stages_of(w.kind)}
    counts: dict[tuple[str, str], int] = {}
    for r in w.records():
        counts[(r.uid, r.stage)] = counts.get((r.uid, r.stage), 0) + 1
    dup = sorted(k for k, n in counts.items() if n > 1)
    assert not dup, f"duplicate records in part files: {dup}"


CHECKS: dict[str, Callable[[WorkerSpec, Path], None]] = {
    "superseded": check_superseded_returns_none,
    "marker_first": check_marker_first,
    "skip_done": check_skip_done,
    "upstream_poison": check_upstream_poison,
    "nv_failure": check_nv_failure,
    "records_valid": check_records_valid,
    "parts_whole": check_parts_whole,
    "preempt": check_preempt_flush_then_mark,
    "receipt_last": check_receipt_last_with_all_parts,
    "idempotent": check_idempotent_rerun,
}

