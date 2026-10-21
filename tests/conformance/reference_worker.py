"""``FakeConformantWorker``: the reference ``ShardWorker`` (passes every conformance check), and
deliberately broken variants, one per check, that prove each check catches its violation.

The "stage" is a fake that emits one valid ``StageRecord`` per stage of the shard kind and
follows the suite's :class:`~tests.conformance.worker_suite.Script` (NVIDIA failures, SIGINT).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.deps import StoreManifestSource
from flora_deploy.jobs.manifest import stages_of
from flora_deploy.jobs.orchestrator import utc
from flora_deploy.jobs.types import EpisodeError, EpisodeRef, ShardKind, ShardReceipt
from flora_deploy.records import Provenance, Segment, StageRecord
from tests.conformance.worker_suite import WorkerEnv, WorkerSpec


class NvSemanticsFailure(Exception):
    """What the fake engine raises where NVIDIA's own pipeline would report a failure."""


@dataclass(frozen=True)
class Breakage:
    ignore_superseded: bool = False
    marker_late: bool = False
    marker_without_token: bool = False
    no_skip_done: bool = False
    run_poisoned: bool = False
    crash_on_nv_failure: bool = False
    foreign_provenance: bool = False
    append_parts: bool = False
    no_preempt_mark: bool = False
    swallow_interrupt: bool = False
    receipt_own_parts: bool = False
    receipt_first: bool = False
    duplicate_records: bool = False


class FakeConformantWorker:
    flush_every = 2  # episodes per part file (and every SIGINT)

    def __init__(self, env: WorkerEnv, kind: ShardKind, broken: Breakage | None = None):
        self.env, self.kind, self.b = env, kind, broken or Breakage()
        self._active: dict | None = None

    # the fake engine ------------------------------------------------------------------------
    def stage(self, ref: EpisodeRef, stage: str, cfg: str) -> StageRecord:
        if ref.uid == self.env.script.interrupt_before:
            raise KeyboardInterrupt("SIGINT")
        if ref.uid in self.env.script.nv_fail:
            raise NvSemanticsFailure(f"NVIDIA returned no segments for {ref.uid}")
        cfg = "someone-else" if self.b.foreign_provenance else cfg
        return StageRecord(uid=ref.uid, dataset=ref.dataset, episode_id=ref.episode_id,
                           stage=stage,  # type: ignore[arg-type]
                           provenance=Provenance(models={"fake": "0"}, engine="fake-engine",
                                                 preset="exact", config_hash=cfg,
                                                 logic_version="ref-1",
                                                 framework="flora-deploy test"),
                           certified=True,
                           segments=(Segment(f"{ref.uid}:0", 0.0, 1.0,
                                             caption="c" if stage == "caption" else None),))

    # helpers --------------------------------------------------------------------------------
    def _flush(self) -> None:
        a = self._active
        if a is None or not a["buf"]:
            return
        if self.b.append_parts:  # one growing file per container: an append in disguise
            a["all"].extend(a["buf"])
            path = P.write_part(a["jf"], self.kind, a["sid"], self.env.container_id, 0, a["all"])
            if path not in a["own"]:
                a["own"].append(path)
        else:
            path = P.write_part(a["jf"], self.kind, a["sid"], self.env.container_id, a["k"],
                                a["buf"])
            a["own"].append(path)
            a["k"] += 1
        a["buf"] = []

    def _marker(self, jf, sid, token) -> object:
        return A.write_attempt(jf, self.kind, sid, self.env.container_id,
                               started_utc=utc(self.env.clock()),
                               orchestrator_attempt=A.token_attempt(token),
                               token="" if self.b.marker_without_token else token,
                               call_id=self.env.call_id)

    def _preempt(self) -> None:
        a = self._active
        if a is None:
            return
        self._flush()
        if not self.b.no_preempt_mark:
            A.mark_preempted(a["jf"], self.kind, a["sid"], self.env.container_id,
                             ended_utc=utc(self.env.clock()), marker=a["marker"])
        self._active = None

    def on_exit(self) -> None:
        self._preempt()

    # the contract ---------------------------------------------------------------------------
    def run_shard(self, job_id: str, shard_id: str, config_hash: str,
                  token: str = "") -> ShardReceipt | None:
        jf = A.JobFiles(self.env.store, job_id, config_hash)
        if not self.b.ignore_superseded and A.superseded(jf, self.kind, shard_id, token):
            return None  # step 0
        src = StoreManifestSource(jf)
        shard = src.shard(shard_id)
        refs = {r.uid: r for r in src.base().episodes}
        marker = None if self.b.marker_late else self._marker(jf, shard_id, token)  # step 1
        self._active = {"jf": jf, "sid": shard_id, "marker": marker, "buf": [], "k": 0,
                        "own": [], "all": []}
        a = self._active
        receipt_written = False
        try:
            prev = P.shard_parts(jf, shard)  # step 2
            done = set() if self.b.no_skip_done else set(prev.done)
            skip: set[str] = set()
            if shard.depends_on and not self.b.run_poisoned:  # step 3
                ups = [s for s in src.load().shards if s.shard_id in shard.depends_on]
                skip = A.upstream_poisoned(jf, shard, ups) - done
                A.write_upstream_failures(jf, shard, skip)
            failed: list[str] = []
            errors: list[EpisodeError] = []
            if self.b.receipt_first:
                P.write_receipt(jf, self._receipt(shard, config_hash, skip, failed, prev, a))
                receipt_written = True
            n = 0
            for uid in shard.episodes:  # step 4
                if uid in done or uid in skip:
                    continue
                try:
                    recs = [self.stage(refs[uid], st, config_hash) for st in stages_of(self.kind)]
                except NvSemanticsFailure as exc:
                    if self.b.crash_on_nv_failure:
                        raise
                    failed.append(uid)
                    errors += [EpisodeError(uid, st, shard_id, self.env.container_id,
                                            "nv_failure", str(exc))
                               for st in stages_of(self.kind)]
                    continue
                a["buf"].extend(recs)
                if self.b.duplicate_records:
                    a["buf"].extend(recs)
                n += 1
                if n % self.flush_every == 0:
                    self._flush()
                if marker is None and not a["buf"]:  # marker_late: after the first part
                    marker = a["marker"] = self._marker(jf, shard_id, token)
            self._flush()
            if errors:
                P.write_errors(jf, self.kind, shard_id, self.env.container_id, 0, errors)
        except KeyboardInterrupt:  # step 5: SIGINT (preemption)
            self._preempt()
            if self.b.swallow_interrupt:
                return None
            raise
        except Exception as exc:
            A.mark_exception(jf, self.kind, shard_id, self.env.container_id,
                             ended_utc=utc(self.env.clock()), error=repr(exc), marker=marker)
            self._active = None
            raise
        rec = self._receipt(shard, config_hash, skip, failed, prev, a)
        if not receipt_written:
            P.write_receipt(jf, rec)  # step 6: LAST
        self._active = None
        return rec

    def _receipt(self, shard, cfg, skip, failed, prev, a) -> ShardReceipt:
        bad = set(skip) | set(failed)
        own = tuple(a["own"])
        parts = own if self.b.receipt_own_parts else tuple(sorted(set(prev.parts) | set(own)))
        return ShardReceipt(shard.shard_id, self.kind, cfg, self.env.container_id,
                            tuple(u for u in shard.episodes if u not in bad),
                            tuple(u for u in shard.episodes if u in bad), parts, None, 0.0,
                            utc(self.env.clock()))


def fake_spec(kind: ShardKind = "segcap", broken: Breakage | None = None,
              name: str | None = None) -> WorkerSpec:
    def factory(env: WorkerEnv) -> FakeConformantWorker:
        return FakeConformantWorker(env, kind, broken)

    return WorkerSpec(name or f"fake-{kind}", kind, factory)


def _b(**kw: bool) -> Breakage:
    return replace(Breakage(), **kw)


# check name -> broken variants that must fail it
BROKEN: dict[str, list[tuple[str, Breakage]]] = {
    "superseded": [("ignores-superseded", _b(ignore_superseded=True))],
    "marker_first": [("marker-late", _b(marker_late=True)),
                     ("marker-without-token", _b(marker_without_token=True))],
    "skip_done": [("no-skip-done", _b(no_skip_done=True))],
    "upstream_poison": [("runs-poisoned", _b(run_poisoned=True))],
    "nv_failure": [("crashes-on-nv-failure", _b(crash_on_nv_failure=True))],
    "records_valid": [("foreign-provenance", _b(foreign_provenance=True))],
    "parts_whole": [("appends-parts", _b(append_parts=True))],
    "preempt": [("no-preempt-mark", _b(no_preempt_mark=True)),
                ("swallows-sigint", _b(swallow_interrupt=True))],
    "receipt_last": [("receipt-own-parts", _b(receipt_own_parts=True)),
                     ("receipt-first", _b(receipt_first=True))],
    "idempotent": [("no-skip-done", _b(no_skip_done=True)),
                   ("duplicate-records", _b(duplicate_records=True))],
}
