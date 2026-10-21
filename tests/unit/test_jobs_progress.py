"""Property tests: random crash/retry/duplicate/resume interleavings over a lagging FakeStore.

A simple fake worker follows the worker contract (marker first, skip what visible parts already
hold, whole-file parts, receipt last) and a stateless fake orchestrator drives it from
``progress.derive`` alone. After the job is complete we compose and check that no
``(uid, stage)`` is lost or duplicated, that quarantine fired within the documented bound, and
that bisection isolated the single poison episode.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

import pytest

from flora_deploy.jobs.attempts import JobFiles, quarantine, write_attempt
from flora_deploy.jobs.manifest import build_manifest, load_manifest, save_manifest, stages_of
from flora_deploy.jobs.progress import (
    compose_input,
    derive,
    shard_parts,
    write_part,
    write_receipt,
)
from flora_deploy.jobs.types import Shard, ShardReceipt
from flora_deploy.records import Provenance, Segment, StageRecord, record_key
from tests.fakes.modal import FakeStore
from tests.unit.test_jobs_manifest import make_settings, random_indexes

MAX_CONCURRENT = 2  # containers of one shard started in one round (duplicate spawns)


def make_record(uid: str, stage: str, cfg: str, note: str = "") -> StageRecord:
    return StageRecord(
        uid=uid, dataset="d", episode_id=uid.split("--")[-1], stage=stage,  # type: ignore[arg-type]
        provenance=Provenance(models={}, engine="fake", preset="exact", config_hash=cfg,
                              logic_version="1", framework="test"),
        certified=True, segments=(Segment(event_id=f"{uid}:{stage}{note}", start_s=0.0,
                                          end_s=1.0),))


class Sim:
    def __init__(self, rng: random.Random, jf: JobFiles, limit: int, poison: str | None):
        self.rng, self.jf, self.limit, self.poison = rng, jf, limit, poison
        self.ids = itertools.count(1)
        self.starts: Counter[str] = Counter()  # all container starts
        self.marked: Counter[str] = Counter()  # starts that wrote a marker
        self.faults: Counter[str] = Counter()

    def fault_for(self, shard: Shard) -> str | None:
        # Random faults only on the first K-2 starts, so a healthy shard always finishes before
        # it can reach the limit; only the poison episode drives quarantine.
        if self.starts[shard.shard_id] > self.limit - 2:
            return None
        return self.rng.choice([None, None, "before_marker", "after_parts", "before_receipt"])

    def run_container(self, shard: Shard) -> None:
        jf, rng = self.jf, self.rng
        cid = f"ta-{next(self.ids):05d}"
        self.starts[shard.shard_id] += 1
        fault = self.fault_for(shard)
        self.faults[str(fault)] += 1
        if fault == "before_marker":
            return  # died before writing anything (an invisible start)
        write_attempt(jf, shard.kind, shard.shard_id, cid, started_utc="t",
                      orchestrator_attempt=self.starts[shard.shard_id])
        self.marked[shard.shard_id] += 1
        prior = shard_parts(jf, shard)
        done = set(prior.done)
        todo = [u for u in shard.episodes if u not in done]
        own: list[str] = []
        buf: list[StageRecord] = []

        def flush() -> None:
            nonlocal buf
            if buf:
                own.append(write_part(jf, shard.kind, shard.shard_id, cid, len(own), buf))
                buf = []

        crash_at = rng.randrange(len(todo) + 1) if fault == "after_parts" else None
        for i, u in enumerate(todo):
            if i == crash_at:
                flush()
                return  # crash after some parts were copied
            if u == self.poison:
                return  # OOM: the buffered tail is lost
            buf += [make_record(u, st, jf.config_hash) for st in stages_of(shard.kind)]
            if rng.random() < 0.5:
                flush()
        flush()
        if fault == "before_receipt":
            return
        write_receipt(jf, ShardReceipt(
            shard_id=shard.shard_id, kind=shard.kind, config_hash=jf.config_hash,
            container_id=cid, episodes_done=shard.episodes, episodes_failed=(),
            parts=prior.parts + tuple(own), gpu=None, wall_seconds=1.0,
            finished_utc=f"2026-10-21T{next(self.ids):08d}"))


def simulate(seed: int):
    rng = random.Random(seed)
    lag = rng.choice([0, 1, 2, 3])
    limit = rng.choice([3, 4])
    preset = rng.choice(["exact", "balanced"])
    rewards = rng.choice([(), ("reward.terminal",)])
    settings = make_settings(preset, target_min=rng.uniform(0.3, 2.0), rewards=rewards,
                             cfg=f"cfg{seed % 3}")
    m = build_manifest(random_indexes(rng, rng.randrange(2, 25)), settings, created_utc="t")
    store = FakeStore(lag=lag)
    save_manifest(store, m)
    jf = JobFiles(store, m.job_id, m.config_hash)
    poison = rng.choice([None, rng.choice(m.episodes).uid])
    sim = Sim(rng, jf, limit, poison)
    bound = limit - 1 + MAX_CONCURRENT * max(lag, 1)
    quarantined_with: dict[str, int] = {}

    for rnd in range(2000):
        if rng.random() < 0.1:  # orchestrator restart: nothing but files survives
            jf = JobFiles(store, m.job_id, m.config_hash)
            m = load_manifest(store, m.job_id)
            sim.jf = jf
        prog = derive(jf, m, limit=limit)
        if prog.complete:
            break
        for shard in prog.due_for_quarantine():
            # quarantine fires within the documented bound even with listing lag
            assert sim.marked[shard.shard_id] <= bound, (shard.shard_id, sim.marked, bound)
            quarantined_with[shard.shard_id] = sim.marked[shard.shard_id]
            sp = shard_parts(jf, shard)
            quarantine(jf, shard, created_utc=f"2026-10-21T{rnd:06d}", done=sp.done,
                       done_parts=sp.parts, starts=prog.shards[shard.shard_id].starts,
                       durations={e.uid: e.duration_s for e in m.episodes})
        for shard in prog.runnable():
            if rng.random() < 0.15:
                continue  # spawn delayed
            for _ in range(MAX_CONCURRENT if rng.random() < 0.2 else 1):
                sim.run_container(shard)
        store.tick()
    else:
        pytest.fail("job never completed")
    return m, jf, prog, sim, poison, quarantined_with



@pytest.mark.parametrize("seed", range(200))
def test_no_lost_or_duplicated_records(seed):
    m, jf, prog, sim, poison, qw = simulate(seed)
    ci = compose_input(jf, prog)
    assert not ci.missing
    assert not ci.conflicts and ci.foreign == 0
    composed = list(ci.records.values())
    keys = [record_key(r) for r in composed]
    assert len(keys) == len(set(keys))
    kinds = {s.kind for s in m.shards}
    want = {(e.uid, st) for e in m.episodes for k in kinds for st in stages_of(k)}
    poisoned_keys = {(poison, st) for k in kinds for st in stages_of(k)} if poison else set()
    assert set(keys) == want - poisoned_keys  # every healthy (uid, stage), exactly once
    assert ci.excused == poisoned_keys
    # bisection isolated exactly the poison episode, in every kind, in single-episode shards
    assert {u for u, _ in prog.poisoned()} == ({poison} if poison else set())
    holders = [s for s in prog.shards.values()
               if any(r.detail.get("poisoned") for r in s.quarantines)]
    assert bool(holders) == bool(poison)
    assert all(h.shard.episodes == (poison,) for h in holders)
    assert prog.resume_set() == {}


def test_simulations_exercise_the_hard_paths():
    """Guard against a vacuous property test: the interleavings must hit every hard path."""
    stats: Counter[str] = Counter()
    for seed in range(60):
        _m, jf, prog, sim, _poison, qw = simulate(seed)
        ci = compose_input(jf, prog)
        stats["duplicates"] += ci.duplicates > 0
        stats["bisected"] += any(s.shard.bisected_from for s in prog.shards.values())
        stats["done_parts"] += any(r.detail["done_parts"] for s in prog.shards.values()
                                   for r in s.quarantines)
        stats["late_quarantine"] += any(v > sim.limit for v in qw.values())
        stats.update(f"fault:{k}" for k, n in sim.faults.items() if n)
    assert min(stats[k] for k in ("duplicates", "bisected", "done_parts", "late_quarantine",
                                  "fault:before_marker", "fault:after_parts",
                                  "fault:before_receipt")) >= 5, stats


def test_resume_set_and_compose_tie_breaking():
    settings = make_settings("exact", rewards=(), target_min=100)
    m = build_manifest(random_indexes(random.Random(0), 3, files=1), settings, created_utc="t")
    (shard,) = m.shards
    jf = JobFiles(FakeStore(), m.job_id, m.config_hash)
    u0, u1, u2 = shard.episodes
    prog = derive(jf, m, limit=3)
    assert prog.resume_set() == {"segcap": {u0, u1, u2}} and prog.runnable() == [shard]
    # an older container wrote u0 (plus a record of another config), then a newer one finished
    p_old = write_part(jf, "segcap", shard.shard_id, "ta-1", 0,
                       [make_record(u0, "segment", jf.config_hash, "old"),
                        make_record(u0, "caption", "other-cfg")])
    p_new = write_part(jf, "segcap", shard.shard_id, "ta-2", 0,
                       [make_record(u, st, jf.config_hash) for u in shard.episodes
                        for st in ("segment", "caption")])
    write_receipt(jf, ShardReceipt(shard.shard_id, "segcap", jf.config_hash, "ta-2",
                                   shard.episodes, (), (p_old, p_new), None, 1.0, "t2"))
    prog = derive(jf, m, limit=3)
    assert prog.complete and prog.resume_set() == {}
    ci = compose_input(jf, prog)
    assert ci.foreign == 1 and ci.duplicates == 1 and ci.conflicts == [(u0, "segment")]
    assert ci.records[(u0, "segment")].segments[0].event_id.endswith("segment")  # latest wins
    assert not ci.missing
