"""Two-phase balanced episodes (WP1a) change scheduling only, never the model inputs.

* The same episodes run through ``BalancedShardRunner._episodes`` on the one-phase path
  (``phased=False``: the whole episode inside ``MicroBatcher.episode()``, as before) and the
  two-phase path (preparation under ``EpisodeGates.prep()``, inference under ``infer()``), with
  the real ``SegmentEpisodeJob`` / ``CaptionEpisodeJob`` and a fake stage and engine. Every
  request payload (prompt, every frame byte, metadata, sampling, key), every record and every
  audit (minus timings) must be identical; the batches may differ.
* A local fake-model balanced job (``run_local``, the real orchestrator and runners) with 64
  episodes completes, and its shard audit rows record the batch policy and the gate bounds.

``tests/nvlink/test_balanced_accel.py`` proves the same identity with NVIDIA's real code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.jobs.types import EpisodeRef, Shard
from flora_deploy.records import Provenance
from flora_deploy.stages import segment as S
from flora_deploy.workers_impl.caption import CaptionEpisodeJob
from flora_deploy.workers_impl.segment import (
    BalancedShardRunner,
    BalancedWorkerSettings,
    EpisodeInputs,
    SegmentEpisodeJob,
    ShardContext,
    StageInfo,
)

N = 24


def payload_digest(r: VllmRequest, sp: Sampling) -> str:
    h = hashlib.sha256(r.prompt.encode() + r.key.encode() + repr(sp).encode())
    for frames, md in r.videos:
        h.update(str((frames.shape, frames.dtype)).encode())
        h.update(np.ascontiguousarray(frames).tobytes())
        h.update(json.dumps(dict(md), sort_keys=True).encode())
    return h.hexdigest()


class RecordingEngine:
    name = "recording"

    def __init__(self):
        self.payloads: list[str] = []
        self.batches: list[int] = []

    def generate_batch(self, requests, samplings):
        self.batches.append(len(requests))
        out = []
        for r, sp in zip(requests, samplings, strict=True):
            d = payload_digest(r, sp)
            self.payloads.append(d)
            out.append(Completion(key=r.key, token_ids=tuple(int(c, 16) for c in d[:4]),
                                  finish_reason="stop", prompt_tokens=len(r.prompt)))
        return out


def frames_for(uid: str, window: int) -> np.ndarray:
    seed = int(hashlib.sha256(f"{uid}/{window}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).integers(0, 255, (4, 8, 8, 3), dtype=np.uint8)


def prepare(nv, paths, profile, **kw):
    uid = paths["main"]
    return SimpleNamespace(uid=uid, pool="human", profile=profile, duration_s=10.0,
                           instruction=kw.get("instruction", ""))


class FakeSegmentStage:
    """Up to three rounds per episode (grid + retries), each window's request built from the
    prepared episode and the previous round's answers, as NVIDIA's segmenter does."""

    nv = None

    async def segment(self, batcher, ep):
        uid, rows, calls, rounds = ep.uid, [], [], 0
        windows = 2 + int(hashlib.sha256(uid.encode()).hexdigest()[0], 16) % 3
        prev = ""
        for rnd in range(3):
            await asyncio.sleep(0.001 * (hash(uid) % 3))  # frame building, interleaved
            reqs = [VllmRequest(prompt=f"<seg {uid} r{rnd} w{w} {prev}>",
                                videos=((frames_for(uid, w + 10 * rnd), {"fps": 4, "w": w}),),
                                key=f"{uid}#{rnd}.{w}") for w in range(windows)]
            outs = await batcher.generate_many(reqs, [Sampling(max_tokens=1024)] * windows)
            rounds += 1
            prev = ",".join(str(o.token_ids) for o in outs)
            calls.append({"round": rnd, "answers": prev})
            if outs[0].token_ids[0] % 2 == 0:  # "parse ok": no retry
                break
        rows = [dict(start_s=float(i), end_s=float(i) + 1, metadata={"answers": prev})
                for i in range(windows)]
        return SimpleNamespace(rows=rows, calls=calls, apredict_rounds=rounds,
                               stats=SimpleNamespace(as_json=lambda: {"rounds": rounds}))


class FakeCaptionStage:
    nv = None

    async def caption(self, batcher, ep, rows):
        reqs = [VllmRequest(prompt=f"<cap {ep.uid} {r['start_s']}>",
                            videos=((frames_for(ep.uid, 100 + i), {"fps": 8}),),
                            key=f"{ep.uid}#{i}") for i, r in enumerate(rows)]
        outs = await batcher.generate_many(reqs, [Sampling(max_tokens=512)] * len(reqs))
        caps = [dict(start_s=r["start_s"], end_s=r["end_s"], caption=str(o.token_ids),
                     status="complete", segmentation_provenance="fine")
                for r, o in zip(rows, outs, strict=True)]
        return SimpleNamespace(rows=caps, audit=[], stats=SimpleNamespace(as_json=lambda: {}))


def refs(n=N) -> dict[str, EpisodeRef]:
    return {f"ds--ep{i:03d}": EpisodeRef(uid=f"ds--ep{i:03d}", dataset="ds",
                                         episode_id=f"ep{i:03d}", pool="human", duration_s=10.0,
                                         oracle_eligible=True, envelope_ok=True, index_ref="x")
            for i in range(n)}


def inputs(ref):
    # probe-like CPU time that varies per episode, so phases interleave differently
    import time

    time.sleep(0.0005 * (int(ref.episode_id[2:]) % 4))
    return EpisodeInputs(uid=ref.uid, dataset=ref.dataset, episode_id=ref.episode_id,
                         paths={"main": ref.uid}, profile="human", yam32=False,
                         instruction=f"task {ref.episode_id}")


def upstream(ref, _ctx):
    prov = Provenance(models={}, engine="t", preset="balanced", config_hash="c",
                      logic_version="t", framework="t")
    n = 1 + int(ref.episode_id[2:]) % 3
    return S.segment_record([dict(start_s=float(i), end_s=i + 0.5, metadata={})
                             for i in range(n)], uid=ref.uid, dataset=ref.dataset,
                            episode_id=ref.episode_id, provenance=prov)


class ListSink:
    def __init__(self):
        self.records, self.audits, self.errors = [], [], []

    def add(self, records=(), audits=(), errors=()):
        self.records += list(records)
        self.audits += list(audits)
        self.errors += list(errors)

    def flush(self):
        return []


def run(kind: str, *, phased: bool, monkeypatch):
    monkeypatch.setattr(S, "prepare_videos", prepare)
    info = StageInfo("vllm-test")
    if kind == "segment":
        job = SegmentEpisodeJob(FakeSegmentStage(), info, inputs, budget=1024)
    else:
        job = CaptionEpisodeJob(FakeCaptionStage(), info, inputs, segments=upstream,
                                budget=512)
    engine = RecordingEngine()
    settings = BalancedWorkerSettings(phased=phased, max_inflight_episodes=6,
                                      prefetch_episodes=3, max_wait_s=0.05, flush_every=10**6)
    runner = BalancedShardRunner(None, engine, job, settings)  # type: ignore[arg-type]
    rs = refs()
    shard = Shard(f"{kind}-0", kind, tuple(rs), 1.0)  # type: ignore[arg-type]
    ctx = ShardContext(None, shard, None, "c")  # type: ignore[arg-type]
    sink = ListSink()
    asyncio.run(runner._episodes(list(rs), rs, sink, ctx))  # type: ignore[arg-type]
    records = sorted(r.to_json() for r in sink.records)
    audits = sorted(json.dumps({k: v for k, v in a.items() if k != "timings_s"},
                               sort_keys=True, default=str) for a in sink.audits)
    return SimpleNamespace(payloads=sorted(engine.payloads), records=records, audits=audits,
                           errors=sink.errors, batches=engine.batches,
                           stats=runner.last_stats, gates=runner.last_gates)


@pytest.mark.parametrize("kind", ["segment", "caption"])
def test_two_phase_episodes_send_identical_model_inputs(kind, monkeypatch):
    old = run(kind, phased=False, monkeypatch=monkeypatch)
    new = run(kind, phased=True, monkeypatch=monkeypatch)
    assert not old.errors and not new.errors
    assert len(old.payloads) > N  # retries happened (segment) / several calls per episode
    assert new.payloads == old.payloads  # every request: prompt, frame bytes, metadata, sampling
    assert new.records == old.records
    assert new.audits == old.audits
    assert len(old.records) == N
    assert new.gates["peak_preparing"] <= 3 and new.gates["peak_inferring"] <= 6
    assert sum(new.batches) == sum(old.batches) == len(old.payloads)


def test_shard_audit_rows_carry_the_policy_digest(monkeypatch):
    from flora_deploy.engines.batch_policy import policy_for

    new = run("segment", phased=True, monkeypatch=monkeypatch)
    want = policy_for("segment", infer_episodes=6, prefetch_episodes=3,
                      max_wait_s=0.05).as_json()["digest"]
    assert {json.loads(a)["batch_policy"] for a in new.audits} == {want}


# --- a 64-episode local fake-model balanced job ---------------------------------------------


TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"


def _prepare_first_episodes_together(monkeypatch, n: int = 2) -> None:
    """Each shard's first ``n`` episodes finish preparing together, so they enter inference in
    one loop step and share the first round. Without it, whether two episodes ever share a
    round is timing: a segment episode makes one call, a round starts as soon as every
    inferring episode is blocked, and on a slow runner each episode's CPU preparation outlasts
    the previous one's inference (CI ``order (3.11)``: every segment round had one request)."""
    from flora_deploy.local import balanced as LB

    orig = LB._FakeJob._prepare

    async def _prepare(self, ref, ctx):
        out = await orig(self, ref, ctx)
        b = ctx.cache.setdefault("test_first_prepared", {"n": 0, "ev": asyncio.Event()})
        b["n"] += 1
        if b["n"] >= n:
            b["ev"].set()
        try:  # a shard of fewer episodes (none here) must not hang
            await asyncio.wait_for(b["ev"].wait(), 30)
        except TimeoutError:
            pass
        return out

    monkeypatch.setattr(LB._FakeJob, "_prepare", _prepare)


def test_local_fake_balanced_job_with_64_episodes(tmp_path, monkeypatch):
    from flora_deploy.export.results import iter_records
    from flora_deploy.jobs.stores import LocalStore
    from flora_deploy.local import run_local
    from flora_deploy.local import sample_dataset as make_sample
    from tests.unit.test_balanced_deps import balanced_yaml

    _prepare_first_episodes_together(monkeypatch)
    d = tmp_path / "demo"
    d.mkdir()
    shutil.copy(TEMPLATE, d / "flora.yaml")
    text = balanced_yaml((d / "flora.yaml").read_text()).replace(
        "shard_target_gpu_minutes: 0.1", "shard_target_gpu_minutes: 30")
    (d / "flora.yaml").write_text(text)
    make_sample(d / "videos", episodes=64, min_s=0.5, max_s=1.0)
    res = run_local(d / "flora.yaml", store=tmp_path / "store")
    assert res.state == "done", res.as_dict()
    st = LocalStore(res.store_root)
    for stage in ("segment", "caption"):
        recs = list(iter_records(st, res.job_id, stage))
        assert len({r.uid for r in recs}) == 64, stage
        c = res.summary["stages"][stage]
        assert c["records"] == c["expected"] == 64
    rows = [json.loads(line) for p in Path(res.store_root).rglob("*.jsonl")
            for line in p.read_text().splitlines() if '"shard_audit"' in line]
    rows = [r for r in rows if r.get("shard_audit")]
    assert {r["stage"] for r in rows} == {"segment", "caption"}
    assert sum(r["episodes_run"] for r in rows if r["stage"] == "segment") >= 64
    for r in rows:
        assert r["phased"] is True
        pol = r["batch_policy"]
        assert pol["schema"] == "flora-batch-policy/v1" and pol["kind"] == r["stage"]
        if r["episodes_run"]:
            assert r["gates"]["peak_preparing"] <= pol["prefetch_episodes"]
            assert r["gates"]["peak_inferring"] <= pol["infer_episodes"]
            assert r["batcher"]["requests"] >= r["episodes_run"]
            assert r["batcher"]["max_batch"] > 1  # episodes shared rounds
