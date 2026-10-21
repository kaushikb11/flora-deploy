from __future__ import annotations

import random
from fractions import Fraction

import pytest

from flora_deploy.jobs.manifest import (
    DEFAULT_RATES,
    NVIDIA_TERMINAL_GROUP,
    build_manifest,
    load_manifest,
    manifest_from_json,
    manifest_to_json,
    save_manifest,
    source_group,
)
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.types import episode_uid, safe_key
from flora_deploy.media.types import EnvelopeVerdict, EpisodeIndex, SourceRef, StreamIndex
from tests.fakes.modal import FakeStore


def make_index(dataset: str, ep: str, uri: str, duration_s: float, *, three_view=False,
               envelope_ok=True, keyframe=True) -> EpisodeIndex:
    def stream(u: str) -> StreamIndex:
        return StreamIndex(source=SourceRef(u, 0.0, duration_s), rate=Fraction(30), pts_us=(),
                           duration_us=int(duration_s * 1_000_000), size=(640, 480), cfr=True,
                           codec="h264", pix_fmt="yuv420p", starts_on_keyframe=keyframe)

    roles = ("main", "left_wrist", "right_wrist") if three_view else ("main",)
    streams = {r: stream(uri if r == "main" else uri.replace(".mp4", f".{r}.mp4")) for r in roles}
    verdict = EnvelopeVerdict(ok=envelope_ok, reasons=() if envelope_ok else ("duration_gt_120s",))
    return EpisodeIndex(episode_id=ep, dataset=dataset, streams=streams, envelope=verdict)


def make_settings(preset="exact", *, target_min=1.0, envelope="warn", rewards=("reward.terminal",),
                  job_id="job-1", cfg="cfg0") -> RunSettings:
    pools = ({"segcap": PoolSettings("H200")} if preset == "exact"
             else {"segment": PoolSettings("L40S"), "caption": PoolSettings("H100!")})
    pools.update({r: PoolSettings("H100!") for r in rewards})
    return RunSettings(job_id=job_id, preset=preset, config_hash=cfg, budget_usd=10.0,
                       max_gpus=8, pools=pools, shard_target_gpu_minutes=target_min,
                       envelope=envelope)


def random_indexes(rng: random.Random, n: int, *, files: int | None = None) -> list[EpisodeIndex]:
    files = files or max(1, n // 3)
    out = []
    for i in range(n):
        f = rng.randrange(files)
        out.append(make_index(f"org/ds{f % 2}", f"episode_{i:06d}", f"/data/file-{f:03d}.mp4",
                              rng.uniform(2.0, 40.0), envelope_ok=rng.random() > 0.15))
    rng.shuffle(out)
    return out


def test_exact_preset_kinds_and_ids():
    idx = [make_index("lerobot/a", f"episode_{i:06d}", "/d/f0.mp4", 10.0) for i in range(4)]
    m = build_manifest(idx, make_settings("exact"), created_utc="t")
    kinds = {s.kind for s in m.shards}
    assert kinds == {"segcap", "reward.terminal"}
    assert all(safe_key(s.shard_id) and s.shard_id.startswith(f"{s.kind}-") for s in m.shards)
    assert {e.uid for e in m.episodes} == {episode_uid("lerobot/a", f"episode_{i:06d}")
                                           for i in range(4)}
    assert m.episodes[0].index_ref.startswith("flora/job-1/index/")


def test_balanced_caption_depends_on_segment_shards():
    rng = random.Random(1)
    m = build_manifest(random_indexes(rng, 30), make_settings("balanced", rewards=()),
                       created_utc="t", rates={"segment": 1.0, "caption": 2.5})
    seg = {u: s.shard_id for s in m.shards if s.kind == "segment" for u in s.episodes}
    caps = [s for s in m.shards if s.kind == "caption"]
    assert caps and {s.kind for s in m.shards} == {"segment", "caption"}
    for c in caps:
        assert set(c.depends_on) == {seg[u] for u in c.episodes}
    assert all(not s.depends_on for s in m.shards if s.kind == "segment")


def test_envelope_policy():
    idx = [make_index("d", "e1", "/f.mp4", 5.0), make_index("d", "e2", "/f.mp4", 5.0,
                                                            envelope_ok=False)]
    enforce = build_manifest(idx, make_settings(envelope="enforce"), created_utc="t")
    warn = build_manifest(idx, make_settings(envelope="warn"), created_utc="t")
    assert [e.episode_id for e in enforce.episodes] == ["e1"]
    assert [(e.episode_id, e.envelope_ok) for e in warn.episodes] == [("e1", True), ("e2", False)]
    assert all(set(s.episodes) <= {e.uid for e in enforce.episodes} for s in enforce.shards)


def test_duplicate_uid_rejected():
    idx = [make_index("d", "e1", "/f.mp4", 5.0)] * 2
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest(idx, make_settings(), created_utc="t")


@pytest.mark.parametrize("seed", range(40))
def test_packing_respects_target_and_groups_by_file(seed):
    rng = random.Random(seed)
    idx = random_indexes(rng, rng.randrange(1, 60), files=rng.randrange(1, 12))
    settings = make_settings(rng.choice(["exact", "balanced"]), target_min=rng.uniform(0.2, 3.0),
                             envelope=rng.choice(["warn", "enforce"]))
    m = build_manifest(idx, settings, created_utc="t")
    by_uid = {e.uid: e for e in m.episodes}
    group = {episode_uid(i.dataset, i.episode_id): source_group(i) for i in idx}
    target = settings.shard_target_gpu_minutes * 60
    for kind in {s.kind for s in m.shards}:
        shards = [s for s in m.shards if s.kind == kind]
        # every manifest episode exactly once per kind
        flat = [u for s in shards for u in s.episodes]
        assert sorted(flat) == sorted(by_uid)
        rate = DEFAULT_RATES[kind]
        if kind == "reward.terminal":  # review 5 M-B: whole NVIDIA groups of the job order
            job = [e.uid for e in m.episodes]
            assert flat == job
            pos = 0
            for a in shards:
                assert pos % NVIDIA_TERMINAL_GROUP == 0  # every boundary is a group boundary
                pos += len(a.episodes)
                est = sum(by_uid[u].duration_s * rate for u in a.episodes)
                assert a.est_gpu_seconds == pytest.approx(est)
                assert est <= target + 1e-6 or len(a.episodes) <= NVIDIA_TERMINAL_GROUP
            for a, b in zip(shards, shards[1:], strict=False):
                nxt = b.episodes[:NVIDIA_TERMINAL_GROUP]
                g_est = sum(by_uid[u].duration_s * rate for u in nxt)
                assert a.est_gpu_seconds + g_est > target - 1e-6
            continue
        for s in shards:
            est = sum(by_uid[u].duration_s * rate for u in s.episodes)
            assert s.est_gpu_seconds == pytest.approx(est)
            assert est <= target + 1e-6 or len(s.episodes) == 1
        # each source file is read by one shard, unless it alone exceeds the target, in which
        # case it occupies consecutive shards
        owners: dict[tuple, list[int]] = {}
        for i, s in enumerate(shards):
            for u in s.episodes:
                owners.setdefault(group[u], []).append(i)
        for g, idxs in owners.items():
            span = sorted(set(idxs))
            assert span == list(range(span[0], span[-1] + 1))
            if len(span) > 1:
                g_est = sum(e.duration_s * rate for u, e in by_uid.items() if group[u] == g)
                assert g_est > target
        # next-fit: a shard was closed only because the next whole group did not fit (a group
        # that is split starts on a fresh shard)
        for a, b in zip(shards, shards[1:], strict=False):
            g = group[b.episodes[0]]
            g_est = sum(e.duration_s * rate for u, e in by_uid.items() if group[u] == g)
            assert a.est_gpu_seconds + g_est > target - 1e-6
    # deterministic regardless of input order
    rng.shuffle(idx)
    assert manifest_to_json(build_manifest(idx, settings, created_utc="t")) == manifest_to_json(m)


def test_three_view_groups_by_all_files():
    a = make_index("d", "e1", "/f0.mp4", 5.0, three_view=True)
    assert source_group(a) == ("/f0.left_wrist.mp4", "/f0.mp4", "/f0.right_wrist.mp4")


def test_manifest_json_roundtrip_and_store():
    rng = random.Random(3)
    m = build_manifest(random_indexes(rng, 20), make_settings("balanced"), created_utc="2026")
    assert manifest_from_json(manifest_to_json(m)) == m
    store = FakeStore()
    save_manifest(store, m)
    assert load_manifest(store, m.job_id) == m


@pytest.mark.parametrize("target_min", [117 / 60, 1.5, 0.01, 100.0])
def test_terminal_shards_hold_whole_nvidia_groups(target_min):
    """Review 5 M-B: 20 episodes of 9 terminal GPU-s each. Next-fit by duration alone cuts
    13 + 7 at a 117 s target; terminal shards are cut on NVIDIA's group boundaries (8) of the
    job's episode list instead, whatever the target, so a group never spans shards. Other
    kinds keep the duration packing."""
    idx = [make_index("d", f"e{i:02d}", f"/f{i:02d}.mp4", 30.0) for i in range(20)]
    m = build_manifest(idx, make_settings(target_min=target_min), created_utc="t",
                       rates={"reward.terminal": 0.3, "segcap": 0.3})
    job = [e.uid for e in m.episodes]
    term = [s.episodes for s in m.shards if s.kind == "reward.terminal"]
    assert [u for s in term for u in s] == job
    cuts, pos = set(), 0
    for s in term[:-1]:
        pos += len(s)
        cuts.add(pos)
    assert all(c % NVIDIA_TERMINAL_GROUP == 0 for c in cuts)
    want = {117 / 60: {8}, 1.5: {8, 16}, 0.01: {8, 16}, 100.0: set()}[target_min]
    assert cuts == want
    if target_min == 117 / 60:  # the duration packing (segcap) is the reviewer's 13 + 7
        seg = [len(s.episodes) for s in m.shards if s.kind == "segcap"]
        assert seg == [13, 7]
