"""Replay of the scale-lite-102223538c21 budget stop (2026-10-22, 19:34:12 UTC).

A healthy balanced job (L40S segment + H100 caption) was stopped by the projection rule at
$1.86 real spend: the warm compile-cache seed made each segment container's startup ~560 s
(the table says 295 s), and the old calibration subtracted the *table* startup from the first
finished call's wall time, charging ~265 s of startup as work on a 19 s shard (16.18x). It also
charged two new H100 caption starts although a pre-warmed caption container was already up.

The numbers below are the job's own: its settings, shard layout, episode durations, call
ledgers, attempt markers and pre-warm interval (only the episode ids are renamed). The old rule
reproduces the recorded stop ($6.6869 against a $6.08 limit, calibration 16.1787); the new one
projects ~$4.2.
"""

from __future__ import annotations

import pytest

from flora_deploy.jobs.attempts import JobFiles, write_attempt
from flora_deploy.jobs.budget import (
    BudgetState,
    CallRecord,
    ProjectionBudgetGate,
    RateTableEstimator,
    parse_utc,
    pinned_rates,
)
from flora_deploy.jobs.pools import (
    ORCHESTRATOR_CORES,
    ORCHESTRATOR_MEMORY_GIB,
    ORCHESTRATOR_PREEMPTIBLE,
)
from flora_deploy.jobs.settings import PoolSettings, RunSettings
from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import Shard

JOB = "scale-lite-102223538c21"
CFG = "76041b3a82a08788199d4f1ab18482afed361b5827df7f468068cfa63489f978"
STAGES = {"caption": "6134f3a01f3a4afee7c236eec154af7f08afdbcd5424ffaf6c7d274d7a2fa7e3",
          "segment": "2ce7be563074c77c27efbe97caccea8957b1b06d5e0ed6028248c48641e65894"}
STOP = parse_utc("2026-10-22T19:34:12.466201Z")  # the budget_stop record
PREWARM = (1792697026.1028736, None)  # caption pre-warm container, started 19:23:46

# video seconds of episodes e00..e31 (the manifest's EpisodeRef.duration_s)
DURATIONS = [39.966667, 10.433333, 8.433333, 54.433333, 7.033333, 40.3, 11.133333, 8.666667,
             6.166667, 53.333333, 11.633333, 37.8, 7.066667, 40.266667, 41.033333, 45.666667,
             50.466667, 7.166667, 11.666667, 45.366667, 8.566667, 8.866667, 49.366667, 31.0,
             9.9, 54.8, 9.2, 7.233333, 11.133333, 9.6, 11.166667, 52.8]
SHARDS = [  # (shard_id, kind, episodes, est_gpu_seconds)
    ("segment-7d63370cf52664c4", "segment", range(0, 10), 119.949999),
    ("segment-d3af1ff55c06e38f", "segment", range(10, 13), 28.25),
    ("segment-52664e1e93ccf075", "segment", range(13, 19), 98.133334),
    ("segment-349b7e758cc93641", "segment", range(19, 22), 31.400001),
    ("segment-917aa5a2c2a5df0c", "segment", range(22, 31), 96.7),
    ("segment-0f17acba1d91d0ff", "segment", range(31, 32), 26.4),
    ("caption-5c59a9aaba9c4c96", "caption", range(0, 4), 113.266666),
    ("caption-4b6f54421a9df6f7", "caption", range(4, 9), 73.3),
    ("caption-aba1a503e7366fb8", "caption", range(9, 13), 109.833333),
    ("caption-86394941fe165d54", "caption", range(13, 15), 81.3),
    ("caption-387af961aa40d8c3", "caption", range(15, 19), 114.966668),
    ("caption-892c473804502b38", "caption", range(19, 22), 62.800001),
    ("caption-1c5356cdcedd0fef", "caption", range(22, 25), 90.266667),
    ("caption-12b029104a2fb889", "caption", range(25, 31), 103.133333),
    ("caption-22b50f3c5706a8b9", "caption", range(31, 32), 52.8),
]
# (call id, shard id, spawned, ended, container, attempt marker started_utc)
CALLS = [
    ("fc-01M5JH30E016X0MVBK971Q9FAS", "segment-7d63370cf52664c4",
     "2026-10-22T19:23:28.248253Z", "2026-10-22T19:34:11.698724Z",
     "ta-01M5JH30NG0GYKTBM7K9VW437R", "2026-10-22T19:32:46.944490Z"),
    ("fc-01M5JH31T8QGJN8MT7MCSMAGT2", "segment-d3af1ff55c06e38f",
     "2026-10-22T19:23:29.968078Z", "2026-10-22T19:33:37.988850Z",
     "ta-01M5JH324N4MAD36FD85S1W5NR", "2026-10-22T19:32:49.668928Z"),
    ("fc-01M5JHNNE6372XBGCEZ9ED934F", "segment-52664e1e93ccf075",
     "2026-10-22T19:33:39.891182Z", "2026-10-22T19:34:28.489218Z",
     "ta-01M5JH324N4MAD36FD85S1W5NR", "2026-10-22T19:33:40.987332Z"),
]


def _uid(i: int) -> str:
    return f"e{i:02d}"


def run_settings(budget: float = 6.4) -> RunSettings:
    return RunSettings(
        job_id=JOB, preset="balanced", config_hash=CFG, budget_usd=budget, max_gpus=4,
        pools={"segment": PoolSettings(gpu="L40S", max_containers=2, memory_gib=48.0),
               "caption": PoolSettings(gpu="H100!", max_containers=2, memory_gib=96.0)},
        stage_hashes=dict(STAGES), scaledown_s=300, shard_target_gpu_minutes=2.0)


def shards() -> list[Shard]:
    return [Shard(sid, kind, tuple(_uid(i) for i in eps), est)  # type: ignore[arg-type]
            for sid, kind, eps, est in SHARDS]


# shard receipts visible at the stop: d3af, and 7d63 (written before its ledger said success)
DONE_AT_STOP = ("segment-d3af1ff55c06e38f", "segment-7d63370cf52664c4")


def build_gate(root, *, with_markers: bool = True, budget: float = 6.4,
               done: tuple[str, ...] = DONE_AT_STOP, **kw) -> ProjectionBudgetGate:
    """The production gate (``jobs.deps.default_budget_gate``) over this job's store."""
    store = LocalStore(root)
    settings = run_settings(budget)
    paths = JobPaths(JOB, config_hash=CFG, stage_hashes=STAGES)
    for sid in done:
        store.put_bytes(paths.receipt(CFG, sid.split("-")[0], sid), b"{}")
    if with_markers:
        jf = JobFiles(store, JOB, CFG, stage_hashes=STAGES)
        for cid, sid, _t0, _t1, container, started in CALLS:
            write_attempt(jf, "segment", sid, container, started_utc=started,
                          orchestrator_attempt=1, call_id=cid)
    durations = {_uid(i): d for i, d in enumerate(DURATIONS)}
    return ProjectionBudgetGate(
        settings=settings, store=store, shards=shards, rates=pinned_rates(),
        scaledown_s=300.0, orchestrator_cores=ORCHESTRATOR_CORES,
        orchestrator_memory_gib=ORCHESTRATOR_MEMORY_GIB,
        orchestrator_preemptible=ORCHESTRATOR_PREEMPTIBLE,
        estimator=RateTableEstimator.default(
            "balanced", durations=durations, packing_rates={"caption": 1.0, "segment": 0.5}),
        stage_hashes=dict(STAGES), **{"clock": lambda: STOP, **kw})


def state_at_stop(*, live_7d63: bool = True, now: float = STOP,
                  prewarm: tuple[float, float | None] = PREWARM,
                  drains: dict | None = None) -> BudgetState:
    """The orchestrator's call history at ``now``: a call is live until it ended. At the stop,
    d3af is done, 52664 live, and 7d63 (which finished 0.8 s earlier) still live in that tick's
    ledgers, as the stop record shows (``live_7d63``)."""
    calls = []
    for cid, sid, t0, t1, _c, _s in CALLS:
        if parse_utc(t0) > now:
            continue  # not spawned yet
        end: float | None = parse_utc(t1)
        if end > now or (live_7d63 and sid.startswith("segment-7d63")):
            end = None
        calls.append(CallRecord(cid, sid, "segment", parse_utc(t0), end,
                                None if end is None else "success"))
    return BudgetState(now, tuple(calls), prewarms={"caption": prewarm}, drains=drains or {})


SEGMENTS_DONE = ("segment-d3af1ff55c06e38f", "segment-7d63370cf52664c4",
                 "segment-52664e1e93ccf075")
DRAINED_AT = 1792697670.1032119  # budget_stopped: both worker classes drained
RESUME = parse_utc("2026-10-23T12:00:00Z")


def old_rule(gate: ProjectionBudgetGate) -> ProjectionBudgetGate:
    """The pre-fix rule on the same gate: no attempt markers (table startup subtracted from a
    cold call's wall time), a plain ratio (no minimum work) and no credit for up containers."""
    gate.calibration_min_work_s = 0.0
    gate._idle_containers = lambda now, observed: ({}, 0.0)  # type: ignore[method-assign]
    return gate


def test_old_rule_reproduces_the_recorded_stop(tmp_path):
    g = old_rule(build_gate(tmp_path, with_markers=False))
    g.observe(state_at_stop())
    d = g.admit(shards()[3], {"segment": 2})
    assert not d.allow
    # the budget_stop record: calibration 16.1787, $6.6869 > $6.08
    assert d.detail["calibration"]["segment"] == pytest.approx(16.1787, abs=1e-3)
    assert d.detail["spend_usd"] == pytest.approx(1.8572, abs=1e-4)
    assert d.detail["in_flight_usd"] == pytest.approx(1.753, abs=1e-3)
    assert d.detail["queued_usd"] == pytest.approx(2.0219, abs=1e-3)
    assert d.detail["startup_usd"] == pytest.approx(1.0548, abs=1e-3)
    assert d.detail["projected_usd"] == pytest.approx(6.6869, abs=1e-3)
    assert d.detail["limit_usd"] == pytest.approx(6.08)


def test_new_rule_admits_the_healthy_job(tmp_path):
    g = build_gate(tmp_path)
    g.observe(state_at_stop())
    # observed startups: both L40S containers took ~559 s to their first marker (table: 295 s)
    obs = g.observed_startups()
    assert sorted(round(x, 1) for x in obs["segment"]) == [558.7, 559.7]
    assert "caption" not in obs  # no caption call yet: the table's 372.6 s
    cal = g.effective_calibration()["segment"]
    # d3af: 48.3 s of work on a 19.3 s estimate (2.5x), padded to 120 s of work at 1.0x
    assert cal == pytest.approx((48.32 + 71.68) / (19.32 + 71.68), abs=0.01)
    d = g.admit(shards()[3], {"segment": 2})
    assert d.allow, d.reason
    p = d.detail
    assert p["spend_usd"] == pytest.approx(1.8572, abs=1e-4)
    # one new H100 start (the ready pre-warm container takes the other slot): 372.6 s
    assert p["startup_usd"] == pytest.approx(372.6 * pinned_rates().per_second(
        "H100", 1, 8.0, 96.0), abs=1e-3)
    assert p["startup_usd"] == pytest.approx(0.5274, abs=1e-3)
    assert p["projected_usd"] == pytest.approx(3.50, abs=0.02)
    # with the old 2 caption starts it would be ~$4.03 (the diag's "~$4.2" did that, at 1.0x)
    assert p["projected_usd"] + 0.5274 == pytest.approx(4.03, abs=0.03)


def test_new_rule_after_7d63_ended_still_admits(tmp_path):
    g = build_gate(tmp_path)
    g.observe(state_at_stop(live_7d63=False))
    d = g.admit(shards()[3], {"segment": 1})
    assert d.allow
    # 7d63's container is idle inside its scale-down window: no new L40S start charged
    idle, _ = g._idle_containers(STOP, g.observed_startups())
    assert idle == {"segment": 1, "caption": 1}
    assert d.detail["projected_usd"] == pytest.approx(3.75, abs=0.02)
    assert d.detail["calibration"]["segment"] == pytest.approx(1.31, abs=0.01)


def test_resume_projection(tmp_path):
    """``flora resume scale-lite-102223538c21``: the three done segment shards are kept; three
    segment and nine caption shards are left. New rule: calibration from the three finished
    calls by estimate (~1.07x; the old mean of per-call ratios was ~7x), observed L40S startup
    ~559 s for the two new segment containers, the table's 372.6 s for two H100 starts."""
    pools = {"caption": ((DRAINED_AT, None),), "segment": ((DRAINED_AT, None),)}
    state = state_at_stop(live_7d63=False, now=RESUME,
                          prewarm=(PREWARM[0], 1792697669.2534943), drains=pools)
    g = build_gate(tmp_path, done=SEGMENTS_DONE, clock=lambda: RESUME)
    g.observe(state)
    cal = g.effective_calibration()["segment"]
    assert cal == pytest.approx(180.6 / 168.6, abs=0.01)
    p = g.projection({})
    old = old_rule(build_gate(tmp_path / "old", with_markers=False, done=SEGMENTS_DONE,
                              clock=lambda: RESUME))
    old.observe(state)
    # table startup subtracted, weighted: (313 + 348.5 + 48.6) / 168.6 (the pre-fix plain mean
    # of the three ratios, 16.18 / 4.25 / 0.72, was ~7.05x)
    assert old.effective_calibration()["segment"] == pytest.approx(4.21, abs=0.02)
    po = old.projection({})
    assert p.spend_usd == pytest.approx(po.spend_usd)
    l40s = pinned_rates().per_second("L40S", 1, 8.0, 48.0)
    h100 = pinned_rates().per_second("H100", 1, 8.0, 96.0)
    # the observed ~559 s L40S startup (not the table's 295 s) for two new segment containers
    assert p.startup_usd == pytest.approx(2 * 559.2 * l40s + 2 * 372.6 * h100, abs=0.01)
    assert po.startup_usd == pytest.approx(2 * 295.0 * l40s + 2 * 372.6 * h100, abs=0.01)
    assert p.spend_usd == pytest.approx(2.04, abs=0.01)
    assert p.total_usd == pytest.approx(4.99, abs=0.02)  # under the $6.08 limit: resumable
    assert p.total_usd < g.limit_usd


def test_starting_prewarm_is_charged_only_its_remaining_startup(tmp_path):
    """19:25:00, before any marker: the caption pre-warm container has been starting for ~74 s.
    It is one of the two caption containers (no new start for it) and only the rest of its
    startup is still ahead; the second caption container is a full start."""
    now = parse_utc("2026-10-22T19:25:00Z")
    g = build_gate(tmp_path, with_markers=False, done=(), clock=lambda: now)
    g.observe(state_at_stop(live_7d63=False, now=now))
    h100 = pinned_rates().per_second("H100", 1, 8.0, 96.0)
    p = g.projection({"segment": 2})
    ahead = PREWARM[0] + 372.6 - now
    assert 290 < ahead < 300
    assert p.startup_usd == pytest.approx((ahead + 372.6) * h100, abs=1e-3)
