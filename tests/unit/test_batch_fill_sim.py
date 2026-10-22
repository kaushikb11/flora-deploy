"""Batch fill (WP1) on the virtual-clock simulation (``tests/fakes/batch_sim.py``): the real
runner, gates and micro-batcher with modelled CPU and GPU time. ``python -m
tests.fakes.batch_sim`` prints the numbers these tests bound."""

from __future__ import annotations

from tests.fakes import batch_sim as B


def test_simulation_replays_wave_h():
    """The one-phase path on the Wave H episodes is close to the measured run (139.5 s stage
    wall, GPU busy 36%, the timeout firing in most rounds), which calibrates the model."""
    r = B.simulate(B.BEFORE, episodes=6, frames_s=10.0)
    assert 120 <= r.makespan_s <= 160, r.as_json()
    assert 0.3 <= r.gpu_busy_fraction <= 0.5, r.as_json()
    assert r.triggers.get("timeout", 0) >= r.rounds - 2, r.as_json()
    assert 3.5 <= r.mean_batch <= 5.5


def test_two_phase_policy_fills_the_gpu():
    before = B.simulate(B.BEFORE, episodes=64, frames_s=0.5)
    after = B.simulate(B.AFTER, episodes=64, frames_s=0.5)
    assert after.makespan_s < 0.85 * before.makespan_s, (before.as_json(), after.as_json())
    assert after.gpu_busy_fraction > 0.95 > before.gpu_busy_fraction + 0.1
    assert after.gates["peak_preparing"] <= after.policy["prefetch_episodes"] == 8
    assert after.gates["peak_inferring"] <= after.policy["infer_episodes"] == 64
    assert after.policy["max_wait_s"] == 0.5


def test_cpu_bound_shards_are_not_slowed_down():
    """With Wave H's frame-building cost the shard is CPU-bound on 8 cores: the new policy
    cannot speed it up, and must not slow it down materially."""
    before = B.simulate(B.BEFORE, episodes=64, frames_s=10.0)
    after = B.simulate(B.AFTER, episodes=64, frames_s=10.0)
    assert after.makespan_s <= 1.05 * before.makespan_s
    assert after.gpu_busy_fraction >= before.gpu_busy_fraction


def test_simulation_is_deterministic():
    a = B.simulate(B.AFTER, episodes=24, frames_s=0.5).as_json()
    b = B.simulate(B.AFTER, episodes=24, frames_s=0.5).as_json()
    assert a == b
