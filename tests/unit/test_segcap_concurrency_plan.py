"""Review 6 H2: plan, factory and runner agree on segcap concurrency.

The measured ``exact x4`` is four **episodes in flight inside one container** (the worker's
in-process batch, ``FastExactEngine.concurrency``), not four Modal inputs: the segcap runner
takes one shard per container (``ConcurrentShardRefused``). So ``flora plan --write`` writes
``resources.segcap.episode_concurrency: 4`` with ``max_inputs: 1``; the factory deploys the
class without ``@modal.concurrent`` and hands the worker ``FLORA_EPISODE_CONCURRENCY=4``; the
worker builds its engine with that concurrency; the orchestrator keeps one live call per
container; budget and planner price the same c=4 rate."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import template_text
from flora_deploy.config.schema import load_config, parse_config
from flora_deploy.modal_app import factory as f

RECEIPTS = {"captioner_bf16": "0" * 64}
ARGS = ["--video-hours", "100", "--budget", "6000", "--deadline", "96h", "--max-gpus", "10"]


def test_plan_factory_and_runner_agree(tmp_path):
    from flora_deploy.modal_app import reference_app as ra

    cfg_path = tmp_path / "flora.yaml"
    cfg_path.write_text(template_text("exact_mp4_dir"))
    r = CliRunner().invoke(app, ["plan", "-c", str(cfg_path), *ARGS, "--preset", "exact",
                                 "--write", "1"])
    assert r.exit_code == 0, r.output
    written = yaml.safe_load(cfg_path.read_text())["resources"]["segcap"]
    # 1. plan: episode concurrency, one Modal input per container
    assert written["max_inputs"] == 1 and written["episode_concurrency"] == 4, written
    cfg = load_config(cfg_path)
    pool = cfg.pools()["segcap"]
    assert (pool.max_inputs, pool.episode_concurrency) == (1, 4)
    # 2. factory: no @modal.concurrent for the segcap worker, the concurrency in its env
    plan = f.plan_app(cfg, derived_receipts=RECEIPTS)
    (spec,) = [o for o in plan.objects if o.kind == "cls" and "segcap" in o.shard_kinds]
    assert spec.max_inputs is None
    assert spec.env["FLORA_EPISODE_CONCURRENCY"] == "4"
    # 3. runner: the worker's engine runs exactly that many episodes at once
    assert ra.episode_concurrency(spec.env, default=1) == 4
    # 4. orchestrator: one live segcap call per container (no refused inputs)
    from flora_deploy.config.schema import to_run_settings

    rs = to_run_settings(cfg, "job-x", config_hash="h" * 64, stage_hashes={})
    p = rs.pools["segcap"]
    assert p.max_inputs == 1 and p.episode_concurrency == 4  # live calls = containers


def test_factory_refuses_concurrent_segcap_inputs():
    d = yaml.safe_load(template_text("exact_mp4_dir"))
    d.setdefault("resources", {}).setdefault("segcap", {})["max_inputs"] = 4
    with pytest.raises(ValueError, match="episode_concurrency"):
        parse_config(d)


def test_plan_json_and_table_label_episode_concurrency(tmp_path):
    """Review 8a R7: ``flora plan --json`` and the table show segcap's x4 as episodes in flight
    (``max_inputs: 1, episode_concurrency: 4``, as ``--write`` writes it), not Modal inputs."""
    import json

    cfg_path = tmp_path / "flora.yaml"
    cfg_path.write_text(template_text("exact_mp4_dir"))
    r = CliRunner().invoke(app, ["plan", "-c", str(cfg_path), *ARGS, "--preset", "exact",
                                 "--json"])
    assert r.exit_code == 0, r.output
    kinds = [k for o in json.loads(r.output)["options"] for k in o["kinds"]]
    seg = [k for k in kinds if k["kind"] == "segcap"]
    assert seg and all(k["max_inputs"] == 1 for k in seg), seg
    assert 4 in {k["episode_concurrency"] for k in seg}, seg
    assert all(k["episode_concurrency"] is None for k in kinds if k["kind"] != "segcap")
    r = CliRunner().invoke(app, ["plan", "-c", str(cfg_path), *ARGS, "--preset", "exact"])
    assert r.exit_code == 0, r.output
    assert "x4 ep" in r.output, r.output


def test_budget_prices_the_episode_concurrency_rate():
    from flora_deploy.jobs.settings import PoolSettings, episode_concurrency

    assert episode_concurrency("segcap", PoolSettings(gpu="H200", episode_concurrency=8)) == 8
    assert episode_concurrency("segcap", PoolSettings(gpu="H200")) == 4  # measured default
    assert episode_concurrency("reward.terminal", PoolSettings(gpu="H100!", max_inputs=2)) == 2
