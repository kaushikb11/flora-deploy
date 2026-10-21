"""flora.yaml schema: validation errors, YAML -> RunSettings, presets."""

from __future__ import annotations

import copy

import pytest

from flora_deploy.config.presets import PRESETS, PresetUnavailableError, get_preset
from flora_deploy.config.schema import ConfigError, loads_config, parse_config, to_run_settings
from flora_deploy.jobs.settings import PoolSettings, RunSettings

SHA = "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"
RECEIPTS = {"captioner_bf16": "c" * 64, "tempseg_merged": "t" * 64}


def base() -> dict:
    return {
        "project": "demo",
        "license_ack": "research-only",
        "modal": {"max_gpus": 4, "budget_usd": 20},
        "preset": "exact",
        "input": {"format": "lerobot_v3", "repo": "org/ds", "revision": SHA},
        "pipeline": {"stages": ["segment", "caption", "reward.terminal"], "embodiment": "human"},
        "output": {"sink": "s3://b/p"},
    }


def with_(d: dict, path: str, value) -> dict:
    d = copy.deepcopy(d)
    *head, last = path.split(".")
    cur = d
    for k in head:
        cur = cur.setdefault(k, {})
    if value is _DEL:
        del cur[last]
    else:
        cur[last] = value
    return d


_DEL = object()


def err(d: dict) -> str:
    with pytest.raises(ConfigError) as e:
        parse_config(d)
    return str(e.value)


def test_base_is_valid():
    cfg = parse_config(base())
    assert cfg.preset == "exact"
    assert cfg.pipeline.envelope == "warn"
    assert cfg.pipeline.episode_instruction.source == "dataset_task"


@pytest.mark.parametrize(
    "path,value,needle",
    [
        ("license_ack", _DEL, "license_ack: required"),
        ("license_ack", "commercial", "license_ack"),
        ("modal.budget_usd", 0, "modal.budget_usd"),
        ("modal.max_gpus", 0, "modal.max_gpus"),
        ("modal.max_gpu", 3, "modal.max_gpu: unknown key"),
        ("input.revision", "main", "full 40-hex commit"),
        ("input.revision", _DEL, "full 40-hex commit"),
        ("input.repo", _DEL, "set `repo`"),
        ("input", {"format": "mp4_dir", "path": "v", "revision": SHA}, "only applies"),
        ("input.format", "rosbag", "input.format"),
        ("input.cameras", {"a": "main", "b": "main"}, "more than once"),
        ("input.cameras", {"a": "left_wrist"}, "needs a 'main'"),
        ("input.cameras", {"a": "main", "b": "left_wrist"}, "1 view"),
        ("pipeline.stages", [], "pipeline.stages"),
        ("pipeline.stages", ["caption"], "`caption` needs `segment`"),
        ("pipeline.stages", ["segment", "segment", "caption"], "more than once"),
        ("pipeline.stages", ["segment"], "segment and caption together"),
        ("pipeline.embodiment", "robot", "pipeline.embodiment"),
        ("pipeline.envelope", "strict", "pipeline.envelope"),
        ("pipeline.episode_instruction", {"source": "fixed"}, "needs `text`"),
        ("output.sink", "/tmp/out", "output.sink"),
        ("output.formats", ["csv"], "output.formats"),
        ("preset", "budget", "not available yet"),
        ("preset", "interactive", "not available yet"),
        ("preset", "turbo", "preset"),
        ("resources", {"segment": {"gpu": "L40S"}}, "as `segcap`"),
        ("resources", {"segcap": {"gpu": "H100!:2"}}, "gpu_count"),
        ("resources", {"segcap": {"gpu": "H 100"}}, "not a Modal GPU"),
        ("resources", {"segcap": {"max_containers": 0}}, "max_containers"),
        ("resources", {"segcap": {"max_containers": 4}}, "can use 5 GPUs"),
    ],
)
def test_validation_errors(path, value, needle):
    assert needle in err(with_(base(), path, value))


def test_errors_are_all_listed_with_paths():
    d = with_(with_(base(), "modal.budget_usd", -1), "output.sink", "nope")
    msg = err(d)
    assert "2 problem(s)" in msg and "modal.budget_usd" in msg and "output.sink" in msg


def test_non_mapping_and_bad_yaml():
    with pytest.raises(ConfigError, match="mapping"):
        parse_config(["x"])
    with pytest.raises(ConfigError, match="not valid YAML"):
        loads_config("a: [1,")


def test_mp4_dir_path_only():
    d = with_(base(), "input", {"format": "mp4_dir", "path": "./videos"})
    assert parse_config(d).input.dataset == "./videos"


def test_stages_are_canonically_ordered():
    d = with_(base(), "pipeline.stages", ["reward.terminal", "caption", "segment"])
    assert parse_config(d).pipeline.stages == ("segment", "caption", "reward.terminal")


def test_exact_to_run_settings():
    d = with_(base(), "modal.tags", {"team": "x"})
    d = with_(d, "resources", {"segcap": {"max_containers": 2, "memory": 128}})
    rs = to_run_settings(parse_config(d), "job-1", derived_receipts=RECEIPTS)
    assert isinstance(rs, RunSettings)
    assert rs.job_id == "job-1" and rs.preset == "exact" and len(rs.config_hash) == 64
    assert rs.budget_usd == 20 and rs.max_gpus == 4 and rs.tags == {"team": "x"}
    assert set(rs.pools) == {"segcap", "reward.terminal"}
    assert rs.pools["segcap"] == PoolSettings(gpu="H200", max_containers=2, cpu_cores=8.0,
                                              memory_gib=128.0)
    assert rs.pools["reward.terminal"].gpu == "H100!"
    assert rs.envelope == "warn" and rs.max_starts_before_quarantine == 3


def test_balanced_pools():
    d = with_(base(), "preset", "balanced")
    d = with_(d, "pipeline.stages", ["segment", "caption", "reward.terminal", "reward.dense"])
    d = with_(d, "resources", {"segment": {"max_inputs": 3}})
    rs = to_run_settings(parse_config(d), "j", derived_receipts=RECEIPTS)
    assert {k: p.gpu for k, p in rs.pools.items()} == {
        "segment": "L40S", "caption": "H100!", "reward.terminal": "H100!", "reward.dense": "H100!"}
    assert rs.pools["segment"].max_inputs == 3 and rs.pools["segment"].cpu_cores == 6.0


def test_explicit_config_hash_and_settings_passthrough():
    d = with_(base(), "retries", {"max_starts_before_quarantine": 5})
    d = with_(d, "pipeline.envelope", "enforce")
    d = with_(d, "pipeline.shard_target_gpu_minutes", 20)
    rs = to_run_settings(parse_config(d), "j", config_hash="h" * 64)
    assert rs.config_hash == "h" * 64
    assert rs.max_starts_before_quarantine == 5 and rs.envelope == "enforce"
    assert rs.shard_target_gpu_minutes == 20


def test_presets():
    exact, balanced = get_preset("exact"), get_preset("balanced")
    assert exact.pools["segcap"].gpu == "H200"
    assert exact.stage_kinds["segment"] == exact.stage_kinds["caption"] == "segcap"
    assert {exact.pools[k].gpu for k in ("reward.terminal", "reward.dense")} == {"H100!"}
    assert balanced.pools["segment"].gpu == "L40S"
    assert balanced.pools["caption"].gpu == "H100!"
    assert {balanced.pools[k].gpu for k in ("reward.terminal", "reward.dense")} == {"H100!"}
    for spec in (exact, balanced):
        assert set(spec.engines) == set(spec.pools) == set(spec.pins)
        assert set(spec.stage_kinds.values()) <= set(spec.pools)
        for p in spec.pools.values():
            assert p.gpu_count >= 1 and p.cpu_cores > 0 and p.memory_gib > 0
    for name in ("budget", "interactive"):
        assert not PRESETS[name].available
        with pytest.raises(PresetUnavailableError, match="not available yet"):
            get_preset(name)
