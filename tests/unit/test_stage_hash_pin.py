"""Pinned stage hashes for a fixed config (review 15 M1).

A print-only change to a module in the logic digest (``config/logic.py``) changes every stage
hash, so finished shards stop matching and a paid run in flight can no longer resume or pair
with its earlier steps. Fix-DX3 did this through ``jobs/indexing.py`` without anyone noticing.

If this test fails, either move the change out of the hashed modules (CLI, local runner,
reports: anything in ``NON_OUTPUT_MODULES``), or, when the change really does affect outputs,
update the pins below on purpose and say so in the commit message.
"""

from __future__ import annotations

from flora_deploy.config import logic
from flora_deploy.config.hashing import stage_hashes
from flora_deploy.config.schema import parse_config

SHA = "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"
RECEIPTS = {"captioner_bf16": "c" * 64, "tempseg_merged": "t" * 64}

FIXTURE = {
    "project": "demo",
    "license_ack": "research-only",
    "modal": {"max_gpus": 8, "budget_usd": 20, "environment": "main", "region": None,
              "tags": {}},
    "preset": "exact",
    "input": {"format": "lerobot_v3", "repo": "org/ds", "revision": SHA, "path": "sub",
              "cameras": {"top": "main", "l": "left_wrist", "r": "right_wrist"}},
    "pipeline": {"stages": ["segment", "caption", "reward.terminal", "reward.dense"],
                 "embodiment": "yam"},
    "output": {"sink": "s3://b/p"},
}

# Source digest of each logic group (main at e9fdc4e).
LOGIC_PINS = {
    "common": "eeae36a4bac1784f470cfa710cb518542f1d498b218cbdc3e8426d6eec8045a4",
    "nvidia-hf": "13a1affd795b7c5530c2842ed0305108b8bfc7c40d3acd16bd73f8629caf3998",
    "vllm-video": "b9b743f6438600b5be9bffebd466b371a590c49382ff009da34c17ca16f790b4",
    "reward": "3a172f70424f27547c66c2e96b6943163f4ca12ed521caaf4cccff55896c9dc1",
    "rubric": "ea090cd9ea807fc456374da0f5b1208870a945f5e619b849431d229d75478720",
}

STAGE_PINS = {
    "segcap": "3dc67ab28186008f570693f8c12881bc04aba94e4920764b276d31b305ad6696",
    "reward.terminal": "7ca57c13c4ece9924c39436eb1d0c1d8af9c8384e4b50aa9a7c1e523e18e04d9",
    "reward.dense": "199acca1b398025f5fcecfc960b59a4dba5c9e8b8f599d60be39fd8888b35041",
}


def test_logic_group_digests_are_pinned():
    got = {g: logic.source_digest(paths) for g, paths in logic.OUTPUT_MODULES.items()}
    changed = sorted(g for g in set(got) | set(LOGIC_PINS) if got.get(g) != LOGIC_PINS.get(g))
    assert not changed, (
        f"logic digest changed for {changed}: a hashed module was edited, which changes stage "
        "hashes (see this file's docstring)")


def test_stage_hashes_of_fixture_are_pinned():
    cfg = parse_config(FIXTURE, allow_uncertified=True)
    assert stage_hashes(cfg, derived_receipts=RECEIPTS) == STAGE_PINS
