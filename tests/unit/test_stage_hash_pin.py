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

# Source digest of each logic group (feat/caption-recert: the v4 caption re-certification,
# the worker startup-failure markers, the envelope fix and balanced certification).
LOGIC_PINS = {
    "common": "a06f85b7e962d2310b12917029bf1fd93761723b753e886ff3aa9716006bfa7d",
    "nvidia-hf": "9aa60b3e17d1b02f2c11a7885c58178b8ff4fe4906737738b2672366c8844cef",
    "vllm-video": "de285d9c0250721a480bc00b1bac12b990cf594b46a5bd34d0be4fc8c11089b0",
    "reward": "5714c300ce363c12e14506dd17d7b63fdab48f7560f8d661f53b0f225af90518",
    "rubric": "96e6658314ff638b9b1f56018a75bdede9172ee66552c7523bc31c4de8554353",
}

STAGE_PINS = {
    "segcap": "aed033db4929c5c619c243772b412bf3ef99079ebfc1123d069406807a57daf9",
    "reward.terminal": "6f7d44e0c3e31ee3208eac6858b33a3f1cdec93ce19fe31f66cd81dbb560b79e",
    "reward.dense": "e02e55d441f109ef2aca099b0d36b972900220de5079bd1da56c377b77e5eed3",
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
