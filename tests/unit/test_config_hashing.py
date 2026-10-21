"""D11 config hash: invariance to resource/cost knobs, sensitivity to everything hashed."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

import flora_deploy
from flora_deploy.config import hashing, pins, presets
from flora_deploy.config.hashing import MissingReceiptError, canonical_json, config_hash
from flora_deploy.config.schema import parse_config

SHA = "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"
RECEIPTS = {"captioner_bf16": "c" * 64, "tempseg_merged": "t" * 64}


def base() -> dict:
    return {
        "project": "demo",
        "license_ack": "research-only",
        "modal": {"max_gpus": 8, "budget_usd": 20, "environment": "main", "region": None,
                  "tags": {}},
        "preset": "exact",
        "input": {"format": "lerobot_v3", "repo": "org/ds", "revision": SHA, "path": "sub",
                  "cameras": {"top": "main", "l": "left_wrist", "r": "right_wrist"}},
        "pipeline": {"stages": ["segment", "caption", "reward.terminal", "reward.dense"],
                     "embodiment": "human"},
        "output": {"sink": "s3://b/p"},
    }


def h(d: dict, receipts=RECEIPTS, **kw) -> str:
    return config_hash(parse_config(d), derived_receipts=receipts, **kw).hash


def set_(d: dict, path: str, value) -> dict:
    d = copy.deepcopy(d)
    *head, last = path.split(".")
    cur = d
    for k in head:
        cur = cur.setdefault(k, {})
    cur[last] = value
    return d


BASE = h(base())


@pytest.mark.parametrize(
    "path,value",
    [
        ("resources", {"segcap": {"gpu": "B200"}}),
        ("resources", {"segcap": {"gpu_count": 2}}),
        ("resources", {"segcap": {"max_containers": 4}, "reward.dense": {"max_containers": 2}}),
        ("resources", {"reward.terminal": {"max_inputs": 3, "cpu": 16, "memory": 200}}),
        ("resources", {"reward.terminal": {"gpu": "A100-80GB"}}),
        ("modal.budget_usd", 999),
        ("modal.max_gpus", 64),
        ("modal.region", "us-east"),
        ("modal.tags", {"team": "x", "run": "y"}),
        ("modal.environment", "staging"),
        ("modal.environment_budget_factor", 2.0),
        ("project", "other"),
        ("input.stage_to", "volume://inputs"),
        ("output", {"sink": "gs://other/q", "formats": ["jsonl", "parquet"]}),
        ("retries", {"max_starts_before_quarantine": 7}),
        ("pipeline.shard_target_gpu_minutes", 33),
        ("observability", {"log_level": "debug", "heartbeat_s": 5, "progress": False}),
    ],
)
def test_invariant_to_non_output_knobs(path, value):
    assert h(set_(base(), path, value)) == BASE


@pytest.mark.parametrize(
    "path,value",
    [
        ("preset", "balanced"),
        ("pipeline.stages", ["segment", "caption", "reward.terminal"]),
        ("pipeline.embodiment", "dex"),
        ("pipeline.envelope", "enforce"),
        ("pipeline.episode_instruction", {"source": "none"}),
        ("pipeline.episode_instruction", {"source": "fixed", "text": "Fold."}),
        ("input.repo", "org/other"),
        ("input.revision", "0" * 40),
        ("input.path", "other"),
        ("input.format", "lerobot_v21"),
        ("input.cameras", {"top": "main"}),
        ("input.cameras", {"top": "main", "l": "right_wrist", "r": "left_wrist"}),
    ],
)
def test_sensitive_to_config_inputs(path, value):
    assert h(set_(base(), path, value)) != BASE


def test_sensitive_to_fixed_text():
    a = set_(base(), "pipeline.episode_instruction", {"source": "fixed", "text": "A."})
    b = set_(base(), "pipeline.episode_instruction", {"source": "fixed", "text": "B."})
    assert h(a) != h(b)


def test_sensitive_to_receipts():
    assert h(base(), receipts={"captioner_bf16": "d" * 64}) != BASE
    with pytest.raises(MissingReceiptError, match="captioner_bf16"):
        h(base(), receipts={})
    # rewards-only needs no derived weights; unused receipts don't enter the hash
    r = set_(base(), "pipeline.stages", ["reward.terminal"])
    assert h(r, receipts={}) == h(r, receipts={"captioner_bf16": "x"})
    bal = set_(base(), "preset", "balanced")
    with pytest.raises(MissingReceiptError, match="tempseg_merged"):
        h(bal, receipts={"captioner_bf16": "c" * 64})


def test_sensitive_to_logic_version(monkeypatch):
    monkeypatch.setattr(hashing, "LOGIC_VERSION", flora_deploy.LOGIC_VERSION + ".1")
    assert h(base()) != BASE


@pytest.mark.parametrize("key", ["tempseg", "tempseg_code", "tempseg_base", "captioner",
                                 "captioner_code", "rewarder"])
@pytest.mark.parametrize("field", ["revision", "plan_digest", "expected", "code_digests"])
def test_sensitive_to_every_used_pin(monkeypatch, key, field):
    p = pins.PINS_BY_KEY[key]
    value = {"revision": "f" * 40, "plan_digest": "e" * 64,
             "expected": (("manifest_sha256", "a" * 64),),
             "code_digests": (("code_manifest_sha256", "b" * 64),)}[field]
    if getattr(p, field) == value:
        pytest.skip("same value")
    monkeypatch.setitem(hashing.PINS_BY_KEY, key, replace(p, **{field: value}))
    assert h(base()) != BASE


def test_unused_pins_do_not_matter(monkeypatch):
    r = set_(base(), "pipeline.stages", ["reward.terminal"])
    before = h(r)
    p = pins.PINS_BY_KEY["captioner"]
    monkeypatch.setitem(hashing.PINS_BY_KEY, "captioner", replace(p, revision="f" * 40))
    assert h(r) == before


@pytest.mark.parametrize("key", ["av", "pillow", "numpy"])
def test_sensitive_to_decode_pins(monkeypatch, key):
    monkeypatch.setitem(hashing.RUNTIME, key, "0.0.0")
    assert h(base()) != BASE


def test_sensitive_to_image_fingerprint():
    fp = hashing._default_fingerprint
    assert h(base(), image_fingerprint=fp) == BASE
    assert h(base(), image_fingerprint=lambda n: fp(n) + ("x" if n == "img_hfref" else "")) \
        != BASE
    assert h(base(), image_fingerprint=lambda n: fp(n) + ("x" if n == "img_vllm" else "")) \
        != BASE


@pytest.mark.parametrize("change", [
    {"name": "other"}, {"version": {"vllm": "0.25.0"}}, {"patches": ("opt1",)},
    {"engine_args": {"max_num_seqs": 8}}, {"sampling": {"temperature": 0.0}},
    {"mm_kwargs": {"fps": 2}}, {"image": "img_cpu"},
])
def test_sensitive_to_engine_identity(change):
    eng = presets.PRESETS["exact"].engines["reward.dense"]
    assert h(base(), engines={"reward.dense": replace(eng, **change)}) != BASE
    assert h(base(), engines={"reward.dense": eng}) == BASE


def test_document_contents_and_exclusions():
    res = config_hash(parse_config(base()), derived_receipts=RECEIPTS)
    doc = res.document
    assert json.loads(res.canonical) == doc
    assert doc["logic_version"] == flora_deploy.LOGIC_VERSION
    assert set(doc["nvidia_pins"]) == {"tempseg", "tempseg_code", "tempseg_base", "captioner",
                                       "captioner_code", "rewarder"}
    assert doc["derived_weights"] == {"captioner_bf16": "c" * 64}
    assert set(doc["shard_kinds"]) == {"segcap", "reward.terminal", "reward.dense"}
    assert doc["decode"]["arch"] == "x86_64"
    flat = res.canonical
    for word in ("H200", "H100", "budget_usd", "max_containers", "region", "s3://"):
        assert word not in flat


def test_canonical_json_order_independent():
    assert canonical_json({"b": 1, "a": {"y": [1, 2], "x": None}}) == \
        canonical_json({"a": {"x": None, "y": [1, 2]}, "b": 1})
    d = base()
    reordered = {k: d[k] for k in reversed(list(d))}
    reordered["input"]["cameras"] = dict(reversed(list(d["input"]["cameras"].items())))
    assert h(reordered) == BASE
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_stable_across_python_runs():
    code = (
        "import json,sys\n"
        "from flora_deploy.config.schema import parse_config\n"
        "from flora_deploy.config.hashing import config_hash\n"
        "d=json.loads(sys.argv[1]); r=json.loads(sys.argv[2])\n"
        "print(config_hash(parse_config(d), derived_receipts=r).hash)\n"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code, json.dumps(base()),
                              json.dumps(RECEIPTS)], env=env, capture_output=True, text=True,
                             check=True)
        outs.add(out.stdout.strip())
    assert outs == {BASE}
