"""D11 config hash: invariance to resource/cost knobs, sensitivity to everything hashed, the
logic digest and per-stage hashes."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import flora_deploy
from flora_deploy.config import hashing, logic, pins, presets
from flora_deploy.config.hashing import (
    MissingReceiptError,
    canonical_json,
    config_hash,
    stage_hashes,
)
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
                     "embodiment": "yam"},
        "output": {"sink": "s3://b/p"},
    }


def h(d: dict, receipts=RECEIPTS, **kw) -> str:
    return config_hash(parse_config(d, allow_uncertified=True), derived_receipts=receipts,
                       **kw).hash


def sh(d: dict, receipts=RECEIPTS, **kw) -> dict[str, str]:
    return stage_hashes(parse_config(d, allow_uncertified=True), derived_receipts=receipts, **kw)


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
        ("input.path", "./sub/"),
        ("input.path", "sub//"),
        ("allow_uncertified", True),
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
        ("pipeline.episode_instruction", {"source": "fixed", "text": "Wipe."}),
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
    assert doc["shared"]["logic_version"] == flora_deploy.LOGIC_VERSION
    assert doc["shared"]["logic_common"] == logic.logic_digest(None)
    kinds = doc["shard_kinds"]
    assert set(kinds) == {"segcap", "reward.terminal", "reward.dense"}
    assert set(kinds["segcap"]["nvidia_pins"]) == {"tempseg", "tempseg_code", "tempseg_base",
                                                   "captioner", "captioner_code"}
    assert set(kinds["reward.dense"]["nvidia_pins"]) == {"rewarder"}
    assert kinds["segcap"]["derived_weights"] == {"captioner_bf16": "c" * 64}
    assert kinds["reward.dense"]["derived_weights"] == {}
    assert kinds["segcap"]["stages"] == ["segment", "caption"]
    assert kinds["segcap"]["logic_digest"] == logic.logic_digest("segcap")
    assert doc["shared"]["decode"]["arch"] == "x86_64"
    assert set(res.stage_hashes) == set(kinds)
    assert len(set(res.stage_hashes.values())) == 3
    flat = res.canonical
    for word in ("H200", "H100", "budget_usd", "max_containers", "region", "s3://",
                 "allow_uncertified"):
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


# ---------------------------------------------------------------------------------------------
# Logic digest (review A1)


@pytest.fixture(scope="module")
def pkg_copy(tmp_path_factory) -> Path:
    """A copy of the flora_deploy source tree, to edit without touching the checkout."""
    dst = tmp_path_factory.mktemp("pkg") / "flora_deploy"
    shutil.copytree(logic.package_root(), dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


class _Edit:
    """Append a line to ``root/rel`` for the duration of a ``with`` block."""

    def __init__(self, root: Path, rel: str):
        self.path = root / rel

    def __enter__(self):
        self.original = self.path.read_bytes()
        self.path.write_bytes(self.original + b"\n# a fix\n")
        return self.path

    def __exit__(self, *exc):
        self.path.write_bytes(self.original)


@pytest.mark.parametrize("rel", ["nv/capture.py", "stages/reference.py", "media/index.py",
                                 "accel/framecache.py", "records.py", "config/hashing.py",
                                 "stages/reward.py", "nv/segment_glue.py", "nv/session.py",
                                 "nv/budget.py", "media/rangedecode.py", "media/cuts.py",
                                 "jobs/indexing.py", "workers_impl/reward.py",
                                 "modal_app/workers_app.py", "media/sources/lerobot_v3.py",
                                 "observe/stdout.py"])
def test_editing_an_output_module_changes_the_hash(pkg_copy, rel):
    assert h(base(), logic_root=pkg_copy) == BASE  # the copy hashes like the checkout
    with _Edit(pkg_copy, rel):
        assert h(base(), logic_root=pkg_copy) != BASE
    assert h(base(), logic_root=pkg_copy) == BASE


@pytest.mark.parametrize("rel", ["cli/init.py", "cli/plan.py", "jobs/orchestrator.py",
                                 "planner/options.py", "observe/aggregate.py", "media/synth.py",
                                 "validate/gatec.py", "config/schema.py", "__init__.py"])
def test_editing_a_non_output_module_keeps_the_hash(pkg_copy, rel):
    with _Edit(pkg_copy, rel):
        assert h(base(), logic_root=pkg_copy) == BASE


def test_logic_digest_is_per_shard_kind(pkg_copy):
    before = sh(base(), logic_root=pkg_copy)
    with _Edit(pkg_copy, "stages/reward.py"):
        after = sh(base(), logic_root=pkg_copy)
    assert after["segcap"] == before["segcap"]
    assert after["reward.dense"] != before["reward.dense"]
    with _Edit(pkg_copy, "stages/reference.py"):
        after = sh(base(), logic_root=pkg_copy)
    assert after["segcap"] != before["segcap"]
    assert after["reward.terminal"] == before["reward.terminal"]
    with _Edit(pkg_copy, "records.py"):  # common: every kind
        after = sh(base(), logic_root=pkg_copy)
    assert all(after[k] != before[k] for k in before)


def test_missing_logic_module_fails_closed(pkg_copy):
    with _Edit(pkg_copy, "nv/session.py") as p:
        p.unlink()
        with pytest.raises(FileNotFoundError, match="nv/session.py"):
            h(base(), logic_root=pkg_copy)


# ---------------------------------------------------------------------------------------------
# Per-stage hashes (review A5)


def test_adding_a_reward_stage_keeps_finished_segcap_shards():
    a = sh(set_(base(), "pipeline.stages", ["segment", "caption"]))
    b = sh(base())
    c = sh(set_(base(), "pipeline.stages", ["segment", "caption", "reward.rubric"]))
    assert a["segcap"] == b["segcap"] == c["segcap"]
    assert set(b) == {"segcap", "reward.terminal", "reward.dense"}
    assert h(set_(base(), "pipeline.stages", ["segment", "caption"])) != BASE  # job hash moves


def test_vllm_image_only_invalidates_rewards():
    fp = hashing._default_fingerprint
    bumped = sh(base(), image_fingerprint=lambda n: fp(n) + ("x" if n == "img_vllm" else ""))
    ref = sh(base())
    assert bumped["segcap"] == ref["segcap"]
    assert bumped["reward.terminal"] != ref["reward.terminal"]


# ---------------------------------------------------------------------------------------------
# Review 4: output-affecting run modes are config fields, hashed per kind (H3, M4)


def test_determinism_is_hashed_into_segcap_only():
    ref = sh(base())
    det = sh(set_(base(), "pipeline.determinism", "deterministic"))
    assert det["segcap"] != ref["segcap"]
    assert det["reward.terminal"] == ref["reward.terminal"]
    assert det["reward.dense"] == ref["reward.dense"]
    assert h(set_(base(), "pipeline.determinism", "deterministic")) != BASE  # job hash too
    assert h(set_(base(), "pipeline.determinism", "nvidia")) == BASE  # the default
    doc = config_hash(parse_config(base()), derived_receipts=RECEIPTS).document
    assert doc["shard_kinds"]["segcap"]["modes"] == {"determinism": "nvidia"}


def test_determinism_needs_the_segcap_worker():
    from flora_deploy.config.schema import ConfigError

    rewards = set_(base(), "pipeline.stages", ["reward.terminal"])
    with pytest.raises(ConfigError, match="determinism"):
        parse_config(set_(rewards, "pipeline.determinism", "deterministic"))
    with pytest.raises(ConfigError, match="determinism"):
        parse_config(set_(base(), "pipeline.determinism", "fast"))


def test_terminal_rationale_is_hashed_into_reward_terminal_only():
    ref = sh(base())
    rat = sh(set_(base(), "pipeline.terminal_rationale", True))
    assert rat["reward.terminal"] != ref["reward.terminal"]
    assert rat["reward.dense"] == ref["reward.dense"]
    assert rat["segcap"] == ref["segcap"]
    doc = config_hash(parse_config(base()), derived_receipts=RECEIPTS).document
    assert doc["shard_kinds"]["reward.terminal"]["modes"] == {"terminal_mode": "terminal"}
    assert doc["shard_kinds"]["reward.dense"]["modes"] == {}


def test_cli_override_goes_through_the_config():
    from flora_deploy.config.schema import output_settings, with_pipeline

    cfg = parse_config(base())
    assert output_settings(cfg) == {"determinism": "nvidia", "terminal_mode": "terminal"}
    det = with_pipeline(cfg, determinism="deterministic", terminal_rationale=True)
    assert output_settings(det) == {"determinism": "deterministic",
                                    "terminal_mode": "terminal_rationale"}
    assert config_hash(det, derived_receipts=RECEIPTS).hash != BASE
    assert cfg.pipeline.determinism == "nvidia"  # the original is unchanged


def test_input_content_digest_is_hashed():
    assert h(base(), input_digest="a" * 64) != BASE
    assert h(base(), input_digest="a" * 64) != h(base(), input_digest="b" * 64)
    assert h(base(), input_digest=None) == BASE


def test_gpu_type_is_not_hashed_but_gpu_count_is_checked():
    """M7: GPU type stays out of the hash (D11; recorded in provenance.gpu), and memory is
    checked per device: a second GPU does not make a single-device engine fit."""
    from flora_deploy.config.schema import ConfigError

    assert h(set_(base(), "resources", {"segcap": {"gpu": "B200"}})) == BASE
    for res in ({"segcap": {"gpu": "L40S", "gpu_count": 2}},
                {"segcap": {"gpu": "H100", "gpu_count": 2}},
                {"segcap": {"gpu_count": 2}},
                {"reward.terminal": {"gpu": "L40S", "gpu_count": 2}},
                {"reward.terminal": {"gpu": "A100-40GB", "gpu_count": 2}}):
        with pytest.raises(ConfigError, match="GPU"):
            parse_config(set_(base(), "resources", res))


def test_reward_pin_only_invalidates_rewards(monkeypatch):
    ref = sh(base())
    p = pins.PINS_BY_KEY["rewarder"]
    monkeypatch.setitem(hashing.PINS_BY_KEY, "rewarder", replace(p, revision="f" * 40))
    bumped = sh(base())
    assert bumped["segcap"] == ref["segcap"] and bumped["reward.dense"] != ref["reward.dense"]


def test_shared_inputs_change_every_stage_hash():
    ref = sh(base())
    for path, value in (("pipeline.embodiment", "dex"), ("input.revision", "0" * 40),
                        ("pipeline.envelope", "enforce")):
        other = sh(set_(base(), path, value))
        assert all(other[k] != ref[k] for k in ref), path


def test_balanced_caption_depends_on_segment():
    bal = set_(set_(base(), "preset", "balanced"), "pipeline.stages",
               ["segment", "caption", "reward.terminal"])
    ref = sh(bal)
    eng = presets.PRESETS["balanced"].engines["segment"]
    seg_changed = sh(bal, engines={"segment": replace(eng, sampling={"temperature": 0.1})})
    assert seg_changed["segment"] != ref["segment"]
    assert seg_changed["caption"] != ref["caption"]  # it consumes segment's outputs
    assert seg_changed["reward.terminal"] == ref["reward.terminal"]


def test_balanced_hash_includes_required_vllm_engine_kwargs():
    from flora_deploy.nv.capture import REQUIRED_VLLM_ENGINE_KWARGS

    bal = set_(base(), "preset", "balanced")
    doc = config_hash(parse_config(bal, allow_uncertified=True),
                      derived_receipts=RECEIPTS).document
    for kind in ("segment", "caption"):
        args = doc["shard_kinds"][kind]["engine"]["engine_args"]
        assert args == dict(REQUIRED_VLLM_ENGINE_KWARGS) and args["mm_processor_cache_gb"] == 0
    eng = presets.PRESETS["balanced"].engines["caption"]
    cached = replace(eng, engine_args={"mm_processor_cache_gb": 4})
    assert h(bal, engines={"caption": cached}) != h(bal)


def test_path_normalization_for_repo_and_local_inputs():
    assert h(set_(base(), "input.path", "./sub")) == BASE
    root = set_(base(), "input.path", None)
    assert h(set_(base(), "input.path", ".")) == h(root)
    local = {"format": "mp4_dir", "path": "./videos/"}
    assert h(set_(base(), "input", local)) == h(set_(base(), "input",
                                                     {**local, "path": "videos"}))
