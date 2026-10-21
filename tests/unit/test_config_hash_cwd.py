"""The config hash of a local input does not depend on the cwd or the machine (H0): a relative
``input.path`` resolves against the config file's directory and is hashed in a canonical
relative form."""

from __future__ import annotations

import shutil
from importlib import resources

import yaml

from flora_deploy.config.hashing import config_hash
from flora_deploy.config.receipts import load_certified
from flora_deploy.config.schema import load_config, with_pipeline

KW = {"derived_receipts": {"captioner_bf16": "0" * 64}, "image_fingerprint": lambda n: "fp"}


def _project(root, path="./videos"):
    d = yaml.safe_load((resources.files("flora_deploy.config") / "templates" /
                        "exact_mp4_dir.yaml").read_text())
    d["input"]["path"] = path
    root.mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos" / "ep0.mp4").write_bytes(b"episode zero")
    (root / "flora.yaml").write_text(yaml.safe_dump(d))
    return root / "flora.yaml"


def test_same_hash_from_two_cwds_and_two_machines(tmp_path, monkeypatch):
    cfg_path = _project(tmp_path / "proj")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "videos").mkdir()  # a decoy the cwd would have picked
    (elsewhere / "videos" / "other.mp4").write_bytes(b"not the dataset")

    monkeypatch.chdir(cfg_path.parent)
    a = config_hash(load_config(cfg_path), **KW)
    monkeypatch.chdir(elsewhere)
    b = config_hash(load_config(cfg_path), **KW)
    assert a.hash == b.hash
    assert a.document["shared"]["input"]["content"] is not None
    assert a.document["shared"]["input"]["dataset"] == "videos"

    # the same project on another machine (another absolute location)
    moved = tmp_path / "machine2" / "home" / "proj"
    shutil.copytree(cfg_path.parent, moved)
    assert config_hash(load_config(moved / "flora.yaml"), **KW).hash == a.hash
    # an absolute path to the same directory hashes like the relative one
    abs_cfg = _project(tmp_path / "abs", path=str(tmp_path / "abs" / "videos"))
    assert config_hash(load_config(abs_cfg), **KW).hash == a.hash

    # the base directory survives re-validation and the certified loader
    cfg = load_config(cfg_path)
    assert with_pipeline(cfg, determinism="deterministic").base_dir == cfg.base_dir
    assert load_certified(cfg_path)[0].base_dir == cfg_path.parent.resolve()

    # the content is bound: changing a file changes the hash
    (cfg_path.parent / "videos" / "ep0.mp4").write_bytes(b"episode zero, edited")
    assert config_hash(load_config(cfg_path), **KW).hash != a.hash
