"""A complete, synthetic Captioner BF16 ``derived_receipt.json`` (no torch, no files)."""

from __future__ import annotations

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.prep import convert_captioner as cc
from flora_deploy.prep.verify import pin_record


def captioner_receipt() -> dict:
    pin = PINS_BY_KEY["captioner"]
    return {
        "schema": 1, "ok": True, "kind": cc.DERIVED_NAME,
        "converter": {"version": 1, "module_sha256": "m" * 64, "git_commit": "c" * 40,
                      "git_dirty": False},
        "source": {"pin": pin_record(pin), "manifest_sha256": dict(pin.expected)[
            "manifest_sha256"], "shard_bytes": 10},
        "dtype_policy": {"load_dtype": "bfloat16", "dtype_plan": {"A_log": "float32"},
                         "non_load_dtype_keys": 96, "non_load_dtype_key_sha256": "k" * 64},
        "casts": {"F32->BF16:loaded": 1088, "F32->F32:loaded": 96},
        "files": [
            {"path": "model-00002-of-00002.safetensors", "size": 7, "sha256": "b" * 64,
             "header_sha256": "h" * 64, "kind": "converted", "source_sha256": "s" * 64,
             "tensors": 3},
            {"path": "config.json", "size": 2, "sha256": "a" * 64, "kind": "copied",
             "source_sha256": "a" * 64},
        ],
        "bytes": 9,
        "tools": {"python": "3.12.10", "platform": "Linux-x86_64-host1",
                  "torch": "2.11.0+cu130", "transformers": "5.8.1", "safetensors": "0.7.0"},
        "timings_s": {"total": 267.1, "per_shard": {"x": 1.0}},
        "created_utc": "2026-10-21T10:00:00Z",
    }
