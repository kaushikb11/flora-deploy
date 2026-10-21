"""Content digest of a derived receipt (``prep.convert_captioner.content_digest``): what
``pins.DerivedPin.content_sha256`` pins. No torch needed."""

from __future__ import annotations

import copy
import json

import pytest

from flora_deploy.config.pins import DERIVED_PINS_BY_KEY, PINS_BY_KEY
from flora_deploy.prep import convert_captioner as cc
from flora_deploy.prep import merge_tempseg as mt
from flora_deploy.prep.verify import pin_record
from tests.fakes.derived import captioner_receipt


def test_same_content_other_build_same_digest():
    r = captioner_receipt()
    other = copy.deepcopy(r)
    other["timings_s"] = {"total": 1.5}
    other["created_utc"] = "2031-01-01T00:00:00Z"
    other["tools"].update(platform="Linux-aarch64-elsewhere", torch="2.11.0+cu128")
    other["converter"].update(git_commit="d" * 40, git_dirty=True, module_sha256="n" * 64)
    other["source"]["shard_bytes"] = 11
    other["bytes"] = 99
    other["files"].reverse()  # file order in the receipt does not matter
    other["files"][0]["tensors"] = 4
    assert cc.content_digest(other) == cc.content_digest(r)
    # the receipt file bytes (what weights.derived records) differ, as they should
    assert json.dumps(other, sort_keys=True) != json.dumps(r, sort_keys=True)


@pytest.mark.parametrize("change", [
    lambda r: r["files"][0].update(sha256="c" * 64),
    lambda r: r["files"][0].update(size=8),
    lambda r: r["files"][0].update(header_sha256="i" * 64),
    lambda r: r["files"][1].update(path="config2.json"),
    lambda r: r["files"].pop(),
    lambda r: r["casts"].update({"F32->BF16:loaded": 1087}),
    lambda r: r["dtype_policy"].update(non_load_dtype_keys=95),
    lambda r: r["converter"].update(version=2),
    lambda r: r["source"].update(manifest_sha256="0" * 64),
    lambda r: r["source"]["pin"].update(revision="0" * 40),
    lambda r: r.update(kind="tempseg-merged"),
])
def test_output_determining_change_changes_digest(change):
    r = captioner_receipt()
    changed = copy.deepcopy(r)
    change(changed)
    assert cc.content_digest(changed) != cc.content_digest(r)


def test_stored_digest_is_checked_not_trusted():
    r = captioner_receipt()
    good = cc.content_digest(r)
    assert cc.checked_content_digest({**r, "content_sha256": good}) == good
    assert cc.checked_content_digest(r) == good  # older receipts have none stored
    with pytest.raises(cc.ContentDigestError, match="stored content_sha256"):
        cc.checked_content_digest({**r, "content_sha256": "0" * 64})
    # the stored field itself is not part of the digest
    assert cc.content_digest({**r, "content_sha256": "0" * 64}) == good
    with pytest.raises(cc.ContentDigestError, match="lacks"):
        cc.content_digest({k: v for k, v in r.items() if k != "casts"})


def tempseg_receipt() -> dict:
    r = captioner_receipt()
    r.pop("converter")
    r.update(kind=mt.DERIVED_NAME,
             merger={"version": mt.MERGER_VERSION, "workers": 2, "torch_threads": 4},
             base={"pin": pin_record(PINS_BY_KEY["tempseg_base"]), "plan_digest": "p" * 64,
                   "prep_receipt_sha256": "q" * 64, "shard_bytes": 5},
             lora={"r": 32, "alpha": 64, "scale": 2.0},
             merge_stats={"mean_abs_change": 1e-3})
    return r


def test_tempseg_digest_covers_base_and_lora_not_stats():
    r = tempseg_receipt()
    d = cc.content_digest(r)
    for change in (lambda x: x["merge_stats"].update(mean_abs_change=2e-3),
                   lambda x: x["merger"].update(workers=3, torch_threads=8),
                   lambda x: x["base"].update(prep_receipt_sha256="z" * 64)):
        other = copy.deepcopy(r)
        change(other)
        assert cc.content_digest(other) == d
    for change in (lambda x: x["lora"].update(scale=1.0),
                   lambda x: x["base"]["pin"].update(revision="0" * 40),
                   lambda x: x["merger"].update(version=1)):
        other = copy.deepcopy(r)
        change(other)
        assert cc.content_digest(other) != d


def test_verify_merged_refuses_a_stale_stored_digest(tmp_path):
    r = tempseg_receipt()
    (tmp_path / mt.DERIVED_RECEIPT).write_text(json.dumps({**r, "content_sha256": "0" * 64}))
    with pytest.raises(mt.MergeError, match="stored content_sha256"):
        mt.verify_merged(tmp_path)


def test_pins_carry_content_digests():
    for key in ("captioner-bf16", "tempseg-merged"):
        p = DERIVED_PINS_BY_KEY[key]
        assert p.content_sha256 is None or len(p.content_sha256) == 64
        assert not hasattr(p, "receipt_sha256")
    assert DERIVED_PINS_BY_KEY["tempseg-merged"].converter_version == mt.MERGER_VERSION
