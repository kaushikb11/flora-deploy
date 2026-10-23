"""A derived-weight receipt counts for certificates only once its content is checked against the
pin (``config.receipts.derived_content``): a made-up receipt never makes a dry-run say
"certified" (newcomer 2, item 2)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli.init import template_text
from flora_deploy.config import receipts as rc
from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
from flora_deploy.config.schema import load_config

pytestmark = pytest.mark.real_receipts

FAKE = {"captioner_bf16": "a" * 64, "tempseg_merged": "b" * 64}


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "flora.yaml"
    p.write_text(template_text("balanced_lerobot"))
    return p


def test_made_up_receipt_does_not_count(cfg_path):
    cfg = load_config(cfg_path, allow_uncertified=True)
    content = rc.derived_content(cfg, FAKE)
    assert content == {}
    assert set(content.unchecked) == set(rc.needed(cfg))
    assert all("unchecked" in why for why in content.unchecked.values()), content.unchecked


def test_known_receipts_count(cfg_path):
    cfg = load_config(cfg_path, allow_uncertified=True)
    known = {n: sha for sha, c in rc.KNOWN_RECEIPTS.items() for n in rc.needed(cfg)
             if DERIVED_PINS_BY_KEY[rc.pin_key(n)].content_sha256 == c}
    content = rc.derived_content(cfg, known)
    assert content == {n: DERIVED_PINS_BY_KEY[rc.pin_key(n)].content_sha256 for n in known}
    assert not content.unchecked


def test_receipt_read_from_volume_counts_and_is_cached(cfg_path, monkeypatch):
    """A reader (a real deploy: the flora-weights Volume) checks the content, and the result is
    remembered, so the next plan / dry-run on this machine counts it without Modal."""
    cfg = load_config(cfg_path, allow_uncertified=True)
    pinned = {n: DERIVED_PINS_BY_KEY[rc.pin_key(n)].content_sha256 for n in rc.needed(cfg)}
    raws = {n: json.dumps({"name": n}).encode() for n in pinned}
    import hashlib

    shas = {n: hashlib.sha256(r).hexdigest() for n, r in raws.items()}
    monkeypatch.setattr(rc, "check_pinned", lambda name, receipt, **kw: pinned[name])
    content = rc.derived_content(cfg, shas, read_receipt=raws.get)
    assert content == pinned and not content.unchecked
    again = rc.derived_content(cfg, shas)  # no reader: the cache
    assert again == pinned
    # a receipt on the Volume that is not the recorded one does not count
    other = rc.derived_content(cfg, FAKE, read_receipt=raws.get)
    assert other == {} and all("not" in w for w in other.unchecked.values())


def test_dry_run_with_made_up_receipt_is_not_certified(cfg_path):
    args = ["deploy", "--dry-run", "-c", str(cfg_path), "--allow-uncertified"]
    for n, sha in FAKE.items():
        args += ["--receipt", f"{n}={sha}"]
    r = CliRunner().invoke(app, args, env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    seg = [ln for ln in r.output.splitlines() if "certificate segment" in ln]
    assert seg and ": certified" not in seg[0] and "not checked" in seg[0], r.output
