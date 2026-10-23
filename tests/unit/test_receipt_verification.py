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


# --- dry-run and doctor read the Volume receipts too (free, read-only) ------------------------


@pytest.fixture
def volume(cfg_path, monkeypatch, tempseg_pinned):
    """A mocked flora-weights Volume holding one receipt per needed derived weight, each with
    its pinned content: ``(shas, calls)``."""
    import hashlib

    cfg = load_config(cfg_path, allow_uncertified=True)
    pinned = {n: DERIVED_PINS_BY_KEY[rc.pin_key(n)].content_sha256 for n in rc.needed(cfg)}
    raws = {n: json.dumps({"name": n}).encode() for n in pinned}
    calls: list[str] = []

    def read(name):
        calls.append(name)
        return raws.get(name)

    monkeypatch.setattr(rc, "read_volume_receipt", read)
    monkeypatch.setattr(rc, "check_pinned", lambda name, receipt, **kw: pinned[name])
    return {n: hashlib.sha256(r).hexdigest() for n, r in raws.items()}, calls


@pytest.fixture
def offline(monkeypatch):
    """The Volume cannot be read (no network / no credentials): the read calls made."""
    calls: list[str] = []

    def read(name):
        calls.append(name)
        raise ConnectionError("no route to api.modal.com")

    monkeypatch.setattr(rc, "read_volume_receipt", read)
    return calls


def _receipt_args(shas):
    return [a for n, sha in shas.items() for a in ("--receipt", f"{n}={sha}")]


def _recorded(cfg_path, shas):
    p = cfg_path.with_name("recorded.yaml")
    p.write_text(cfg_path.read_text())
    rc.record(p, shas)
    return p


def test_dry_run_reads_volume_receipts_and_caches_them(cfg_path, volume):
    shas, calls = volume
    args = ["deploy", "--dry-run", "-c", str(cfg_path), "--allow-uncertified",
            *_receipt_args(shas)]
    r = CliRunner().invoke(app, args, env={"COLUMNS": "400"})
    assert r.exit_code == 0, r.output
    assert sorted(set(calls)) == sorted(shas), calls
    assert "could not read" not in r.output and "not checked" not in r.output, r.output
    for n, sha in shas.items():  # remembered: the next check on this machine reads nothing
        assert rc.verified_content(sha, n) == DERIVED_PINS_BY_KEY[rc.pin_key(n)].content_sha256
    calls.clear()
    r = CliRunner().invoke(app, args, env={"COLUMNS": "400"})
    assert r.exit_code == 0 and calls == [], r.output


def test_dry_run_volume_unreachable_is_not_checked_with_reason(cfg_path, offline):
    args = ["deploy", "--dry-run", "-c", str(cfg_path), "--allow-uncertified",
            *_receipt_args(FAKE)]
    r = CliRunner().invoke(app, args, env={"COLUMNS": "400"})
    assert r.exit_code == 0, r.output
    assert len(offline) == 1  # unreachable once: the other receipts are not retried
    seg = [ln for ln in r.output.splitlines() if "certificate segment" in ln]
    assert seg and "not checked" in seg[0] and "could not read" in seg[0], r.output
    assert "ConnectionError" in seg[0] and "flora-weights" in seg[0]


def test_emit_reads_no_volume(cfg_path, tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "read_volume_receipt", lambda n: pytest.fail("read the Volume"))
    r = CliRunner().invoke(app, ["deploy", "-c", str(cfg_path), "--allow-uncertified",
                                 "--emit", str(tmp_path / "app.py"), *_receipt_args(FAKE)])
    assert r.exit_code == 0, r.output


def test_doctor_reads_volume_receipts(cfg_path, volume):
    from flora_deploy.cli import doctor as d

    shas, calls = volume
    p = _recorded(cfg_path, shas)
    probes = d.RealProbes()
    check, cfg = d.check_config(p, probes.read_receipt)
    assert check.status == "pass" and sorted(calls) == sorted(shas), (check, calls)
    st = probes.certificate_status(cfg)  # verified and cached by the config check
    assert sorted(calls) == sorted(shas)
    assert not any("not checked" in v or "could not read" in v for v in st.values()), st


def test_doctor_run_checks_passes_the_probes_reader(cfg_path, monkeypatch):
    from flora_deploy.cli import doctor as d

    class Probes:
        def read_receipt(self, name):
            return None

        def __getattr__(self, name):  # every other probe: nothing there
            return lambda *a, **kw: None

    probes = Probes()
    seen = []

    def check_config(path, read_receipt=None):
        seen.append(read_receipt)
        return d.Check("config", "fail", "stop here"), None

    monkeypatch.setattr(d, "check_config", check_config)
    monkeypatch.setattr(d, "check_modal", lambda p, env: [])
    monkeypatch.setattr(d, "check_hf", lambda p: ([], None))
    d.run_checks(cfg_path, probes)
    assert seen == [probes.read_receipt]  # the config check reads the Volume via the probes


def test_doctor_volume_unreachable_says_why(cfg_path, tempseg_pinned, offline):
    from flora_deploy.cli import doctor as d

    p = _recorded(cfg_path, FAKE)
    probes = d.RealProbes()
    check, cfg = d.check_config(p, probes.read_receipt)
    assert check.status == "pass", check
    st = probes.certificate_status(cfg)
    assert len(offline) == 1  # memoized across the config and certificate checks
    unchecked = [v for v in st.values() if "not checked" in v]
    assert unchecked and all("could not read" in v and "ConnectionError" in v
                             for v in unchecked), st
    lines = d.check_certificates(probes, cfg)
    assert any("could not read" in c.message for c in lines), lines
