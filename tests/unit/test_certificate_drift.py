"""Certificate drift guard (``validate/certfreeze.py``, ``docs/certificate-freeze.md``).

``test_committed_certificates_still_match`` is the guard: every certificate committed under
``src/flora_deploy/config/certificates/`` must still match the key the current code computes,
or the PR fails with a diff of the key documents and the choice (revert, or re-certify). With
no certificate committed it passes trivially. Every other test drives the mechanism with
synthetic certificates built here, in ``tmp_path``; none is ever committed.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from flora_deploy.config.logic import package_root
from flora_deploy.validate import certfreeze as F
from flora_deploy.validate import certkey as K
from flora_deploy.workers_impl.certstamp import pinned_content

ROOT = Path(__file__).resolve().parents[2]
STAMP = "2026-11-01T00:00:00Z"
LATER = "2026-11-02T00:00:00Z"
SEG_METRICS = ("iou_divergence", "boundary_error_s", "count_mismatch", "label_disagreement")


# ---------------------------------------------------------------------------------------------
# The guard


def test_committed_certificates_still_match():
    """Fails (message first, no traceback) when a committed certificate drifted."""
    report = F.check()
    for f in report.by("warning"):
        warnings.warn(f"certificate {f.cert} [{f.status}]: {f.message}", stacklevel=1)
    if not report.ok:
        pytest.fail(F.format_report(report), pytrace=False)


# ---------------------------------------------------------------------------------------------
# Synthetic certificates (tmp_path only)


def current_doc(kind="segment", gpu="L40S", preset="balanced", **kw):
    """The key document a worker would stamp now (pinned derived content digests)."""
    kw.setdefault("derived_content", pinned_content(preset, kind))
    return K.key_document(preset, kind, gpu, **kw)


def make_cert(doc, *, verdict="certified", gpu="L40S", pools=("human",), stamp=STAMP,
              preset=None, kind=None, tolerance=1.5):
    """A certificate body for ``doc`` that passes ``certkey.policy_problems`` (when certified)."""
    zeros = dict.fromkeys(SEG_METRICS, 0.0)
    body = {
        "schema": K.CERT_SCHEMA,
        "generator": {"name": K.GENERATOR, "version": K.GENERATOR_VERSIONS[-1],
                      "framework": "test"},
        "generated_at": stamp, "verdict": verdict, "reasons": [],
        "unlocks": verdict == "certified",
        "preset": preset or doc["preset"], "stage": doc["kind"], "kind": kind or doc["kind"],
        "gpu": gpu, "covered_pools": list(pools), "stage_hash": "b" * 64,
        "engine_spec_hash": K.sha256_hex(doc),
        "engine": {**{k: doc["engine"].get(k) for k in ("name", "version", "image",
                                                        "image_fingerprint", "patches")},
                   "vllm": doc.get("vllm_engine")},
        "modes": dict(doc.get("modes") or {}),
        "logic": {"version": doc["logic_version"], "digest": doc["logic_digest"]},
        "pins": {k: v["revision"] for k, v in sorted(doc["nvidia_pins"].items())},
        "derived_weights": dict(doc["derived_weights"]),
        "metrics": {"candidate": {p: {"n": 12, "means": zeros} for p in pools},
                    "noise": {"n": 5, "means": zeros}, "coverage": 1.0, "missing": 0},
        "thresholds": {"tolerance": tolerance, "min_samples": 10, "min_noise_samples": 5,
                       "max_missing": K.MAX_MISSING, "slack": {},
                       "noise_floor": {m: K.NOISE_FLOOR[m] for m in SEG_METRICS},
                       "absolute_floor": {"count_mismatch": K.ABSOLUTE_FLOOR["count_mismatch"]}},
    }
    body["id"] = K.certificate_id(body)
    return body


def commit(d: Path, cert, doc=None, root=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cert['id']}.json").write_text(json.dumps(cert, indent=1, sort_keys=True))
    if doc is not None:
        F.write_freeze(cert, doc, d, root)
    return cert


def errors(rep):
    return [f for f in rep.by("error")]


def statuses(rep, severity=None):
    return sorted(f.status for f in rep.findings if severity in (None, f.severity))


def text(rep):
    return F.format_report(rep, verbose=True)


def assert_choice(msg: str):
    assert "revert" in msg and "re-certify" in msg
    assert "scripts/wave_h.py --plan certified" in msg and "flora certify" in msg
    assert "scripts/cert_freeze.py --lock" in msg and "docs/certificate-freeze.md" in msg


# ---------------------------------------------------------------------------------------------
# Matching and the trivial case


def test_no_certificates_passes_trivially(tmp_path):
    rep = F.check(tmp_path / "certificates")
    assert rep.ok and rep.certificates == 0 and not rep.findings
    assert "nothing is frozen" in F.format_report(rep)


def test_matching_certificate_is_ok_and_selected_by_matching(tmp_path):
    doc = current_doc()
    c = commit(tmp_path, make_cert(doc), doc)
    rep = F.check(tmp_path)
    assert rep.ok and statuses(rep) == ["ok"], text(rep)
    # the same answer as the registry's own lookup, end to end
    assert F.read_freeze(tmp_path, c)["modules"] == F.module_hashes("segment")


def test_missing_lock_is_a_warning_and_lock_backfills_it(tmp_path):
    doc = current_doc("caption", "H100!")
    commit(tmp_path, make_cert(doc, gpu="H100!"))
    rep = F.check(tmp_path)
    assert rep.ok and "no_lock" in statuses(rep, "warning")
    from scripts import cert_freeze

    assert cert_freeze.main(["--lock", "--cert-dir", str(tmp_path)]) == 0
    assert statuses(F.check(tmp_path)) == ["ok"]


# ---------------------------------------------------------------------------------------------
# Drift, with the diff of the key documents


@pytest.fixture
def drifted_root(tmp_path):
    """A copy of the package source with one code edit on the vLLM video path."""
    root = tmp_path / "old_src"
    shutil.copytree(package_root(), root, ignore=shutil.ignore_patterns("__pycache__"))
    p = root / "engines" / "microbatch.py"
    p.write_text(p.read_text() + "\n_OLD_BEHAVIOUR = 1\n")
    return root


def test_code_edit_names_the_changed_module(tmp_path, drifted_root):
    old = current_doc(logic_root=drifted_root)
    certs = tmp_path / "certs"
    commit(certs, make_cert(old), old, root=drifted_root)
    rep = F.check(certs)
    assert not rep.ok
    (f,) = errors(rep)
    assert f.status == "drift"
    msg = F.format_report(rep)
    assert msg.startswith("certificate drift: 1 error(s)")
    assert "logic_digest:" in msg
    assert "engines/microbatch.py: code changed" in msg
    assert "modal_app/balanced_app.py" not in msg  # only the module that changed
    assert_choice(msg)


def test_comment_and_docstring_edits_do_not_drift(tmp_path):
    root = tmp_path / "src"
    shutil.copytree(package_root(), root, ignore=shutil.ignore_patterns("__pycache__"))
    p = root / "engines" / "microbatch.py"
    p.write_text(p.read_text() + "\n# a comment only\n")
    doc = current_doc(logic_root=root)
    assert doc == current_doc()


def test_engine_kwargs_pins_and_derived_digests_are_diffed(tmp_path):
    old = copy.deepcopy(current_doc())
    old["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 31
    old["nvidia_pins"]["tempseg"]["revision"] = "0" * 40
    old["derived_weights"]["tempseg_merged"] = "e" * 64
    old["decode"]["av"] = "15.0.0"
    commit(tmp_path, make_cert(old), old)
    msg = F.format_report(F.check(tmp_path))
    now = current_doc()["vllm_engine"]["engine_kwargs"]["max_num_seqs"]
    assert f"vllm_engine.engine_kwargs.max_num_seqs: 31 -> {now}" in msg
    assert 'nvidia_pins.tempseg.revision: "' + "0" * 40 in msg
    assert 'derived_weights.tempseg_merged: "' + "e" * 64 in msg
    assert 'decode.av: "15.0.0" -> ' in msg
    assert "logic_digest" not in msg
    assert_choice(msg)


def test_without_lock_the_recorded_fields_are_diffed(tmp_path, drifted_root):
    old = copy.deepcopy(current_doc(logic_root=drifted_root))
    old["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 31
    commit(tmp_path, make_cert(old))  # no freeze lock
    rep = F.check(tmp_path, git_history=False)
    msg = F.format_report(rep)
    assert not rep.ok
    assert "key fields the certificate records" in msg
    now = current_doc(logic_root=drifted_root)["vllm_engine"]["engine_kwargs"]["max_num_seqs"]
    assert f"vllm_engine.engine_kwargs.max_num_seqs: 31 -> {now}" in msg
    assert "logic_digest:" in msg and "per-module breakdown unavailable" in msg


def test_unrecorded_field_drift_is_still_caught(tmp_path):
    old = copy.deepcopy(current_doc())
    old["decode"]["numpy"] = "1.0.0"  # not in the certificate body
    commit(tmp_path, make_cert(old))
    msg = F.format_report(F.check(tmp_path, git_history=False))
    assert "key drift" in msg and "does not record" in msg


def test_repinned_derived_weight_drifts(tmp_path):
    """The workers load only the pinned content: a re-pin invalidates the certificate."""
    old = current_doc(derived_content={"tempseg_merged": "d" * 64})
    commit(tmp_path, make_cert(old), old)
    msg = F.format_report(F.check(tmp_path))
    assert "derived_weights.tempseg_merged" in msg


def test_image_fingerprint_drift(tmp_path):
    doc = current_doc()
    commit(tmp_path, make_cert(doc), doc)
    rep = F.check(tmp_path, image_fingerprint=lambda name: "f" * 64)
    assert "engine.image_fingerprint" in F.format_report(rep)


# ---------------------------------------------------------------------------------------------
# Verdicts and the newest-wins rule


def test_provisional_drift_is_a_warning(tmp_path):
    old = copy.deepcopy(current_doc())
    old["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 31
    commit(tmp_path, make_cert(old, verdict="provisional"), old)
    rep = F.check(tmp_path)
    assert rep.ok and "drift" in statuses(rep, "warning")


def test_newer_fail_revokes_and_is_not_checked(tmp_path):
    old = copy.deepcopy(current_doc())
    old["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 31
    commit(tmp_path, make_cert(old), old)
    commit(tmp_path, make_cert(old, verdict="fail", stamp=LATER), old)
    rep = F.check(tmp_path)
    assert rep.ok and statuses(rep) == ["revoked", "superseded"], text(rep)


def test_tie_goes_to_fail(tmp_path):
    doc = current_doc()
    commit(tmp_path, make_cert(doc), doc)
    commit(tmp_path, make_cert(doc, verdict="fail"), doc)
    assert statuses(F.check(tmp_path)) == ["revoked", "superseded"]


def test_newest_certified_decides_over_older_provisional(tmp_path):
    doc = current_doc()
    old = commit(tmp_path, make_cert(doc, verdict="provisional"), doc)
    new = commit(tmp_path, make_cert(doc, stamp=LATER), doc)
    rep = F.check(tmp_path)
    by_id = {f.cert: f.status for f in rep.findings}
    assert by_id == {old["id"]: "superseded", new["id"]: "ok"}


def test_newest_wins_per_pool(tmp_path):
    doc = current_doc()
    both = commit(tmp_path, make_cert(doc, pools=("human", "yam")), doc)
    commit(tmp_path, make_cert(doc, verdict="fail", pools=("yam",), stamp=LATER), doc)
    rep = F.check(tmp_path)
    assert rep.ok
    assert {f.cert for f in rep.findings if f.status == "ok"} == {both["id"]}
    assert F.deciding(K.load_certificates(tmp_path))[both["id"]] == ["human"]


def test_stale_certificate_after_recertification_must_be_deleted(tmp_path):
    """Another key for the same slot: both are live; the drifted one fails until deleted."""
    old = copy.deepcopy(current_doc())
    old["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 31
    stale = commit(tmp_path, make_cert(old), old)
    doc = current_doc()
    commit(tmp_path, make_cert(doc, stamp=LATER), doc)
    (f,) = errors(F.check(tmp_path))
    assert f.cert == stale["id"] and "delete the superseded certificate" in f.details[-1]
    (tmp_path / f"{stale['id']}.json").unlink()
    assert F.check(tmp_path).ok


# ---------------------------------------------------------------------------------------------
# Orphans, invalid files, policy


@pytest.mark.parametrize(("change", "needle"), [
    ({"gpu": "T5"}, "not a known Modal GPU"),
    ({"gpu": "T4"}, "no longer usable"),
    ({"covered_pools": ["octopus"]}, "no longer exist"),
    ({"preset": "budget"}, "not an available preset"),
    ({"kind": "segcap"}, "is not in preset"),
])
def test_orphan_certificates(tmp_path, change, needle):
    doc = current_doc()
    body = {**make_cert(doc), **change}
    body["metrics"]["candidate"] = {p: {"n": 12, "means": dict.fromkeys(SEG_METRICS, 0.0)}
                                    for p in body["covered_pools"]}
    body["id"] = K.certificate_id(body)
    commit(tmp_path, body)
    (f,) = errors(F.check(tmp_path))
    assert f.status == "orphan" and needle in " ".join(f.details)


def test_gpu_without_engine_limits_cannot_compute_a_key(tmp_path):
    doc = current_doc()
    body = {**make_cert(doc), "gpu": "A100-80GB", "kind": "caption"}  # no captioner limits
    body["id"] = K.certificate_id(body)
    commit(tmp_path, body)
    (f,) = errors(F.check(tmp_path))
    assert f.status == "drift" and "cannot be computed" in f.message


def test_hand_edited_certificate_is_invalid(tmp_path):
    doc = current_doc()
    c = make_cert(doc)
    c["thresholds"]["tolerance"] = 1.9  # id no longer its content address
    commit(tmp_path, c)
    (f,) = errors(F.check(tmp_path))
    assert f.status == "invalid" and "content address" in f.message


def test_certificate_outside_policy(tmp_path):
    doc = current_doc()
    commit(tmp_path, make_cert(doc, tolerance=3.0), doc)
    (f,) = errors(F.check(tmp_path))
    assert f.status == "policy"


def test_orphan_lock_is_a_warning_and_lock_prunes_it(tmp_path):
    doc = current_doc()
    c = commit(tmp_path, make_cert(doc), doc)
    (tmp_path / f"{c['id']}.json").unlink()
    assert "orphan_lock" in statuses(F.check(tmp_path), "warning")
    from scripts import cert_freeze

    cert_freeze.main(["--lock", "--cert-dir", str(tmp_path)])
    assert not F.check(tmp_path).findings


# ---------------------------------------------------------------------------------------------
# flora certify writes the lock; the script; CPU only


def test_flora_certify_writes_the_freeze_lock(tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli import app
    from tests.fakes.certrecords import L0_PASS, key, segment_set
    from tests.unit.test_certify import _balanced_yaml, _jsonl

    content = pinned_content("balanced", "segment")
    doc = key("balanced", "segment", "L40S", fp=None, content=content)
    ex = _jsonl(tmp_path / "exact.jsonl", segment_set("exact", 12, fp=None))
    bal = _jsonl(tmp_path / "bal.jsonl", segment_set("balanced", 12, fp=None, cert_key=doc))
    nz = _jsonl(tmp_path / "noise.jsonl", segment_set("exact", 5, fp=None))
    lab = tmp_path / "lab.json"
    lab.write_text(json.dumps(L0_PASS))
    out = tmp_path / "certs"
    r = CliRunner().invoke(app, [
        "certify", "--exact", str(ex), "--balanced", str(bal), "--noise", str(nz), "--l0",
        str(lab), "-c", str(_balanced_yaml(tmp_path)), "--out-dir", str(out),
        "--derived-content", f"tempseg_merged={content['tempseg_merged']}", "--json"])
    assert r.exit_code == 0, r.output
    (cert,) = json.loads(r.output)
    assert cert["verdict"] == "certified"
    assert F.read_freeze(out, cert)["key_document"] == doc
    rep = F.check(out)
    assert rep.ok and statuses(rep) == ["ok"], text(rep)


def test_script_check_and_listing(tmp_path, capsys):
    from scripts import cert_freeze

    old = copy.deepcopy(current_doc())
    old["vllm_engine"]["engine_kwargs"]["max_num_seqs"] = 31
    c = commit(tmp_path, make_cert(old), old)
    assert cert_freeze.main(["--check", "--cert-dir", str(tmp_path), "--github"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("certificate drift: 1 error(s)")
    assert "::error title=Certificate drift (drift)::" in out
    assert cert_freeze.main(["--cert-dir", str(tmp_path)]) == 0
    listing = capsys.readouterr().out
    assert "## balanced / segment / L40S (default pool)" in listing
    assert f"`{c['id']}` certified" in listing and "DOES NOT match" in listing
    assert "`engines/microbatch.py`" in listing
    assert "## balanced / caption / H100! (default pool)" in listing
    assert "**no certificate**" in listing


def test_drift_check_needs_no_nvidia_code(tmp_path):
    """The key needs NVIDIA pins only as revisions and digests: no torch, vLLM or NVIDIA
    module is imported to compute it (so the guard runs in the CPU test job)."""
    doc = current_doc()
    commit(tmp_path, make_cert(doc), doc)
    code = ("import sys\nfrom pathlib import Path\nfrom flora_deploy.validate import certfreeze"
            f"\nr = certfreeze.check(Path({str(tmp_path)!r}))\nassert r.ok, r\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'vllm', 'transformers', 'batch_utils', 'modal'))\nprint(bad)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
