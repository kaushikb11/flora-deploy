"""A partly certified ``balanced``: segment and caption certified for ``human`` on L40S / H100!,
other embodiments not; the rewards are the reference by identity (exact's own engine) on
exact's GPU, and not certified on any other. Fake certificates are written by
``validate.certify`` into a temporary registry (``certkey.CERT_DIR`` patched); none is ever
committed.

Covers: the opt-in rule (fully covered needs no ``allow_uncertified``, partly covered names
each gap, none keeps the old message), each command's plain-words status (plan, doctor, deploy,
results) and the README blocks ``scripts/wave_h_finalize.py`` regenerates."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.config import receipts as R
from flora_deploy.config.hashing import kind_modes
from flora_deploy.config.schema import ConfigError, parse_config
from flora_deploy.validate import certify as C
from flora_deploy.validate import certkey
from tests.fakes.certrecords import (
    STAGE_SETS,
    cold_witness,
    isolated,
    key,
    lab_for,
    rec,
    segs,
)
from tests.unit.test_wave_h_finalize import REPO, fin

runner = CliRunner()
STAMP = "2026-10-22T00:00:00Z"
# Recorded receipts: their content digests are the pinned ones (``receipts.derived_content``).
WEIGHTS = {"derived": {"tempseg_merged": "a" * 64, "captioner_bf16": "b" * 64}}
# reward.terminal on a GPU other than exact's (H100!): not the reference, so not certified
H100 = {"reward.terminal": {"gpu": "H100"}}
REWARDS = ("reward.terminal", "reward.dense")  # (reward.rubric needs pipeline.rubric)


def config(stages=("segment", "caption"), embodiment="human", **extra) -> dict:
    return {"project": "demo", "license_ack": "research-only",
            "modal": {"max_gpus": 6, "budget_usd": 100}, "preset": "balanced",
            "input": {"format": "lerobot_v3", "repo": "org/ds",
                      "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"},
            "pipeline": {"stages": list(stages), "embodiment": embodiment,
                         "episode_instruction": {"source": "dataset_task"}},
            "weights": WEIGHTS, "output": {"sink": "s3://b/p"}, **extra}


def content() -> dict[str, str]:
    got = R.derived_content(parse_config(config(), allow_uncertified=True))
    assert set(got) == {"tempseg_merged", "captioner_bf16"}  # both pinned
    return got


def write_cert(cert_dir: Path, stage: str, embodiment: str = "human", *, noise: bool = True,
               stamp: str = STAMP) -> dict:
    """A fake ``balanced`` certificate for ``stage`` on its pool GPU, covering ``embodiment``,
    keyed to the stack as the code stands (real image fingerprint, pinned weights)."""
    cfg = parse_config(config(stages=("segment", "caption", "reward.terminal"),
                              embodiment=embodiment), allow_uncertified=True)
    mk = STAGE_SETS[stage]
    gpu = {"segment": "L40S"}.get(stage, "H100!")
    modes = kind_modes(cfg, stage)
    doc = key("balanced", stage, gpu, fp=None, modes=modes, content=content())
    lab = lab_for(stage)
    bal = mk("balanced", 10, fp=None, cert_key=doc, embodiment=embodiment)
    ex = mk("exact", 10, fp=None, embodiment=embodiment)
    nz = mk("exact", 5, fp=None, embodiment=embodiment) if noise else None
    wit = cold_witness(bal) if stage in C.L0_STAGES else None
    extra: dict = {}
    if stage == "caption":  # v4: stage-isolated evidence (oracle segments, digests)
        bal, wit, extra = isolated(ex, bal, wit, nz, directory=cert_dir.parent)
    cert = C.certify_stage(
        stage, exact=ex, balanced=bal, witness=wit, noise=nz,
        engine_spec=doc, gpu=gpu, l0=lab if stage in C.L0_STAGES else None,
        generated_at=stamp, **extra)
    C.write_certificate(cert, cert_dir)
    return cert


@pytest.fixture
def registry(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "certificates"
    d.mkdir()
    monkeypatch.setattr(certkey, "CERT_DIR", d)
    return d


@pytest.fixture
def human_certs(registry) -> dict[str, dict]:
    """The expected first certificate: segment and caption, human, L40S / H100!."""
    return {s: write_cert(registry, s) for s in ("segment", "caption")}


def refusal(d: dict) -> str:
    with pytest.raises(ConfigError) as e:
        parse_config(d, derived_content=content())
    return " ".join(str(e.value).split())


# ---------------------------------------------------------------------------------------------
# the opt-in rule


def test_fully_covered_needs_no_opt_in(human_certs):
    cfg = parse_config(config(), derived_content=content())
    assert cfg.preset == "balanced" and not cfg.allow_uncertified


def test_partly_covered_names_each_gap(human_certs):
    msg = refusal(config(stages=("segment", "caption", "reward.terminal"), resources=H100))
    assert ("not certified for every stage of this config (certified: segment, caption)"
            in msg)
    assert "reward.terminal on H100: no certificate" in msg
    assert "Add `allow_uncertified: true`" in msg and "segment on L40S" not in msg
    msg = refusal(config(stages=("segment", "caption", "reward.terminal"), embodiment="yam",
                         resources=H100))
    assert ("reward.terminal on H100: no certificate; embodiment yam: segment/caption "
            "certified for human only") in msg
    msg = refusal(config(stages=("segment", "caption", "reward.terminal"), embodiment="yam"))
    assert "(reference: reward.terminal)" in msg and "reward.terminal on" not in msg
    # one boolean opts in to the rest
    ok = parse_config(config(stages=("segment", "caption", "reward.terminal"), resources=H100,
                             allow_uncertified=True), derived_content=content())
    assert ok.allow_uncertified


def test_provisional_and_missing_weights_are_named(registry):
    write_cert(registry, "segment", noise=False)  # provisional: does not count
    write_cert(registry, "caption")
    msg = refusal(config())
    assert "segment on L40S: only a provisional certificate" in msg
    with pytest.raises(ConfigError, match="segment on L40S: certificate not checked without "
                                          "the prepared weights"):
        parse_config(config())  # no derived content: fail closed, and say why


def test_no_certificate_keeps_todays_message(registry):
    msg = refusal(config())
    assert ("preset 'balanced' has no accuracy certificate for segment on L40S, caption on "
            "H100!") in msg
    assert "--allow-uncertified" in msg


def test_template_for_human_data_needs_no_opt_in(human_certs):
    text = (resources.files("flora_deploy.config") / "templates" / "balanced_mp4_dir.yaml"
            ).read_text()
    assert "certificates yet" not in text and "remove the line" in text
    d = yaml.safe_load(text)
    assert d["pipeline"]["stages"] == ["segment", "caption"]
    assert d["pipeline"]["embodiment"] == "human"
    d.pop("allow_uncertified")
    d["weights"] = WEIGHTS
    assert parse_config(d, derived_content=content()).preset == "balanced"


# ---------------------------------------------------------------------------------------------
# what users see


def cfg_file(tmp_path: Path, **kw) -> Path:
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(config(**kw)))
    return p


def test_plan_recommends_the_certified_option(human_certs, tmp_path):
    p = cfg_file(tmp_path)
    r = runner.invoke(app, ["plan", "-c", str(p), "--video-hours", "1", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    best = d["options"][d["recommended"] - 1]
    assert best["preset"] == "balanced" and best["certification"] == "certified"
    assert {k: c["label"] for k, c in best["certificates"].items()} == {
        "segment": "certified (human, L40S)", "caption": "certified (human, H100!)"}
    out = runner.invoke(app, ["plan", "-c", str(p), "--video-hours", "1"],
                        env={"COLUMNS": "80"}).output
    assert "(recommended): ok, certified" in out
    assert ("certificates: segment certified (human, L40S); caption certified (human, H100!)"
            in " ".join(out.split()))


def test_plan_shows_partly_certified(human_certs, tmp_path):
    p = cfg_file(tmp_path, stages=("segment", "caption", "reward.terminal"), resources=H100,
                 allow_uncertified=True)
    out = " ".join(runner.invoke(app, ["plan", "-c", str(p), "--video-hours", "1"],
                                 env={"COLUMNS": "80"}).output.split())
    assert "ok, partly certified" in out
    assert "reward.terminal not certified" in out and "segment certified (human, L40S)" in out
    assert "Gate" not in out


def doctor_checks(p: Path):
    from flora_deploy.cli import doctor as d

    cfg = R.load_certified(p)[0]
    return {c.name: c for c in d.check_certificates(d.RealProbes(), cfg)}


def test_doctor_says_each_stage_in_plain_words(human_certs, tmp_path):
    c = doctor_checks(cfg_file(tmp_path, stages=("segment", "caption", "reward.terminal"),
                               embodiment="yam", allow_uncertified=True))
    assert c["certificates.caption,segment"].message.startswith(
        "caption, segment: certified for human, not for yam; runs uncertified")
    assert c["certificates.reward.terminal"].message == (
        "reward.terminal on H100!: reference (NVIDIA's own engine, no certificate needed)")
    assert all(x.status == "pass" for x in c.values())
    c = doctor_checks(cfg_file(tmp_path, stages=("segment", "caption", "reward.terminal"),
                               resources=H100, allow_uncertified=True))
    assert c["certificates.segment"].message == "segment: certified (human, L40S)"
    assert c["certificates.caption"].message == "caption: certified (human, H100!)"
    assert c["certificates.reward.terminal"].message.startswith(
        "reward.terminal: not certified; runs uncertified")
    assert "allow_uncertified" not in c  # still needed for reward.terminal on H100
    # fully covered but still opted in: say the opt-in is no longer needed
    c = doctor_checks(cfg_file(tmp_path, allow_uncertified=True))
    assert c["allow_uncertified"].status == "warn"
    assert "every stage is certified for human" in c["allow_uncertified"].message
    for x in c.values():
        assert "Gate" not in x.message and "D16" not in x.message


def test_doctor_config_fix_names_the_opt_in(human_certs, tmp_path):
    from flora_deploy.cli import doctor as d

    chk, cfg = d.check_config(cfg_file(tmp_path, stages=("segment", "caption", "reward.terminal"),
                                       resources=H100))
    assert cfg is None and chk.status == "fail"
    assert "reward.terminal on H100: no certificate" in chk.message
    assert "allow_uncertified: true" in chk.fix and "no certificates yet" not in chk.fix


def test_deploy_dry_run_shows_each_stage(human_certs, tmp_path):
    def dry_run(**kw) -> str:
        p = cfg_file(tmp_path, stages=("segment", "caption", "reward.terminal"), **kw)
        r = runner.invoke(app, ["deploy", "-c", str(p), "--dry-run"], env={"COLUMNS": "200"})
        assert r.exit_code == 0, r.output
        return " ".join(r.output.split())

    out = dry_run(resources=H100, allow_uncertified=True)
    assert "certificate segment on L40S: certified (human, L40S) - cert_" in out
    assert "certificate caption on H100!: certified (human, H100!)" in out
    assert "certificate reward.terminal on H100: not certified" in out
    assert "Gate" not in out
    out = dry_run()  # no opt-in: segment/caption certified, the rewards the reference
    assert "certificate reward.terminal on H100!: reference (NVIDIA's own engine" in out


def test_results_say_certified_in_plain_words_with_the_certificate_id(human_certs):
    from flora_deploy.cli.results import _certified, label_table

    cid = human_certs["caption"]["id"]
    ok = rec("u1", "caption", "balanced", segments=segs([(0.0, 1.0)], ["pick"]), fp=None,
             embodiment="human")
    ok = replace(ok, certified=True, uncertified_reason=None,
                 provenance=replace(ok.provenance, certificate=cid))
    yam = rec("u2", "caption", "balanced", segments=segs([(0.0, 1.0)], ["pick"]), fp=None,
              embodiment="yam")
    yam = replace(yam, uncertified_reason="uncovered_pool")
    plain = rec("u3", "caption", "balanced", fp=None)
    assert _certified(ok) == f"yes (certificate {cid[:13]})"
    assert _certified(yam) == "no (not certified for yam)"
    assert _certified(plain) == "no (no certificate for this stage)"
    exact = replace(plain, certified=True, uncertified_reason=None)
    assert _certified(exact) == "yes (reference)"
    cols, rows = label_table("caption", [ok, yam])
    assert cols[-1] == "certified" and rows[0][-1].startswith("yes (certificate cert_")
    assert json.loads(ok.to_json())["provenance"]["certificate"] == cid  # the record carries it


# ---------------------------------------------------------------------------------------------
# README blocks (scripts/wave_h_finalize.py)


def blocks(text: str) -> dict[str, str]:
    """The README's generated certification block(s) present in ``text`` (today only
    ``allow-uncertified``; the ``certified`` status block is retired)."""
    return {n: text.split(f"<!-- wave-h:{n} -->\n")[1].split(f"<!-- /wave-h:{n} -->")[0]
            for n in ("certified", "allow-uncertified") if f"<!-- wave-h:{n} -->\n" in text}


def no_certificates_readme(tmp_path: Path) -> str:
    """The committed README with its certification block as for an empty registry."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    return fin.certification_blocks(readme, [tmp_path / "none"])


def test_readme_blocks_as_committed_match_the_committed_certificates(tmp_path):
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    # what scripts/wave_h_finalize.py writes for the committed registry (idempotent)
    assert fin.certification_blocks(readme, [certkey.CERT_DIR]) == readme
    assert "allow_uncertified: true           # required for balanced today (see the note below)" \
        in readme
    assert "<!-- wave-h:certified -->" not in readme  # retired
    b = blocks(no_certificates_readme(tmp_path))
    assert list(b) == ["allow-uncertified"]
    assert "segment and caption records currently say `certified: false`" in b["allow-uncertified"]
    assert "`--allow-uncertified` when you export" in b["allow-uncertified"]


def test_readme_blocks_follow_the_certificates(human_certs, registry, tmp_path):
    readme = no_certificates_readme(tmp_path)
    new = fin.certification_blocks(readme, [registry])
    b = blocks(new)
    assert b["allow-uncertified"] == ("Every `balanced` stage is certified for `human` data; "
                                      "remove `allow_uncertified`.\n")
    assert "cert_" not in new.replace(readme, "")
    # only the block changed, and a second pass changes nothing
    def outside(t: str) -> str:
        for name, body in blocks(t).items():
            t = t.replace(f"<!-- wave-h:{name} -->\n{body}", f"<!-- wave-h:{name} -->\n")
        return t

    assert outside(new) == outside(readme)
    assert fin.certification_blocks(new, [registry]) == new
    # a newer provisional certificate for the same key decides: back to the uncertified note
    later = tmp_path / "later"
    write_cert(later, "segment", noise=False, stamp="2026-10-23T00:00:00Z")
    write_cert(later, "caption", noise=False, stamp="2026-10-23T00:00:00Z")
    back = fin.certification_blocks(new, [registry, later])
    assert back == readme


# ---------------------------------------------------------------------------------------------
# reference by identity: balanced's rewards run exact's own engine


def modes_of(stages=REWARDS) -> dict:
    cfg = parse_config(config(stages=stages), allow_uncertified=True)
    return {k: kind_modes(cfg, k) for k in stages}


def test_rewards_are_the_reference_today_and_segment_caption_never():
    modes = modes_of()
    for kind in REWARDS:
        assert certkey.reference_identical("balanced", kind, "H100!", modes=modes), kind
    assert certkey.reference_identical("balanced", "reward.rubric", "H100!")
    for kind, gpu in (("segment", "L40S"), ("caption", "H100!")):
        assert not certkey.reference_identical("balanced", kind, gpu, derived_content=content())
    pools = parse_config(config(stages=REWARDS)).pools()
    st = certkey.status("balanced", pools, embodiment="yam", modes=modes)
    assert {k: v["status"] for k, v in st.items()} == dict.fromkeys(REWARDS, "reference")


def test_a_rewards_only_config_needs_no_opt_in(registry):
    cfg = parse_config(config(stages=REWARDS, embodiment="yam"))
    assert cfg.preset == "balanced" and not cfg.allow_uncertified
    msg = refusal(config(stages=REWARDS, resources=H100))
    assert "no accuracy certificate for reward.terminal on H100 (" in msg  # dense: reference


def test_any_divergence_from_exact_is_not_the_reference(monkeypatch):
    from flora_deploy.config.presets import get_preset

    modes = modes_of()
    kind = "reward.terminal"
    eng = get_preset("balanced").engines[kind]
    for e in (replace(eng, engine_args={"max_num_seqs": 4}), replace(eng, image="img_hfref")):
        assert not certkey.reference_identical("balanced", kind, "H100!", modes=modes,
                                               engines={kind: e})
    assert not certkey.reference_identical("balanced", kind, "H100", modes=modes)  # GPU
    real = certkey.key_document

    def drifted(preset, k, gpu, **kw):  # the balanced side's logic digest drifts
        doc = real(preset, k, gpu, **kw)
        return {**doc, "logic_digest": "0" * 64} if preset == "balanced" else doc

    monkeypatch.setattr(certkey, "key_document", drifted)
    assert not certkey.reference_identical("balanced", kind, "H100!", modes=modes)
    with pytest.raises(ConfigError, match="no accuracy certificate for reward.terminal on H100!"):
        parse_config(config(stages=REWARDS))


def test_a_matching_fail_certificate_overrides_reference_by_identity(registry):
    """Review 13 S3-7: a ``fail`` measured under the very key the identity argument covers
    revokes it; the kind then needs a certificate like any other."""
    from tests.fakes.certrecords import terminal_set

    kind = "reward.terminal"
    modes = modes_of()
    pools = parse_config(config(stages=REWARDS)).pools()
    assert kind in certkey.reference_kinds("balanced", pools, embodiment="yam", modes=modes)
    doc = key("balanced", kind, "H100!", fp=None, modes=modes[kind], content=content())
    cert = C.certify_stage(
        kind, exact=terminal_set("exact", 10, fp=None, embodiment="yam"),
        balanced=terminal_set("balanced", 10, score=2, es=2.0, fp=None, cert_key=doc,
                              embodiment="yam"),
        noise=terminal_set("exact", 5, fp=None, embodiment="yam"), engine_spec=doc,
        gpu="H100!", generated_at=STAMP)
    assert cert["verdict"] == "fail"
    C.write_certificate(cert, registry)
    ref = certkey.reference_kinds("balanced", pools, embodiment="yam", modes=modes)
    assert kind not in ref and "reward.dense" in ref
    st = certkey.status("balanced", pools, embodiment="yam", modes=modes)
    assert st[kind] == {"gpu": "H100!", "status": "fail", "certificate": cert["id"]}
    assert st["reward.dense"]["status"] == "reference"
    from flora_deploy.config.presets import find_certificates

    found = find_certificates("balanced", pools, modes=modes, embodiment="yam")
    assert kind not in found and found["reward.dense"].verdict == "reference"
    # another embodiment's pool has no matching fail: still the reference there
    assert kind in certkey.reference_kinds("balanced", pools, embodiment="dex", modes=modes)
