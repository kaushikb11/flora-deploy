"""validate/certify.py + validate/certkey.py: metrics, pass rule, certificate content, matching.

The review 7 repros (H1-H7, M1, M2, M4, M6) are in ``test_certify_review7.py``."""

from __future__ import annotations

import json
import random
import string

import pytest

from flora_deploy.config import presets
from flora_deploy.config.presets import PoolSettings
from flora_deploy.records import (
    DenseReward,
    Provenance,
    Segment,
    StageRecord,
    TerminalReward,
)
from flora_deploy.validate import certify as C
from flora_deploy.validate import certkey
from tests.fakes.certrecords import (
    BOUNDS,
    CAND_HASH,
    CONTENT,
    FP,
    L0_PASS,
    STAGE_SETS,
    key,
    prov,
    rec,
    segment_set,
    segs,
    terminal_set,
)

STAMP = "2026-10-22T00:00:00Z"
MATCH = {"embodiment": "yam", "derived_content": CONTENT, **FP}


def cert_for(stage="segment", *, n=12, noise=True, n_noise=5, shift=0.0, noise_shift=0.0,
             l0=L0_PASS, gpu="L40S", stamp=STAMP, **kw):
    noise_recs = segment_set("exact", n_noise, noise_shift) if noise else None
    return C.certify_stage(stage, exact=segment_set("exact", n), balanced=segment_set(
        "balanced", n, shift), noise=noise_recs, engine_spec=key("balanced", stage, gpu),
        gpu=gpu, l0=l0, generated_at=stamp, **kw)


# ---------------------------------------------------------------------------------------------
# metrics


def test_alignment_and_segment_metrics():
    a = rec("u", "segment", "exact", segments=segs(BOUNDS))
    b = rec("u", "segment", "balanced", segments=segs([(0.0, 2.0), (2.0, 9.0)]))
    pairs = C.align(a.segments, b.segments)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (2, 1)]  # IoU 4/7 beats 3/7
    m = C.segment_metrics(a, b)
    assert m["count_mismatch"] == 1.0
    assert m["iou_divergence"] == pytest.approx(1 - (1 + 4 / 7) / 3)
    assert m["boundary_error_s"] == pytest.approx((0 + 3 / 2) / 2)
    assert m["label_disagreement"] == 0.0
    assert C.segment_metrics(a, a) == {"iou_divergence": 0.0, "boundary_error_s": 0.0,
                                       "count_mismatch": 0.0, "label_disagreement": 0.0}


def test_caption_metrics_and_similarity_hook():
    a = rec("u", "caption", "exact", segments=segs(BOUNDS, ["pick cup", "place cup", "x"]))
    b = rec("u", "caption", "balanced", segments=segs(BOUNDS[:2], ["pick cup", "place mug"]))
    m = C.caption_metrics(a, b)
    assert m["caption_mismatch"] == pytest.approx(1 - 1 / 3)
    assert m["caption_edit_distance"] == pytest.approx((0 + 2 / 9 + 1) / 3)
    assert C.SIMILARITY_METRIC not in m
    m = C.caption_metrics(a, b, similarity=lambda x, y: 1.0)
    assert m[C.SIMILARITY_METRIC] == pytest.approx(1 / 3)  # the unmatched slot counts fully
    assert C.normalized_edit_distance("", "") == 0.0
    assert C.levenshtein("kitten", "sitting") == 3


def test_reward_metrics():
    t = lambda s, e: TerminalReward(score=s, expected_score=e,  # noqa: E731
                                    probabilities=(0.0, 0.0, 0.0, 0.5, 0.5))
    m = C.terminal_metrics(rec("u", "reward.terminal", "exact", terminal=t(4, 4.5)),
                           rec("u", "reward.terminal", "balanced", terminal=t(5, 4.75)))
    assert m == {"expected_score_abs_delta": 0.25, "score_mismatch": 1.0}
    d = lambda xs: DenseReward(segments=(), per_step=tuple(xs))  # noqa: E731
    m = C.dense_metrics(rec("u", "reward.dense", "exact", dense=d([3, 3, 1, 2])),
                        rec("u", "reward.dense", "balanced", dense=d([3, 3, 2, 1])))
    assert m["per_step_abs_delta"] == 0.5 and m["length_mismatch"] == 0.0
    assert 0 < m["rank_divergence"] < 2
    same = rec("u", "reward.dense", "exact", dense=d([3, 3, 3]))
    assert C.dense_metrics(same, same)["rank_divergence"] == 0.0  # constant and equal
    assert C.spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert C.spearman([1, 1], [1, 2]) is None


# ---------------------------------------------------------------------------------------------
# pass rule


def test_identical_records_with_noise_and_samples_are_certified():
    c = cert_for()
    assert c["verdict"] == "certified" and c["unlocks"] is True and c["reasons"] == []
    assert all(ch["pass"] for ch in c["checks"])
    assert c["metrics"]["noise"]["n"] == 5 and c["metrics"]["coverage"] == 1.0
    assert c["l0"]["status"] == "pass" and c["gate_a"]["greedy_identical"] == 6
    assert c["id"] == certkey.certificate_id(c) and c["id"].startswith("cert_")
    assert c["stage_hash"] == CAND_HASH and c["covered_pools"] == ["yam"]
    assert c["engine_spec_hash"] == certkey.engine_spec_hash(
        "balanced", "segment", "L40S", derived_content=CONTENT, **FP)
    assert c["episodes"]["count"] == 12 and len(c["episodes"]["uid_sha256"]) == 12
    assert c["generated_at"] == STAMP and certkey.policy_problems(c) == []


def test_divergence_above_noise_floor_fails():
    c = cert_for(shift=0.5, noise_shift=0.1)
    assert c["verdict"] == "fail" and "divergence_above_noise_floor" in c["reasons"]
    bad = {ch["metric"] for ch in c["checks"] if ch["pass"] is False}
    assert {"iou_divergence", "boundary_error_s"} <= bad
    # within tolerance x noise: certified
    assert cert_for(shift=0.12, noise_shift=0.1)["verdict"] == "certified"
    # a bounded slack lifts the threshold by its value
    th = C.Thresholds(slack={"boundary_error_s": 0.1})
    t0 = cert_for(noise_shift=0.1)["thresholds"]["per_metric"]["boundary_error_s"]["threshold"]
    t1 = cert_for(noise_shift=0.1, thresholds=th)["thresholds"]["per_metric"][
        "boundary_error_s"]["threshold"]
    assert t1 == pytest.approx(t0 + 0.1)


def test_no_noise_run_is_provisional_and_never_unlocks():
    c = cert_for(noise=False)
    assert c["verdict"] == "provisional" and c["unlocks"] is False
    assert "no_noise_run" in c["reasons"]
    assert all(ch["pass"] is None and ch["threshold"] is None for ch in c["checks"])
    # even a large divergence cannot fail or certify without a noise floor
    assert cert_for(noise=False, shift=3.0)["verdict"] == "provisional"


@pytest.mark.parametrize("kw,reason", [
    ({"n": 9}, "too_few_samples"),
    ({"l0": None}, "no_l0"),
    ({"n_noise": 4}, "too_few_noise_samples"),
])
def test_insufficient_evidence_is_provisional(kw, reason):
    c = cert_for(**kw)
    assert c["verdict"] == "provisional" and reason in c["reasons"]


def test_l0_failure_fails_and_wrong_lab_kind_or_gpu_is_refused():
    bad = json.loads(json.dumps(L0_PASS))
    bad["baseline"]["l0"][0]["grid_equal"] = False
    c = cert_for(l0=bad)
    assert c["verdict"] == "fail" and "l0_failed" in c["reasons"]
    with pytest.raises(C.CertifyError, match="kind"):
        cert_for(l0={**L0_PASS, "kind": "captioner"})
    with pytest.raises(C.CertifyError, match="measured on"):
        cert_for(l0={**L0_PASS, "baseline": {**L0_PASS["baseline"], "gpu": "H100!"}})


def test_rewards_need_no_l0():
    c = C.certify_stage("reward.terminal", exact=terminal_set("exact", 10),
                        balanced=terminal_set("balanced", 10), noise=terminal_set("exact", 5),
                        engine_spec=key("balanced", "reward.terminal", "H100!"), gpu="H100!")
    assert c["l0"] == {"status": "not_applicable"} and c["verdict"] == "certified"


def test_input_consistency_is_checked():
    ex = segment_set("exact", 12)
    spec = key("balanced", "segment", "L40S")
    with pytest.raises(C.CertifyError, match="one stage hash"):
        C.certify_stage("segment", exact=ex, engine_spec=spec, gpu="L40S",
                        balanced=[*segment_set("balanced", 11),
                                  rec("u11", "segment", "balanced", config_hash="e" * 64)])
    with pytest.raises(C.CertifyError, match="not produced on"):
        C.certify_stage("segment", exact=ex, engine_spec=spec, gpu="H100",
                        balanced=segment_set("balanced", 12))
    with pytest.raises(C.CertifyError, match="preset 'exact'"):
        C.certify_stage("segment", exact=segment_set("balanced", 12), engine_spec=spec,
                        gpu="L40S", balanced=segment_set("balanced", 12))


# ---------------------------------------------------------------------------------------------
# registry and matching


POOLS = {"segment": PoolSettings(gpu="L40S", cpu_cores=6.0, memory_gib=48.0)}


def test_written_certificate_matches_only_its_kind_gpu_and_engine_spec(tmp_path):
    p = C.write_certificate(cert_for(), tmp_path)
    assert p.name.startswith("cert_") and json.loads(p.read_text())["verdict"] == "certified"
    found = presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, **MATCH)
    assert found["segment"].verdict == "certified" and found["segment"].gpu == "L40S"
    # GPU mismatch
    h100 = {"segment": PoolSettings(gpu="H100!", cpu_cores=6.0, memory_gib=48.0)}
    assert presets.find_certificates("balanced", h100, cert_dir=tmp_path, **MATCH) == {}
    # key mismatch: a different image, mode, or derived weights
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, embodiment="yam",
                                     derived_content=CONTENT,
                                     image_fingerprint=lambda n: "other") == {}
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path,
                                     modes={"segment": {"x": 1}}, **MATCH) == {}
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path,
                                     **{**MATCH, "derived_content": {"tempseg_merged": "f" * 64}}
                                     ) == {}
    # another preset or kind
    assert presets.find_certificates("budget", POOLS, cert_dir=tmp_path, **MATCH) == {}
    assert certkey.matching("balanced", {"caption": POOLS["segment"]}, cert_dir=tmp_path,
                            **MATCH) == {}


def test_logic_change_invalidates(tmp_path):
    C.write_certificate(cert_for(), tmp_path / "certs")
    root = tmp_path / "pkg"
    from flora_deploy.config.logic import modules_for, package_root

    for rel in modules_for("segment"):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((package_root() / rel).read_bytes())
    assert certkey.matching("balanced", POOLS, cert_dir=tmp_path / "certs", logic_root=root,
                            **MATCH)
    (root / "stages/segment.py").write_text("X = 1\n")
    assert certkey.matching("balanced", POOLS, cert_dir=tmp_path / "certs", logic_root=root,
                            **MATCH) == {}


def test_provisional_is_listed_but_does_not_unlock(tmp_path):
    C.write_certificate(cert_for(noise=False, stamp="2026-10-21T00:00:00Z"), tmp_path)
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, **MATCH) == {}
    st = certkey.status("balanced", POOLS, cert_dir=tmp_path, **MATCH)
    assert st["segment"]["status"] == "provisional" and st["segment"]["certificate"]
    assert certkey.status("balanced", POOLS, cert_dir=tmp_path / "none", **MATCH)["segment"][
        "status"] == "none"
    assert certkey.status("exact", POOLS)["segment"]["status"] == "reference"
    # a newer certified one decides
    C.write_certificate(cert_for(), tmp_path)
    assert certkey.status("balanced", POOLS, cert_dir=tmp_path, **MATCH)["segment"][
        "status"] == "certified"


def test_tampered_certificate_is_ignored(tmp_path):
    p = C.write_certificate(cert_for(noise=False), tmp_path)
    body = json.loads(p.read_text())
    body["verdict"] = "certified"
    p.write_text(json.dumps(body))
    assert certkey.load_certificates(tmp_path) == []
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, **MATCH) == {}
    with pytest.raises(C.CertifyError, match="content address"):
        C.write_certificate(body, tmp_path)


def _write_stage(tmp_path, cfg, stage, *, noise=True, content=CONTENT, stamp=STAMP):
    from flora_deploy.config.hashing import kind_modes

    mk = STAGE_SETS[stage]
    gpu = {"segment": "L40S"}.get(stage, "H100!")
    modes = kind_modes(cfg, stage)
    lab = {**L0_PASS, "kind": C.L0_STAGES.get(stage, "x")}
    doc = key("balanced", stage, gpu, fp=None, modes=modes, content=content)
    C.write_certificate(C.certify_stage(
        stage, exact=mk("exact", 10, fp=None), balanced=mk("balanced", 10, fp=None, modes=modes),
        noise=mk("exact", 5, fp=None) if noise else None, engine_spec=doc, gpu=gpu,
        l0=lab if stage in C.L0_STAGES else None, generated_at=stamp), tmp_path)


SCHEMA_CFG = {"project": "demo", "license_ack": "research-only",
              "modal": {"max_gpus": 4, "budget_usd": 20}, "preset": "balanced",
              "input": {"format": "lerobot_v3", "repo": "org/ds",
                        "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"},
              "pipeline": {"stages": ["segment", "caption", "reward.terminal"],
                           "embodiment": "yam"},
              "output": {"sink": "s3://b/p"}}


def test_schema_gate_reads_the_registry(tmp_path, monkeypatch):
    """End to end: certified certificates for every balanced pool unlock the preset without
    allow_uncertified, given the derived content digests; a provisional one does not."""
    from flora_deploy.config.schema import ConfigError, parse_config

    monkeypatch.setattr(certkey, "CERT_DIR", tmp_path)
    d = SCHEMA_CFG
    with pytest.raises(ConfigError, match="no accuracy certificate"):
        parse_config(d, derived_content=CONTENT)
    cfg = parse_config(d, allow_uncertified=True)
    _write_stage(tmp_path, cfg, "segment")
    _write_stage(tmp_path, cfg, "caption", noise=False)
    _write_stage(tmp_path, cfg, "reward.terminal")
    with pytest.raises(ConfigError, match="caption on H100!") as e:
        parse_config(d, derived_content=CONTENT)
    assert "segment on L40S" not in str(e.value)
    _write_stage(tmp_path, cfg, "caption", stamp="2026-10-23T00:00:00Z")  # newer: decides
    assert parse_config(d, derived_content=CONTENT).preset == "balanced"
    # fail closed: without the derived weights' content digests nothing with weights matches
    with pytest.raises(ConfigError, match="segment on L40S, caption on H100!"):
        parse_config(d)
    # another embodiment is not covered (D16)
    with pytest.raises(ConfigError, match="no accuracy certificate"):
        parse_config({**d, "pipeline": {**d["pipeline"], "embodiment": "human"}},
                     derived_content=CONTENT)


# ---------------------------------------------------------------------------------------------
# no record text in a certificate


def _rand(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(12))


def _random_record(rng: random.Random, uid: str, stage: str, preset: str) -> StageRecord:
    n = rng.randint(0, 4)
    t = 0.0
    sg = []
    for i in range(n):
        d = rng.uniform(0.5, 3.0)
        sg.append(Segment(event_id=_rand(rng), start_s=t, end_s=t + d, caption=_rand(rng),
                          fine_event_id=_rand(rng), segmentation_provenance=_rand(rng),
                          degradation={_rand(rng): _rand(rng)},
                          nv_metadata={_rand(rng): _rand(rng), "i": i}))
        t += d
    p = prov(preset, stage, framework=_rand(rng), certificate=_rand(rng),
             source={_rand(rng): {"uri": _rand(rng)}})
    p = Provenance(**{**p.__dict__, "models": {**p.models, _rand(rng): _rand(rng)}})
    terminal = TerminalReward(score=rng.randint(1, 5), expected_score=rng.uniform(1, 5),
                              probabilities=(0.2,) * 5, rationale=_rand(rng))
    dense = DenseReward(segments=({"label": _rand(rng), "rationale": _rand(rng)},),
                        per_step=tuple(rng.randint(1, 3) for _ in range(rng.randint(0, 6))))
    return StageRecord(uid=uid, dataset=_rand(rng), episode_id=_rand(rng), stage=stage,
                       provenance=p, certified=False, uncertified_reason="uncertified_stack",
                       segments=tuple(sg), terminal=terminal, dense=dense,
                       audit_ref=_rand(rng))


def _strings(obj, out: set[str]) -> set[str]:
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _strings(k, out)
            _strings(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _strings(v, out)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_certificate_never_contains_a_record_string(seed):
    """Property: no free-text field of any input record (captions, rationales, metadata, ids,
    dataset names, provenance text) appears anywhere in the certificate. The binding values
    (stage hashes, the key document, which must equal this checkout's) are ours by design."""
    from dataclasses import asdict, fields

    rng = random.Random(seed)
    stage = rng.choice(list(C.STAGE_METRICS))
    uids = [_rand(rng) for _ in range(rng.randint(1, 7))]
    ex = [_random_record(rng, u, stage, "exact") for u in uids]
    bal = [_random_record(rng, u, stage, "balanced") for u in uids if rng.random() < 0.9] or [
        _random_record(rng, uids[0], stage, "balanced")]
    nz = [_random_record(rng, u, stage, "exact") for u in uids[:2]]
    lab = {**L0_PASS, "kind": C.L0_STAGES.get(stage, "x")}
    gpu = {"segment": "L40S"}.get(stage, "H100!")
    cert = C.certify_stage(stage, exact=ex, balanced=bal, noise=nz,
                           engine_spec=key("balanced", stage, gpu), gpu=gpu,
                           l0=lab if stage in C.L0_STAGES else None,
                           similarity=(lambda a, b: 0.5) if stage == "caption" else None)
    ours = {"complete", "uncertified_stack", "flora_labels_v1", stage, "exact", "balanced"}
    ours |= {f.name for cls in (StageRecord, Provenance, Segment, TerminalReward, DenseReward)
             for f in fields(cls)}
    for r in ex + bal + nz:  # the binding fields: validated and ours
        ours |= _strings(r.provenance.cert_key, set()) | {r.provenance.config_hash}

    def leaks(text: str, records) -> set[str]:
        return {s for r in records for s in _strings(asdict(r), set())
                if s not in ours and len(s) >= 8 and s in text}

    assert not leaks(json.dumps(cert), ex + bal + nz)


# ---------------------------------------------------------------------------------------------
# CLI


def _balanced_yaml(tmp_path, embodiment="yam"):
    import yaml

    d = {**SCHEMA_CFG, "allow_uncertified": True,
         "pipeline": {"stages": ["segment", "caption"], "embodiment": embodiment}}
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def _jsonl(path, records):
    path.write_text("".join(r.to_json() + "\n" for r in records))
    return path


def _content_args():
    return [a for n, h in CONTENT.items() for a in ("--derived-content", f"{n}={h}")]


def test_cli_certify_writes_metrics_only_certificates(tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli import app

    cfg = _balanced_yaml(tmp_path)
    ex = _jsonl(tmp_path / "exact.jsonl", segment_set("exact", 12, fp=None))
    bal = _jsonl(tmp_path / "bal.jsonl", segment_set("balanced", 12, fp=None))
    nz = _jsonl(tmp_path / "noise.jsonl", segment_set("exact", 5, fp=None))
    lab = tmp_path / "lab.json"
    lab.write_text(json.dumps(L0_PASS))
    out = tmp_path / "certs"
    args = ["certify", "--exact", str(ex), "-c", str(cfg), "--out-dir", str(out),
            *_content_args()]
    r = CliRunner().invoke(app, [*args, "--balanced", str(bal), "--noise", str(nz),
                                 "--l0", str(lab), "--json"])
    assert r.exit_code == 0, r.output
    (cert,) = json.loads(r.output)
    assert cert["verdict"] == "certified" and cert["gpu"] == "L40S"
    assert (out / f"{cert['id']}.json").is_file()
    assert cert["engine_spec_hash"] == certkey.engine_spec_hash(
        "balanced", "segment", "L40S", derived_content=CONTENT)
    # without the noise run: provisional, still written, exit 0
    r = CliRunner().invoke(app, [*args, "--balanced", str(bal)])
    assert r.exit_code == 0 and "provisional" in r.output and "allow_uncertified" in r.output
    # a failing stack exits 1
    bad = _jsonl(tmp_path / "bad.jsonl", segment_set("balanced", 12, shift=1.0, fp=None))
    r = CliRunner().invoke(app, [*args, "--balanced", str(bad), "--noise", str(nz),
                                 "--l0", str(lab)])
    assert r.exit_code == 1 and "fail" in r.output
    # no content digests: refused
    r = CliRunner().invoke(app, ["certify", "--exact", str(ex), "-c", str(cfg), "--balanced",
                                 str(bal)])
    assert r.exit_code == 2 and "derived-content" in r.output
    # knobs outside policy: refused
    r = CliRunner().invoke(app, [*args, "--balanced", str(bal), "--tolerance", "4"])
    assert r.exit_code == 2 and "outside policy" in r.output
    # the config's embodiment must be the records'
    r = CliRunner().invoke(app, ["certify", "--exact", str(ex), "--balanced", str(bal), "-c",
                                 str(_balanced_yaml(tmp_path / "..", "human")), *_content_args()])
    assert r.exit_code == 2 and "embodiment" in r.output
