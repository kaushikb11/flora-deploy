"""validate/certify.py + validate/certkey.py: metrics, pass rule, certificate content, matching."""

from __future__ import annotations

import json
import random
import string

import pytest

from flora_deploy import LOGIC_VERSION
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

FP = {"image_fingerprint": lambda name: f"fp-{name}"}
STAGE_HASH = "a" * 64


def prov(preset: str, gpu: str | None = None, config_hash: str = STAGE_HASH, **kw) -> Provenance:
    return Provenance(models={"tempseg": "r"}, engine=kw.pop("engine", "e"), preset=preset,
                      config_hash=config_hash, logic_version=LOGIC_VERSION,
                      framework="flora-deploy test", gpu=gpu, **kw)


def rec(uid: str, stage: str, preset: str, *, gpu: str | None = None, segments=(),
        terminal=None, dense=None, **kw) -> StageRecord:
    return StageRecord(uid=uid, dataset="ds", episode_id=uid, stage=stage,
                       provenance=prov(preset, gpu, **kw), certified=preset == "exact",
                       uncertified_reason=None if preset == "exact" else "uncertified_stack",
                       segments=tuple(segments), terminal=terminal, dense=dense)


def segs(bounds, captions=None) -> list[Segment]:
    return [Segment(event_id=f"event_{i:04d}", start_s=s, end_s=e,
                    caption=(captions[i] if captions else None), segmentation_provenance="fine")
            for i, (s, e) in enumerate(bounds)]


BOUNDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 9.0)]


def segment_set(preset: str, n: int, shift: float = 0.0, gpu: str | None = None):
    return [rec(f"u{i}", "segment", preset, gpu=gpu,
                segments=segs([(s + shift, e + shift) for s, e in BOUNDS])) for i in range(n)]


def engine_spec(kind: str = "segment", **kw):
    return certkey.engine_spec_document("balanced", kind, **FP, **kw)


L0_PASS = {"kind": "tempseg", "baseline": {"l0": [
    {"key": "k", "ids_equal": True, "render_ids_equal": True, "grid_equal": True}] * 6},
    "gate_a": {"requests": 6, "greedy_identical": 6, "positions": 10, "max_abs_dlogit": 0.1,
               "mean_abs_dlogit": 0.01, "first_divergence": []}}


def cert_for(kind="segment", *, n=6, noise=True, shift=0.0, noise_shift=0.0, l0=L0_PASS,
             gpu="L40S", **kw):
    noise_recs = segment_set("exact", 3, noise_shift) if noise else None
    return C.certify_stage(kind, exact=segment_set("exact", n), balanced=segment_set(
        "balanced", n, shift), noise=noise_recs, engine_spec=engine_spec(kind), gpu=gpu,
        l0=l0, **kw)


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
    assert c["metrics"]["noise"]["n"] == 3 and c["metrics"]["coverage"] == 1.0
    assert c["l0"]["status"] == "pass" and c["gate_a"]["greedy_identical"] == 6
    assert c["id"] == certkey.certificate_id(c) and c["id"].startswith("cert_")
    assert c["stage_hash"] == STAGE_HASH
    assert c["engine_spec_hash"] == certkey.engine_spec_hash("balanced", "segment", **FP)
    assert c["episodes"]["count"] == 6 and len(c["episodes"]["uid_sha256"]) == 6


def test_divergence_above_noise_floor_fails():
    c = cert_for(shift=0.5, noise_shift=0.1)
    assert c["verdict"] == "fail" and "divergence_above_noise_floor" in c["reasons"]
    bad = {ch["metric"] for ch in c["checks"] if ch["pass"] is False}
    assert {"iou_divergence", "boundary_error_s"} <= bad
    # within tolerance x noise: certified
    assert cert_for(shift=0.12, noise_shift=0.1)["verdict"] == "certified"
    # slack lifts the threshold
    th = C.Thresholds(slack={"iou_divergence": 1.0, "boundary_error_s": 1.0})
    assert cert_for(shift=0.5, noise_shift=0.1, thresholds=th)["verdict"] == "certified"


def test_no_noise_run_is_provisional_and_never_unlocks():
    c = cert_for(noise=False)
    assert c["verdict"] == "provisional" and c["unlocks"] is False
    assert "no_noise_run" in c["reasons"]
    assert all(ch["pass"] is None and ch["threshold"] is None for ch in c["checks"])
    # even a large divergence cannot fail or certify without a noise floor
    assert cert_for(noise=False, shift=3.0)["verdict"] == "provisional"


@pytest.mark.parametrize("kw,reason", [
    ({"n": 3}, "too_few_samples"),
    ({"l0": None}, "no_l0"),
    ({"thresholds": C.Thresholds(min_noise_samples=5)}, "too_few_noise_samples"),
])
def test_insufficient_evidence_is_provisional(kw, reason):
    c = cert_for(**kw)
    assert c["verdict"] == "provisional" and reason in c["reasons"]


def test_samples_counted_per_embodiment_class():
    classes = {f"u{i}": ("yam" if i < 5 else "human") for i in range(6)}
    c = cert_for(classes=classes)
    assert c["metrics"]["candidate"]["human"]["n"] == 1
    assert c["verdict"] == "provisional" and "too_few_samples" in c["reasons"]


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
    t = TerminalReward(score=4, expected_score=4.2, probabilities=(0, 0, 0, 0.8, 0.2))
    mk = lambda p, n: [rec(f"u{i}", "reward.terminal", p, terminal=t)  # noqa: E731
                       for i in range(n)]
    c = C.certify_stage("reward.terminal", exact=mk("exact", 5), balanced=mk("balanced", 5),
                        noise=mk("exact", 2), engine_spec=engine_spec("reward.terminal"),
                        gpu="H100!")
    assert c["l0"] == {"status": "not_applicable"} and c["verdict"] == "certified"


def test_input_consistency_is_checked():
    ex = segment_set("exact", 6)
    with pytest.raises(C.CertifyError, match="one stage hash"):
        C.certify_stage("segment", exact=ex, engine_spec=engine_spec(), gpu="L40S",
                        balanced=[*segment_set("balanced", 5),
                                  rec("u9", "segment", "balanced", config_hash="b" * 64)])
    with pytest.raises(C.CertifyError, match="not produced on"):
        C.certify_stage("segment", exact=ex, engine_spec=engine_spec(), gpu="L40S",
                        balanced=segment_set("balanced", 6, gpu="NVIDIA H100 80GB HBM3"))
    with pytest.raises(C.CertifyError, match="preset 'exact'"):
        C.certify_stage("segment", exact=segment_set("balanced", 6), engine_spec=engine_spec(),
                        gpu="L40S", balanced=segment_set("balanced", 6))
    assert C.certify_stage("segment", exact=ex, engine_spec=engine_spec(), gpu="L40S",
                           balanced=segment_set("balanced", 6, gpu="NVIDIA L40S"))


# ---------------------------------------------------------------------------------------------
# registry and matching


POOLS = {"segment": PoolSettings(gpu="L40S", cpu_cores=6.0, memory_gib=48.0)}


def test_written_certificate_matches_only_its_kind_gpu_and_engine_spec(tmp_path):
    p = C.write_certificate(cert_for(), tmp_path)
    assert p.name.startswith("cert_") and json.loads(p.read_text())["verdict"] == "certified"
    found = presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, **FP)
    assert found["segment"].verdict == "certified" and found["segment"].gpu == "L40S"
    # GPU mismatch
    h100 = {"segment": PoolSettings(gpu="H100!", cpu_cores=6.0, memory_gib=48.0)}
    assert presets.find_certificates("balanced", h100, cert_dir=tmp_path, **FP) == {}
    # engine spec / stage hash mismatch: a different image, mode or logic tree
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path,
                                     image_fingerprint=lambda n: "other") == {}
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path,
                                     modes={"segment": {"x": 1}}, **FP) == {}
    # another preset or kind
    assert presets.find_certificates("budget", POOLS, cert_dir=tmp_path, **FP) == {}
    assert certkey.matching("balanced", {"caption": POOLS["segment"]}, cert_dir=tmp_path,
                            **FP) == {}


def test_logic_change_invalidates(tmp_path):
    C.write_certificate(cert_for(), tmp_path / "certs")
    root = tmp_path / "pkg"
    from flora_deploy.config.logic import modules_for, package_root

    for rel in modules_for("segment"):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((package_root() / rel).read_bytes())
    assert certkey.matching("balanced", POOLS, cert_dir=tmp_path / "certs", logic_root=root,
                            **FP)
    (root / "stages/segment.py").write_text("# changed\n")
    assert certkey.matching("balanced", POOLS, cert_dir=tmp_path / "certs", logic_root=root,
                            **FP) == {}


def test_provisional_is_listed_but_does_not_unlock(tmp_path):
    C.write_certificate(cert_for(noise=False), tmp_path)
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, **FP) == {}
    st = certkey.status("balanced", POOLS, cert_dir=tmp_path, **FP)
    assert st["segment"]["status"] == "provisional" and st["segment"]["certificate"]
    assert certkey.status("balanced", POOLS, cert_dir=tmp_path / "none", **FP)["segment"][
        "status"] == "none"
    assert certkey.status("exact", POOLS)["segment"]["status"] == "reference"
    # certified wins over provisional for the same pool
    C.write_certificate(cert_for(), tmp_path)
    assert certkey.status("balanced", POOLS, cert_dir=tmp_path, **FP)["segment"][
        "status"] == "certified"


def test_tampered_certificate_is_ignored(tmp_path):
    p = C.write_certificate(cert_for(noise=False), tmp_path)
    body = json.loads(p.read_text())
    body["verdict"] = "certified"
    p.write_text(json.dumps(body))
    assert certkey.load_certificates(tmp_path) == []
    assert presets.find_certificates("balanced", POOLS, cert_dir=tmp_path, **FP) == {}
    with pytest.raises(C.CertifyError, match="content address"):
        C.write_certificate(body, tmp_path)


def test_schema_gate_reads_the_registry(tmp_path, monkeypatch):
    """End to end: certified certificates for every balanced pool unlock the preset without
    allow_uncertified; a provisional one does not."""
    from flora_deploy.config.schema import ConfigError, parse_config

    monkeypatch.setattr(certkey, "CERT_DIR", tmp_path)
    d = {"project": "demo", "license_ack": "research-only",
         "modal": {"max_gpus": 4, "budget_usd": 20}, "preset": "balanced",
         "input": {"format": "lerobot_v3", "repo": "org/ds",
                   "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"},
         "pipeline": {"stages": ["segment", "caption", "reward.terminal"],
                      "embodiment": "human"},
         "output": {"sink": "s3://b/p"}}
    with pytest.raises(ConfigError, match="no Gate certificate"):
        parse_config(d)
    cfg = parse_config(d, allow_uncertified=True)
    from flora_deploy.config.hashing import kind_modes

    def write(stage, gpu, noise=True):
        mk = lambda p, n: segment_set(p, n)  # noqa: E731
        if stage == "caption":
            mk = lambda p, n: [rec(f"u{i}", "caption", p,  # noqa: E731
                                   segments=segs(BOUNDS, ["a", "b", "c"])) for i in range(n)]
        if stage == "reward.terminal":
            t = TerminalReward(score=4, expected_score=4.0, probabilities=(0, 0, 0, 1, 0))
            mk = lambda p, n: [rec(f"u{i}", stage, p, terminal=t)  # noqa: E731
                               for i in range(n)]
        lab = {**L0_PASS, "kind": C.L0_STAGES.get(stage, "x")}
        doc = certkey.engine_spec_document("balanced", stage, modes=kind_modes(cfg, stage))
        C.write_certificate(C.certify_stage(
            stage, exact=mk("exact", 5), balanced=mk("balanced", 5),
            noise=mk("exact", 2) if noise else None, engine_spec=doc, gpu=gpu,
            l0=lab if stage in C.L0_STAGES else None), tmp_path)

    write("segment", "L40S")
    write("caption", "H100!", noise=False)
    write("reward.terminal", "H100!")
    with pytest.raises(ConfigError, match="caption on H100!") as e:
        parse_config(d)
    assert "segment on L40S" not in str(e.value)
    write("caption", "H100!")
    assert parse_config(d).preset == "balanced"


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
    p = Provenance(models={_rand(rng): _rand(rng)}, engine=_rand(rng), preset=preset,
                   config_hash=STAGE_HASH, logic_version=LOGIC_VERSION, framework=_rand(rng),
                   gpu=None, certificate=_rand(rng), source={_rand(rng): {"uri": _rand(rng)}})
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


@pytest.mark.parametrize("seed", range(40))
def test_certificate_never_contains_a_record_string(seed):
    """Property: no string field of any input record (captions, rationales, metadata, ids,
    dataset names, provenance text) appears anywhere in the certificate. The one exception by
    design is ``provenance.config_hash``, the certified stage hash."""
    from dataclasses import asdict

    rng = random.Random(seed)
    stage = rng.choice(list(C.STAGE_METRICS))
    uids = [_rand(rng) for _ in range(rng.randint(1, 7))]
    ex = [_random_record(rng, u, stage, "exact") for u in uids]
    bal = [_random_record(rng, u, stage, "balanced") for u in uids if rng.random() < 0.9] or [
        _random_record(rng, uids[0], stage, "balanced")]
    nz = [_random_record(rng, u, stage, "exact") for u in uids[:2]]
    classes = {u: _rand(rng) for u in uids} if rng.random() < 0.5 else None
    lab = {**L0_PASS, "kind": C.L0_STAGES.get(stage, "x")}
    cert = C.certify_stage(stage, exact=ex, balanced=bal, noise=nz,
                           engine_spec=engine_spec(stage),
                           gpu="L40S", classes=None, l0=lab if stage in C.L0_STAGES else None,
                           similarity=(lambda a, b: 0.5) if stage == "caption" else None)
    # Our own enum vocabulary (stage, preset, status, schema names) is not record text.
    ours = {STAGE_HASH, "complete", "uncertified_stack", "flora_labels_v1", stage, "exact",
            "balanced"}
    # ... nor are the record schema's own field names (dict keys of ``asdict``).
    from dataclasses import fields

    ours |= {f.name for cls in (StageRecord, Provenance, Segment, TerminalReward, DenseReward)
             for f in fields(cls)}

    def leaks(text: str, records) -> set[str]:
        return {s for r in records for s in _strings(asdict(r), set())
                if s not in ours and len(s) >= 8 and s in text}

    assert not leaks(json.dumps(cert), ex + bal + nz)
    if classes:  # embodiment class names are ours, but they still never carry record text
        cert2 = C.certify_stage(stage, exact=ex, balanced=bal, noise=nz,
                                engine_spec=engine_spec(stage), gpu="L40S", classes=classes,
                                l0=lab if stage in C.L0_STAGES else None)
        assert not leaks(json.dumps(cert2), ex + bal + nz)


# ---------------------------------------------------------------------------------------------
# CLI


def _balanced_yaml(tmp_path):
    import yaml

    d = {"project": "demo", "license_ack": "research-only",
         "modal": {"max_gpus": 4, "budget_usd": 20}, "preset": "balanced",
         "allow_uncertified": True,
         "input": {"format": "lerobot_v3", "repo": "org/ds",
                   "revision": "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"},
         "pipeline": {"stages": ["segment", "caption"], "embodiment": "human"},
         "output": {"sink": "s3://b/p"}}
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def _jsonl(path, records):
    path.write_text("".join(r.to_json() + "\n" for r in records))
    return path


def test_cli_certify_writes_metrics_only_certificates(tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli import app

    cfg = _balanced_yaml(tmp_path)
    ex = _jsonl(tmp_path / "exact.jsonl", segment_set("exact", 6))
    bal = _jsonl(tmp_path / "bal.jsonl", segment_set("balanced", 6, gpu="NVIDIA L40S"))
    nz = _jsonl(tmp_path / "noise.jsonl", segment_set("exact", 3))
    lab = tmp_path / "lab.json"
    lab.write_text(json.dumps(L0_PASS))
    out = tmp_path / "certs"
    args = ["certify", "--exact", str(ex), "-c", str(cfg), "--out-dir", str(out)]
    r = CliRunner().invoke(app, [*args, "--balanced", str(bal), "--noise", str(nz),
                                 "--l0", str(lab), "--json"])
    assert r.exit_code == 0, r.output
    (cert,) = json.loads(r.output)
    assert cert["verdict"] == "certified" and cert["gpu"] == "L40S"
    assert (out / f"{cert['id']}.json").is_file()
    assert cert["engine_spec_hash"] == certkey.engine_spec_hash("balanced", "segment")
    # without the noise run: provisional, still written, exit 0
    r = CliRunner().invoke(app, [*args, "--balanced", str(bal)])
    assert r.exit_code == 0 and "provisional" in r.output and "allow_uncertified" in r.output
    # a failing stack exits 1
    bad = _jsonl(tmp_path / "bad.jsonl", segment_set("balanced", 6, shift=1.0))
    r = CliRunner().invoke(app, [*args, "--balanced", str(bad), "--noise", str(nz),
                                 "--l0", str(lab)])
    assert r.exit_code == 1 and "fail" in r.output
