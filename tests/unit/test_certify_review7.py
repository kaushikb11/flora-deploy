"""Review 7 repros (scratchpad/review7_cert): each test is a reviewer probe that the W1-Certify
code got wrong (it certified, accepted or refused the wrong thing), now pinned."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from flora_deploy.config import presets
from flora_deploy.config.presets import PoolSettings
from flora_deploy.records import DenseReward, Provenance, StageRecord
from flora_deploy.validate import certify as C
from flora_deploy.validate import certkey
from tests.fakes.certrecords import (
    CONTENT,
    FP,
    L0_PASS,
    ORACLE_HASH,
    dense_set,
    key,
    rec,
    segment_set,
    terminal,
    terminal_set,
)

STAMP = "2026-10-22T00:00:00Z"
SEG = {"segment": PoolSettings(gpu="L40S", cpu_cores=6.0, memory_gib=48.0)}
MATCH = {"embodiment": "yam", "derived_content": CONTENT, **FP}


def seg_cert(*, exact=None, balanced=None, noise=None, n=12, n_noise=5, stamp=STAMP, **kw):
    return C.certify_stage(
        "segment", exact=exact if exact is not None else segment_set("exact", n),
        balanced=balanced if balanced is not None else segment_set("balanced", n),
        noise=noise if noise is not None else segment_set("exact", n_noise),
        engine_spec=key("balanced", "segment", "L40S"), gpu="L40S", l0=L0_PASS,
        generated_at=stamp, **kw)


def term_cert(*, exact, balanced, noise, **kw):
    return C.certify_stage("reward.terminal", exact=exact, balanced=balanced, noise=noise,
                           engine_spec=key("balanced", "reward.terminal", "H100!"), gpu="H100!",
                           generated_at=STAMP, **kw)


def _with(records, **prov_changes):
    return [replace(r, provenance=replace(r.provenance, **prov_changes)) for r in records]


# ---------------------------------------------------------------------------------------------
# H1: the key is derived from the records


@pytest.mark.parametrize("fake", ["local-fake-models", "fake-models"])
def test_h1_fake_model_records_are_refused(fake):
    """P2: every input from `flora run --local --fake-models` came out `certified`."""
    with pytest.raises(C.CertifyRefused) as e:
        term_cert(exact=terminal_set("exact", 10), noise=terminal_set("exact", 5),
                  balanced=_with(terminal_set("balanced", 10), certificate=fake))
    assert e.value.reason == "fake_models"
    with pytest.raises(C.CertifyRefused, match="fake"):
        term_cert(exact=_with(terminal_set("exact", 10), engine="flora-local-fake"),
                  balanced=terminal_set("balanced", 10), noise=terminal_set("exact", 5))


def test_h1_records_without_binding_provenance_fail_closed():
    """Records as today's workers write them (no cert_key / embodiment / GPU) certify nothing,
    and the error names the fields to add."""
    old = _with(segment_set("balanced", 12), cert_key=None, embodiment=None, gpu=None)
    with pytest.raises(C.CertifyRefused) as e:
        seg_cert(balanced=old)
    assert e.value.reason == "missing_provenance"
    for f in ("cert_key", "embodiment", "gpu"):
        assert f in str(e.value)
    bad_hash = _with(segment_set("balanced", 12), config_hash="not-a-hash")
    with pytest.raises(C.CertifyRefused, match="64-hex"):
        seg_cert(balanced=bad_hash)
    broken = _with(segment_set("balanced", 12), cert_key={"schema": "x"})
    with pytest.raises(C.CertifyRefused, match="cert_key"):
        seg_cert(balanced=broken)


@pytest.mark.parametrize("change", [
    {"fp": {"image_fingerprint": lambda n: "older-image"}},
    {"dtbias_fp32": True},
    {"content": {**CONTENT, "tempseg_merged": "e" * 64}},
    {"modes": {"x": 1}},
])
def test_h1_records_of_another_stack_do_not_certify_this_checkout(change):
    """The key used to come from the checkout and `--config`: records from an older image, the
    dt_bias plugin, other weights or other modes certified today's stack."""
    other = key("balanced", "segment", "L40S", **change)
    recs = _with(segment_set("balanced", 12), cert_key=other,
                 engine=other["vllm_engine"]["label"])
    with pytest.raises(C.CertifyError, match="another stack"):
        seg_cert(balanced=recs)


def test_h1_key_engine_and_pins_must_be_uniform_and_consistent():
    mixed = [*segment_set("balanced", 6),
             *_with(segment_set("balanced", 6, start=6),
                    cert_key=key("balanced", "segment", "L40S", dtbias_fp32=True),
                    engine="vllm-0.24.0-offline+dtbias")]
    with pytest.raises(C.CertifyError, match="one certificate key"):
        seg_cert(balanced=mixed)
    with pytest.raises(C.CertifyError, match="provenance.engine"):
        seg_cert(balanced=_with(segment_set("balanced", 12), engine="vllm-0.24.0-offline+dtbias"))
    with pytest.raises(C.CertifyError, match="models disagree"):
        seg_cert(balanced=_with(segment_set("balanced", 12), models={"tempseg": "other"}))
    with pytest.raises(C.CertifyError, match="unknown GPU"):
        seg_cert(balanced=_with(segment_set("balanced", 12), gpu="NVIDIA B200"))


def test_h1_exact_substring_gpu_match_is_gone():
    """M8 (with H1): "L4" matched "NVIDIA L40S"; None matched anything."""
    with pytest.raises(C.CertifyError, match="not produced on"):
        C.certify_stage("segment", exact=segment_set("exact", 12),
                        balanced=segment_set("balanced", 12),
                        engine_spec=key("balanced", "segment", "L40S"), gpu="A100-80GB")


# ---------------------------------------------------------------------------------------------
# H2: the noise run repeats the oracle run


def test_h2_noise_from_another_stack_is_refused():
    """P3: noise from another stack (another stage hash, GPU) inflated the floor until a
    balanced run that flips every terminal score was `certified`."""
    ex = terminal_set("exact", 10)
    with pytest.raises(C.CertifyError, match="stage_hash"):
        term_cert(exact=ex, balanced=terminal_set("balanced", 10),
                  noise=terminal_set("exact", 5, config_hash="f" * 64))
    with pytest.raises(C.CertifyError, match="gpu"):
        term_cert(exact=ex, balanced=terminal_set("balanced", 10),
                  noise=terminal_set("exact", 5, gpu="H200"))


def test_h2_oracle_must_be_one_run_and_noise_a_subset_not_a_copy():
    """P3b: oracle records from two stage hashes, with noise = the oracle file, were accepted."""
    two = [*segment_set("exact", 6), *segment_set("exact", 6, start=6, config_hash="f" * 64)]
    with pytest.raises(C.CertifyError, match="one stage hash"):
        seg_cert(exact=two)
    with pytest.raises(C.CertifyError, match="subset"):
        seg_cert(noise=segment_set("exact", 5, start=20))
    ex = [replace(r, audit_ref=f"audit/{r.uid}") for r in segment_set("exact", 12)]
    with pytest.raises(C.CertifyError, match="copies"):
        seg_cert(exact=ex, noise=ex[:5])


def test_h2_determinism_mode_must_match_and_is_recorded():
    det = segment_set("exact", 5, modes={"determinism": "deterministic"})
    with pytest.raises(C.CertifyError, match="differ from the oracle"):
        seg_cert(noise=det)
    c = seg_cert()
    assert c["oracle"] == {"stage_hash": ORACLE_HASH, "gpu": "H200", "determinism": "nvidia",
                           "engine_spec_hash": certkey.sha256_hex(key("exact", "segment",
                                                                      "H200"))}


# ---------------------------------------------------------------------------------------------
# H3: bounded knobs, absolute caps, consumption re-checks them


@pytest.mark.parametrize("kw", [
    {"tolerance": 4.0}, {"tolerance": 0.5},
    {"slack": {"expected_score_abs_delta": 4.0}},  # P6
    {"slack": {"score_mismatch": 1.0}},
    {"min_samples": 1}, {"min_noise_samples": 1},  # P6b
    {"min_coverage": 0.5},
    {"slack": {"made_up": 0.0}},
])
def test_h3_knobs_outside_policy_are_refused(kw):
    with pytest.raises(C.CertifyError, match="outside policy"):
        C.Thresholds(**kw)


def test_h3_absolute_cap_on_terminal_expected_score():
    """P5: an E[s] drift of 0.42 certified against a noisy floor; PLAN §2 caps it at 0.05."""
    noisy = [rec(f"u{i}", "reward.terminal", "exact", terminal=terminal(4, 4.2 + 0.6))
             for i in range(5)]
    bal = terminal_set("balanced", 10, es=4.62)
    c = term_cert(exact=terminal_set("exact", 10), balanced=bal, noise=noisy)
    t = c["thresholds"]["per_metric"]["expected_score_abs_delta"]
    assert t["cap"] == 0.05 and t["threshold"] == 0.05
    assert c["verdict"] == "fail"


def _forge(cert, **changes):
    body = json.loads(json.dumps(cert))
    for path, value in changes.items():
        node = body
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    body["id"] = certkey.certificate_id(body)
    return body


def test_h3_matching_rejects_forged_or_weak_certificates(tmp_path):
    """P14d: a body edited and re-addressed loads; matching must still refuse one that is
    weaker than policy, or whose checks do not recompute from its own metrics."""
    good = seg_cert(balanced=segment_set("balanced", 12, shift=0.5),
                    noise=segment_set("exact", 5, shift=0.1))
    assert good["verdict"] == "fail"
    for forged in (
        _forge(good, verdict="certified", reasons=[], unlocks=True),  # checks do not recompute
        _forge(seg_cert(), **{"thresholds.tolerance": 10.0}),
        _forge(seg_cert(), **{"thresholds.min_samples": 1}),
        _forge(seg_cert(), **{"generator.version": "1"}),
    ):
        d = tmp_path / forged["id"]
        C.write_certificate(forged, d)
        assert certkey.load_certificates(d)  # well-formed and self-consistent ...
        assert presets.find_certificates("balanced", SEG, cert_dir=d, **MATCH) == {}  # ... no


# ---------------------------------------------------------------------------------------------
# H4: Wave H-lite (6 + 2) is provisional by default


def test_h4_wave_h_lite_without_pools_is_provisional():
    """P1: 6 episodes and a 2-episode noise run came out `certified` without `--pools`."""
    c = seg_cert(n=6, n_noise=2)
    assert c["verdict"] == "provisional" and not c["unlocks"]
    assert {"too_few_samples", "too_few_noise_samples"} <= set(c["reasons"])


# ---------------------------------------------------------------------------------------------
# H5: a certificate covers only the embodiment pools it measured (D16)


def test_h5_matching_checks_the_configs_embodiment(tmp_path):
    C.write_certificate(seg_cert(), tmp_path)
    assert presets.find_certificates("balanced", SEG, cert_dir=tmp_path, **MATCH)
    for emb in ("human", "dex", None):
        assert presets.find_certificates("balanced", SEG, cert_dir=tmp_path,
                                         **{**MATCH, "embodiment": emb}) == {}
    with pytest.raises(C.CertifyError, match="different embodiments"):
        seg_cert(balanced=segment_set("balanced", 12, embodiment="human"))
    with pytest.raises(C.CertifyError, match="unknown embodiment"):
        seg_cert(balanced=segment_set("balanced", 12, embodiment="robot-x"),
                 exact=segment_set("exact", 12, embodiment="robot-x"),
                 noise=segment_set("exact", 5, embodiment="robot-x"))


# ---------------------------------------------------------------------------------------------
# H6 / H7: derived weights by content digest, fail closed; one lookup for every caller


def test_h6_derived_weights_are_required_and_matched_by_content(tmp_path):
    """P14b: `derived_receipts=None` matched; every caller passed None."""
    C.write_certificate(seg_cert(), tmp_path)
    assert presets.find_certificates("balanced", SEG, cert_dir=tmp_path, **{
        **MATCH, "derived_content": None}) == {}
    assert presets.find_certificates("balanced", SEG, cert_dir=tmp_path, **{
        **MATCH, "derived_content": {"tempseg_merged": "e" * 64}}) == {}
    st = certkey.status("balanced", SEG, cert_dir=tmp_path, **{**MATCH, "derived_content": None})
    assert st["segment"]["status"] == "unverified_weights"
    assert seg_cert()["derived_weights"] == {"tempseg_merged": CONTENT["tempseg_merged"]}


def _rewards_cfg(tmp_path):
    from flora_deploy.cli.init import template_text
    from flora_deploy.config.schema import loads_config

    text = template_text("rewards_only").replace("preset: exact", "preset: balanced")
    (tmp_path / "rollouts").mkdir()
    text = text.replace("./rollouts", str(tmp_path / "rollouts"))
    return text, loads_config(text, source="x", allow_uncertified=True)


def test_h7_config_validation_and_plan_app_agree_on_reward_certificates(tmp_path, monkeypatch):
    """probe_deploy: config validation accepted a reward.terminal certificate that `plan_app`
    then refused (it passed no modes)."""
    from flora_deploy.config.hashing import kind_modes
    from flora_deploy.config.schema import loads_config
    from flora_deploy.modal_app.factory import DeployError, plan_app

    text, cfg = _rewards_cfg(tmp_path)
    cdir = tmp_path / "certs"
    monkeypatch.setattr(certkey, "CERT_DIR", cdir)
    emb = cfg.pipeline.embodiment
    gpu = cfg.pools()["reward.terminal"].gpu
    for kind in ("reward.terminal", "reward.dense"):
        modes = kind_modes(cfg, kind)
        mk = terminal_set if kind == "reward.terminal" else dense_set
        kw = {"fp": None, "embodiment": emb, "gpu": gpu}
        C.write_certificate(C.certify_stage(
            kind, exact=mk("exact", 10, **kw), balanced=mk("balanced", 10, modes=modes, **kw),
            noise=mk("exact", 5, **kw), gpu=gpu, generated_at=STAMP,
            engine_spec=key("balanced", kind, gpu, fp=None, modes=modes)), cdir)
    cfg2 = loads_config(text, source="x")  # validation: accepted without allow_uncertified
    assert plan_app(cfg2, store="s3://b/x")  # deploy: accepted too
    # and both refuse once a newer fail revokes the terminal certificate
    gpu_kw = {"fp": None, "embodiment": emb, "gpu": gpu}
    modes = kind_modes(cfg, "reward.terminal")
    C.write_certificate(C.certify_stage(
        "reward.terminal", exact=terminal_set("exact", 10, **gpu_kw),
        balanced=terminal_set("balanced", 10, score=1, es=1.0, modes=modes, **gpu_kw),
        noise=terminal_set("exact", 5, **gpu_kw), gpu=gpu, generated_at="2026-10-23T00:00:00Z",
        engine_spec=key("balanced", "reward.terminal", gpu, fp=None, modes=modes)), cdir)
    with pytest.raises(DeployError, match="Gate certificate"):
        plan_app(cfg, store="s3://b/x")


# ---------------------------------------------------------------------------------------------
# M1: the certificate logic digest ignores comments, docstrings and the registry


def test_m1_normalized_source_drops_comments_and_docstrings():
    a = '"""Doc."""\nimport os  # why\n\n\ndef f(x):\n    """More."""\n    return x  # ok\n'
    b = "import os\ndef f(x):\n    return x\n"
    assert certkey.normalized_source(a) == certkey.normalized_source(b)
    assert certkey.normalized_source("def f(): return 1\n") != certkey.normalized_source(
        "def f(): return 2\n")
    assert certkey.normalized_source('def f(): """d"""\n') == 'def f(): """d"""'  # kept: same line


def test_m1_cert_digest_is_stable_under_unrelated_edits(tmp_path):
    import flora_deploy

    root = tmp_path / "flora_deploy"
    shutil.copytree(flora_deploy.__path__[0], root, ignore=shutil.ignore_patterns("__pycache__"))
    base = {k: certkey.cert_logic_digest(k, root) for k in ("segment", "reward.terminal")}

    def after(rel: str, text: str, append: bool = True) -> dict[str, bool]:
        p = root / rel
        orig = p.read_text() if p.exists() else ""
        p.write_text(orig + text if append else text)
        now = {k: certkey.cert_logic_digest(k, root) != base[k] for k in base}
        if rel.startswith("cli/new_"):
            p.unlink()
        else:
            p.write_text(orig)
        return now

    for rel in ("config/logic.py", "config/hashing.py", "validate/certkey.py",
                "config/schema.py", "stages/segment.py", "modal_app/factory.py"):
        assert after(rel, "\n# a comment-only edit\n") == {"segment": False,
                                                           "reward.terminal": False}, rel
    assert after("cli/new_module.py", "X = 1\n", append=False) == {
        "segment": False, "reward.terminal": False}
    assert after("config/logic.py", "\nNEW_REGISTRY_ENTRY = ('cli/new_module.py',)\n") == {
        "segment": False, "reward.terminal": False}
    assert after("stages/segment.py", "\nX = 1\n") == {"segment": True, "reward.terminal": False}
    assert after("stages/reward.py", "\nX = 1\n") == {"segment": False, "reward.terminal": True}


# ---------------------------------------------------------------------------------------------
# M2: a newer fail revokes an older certified


def test_m2_newest_certificate_decides_and_fail_wins_ties(tmp_path):
    """P14: a certified and a failed certificate for one key gave `certified`."""
    ok = seg_cert(stamp="2026-10-20T00:00:00Z")
    bad = seg_cert(balanced=segment_set("balanced", 12, shift=0.5),
                   noise=segment_set("exact", 5, shift=0.1), stamp="2026-10-21T00:00:00Z")
    assert ok["verdict"] == "certified" and bad["verdict"] == "fail"
    assert ok["engine_spec_hash"] == bad["engine_spec_hash"]
    C.write_certificate(ok, tmp_path)
    C.write_certificate(bad, tmp_path)
    assert presets.find_certificates("balanced", SEG, cert_dir=tmp_path, **MATCH) == {}
    assert certkey.status("balanced", SEG, cert_dir=tmp_path, **MATCH)["segment"][
        "status"] == "fail"
    # a newer certified (e.g. after more data) decides again
    C.write_certificate(seg_cert(stamp="2026-10-22T00:00:00Z"), tmp_path)
    assert presets.find_certificates("balanced", SEG, cert_dir=tmp_path, **MATCH)
    # same time: fail wins
    tie = tmp_path / "tie"
    C.write_certificate(seg_cert(stamp=STAMP), tie)
    C.write_certificate(replace_stamp(bad, STAMP), tie)
    assert certkey.status("balanced", SEG, cert_dir=tie, **MATCH)["segment"]["status"] == "fail"
    # a certificate without a generation time is not loaded
    no_time = _forge(ok, generated_at=None)
    C.write_certificate(no_time, tmp_path / "nt")
    assert certkey.load_certificates(tmp_path / "nt") == []


def replace_stamp(cert, stamp):
    return _forge(cert, generated_at=stamp)


# ---------------------------------------------------------------------------------------------
# M4: zero measured noise does not demand a bit-identical engine


def test_m4_zero_noise_floor_uses_the_absolute_floor():
    """P7: noise 0 and balanced boundaries 10 ms off gave `fail`."""
    c = seg_cert(balanced=segment_set("balanced", 12, shift=0.01))
    assert c["metrics"]["noise"]["means"]["boundary_error_s"] == 0.0
    assert c["verdict"] == "certified", c["checks"]
    assert c["thresholds"]["per_metric"]["boundary_error_s"]["floor"] == 0.1
    # a full TempSeg tick (0.25 s at 4 fps) on every boundary still fails
    assert seg_cert(balanced=segment_set("balanced", 12, shift=0.25))["verdict"] == "fail"


# ---------------------------------------------------------------------------------------------
# M6: constant dense series


def test_m6_constant_dense_series():
    """P11: one `2` in an all-`3` series scored the maximal rank divergence."""
    d = lambda xs: DenseReward(segments=(), per_step=tuple(xs))  # noqa: E731
    a = rec("u", "reward.dense", "exact", dense=d([3] * 10))
    b = rec("u", "reward.dense", "balanced", dense=d([3] * 9 + [2]))
    assert C.dense_metrics(a, b)["rank_divergence"] == pytest.approx(0.1)
    assert C.dense_metrics(a, a)["rank_divergence"] == 0.0
    c = rec("u", "reward.dense", "balanced", dense=d([2] * 10))
    assert C.dense_metrics(a, c)["rank_divergence"] == pytest.approx(1.0)  # |delta|


def test_records_round_trip_the_binding_fields():
    r = segment_set("balanced", 1)[0]
    back = StageRecord.from_json(r.to_json())
    assert back.provenance.cert_key == r.provenance.cert_key
    assert back.provenance.embodiment == "yam"
    assert isinstance(back.provenance, Provenance)


# ---------------------------------------------------------------------------------------------
# Consumption: the planner prefers a certified balanced when it is cheaper


def test_recommended_prefers_a_cheaper_certified_balanced():
    from flora_deploy.planner.options import PlanRequest, WorkloadSummary, plan_options, recommended

    req = PlanRequest(WorkloadSummary.from_hours(100), deadline_s=96 * 3600, budget_usd=6000,
                      max_gpus=10)
    # balanced as if measured and deployable (its rates and workers are not, today): only its
    # certification differs from exact's
    opts = [replace(o, rate_status="measured", unavailable=tuple(
        u for u in o.unavailable if u.startswith("uncertified"))) if o.preset == "balanced"
        else o for o in plan_options(req)]
    bal = next(o for o in opts if o.preset == "balanced")
    ex = recommended(opts)
    assert ex.preset == "exact" and bal.cost_usd < ex.cost_usd  # uncertified: exact
    assert recommended(opts, certified={bal.name}) is bal
