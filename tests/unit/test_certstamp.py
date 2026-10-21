"""``workers_impl.certstamp``: the one helper that stamps certification provenance
(``embodiment``, ``cert_key``, ``gpu``; the sink stamps the 64-hex ``config_hash``) into the
records of **every** worker kind, from the env the factory sets, in production, local and fake
mode (``docs/certification.md``, "Provenance a record must carry"). ``flora certify``
(``validate.certify``) must accept the records' provenance and refuse fake ones only because
they are fake."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from flora_deploy.config.presets import AVAILABLE_PRESETS, PRESETS
from flora_deploy.config.schema import load_config, parse_config
from flora_deploy.export.results import iter_records
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunner
from flora_deploy.local import sample_dataset as make_sample
from flora_deploy.modal_app import factory as F
from flora_deploy.records import Provenance, StageRecord
from flora_deploy.validate import certify as C
from flora_deploy.workers_impl import certstamp as CS
from tests.unit.test_balanced_deps import balanced_yaml

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ALL_KINDS = sorted({k for n in AVAILABLE_PRESETS for k in PRESETS[n].pools})
WORKER_KINDS = [k for k in ALL_KINDS if k not in F.PENDING_WORKERS]


def _record(stage: str = "segment", **prov) -> StageRecord:
    p = Provenance(models={}, engine="e", preset="exact", config_hash="c" * 64,
                   logic_version="t", framework="flora-deploy test", **prov)
    return StageRecord(uid="u0", dataset="ds", episode_id="u0", stage=stage, provenance=p,
                       certified=True)


def _check(rec: StageRecord, preset: str, kind: str) -> None:
    p = rec.provenance
    assert C.missing_provenance(p) == [], (preset, kind, C.missing_provenance(p))
    assert HEX64.match(p.config_hash) and p.gpu and p.embodiment
    assert (p.cert_key["preset"], p.cert_key["kind"]) == (preset, kind)


# ---------------------------------------------------------------------------------------------
# Production env: exactly what ``factory.plan_app`` sets on each worker class


def _plan(preset: str):
    text = TEMPLATE.read_text().replace("stage_to: null",
                                        "stage_to: volume://flora-media/run/videos")
    if preset == "balanced":
        text = balanced_yaml(text)
    cfg = parse_config(yaml.safe_load(text))
    receipts = {n: "0" * 64 for k in PRESETS[preset].derived_weights.values() for n in k}
    return cfg, F.plan_app(cfg, derived_receipts=receipts, allow_uncertified=True,
                           missing_workers="skip")


@pytest.mark.parametrize("preset", AVAILABLE_PRESETS)
def test_production_env_resolves_every_worker_class_stamp(preset):
    cfg, plan = _plan(preset)
    assert plan.worker_classes, preset
    seen = set()
    for obj in (o for o in plan.objects if o.kind == "cls"):
        env = dict(obj.env)
        # the derived digests the worker verifies at start (stand-ins where none is pinned)
        content = {n: CS.pinned_content(preset, k).get(n, "d" * 64)
                   for k in obj.shard_kinds for n in PRESETS[preset].derived_weights.get(k, ())}
        stamps = CS.container_stamps(env, gpu="NVIDIA H200", derived_content=content)
        assert sorted(stamps) == sorted(obj.shard_kinds)
        for kind, stamp in stamps.items():
            assert stamp.embodiment == cfg.pipeline.embodiment
            assert C.document_problems(stamp.cert_key) == []
            _check(stamp.apply(_record()), preset, kind)
        seen |= set(stamps)
    assert seen == set(plan.worker_classes)
    assert seen == {k for k in PRESETS[preset].pools if k in cfg.pools()} - set(plan.skipped)


def test_exact_segcap_stamp_uses_the_pinned_bf16_content():
    content = CS.pinned_content("exact", "segcap")
    assert set(content) == {"captioner_bf16"} and HEX64.match(content["captioner_bf16"])
    _cfg, plan = _plan("exact")
    env = dict(plan.object(plan.worker_classes["segcap"]).env)
    stamp = CS.stamp_from_env("segcap", env, gpu="NVIDIA H200", derived_content=content)
    assert stamp.cert_key["derived_weights"] == content
    assert stamp.cert_key["modes"] == {"determinism": "nvidia"}


def test_fails_closed_without_env_or_content():
    _cfg, plan = _plan("exact")
    env = dict(plan.object(plan.worker_classes["segcap"]).env)
    for drop in (CS.POOL_GPUS_ENV, CS.KIND_MODES_ENV, CS.EMBODIMENT_ENV, CS.PRESET_ENV,
                 CS.SHARD_KINDS_ENV):
        with pytest.raises(CS.CertStampError):
            CS.container_stamps({k: v for k, v in env.items() if k != drop},
                                derived_content=CS.pinned_content("exact", "segcap"))
    with pytest.raises(CS.CertStampError, match="captioner_bf16"):
        CS.stamp_from_env("segcap", env, derived_content={})


def test_pending_kinds_need_only_the_helper():
    """``reward.rubric`` (no worker yet): the env the factory would set already resolves."""
    for preset in AVAILABLE_PRESETS:
        cfg, _ = _plan(preset)
        for kind in (k for k in PRESETS[preset].pools if k in F.PENDING_WORKERS):
            env = {CS.PRESET_ENV: preset, CS.SHARD_KINDS_ENV: kind,
                   **CS.cert_env(cfg, {kind: PRESETS[preset].pools[kind]})}
            for fake in (False, True):
                stamp = CS.container_stamps(env, gpu="NVIDIA H100", fake=fake)[kind]
                _check(stamp.apply(_record("reward.rubric")), preset, kind)


def test_a_records_own_fields_win():
    stamp = CS.CertStamp("yam", {"x": 1}, "NVIDIA H200")
    out = stamp.apply(_record(gpu="NVIDIA H200 NVL", embodiment="human"))
    assert (out.provenance.gpu, out.provenance.embodiment) == ("NVIDIA H200 NVL", "human")
    assert out.provenance.cert_key == {"x": 1}


# ---------------------------------------------------------------------------------------------
# Local fake mode, end to end: exact + balanced, then ``flora certify``


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("certstamp")
    make_sample(d / "videos")
    out = {}
    for preset in AVAILABLE_PRESETS:
        text = TEMPLATE.read_text()
        if preset == "balanced":
            text = balanced_yaml(text)
        (d / f"{preset}.yaml").write_text(text)
        r = LocalRunner(d / f"{preset}.yaml", store=d / f"store-{preset}")
        try:
            res = r.run()
        finally:
            r.close()
        assert res.state == "done", res.as_dict()
        st = LocalStore(d / f"store-{preset}")
        recs = {s: list(iter_records(st, r.job_id, s)) for s in STAGES}
        for s, rows in recs.items():
            assert rows, (preset, s)
            (d / f"{preset}-{s}.jsonl").write_text("".join(x.to_json() + "\n" for x in rows))
        (d / f"{preset}.jsonl").write_text(
            "".join(x.to_json() + "\n" for rows in recs.values() for x in rows))
        out[preset] = recs
    out["dir"] = d
    return out


def test_every_kind_stamps_valid_provenance_in_local_fake_mode(runs):
    covered = set()
    for preset in AVAILABLE_PRESETS:
        spec = PRESETS[preset]
        for stage, rows in runs[preset].items():
            kind = spec.stage_kinds[stage]
            for rec in rows:
                assert C.is_fake(rec.provenance)
                _check(rec, preset, kind)
            covered.add(kind)
    assert sorted(covered) == WORKER_KINDS  # every kind with a worker


def test_certify_bind_accepts_the_provenance_and_refuses_only_as_fake(runs, monkeypatch):
    for preset in AVAILABLE_PRESETS:
        for stage, rows in runs[preset].items():
            recs = {r.uid: r for r in rows}
            with pytest.raises(C.CertifyRefused) as e:
                C.bind(recs, stage, preset)
            assert e.value.reason == "fake_models"
    # test seam: were they not fake, the provenance check passes (no missing_provenance)
    monkeypatch.setattr(C, "is_fake", lambda p: False)
    for preset in AVAILABLE_PRESETS:
        for stage, rows in runs[preset].items():
            try:
                C.bind({r.uid: r for r in rows}, stage, preset)
            except C.CertifyRefused as e:  # pragma: no cover - the failure being tested
                pytest.fail(f"{preset} {stage}: refused {e.reason}: {e}")
            except C.CertifyError as e:  # fake model revisions / engine label: not provenance
                assert "lack provenance" not in str(e)


def test_flora_certify_refuses_local_fake_exact_and_balanced_only_as_fake(runs):
    from typer.testing import CliRunner

    from flora_deploy.cli._app import app

    d = runs["dir"]
    contents = [f"--derived-content={n}={v}" for k in ("segment", "caption")
                for n, v in CS.fake_content("balanced", k).items()]
    for stage in STAGES:
        res = CliRunner().invoke(app, [
            "certify", "--exact", str(d / "exact.jsonl"), "--balanced", str(d / "balanced.jsonl"),
            "--config", str(d / "balanced.yaml"), "--stage", stage, *contents])
        out = " ".join(res.output.split())
        assert res.exit_code == 1, (stage, res.output)
        assert "certified: false (fake_models)" in out, (stage, res.output)
        assert "missing_provenance" not in out


def test_records_round_trip_the_stamp(runs):
    rec = runs["exact"]["segment"][0]
    again = StageRecord.from_json(rec.to_json())
    assert again.provenance.cert_key == rec.provenance.cert_key
    assert json.loads(rec.to_json())["provenance"]["embodiment"] == "human"


def test_local_config_matches_the_runs(runs):
    for preset in AVAILABLE_PRESETS:
        cfg = load_config(runs["dir"] / f"{preset}.yaml", allow_uncertified=True)
        embodiments = {r.provenance.embodiment for rows in runs[preset].values() for r in rows}
        assert embodiments == {cfg.pipeline.embodiment}
