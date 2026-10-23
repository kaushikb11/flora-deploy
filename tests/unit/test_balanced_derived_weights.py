"""Review 10 (Wire-balanced) findings: M1 unpinned derived weights refused before any GPU, M2
validated worker shapes, and the balanced worker's ``build_stage`` split."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest
import yaml

from flora_deploy.config import receipts as rc
from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
from flora_deploy.config.schema import parse_config
from flora_deploy.modal_app import factory as f

BAL_RECEIPTS = {"captioner_bf16": "0" * 64, "tempseg_merged": "1" * 64}


def balanced_cfg(**update):
    d = yaml.safe_load((resources.files("flora_deploy.config") / "templates"
                        / "exact_mp4_dir.yaml").read_text())
    d.update(preset="balanced", allow_uncertified=True, resources={})
    d["modal"]["max_gpus"] = 4
    d.update(update)
    return parse_config(d)


@pytest.fixture
def tempseg_unpinned(monkeypatch):
    pin = replace(DERIVED_PINS_BY_KEY["tempseg-merged"], content_sha256=None)
    monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", pin)
    return pin


@pytest.fixture
def tempseg_pinned(monkeypatch):
    pin = replace(DERIVED_PINS_BY_KEY["tempseg-merged"], content_sha256="c" * 64)
    monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", pin)
    return pin


# --- M1: refused at plan / deploy time, with the fix ----------------------------------------


def test_m1_plan_app_refuses_an_unpinned_derived_weight(tempseg_unpinned):
    with pytest.raises(f.DeployError) as e:
        f.plan_app(balanced_cfg(), derived_receipts=BAL_RECEIPTS)
    msg = str(e.value)
    assert "tempseg-merged" in msg and "segment" in msg  # which weight, which kind
    assert "not pinned" in msg  # why
    assert "flora prepare --show-receipt" in msg and "preset: exact" in msg  # the fix
    assert "convert_app.py::tempseg --step equivalence" in msg


def test_m1_plan_app_accepts_a_pinned_one_and_fake_models_skip_it(tempseg_pinned, monkeypatch):
    plan = f.plan_app(balanced_cfg(), derived_receipts=BAL_RECEIPTS)
    assert plan.worker_classes["segment"] == "SegmentWorker"
    unpinned = replace(tempseg_pinned, content_sha256=None)
    monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", unpinned)
    fake = f.plan_app(balanced_cfg(), fake_models=True)  # no weights are read
    assert fake.worker_classes["segment"] == "FakeSegmentWorker"


def test_m1_caption_only_balanced_does_not_need_tempseg(tempseg_unpinned):
    cfg = balanced_cfg()
    assert rc.needed_by_kind(cfg) == {"tempseg_merged": ["segment"],
                                      "captioner_bf16": ["caption"]}
    assert rc.unpinned(["captioner_bf16", "tempseg_merged"]) == ["tempseg_merged"]
    rc.require_pinned(["captioner_bf16"])  # pinned: no error


def test_m1_flora_deploy_dry_run_refuses_with_the_fix(tempseg_unpinned, tmp_path):
    from typer.testing import CliRunner

    from flora_deploy.cli import app

    d = yaml.safe_load((resources.files("flora_deploy.config") / "templates"
                        / "exact_mp4_dir.yaml").read_text())
    d.update(preset="balanced", allow_uncertified=True, resources={},
             weights={"derived": BAL_RECEIPTS})
    (tmp_path / "data").mkdir()
    d["input"]["path"] = str(tmp_path / "data")
    p = tmp_path / "flora.yaml"
    p.write_text(yaml.safe_dump(d))
    r = CliRunner().invoke(app, ["deploy", "--config", str(p), "--dry-run"])
    assert r.exit_code != 0
    text = " ".join(r.output.split())
    assert "tempseg-merged" in text and "not pinned" in text and "--show-receipt" in text


def test_m1_doctor_fails_with_the_fix(tempseg_unpinned):
    from types import SimpleNamespace

    from flora_deploy.cli import doctor as d
    from tests.unit.test_doctor import FakeProbes

    cfg = balanced_cfg()
    cfg = SimpleNamespace(**{k: getattr(cfg, k) for k in ("modal", "preset", "pipeline")},
                          weights=SimpleNamespace(derived=dict(BAL_RECEIPTS)))
    checks = {c.name: c for c in d.check_weights(FakeProbes(), cfg)}
    c = checks["weights.derived.pins"]
    assert c.status == "fail" and "tempseg-merged" in c.message
    assert "--show-receipt" in c.fix and "preset: exact" in c.fix


def test_m1_worker_message_is_the_shared_one(tempseg_unpinned, tmp_path):
    from flora_deploy.modal_app import balanced_app as ba

    with pytest.raises(ba.DerivedWeightsError) as e:
        ba.check_pinned_kind("segment", "balanced")
    assert isinstance(e.value, ba.UnpinnedDerivedWeights)
    assert "--show-receipt" in str(e.value) and "preset: exact" in str(e.value)
    ba.check_pinned_kind("caption", "balanced")  # captioner-bf16 is pinned


# --- M1: a refused start ends the shard at once (no Modal retry, no respawn, not D15) --------


def test_m1_refused_start_fails_the_shard_terminally(tmp_path):
    from flora_deploy.jobs import attempts as A
    from flora_deploy.jobs.deps import StoreAttemptLedger
    from flora_deploy.jobs.manifest import save_manifest
    from flora_deploy.jobs.stores.local import LocalStore
    from flora_deploy.jobs.types import EpisodeRef, Manifest, Shard
    from flora_deploy.modal_app import balanced_app as ba

    store = LocalStore(tmp_path / "s")
    jf = A.JobFiles(store, "job", "cfg", stage_hashes={})
    ep = EpisodeRef(uid="ds--a", dataset="ds", episode_id="a", pool="human", duration_s=1.0,
                    oracle_eligible=True, envelope_ok=True, index_ref="idx")
    shard = Shard("segment-00000", "segment", ("ds--a",), 1.0)
    save_manifest(store, Manifest("job", "cfg", (ep,), (shard,), "2026-10-22T00:00:00Z"))
    runner = ba.RefusedRunner(store, "tempseg-merged is not pinned", container_id="ct-1",
                              call_id=lambda: "fc-1")
    token = A.spawn_token("job", "cfg", shard.shard_id, 0)
    assert runner.run_shard("job", shard.shard_id, "cfg", token=token) is None  # no Modal retry
    (m,) = A.markers(jf, "segment", shard.shard_id)
    assert m.error and "not pinned" in m.error and not m.crashed  # never D15 (no poison pill)
    assert A.counted_starts(jf, "segment", shard.shard_id) == 0
    failed = StoreAttemptLedger(jf, limit=3).failed()
    assert failed[shard.shard_id].reason == ba.DEPLOYMENT_REFUSED
    assert "not pinned" in failed[shard.shard_id].detail["error"]
    assert runner.preempt() is False


# --- M2: the deployed worker shapes are the lab-validated ones -------------------------------


def test_m2_balanced_pools_match_the_validated_lab_shape():
    from flora_deploy.config.presets import get_preset
    from flora_deploy.modal_app import balanced_lab as lab

    pools = get_preset("balanced").pools
    for kind, lab_kind in (("segment", "tempseg"), ("caption", "captioner")):
        assert pools[kind].cpu_cores >= lab.CPU, kind
        assert pools[kind].memory_gib >= lab.MEM_GIB[lab_kind], kind
    assert pools["caption"].memory_gib >= 96 and pools["segment"].cpu_cores >= 8


def test_m2_planner_prices_follow_the_pool_shape(tempseg_pinned):
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.planner.options import PlanRequest, WorkloadSummary, plan_options

    plan = f.plan_app(balanced_cfg(), derived_receipts=BAL_RECEIPTS)
    cap, seg = plan.object("CaptionWorker"), plan.object("SegmentWorker")
    assert cap.memory[0] == 96 * 1024 and seg.cpu == 8.0
    r = pinned_rates()
    opts = plan_options(PlanRequest(workload=WorkloadSummary.from_hours(1.0),
                                    presets=("balanced",)), rates=r)
    assert opts
    for o in opts:
        for k in o.kinds:
            want = {"segment": ("L40S", 8.0, 48.0), "caption": ("H100!", 8.0, 96.0)}[k.kind]
            assert (k.gpu, k.cpu_cores, k.memory_gib) == want
            per_s = r.per_second(k.gpu, k.gpu_count, k.cpu_cores, k.memory_gib)
            assert k.cost_usd == pytest.approx(per_s * (k.work_s + k.containers * k.startup_s),
                                               abs=1e-3)


def test_m2_docs_state_the_deployed_shape():
    text = (Path(__file__).resolve().parents[2] / "docs" / "balanced-engines.md").read_text()
    assert "8 cores, 48 GiB" in text and "8 cores, 96 GiB" in text


# --- build_stage: the balanced worker's stage, reusable by a future online service ----------


class _Spec:
    label = "vllm-0.24.0"

    def __init__(self, kind, model, gpu):
        self.kind, self.model, self.gpu = kind, model, gpu

    def hashed(self):
        return {"kind": self.kind, "gpu": self.gpu}

    def as_json(self):
        return {"kind": self.kind, "model": self.model, "gpu": self.gpu}


@pytest.fixture
def fake_stack(monkeypatch, tmp_path, tempseg_pinned):
    """Every heavy dependency of the balanced worker start replaced by a recorder."""
    from types import SimpleNamespace

    from flora_deploy.accel import captioner_load
    from flora_deploy.engines import offline_llm
    from flora_deploy.jobs.stores.local import LocalStore
    from flora_deploy.modal_app import balanced_app as ba
    from flora_deploy.modal_app import workers_app
    from flora_deploy.nv import linker
    from flora_deploy.stages import caption as stc
    from flora_deploy.stages import segment as sts
    from flora_deploy.workers_impl import balanced as wb
    from flora_deploy.workers_impl import certstamp

    calls: list[str] = []
    store = LocalStore(tmp_path / "store")
    monkeypatch.setattr(workers_app, "store_from_env", lambda env: store)
    monkeypatch.setattr(workers_app, "preflight",
                        lambda env, store: calls.append("preflight") or {"store": "s"})
    monkeypatch.setattr(workers_app, "gpu_name", lambda: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(ba, "derived_dir", lambda n: tmp_path / n)
    monkeypatch.setattr(ba, "check_extra_state", lambda d: [])
    cache = SimpleNamespace(env={}, status=SimpleNamespace(as_json=lambda: {"state": "cold"}))

    def prepare_cache(kind, spec, content, env):
        calls.append(f"cache:{kind}:{content[:1]}")
        return cache

    monkeypatch.setattr(ba, "prepare_cache", prepare_cache)
    monkeypatch.setattr(captioner_load, "full_verify_default", lambda: True)

    def verify(name, directory, recorded, *, background=False, **_):
        calls.append(f"verify:{name}:{recorded}:{background}")
        return {"name": name, "dir": str(directory), "receipt_sha256": recorded,
                "content_sha256": "d" * 64,
                "files": lambda: calls.append(f"files:{name}") or {"ok": True}}

    monkeypatch.setattr(ba, "verify_derived_weights", verify)
    stamp = certstamp.CertStamp(embodiment="human", cert_key={"k": 1}, gpu="H100")

    def stamp_from_env(kind, env, *, gpu=None, derived_content=None, **_):
        calls.append(f"stamp:{kind}:{gpu}:{sorted(derived_content)}")
        return stamp

    monkeypatch.setattr(certstamp, "stamp_from_env", stamp_from_env)
    nv = SimpleNamespace(models=SimpleNamespace(
        processor=lambda kind, d: calls.append(f"processor:{kind}") or f"proc-{kind}"))
    monkeypatch.setattr(linker, "load_family", lambda fam, path: calls.append("nv") or nv)
    monkeypatch.setattr(offline_llm, "spec_for",
                        lambda kind, model, gpu: calls.append(f"spec:{kind}:{gpu}")
                        or _Spec(kind, model, gpu))

    class Engine:
        gdn_dtypes = {"A_log": "float32"}

        def __init__(self, spec):
            calls.append("engine")
            self.spec = spec

    monkeypatch.setattr(offline_llm, "OfflineEngine", Engine)
    monkeypatch.setattr(sts, "SegmentStage", lambda nv_, proc: SimpleNamespace(nv=nv_, proc=proc))
    monkeypatch.setattr(stc, "CaptionStage", lambda nv_, proc: SimpleNamespace(nv=nv_, proc=proc))
    from flora_deploy.workers_impl import caption as wc
    from flora_deploy.workers_impl import segment as ws

    def job_cls(stage_name):
        class Job:
            stage = stage_name

            def __init__(self, st, info, inputs, segments=None):
                self.st, self.info, self.inputs = st, info, inputs
                self.segments = segments

        return Job

    monkeypatch.setattr(ws, "SegmentEpisodeJob", job_cls("segment"))
    monkeypatch.setattr(wc, "CaptionEpisodeJob", job_cls("caption"))
    active = SimpleNamespace(provenance=lambda: {"patches": ["p"]}, episode=None)
    monkeypatch.setattr(wb, "apply_accel", lambda nv_: calls.append("accel") or active)
    env = {"FLORA_PRESET": "balanced",
           "FLORA_POOL_GPUS": json.dumps({"segment": "L40S", "caption": "H100!"}),
           "FLORA_DERIVED_RECEIPTS": json.dumps(BAL_RECEIPTS),
           "FLORA_MEDIA_ROOT": "/media"}
    return SimpleNamespace(calls=calls, store=store, stamp=stamp, active=active, env=env)


def _expected_calls(kind):
    name = {"segment": "tempseg_merged", "caption": "captioner_bf16"}[kind]
    proc = {"segment": "tempseg", "caption": "captioner"}[kind]
    gpu = "L40S" if kind == "segment" else "H100!"
    return name, gpu, [
        "preflight", f"verify:{name}:{BAL_RECEIPTS[name]}:True",
        f"stamp:{kind}:NVIDIA H100 80GB HBM3:['{name}']", f"spec:{kind}:{gpu}",
        f"cache:{kind}:d", "nv", f"processor:{proc}", "engine", f"files:{name}", "accel"]


@pytest.mark.parametrize("kind", ["segment", "caption"])
def test_build_runner_order_and_parts(fake_stack, kind):
    """The start order (fail closed: weights, stamp, then the engine; files joined before
    ready) and the runner's parts. Pinned before and after the build_stage split."""
    from flora_deploy.modal_app import balanced_app as ba
    from flora_deploy.workers_impl.balanced import AccelJob, StampedJob

    runner, startup = ba.build_runner(kind, fake_stack.env)
    name, gpu, calls = _expected_calls(kind)
    assert fake_stack.calls == calls
    assert runner.store is fake_stack.store and runner.engine.spec.kind == kind
    assert runner.settings.gpu == "NVIDIA H100 80GB HBM3"
    job = runner.job
    assert isinstance(job, StampedJob) and job.stamp is fake_stack.stamp
    assert isinstance(job.job, AccelJob) and job.job.active is fake_stack.active
    inner = job.job.job
    assert inner.stage == kind
    assert inner.info.engine_label == "vllm-0.24.0" and inner.info.engine_spec == {
        "kind": kind, "gpu": gpu}
    assert inner.info.config_extra == {"derived": {name: "d" * 64}}
    assert startup["kind"] == kind and startup["derived_files"] == {name: True}
    assert startup["spec"]["gpu"] == gpu and startup["accel"] == {"patches": ["p"]}
    assert set(startup["marks"]) == {"preflight", "derived_receipts", "cert_key",
                                     "compile_cache", "engine", "derived_files"}
    assert startup["compile_cache"] == {"state": "cold"}
    assert runner.compile_cache is not None


@pytest.mark.parametrize("kind", ["segment", "caption"])
def test_build_stage_parts_and_stamp_fn(fake_stack, kind):
    from flora_deploy.modal_app import balanced_app as ba
    from flora_deploy.workers_impl.balanced import AccelJob

    st = ba.build_stage(kind, fake_stack.env)
    assert fake_stack.calls == _expected_calls(kind)[2]
    engine, job, stamp_fn, info = st[:4]
    assert engine.spec.kind == kind and isinstance(job, AccelJob) and job.job.info is info
    assert stamp_fn() is fake_stack.stamp  # the batch stamp: FLORA_EMBODIMENT
    other = stamp_fn("yam")  # a service's per-request embodiment: the same cert_key
    assert other.embodiment == "yam" and other.cert_key == fake_stack.stamp.cert_key
    assert st.store is fake_stack.store and st.startup["kind"] == kind


def test_refused_start_serves_a_refusing_runner(fake_stack, monkeypatch):
    from flora_deploy.modal_app import balanced_app as ba

    pin = replace(DERIVED_PINS_BY_KEY["tempseg-merged"], content_sha256=None)
    monkeypatch.setitem(DERIVED_PINS_BY_KEY, "tempseg-merged", pin)
    with pytest.raises(ba.UnpinnedDerivedWeights) as e:
        ba.build_stage("segment", fake_stack.env)
    assert e.value.store is fake_stack.store
    assert fake_stack.calls == []  # refused before preflight, any probe or the engine
    real = ba.build_runner
    monkeypatch.setattr(ba, "build_runner", lambda kind, **kw: real(kind, fake_stack.env, **kw))
    runner, startup = ba._start("segment")  # what @modal.enter runs: it does not raise
    assert isinstance(runner, ba.RefusedRunner) and runner.store is fake_stack.store
    assert "tempseg-merged" in startup["refused"]
    runner, startup = ba._start("caption")  # the Captioner is pinned: unaffected
    assert not isinstance(runner, ba.RefusedRunner)


# --- L3: the accel identity test digests what vLLM receives -----------------------------------


def test_l3_prompt_dict_digest_sees_wrapping_dtype_and_mm_kwargs():
    import numpy as np

    from flora_deploy.engines.base import VllmRequest
    from flora_deploy.stages import segment as S
    from tests.fakes.request_digest import prompt_dict_digest

    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    md = {"fps": 2.0}
    wrapped = f"a {S.VISION_START}{S.VIDEO_TOKEN}{S.VISION_END} b"
    base = VllmRequest(prompt=wrapped, videos=((frames, md),), key="k")
    d = prompt_dict_digest(base)
    assert d == prompt_dict_digest(replace(base, key="other"))  # the key is not sent
    bare = replace(base, prompt=S.unwrap_prompt(wrapped))
    assert S.request_digest(bare.prompt, bare.videos) == S.request_digest(wrapped, base.videos)
    assert prompt_dict_digest(bare) != d  # the bare digest misses the wrapping
    as_i8 = replace(base, videos=((frames.view(np.int8), md),))
    assert prompt_dict_digest(as_i8) != d  # same bytes, other dtype
    assert prompt_dict_digest(replace(base, mm_kwargs={"fps": 1})) != d
