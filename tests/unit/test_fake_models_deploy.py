"""``flora deploy --fake-models`` (H0): the factory's fake-models plan, its hash separation, and
the deployed fake workers' runners (``workers_app.fake_segcap_runner`` / ``fake_reward_runner``
with ``FakeFaultHook``) driven end to end by G2's local orchestrator with the store-file fault
spec the H0 script writes (preemption + poison)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flora_deploy.config.hashing import config_hash, hash_document
from flora_deploy.config.schema import load_config
from flora_deploy.export.results import iter_records
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunner
from flora_deploy.local import sample_dataset as make_sample
from flora_deploy.local.faults import ContainerCrash
from flora_deploy.modal_app import factory as F
from flora_deploy.modal_app import workers_app as W

TEMPLATE = Path(__file__).resolve().parents[2] / "tests/fixtures/local_demo_exact.yaml"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("h0demo")
    text = TEMPLATE.read_text().replace("stage_to: null",
                                        "stage_to: volume://flora-h0-media/run/videos")
    (d / "flora.yaml").write_text(text)
    make_sample(d / "videos")
    return d / "flora.yaml"


def _fake_hashes(cfg):
    return config_hash(cfg, derived_receipts={"captioner_bf16": F.FAKE_RECEIPT},
                       image_fingerprint=lambda n: "img").stage_hashes


def test_hash_component_is_additive_and_separates(demo):
    cfg = load_config(demo)
    kw = {"derived_receipts": {"captioner_bf16": "r"}, "image_fingerprint": lambda n: "img"}
    doc = hash_document(cfg, **kw)
    assert "fake_models" not in doc["shared"]  # real documents are unchanged
    real = config_hash(cfg, **kw)
    fake = config_hash(cfg, fake_models=True, **kw)
    assert fake.document["shared"]["fake_models"] is True
    assert real.hash != fake.hash
    assert all(real.stage_hashes[k] != fake.stage_hashes[k] for k in real.stage_hashes)


def test_fake_plan_is_the_real_app_on_cpu(demo):
    cfg = load_config(demo)
    real = F.plan_app(cfg, derived_receipts={"captioner_bf16": F.FAKE_RECEIPT})
    fake = F.plan_app(cfg, fake_models=True)  # no --receipt needed
    assert fake.fake_models and not real.fake_models
    assert "-fake-" in fake.app_name and fake.app_name != real.app_name
    assert fake.watch_dict != real.watch_dict
    assert all(fake.stage_hashes[k] != real.stage_hashes[k] for k in real.stage_hashes)
    assert set(fake.worker_classes) == set(real.worker_classes)
    assert fake.worker_classes == {"segcap": "FakeSegcapWorker",
                                   "reward.terminal": "FakeRewardWorker",
                                   "reward.dense": "FakeRewardWorker"}
    by = {o.name: o for o in fake.objects}
    rb = {o.name: o for o in real.objects}
    assert {o.name for o in fake.objects} == ({o.name for o in real.objects}
                                              - {"ReferenceWorker", "RewardWorker",
                                                 F.WARM_CACHE_FUNCTION}
                                              | {"FakeSegcapWorker", "FakeRewardWorker"})
    for fake_cls, real_cls in (("FakeSegcapWorker", "ReferenceWorker"),
                               ("FakeRewardWorker", "RewardWorker")):
        o, r = by[fake_cls], rb[real_cls]
        assert o.gpu is None and o.image == "img_cpu" and r.gpu
        assert o.cpu == F.FAKE_WORKER_CPU
        # everything but resources and hashes is the real deployment's
        assert (o.volumes, o.secrets, o.timeout, o.retries, o.max_containers,
                o.scaledown_window, o.shard_kinds) == (r.volumes, r.secrets, r.timeout,
                                                       r.retries, r.max_containers,
                                                       r.scaledown_window, r.shard_kinds)
        diff = {k for k in set(o.env) | set(r.env) if o.env.get(k) != r.env.get(k)}
        assert diff <= {F.APP_ENV, "FLORA_CONFIG_HASH", "FLORA_STAGE_HASHES",
                        F.FAKE_MODELS_ENV}, diff
        assert o.env[F.FAKE_MODELS_ENV] == "1" and F.FAKE_MODELS_ENV not in r.env
        assert o.env["FLORA_MEDIA_ROOT"] == "/media/flora-h0-media/run/videos"
    assert F.WARM_CACHE_FUNCTION in fake.skipped
    assert all(o.gpu is None for o in fake.objects)
    rec = F.deployment_record(fake)
    assert rec["fake_models"] is True and rec["certificate"] == "fake-models"
    assert "fake_models" not in F.deployment_record(real)
    assert fake.tags["flora-fake-models"] == "true"
    # the fake app materializes offline (Modal resolves the classes by module + qualname)
    funcs, classes = F.registered(F.materialize(fake))
    assert {"FakeSegcapWorker", "FakeRewardWorker"} <= set(classes)
    assert W.FAKE_MODELS_ENV == F.FAKE_MODELS_ENV and W.FAKE_CERTIFICATE == F.FAKE_CERTIFICATE


def test_determinism_flows_into_fake_hash_and_env(demo):
    cfg = load_config(demo)
    a = F.plan_app(cfg, fake_models=True, determinism="deterministic")
    b = F.plan_app(cfg, fake_models=True, determinism="nvidia")
    assert a.app_name != b.app_name
    env = a.object("FakeSegcapWorker").env
    assert env["FLORA_DETERMINISM"] == "deterministic"
    assert W.configured_determinism_env(env) == "deterministic"


def test_preempt_matches():
    assert W.preempt_matches("segcap", "segcap-0123456789abcdef")
    assert not W.preempt_matches("segcap", "segcap-ep-ds--ep1")
    assert not W.preempt_matches("segcap", "reward.dense-0123")
    assert W.preempt_matches("reward.*", "reward.dense-0123")


def test_preempt_fires_once_when_two_shards_reach_it_together(tmp_path):
    """``preempt`` is once per target and job. Two segcap containers reaching their 2nd episode
    together both saw no fired file and both preempted (``r.preempted == 2`` in the end-to-end
    test below, 2 of 40 runs under xdist): the store holds both hooks inside the check until
    both arrived (or 1 s passed), so without an atomic check-and-claim this always fires
    twice."""
    import threading

    inner, gate = LocalStore(tmp_path), threading.Barrier(2, timeout=1.0)

    class Racing:
        def __getattr__(self, name):
            return getattr(inner, name)

        def exists(self, path):
            if "/fake_faults/fired/" in path:
                try:
                    gate.wait()
                except threading.BrokenBarrierError:
                    pass
            return inner.exists(path)

    store, job = Racing(), "j.fake"
    inner.put_bytes(W.fake_faults_path(job), json.dumps({"preempt": {"segcap": 2}}).encode())
    fired: list[str] = []

    def container(shard_id):
        def preempt(_wait):
            fired.append(shard_id)
            raise KeyboardInterrupt

        hook = W.FakeFaultHook(store, preempt_fn=preempt, sleep=lambda s: None)
        hook.start_shard(job, shard_id)
        try:
            for uid in ("ds--a", "ds--b"):
                hook.before_episode(uid)
        except KeyboardInterrupt:
            pass

    ts = [threading.Thread(target=container, args=(f"segcap-{c * 16}",)) for c in "ab"]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert len(fired) == 1, fired


def test_fake_workers_refuse_outside_a_fake_app():
    with pytest.raises(W.PreflightError, match=W.FAKE_MODELS_ENV):
        W.fake_preflight({"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": "/nonexistent"})


def cert_env_of(plan, kind: str) -> dict:
    """The certification env of ``kind``'s own worker class in ``plan``."""
    from flora_deploy.workers_impl import certstamp as CS

    if plan is None:
        return {}
    env = plan.object(plan.worker_classes[kind]).env
    return {k: env[k] for k in (CS.POOL_GPUS_ENV, CS.KIND_MODES_ENV, CS.EMBODIMENT_ENV,
                                CS.SHARD_KINDS_ENV)}


class _DeployedFakes(LocalRunner):
    """G2's local orchestrator + spawner, but each container runs the *deployed* fake workers'
    runners and fault hook, configured only through the env the factory sets. Spend is priced
    as the deployed app's CPU containers (``RunSettings.fake_models``), as ``sdk.submit`` does
    for a ``--fake-models`` plan."""

    bill_as_fake_app = True

    def __init__(self, *a, env: dict, app_plan=None, **kw):
        super().__init__(*a, **kw)
        self.env = env
        self.app_plan = app_plan  # each kind's own worker-class env for its certification stamp
        self.preempted = 0

    def _runner(self, kind, container_id, call_id):
        env = {**self.env, **cert_env_of(self.app_plan, kind),
               "FLORA_MEDIA_ROOT": str(self.media_root)}

        def crash(code):
            raise ContainerCrash(f"os._exit({code})")

        def preempt(_wait):
            self.preempted += 1
            raise KeyboardInterrupt("simulated preemption")

        hook = W.FakeFaultHook(self.store, exit_fn=crash, preempt_fn=preempt,
                               sleep=lambda s: None)
        make = W.fake_segcap_runner if kind == "segcap" else W.fake_reward_runner
        runner = make(self.store, env, hook, call_id=call_id)
        inner = runner.run_shard

        def run_shard(job_id, shard_id, config_hash, token=""):
            hook.start_shard(job_id, shard_id)
            return inner(job_id, shard_id, config_hash, token=token)

        runner.run_shard = run_shard
        return runner


def test_deployed_fake_runners_end_to_end_with_preempt_and_poison(demo, tmp_path):
    cfg = load_config(demo)
    plan = F.plan_app(cfg, fake_models=True, determinism="deterministic")
    env = dict(plan.object("FakeSegcapWorker").env)
    env.update({"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(tmp_path / "store")})
    store_dir = tmp_path / "store"
    r = _DeployedFakes(cfg, config_path=demo, store=store_dir, env=env, app_plan=plan)
    try:
        r.index()
        uids = sorted(json.loads(p.read_text())["uid"]
                      for p in (store_dir / "flora").rglob("*.json")
                      if json.loads(p.read_text()).get("schema") == "flora_index_receipt_v1"
                      and json.loads(p.read_text()).get("uid"))
        poison = uids[-1]
        st = LocalStore(store_dir)
        st.put_bytes(W.fake_faults_path(r.job_id), json.dumps(
            {"preempt": {"segcap": 2}, "poison": [poison]}).encode())
        r.plan()
        outcome = r.orchestrate()
        summary = r.compose(outcome)
        res = r.result(outcome, summary, 0.0)
    finally:
        r.close()
    assert res.state == "completed_with_errors", res.as_dict()
    assert r.preempted == 1
    for stage in STAGES:
        recs = list(iter_records(st, r.job_id, stage))
        got = [x.uid for x in recs]
        assert sorted(got) == [u for u in uids if u != poison], stage
        assert {x.provenance.certificate for x in recs} == {"fake-models"}
        assert {v for x in recs for v in x.provenance.models.values()} == {"fake"}
        assert summary["stages"][stage]["excused"] == 1
    assert any(poison in p for p in summary["poison_pills"])
    jf = A.JobFiles(st, r.job_id, res.config_hash)
    seg = [s for s in r.manifest.shards if s.kind == "segcap" and not s.bisected_from]
    assert any(m.preempted for s in seg for m in A.markers(jf, "segcap", s.shard_id))
    audits = [json.loads(line) for p in store_dir.rglob("*.jsonl") if "/audit/segcap/" in
              p.as_posix() for line in p.read_text().splitlines()]
    assert audits and {a["runtime"]["determinism"]["mode"] for a in audits} == {"deterministic"}
    shutil.rmtree(store_dir, ignore_errors=True)
