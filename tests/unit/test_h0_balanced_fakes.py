"""``flora deploy --fake-models`` with ``preset: balanced``: the factory plans CPU
``FakeSegmentWorker`` / ``FakeCaptionWorker``, and their runners (``workers_app.
fake_balanced_runner`` + ``FakeFaultHook``, configured only through the env the factory sets)
complete a job driven by G2's local orchestrator, with the store-file fault spec the H0 script
writes: one preemption and one poison episode, poisoned at ``segment`` in one run and at
``caption`` in the other."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flora_deploy.config.schema import load_config
from flora_deploy.export.results import iter_records
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.local import LocalRunner
from flora_deploy.local import sample_dataset as make_sample
from flora_deploy.local.faults import ContainerCrash
from flora_deploy.modal_app import factory as F
from flora_deploy.modal_app import workers_app as W
from tests.unit.test_balanced_deps import balanced_yaml

TEMPLATE = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/local_demo.yaml"
STAGES = ("segment", "caption", "reward.terminal", "reward.dense")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("h0bal")
    text = balanced_yaml(TEMPLATE.read_text()).replace(
        "stage_to: null", "stage_to: volume://flora-h0-media/run/videos")
    (d / "flora.yaml").write_text(text)
    make_sample(d / "videos")
    return d / "flora.yaml"


def test_fake_balanced_plan(demo):
    cfg = load_config(demo)
    fake = F.plan_app(cfg, fake_models=True)
    assert fake.worker_classes == {"segment": "FakeSegmentWorker",
                                   "caption": "FakeCaptionWorker",
                                   "reward.terminal": "FakeRewardWorker",
                                   "reward.dense": "FakeRewardWorker"}
    assert all(o.gpu is None for o in fake.objects)
    env = fake.object("FakeCaptionWorker").env
    assert json.loads(env["FLORA_DERIVED_RECEIPTS"]) == {"captioner_bf16": F.FAKE_RECEIPT}
    assert env["FLORA_EMBODIMENT"] == "human"
    _funcs, classes = F.registered(F.materialize(fake))
    assert {"FakeSegmentWorker", "FakeCaptionWorker"} <= set(classes)


def cert_env_of(plan, kind: str) -> dict:
    """The certification env of ``kind``'s own worker class in ``plan``."""
    from flora_deploy.workers_impl import certstamp as CS

    if plan is None:
        return {}
    env = plan.object(plan.worker_classes[kind]).env
    return {k: env[k] for k in (CS.POOL_GPUS_ENV, CS.KIND_MODES_ENV, CS.EMBODIMENT_ENV,
                                CS.SHARD_KINDS_ENV)}


class _DeployedFakes(LocalRunner):
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
        if kind in ("segment", "caption"):
            runner = W.fake_balanced_runner(kind, self.store, env, hook, call_id=call_id)
            inner = runner.run_shard

            def run_shard(job_id, shard_id, config_hash, token=""):  # FakeBalancedWorker._run
                hook.start_shard(job_id, shard_id, kind)
                runner.flora_hook_slot[0] = hook.bound()
                return inner(job_id, shard_id, config_hash, token=token)
        else:
            runner = W.fake_reward_runner(self.store, env, hook, call_id=call_id)
            inner = runner.run_shard

            def run_shard(job_id, shard_id, config_hash, token=""):
                hook.start_shard(job_id, shard_id, kind)
                return inner(job_id, shard_id, config_hash, token=token)

        runner.run_shard = run_shard
        return runner


@pytest.mark.parametrize("poison_at", ["segment", "caption"])
def test_deployed_balanced_fakes_with_preempt_and_poison(demo, tmp_path, poison_at):
    cfg = load_config(demo)
    plan = F.plan_app(cfg, fake_models=True)
    env = dict(plan.object("FakeSegmentWorker").env)
    store_dir = tmp_path / "store"
    env.update({"FLORA_STORE_KIND": "local", "FLORA_STORE_ROOT": str(store_dir)})
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
            {"preempt": {"segment": 2}, "poison": {poison_at: [poison]}}).encode())
        r.plan()
        outcome = r.orchestrate()
        summary = r.compose(outcome)
        res = r.result(outcome, summary, 0.0)
    finally:
        r.close()
    assert res.state == "completed_with_errors", res.as_dict()
    assert r.preempted == 1
    lost = {"segment": {"segment", "caption"}, "caption": {"caption"}}[poison_at]
    for stage in STAGES:
        got = sorted(x.uid for x in iter_records(st, r.job_id, stage))
        want = [u for u in uids if u != poison] if stage in lost else uids
        assert got == want, stage
        assert summary["stages"][stage]["excused"] == (1 if stage in lost else 0)
    assert summary["poison_pills"] == [f"{poison}/{poison_at}"]
    assert [p.split(":")[0] for p in summary["problems"]] == [poison_at]  # counted once
    jf = A.JobFiles(st, r.job_id, res.config_hash)
    seg = [s for s in r.manifest.shards if s.kind == "segment" and not s.bisected_from]
    assert any(m.preempted for s in seg for m in A.markers(jf, "segment", s.shard_id))
    caps = list(iter_records(st, r.job_id, "caption"))
    assert caps and {x.provenance.certificate for x in caps} == {"fake-models"}
    assert {x.provenance.embodiment for x in caps} == {"human"}


def _h0():
    import importlib.util
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    if "h0_rehearsal" in sys.modules:
        return sys.modules["h0_rehearsal"]
    spec = importlib.util.spec_from_file_location("h0_rehearsal", scripts / "h0_rehearsal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h0_rehearsal"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_h0_balanced_jobs_cover_the_same_episodes_and_both_poison_kinds(tmp_path, monkeypatch):
    h0 = _h0()
    bal = [j for j in h0.JOBS if j.preset == "balanced"]
    assert {j.data for j in bal} == set(h0.EPISODES)  # the same 6 episodes
    assert {j.poison_at for j in bal} == {None, "segment", "caption"}
    seg, cap = h0.JOB_SPECS["bal-yam-seg"], h0.JOB_SPECS["bal-yam-cap"]
    assert seg.faults("u") == {"preempt": {"segment": 2}, "poison": {"segment": ["u"]}}
    assert cap.faults("u") == {"preempt": {"caption": 2}, "poison": {"caption": ["u"]}}
    assert [s for s in STAGES if seg.lost(s)] == ["segment", "caption"]
    assert [s for s in STAGES if cap.lost(s)] == ["caption"]
    assert all(h0.JOB_SPECS["yam"].lost(s) for s in STAGES)
    for data in h0.EPISODES:
        (tmp_path / data).mkdir()
    monkeypatch.chdir(tmp_path)
    for j in h0.JOBS:
        pl = h0.plans(h0.write_config(tmp_path, j.name, "run1"))
        f = pl["fake"]
        assert f.preset == j.preset and all(o.gpu is None for o in f.objects)
        assert "-fake-" in f.app_name and f.app_name != pl["real"].app_name
        if j.preset == "balanced":
            assert (f.worker_classes["segment"], f.worker_classes["caption"]) == \
                ("FakeSegmentWorker", "FakeCaptionWorker")
            assert pl["real"].worker_classes["segment"] == "SegmentWorker"
