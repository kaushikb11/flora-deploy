"""``modal_app.factory``: plan and build the per-config Modal app, offline (no Modal calls)."""

from __future__ import annotations

import re
import sys
import types
from importlib import resources

import modal
import pytest
import yaml

from flora_deploy.config.schema import parse_config
from flora_deploy.layout import CACHE_MOUNT, WEIGHTS_MOUNT
from flora_deploy.modal_app import factory as f

RECEIPTS = {"captioner_bf16": "0" * 64}


class RewardWorker:
    """Stand-in for ``workers_app.RewardWorker`` (module scope, undecorated)."""

    @modal.enter()
    def start(self) -> None:  # pragma: no cover - never runs
        pass

    @modal.method()
    def run_shard(self, job_id: str, shard_id: str, config_hash: str, token: str = ""):
        return None  # pragma: no cover


@pytest.fixture
def fake_workers_app(monkeypatch):
    mod = types.ModuleType("flora_deploy.modal_app.workers_app")
    mod.RewardWorker = RewardWorker
    monkeypatch.setitem(sys.modules, "flora_deploy.modal_app.workers_app", mod)
    return mod


def template(name: str) -> dict:
    text = (resources.files("flora_deploy.config") / "templates" / name).read_text()
    return yaml.safe_load(text)


def exact_cfg(**changes):
    d = template("exact_mp4_dir.yaml")
    d.update(changes)
    return parse_config(d)


def rewards_cfg(resources_=None, **changes):
    d = template("rewards_only.yaml")
    if resources_ is not None:
        d["resources"] = resources_
    d.update(changes)
    return parse_config(d)


# ---------------------------------------------------------------------------------------------
# exact (segcap -> ReferenceWorker)


def test_exact_plan_and_app():
    plan = f.plan_app(exact_cfg(), derived_receipts=RECEIPTS)
    assert plan.worker_classes == {"segcap": "ReferenceWorker"}
    assert [o.name for o in plan.objects] == ["ReferenceWorker", "orchestrator_run", "reconcile"]
    w = plan.object("ReferenceWorker")
    assert (w.kind, w.module, w.image) == ("cls", "flora_deploy.modal_app.reference_app",
                                           "img_hfref")
    assert w.gpu == "H200" and w.cpu == 8.0
    assert w.memory == (96 * 1024, int(96 * 1024 * f.MEMORY_LIMIT_FACTOR))
    assert (w.min_containers, w.max_containers) == (0, 2)  # resources.segcap.max_containers
    assert w.max_inputs is None  # 1 input per container: no @modal.concurrent
    assert w.retries == f.WORKER_RETRIES and w.timeout >= f.MIN_WORKER_TIMEOUT_S
    assert w.volumes[str(WEIGHTS_MOUNT)] == f.VolumeMount("flora-weights", read_only=True)
    assert str(CACHE_MOUNT) not in w.volumes  # NVIDIA's HF reference compiles nothing
    # the store (sink volume://flora-results/my-mp4-labels) is a mounted Volume
    assert w.volumes["/vol/flora-results"] == f.VolumeMount("flora-results", create=True)
    assert w.env["FLORA_STORE_ROOT"] == "/vol/flora-results/my-mp4-labels"
    assert w.env["HF_HUB_OFFLINE"] == "1" and "FLORA_MODEL_CACHE" in w.env
    assert w.env["FLORA_APP_NAME"] == plan.app_name
    assert w.env["FLORA_SHARD_KINDS"] == "segcap" and "CUBLAS_WORKSPACE_CONFIG" not in w.env
    assert w.secrets == ()
    orch = plan.object("orchestrator_run")
    assert orch.nonpreemptible and orch.gpu is None and orch.cpu <= 0.5
    assert orch.timeout == 24 * 3600 and orch.retries == f.ORCH_RETRIES
    assert orch.env["FLORA_WATCH_DICT"] == plan.watch_dict
    assert set(plan.image_fingerprints) == {"img_hfref", "img_cpu"}
    assert "reconcile_watched" in plan.skipped and "index" in plan.skipped

    app = f.build_app(plan)
    assert app.name == plan.app_name
    fns, classes = f.registered(app)
    assert set(classes) == {"ReferenceWorker"}
    assert set(fns) == {"ReferenceWorker.*", "orchestrator_run", "reconcile"}
    spec = fns["ReferenceWorker.*"]._spec
    assert spec.gpus == "H200" and spec.cpu == 8.0 and spec.memory == w.memory
    assert set(spec.volumes) == {str(WEIGHTS_MOUNT), "/vol/flora-results"}
    ospec = fns["orchestrator_run"]._spec
    assert "nonpreemptible: true" in str(ospec.scheduler_placement) and ospec.gpus is None
    for name in ("orchestrator_run", "reconcile"):
        raw = fns[name]._raw_f
        assert raw.__module__ == f.__name__ and raw.__qualname__ == name  # module scope


def test_determinism_toggle_sets_the_cublas_env():
    plan = f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, determinism="deterministic")
    env = plan.object("ReferenceWorker").env
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["FLORA_DETERMINISM"] == "deterministic"
    with pytest.raises(ValueError):
        f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, determinism="fast")


def test_exact_needs_the_bf16_receipt():
    from flora_deploy.config.hashing import MissingReceiptError

    with pytest.raises(MissingReceiptError):
        f.plan_app(exact_cfg())


def test_reconcile_cron_registers_a_schedule():
    plan = f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, reconcile_cron="on")
    assert plan.object("reconcile_watched").schedule == f.RECONCILE_EVERY
    plan2 = f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, reconcile_cron="0 * * * *")
    assert plan2.object("reconcile_watched").schedule == "0 * * * *"
    fns, _ = f.registered(f.build_app(plan))
    assert "reconcile_watched" in fns


# ---------------------------------------------------------------------------------------------
# rewards (reward.* -> one RewardWorker)


def test_rewards_only_share_one_worker_class(fake_workers_app):
    cfg = rewards_cfg({"reward.terminal": {"gpu": "H100!", "max_containers": 1, "max_inputs": 3},
                       "reward.dense": {"gpu": "H100!", "max_containers": 1, "max_inputs": 3,
                                        "memory": 96}},
                      output={"sink": "s3://my-bucket/flora/rr", "formats": ["parquet"]})
    plan = f.plan_app(cfg)
    assert plan.worker_classes == {"reward.terminal": "RewardWorker",
                                   "reward.dense": "RewardWorker"}
    w = plan.object("RewardWorker")
    assert w.shard_kinds == ("reward.terminal", "reward.dense")
    assert w.gpu == "H100!" and w.image == "img_vllm" and w.max_inputs == 3
    assert w.max_containers == 2  # summed: each kind keeps its own pool
    assert w.memory[0] == 96 * 1024  # the larger of the two
    assert w.volumes[str(CACHE_MOUNT)] == f.VolumeMount("flora-cache", create=True)
    assert w.secrets == ("flora-bucket",)  # bucket store: credentials by Secret name
    assert w.env["FLORA_STORE_KIND"] == "bucket" and w.env["FLORA_STORE_BUCKET"] == "my-bucket"
    assert w.env["FLORA_STORE_PREFIX"] == "flora/rr"
    assert "VLLM_CACHE_ROOT" in w.env and "FLORA_IMG_VLLM_FINGERPRINT" in w.env
    assert set(plan.image_fingerprints) == {"img_vllm", "img_cpu"}
    fns, classes = f.registered(f.build_app(plan))
    assert set(classes) == {"RewardWorker"}
    spec = fns["RewardWorker.*"]._spec
    assert spec.gpus == "H100!" and len(spec.secrets) == 2  # flora-bucket + the env dict


def test_gpu_count_is_rendered_as_type_count(fake_workers_app):
    from flora_deploy.config.schema import ConfigError
    from flora_deploy.jobs.settings import PoolSettings

    assert f.gpu_string(PoolSettings(gpu="H100!", gpu_count=2)) == "H100!:2"
    assert f.gpu_string(PoolSettings(gpu="H100!")) == "H100!"
    # review 4 M7: terminal/dense engines use one device, so a second GPU is refused up front
    with pytest.raises(ConfigError, match="single-device"):
        rewards_cfg({"reward.terminal": {"gpu": "H100!", "gpu_count": 2, "max_containers": 1},
                     "reward.dense": {"gpu": "H100!", "gpu_count": 2, "max_containers": 1}},
                    modal={"max_gpus": 4, "budget_usd": 5})


def test_shared_class_pools_must_agree(fake_workers_app):
    cfg = rewards_cfg({"reward.terminal": {"gpu": "H100!"}, "reward.dense": {"gpu": "H200"}})
    with pytest.raises(f.DeployError, match="must agree"):
        f.plan_app(cfg)


def test_missing_worker_module_is_refused_or_skipped(monkeypatch):
    monkeypatch.setitem(sys.modules, "flora_deploy.modal_app.workers_app", None)
    with pytest.raises(f.WorkerUnavailable, match="reward.terminal"):
        f.plan_app(rewards_cfg())
    plan = f.plan_app(rewards_cfg(), missing_workers="skip")
    assert plan.worker_classes == {} and {"reward.terminal", "reward.dense"} <= set(plan.skipped)
    assert [o.name for o in plan.objects] == ["orchestrator_run", "reconcile"]


def test_rubric_has_no_worker_yet(fake_workers_app):
    d = template("rewards_only.yaml")
    d["pipeline"]["stages"] = ["reward.rubric"]
    d["resources"] = {}
    d["modal"]["max_gpus"] = 4
    with pytest.raises(f.WorkerUnavailable, match="RubricWorker"):
        f.plan_app(parse_config(d))


# ---------------------------------------------------------------------------------------------
# refusals


def test_balanced_is_refused():
    d = template("exact_mp4_dir.yaml")
    d.update(preset="balanced", allow_uncertified=True, resources={})
    d["modal"]["max_gpus"] = 4
    cfg = parse_config(d)
    receipts = {"captioner_bf16": "0" * 64, "tempseg_merged": "1" * 64}
    # no SegmentWorker / CaptionWorker yet (P1b-1)
    with pytest.raises(f.WorkerUnavailable, match="P1b-1"):
        f.plan_app(cfg, derived_receipts=receipts)
    # and without the opt-in, the certificate rule refuses first
    strict = cfg.model_copy(update={"allow_uncertified": False})
    with pytest.raises(f.DeployError, match="certificate"):
        f.plan_app(strict, derived_receipts=receipts)


def test_decorated_worker_is_refused():
    reg = {"segcap": f.WorkerRef("flora_deploy.modal_app.reference_app", "ReferenceWorkerSmoke",
                                 "img_hfref", startup_timeout_s=60, recipe="segcap")}
    with pytest.raises(f.DeployError, match="undecorated"):
        f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, registry=reg)


def test_worker_image_must_match_the_preset():
    reg = {"segcap": f.WorkerRef("flora_deploy.modal_app.reference_app", "ReferenceWorker",
                                 "img_vllm", startup_timeout_s=60, recipe="segcap")}
    with pytest.raises(f.DeployError, match="img_hfref"):
        f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, registry=reg)


@pytest.mark.parametrize("uri", ["file:///tmp/x", "/tmp/x", "ftp://h/x", "modal-volume://"])
def test_unreachable_stores_are_refused(uri):
    with pytest.raises(f.DeployError):
        f.parse_store(uri)


def test_parse_store_normalizes():
    s = f.parse_store("volume://flora-jobs/a/b/")
    assert (s.uri, s.volume, s.mount, s.root) == ("modal-volume://flora-jobs/a/b", "flora-jobs",
                                                  "/vol/flora-jobs", "a/b")
    b = f.parse_store("r2://bkt/pre/", bucket_secret="r2-creds")
    assert (b.kind, b.bucket, b.root, b.secret) == ("bucket", "bkt", "pre", "r2-creds")


def test_negative_min_containers_refused():
    with pytest.raises(f.DeployError):
        f.plan_app(exact_cfg(), derived_receipts=RECEIPTS, min_containers=-1)


# ---------------------------------------------------------------------------------------------
# names


def test_app_name_is_stable_and_follows_the_stage_hashes(fake_workers_app):
    a = f.plan_app(rewards_cfg()).app_name
    assert a == f.plan_app(rewards_cfg()).app_name
    assert re.fullmatch(r"flora-my-rollout-rewards-[0-9a-f]{12}", a) and len(a) <= 64
    # resource-only edits (not hashed, D11) redeploy the same app
    more = rewards_cfg({"reward.terminal": {"gpu": "H100!", "max_containers": 1},
                        "reward.dense": {"gpu": "H100!", "max_containers": 1, "cpu": 12}})
    assert f.plan_app(more).app_name == a
    # different outputs -> a different app
    d = template("rewards_only.yaml")
    d["pipeline"]["stages"] = ["reward.terminal"]
    d["resources"] = {}
    assert f.plan_app(parse_config(d)).app_name != a
    other = rewards_cfg(project="Other_Project.v2")
    assert f.plan_app(other).app_name.startswith("flora-other-project-v2-")
    # the class/function names ModalSpawner relies on never change
    assert f.worker_classes(rewards_cfg()) == {"reward.terminal": "RewardWorker",
                                               "reward.dense": "RewardWorker"}


def test_long_project_names_are_truncated():
    name = f.app_name_for("p" * 63, {"segcap": "0" * 64})
    assert len(name) <= 64 and name.startswith("flora-ppp")


def test_two_configs_build_side_by_side(fake_workers_app):
    a = f.build_app(f.plan_app(exact_cfg(), derived_receipts=RECEIPTS))
    b = f.build_app(f.plan_app(rewards_cfg()))
    assert a.name != b.name
    assert set(f.registered(a)[1]) == {"ReferenceWorker"}
    assert set(f.registered(b)[1]) == {"RewardWorker"}


# ---------------------------------------------------------------------------------------------
# orchestrator wrappers (container side), with Modal stubbed


def test_orchestrator_wrappers_spawn_in_their_own_app(monkeypatch):
    from flora_deploy.modal_app import orchestrator_app as oa

    monkeypatch.setenv(f.APP_ENV, "flora-x-abc")
    monkeypatch.setenv(f.WATCH_ENV, "flora-watch-abc")
    looked_up = []

    class _Fn:
        def __init__(self, name):
            self.name = name

        def spawn(self, *a):
            return a

    def own(name):
        looked_up.append(name)
        return _Fn(name)

    kv = {"job-1": "pkg.mod:factory"}

    class _KV(dict):
        def items(self):  # noqa: D401 - dict API
            return list(super().items())

    store = _KV(kv)
    monkeypatch.setattr(f, "_own", own)
    monkeypatch.setattr(f, "_watch_kv", lambda: store)
    calls = {}

    def run_generation(job_id, factory, generation, watch, *, spawn_successor, watch_kv):
        calls["run"] = (job_id, factory, generation, watch, spawn_successor, watch_kv)
        return {"ok": True}

    def reconcile_job(job_id, factory, revive=True, *, spawn_run=None):
        calls.setdefault("reconcile", []).append((job_id, factory, revive, spawn_run))
        return {"state": "done", "revived": False}

    monkeypatch.setattr(oa, "run_generation", run_generation)
    monkeypatch.setattr(oa, "_reconcile_job", reconcile_job)
    assert f.orchestrator_run("job-1", "pkg.mod:factory", 2, True) == {"ok": True}
    assert calls["run"][:4] == ("job-1", "pkg.mod:factory", 2, True)
    assert calls["run"][4]("a") == ("a",) and calls["run"][5] is store
    f.reconcile("job-1", "pkg.mod:factory")
    out = f.reconcile_watched()
    assert out == {"job-1": {"state": "done", "revived": False}}
    assert "job-1" not in store  # terminal jobs leave the watch list
    assert set(looked_up) == {"orchestrator_run"}


def test_orchestrator_app_run_still_spawns_itself(monkeypatch):
    """The refactor kept ``orchestrator_app.run``: it delegates to ``run_generation``."""
    import inspect

    from flora_deploy.modal_app import orchestrator_app as oa

    src = inspect.getsource(oa.run._raw_f_)
    assert "run_generation(" in src and "spawn_successor=run.spawn" in src
