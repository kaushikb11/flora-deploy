"""``flora_deploy.sdk``: submit / status / wait / results / report / resume / cancel / logs and
the deployed deps factory, against a fake backend (``tests/fakes/flora_backend.py``): the
client-side code, the run spec and ``sdk.deployed_deps`` are the real ones; the "deployed app"
runs the real orchestrator in-process with G2's local spawner and fake engines. No Modal."""

from __future__ import annotations

import json

import pytest

from flora_deploy import sdk
from flora_deploy.jobs.orchestrator import ShardLedger, utc
from flora_deploy.jobs.store import JobPaths
from flora_deploy.modal_app.factory import deployment_path
from tests.fakes.flora_backend import JOB, RECEIPTS, DeployedApp


def submit(app: DeployedApp, **kw) -> sdk.Job:
    return sdk.submit(app.config_path, receipts=RECEIPTS, backend=app.backend, **kw)


def test_submit_wait_results_report_end_to_end(tmp_path):
    app = DeployedApp(tmp_path)
    job = submit(app)
    # the spawn went to the deployed app's orchestrator with the SDK's factory
    (sp,) = app.backend.spawned
    assert (sp["app"], sp["function"]) == (app.plan.app_name, "orchestrator_run")
    assert sp["args"] == (JOB, sdk.FACTORY, 0, True) and sp["environment"] == "main"
    assert app.outcomes == ["done"]
    spec = sdk.read_run_spec(app.store, JOB, app.plan.config_hash)
    assert spec["worker_classes"] == app.plan.worker_classes
    assert spec["launches"][0]["call_id"] == sp["call_id"]
    assert sdk.read_run_spec(app.store, JOB) == spec  # the job-level pointer

    v = job.wait(poll_s=0.0)
    assert v.state == "done" and v.exit_code == 0 and v.composed
    assert v.counts["segcap"]["done"] == sum(v.counts["segcap"].values())
    assert v.spend_usd > 0 and v.budget_usd == 5.0 and v.eta_s is None
    recs = job.results("caption")
    # fake engines: never certified (review 6 M7)
    assert len(recs) == 6 and {r.uncertified_reason for r in recs} == {"fake_models"}

    r = job.report()
    assert r["state"] == "done" and r["composed"]
    assert set(r["spend"]["by_kind"]) == {"segcap", "reward.terminal", "reward.dense"}
    assert r["usd_per_video_hour"] and r["usd_per_video_hour"] > 0
    assert r["video_hours"]["done"] == r["video_hours"]["planned"] > 0
    assert r["certification"]["preset"] == "exact"
    assert not r["certification"]["needs_certificate"]
    assert r["outputs"]["caption"]["jsonl"].startswith(app.store_uri + "/flora/" + JOB)
    assert r["errors"]["poison_pills"] == []

    again = sdk.open_job(JOB, store=app.store_uri, backend=app.backend)
    assert again.config_hash == job.config_hash and again.status().state == "done"


def test_deployed_deps_reads_the_run_spec(tmp_path):
    app = DeployedApp(tmp_path, run=False)
    job = submit(app, budget_usd=7.5)
    env = {**app.env(), "FLORA_MIN_CONTAINERS": "1", "FLORA_SCALEDOWN_S": "120"}
    deps = sdk.deployed_deps(JOB, env)
    assert deps.settings.config_hash == job.config_hash
    assert deps.settings.budget_usd == 7.5
    assert (deps.settings.min_containers, deps.settings.scaledown_s) == (1, 120.0)
    assert deps.worker_app == app.plan.app_name
    assert dict(deps.worker_classes) == app.plan.worker_classes
    assert deps.settings.pools == job.settings.pools
    with pytest.raises(sdk.JobNotFound):
        sdk.deployed_deps("other-job", env)


def test_submit_refusals(tmp_path):
    app = DeployedApp(tmp_path / "a", deploy=False)
    with pytest.raises(sdk.NotDeployed, match="flora deploy"):
        submit(app)
    with pytest.raises(sdk.FloraError, match="captioner_bf16"):  # missing receipt
        sdk.submit(app.config_path, backend=app.backend)

    app2 = DeployedApp(tmp_path / "b", run=False)
    rec = json.loads(app2.store.get_bytes(deployment_path(app2.plan.app_name)))
    rec["config_hash"] = "f" * 64
    app2.store.put_bytes(deployment_path(app2.plan.app_name), json.dumps(rec).encode())
    with pytest.raises(sdk.DeploymentMismatch, match="redeploy"):
        submit(app2)

    app3 = DeployedApp(tmp_path / "c", index=False)
    with pytest.raises(sdk.FloraError, match="flora index"):
        submit(app3)
    assert app3.backend.spawned == []


def test_resubmit_attaches_restarts_or_refuses(tmp_path):
    app = DeployedApp(tmp_path, run=False)
    job = submit(app)
    assert len(app.backend.spawned) == 1
    app.backend.kv.put(f"orchestrator/{JOB}/hint", 0)  # nothing holds it: restart
    submit(app)
    assert len(app.backend.spawned) == 2
    assert len(sdk.read_run_spec(app.store, JOB)["launches"]) == 2
    with pytest.raises(sdk.FloraError, match="already running with budget"):
        submit(app, budget_usd=99)
    app.backend.orchestrate = app._orchestrate
    job.launch()
    job.wait(poll_s=0)
    with pytest.raises(sdk.FloraError, match="already submitted .* done"):
        submit(app)


def test_completed_with_errors_from_a_poison_pill(tmp_path):
    app = DeployedApp(tmp_path, faults="oom:ep001@segcap")
    job = submit(app)
    v = job.wait(poll_s=0)
    assert v.state == "completed_with_errors" and v.exit_code == 3
    r = job.report()
    assert r["errors"]["poison_pills"] and r["errors"]["problems"]
    assert len(job.results("caption")) == 5


def test_budget_stop_and_resume_with_a_higher_budget(tmp_path):
    app = DeployedApp(tmp_path)
    job = submit(app, budget_usd=0.01)
    v = job.wait(poll_s=0)
    assert v.state == "budget_stopped" and v.exit_code == 4
    with pytest.raises(sdk.FloraError, match="--budget-usd"):
        job.resume()
    with pytest.raises(sdk.FloraError, match="positive"):
        job.resume(budget_usd=-1)
    if v.spend_usd > 0:
        with pytest.raises(sdk.FloraError, match="below the spend"):
            job.resume(budget_usd=v.spend_usd / 2)
    call = job.resume(budget_usd=50)
    assert call == app.backend.spawned[-1]["call_id"]
    assert sdk.read_run_spec(app.store, JOB)["settings"]["budget_usd"] == 50
    v = job.wait(poll_s=0)
    assert v.state == "done" and v.budget_usd == 50
    with pytest.raises(sdk.FloraError, match="is done"):
        job.resume()


def test_cancel_without_an_orchestrator_cancels_live_calls(tmp_path):
    app = DeployedApp(tmp_path, run=False)
    job = submit(app)
    assert job.status().state == "planned"
    m = job._deps().manifests.load()
    s = m.shards[0]
    paths = JobPaths(JOB, config_hash=job.config_hash, stage_hashes=dict(m.stage_hashes))
    led = ShardLedger(s.shard_id, s.kind, status="active", call_id="fc-live", spawn_n=1,
                      calls=[{"call_id": "fc-live", "outcome": "spawned", "utc": utc(1.0)}],
                      updated_utc=utc(1.0))
    app.store.put_bytes(paths.call(job.config_hash, s.kind, s.shard_id), led.to_bytes())
    assert job.cancel("test") == ["fc-live"]
    assert app.backend.cancelled == ["fc-live"]
    assert job.status().state == "cancelled" and job.status().exit_code == 5
    with pytest.raises(sdk.FloraError, match="stays cancelled"):
        job.resume()


def test_cancel_with_a_live_orchestrator_only_writes_the_record(tmp_path):
    from flora_deploy.jobs.orchestrator import DictLease

    app = DeployedApp(tmp_path, run=False)
    job = submit(app)
    assert DictLease(app.backend.kv, f"orchestrator/{JOB}").acquire("orch-x", 300)
    assert job.status().orchestrator_alive
    assert job.cancel() == [] and app.backend.cancelled == []
    assert job.status().state == "cancelled"  # the record alone decides the state
    with pytest.raises(sdk.FloraError, match="stays cancelled"):
        job.resume()


def test_wait_reports_a_dead_orchestrator(tmp_path):
    app = DeployedApp(tmp_path, run=False)
    job = submit(app)
    t = [1000.0]
    job.clock = lambda: t[0]

    def sleep(s: float) -> None:
        t[0] += 60

    with pytest.raises(sdk.FloraError, match="flora resume"):
        job.wait(poll_s=60, sleep=sleep, dead_after_s=180)
    assert job.wait(poll_s=60, sleep=sleep, timeout_s=0).state == "planned"


def test_logs_by_kind(tmp_path):
    app = DeployedApp(tmp_path)
    job = submit(app)
    assert list(job.logs()) == ["line 1", "line 2"]
    assert app.backend.log_calls[-1]["function_call"] is None
    out = list(job.logs(kind="orchestrator"))
    assert out[0].startswith("--- orchestrator call fc-0001")
    assert app.backend.log_calls[-1]["function_call"] == "fc-0001"
    out = list(job.logs(kind="segcap", follow=True))
    assert app.backend.log_calls[-1]["follow"] is True
    assert sum(x.startswith("---") for x in out) == 1
    with pytest.raises(sdk.FloraError, match="this job runs"):
        list(job.logs(kind="caption"))


def test_plan_options():
    opts = sdk.plan(sdk_cfg(), video_hours=2.0, episodes=100, budget_usd=500)
    assert 1 <= len(opts) <= 4
    assert all(o.cost_usd > 0 for o in opts)
    with pytest.raises(sdk.FloraError):
        sdk.plan(sdk_cfg())


def sdk_cfg():
    from importlib import resources

    from flora_deploy.config.schema import loads_config

    return loads_config((resources.files("flora_deploy.config") / "templates" /
                         "exact_mp4_dir.yaml").read_text())


def test_exit_codes():
    assert [sdk.exit_code(s) for s in ("done", "failed", "completed_with_errors",
                                       "budget_stopped", "cancelled", "running", "planned",
                                       "lease_held")] == [0, 1, 3, 4, 5, 0, 0, 5]


class FakeVolume:
    """``modal.Volume`` subset ``RemoteVolumeStore`` uses."""

    def __init__(self):
        self.files: dict[str, bytes] = {}

    class _Batch:
        def __init__(self, vol):
            self.vol = vol

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def put_file(self, f, path):
            self.vol.files[path.lstrip("/")] = f.read()

    def batch_upload(self, force=False):
        return self._Batch(self)

    def read_file(self, path):
        p = path.lstrip("/")
        if p not in self.files:
            raise FileNotFoundError(p)
        yield self.files[p]

    def listdir(self, path, recursive=False):
        p = path.strip("/")
        hits = [k for k in self.files if not p or k.startswith(p + "/")]
        if not hits:
            raise FileNotFoundError(p)
        from types import SimpleNamespace

        return [SimpleNamespace(path=k, type=1) for k in hits]


def test_remote_volume_store():
    vol = FakeVolume()
    st = sdk.RemoteVolumeStore(vol, "/sub")
    st.put_bytes("flora/j/a/x.json", b"1")
    st.put_bytes("flora/j/a/y.json", b"2")
    st.put_bytes("flora/j/b.json", b"3")
    assert set(vol.files) == {"sub/flora/j/a/x.json", "sub/flora/j/a/y.json",
                              "sub/flora/j/b.json"}
    assert st.get_bytes("flora/j/a/x.json") == b"1"
    assert st.exists("flora/j/b.json") and not st.exists("flora/j/c.json")
    with pytest.raises(FileNotFoundError):
        st.get_bytes("nope")
    assert list(st.list("flora/j/a/")) == ["flora/j/a/x.json", "flora/j/a/y.json"]
    assert list(st.list("flora/j/")) == ["flora/j/a/x.json", "flora/j/a/y.json", "flora/j/b.json"]
    assert list(st.list("flora/missing/")) == []
    with pytest.raises(ValueError):
        st.put_bytes("../escape", b"")
