"""Review 8b M5: fake-models jobs (``flora run --fake-models`` on Modal, ``flora run --local
--fake-models``) live under their own job id (``<job>.fake``), so with the same ``--job`` and
store they never share ``manifest.json``, ``run.json``, ``index/`` or the
``orchestrator/<job>`` lease with a real job, and ``flora results`` / ``flora status`` refuse
to mix the two."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from flora_deploy import sdk
from flora_deploy.cli import app
from flora_deploy.jobs.store import JobPaths, fake_job_id, is_fake_job_id
from tests.fakes.flora_backend import JOB, RECEIPTS, DeployedApp

cli = CliRunner()
FAKE_JOB = f"{JOB}.fake"


def test_fake_job_ids_are_their_own_namespace():
    assert fake_job_id(JOB) == FAKE_JOB and fake_job_id(FAKE_JOB) == FAKE_JOB
    assert is_fake_job_id(FAKE_JOB) and not is_fake_job_id(JOB)
    real, fake = JobPaths(JOB), JobPaths(FAKE_JOB)
    for a, b in ((real.manifest, fake.manifest), (real.job_cache, fake.job_cache),
                 (real.index("u"), fake.index("u")), (sdk.run_pointer_path(JOB),
                                                      sdk.run_pointer_path(FAKE_JOB))):
        assert a != b and not b.startswith(real.root + "/")


@pytest.fixture
def both(tmp_path):
    """One store with the real app and the fake-models app of the same config deployed, and the
    dataset indexed under both job ids."""
    from flora_deploy.jobs.indexing import IndexJob, open_dataset, run_index
    from flora_deploy.modal_app.factory import deployment_path, deployment_record, plan_app

    a = DeployedApp(tmp_path, run=False)
    fake = plan_app(a.cfg, fake_models=True, derived_receipts=RECEIPTS)
    assert fake.store.uri == a.store_uri
    rec = deployment_record(fake, app_id="ap-fake2", deployed_utc="2026-10-21T00:00:00Z")
    a.store.put_bytes(deployment_path(fake.app_name), json.dumps(rec).encode())
    ijob = IndexJob.from_config(a.cfg, FAKE_JOB)
    run_index(ijob, a.store, open_dataset(ijob.dataset, local_root=a.videos), log=lambda m: None)
    a.fake_plan = fake
    sdk.set_backend(a.backend)
    yield a
    sdk.set_backend(None)


def _fake_planner(cfg, **kw):
    from flora_deploy.modal_app.factory import plan_app

    return plan_app(cfg, fake_models=True, **kw)


def test_fake_and_real_submits_with_the_same_job_id_share_nothing(both):
    a = both
    fake = sdk.submit(a.cfg, job_id=JOB, receipts=RECEIPTS, planner=_fake_planner)
    assert fake.job_id == FAKE_JOB and fake.spec["fake_models"] is True
    assert not a.store.exists(sdk.run_pointer_path(JOB))  # the real pointer is untouched
    assert not a.store.exists(JobPaths(JOB).manifest)
    assert a.backend.spawned[-1]["args"][0] == FAKE_JOB  # lease orchestrator/<job>.fake
    real = sdk.submit(a.cfg, job_id=JOB, receipts=RECEIPTS)
    assert real.job_id == JOB and real.spec["fake_models"] is False
    assert real.config_hash != fake.config_hash
    assert sdk.read_run_spec(a.store, FAKE_JOB)["config_hash"] == fake.config_hash
    assert sdk.read_run_spec(a.store, JOB)["config_hash"] == real.config_hash
    assert sdk.open_job(JOB, store=a.store_uri).config_hash == real.config_hash
    assert sdk.open_job(FAKE_JOB, store=a.store_uri).config_hash == fake.config_hash


def test_a_real_submit_refuses_a_fake_job_id(both):
    with pytest.raises(sdk.FloraError, match="fake-models"):
        sdk.submit(both.cfg, job_id=FAKE_JOB, receipts=RECEIPTS)


def test_open_job_points_at_the_fake_run_and_refuses_a_mixed_spec(both):
    a = both
    sdk.submit(a.cfg, job_id=JOB, receipts=RECEIPTS, planner=_fake_planner)
    with pytest.raises(sdk.JobNotFound, match=r"local-demo\.fake"):
        sdk.open_job(JOB, store=a.store_uri)
    # a pointer from before the namespace (a fake spec under the real job id) is refused
    spec = dict(sdk.read_run_spec(a.store, FAKE_JOB), job_id=JOB)
    sdk.write_run_spec(a.store, spec)
    with pytest.raises(sdk.FloraError, match="fake-models"):
        sdk.open_job(JOB, store=a.store_uri)
    st = cli.invoke(app, ["status", JOB, "--store", a.store_uri, "--config",
                          str(a.tmp / "absent.yaml")])
    assert st.exit_code == 2 and "fake-models" in st.output


def test_local_fake_runs_use_the_fake_namespace(tmp_path):
    from flora_deploy.local import run_local
    from flora_deploy.local import sample_dataset as make_sample
    from tests.fakes.flora_backend import TEMPLATE

    (tmp_path / "flora.yaml").write_text(TEMPLATE.read_text())
    make_sample(tmp_path / "videos")
    store = tmp_path / "store"
    res = run_local(tmp_path / "flora.yaml", store=store)
    assert res.state == "done" and res.job_id == FAKE_JOB
    assert (store / "flora" / FAKE_JOB / "run.json").is_file()
    assert not (store / "flora" / JOB).exists()
    ok = cli.invoke(app, ["results", "--job", FAKE_JOB, "--stage", "caption", "--store",
                          str(store), "--format", "jsonl"])
    assert ok.exit_code == 0, ok.output
    no = cli.invoke(app, ["results", "--job", JOB, "--stage", "caption", "--store", str(store)])
    assert no.exit_code == 2 and FAKE_JOB in no.output


def test_results_refuses_a_fake_spec_under_a_real_job_id(tmp_path):
    from flora_deploy.jobs.stores import LocalStore

    st = LocalStore(tmp_path)
    sdk.write_run_spec(st, {"schema": sdk.RUN_SPEC_SCHEMA, "job_id": JOB, "config_hash": "c" * 16,
                            "app_name": "flora-x-fake-0123456789ab", "fake_models": True})
    o = cli.invoke(app, ["results", "--job", JOB, "--stage", "caption", "--store", str(tmp_path)])
    assert o.exit_code == 2 and "fake-models" in o.output
