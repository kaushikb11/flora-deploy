"""`flora compose` / `flora results` over a LocalStore (file:// URI)."""

from __future__ import annotations

import io
import json

import pyarrow.parquet as pq
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.export.results import records_from_table
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.records import StageRecord
from tests.unit.test_export_compose import JOB, setup_job

runner = CliRunner()
ENV = {"COLUMNS": "200"}


def _job(tmp_path):
    root = tmp_path / "store"
    m, _ = setup_job(LocalStore(root))
    return m, f"file://{root}"


def test_compose_then_results_jsonl_table_parquet(tmp_path):
    m, uri = _job(tmp_path)
    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "caption", "--store", uri],
                      env=ENV)
    assert r.exit_code == 1 and "not been composed" in r.output

    r = runner.invoke(app, ["compose", "--job", JOB, "--store", uri], env=ENV)
    assert r.exit_code == 0, r.output
    assert "caption: 4 records (12 rows), certified 75.0%" in r.output

    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "caption", "--store", uri,
                            "--format", "jsonl"], env=ENV)
    assert r.exit_code == 0, r.output
    recs = [StageRecord.from_json(x) for x in r.stdout.splitlines() if x.strip()]
    assert sorted(x.uid for x in recs) == sorted(e.uid for e in m.episodes)

    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "reward.terminal",
                            "--store", uri], env=ENV)
    assert r.exit_code == 0, r.output
    assert "4 records" in r.output and "outside_reference_envelope" in r.output

    out = tmp_path / "out" / "seg.parquet"
    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "segment", "--store", uri,
                            "--format", "parquet", "--out", str(out)], env=ENV)
    assert r.exit_code == 0, r.output
    table = pq.read_table(io.BytesIO(out.read_bytes()))
    assert table.num_rows == 12 and len(records_from_table(table)) == 4

    out2 = tmp_path / "t.jsonl"
    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "segment", "--store", uri,
                            "--out", str(out2)], env=ENV)
    assert r.exit_code == 0 and len(out2.read_text().splitlines()) == 4
    assert json.loads(out2.read_text().splitlines()[0])["segments"] == 3


def test_cli_errors(tmp_path):
    _m, uri = _job(tmp_path)
    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "nope", "--store", uri])
    assert r.exit_code == 2 and "unknown stage" in r.output
    r = runner.invoke(app, ["results", "--job", JOB, "--stage", "segment", "--store", uri,
                            "--format", "parquet"])
    assert r.exit_code == 2 and "--out" in r.output
    r = runner.invoke(app, ["compose", "--job", "ghost", "--store", uri])
    assert r.exit_code == 1 and "no manifest" in r.output
    r = runner.invoke(app, ["compose", "--job", JOB, "--store", "gs://x"])
    assert r.exit_code == 2
    r = runner.invoke(app, ["compose", "--job", JOB, "--store", uri, "--config-hash", "zz"])
    assert r.exit_code == 1 and "not the manifest" in r.output


def test_store_from_env(tmp_path):
    _m, uri = _job(tmp_path)
    r = runner.invoke(app, ["compose", "--job", JOB], env={**ENV, "FLORA_STORE": uri})
    assert r.exit_code == 0, r.output
