"""``scripts/h0_rehearsal.py`` cleanup (review 8b L3): each run deletes only its own Volumes and
its own leases, never another H0 run's. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(scope="module")
def h0():
    spec = importlib.util.spec_from_file_location("h0_rehearsal_cleanup", SCRIPTS /
                                                  "h0_rehearsal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


class _Leases(dict):
    def keys(self):  # modal.Dict.keys() is an iterator
        return iter(list(super().keys()))


@pytest.fixture
def modal_calls(monkeypatch):
    calls: list[list[str]] = []
    leases = _Leases({"orchestrator/h0-r1-yam.fake": 1, "orchestrator/h0-r1-yam.fake2": 1,
                      "orchestrator/h0-r1-human.fake": 1, "orchestrator/h0-r10-yam.fake": 1,
                      "orchestrator/real-job": 1})

    def run(args, **kw):
        calls.append(list(args))
        out = json.dumps([]) if args[:3] == ["modal", "app", "list"] else ""
        return subprocess.CompletedProcess(args, 0, out, "")

    fake_modal = types.SimpleNamespace(Dict=types.SimpleNamespace(
        from_name=lambda name, environment_name=None: leases))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(subprocess, "run", run)
    return calls, leases


def test_volumes_are_per_run(h0):
    assert h0.store_volume("r1") != h0.store_volume("r2")
    assert h0.media_volume("r1") != h0.media_volume("r2")
    assert "r1" in h0.store_volume("r1") and "r1" in h0.media_volume("r1")
    text = h0.write_config(Path(__import__("tempfile").mkdtemp()), "yam", "r1").read_text()
    assert f"volume://{h0.store_volume('r1')}/" in text
    assert f"volume://{h0.media_volume('r1')}/" in text


def test_job_ids_are_fake_namespaced(h0):
    from flora_deploy.jobs.store import is_fake_job_id

    assert is_fake_job_id(h0.job_id_for("r1", "yam"))


def test_cleanup_deletes_only_this_runs_volumes_and_leases(h0, modal_calls):
    calls, leases = modal_calls
    h0.cleanup([], "r1", ["h0-r1-yam.fake", "h0-r1-human.fake"], [], keep=False)
    deleted = [c[3] for c in calls if c[:3] == ["modal", "volume", "delete"]]
    assert sorted(deleted) == sorted([h0.store_volume("r1"), h0.media_volume("r1")])
    assert h0.STORE_VOLUME not in deleted and h0.MEDIA_VOLUME not in deleted
    assert set(leases) == {"orchestrator/h0-r1-yam.fake2", "orchestrator/h0-r10-yam.fake",
                           "orchestrator/real-job"}


def test_cleanup_keep_deletes_no_volume(h0, modal_calls):
    calls, _ = modal_calls
    h0.cleanup([], "r1", [], [], keep=True)
    assert not [c for c in calls if c[:3] == ["modal", "volume", "delete"]]


def test_watch_dicts_are_per_run(h0, tmp_path, monkeypatch, request):
    """Review 9a M2: the fake app's watch Dict was named from the stage hashes only, so two H0
    runs (or any two fake deploys of the same configs) shared one, and either run's cleanup
    deleted the other's."""
    names: dict[str, list[str]] = {}
    for rid in ("10220142abcd", "10220143ef01"):
        w = tmp_path / rid
        for job in ("human", "yam"):
            (w / h0.JOB_SPECS[job].data).mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(w)
        names[rid] = [h0.plans(h0.write_config(w, job, rid))["fake"].watch_dict
                      for job in ("human", "yam")]
    mine, theirs = names.values()
    assert len(set(mine)) == 2 and not set(mine) & set(theirs), names
    calls, _ = request.getfixturevalue("modal_calls")  # after planning: it stubs out modal
    h0.cleanup([], "10220142abcd", [], mine, keep=True)
    deleted = [c[4] for c in calls if c[:3] == ["modal", "dict", "delete"]]
    assert deleted == mine
