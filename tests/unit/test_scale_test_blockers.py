"""``scripts/scale_test.py`` blockers the fake-modal rehearsal found before the real GPU run:
stop rules that could not cancel (``flora cancel`` waiting at its [y/N] prompt), a wall limit
timed from before a slow submit, and the run's Volumes left behind when the report failed.
Nothing here contacts Modal."""

from __future__ import annotations

import pytest

from tests.unit.test_scale_test_runbook import _Harness, full_profile, offline, sc, wh

__all__ = ["full_profile", "offline"]  # fixtures, used by name


def _cancels(h: _Harness) -> list[list[str]]:
    return [c for c in h.calls if "cancel" in c]


def test_the_spend_stop_rule_cancels_without_a_prompt(monkeypatch, tmp_path):
    gpu = {"ta-seg": {"start": 0.0, "end": 1e7}}  # far past any budget
    h = _Harness(monkeypatch, tmp_path, states=("running",), containers=gpu)
    monkeypatch.setattr(sc, "_container_kinds", lambda ctx: {"ta-seg": "segment"})
    out = sc.run_run(h.ctx)
    assert out["stopped"].startswith("observed spend")
    (c,) = _cancels(h)
    assert c[:3] == ["flora", "cancel", h.ctx.job_id] and "--yes" in c
    assert c[c.index("--store") + 1] == h.ctx.store_uri  # the job's own store


def test_the_wall_limit_counts_from_the_orchestrators_spawn(monkeypatch, tmp_path):
    """A 5.5-min client-side submit (fake-modal rehearsal) is not run time: the wall limit
    starts when ``flora run --detach`` returns (it spawns the orchestrator last)."""
    clock = [1000.0]
    monkeypatch.setattr(sc.time, "time", lambda: clock[0])
    monkeypatch.setattr(sc, "run_wall_s", lambda *a, **k: 60.0)
    limit = 60.0 + sc.CLIENT_SLACK_S
    h = _Harness(monkeypatch, tmp_path, states=("running", "running", "done"))
    inner = h.ctx.sh
    timeouts = {}

    def sh(args, *, timeout=1800, check=True):
        if "--detach" in args:
            timeouts["submit"] = timeout
            clock[0] += limit + 30  # a submit longer than the whole wall limit
        return inner(args, timeout=timeout, check=check)

    h.ctx.sh = sh  # type: ignore[method-assign]
    monkeypatch.setattr(sc.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 10))
    out = sc.run_run(h.ctx)
    assert "stopped" not in out, out.get("stopped")
    assert out["state"] == "done" and out["submit_s"] == pytest.approx(limit + 30)
    assert timeouts["submit"] == sc.SUBMIT_TIMEOUT_S >= 1800
    assert not _cancels(h)
    # and it still stops a run that overstays the limit after the spawn
    clock[0] = 1000.0
    h2 = _Harness(monkeypatch, tmp_path, states=("running",))
    monkeypatch.setattr(sc.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 100))
    out = sc.run_run(h2.ctx)
    assert out["stopped"] == f"wall limit {limit:.0f} s"
    (c,) = _cancels(h2)
    assert "--yes" in c


def test_chaos_container_stop_does_not_prompt(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(sc, "_running_container", lambda ctx, kind: "ta-1")
    res = sc._chaos_modal(h.ctx, "preempt_segment")
    assert res["container"] == "ta-1"
    assert ["modal", "container", "stop", "--yes", "ta-1"] in h.calls


def test_every_flora_cancel_in_the_runbooks_passes_yes():
    """``flora cancel`` asks [y/N] unless ``--yes``; with no terminal it cancels nothing and
    exits 2. ``modal container stop`` / ``app stop`` / ``volume delete`` ask too."""
    import re
    from pathlib import Path

    scripts = Path(sc.__file__).resolve().parent
    bad = []
    for f in sorted(scripts.glob("*.py")):
        text = f.read_text()
        for m in re.finditer(r'\[(?:fl|_flora\(\)|"flora"), "cancel"[^\]]*\]', text):
            if '"--yes"' not in m.group(0):
                bad.append(f"{f.name}: {m.group(0)}")
        for m in re.finditer(r'\["modal", "(?:container|app|volume|dict)", "(?:stop|delete)"'
                             r'[^\]]*\]', text):
            if '"--yes"' not in m.group(0):
                bad.append(f"{f.name}: {m.group(0)}")
    assert not bad, bad


# ---------------------------------------------------------------------------------------------
# fake-modal: the run's own Volumes go on any terminal state


@pytest.fixture
def deletes(monkeypatch):
    out: list[list[str]] = []

    def run(args, **kw):
        out.append(list(args))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sc.subprocess, "run", run)
    return out


def _vol_deletes(calls):
    return [c[3] for c in calls if c[:3] == ["modal", "volume", "delete"]]


def test_a_failed_report_still_deletes_the_runs_volumes(offline, deletes, tmp_path, monkeypatch):
    monkeypatch.setitem(sc.RUNNERS, "report", lambda ctx: {"ok": False, "tracked_usd": 0.0,
                                                           "apps": [], "error": "boom"})
    state = tmp_path / "state.json"
    args = ["--mode", "fake-modal", "--yes-spend", "--state", str(state),
            "--work", str(tmp_path / "w")]
    sc.main(args)
    st = wh.load_state(state)
    assert st.step("report").status == "failed"
    assert _vol_deletes(deletes) == sc.fake_volumes(st.run_id)  # only this run's two
    assert all(v.endswith(f"-{st.run_id}") for v in _vol_deletes(deletes))
    assert all("--yes" in c for c in deletes if c[:3] == ["modal", "volume", "delete"])
    # the run is over: a second pass refuses rather than resuming onto deleted Volumes
    deletes.clear()
    assert sc.main(args) == 2 and deletes == []


def test_a_done_run_deletes_and_keep_keeps(offline, deletes, tmp_path):
    state = tmp_path / "s1.json"
    sc.main(["--mode", "fake-modal", "--yes-spend", "--state", str(state),
             "--work", str(tmp_path / "w1")])
    assert len(_vol_deletes(deletes)) == 2
    deletes.clear()
    sc.main(["--mode", "fake-modal", "--yes-spend", "--keep", "--state",
             str(tmp_path / "s2.json"), "--work", str(tmp_path / "w2")])
    assert _vol_deletes(deletes) == []


def test_a_resumable_run_keeps_its_volumes(offline, deletes, tmp_path, monkeypatch):
    """A step the stop rule refused *before* it started (no attempt: a wait for billing) is
    not the end of the run."""
    st = wh.State("r9", 0.0, "2026-10-22")
    st.step("run").status = "stopped"  # refused before it started: no attempt
    assert sc.fake_run_over(st) is None
    st.step("index").status = "failed"
    st.step("index").attempts.append(wh.Attempt(started=0.0))
    assert sc.fake_run_over(st) == "index failed"
    # nothing indexed yet (no Volume created): nothing to delete
    monkeypatch.setitem(sc.RUNNERS, "verify", lambda ctx: {"ok": False, "tracked_usd": 0.0,
                                                           "apps": []})
    sc.main(["--mode", "fake-modal", "--yes-spend", "--state", str(tmp_path / "s.json"),
             "--work", str(tmp_path / "w")])
    assert _vol_deletes(deletes) == []


def test_real_media_is_staged_on_the_store_volume():
    """Modal-side `flora index` mounts only the store's Volume, so real media must be on it."""
    import yaml

    cfg = yaml.safe_load(sc.config_text("real", project="p", run_id="r1", budget_usd=1.0))
    stage, sink = cfg["input"]["stage_to"], cfg["output"]["sink"]
    assert stage.split("/")[2] == sink.split("/")[2] == sc.STORE_VOLUME["real"]
