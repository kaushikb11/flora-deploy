"""``--max-parallel``: independent runbook steps at once (``scripts/_runbook.py``
``run_parallel``), with fakes only. The certified plan's DAG (b, f, d at once; d2 after d; e
after all), the reservation each running step holds against the global cap, failure and kill
isolation, ``--status``, the wall-clock estimate, and ``--max-parallel 1`` being the
sequential loop. Nothing here contacts Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wh = _load("wave_h")
rb = sys.modules["_runbook"]
CERT = wh.PLANS[wh.CERTIFIED]  # exact_e2e, noise, balanced, lab, caption_iso, certify
UPPER = {"exact_e2e": 4.0, "noise": 2.0, "balanced": 4.5, "lab": 2.3, "caption_iso": 2.9,
         "certify": 0.0}
CAPS = {"exact_e2e": 4.2, "noise": 2.15, "balanced": 4.55, "lab": 2.35, "caption_iso": 2.95,
        "certify": 0.0}
GCAP = round(sum(CAPS.values()), 2)


def _proj(upper: float, rate: float = 0.001) -> SimpleNamespace:
    return SimpleNamespace(usd=upper / 2, usd_upper=upper, peak_usd_per_s=rate)


PROJ = {n: _proj(u) for n, u in UPPER.items()}


class Ctx:
    step = ""


class Fake:
    """A runner that records when each step ran and how many ran at once. ``gate``: steps
    that wait inside the runner until all of them are in (a barrier), so they must overlap."""

    def __init__(self, *, gate=(), fail=(), spend=None, hold=None):
        self.lock = threading.Lock()
        self.active: set[str] = set()
        self.peak = 0
        self.log: list[tuple[str, str, frozenset]] = []
        self.barrier = threading.Barrier(len(gate), timeout=10) if gate else None
        self.gate, self.fail, self.spend = set(gate), set(fail), spend or {}
        self.hold = hold or {}  # step -> Event it waits for inside the runner
        self.ctx_steps: dict[str, str] = {}

    def __call__(self, name, ctx):
        with self.lock:
            self.active.add(name)
            self.peak = max(self.peak, len(self.active))
            self.log.append(("start", name, frozenset(self.active)))
        self.ctx_steps[name] = ctx.step
        try:
            print(f"working on {name}")
            if name in self.gate:
                self.barrier.wait()
            if name in self.hold:
                assert self.hold[name].wait(10)
            if name in self.fail:
                raise RuntimeError(f"{name} broke")
            usd = self.spend.get(name, 0.0 if name == "certify" else 0.5)
            return {"ok": True, "tracked_usd": usd, "apps": [],
                    "jobs": {"human": {"job_id": f"job-{name}"}}}
        finally:
            with self.lock:
                self.active.discard(name)
                self.log.append(("end", name, frozenset(self.active)))

    def started_after_end(self, a: str, b: str) -> bool:
        """``a`` started after ``b`` ended."""
        at = {x[:2]: k for k, x in reversed(list(enumerate(self.log)))}  # first index of each
        return ("end", b) in at and ("start", a) in at and at[("end", b)] < at[("start", a)]


def _run(st, path, run, *, steps=CERT, deps=None, max_parallel=3, caps=CAPS, gcap=GCAP,
         proj=PROJ, out=None, keep_later=True, **kw):
    """``keep_later=False``: the decision keeps nothing for the later steps (each decided as
    if alone, as ``--only`` is), so only the running steps' reservations protect the cap."""
    return rb.run_steps(
        st, steps, state_path=path, ctx=Ctx(), run=run, projection=proj.__getitem__,
        decide=lambda n, p: rb.decide(st, steps if keep_later else [n], n, proj, caps=caps,
                                      global_cap=gcap),
        check_after=lambda n: rb.check_after(st, n, caps=caps, global_cap=gcap),
        deps=wh.CERT_DEPS if deps is None else deps, max_parallel=max_parallel,
        out=out or (lambda m: None), **kw)


# ---------------------------------------------------------------------------------------------
# The DAG


def test_certified_dag_b_f_d_independent_d2_after_d_e_after_all():
    assert wh.plan_deps(wh.CERTIFIED) == {
        "exact_e2e": (), "noise": (), "balanced": (), "lab": ("balanced",),
        "caption_iso": ("exact_e2e",),  # g captions (b)'s own segment records
        "certify": ("exact_e2e", "noise", "balanced", "lab", "caption_iso")}
    for plan in ("balanced-only", "full"):  # one chain: nothing runs beside another
        deps = wh.plan_deps(plan)
        steps = wh.PLANS[plan]
        assert all(deps[s] == ((steps[i - 1],) if i else ()) for i, s in enumerate(steps))
    assert wh.max_parallel_for(wh.CERTIFIED, None, 3) == 3
    assert wh.max_parallel_for(wh.CERTIFIED, "lab", 3) == 1  # --only: one step
    assert wh.max_parallel_for("full", None, 3) == 1


def test_b_f_d_run_at_once_and_e_waits_for_all(tmp_path, capsys):
    run = Fake(gate=("exact_e2e", "noise", "balanced"))  # the barrier: all three at once
    st = rb.State("r", 0.0, "x")
    rc = _run(st, tmp_path / "s.json", run, out=print)
    assert rc == 0, [st.step(n).attempts[0].notes for n in CERT if st.step(n).attempts]
    assert all(st.step(n).status == "done" for n in CERT)
    assert run.peak == 3
    assert run.started_after_end("lab", "balanced")
    for dep in ("exact_e2e", "noise", "balanced", "lab"):
        assert run.started_after_end("certify", dep)
    assert run.ctx_steps == {n: n for n in CERT}  # each step its own ctx copy
    # every attempt reserved its budget (the cap: the global cap is the sum of the caps)
    for n in CERT:
        a = st.step(n).attempts[0]
        assert a.reserved_usd == a.budget_usd == CAPS[n] and a.status == "done"
    text = capsys.readouterr().out
    assert "[balanced] working on balanced" in text and "[lab] working on lab" in text
    assert "start exact_e2e: reserves $4.20 of the global cap" in text


def test_results_match_a_sequential_run(tmp_path):
    seq, par = rb.State("r", 0.0, "x"), rb.State("r", 0.0, "x")
    assert _run(seq, tmp_path / "a.json", Fake(), max_parallel=1) == 0
    assert _run(par, tmp_path / "b.json", Fake(), max_parallel=3) == 0
    for n in CERT:
        a, b = seq.step(n), par.step(n)
        assert (a.status, a.result, a.tracked_usd) == (b.status, b.result, b.tracked_usd)
        assert a.attempts[0].budget_usd == b.attempts[0].budget_usd
    assert seq.spent == par.spent


# ---------------------------------------------------------------------------------------------
# Reservations


def test_running_reservation_counts_against_the_cap():
    st = rb.State("r", 0.0, "x")
    a = rb.Attempt(started=1.0, budget_usd=2.0, reserved_usd=2.0, tracked_usd=0.5)
    st.step("a").attempts.append(a)
    st.step("a").status = "running"
    caps = {"a": 2.0, "b": 1.5}
    # spent 0.5 (a's heartbeat) + a still keeps 1.5 = 2.0 of the 3.0 cap: 1.0 left for b
    b = rb.budget_for(st, "b", caps=caps, global_cap=3.0)
    assert b.usd == pytest.approx(1.0) and b.binding == "running a"
    assert b.terms["running"] == pytest.approx(1.5)
    d = rb.decide(st, ["a", "b"], "b", {"a": _proj(2.0), "b": _proj(1.2)}, caps=caps,
                  global_cap=3.0)
    assert d.action == "stop" and "reserved by the running steps" in d.message
    # spent + running reservations + b's upper <= cap: an upper of 1.0 fits exactly
    assert rb.decide(st, ["a", "b"], "b", {"a": _proj(2.0), "b": _proj(1.0)}, caps=caps,
                     global_cap=3.0).runs
    # no step running: today's rule (b fits its cap)
    st.step("a").attempts[0].status, st.step("a").status = "done", "done"
    assert rb.budget_for(st, "b", caps=caps, global_cap=3.0).binding == "attempt cap"


def test_in_step_deadline_uses_the_steps_own_reservation_only():
    st = rb.State("r", 0.0, "x")
    st.step("a").attempts.append(rb.Attempt(started=1.0, budget_usd=1.0, reserved_usd=1.0))
    st.step("a").status = "running"
    caps = {"a": 5.0}
    b = rb.budget_for(st, "a", caps=caps, global_cap=50.0, spent_in_step=0.25)
    assert b.usd == pytest.approx(0.75) and b.binding == "reservation"  # not 5.0 - 0.25
    assert rb.gpu_deadline_s(st, "a", 0.001, caps=caps, global_cap=50.0, slack_s=100) == \
        pytest.approx((1.0 - 0.1) / 0.001)
    # a sequential attempt (no reservation): recomputed as always
    st.step("a").attempts[0].reserved_usd = None
    assert rb.budget_for(st, "a", caps=caps, global_cap=50.0).usd == pytest.approx(5.0)


def test_queued_earlier_steps_are_kept_for():
    st = rb.State("r", 0.0, "x")
    proj = {"a": _proj(1.0), "b": _proj(1.0), "c": _proj(1.0)}
    assert rb.reserve_for(st, ["a", "b", "c"], "b", proj) == {"c": 1.0}
    st.queued = {"a"}  # a parallel run: a waits for a dependency, b may start first
    assert rb.reserve_for(st, ["a", "b", "c"], "b", proj) == {"a": 1.0, "c": 1.0}
    st.step("c").attempts.append(rb.Attempt(started=1.0, budget_usd=1.0))
    st.step("c").status = "running"  # a running step is in reservations(), not the reserve
    assert rb.reserve_for(st, ["a", "b", "c"], "b", proj) == {"a": 1.0}
    assert st.reservations() == {"c": 1.0}


def test_a_step_that_does_not_fit_beside_the_running_ones_waits(tmp_path):
    """Cap 3.0: a (cap 2.0) runs; b (upper 1.5) does not fit beside a's reservation, so it
    waits and starts once a has ended (spending 0.4): it never overlaps a."""
    steps = ("a", "b")
    proj, caps = {"a": _proj(1.9), "b": _proj(1.5)}, {"a": 2.0, "b": 1.5}
    gate = threading.Event()
    run = Fake(hold={"a": gate}, spend={"a": 0.4})
    st = rb.State("r", 0.0, "x")
    msgs: list[str] = []

    def out(m):
        msgs.append(m)
        if m.startswith("wait b"):
            gate.set()  # b was refused while a runs: let a finish

    assert _run(st, tmp_path / "s.json", run, steps=steps, deps={}, caps=caps, gcap=3.0,
                proj=proj, out=out, keep_later=False) == 0
    assert run.peak == 1 and run.started_after_end("b", "a")
    assert any(m.startswith("wait b:") and "reserved by the running steps" in m for m in msgs)
    assert st.step("b").status == "done"


def test_the_cap_stops_a_waiting_step_when_the_running_one_spent_its_share(tmp_path):
    steps = ("a", "b")
    proj, caps = {"a": _proj(1.9), "b": _proj(1.5)}, {"a": 2.0, "b": 1.5}
    gate = threading.Event()
    run = Fake(hold={"a": gate}, spend={"a": 1.9})
    st = rb.State("r", 0.0, "x")

    def out(m):
        if m.startswith("wait b"):
            gate.set()

    rc = _run(st, tmp_path / "s.json", run, steps=steps, deps={}, caps=caps, gcap=3.0,
              proj=proj, out=out, keep_later=False)
    assert rc == 3 and st.step("b").status == "stopped" and not st.step("b").attempts
    assert [n for _, n, _ in run.log] == ["a", "a"]  # b never ran
    assert any("stop rule before" in n for n in st.step("b").notes)


# ---------------------------------------------------------------------------------------------
# Failure and kill isolation


def test_one_step_failing_does_not_corrupt_the_others(tmp_path):
    gate = threading.Event()
    run = Fake(fail=("noise",), hold={"balanced": gate})
    st = rb.State("r", 0.0, "x")
    path = tmp_path / "s.json"
    msgs: list[str] = []

    def out(m):
        msgs.append(m)

    th = threading.Thread(target=lambda: msgs.append(("rc", _run(st, path, run, out=out))))
    th.start()
    deadline = time.time() + 10
    while st.step("noise").status != "failed" and time.time() < deadline:
        time.sleep(0.01)
    gate.set()  # balanced finishes after noise failed
    th.join(10)
    assert ("rc", 1) in msgs
    assert st.step("noise").status == "failed"
    assert st.step("noise").attempts[0].notes[-1] == "error: RuntimeError('noise broke')"
    for n in ("exact_e2e", "balanced"):  # running beside it: finished and recorded
        assert st.step(n).status == "done" and st.step(n).result["jobs"]["human"]
    # the failure starts nothing new: lab (after balanced) and certify never ran
    assert st.step("lab").status == "pending" and st.step("certify").status == "pending"
    saved = rb.load_state(path)  # the file is whole and says the same
    assert {n: s.status for n, s in saved.steps.items()} == \
        {n: s.status for n, s in st.steps.items()}
    assert any(m.startswith("not started: lab, certify") for m in msgs if isinstance(m, str))


def test_a_stop_rule_keeps_exit_3_when_running_steps_fail_after_it(tmp_path):
    """b hits an in-step stop rule (exit 3, as sequentially); f and d, already running beside
    it, fail afterwards. Their failures are recorded but never replace the halting code."""
    stopped = threading.Event()
    run = Fake(fail=("noise", "balanced"))
    base = run.__call__

    def runner(name, ctx):
        if name == "exact_e2e":
            try:
                raise rb.StopRule("in-step stop")
            finally:
                stopped.set()
        assert stopped.wait(10)
        time.sleep(0.05)
        return base(name, ctx)

    st = rb.State("r", 0.0, "x")
    rc = _run(st, tmp_path / "s.json", runner)
    assert st.step("exact_e2e").status == "stopped"
    assert st.step("noise").status == st.step("balanced").status == "failed"
    assert rc == 3


def test_ctrl_c_waits_for_the_running_steps_cleanup(tmp_path, monkeypatch):
    """Ctrl-C reaches the main thread while it waits for a step. The steps' threads are
    daemons, so re-raising at once would kill them before their ``finally`` stopped the apps
    they deployed (sequentially that ``finally`` runs). The main thread waits for them first."""
    import queue as q

    cleaned: list[str] = []
    started = threading.Barrier(4, timeout=10)  # the three steps and the main thread

    def runner(name, ctx):
        try:
            started.wait()
            time.sleep(0.2)  # the subprocess got the SIGINT too and is winding down
            return {"ok": False, "tracked_usd": 0.1}
        finally:
            cleaned.append(name)  # `modal app stop` in the real runner

    class Interrupted(q.Queue):
        def get(self, *a, **k):
            started.wait()  # all three are running
            raise KeyboardInterrupt

    monkeypatch.setattr(rb, "queue", SimpleNamespace(Queue=Interrupted))
    st = rb.State("r", 0.0, "x")
    msgs: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        _run(st, tmp_path / "s.json", runner, out=msgs.append)
    assert sorted(cleaned) == ["balanced", "exact_e2e", "noise"]
    assert any(m.startswith("interrupted: waiting up to") for m in msgs)


def test_a_failed_dependency_blocks_its_dependents(tmp_path):
    run = Fake(fail=("balanced",))
    st = rb.State("r", 0.0, "x")
    msgs: list[str] = []
    rc = _run(st, tmp_path / "s.json", run, out=msgs.append,
              keep_going=lambda n: n == "balanced")  # keep going: the rest still decides
    assert rc == 1
    assert "blocked lab: needs balanced (failed)" in msgs
    assert any(m.startswith("blocked certify: needs") for m in msgs)
    assert st.step("exact_e2e").status == st.step("noise").status == "done"
    assert not st.step("lab").attempts and not st.step("certify").attempts


def test_a_killed_run_marks_every_running_step_killed(tmp_path):
    """The state on disk while b, f and d run is what a killed process leaves: the next run
    marks all three killed (charged their bound or budget), their steps failed."""
    gate = threading.Event()
    run = Fake(hold={n: gate for n in ("exact_e2e", "noise", "balanced")})
    st = rb.State("r", 0.0, "x")
    path = tmp_path / "s.json"
    th = threading.Thread(target=lambda: _run(st, path, run))
    th.start()
    deadline = time.time() + 10
    while len(run.active) < 3 and time.time() < deadline:
        time.sleep(0.01)
    killed = rb.load_state(path)  # the process dies here
    gate.set()
    th.join(10)
    assert set(killed.running()) == {"exact_e2e", "noise", "balanced"}
    for n in ("exact_e2e", "noise", "balanced"):
        a = killed.step(n).attempts[-1]
        a.apps = [f"app-{n}"]
    apps = killed.mark_killed()
    assert apps == ["app-balanced", "app-exact_e2e", "app-noise"]
    for n in ("exact_e2e", "noise", "balanced"):
        s = killed.step(n)
        assert s.status == "failed" and s.attempts[-1].status == "killed"
        assert s.charged() == pytest.approx(CAPS[n])  # its budget until billing covers it
    assert not killed.running() and killed.reservations() == {}


def test_state_saves_survive_concurrent_writers(tmp_path):
    st = rb.State("r", 0.0, "x")
    path = tmp_path / "s.json"
    stop = threading.Event()

    def grow():
        i = 0
        while not stop.is_set():
            st.step("a").result[f"k{i}"] = i  # a step's thread growing its result
            i += 1
            time.sleep(0.0005)

    th = threading.Thread(target=grow)
    th.start()
    try:
        for _ in range(50):
            rb.save_state(st, path)
            json.loads(path.read_text())
    finally:
        stop.set()
        th.join()


# ---------------------------------------------------------------------------------------------
# --status, the dry run, --max-parallel 1


def _running_cert_state() -> rb.State:
    st = rb.State("r", 0.0, "x", plan=wh.CERTIFIED)
    for n, budget, spent in (("exact_e2e", 4.2, 1.0), ("balanced", 4.55, 0.5)):
        st.step(n).status = "running"
        st.step(n).attempts.append(rb.Attempt(started=time.time() - 60, budget_usd=budget,
                                              reserved_usd=budget, tracked_usd=spent,
                                              heartbeat=time.time()))
    return st


def test_status_shows_running_steps_and_their_reservations():
    st = _running_cert_state()
    m = wh.offline_manifest(0.0, wh.CERT_GOLDEN)
    projs = wh.plan_projections(wh.CERTIFIED, m)
    lines: list[str] = []
    d = wh.print_status(st, projs, plan=wh.CERTIFIED, out=lines.append)
    text = "\n".join(lines)
    assert "running: b exact_e2e: reserved $4.20, charged $1.00 so far, $3.20 still kept" in text
    assert "running: d balanced: reserved $4.55, charged $0.50 so far, $4.05 still kept" in text
    assert "reserved by running steps: $7.25 (spent $1.50)" in text
    assert "ready to start: noise (what each needs is done)" in text
    assert d is not None and d.step == "noise"  # the next decision skips the running steps
    assert d.budget.terms["running"] == pytest.approx(7.25)


def test_status_next_never_names_a_step_whose_dependency_runs():
    """b, f and g done, d running: lab needs d and certify needs everything, so nothing can
    start; --status must not say `next: lab: run` (cert_preflight reads that line), nor that
    every step is done."""
    st = rb.State("r", 0.0, "x", plan=wh.CERTIFIED)
    for n in ("exact_e2e", "noise", "caption_iso"):
        st.step(n).status = "done"
    st.step("balanced").status = "running"
    st.step("balanced").attempts.append(rb.Attempt(
        started=time.time() - 60, budget_usd=4.55, reserved_usd=4.55, tracked_usd=0.5,
        heartbeat=time.time()))
    projs = wh.plan_projections(wh.CERTIFIED, wh.offline_manifest(0.0, wh.CERT_GOLDEN))
    lines: list[str] = []
    assert wh.print_status(st, projs, plan=wh.CERTIFIED, out=lines.append) is None
    text = "\n".join(lines)
    assert "next: lab" not in text and "every step is done" not in text
    assert "next: none until balanced ends" in text


def test_dry_run_prints_wall_clock_sequential_vs_parallel():
    m = wh.offline_manifest(0.0, wh.CERT_GOLDEN)
    projs = wh.plan_projections(wh.CERTIFIED, m)
    st = rb.State("r", 0.0, "x")
    lines = wh.wall_lines(st, projs, wh.CERTIFIED, CERT, wh.titles(wh.CERTIFIED))
    assert lines[0] == "  order: b, f, d first; d2 after d; g after b; e after b, f, d, d2, g"
    walls = {n: projs[n].wall_s() for n in CERT}
    seq = sum(walls.values())
    par = max(walls["exact_e2e"] + walls["caption_iso"], walls["noise"],
              walls["balanced"] + walls["lab"])
    assert rb.wall_clock(CERT, wh.CERT_DEPS, walls, 1) == pytest.approx(seq)
    assert rb.wall_clock(CERT, wh.CERT_DEPS, walls, 3) == pytest.approx(par)
    assert f"sequential {seq / 60:.0f} /" in lines[1] and f"--max-parallel 3 {par / 60:.0f} /" \
        in lines[1]
    assert "same cost" in lines[1] and lines[2].startswith("  with d2 skipped")
    # (d)'s wall: its job's two pools at once, then the witness's two at once
    secs = [i["seconds"] for i in projs["balanced"].items]
    assert len(secs) == 4
    assert walls["balanced"] == pytest.approx(max(secs[:2]) + max(secs[2:]))
    assert walls["lab"] == pytest.approx(sum(i["seconds"] for i in projs["lab"].items))


def test_wall_clock_list_scheduling():
    deps = {"c": ("a",), "d": ("a", "b")}
    w = {"a": 10.0, "b": 30.0, "c": 5.0, "d": 1.0}
    assert rb.wall_clock(["a", "b", "c", "d"], deps, w, 1) == 46.0
    assert rb.wall_clock(["a", "b", "c", "d"], deps, w, 2) == 31.0  # a+c beside b, then d
    assert rb.wall_clock(["a", "b", "c", "d"], rb.chain("abcd"), w, 3) == 46.0


def test_max_parallel_1_is_the_sequential_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "run_parallel", lambda *a, **k: pytest.fail("not sequential"))
    a, b = rb.State("r", 0.0, "x"), rb.State("r", 0.0, "x")
    ra, rb_ = Fake(), Fake()
    _run(a, tmp_path / "a.json", ra, max_parallel=1)  # with the DAG, one at a time
    rb.run_steps(b, CERT, state_path=tmp_path / "b.json", ctx=Ctx(), run=rb_,
                 projection=PROJ.__getitem__,
                 decide=lambda n, p: rb.decide(b, CERT, n, PROJ, caps=CAPS, global_cap=GCAP),
                 check_after=lambda n: rb.check_after(b, n, caps=CAPS, global_cap=GCAP),
                 out=lambda m: None)  # today's call: no deps, no max_parallel
    assert [x[:2] for x in ra.log] == [x[:2] for x in rb_.log]  # plan order, never two at once
    assert ra.peak == rb_.peak == 1
    strip = {"started", "finished", "heartbeat"}

    def norm(st):
        d = st.to_json()["steps"]
        return {n: {**s, "attempts": [{k: v for k, v in x.items() if k not in strip}
                                      for x in s["attempts"]]} for n, s in d.items()}

    assert norm(a) == norm(b)
    assert all(x.reserved_usd is None for s in a.steps.values() for x in s.attempts)


def test_wave_h_main_passes_the_dag_and_max_parallel(tmp_path, monkeypatch):
    got = {}

    def fake_run_steps(st, steps, **kw):
        got.update(steps=list(steps), deps=kw["deps"], max_parallel=kw["max_parallel"])
        return 0

    monkeypatch.setattr(rb, "run_steps", fake_run_steps)
    monkeypatch.setattr(wh, "load_manifest",
                        lambda offline, version="golden/v1": wh.offline_manifest(0.0, version))
    monkeypatch.setattr(wh, "billed_by_app", lambda since: {})
    monkeypatch.setattr(wh, "refresh_billing", lambda st, **kw: True)
    monkeypatch.setattr(wh, "stop_killed", lambda st: None)
    monkeypatch.setattr(wh, "lab_already_done", lambda *a, **k: (False, "test"))
    monkeypatch.chdir(tmp_path)
    base = ["--plan", "certified", "--yes-spend", "--state", str(tmp_path / "s.json"),
            "--work", str(tmp_path / "w")]
    assert wh.main(base) == 0
    assert got["steps"] == list(CERT) and got["deps"] == wh.CERT_DEPS
    assert got["max_parallel"] == rb.DEFAULT_MAX_PARALLEL == 3
    assert wh.main([*base, "--max-parallel", "1"]) == 0 and got["max_parallel"] == 1
    assert wh.main([*base, "--only", "noise"]) == 0 and got["max_parallel"] == 1


# ---------------------------------------------------------------------------------------------
# The steps' own concurrency safety


def test_cold_witness_env_is_its_own_never_os_environ(tmp_path, monkeypatch):
    seen = {}

    def run_job(ctx, step, cfg, job_id, **kw):
        seen["env"] = dict(ctx.env)
        seen["os"] = wh.os.environ.get(wh.COMPILE_CACHE_ENV)
        return {"app": "a", "job_id": job_id, "state": "done", "spend": 0.1}

    monkeypatch.setattr(wh, "_run_job", run_job)
    monkeypatch.setattr(wh, "_index", lambda *a, **k: None)
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")
    monkeypatch.delenv(wh.COMPILE_CACHE_ENV, raising=False)
    cfg = tmp_path / "flora-balanced-human.yaml"
    cfg.write_text("project: p\npreset: balanced\n")
    spec = wh.SPECS[wh.CERTIFIED]
    ctx = wh.Ctx(rb.State("r", 0.0, "x"), {}, tmp_path, spec=spec)
    wh.run_cold_witness(ctx, cfg, 0.001, spent=0.0)
    assert seen == {"env": {wh.COMPILE_CACHE_ENV: "off"}, "os": None}
    assert ctx.env == {}  # the step's own ctx is untouched


def test_certified_noise_has_its_own_app_and_the_oracles_stage_hash(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(wh, "require_tempseg_pin", lambda ctx: {})
    monkeypatch.setattr(wh, "resolve_receipts", lambda names, **kw: {})
    monkeypatch.setattr(wh, "balanced_episodes", lambda m, spec, **kw: spec.episodes)
    monkeypatch.setattr(wh, "stage_data", lambda ctx, data, ids: calls.append(("stage", data)))
    monkeypatch.setattr(wh, "_index", lambda ctx, cfg, job_id, tag, *, limit=None:
                        calls.append(("index", job_id, limit)))
    monkeypatch.setattr(wh, "noise_index_problems", lambda *a: "")
    monkeypatch.setattr(wh, "stage_hashes", lambda cfg: {"segment": "s1", "caption": "c1"})

    def run_job(ctx, step, cfg, job_id, **kw):
        calls.append(("run", step, job_id, Path(cfg).read_text()))
        return {"app": f"app-{job_id}", "job_id": job_id, "state": "done", "spend": 0.3}

    monkeypatch.setattr(wh, "_run_job", run_job)
    spec = wh.SPECS[wh.CERTIFIED]
    st = rb.State("r", 0.0, "x", plan=wh.CERTIFIED)
    st.step("noise").attempts.append(rb.Attempt(started=1.0))
    ctx = wh.Ctx(st, wh.offline_manifest(0.0, wh.CERT_GOLDEN), tmp_path, spec=spec,
                 caps=dict(CAPS), step="noise")
    res = wh.run_noise(ctx)  # (b) has not run: (f) needs nothing from it
    assert res["ok"] and res["episodes"] == len(spec.noise)
    assert ("stage", wh.CERT_DATA) in calls
    assert ("index", "wave-h-r-noise-human-a1", len(spec.noise)) in calls
    (_, step, job_id, text), = [c for c in calls if c[0] == "run"]
    assert step == "noise" and job_id == "wave-h-r-noise-human-a1"
    assert "project: wave-h-r-noise-human-a1\n" in text and f"path: ./{wh.CERT_DATA}\n" in text
    assert res["apps"] == ["app-wave-h-r-noise-human-a1"]  # not (b)'s app
    # the oracle text it was derived from: (b)'s own config inputs, (b)'s project
    oracle = (tmp_path / "flora-noise-human-oracle.yaml").read_text()
    assert "project: wave-h-r-exact_e2e-human\n" in oracle
    assert oracle.replace("wave-h-r-exact_e2e-human", "wave-h-r-noise-human-a1") == text
    # (b) ran with other stage hashes: refused before any app ($0)
    st.step("exact_e2e").result = {"jobs": {"human": {"stage_hashes": {"segment": "other"}}}}
    calls.clear()
    with pytest.raises(RuntimeError, match="not \\(b\\)'s"):
        wh.run_noise(ctx)
    assert not [c for c in calls if c[0] in ("run", "index")]


def test_certify_reads_only_the_finished_steps_jobs():
    st = rb.State("r", 0.0, "x", plan=wh.CERTIFIED)
    for n, status in (("exact_e2e", "done"), ("noise", "failed"), ("balanced", "running")):
        st.step(n).status = status
        st.step(n).result = {"jobs": {"human": {"job_id": f"job-{n}"}}}
    view = wh.finished_view(st)
    assert view.step("exact_e2e").result["jobs"]["human"]["job_id"] == "job-exact_e2e"
    assert view.step("noise").result == {} and view.step("balanced").result == {}
    assert st.step("noise").result  # the state itself is untouched


def test_shared_data_dir_is_staged_and_uploaded_once(tmp_path, monkeypatch):
    uploads, inside = [], []
    lock_seen = []

    def stage_local(ctx, data, ids):
        lock_seen.append(wh._STAGE_LOCK.locked())
        inside.append(data)
        time.sleep(0.02)

    monkeypatch.setattr(wh, "stage_local", stage_local)
    monkeypatch.setattr(wh.h0, "upload_dir", lambda vol, local, remote: uploads.append(remote))
    monkeypatch.setattr(wh, "_UPLOADED", set())
    ctx = wh.Ctx(rb.State("r", 0.0, "x"), {}, tmp_path)
    ths = [threading.Thread(target=wh.stage_data, args=(ctx, wh.CERT_DATA, ("e",)))
           for _ in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert uploads == [f"r/{wh.CERT_DATA}"] and all(lock_seen) and len(inside) == 3


def test_rubric_smoke_runs_its_jobs_at_once_and_check_after():
    rs = _load("rubric_smoke")
    steps = rs.planned(None)
    assert rs.step_deps(steps) == {rs.CHECK: tuple(j.task for j in rs.JOBS)}
    assert rs.step_deps(rs.planned("lid")) == {rs.CHECK: ("lid",)}
    projs = rs.projections(rs.offline_manifest())
    walls = {n: projs[n].wall_s() for n in steps}
    assert rb.wall_clock(steps, rs.step_deps(steps), walls, 3) == pytest.approx(
        max(walls[j.task] for j in rs.JOBS))


def test_scale_test_is_one_chain_its_wall_is_its_schedule():
    st_ = _load("scale_test")
    projs = st_.projections("real")
    assert projs["run"].wall == (pytest.approx(st_.run_wall_s("real")),
                                 pytest.approx(st_.run_wall_s("real", upper=True)
                                               + st_.CLIENT_SLACK_S))
    walls = {n: projs[n].wall_s() for n in st_.STEPS}
    chain = rb.chain(st_.STEPS)
    assert rb.wall_clock(st_.STEPS, chain, walls, 3) == rb.wall_clock(st_.STEPS, chain, walls)


# ---------------------------------------------------------------------------------------------
# State compatibility with the code before ``--max-parallel`` (review 15 M2)

# The Attempt / StepState fields of the runbook before ``--max-parallel`` (e9fdc4e). That code
# loads a state with ``Attempt(**a)`` and ``StepState(**{**v, "attempts": atts})``.
OLD_ATTEMPT = ("started", "status", "finished", "heartbeat", "rate_usd_per_s", "budget_usd",
               "tracked_usd", "apps", "report_error", "notes", "app_ids", "billed_usd",
               "stopped_at", "slack_s", "hard_timeout_s")
OLD_STEP = ("status", "projected_usd", "projected_upper_usd", "billed_usd", "attempts", "notes",
            "result")


def _old_load(d: dict) -> dict:
    """The old ``State.from_json`` step loading, on dataclasses with the old fields only."""
    import dataclasses

    old_a = dataclasses.make_dataclass(
        "OldAttempt", [("started", float)] + [(f, object, None) for f in OLD_ATTEMPT[1:]])
    old_s = dataclasses.make_dataclass("OldStep", [(f, object, None) for f in OLD_STEP])
    steps = {}
    for k, v in d["steps"].items():
        v = dict(v)
        atts = [old_a(**a) for a in v.pop("attempts", [])]
        steps[k] = old_s(**{**v, "attempts": atts})
    return steps


def test_the_old_loader_replica_rejects_reserved_usd():
    # the crash review 15 M2 found, so the replica below is faithful
    with pytest.raises(TypeError, match="reserved_usd"):
        _old_load({"steps": {"a": {"attempts": [{"started": 1.0, "reserved_usd": None}]}}})


def test_a_sequential_state_loads_in_the_old_code(tmp_path):
    st = rb.State("r", 0.0, "x")
    assert _run(st, tmp_path / "s.json", Fake(), max_parallel=1) == 0
    saved = json.loads((tmp_path / "s.json").read_text())
    for s in saved["steps"].values():
        assert set(s) <= set(OLD_STEP)
        for a in s["attempts"]:
            assert "reserved_usd" not in a and set(a) <= set(OLD_ATTEMPT)
    steps = _old_load(saved)
    assert {n: s.status for n, s in steps.items()} == {n: "done" for n in CERT}
    # and it round-trips in the new code
    back = rb.State.from_json(saved)
    assert back.to_json() == st.to_json()
    assert all(x.reserved_usd is None for s in back.steps.values() for x in s.attempts)


def test_a_parallel_reservation_is_kept_including_zero():
    st = rb.State("r", 0.0, "x")
    st.step("a").attempts.append(rb.Attempt(started=1.0, budget_usd=2.0, reserved_usd=2.0))
    st.step("e").attempts.append(rb.Attempt(started=1.0, budget_usd=0.0, reserved_usd=0.0))
    d = json.loads(json.dumps(st.to_json()))
    assert d["steps"]["a"]["attempts"][0]["reserved_usd"] == 2.0
    assert d["steps"]["e"]["attempts"][0]["reserved_usd"] == 0.0  # a reservation, not unset
    back = rb.State.from_json(d)
    assert back.step("a").attempts[0].reserved_usd == 2.0
    assert back.step("e").attempts[0].reserved_usd == 0.0


def test_an_old_state_loads_in_the_new_code():
    old = {"run_id": "r", "created": 0.0, "billing_since": "x", "billing_baseline": {},
           "steps": {"exact_e2e": {
               "status": "done", "projected_usd": 3.0, "projected_upper_usd": 4.0,
               "billed_usd": None, "notes": [], "result": {"ok": True},
               "attempts": [{"started": 1.0, "status": "done", "finished": 2.0,
                             "heartbeat": 2.0, "rate_usd_per_s": 0.001, "budget_usd": 4.2,
                             "tracked_usd": 1.5, "apps": ["a"], "report_error": False,
                             "notes": [], "app_ids": ["ap-1"], "billed_usd": None,
                             "stopped_at": 2.0, "slack_s": None, "hard_timeout_s": None}]}}}
    st = rb.State.from_json(old)
    a = st.step("exact_e2e").attempts[0]
    assert a.reserved_usd is None and a.tracked_usd == 1.5 and a.status == "done"
    assert st.to_json()["steps"] == old["steps"]  # re-saved unchanged: still old-loadable


def test_unknown_keys_from_newer_code_are_ignored():
    d = {"run_id": "r", "created": 0.0, "billing_since": "x",
         "steps": {"a": {"status": "done", "from_the_future": 1,
                         "attempts": [{"started": 1.0, "status": "done", "later_field": "x"}]}}}
    st = rb.State.from_json(d)
    assert st.step("a").status == "done" and st.step("a").attempts[0].status == "done"


# ---------------------------------------------------------------------------------------------
# GPUs across apps (review 15 M3): --max-parallel counts steps; --max-gpus caps GPUs


def _gpus_held(run, gpus) -> int:
    return max((sum(sum(gpus[n].values()) for n in active) for _, _, active in run.log),
               default=0)


def test_peak_is_4_gpus_2_h200_and_the_default_cap_holds_nothing_back(tmp_path):
    caps = wh.parse_gpu_caps(None)
    assert caps == {"total": 4}
    b, f, d = (wh.CERT_GPUS[n] for n in ("exact_e2e", "noise", "balanced"))
    assert sum(sum(x.values()) for x in (b, f, d)) == 4
    assert b.get("H200", 0) + f.get("H200", 0) == 2
    run = Fake(gate=("exact_e2e", "noise", "balanced"))  # all three must overlap
    st = rb.State("r", 0.0, "x")
    assert _run(st, tmp_path / "s.json", run, gpus=wh.CERT_GPUS, gpu_caps=caps) == 0
    assert run.peak == 3 and _gpus_held(run, wh.CERT_GPUS) == 4


def test_h200_cap_runs_f_after_b_with_d_beside_b(tmp_path):
    msgs: list[str] = []
    run = Fake(gate=("exact_e2e", "balanced"))  # b and d must overlap
    st = rb.State("r", 0.0, "x")
    rc = _run(st, tmp_path / "s.json", run, out=msgs.append, gpus=wh.CERT_GPUS,
              gpu_caps=wh.parse_gpu_caps(["H200=1"]))
    assert rc == 0 and all(st.step(n).status == "done" for n in CERT)
    assert run.started_after_end("noise", "exact_e2e")
    assert not any({"exact_e2e", "noise"} <= active for _, _, active in run.log)
    assert any(m.startswith("wait noise: GPU cap (H200 2 > 1)") for m in msgs), msgs
    assert sum(m.startswith("wait noise") for m in msgs) == 1  # said once


def test_total_gpu_cap_holds_the_running_steps_under_it(tmp_path):
    run = Fake()
    st = rb.State("r", 0.0, "x")
    assert _run(st, tmp_path / "s.json", run, gpus=wh.CERT_GPUS,
                gpu_caps=wh.parse_gpu_caps(["3"])) == 0
    assert all(st.step(n).status == "done" for n in CERT)
    assert _gpus_held(run, wh.CERT_GPUS) <= 3


def test_a_step_over_the_cap_alone_still_starts(tmp_path):
    assert rb.gpu_wait("balanced", [], wh.CERT_GPUS, {"total": 1}) is None
    run, st = Fake(), rb.State("r", 0.0, "x")
    assert _run(st, tmp_path / "s.json", run, gpus=wh.CERT_GPUS, gpu_caps={"total": 1}) == 0
    assert all(st.step(n).status == "done" for n in CERT) and run.peak == 1
    assert rb.gpu_wait("balanced", ["noise"], wh.CERT_GPUS, {"total": 2}) == (
        "GPU cap (total 3 > 2)")
    assert rb.gpu_wait("certify", ["balanced"], wh.CERT_GPUS, {"total": 1}) is None


def test_parse_gpu_caps_and_the_help_text(capsys):
    assert wh.parse_gpu_caps(["H200=1", "5"]) == {"total": 5, "H200": 1}
    for bad in ("0", "H200=", "x", "H200=-1"):
        with pytest.raises(ValueError):
            wh.parse_gpu_caps([bad])
    with pytest.raises(SystemExit):
        wh.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    par = text.split("--max-parallel N")[-1].split("--max-gpus N|TYPE=N")[0]
    assert "f after b" not in par and "b, f and d at once" in par
    assert "4 GPUs at once (2x H200, L40S, H100)" in par
    assert "--max-gpus H200=1 runs f after b" in text
