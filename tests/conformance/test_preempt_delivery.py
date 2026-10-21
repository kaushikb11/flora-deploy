"""A delivered preemption is marked ``preempted`` (never ``crashed`` or ``error``) by every shard
worker kind, end to end, with the delivery the H0 rehearsal uses on Modal.

Each case runs a whole job with G2's local orchestrator, but every container runs the
``--fake-models`` workers' runners and their :class:`~flora_deploy.modal_app.workers_app.
FakeFaultHook` (built as the ``Fake*Worker`` classes build them, from the env the factory sets)
over the real store-file fault spec (``fake_faults.json``: ``{"preempt": {<kind>: 1}}``). The
hook delivers the preemption itself, as on Modal: ``modal.exception.simulate_preemption`` (a
``SIGALRM`` whose handler raises ``KeyboardInterrupt`` in the main thread) or, from a worker
thread, the process ``SIGINT`` Modal sends. Calls run inline in the main thread, as a Modal
container with ``max_inputs=1`` runs its input, so the signal lands where it would there.

The H0 rehearsal found the balanced ``segment`` / ``caption`` workers recording this as an
infra error (``ValueError: signal only works in main thread``: the hook ran
``simulate_preemption`` on the fake job's worker thread), so a preemption ate a
``modal.Retries`` slot instead of being excused. The case asserts, per kind: the fault fired,
the job completed with every record, the interrupted start's marker says ``preempted`` and
neither ``crashed`` nor ``error``, the D15 count leaves it out, and nothing was quarantined.
"""

from __future__ import annotations

import json
import signal
import time
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

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "src/flora_deploy/config/templates/local_demo.yaml"

# (config flavour, preempted kind): every shard worker kind
CASES = [("exact", "segcap"), ("balanced", "segment"), ("balanced", "caption"),
         ("exact", "reward.terminal"), ("exact", "reward.dense"), ("rubric", "reward.rubric")]


def _config(d: Path, flavour: str) -> Path:
    if flavour == "rubric":
        from tests.unit.test_local_rubric import rubric_config

        path = rubric_config(d)
        text = path.read_text()
    else:
        text = TEMPLATE.read_text()
        if flavour == "balanced":
            from tests.unit.test_balanced_deps import balanced_yaml

            text = balanced_yaml(text)
        path = d / "flora.yaml"
    path.write_text(text.replace("stage_to: null", "stage_to: volume://flora-h0-media/run/videos"))
    make_sample(d / "videos")
    return path


class _InlinePool:
    """The spawner's executor, running each call in the calling (main) thread."""

    def submit(self, fn, *args):
        fn(*args)

    def shutdown(self, **_kw):
        pass


class _DeployedFakes(LocalRunner):
    """Every container runs the deployed fake worker of its kind (``workers_app.Fake*Worker``),
    configured only through its class env, with the fault hook's own preemption delivery."""

    bill_as_fake_app = True

    def __init__(self, *a, app_plan, store_env: dict, **kw):
        super().__init__(*a, **kw)
        self.app_plan, self.store_env = app_plan, store_env

    def _runner(self, kind, container_id, call_id):
        plan = self.app_plan
        env = {**plan.object(plan.worker_classes[kind]).env, **self.store_env,
               "FLORA_MEDIA_ROOT": str(self.media_root)}

        def crash(code):
            raise ContainerCrash(f"os._exit({code})")

        # the alarm fires after preempt_wait_s (1 s); the hook's 60 s wait is cut short
        hook = W.FakeFaultHook(self.store, exit_fn=crash,
                               sleep=lambda s: time.sleep(min(s, 3.0)))
        if kind == "segcap":
            runner = W.fake_segcap_runner(self.store, env, hook, call_id=call_id)
        elif kind in ("segment", "caption"):
            runner = W.fake_balanced_runner(kind, self.store, env, hook, call_id=call_id)
        else:
            runner = W.fake_reward_runner(self.store, env, hook, call_id=call_id)
        inner = runner.run_shard

        def run_shard(job_id, shard_id, config_hash, token=""):  # Fake*Worker.run_shard
            hook.start_shard(job_id, shard_id, kind)
            slot = getattr(runner, "flora_hook_slot", None)
            if slot is not None:
                slot[0] = hook.bound()
            return inner(job_id, shard_id, config_hash, token=token)

        runner.run_shard = run_shard
        return runner


@pytest.fixture
def one_alarm(monkeypatch):
    """``simulate_preemption`` re-arms a second ``SIGALRM`` 30 s after the first (Modal's
    SIGKILL stand-in); a test process must not get it later, so it is not re-armed here."""
    import modal.exception as me

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Simulated preemption interrupt from modal-client!")

    monkeypatch.setattr(me, "_simulate_preemption_interrupt", interrupt)
    old = signal.getsignal(signal.SIGALRM)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize(("flavour", "kind"), CASES, ids=[k for _f, k in CASES])
def test_delivered_preemption_is_marked_preempted(flavour, kind, tmp_path, one_alarm):
    cfg_path = _config(tmp_path, flavour)
    cfg = load_config(cfg_path)
    plan = F.plan_app(cfg, fake_models=True)
    assert kind in plan.worker_classes, plan.worker_classes
    store_dir = tmp_path / "store"
    r = _DeployedFakes(cfg, config_path=cfg_path, store=store_dir, app_plan=plan,
                       store_env={"FLORA_STORE_KIND": "local",
                                  "FLORA_STORE_ROOT": str(store_dir)})
    st = LocalStore(store_dir)
    try:
        r.index()
        st.put_bytes(W.fake_faults_path(r.job_id),
                     json.dumps({"preempt": {kind: 1}}).encode())
        r.plan()
        r.spawner._pool = _InlinePool()
        outcome = r.orchestrate()
        summary = r.compose(outcome)
        res = r.result(outcome, summary, 0.0)
    finally:
        r.close()

    fired = [p for p in st.list(f"flora/{r.job_id}/fake_faults/fired/")]
    assert fired, "the preemption never fired"
    assert res.state == "done", res.as_dict()
    assert not summary["poison_pills"] and not any(s.bisected_from for s in r.manifest.shards)
    stage = "reward.rubric" if kind == "reward.rubric" else ("caption" if kind == "segcap"
                                                             else kind)
    assert list(iter_records(st, r.job_id, stage)), f"no {stage} records"

    jf = A.JobFiles(st, r.job_id, res.config_hash)
    shards = [s for s in r.manifest.shards if s.kind == kind]
    marks = {s.shard_id: A.markers(jf, kind, s.shard_id) for s in shards}
    hit = [sid for sid, ms in marks.items() if any(m.preempted for m in ms)]
    assert len(hit) == 1, {sid: [(m.preempted, m.crashed, m.error) for m in ms]
                           for sid, ms in marks.items()}
    for sid, ms in marks.items():
        bad = [(m.container_id, m.crashed, m.error) for m in ms if m.crashed or m.error]
        assert not bad, f"{sid}: a preempted start recorded as a crash / error: {bad}"
    ms = marks[hit[0]]
    unmarked = [m for m in ms if not m.preempted]
    assert len(ms) == 2 and len(unmarked) == 1, [(m.preempted, m.error) for m in ms]
    # D15: the preempted start is excluded; only the finishing start is a marker that counts
    assert A.counted_starts(jf, kind, hit[0]) == 1
    assert A.count_starts(jf, kind, hit[0]) == 2
