"""Modal-facing pieces, offline: what ``FunctionCall.get(timeout=0)`` really raises in the
installed modal (1.6.2), ModalSpawner over fake Cls / FunctionCall APIs, the Dict lease, the
TickDriver (refresh, budget hard cap hook, transient-error retry), the scheduled reconciler and
the app's module-scope definitions."""

from __future__ import annotations

import builtins
import importlib.metadata
import inspect
from dataclasses import asdict

import pytest
from modal import exception as mx

from flora_deploy.jobs.orchestrator import DictLease
from flora_deploy.jobs.types import ShardReceipt
from flora_deploy.modal_app.orchestrator_app import (
    ModalErrors,
    ModalSpawner,
    modal_errors,
    status_from_get,
)
from tests.fakes.jobs import FakeClock, FakeDict


class FakeInputCancellation(BaseException):
    pass


# Offline errors for the spawner tests: same shape as modal_errors(), self-made classes.
class FakeOutputExpiredError(Exception):
    pass


class FakeFunctionTimeoutError(Exception):
    pass


FAKE = ModalErrors((builtins.TimeoutError,), FakeOutputExpiredError, FakeFunctionTimeoutError,
                   (FakeInputCancellation,))


def _raise(exc):
    def get():
        raise exc
    return get


RECEIPT = ShardReceipt("s1", "segcap", "c", "ta-1", ("u",), (), ("p",), "H200", 1.0, "t")
REAL = modal_errors()


def test_still_running_call_is_pending_exactly_as_modal_raises_it():
    """modal 1.6.2 ``poll_function``: ``raise TimeoutError()``, the *builtin*, no args."""
    assert status_from_get("fc", _raise(builtins.TimeoutError()), REAL).state == "pending"
    # the review's trap: before the fix this came back "failed" and every running shard was
    # respawned without cancelling the original call
    assert REAL.pending == (builtins.TimeoutError,)


def test_real_modal_terminal_errors_map_to_terminal_states():
    assert status_from_get("fc", _raise(mx.OutputExpiredError()), REAL).state == "expired"
    st = status_from_get("fc", _raise(mx.FunctionTimeoutError("24h")), REAL)
    assert st.state == "failed" and "FunctionTimeoutError" in st.error
    assert status_from_get("fc", _raise(mx.InputCancellation()), REAL).state == "cancelled"
    st = status_from_get("fc", _raise(mx.InternalFailure("boom")), REAL)  # a call result
    assert st.state == "failed"
    st = status_from_get("fc", _raise(ValueError("user bug")), REAL)
    assert st.state == "failed" and "user bug" in st.error
    # a worker's own TimeoutError("...") re-raised by get(): a failure, not pending forever
    st = status_from_get("fc", _raise(builtins.TimeoutError("hf download timed out")), REAL)
    assert st.state == "failed" and "hf download" in st.error
    # modal's own TimeoutError family is not "pending" (only the builtin is)
    assert status_from_get("fc", _raise(mx.TimeoutError()), REAL).state == "failed"


def test_transient_client_errors_keep_the_call_pending():
    for exc in (mx.ConnectionError("reset"), mx.ServiceError("503"), mx.InternalError("x"),
                mx.ResourceExhaustedError("slow down")):
        st = status_from_get("fc", _raise(exc), REAL)
        assert st.state == "pending" and st.error.startswith("transient"), exc


def test_exception_classes_and_order_match_installed_modal():
    """Subclass facts the except-clause order in ``status_from_get`` relies on."""
    assert not issubclass(builtins.TimeoutError, mx.TimeoutError)
    assert not issubclass(mx.TimeoutError, builtins.TimeoutError)
    for cls in (mx.OutputExpiredError, mx.FunctionTimeoutError):
        assert issubclass(cls, mx.TimeoutError) and not issubclass(cls, builtins.TimeoutError)
    for cls in REAL.transient:
        assert not issubclass(cls, (builtins.TimeoutError, mx.OutputExpiredError,
                                    mx.FunctionTimeoutError))
    assert issubclass(mx.InputCancellation, BaseException)
    assert not issubclass(mx.InputCancellation, Exception)


def test_installed_modal_still_raises_what_we_assume():
    """Fails loudly if a modal upgrade changes what a pending/expired ``get(timeout=0)`` raises:
    re-check ``modal_errors()`` and the notes in ``orchestrator_app`` before bumping the pin."""
    import modal._functions as mf
    from modal._utils import function_utils as fu

    assert importlib.metadata.version("modal") == "1.6.2"
    src = inspect.getsource(mf._Invocation.poll_function)
    assert "raise OutputExpiredError()" in src
    assert "raise TimeoutError()" in src
    # the bare name resolves to the builtin: _functions does not import modal's TimeoutError
    assert "TimeoutError" not in vars(mf)
    assert mf.OutputExpiredError is mx.OutputExpiredError
    src = inspect.getsource(mf._FunctionCall.get)
    assert "poll_function(timeout=timeout" in src
    src = inspect.getsource(fu._process_result)
    assert "raise FunctionTimeoutError(" in src and "raise InternalFailure(" in src
    assert fu.FunctionTimeoutError is mx.FunctionTimeoutError


@pytest.mark.parametrize("result", [RECEIPT, asdict(RECEIPT)], ids=["dataclass", "dict"])
def test_success_carries_receipt(result):
    st = status_from_get("fc", lambda: result, FAKE)
    assert st.state == "success" and st.receipt == RECEIPT


class _FC:
    registry: dict[str, _FC] = {}

    def __init__(self, oid, outcome):
        self.object_id, self.outcome, self.cancelled = oid, outcome, False
        _FC.registry[oid] = self

    def get(self, timeout=None):
        assert timeout == 0
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def cancel(self, terminate_containers=False):
        self.cancelled = "terminate" if terminate_containers else True


class _CallAPI:
    @staticmethod
    def from_id(oid):
        return _FC.registry[oid]


class _Method:
    def __init__(self, cls_name, log):
        self.cls_name, self.log = cls_name, log

    def spawn(self, *args, token=""):
        self.log.append((self.cls_name, args, token))
        return _FC(f"fc-{len(self.log)}", builtins.TimeoutError())


class _ClsAPI:
    log: list = []
    lookups: list = []

    @classmethod
    def from_name(cls, app, name):
        cls.lookups.append((app, name))
        return lambda: type("Obj", (), {"run_shard": _Method(name, cls.log)})()


def test_modal_spawner_spawn_poll_cancel_with_fake_apis():
    sp = ModalSpawner("flora-workers", {"segcap": "ReferenceWorker"}, errors=FAKE,
                      cls_api=_ClsAPI, call_api=_CallAPI)
    cid = sp.spawn("segcap", "job", "s1", "cfg", token="tok-1")
    cid2 = sp.spawn("segcap", "job", "s2", "cfg")
    assert _ClsAPI.lookups == [("flora-workers", "ReferenceWorker")]  # handle cached
    assert _ClsAPI.log[-2:] == [("ReferenceWorker", ("job", "s1", "cfg"), "tok-1"),
                                ("ReferenceWorker", ("job", "s2", "cfg"), "")]
    assert sp.poll(cid).state == "pending"
    _FC.registry[cid2].outcome = FakeOutputExpiredError()
    assert sp.poll(cid2).state == "expired"
    sp.cancel(cid)
    assert _FC.registry[cid].cancelled is True
    sp.terminate(cid2)
    assert _FC.registry[cid2].cancelled == "terminate"
    with pytest.raises(ValueError, match="no worker class"):
        sp.spawn("caption", "job", "s3", "cfg")


def test_dict_lease_single_holder_takeover_and_release():
    kv, clock = FakeDict(), FakeClock()
    a, b = DictLease(kv, "orch/j", clock), DictLease(kv, "orch/j", clock)
    assert a.acquire("A", 60)
    assert not b.acquire("B", 60)
    assert a.current().holder == "A" and a.current().generation == 1
    clock.advance(30)
    assert a.renew("A", 60)
    clock.advance(50)
    assert not b.acquire("B", 60)  # renewed, still live
    clock.advance(31)
    assert a.current() is None  # expired
    assert b.acquire("B", 60)  # exactly one creator of generation 2
    assert not a.renew("A", 60)  # A sees generation 2 and knows it lost
    assert b.current().holder == "B" and b.current().generation == 2
    a.release("A")  # a stale holder's release is a no-op
    assert b.current().holder == "B"
    b.release("B")
    assert b.current() is None
    assert a.acquire("A", 60) and a.current().generation == 3


def test_dict_lease_takeover_race_has_one_winner():
    kv, clock = FakeDict(), FakeClock()
    DictLease(kv, "orch/j", clock).acquire("old", 10)
    clock.advance(11)
    contenders = [DictLease(kv, "orch/j", clock) for _ in range(3)]
    heads = [c._head() for c in contenders]  # all observe the same expired generation
    assert len({g for g, _ in heads}) == 1
    wins = [c.acquire(f"c{i}", 10) for i, c in enumerate(contenders)]
    assert wins.count(True) == 1


def test_orchestrator_app_definitions_are_module_scope():
    from flora_deploy.modal_app import orchestrator_app as oa

    assert oa.app.name == "flora-orchestrator"
    assert oa.TIMEOUT_S == 24 * 3600 and oa.CPU_CORES <= 0.5
    for fn in (oa.run, oa.reconcile, oa.reconcile_watched):
        raw = fn._raw_f_  # private, but the only offline handle on the decorated function
        assert raw.__qualname__ == raw.__name__  # module scope, not nested/serialized
    assert "nonpreemptible: true" not in str(oa.run._spec_.scheduler_placement)  # jobs.pools
    assert oa.run._spec_.gpus is None and oa.run._spec_.cpu == oa.CPU_CORES
    with pytest.raises(ValueError, match="module:function"):
        oa.load_factory("nocolon")
