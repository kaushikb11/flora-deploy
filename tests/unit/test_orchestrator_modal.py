"""Modal-facing pieces, offline: the TimeoutError-subclass trap, ModalSpawner over fake Cls /
FunctionCall APIs, the Dict lease, and the app's module-scope definitions."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from flora_deploy.jobs.orchestrator import DictLease
from flora_deploy.jobs.types import ShardReceipt
from flora_deploy.modal_app.orchestrator_app import (
    ModalErrors,
    ModalSpawner,
    modal_errors,
    status_from_get,
)
from tests.fakes.jobs import FakeClock, FakeDict


# Mirrors modal.exception: both "terminal" errors subclass the "pending" TimeoutError.
class FakeTimeoutError(Exception):
    pass


class FakeOutputExpiredError(FakeTimeoutError):
    pass


class FakeFunctionTimeoutError(FakeTimeoutError):
    pass


class FakeInputCancellation(BaseException):
    pass


FAKE = ModalErrors(FakeTimeoutError, FakeOutputExpiredError, FakeFunctionTimeoutError,
                   (FakeInputCancellation,))


def _raise(exc):
    def get():
        raise exc
    return get


RECEIPT = ShardReceipt("s1", "segcap", "c", "ta-1", ("u",), (), ("p",), "H200", 1.0, "t")


@pytest.mark.parametrize("errors", [FAKE, modal_errors()], ids=["fake", "modal"])
def test_timeout_subclasses_are_not_read_as_pending(errors):
    assert issubclass(errors.output_expired, errors.timeout)
    assert issubclass(errors.function_timeout, errors.timeout)
    assert status_from_get("fc", _raise(errors.timeout()), errors).state == "pending"
    assert status_from_get("fc", _raise(errors.output_expired()), errors).state == "expired"
    st = status_from_get("fc", _raise(errors.function_timeout("24h")), errors)
    assert st.state == "failed" and "FunctionTimeoutError" in st.error
    assert status_from_get("fc", _raise(errors.cancelled[0]()), errors).state == "cancelled"
    st = status_from_get("fc", _raise(ValueError("user bug")), errors)
    assert st.state == "failed" and "user bug" in st.error


def test_naive_ordering_would_fall_into_the_trap():
    """Documents why the order matters: a TimeoutError-first handler swallows expiry."""
    def naive(get):
        try:
            get()
        except FakeTimeoutError:
            return "pending"
        except FakeOutputExpiredError:  # unreachable
            return "expired"

    assert naive(_raise(FakeOutputExpiredError())) == "pending"
    assert status_from_get("fc", _raise(FakeOutputExpiredError()), FAKE).state == "expired"


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
        self.cancelled = True


class _CallAPI:
    @staticmethod
    def from_id(oid):
        return _FC.registry[oid]


class _Method:
    def __init__(self, cls_name, log):
        self.cls_name, self.log = cls_name, log

    def spawn(self, *args):
        self.log.append((self.cls_name, args))
        return _FC(f"fc-{len(self.log)}", FakeTimeoutError())


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
    cid = sp.spawn("segcap", "job", "s1", "cfg")
    cid2 = sp.spawn("segcap", "job", "s2", "cfg")
    assert _ClsAPI.lookups == [("flora-workers", "ReferenceWorker")]  # handle cached
    assert _ClsAPI.log[-2:] == [("ReferenceWorker", ("job", "s1", "cfg")),
                                ("ReferenceWorker", ("job", "s2", "cfg"))]
    assert sp.poll(cid).state == "pending"
    _FC.registry[cid2].outcome = FakeOutputExpiredError()
    assert sp.poll(cid2).state == "expired"
    sp.cancel(cid)
    assert _FC.registry[cid].cancelled
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
    for fn in (oa.run, oa.reconcile):
        raw = fn._raw_f_  # private, but the only offline handle on the decorated function
        assert raw.__qualname__ == raw.__name__  # module scope, not nested/serialized
    assert "nonpreemptible: true" in str(oa.run._spec_.scheduler_placement)
    assert oa.run._spec_.gpus is None and oa.run._spec_.cpu == oa.CPU_CORES
    with pytest.raises(ValueError, match="module:function"):
        oa.load_factory("nocolon")
