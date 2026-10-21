"""CachedSession against a fake NVIDIA Session (no weights)."""

from __future__ import annotations

import threading
import time

from flora_deploy.nv.session import CachedSession


class FakeSession:
    """Mimics NVIDIA models.Session: one model at a time, load() closes the previous one."""

    instances: list[FakeSession] = []

    def __init__(self):
        self.model = self.proc = None
        self.audit = {}
        self.loads: list[str] = []
        self.closed = 0
        FakeSession.instances.append(self)

    def load(self, kind):
        self.close()
        time.sleep(0.01)
        self.loads.append(kind)
        self.model, self.proc = f"model:{kind}", f"proc:{kind}"
        self.audit[kind] = {"kind": kind}
        return self.model, self.proc

    def close(self):
        self.model = self.proc = None
        self.closed += 1


def setup_function():
    FakeSession.instances.clear()


def test_loads_once_per_kind_in_separate_sessions():
    s = CachedSession(FakeSession)
    assert s.load("tempseg") == ("model:tempseg", "proc:tempseg")
    assert s.load("captioner") == ("model:captioner", "proc:captioner")
    assert s.load("tempseg") == ("model:tempseg", "proc:tempseg")
    assert len(FakeSession.instances) == 2
    assert [i.loads for i in FakeSession.instances] == [["tempseg"], ["captioner"]]
    # the first model was not closed by loading the second kind
    assert FakeSession.instances[0].model == "model:tempseg"
    assert s.loaded_kinds == ("tempseg", "captioner")
    assert s.audit == {"tempseg": {"kind": "tempseg"}, "captioner": {"kind": "captioner"}}


def test_close_is_noop_and_close_all_releases():
    s = CachedSession(FakeSession)
    s.load("tempseg")
    closed_before = FakeSession.instances[0].closed
    s.close()
    assert FakeSession.instances[0].closed == closed_before
    assert s.load("tempseg") == ("model:tempseg", "proc:tempseg")
    s.close_all()
    assert FakeSession.instances[0].closed == closed_before + 1
    assert FakeSession.instances[0].model is None
    assert s.loaded_kinds == ()
    s.load("tempseg")  # loads again after close_all
    assert len(FakeSession.instances) == 2


def test_context_manager_closes_all():
    with CachedSession(FakeSession) as s:
        s.load("captioner")
    assert FakeSession.instances[0].model is None


def test_concurrent_loads_build_once():
    s = CachedSession(FakeSession)
    results = []
    threads = [threading.Thread(target=lambda: results.append(s.load("tempseg")))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(FakeSession.instances) == 1
    assert results == [("model:tempseg", "proc:tempseg")] * 8


def test_from_family_uses_nvidia_session_class():
    class Models:
        Session = FakeSession

    class Handle:
        models = Models

    s = CachedSession.from_family(Handle)
    s.load("tempseg")
    assert isinstance(FakeSession.instances[0], FakeSession)
