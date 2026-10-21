"""Patch registry mechanics on a fake family handle (no NVIDIA code needed)."""

from __future__ import annotations

import types

import pytest
from PIL import Image

from flora_deploy.accel import ALL_PATCHES, Patch, PatchError, apply_patches, registered
from flora_deploy.accel import registry as reg
from flora_deploy.accel.state import AccelState, ByteLRU, image_key
from flora_deploy.nv import Family


class FakeNv:
    family = Family.SEGCAP

    def __init__(self):
        self.mods = {
            "pipeline": types.ModuleType("fake.pipeline"),
            "media": types.ModuleType("fake.media"),
        }
        self.mods["pipeline"].f = lambda x: x + 1
        self.mods["pipeline"].g = lambda x: x * 2
        self.mods["media"].f = self.mods["pipeline"].f

        class Klass:
            def m(self):
                return "orig"

        self.mods["pipeline"].Klass = Klass
        self.vmods = {"plans": types.ModuleType("fake.plans")}
        self.vmods["plans"].h = lambda: "h"

    def module(self, name):
        if name not in self.mods:
            raise ModuleNotFoundError(name)
        return self.mods[name]

    def vendor(self, name):
        if name not in self.vmods:
            raise ModuleNotFoundError(name)
        return self.vmods[name]


def _wrap(tag):
    def factory(nv, current, state):
        out = {}
        for t, fn in current.items():
            def wrapped(*a, _fn=fn, **k):
                return (tag, _fn(*a, **k))
            out[t] = wrapped
        return out
    return factory


@pytest.fixture
def test_patches():
    names = []

    def make(name, targets, factory=None, family=Family.SEGCAP):
        p = Patch(name=name, family=family, targets=tuple(targets),
                  factory=factory or _wrap(name), parity_test="tests/x.py::t",
                  description="test")
        reg.register(p)
        names.append(name)
        return p

    yield make
    for n in names:
        reg._REGISTRY.pop(n, None)


def test_builtin_patches_registered_in_order():
    assert list(ALL_PATCHES) == ["decode_once", "transport_pool", "png_inmem", "geometry_pool"]
    regs = registered()
    for name in ALL_PATCHES:
        p = regs[name]
        assert p.family is Family.SEGCAP
        assert p.parity_test.startswith("tests/nvlink/test_accel_parity.py::")
        assert p.targets and p.description


def test_apply_restore_and_provenance(test_patches):
    nv = FakeNv()
    test_patches("t_a", ["batch_utils.pipeline.f", "batch_utils.media.f"])
    test_patches("t_b", ["batch_utils.pipeline.f", "vendor.plans.h",
                         "batch_utils.pipeline.Klass.m"])
    orig_f, orig_h, orig_m = nv.mods["pipeline"].f, nv.vmods["plans"].h, nv.mods["pipeline"].Klass.m
    active = apply_patches(nv, ["t_b", "t_a"], workers=1)
    assert active.names == ["t_a", "t_b"]  # registry order, not request order
    assert nv.mods["pipeline"].f(1) == ("t_b", ("t_a", 2))  # composed in order
    assert nv.mods["media"].f(1) == ("t_a", 2)
    assert nv.vmods["plans"].h() == ("t_b", "h")
    assert nv.mods["pipeline"].Klass().m() == ("t_b", "orig")
    prov = active.provenance()
    assert prov["schema"] == reg.PATCH_SCHEMA
    assert [p["name"] for p in prov["patches"]] == ["t_a", "t_b"]
    active.restore()
    active.restore()  # idempotent
    assert nv.mods["pipeline"].f is orig_f and nv.mods["media"].f is orig_f
    assert nv.vmods["plans"].h is orig_h and nv.mods["pipeline"].Klass.m is orig_m


def test_patches_are_individually_toggleable(test_patches):
    nv = FakeNv()
    test_patches("t_a", ["batch_utils.pipeline.f"])
    test_patches("t_b", ["batch_utils.pipeline.g"])
    with apply_patches(nv, ["t_b"]):
        assert nv.mods["pipeline"].f(1) == 2
        assert nv.mods["pipeline"].g(2) == ("t_b", 4)
    assert nv.mods["pipeline"].g(2) == 4


def test_errors_and_rollback(test_patches):
    nv = FakeNv()
    test_patches("t_a", ["batch_utils.pipeline.f"])
    test_patches("t_rewards", ["batch_utils.pipeline.f"], family=Family.REWARDER)
    with pytest.raises(PatchError, match="unknown"):
        apply_patches(nv, ["nope"])
    with pytest.raises(PatchError, match="rewarder"):
        apply_patches(nv, ["t_rewards"])
    first = apply_patches(nv, ["t_a"])
    with pytest.raises(PatchError, match="already active"):
        apply_patches(nv, ["t_a"])
    first.restore()

    def boom(nv, current, state):
        raise RuntimeError("factory failed")

    test_patches("t_boom", ["batch_utils.pipeline.g"], factory=boom)
    orig = nv.mods["pipeline"].f
    with pytest.raises(RuntimeError, match="factory failed"):
        apply_patches(nv, ["t_a", "t_boom"])
    assert nv.mods["pipeline"].f is orig  # t_a rolled back
    test_patches("t_bad", ["batch_utils.nope.f"])
    with pytest.raises(PatchError):
        apply_patches(nv, ["t_bad"])
    test_patches("t_attr", ["batch_utils.pipeline.missing"])
    with pytest.raises(PatchError, match="no attribute"):
        apply_patches(nv, ["t_attr"])


def test_restore_is_lifo_across_patch_sets(test_patches):
    nv = FakeNv()
    test_patches("t_a", ["batch_utils.pipeline.f"])
    test_patches("t_b", ["batch_utils.pipeline.f"])
    orig = nv.mods["pipeline"].f
    outer = apply_patches(nv, ["t_a"])
    inner = apply_patches(nv, ["t_b"])
    with pytest.raises(PatchError, match="re-patched"):
        outer.restore()
    inner.restore()
    outer.restore()
    assert nv.mods["pipeline"].f is orig


def test_byte_lru_evicts_and_demotes():
    lru = ByteLRU(30, sizeof=len)
    lru.put("a", "x" * 10)
    lru.put("b", "x" * 10)
    lru.put("c", "x" * 10)
    assert lru.get("a") is not None  # a is now most recent
    lru.demote("c")
    lru.put("d", "x" * 10)
    assert lru.get("c") is None and lru.get("b") is not None and lru.get("d") is not None
    lru.put("huge", "x" * 31)  # larger than the budget: not kept
    assert lru.get("huge") is None and lru.nbytes <= 30
    lru.clear()
    assert len(lru) == 0 and lru.nbytes == 0


def test_image_key_tracks_everything_an_encoder_reads():
    a = Image.new("RGB", (4, 3), (1, 2, 3))
    assert image_key(a) == image_key(a.copy())
    b = a.copy()
    b.putpixel((0, 0), (1, 2, 4))
    assert image_key(a) != image_key(b)
    assert image_key(a) != image_key(a.convert("RGBA"))
    assert image_key(a) != image_key(Image.new("RGB", (3, 4), (1, 2, 3)))
    c = a.copy()
    c.info["icc_profile"] = b"x"
    assert image_key(a) != image_key(c)


def test_state_inline_and_pool_submit_and_episode():
    for workers in (1, 3):
        st = AccelState(workers=workers)
        assert st.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
        fut = st.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            fut.result()
        st.frames.put("outside", Image.new("RGB", (2, 2)))  # default scope
        with st.episode():
            assert st.frames.get("outside") is None  # an episode has its own scope
            st.frames.put("k", Image.new("RGB", (2, 2)))
            scope = st.scope()
            with st.episode():
                assert st.scope() is scope  # nested: same scope
            assert st.frames.get("k") is not None  # inner exit keeps caches
            # pool tasks run in the submitting context: same scope
            assert st.submit(st.scope).result() is scope
        assert scope.closed and len(scope.frames) == 0  # outer exit clears
        assert st.frames.get("k") is None and st.frames.get("outside") is not None
        st.reset()
        assert st.frames.get("outside") is None
        st.close()


def test_byte_budget_is_shared_across_lrus():
    from flora_deploy.accel.state import ByteBudget

    budget = ByteBudget(50)
    a, b = ByteLRU(40, sizeof=len, budget=budget), ByteLRU(40, sizeof=len, budget=budget)
    for i in range(3):
        a.put(f"a{i}", "x" * 10)
    for i in range(3):
        b.put(f"b{i}", "x" * 10)
    assert a.nbytes + b.nbytes == budget.total <= 50 and budget.peak <= 50
    assert b.get("b2") is not None and len(a) + len(b) == 5  # one oldest entry gave way
    a.put("a9", "x" * 30)  # a is now the largest: it gives way first
    assert budget.total <= 50 and a.get("a9") is not None and len(b) >= 1
    a.close()
    assert budget.total == b.nbytes
    a.put("late", "x")  # closed: dropped
    assert a.get("late") is None
