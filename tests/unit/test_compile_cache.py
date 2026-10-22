"""Warm start: keyed, verified vLLM compile caches (``engines.compile_cache``), the balanced
worker's use of them (``modal_app.balanced_app``) and warm/cold startup pricing."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flora_deploy.engines import compile_cache as cc
from flora_deploy.engines.offline_llm import EngineSpec, spec_for

MODEL = "/weights/derived/tempseg-merged@abc"
RUNTIME = {"vllm": "0.24.0", "torch": "2.11.0", "triton": "3.6.0", "torch_cuda": "13.0"}
VERSIONS = {"vllm": "0.24.0", "torch": "2.11.0", "triton": "3.6.0", "cuda_base": "cuda:13"}


def key_of(spec, **kw):
    args = {"model_content": "c" * 64, "image_fp": "img", "versions": VERSIONS,
            "worker_sources": "w" * 64, **kw}
    return cc.spec_key(spec, **args)


def seg_key(**kw):
    return key_of(spec_for("segment", MODEL, "L40S"), **kw)


def fill_tree(local: Path, extra: dict[str, bytes] | None = None) -> None:
    files = {"vllm/torch_compile_cache/torch_aot_compile/h/rank_0_0/model": b"aot" * 100,
             "vllm/torch_compile_cache/abc/rank_0_0/backbone/vllm_compile_cache.py": b"x = 1",
             "triton/K1/kernel.cubin": b"\x00\x01cubin", "xdg/flashinfer/a.so": b"so",
             "triton/K1/held.lock": b"", **(extra or {})}
    for rel, data in files.items():
        p = local / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def published_cache(tmp_path, key=None):
    """A Volume (``vol``) holding one publish of ``key``, made by :func:`cc.publish`."""
    key = key or seg_key()
    src = tmp_path / "src"
    fill_tree(src)
    vol = tmp_path / "vol"
    res = cc.publish(key, src, cc.DirSink(vol), cc.FileLeases(tmp_path / "leases"),
                     holder="ta-1", runtime=RUNTIME, log=lambda m: None)
    assert res["published"], res
    return key, vol, res


# --- key ------------------------------------------------------------------------------------


def test_lab_gate_a_engine_has_the_worker_key():
    """The lab's Gate A engines differ from the workers only in what logprob requests return:
    vLLM 0.24 ignores logprobs_mode in every compile hash and hashes max_logprobs by value (its
    default is the lab's 20), so the key is the same and the lab caches apply."""
    worker = spec_for("segment", MODEL, "L40S")
    lab = EngineSpec.for_gpu("tempseg", MODEL, "L40S", logprobs_mode="raw_logits",
                             max_logprobs=20)
    assert worker.engine_kwargs() != lab.engine_kwargs()
    assert key_of(worker) == key_of(lab)
    other = EngineSpec.for_gpu("tempseg", MODEL, "L40S", max_logprobs=5)
    assert key_of(other) != key_of(worker)  # a different max_logprobs is a different compile


def test_key_changes_with_every_compile_factor():
    base = seg_key()
    variants = [
        key_of(spec_for("segment", MODEL, "H100")),  # GPU (vLLM's AOT key has none)
        key_of(spec_for("segment", MODEL, "L40S", dtbias_fp32=True)),  # plugin on
        key_of(spec_for("segment", MODEL + "x", "L40S")),  # model path
        key_of(spec_for("caption", MODEL, "H100")),
        seg_key(image_fp="img2"), seg_key(model_content="d" * 64),
        seg_key(versions={**VERSIONS, "torch": "2.12.0"}),
        seg_key(versions={**VERSIONS, "vllm": "0.24.1"}),
        seg_key(versions={**VERSIONS, "cuda_base": "cuda:13.1"}),
        seg_key(worker_sources="v" * 64),
    ]
    assert len({k.digest for k in [base, *variants]}) == len(variants) + 1
    kw = spec_for("segment", MODEL, "L40S").engine_kwargs()
    for change in ({"dtype": "float16"}, {"max_model_len": 16384},
                   {"gdn_prefill_backend": "flashinfer"}, {"max_num_seqs": 8}):
        f = cc.key_factors(engine_kind="tempseg", gpu="L40S", engine_kwargs={**kw, **change},
                           dtbias_fp32=False, model_content="c" * 64, image_fp="img",
                           versions=VERSIONS, worker_sources="w" * 64)
        assert cc.CacheKey(f).digest != base.digest, change
    assert base.name.startswith("tempseg-L40S-") and len(base.name.split("-")[-1]) == 24
    assert key_of(spec_for("segment", MODEL, "L40S", dtbias_fp32=True)).name.startswith(
        "tempseg-dtbias-L40S-")
    assert key_of(spec_for("segment", MODEL, "L40S!")) == base  # GPU names are normalised


def test_worker_key_is_derivable_on_a_laptop():
    k = cc.worker_key("segment", "L40S")
    assert k.factors["versions"]["vllm"] == "0.24.0" and k.factors["versions"]["torch"]
    assert k.factors["versions"]["triton"] and k.factors["model_content"]
    assert k.factors["engine_kwargs"]["model"].startswith("/weights/derived/tempseg-merged@")
    assert k == cc.worker_key("segment", "L40S")  # stable
    assert cc.worker_key("caption", "H100!").name.startswith("captioner-H100-")


# --- leases ---------------------------------------------------------------------------------


def test_lease_one_holder_takeover_and_done(tmp_path):
    leases, now = cc.FileLeases(tmp_path), [1000.0]
    clock = lambda: now[0]  # noqa: E731
    a, why = cc.acquire(leases, "k", "A", ttl_s=60, clock=clock)
    assert a is not None and a.generation == 0 and why == "acquired"
    b, why = cc.acquire(leases, "k", "B", ttl_s=60, clock=clock)
    assert b is None and "leased by A" in why
    now[0] += 61  # A's lease expired: exactly one successor wins generation 1
    b, _ = cc.acquire(leases, "k", "B", ttl_s=60, clock=clock)
    c, why = cc.acquire(leases, "k", "C", ttl_s=60, clock=clock)
    assert b is not None and b.generation == 1 and c is None and "leased by B" in why
    assert not cc.still_held(leases, a) and cc.still_held(leases, b)
    leases.put(b.key, {"holder": "B", "done": True, "expires": float("inf")})
    d, why = cc.acquire(leases, "k", "D", clock=lambda: now[0] + 1e9)
    assert d is None and "already published by B" in why


def test_dict_leases_use_put_if_absent():
    store: dict = {}

    class FakeDict:
        def put(self, k, v, skip_if_exists=False):
            if skip_if_exists and k in store:
                return False
            store[k] = v
            return True

        def get(self, k):
            return store.get(k)

    leases = cc.DictLeases(FakeDict())
    assert cc.acquire(leases, "k", "A")[0] is not None
    assert cc.acquire(leases, "k", "B")[0] is None


# --- publish and load -------------------------------------------------------------------------


def test_publish_then_seed_round_trip(tmp_path):
    key, vol, res = published_cache(tmp_path)
    base = vol / cc.ROOT / key.name
    m = json.loads((base / cc.MANIFEST).read_text())
    assert m["publish_id"] == res["publish_id"] and m["runtime"] == RUNTIME
    assert "triton/K1/held.lock" not in m["files"]  # lock files are never published
    assert all((base / m["publish_id"] / rel).is_file() for rel in m["files"])
    local = tmp_path / "local" / key.name
    st = cc.seed(key, vol, local, runtime=RUNTIME)
    assert st.state == "warm" and st.files == len(m["files"]) and st.publish_id == m["publish_id"]
    assert (local / "triton/K1/kernel.cubin").read_bytes() == b"\x00\x01cubin"
    env = cc.env_for(local)
    assert env["VLLM_CACHE_ROOT"] == str(local / "vllm") and Path(env["CUDA_CACHE_PATH"]).is_dir()


def test_publish_is_once_per_key(tmp_path):
    key, vol, _ = published_cache(tmp_path)
    leases = cc.FileLeases(tmp_path / "leases")
    again = cc.publish(key, tmp_path / "src", cc.DirSink(vol), leases, holder="ta-2",
                       runtime=RUNTIME, log=lambda m: None)
    assert not again["published"] and "already published by ta-1" in again["reason"]
    seen = cc.publish(key, tmp_path / "src", cc.DirSink(vol), cc.FileLeases(tmp_path / "l2"),
                      holder="ta-3", runtime=RUNTIME, mount=vol, log=lambda m: None)
    assert not seen["published"] and "manifest already exists" in seen["reason"]


def test_concurrent_publishers_one_manifest(tmp_path):
    """Two containers race: one lease, one tree, one manifest naming it."""
    key = seg_key()
    leases, vol = cc.FileLeases(tmp_path / "leases"), tmp_path / "vol"
    results = []
    for who in ("A", "B"):
        src = tmp_path / who
        fill_tree(src, {f"triton/{who}/only.bin": who.encode()})
        results.append(cc.publish(key, src, cc.DirSink(vol), leases, holder=who,
                                  runtime=RUNTIME, log=lambda m: None))
    assert [r["published"] for r in results] == [True, False]
    trees = [p.name for p in (vol / cc.ROOT / key.name).iterdir() if p.is_dir()]
    assert trees == [results[0]["publish_id"]]


class _FailingSink(cc.DirSink):
    def __init__(self, root, fail_after):
        super().__init__(root)
        self.left = fail_after

    def _write(self, remote, data):
        if self.left == 0:
            raise OSError("upload interrupted")
        self.left -= 1
        super()._write(remote, data)


def test_interrupted_publish_is_invisible(tmp_path):
    key = seg_key()
    src = tmp_path / "src"
    fill_tree(src)
    vol = tmp_path / "vol"
    res = cc.publish(key, src, _FailingSink(vol, 2), cc.FileLeases(tmp_path / "l"),
                     holder="A", runtime=RUNTIME, log=lambda m: None)
    assert not res["published"] and "upload interrupted" in res["reason"]
    assert not (vol / cc.ROOT / key.name / cc.MANIFEST).exists()  # partial tree, no manifest
    st = cc.seed(key, vol, tmp_path / "local", runtime=RUNTIME)
    assert st.state == "cold" and st.reason == cc.NO_CACHE


def test_lost_lease_never_writes_the_manifest(tmp_path):
    key = seg_key()
    src = tmp_path / "src"
    fill_tree(src)
    leases = cc.FileLeases(tmp_path / "l")

    class Sink(cc.DirSink):
        def upload(self, files):
            super().upload(files)
            leases.put_if_absent(f"{key.name}/1", {"holder": "B", "expires": 1e18})

    res = cc.publish(key, src, Sink(tmp_path / "vol"), leases, holder="A", runtime=RUNTIME,
                     log=lambda m: None)
    assert not res["published"] and "lease lost" in res["reason"]
    assert not (tmp_path / "vol" / cc.ROOT / key.name / cc.MANIFEST).exists()


def _corrupt_byte(vol, key):
    m = json.loads((vol / cc.ROOT / key.name / cc.MANIFEST).read_text())
    p = vol / cc.ROOT / key.name / m["publish_id"] / "triton/K1/kernel.cubin"
    p.write_bytes(b"\xff" + p.read_bytes()[1:])


def _truncate(vol, key):
    m = json.loads((vol / cc.ROOT / key.name / cc.MANIFEST).read_text())
    p = vol / cc.ROOT / key.name / m["publish_id"] / "xdg/flashinfer/a.so"
    p.write_bytes(b"s")


def _delete_file(vol, key):
    m = json.loads((vol / cc.ROOT / key.name / cc.MANIFEST).read_text())
    (vol / cc.ROOT / key.name / m["publish_id"] / "xdg/flashinfer/a.so").unlink()


def _edit_manifest(**changes):
    def edit(vol, key):
        p = vol / cc.ROOT / key.name / cc.MANIFEST
        m = json.loads(p.read_text())
        for k, v in changes.items():
            m[k] = v(m) if callable(v) else v
        p.write_text(json.dumps(m))
    return edit


def _garbage_manifest(vol, key):
    (vol / cc.ROOT / key.name / cc.MANIFEST).write_text('{"schema": "flora-compile-ca')


@pytest.mark.parametrize("damage,why", [
    (_corrupt_byte, "sha256"),
    (_truncate, "bytes"),
    (_delete_file, "FileNotFoundError"),
    (_garbage_manifest, "not JSON"),
    (_edit_manifest(schema="other/v0"), "manifest"),
    (_edit_manifest(digest="0" * 64), "not 'tempseg"),
    (_edit_manifest(factors=lambda m: {**m["factors"], "gpu": "H100"}), "factors differ"),
    (_edit_manifest(runtime=lambda m: {**m["runtime"], "torch": "2.12.0"}), "another runtime"),
    (_edit_manifest(publish_id="../../etc"), "bad publish id"),
    (_edit_manifest(files=lambda m: {"../escape": {"size": 1, "sha256": "0" * 64}}),
     "unsafe path"),
    (_edit_manifest(files={}), "no files"),
])
def test_bad_cache_falls_back_cold(tmp_path, damage, why):
    key, vol, _ = published_cache(tmp_path)
    damage(vol, key)
    local = tmp_path / "local" / key.name
    (local / "vllm").mkdir(parents=True)
    (local / "vllm" / "stale").write_text("old")  # a previous attempt's leftovers go too
    logs: list[str] = []
    st = cc.seed(key, vol, local, runtime=RUNTIME, log=logs.append)
    assert st.state == "cold" and why in st.reason, st.reason
    assert logs and "compiling cold" in logs[0]
    assert sorted(p.name for p in local.iterdir()) == sorted(cc.TREE_DIRS)
    assert not any(p.is_file() for p in local.rglob("*"))  # empty dirs only
    assert not list(local.parent.glob(".*staging*"))


def test_other_key_is_never_used(tmp_path):
    key, vol, _ = published_cache(tmp_path)
    other = key_of(spec_for("segment", MODEL, "H100"))
    assert cc.seed(other, vol, tmp_path / "o", runtime=RUNTIME).state == "cold"
    assert cc.seed(key, vol, tmp_path / "k", runtime={**RUNTIME, "vllm": "0.25.0"}).state \
        == "cold"


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(root.rglob("*"))}


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_consumers_read_a_read_only_mount(tmp_path):
    key, vol, _ = published_cache(tmp_path)
    before = _snapshot(vol)
    dirs = [vol, *[p for p in vol.rglob("*") if p.is_dir()]]
    for d in dirs:
        d.chmod(0o555)
    try:
        for i in range(2):
            st = cc.seed(key, vol, tmp_path / f"c{i}" / key.name, runtime=RUNTIME)
            assert st.state == "warm"
        assert _snapshot(vol) == before  # nothing written next to the cache
    finally:
        for d in dirs:
            d.chmod(0o755)


def test_legacy_seed(tmp_path):
    key = seg_key()
    legacy = tmp_path / "balanced" / "tempseg"
    fill_tree(legacy)
    st = cc.seed_legacy(legacy, tmp_path / "local", key)
    assert st.state == "legacy" and (tmp_path / "local/vllm").is_dir()
    assert not (tmp_path / "local/triton/K1/held.lock").exists()
    missing = cc.seed_legacy(tmp_path / "none", tmp_path / "l2", key)
    assert missing.state == "cold"


# --- the worker's view ------------------------------------------------------------------------


def _worker_cache(tmp_path, state="cold", **kw):
    key = seg_key()
    local = tmp_path / "local" / key.name
    fill_tree(local)
    sinks: list = []

    def sink():
        s = cc.DirSink(tmp_path / "vol")
        sinks.append(s)
        return s

    wc = cc.WorkerCache(key, local, cc.CacheStatus(state, key.name), holder="ta-9",
                        sink=sink, leases=lambda: cc.FileLeases(tmp_path / "leases"),
                        log=lambda m: None, **kw)
    return wc, sinks


def test_worker_publishes_once_after_generates(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "runtime_versions", lambda: RUNTIME)
    wc, sinks = _worker_cache(tmp_path)
    assert wc.maybe_publish(SimpleNamespace(stats={"requests": 0}, dead=None)) is None
    assert wc.maybe_publish(SimpleNamespace(stats={"requests": 3}, dead="EngineDead")) is None
    assert sinks == []
    res = wc.maybe_publish(SimpleNamespace(stats={"requests": 3}, dead=None))
    assert res["published"] and wc.status.published == res
    assert wc.maybe_publish(SimpleNamespace(stats={"requests": 9}, dead=None)) is None
    st = cc.seed(wc.key, tmp_path / "vol", tmp_path / "next", runtime=RUNTIME)
    assert st.state == "warm"  # the next container loads it


def test_worker_never_republishes_a_verified_cache(tmp_path):
    wc, sinks = _worker_cache(tmp_path, state="warm")
    assert wc.maybe_publish(SimpleNamespace(stats={"requests": 3}, dead=None)) is None
    off, sinks2 = _worker_cache(tmp_path / "b", publish_enabled=False)
    assert off.maybe_publish(SimpleNamespace(stats={"requests": 3}, dead=None)) is None
    assert sinks == sinks2 == []


def test_publish_errors_never_reach_the_worker(tmp_path):
    def broken():
        raise RuntimeError("no Modal client")

    wc, _ = _worker_cache(tmp_path)
    wc.leases = broken
    res = wc.maybe_publish(SimpleNamespace(stats={"requests": 1}, dead=None))
    assert res == {"key": wc.key.name, "published": False,
                   "reason": "RuntimeError: no Modal client"}


# --- balanced_app ----------------------------------------------------------------------------


@pytest.fixture
def ba(monkeypatch):
    from flora_deploy.modal_app import balanced_app

    monkeypatch.setattr(cc, "runtime_versions", lambda: RUNTIME)
    monkeypatch.setattr(cc, "declared_versions", lambda lock=None: VERSIONS)
    return balanced_app


def test_prepare_cache_warm_legacy_and_cold(tmp_path, ba):
    spec = spec_for("segment", MODEL, "L40S")
    key = cc.spec_key(spec, model_content="c" * 64, image_fp="img")
    mount = tmp_path / "mount"
    kw = dict(mount=mount, local_root=tmp_path / "local", image_fp="img", log=lambda m: None)
    cold = ba.prepare_cache("segment", spec, "c" * 64, {}, **kw)
    assert cold.status.state == "cold" and cold.publish_enabled and cold.key == key
    fill_tree(mount / "balanced" / "tempseg")  # the lab's unkeyed cache, same GPU
    legacy = ba.prepare_cache("segment", spec, "c" * 64, {}, **kw)
    assert legacy.status.state == "legacy"
    h100 = spec_for("segment", MODEL, "H100")  # the lab compiled on an L40S: not offered
    assert ba.prepare_cache("segment", h100, "c" * 64, {}, **kw).status.state == "cold"
    plugin = spec_for("segment", MODEL, "L40S", dtbias_fp32=True)  # no tempseg-dtbias here
    assert ba.prepare_cache("segment", plugin, "c" * 64, {}, **kw).status.state == "cold"
    src = tmp_path / "src"
    fill_tree(src)
    cc.publish(key, src, cc.DirSink(mount), cc.FileLeases(tmp_path / "l"), holder="x",
               runtime=RUNTIME, log=lambda m: None)
    warm = ba.prepare_cache("segment", spec, "c" * 64, {ba.PUBLISH_ENV: "0"}, **kw)
    assert warm.status.state == "warm" and not warm.publish_enabled
    assert warm.env["VLLM_CACHE_ROOT"] == str(tmp_path / "local" / key.name / "vllm")


def test_prepare_cache_without_image_fingerprint(tmp_path, ba, monkeypatch):
    monkeypatch.setattr(ba, "_image_fp", lambda env: None)
    wc = ba.prepare_cache("caption", spec_for("caption", MODEL, "H100"), "c" * 64, {},
                          mount=tmp_path, local_root=tmp_path / "l", log=lambda m: None)
    assert wc.status.state == "cold" and not wc.publish_enabled and "fingerprint" in \
        wc.status.reason


def test_forced_cold_without_a_key_still_records_cold_forced(tmp_path, ba, monkeypatch):
    """Review 14 W3: the unkeyed path keeps the forced-cold identity the witness needs."""
    monkeypatch.setattr(ba, "_image_fp", lambda env: None)
    logs = []
    wc = ba.prepare_cache("caption", spec_for("caption", MODEL, "H100"), "c" * 64,
                          {ba.COMPILE_CACHE_ENV: "off"}, mount=tmp_path,
                          local_root=tmp_path / "l", log=logs.append)
    assert wc.status.state == "cold" and not wc.publish_enabled
    assert wc.status.reason == cc.FORCED_COLD and wc.status.identity() == "cold:forced"
    assert any("fingerprint" in m for m in logs)  # why there was no key is still logged


def test_legacy_lab_gpus_match_the_lab():
    from flora_deploy.engines.offline_llm import normalize_gpu
    from flora_deploy.modal_app import balanced_app, balanced_lab

    assert balanced_app.LEGACY_LAB_GPUS == {k: normalize_gpu(g)
                                            for k, g in balanced_lab.GPUS.items()}


def test_build_engine_retries_cold_once(tmp_path, ba):
    from flora_deploy.engines.offline_llm import EngineVerifyError

    wc, _ = _worker_cache(tmp_path, state="warm")
    builds: list[int] = []

    def engine(spec):
        builds.append(len(list(wc.local.rglob("*.cubin"))))
        if len(builds) == 1:
            raise RuntimeError("Inductor artifact load crashed")
        return "engine"

    assert ba.build_engine("spec", wc, engine) == "engine"
    assert builds == [1, 0] and wc.status.state == "cold" and "build failed" in wc.status.reason

    cold, _ = _worker_cache(tmp_path / "c")
    with pytest.raises(RuntimeError):  # a cold build's failure is the real failure
        ba.build_engine("spec", cold, lambda s: (_ for _ in ()).throw(RuntimeError("oom")))
    warm, _ = _worker_cache(tmp_path / "w", state="warm")
    with pytest.raises(EngineVerifyError):  # deterministic: no retry
        ba.build_engine("spec", warm,
                        lambda s: (_ for _ in ()).throw(EngineVerifyError("dtypes")))
    assert warm.status.state == "warm"


def test_after_shard_publishes(tmp_path, ba, monkeypatch):
    wc, _ = _worker_cache(tmp_path)
    runner = SimpleNamespace(compile_cache=wc, engine=SimpleNamespace(stats={"requests": 2},
                                                                      dead=None))
    ba.after_shard(runner)
    assert wc.status.published["published"]
    ba.after_shard(SimpleNamespace())  # a refused runner has no cache: nothing happens


# --- pricing ---------------------------------------------------------------------------------


def _table(*rows):
    from flora_deploy.planner.rates import RateTable

    base = {"uncertainty": 0.5, "status": "measured", "provenance": "x", "date": "d"}
    return RateTable.from_dict({"schema": 1, "startup": [{**base, **r} for r in rows]},
                               source="t")


def test_startup_for_warm_and_cold():
    t = _table({"preset": "balanced", "kind": "segment", "startup_s": 300,
                "compile_cache": "cold"},
               {"preset": "balanced", "kind": "segment", "startup_s": 120,
                "compile_cache": "warm", "status": "placeholder"},
               {"preset": "balanced", "kind": "*", "startup_s": 130, "status": "projected"})
    assert t.startup_for("balanced", "segment").startup_s == 300  # default: cold
    assert t.startup_for("balanced", "segment", warm=True).startup_s == 120
    assert t.startup_for("balanced", "caption", warm=True).startup_s == 130  # no warm row
    only_warm = _table({"preset": "balanced", "kind": "segment", "startup_s": 120,
                        "compile_cache": "warm"})
    assert only_warm.startup_for("balanced", "segment").status == "placeholder"  # never warm
    assert only_warm.startup_for("balanced", "segment").startup_s == 0


def test_shipped_warm_rows_are_placeholders():
    from flora_deploy.planner.rates import RateTable, startup_cache

    t = RateTable.load()
    for kind in ("segment", "caption"):
        cold, warm = t.startup_for("balanced", kind), t.startup_for("balanced", kind, warm=True)
        assert cold.compile_cache == "cold" and cold.status == "measured"
        assert warm.compile_cache == "warm" and warm.status == "placeholder"
        assert warm.startup_s < cold.startup_s
    assert startup_cache("segment") is False
    assert startup_cache("segment", warm_kinds=("segment",)) is True
    assert startup_cache("caption", assume_warm=True) is True


def test_planner_prices_warm_only_with_a_cache_or_assume_warm():
    from flora_deploy.jobs.budget import pinned_rates
    from flora_deploy.planner.options import PlanRequest, WorkloadSummary, build_option
    from flora_deploy.planner.rates import RateTable

    t = RateTable.load()
    req = PlanRequest(WorkloadSummary.from_hours(1.0), presets=("balanced",), max_gpus=4)

    def option(r):
        o = build_option(r, t, pinned_rates(), "balanced", 1)
        return o, {k.kind: k for k in o.kinds}

    cold, ck = option(req)
    assert {k.startup_cache for k in ck.values()} == {"cold"}
    assert not any("priced warm" in f for f in cold.flags)
    seg, sk = option(replace(req, warm_kinds=("segment",)))
    assert sk["segment"].startup_cache == "warm" and sk["caption"].startup_cache == "cold"
    assert sk["segment"].startup_s == t.startup_for("balanced", "segment", warm=True).startup_s
    assert seg.cost_usd < cold.cost_usd
    assert any(f.startswith("startup priced warm (a published compile cache): segment")
               for f in seg.flags)
    proj, pk = option(replace(req, assume_warm=True))
    assert {k.startup_cache for k in pk.values()} == {"warm"}
    assert any("(--assume-warm)" in f and "placeholder" in f for f in proj.flags)


def test_budget_charges_cold():
    from flora_deploy.jobs.budget import RateTableEstimator

    est = RateTableEstimator.default("balanced")
    assert est.startup_s("segment") == est.table.startup_for("balanced", "segment").startup_s
    assert est.table.startup_for("balanced", "segment").compile_cache == "cold"


def test_published_kinds_and_plan_check(tmp_path, monkeypatch):
    k = cc.worker_key("segment", "L40S")
    paths = {f"{cc.ROOT}/{k.name}/{cc.MANIFEST}"}
    assert cc.published_kinds({"segment": "L40S", "caption": "H100!",
                               "reward.terminal": "H100!"}, paths.__contains__) == ("segment",)
    assert cc.published_kinds({"segment": "H100"}, paths.__contains__) == ()  # other GPU

    def boom(path):
        raise OSError("network")

    assert cc.published_kinds({"segment": "L40S"}, boom) == ()
    from flora_deploy.cli.plan import published_cache_kinds

    cfg = SimpleNamespace(preset="exact", pools=lambda: {})
    assert published_cache_kinds(cfg, paths.__contains__) == ("segment",)


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_cc_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scripts_price_warm_pools():
    from flora_deploy.planner.rates import RateTable

    t = RateTable.load()
    st = _load("scale_test")
    cold, warm = st.kind_rates("real", t), st.kind_rates("real", t, warm_kinds=("caption",))
    assert warm["caption"].startup_s == t.startup_for("balanced", "caption", warm=True).startup_s
    assert warm["segment"] == cold["segment"] and warm["caption"].startup_s < \
        cold["caption"].startup_s
    wh = _load("wave_h")
    c, w = wh.balanced_job_seconds(100.0, t), wh.balanced_job_seconds(
        100.0, t, warm_kinds=("segment", "caption"))
    assert w["segment"][0] < c["segment"][0] and w["caption"][0] < c["caption"][0]
    assert wh.warm_startup_s(t.startup_for("balanced", "segment")) is None


# --- review 13 W1/W2: which cache a container ran is recorded; one can be forced cold --------


def test_cache_identity_names_the_manifest(tmp_path, ba):
    """A warm start replays the publisher's Inductor autotune picks (``.best_config``), so a
    record must say which cache its container loaded: key + manifest hash, or cold."""
    import hashlib

    spec = spec_for("segment", MODEL, "L40S")
    key = cc.spec_key(spec, model_content="c" * 64, image_fp="img")
    mount = tmp_path / "mount"
    kw = dict(mount=mount, local_root=tmp_path / "local", image_fp="img", log=lambda m: None)
    cold = ba.prepare_cache("segment", spec, "c" * 64, {}, **kw)
    assert cold.status.identity() == "cold"
    src = tmp_path / "src"
    fill_tree(src)
    cc.publish(key, src, cc.DirSink(mount), cc.FileLeases(tmp_path / "l"), holder="x",
               runtime=RUNTIME, log=lambda m: None)
    warm = ba.prepare_cache("segment", spec, "c" * 64, {}, **kw)
    sha = hashlib.sha256((mount / cc.ROOT / key.name / cc.MANIFEST).read_bytes()).hexdigest()
    assert warm.status.manifest_sha256 == sha
    assert warm.status.identity() == f"warm:{key.name}:{sha[:16]}"
    assert warm.status.as_json()["identity"] == warm.status.identity()
    assert cc.CacheStatus("legacy", key.name).identity() == f"legacy:{key.name}"


def test_compile_cache_off_forces_one_container_cold(tmp_path, ba):
    """``FLORA_COMPILE_CACHE=off``: no seed even when a verified cache exists, no publish."""
    spec = spec_for("segment", MODEL, "L40S")
    key = cc.spec_key(spec, model_content="c" * 64, image_fp="img")
    mount = tmp_path / "mount"
    src = tmp_path / "src"
    fill_tree(src)
    cc.publish(key, src, cc.DirSink(mount), cc.FileLeases(tmp_path / "l"), holder="x",
               runtime=RUNTIME, log=lambda m: None)
    fill_tree(mount / "balanced" / "tempseg")  # a legacy one is not offered either
    kw = dict(mount=mount, local_root=tmp_path / "local", image_fp="img", log=lambda m: None)
    wc = ba.prepare_cache("segment", spec, "c" * 64, {ba.COMPILE_CACHE_ENV: "off"}, **kw)
    assert wc.status.state == "cold" and wc.status.reason == cc.FORCED_COLD
    assert wc.status.identity() == "cold:forced" and not wc.publish_enabled
    assert not any((tmp_path / "local" / key.name / "vllm").iterdir())
    with pytest.raises(ValueError, match="FLORA_COMPILE_CACHE"):
        ba.prepare_cache("segment", spec, "c" * 64, {ba.COMPILE_CACHE_ENV: "maybe"}, **kw)


def test_deploy_forwards_the_compile_cache_switch(monkeypatch):
    from flora_deploy.modal_app import balanced_app as ba
    from flora_deploy.modal_app import factory

    monkeypatch.delenv(ba.COMPILE_CACHE_ENV, raising=False)
    monkeypatch.delenv(ba.PUBLISH_ENV, raising=False)
    assert ba.COMPILE_CACHE_ENV not in factory._recipe_env("balanced", "nvidia")
    monkeypatch.setenv(ba.COMPILE_CACHE_ENV, "off")
    monkeypatch.setenv(ba.PUBLISH_ENV, "0")
    env = factory._recipe_env("balanced", "nvidia")
    assert env[ba.COMPILE_CACHE_ENV] == "off" and env[ba.PUBLISH_ENV] == "0"
    assert ba.COMPILE_CACHE_ENV not in factory._recipe_env("segcap", "nvidia")
