"""Balanced worker plumbing: index receipt -> NVIDIA arguments, upstream segment records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs import progress as P
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import EpisodeRef, Shard
from flora_deploy.records import Provenance
from flora_deploy.stages import segment as S
from flora_deploy.workers_impl.caption import MissingUpstream, UpstreamSegments
from flora_deploy.workers_impl.segment import InputResolver, ShardContext


def receipt(**kw):
    d = {"schema": "flora_index_receipt_v1", "status": "indexed", "uid": "ds--ep0",
         "episode_id": "ep0", "dataset": {"name": "ds"}, "pool": "yam", "profile": "yam",
         "yam32": False, "instruction": "dataset_task", "task": "fold the towel",
         "streams": {r: {"source": {"uri": f"videos/{r}.mp4"}}
                     for r in ("main", "left_wrist", "right_wrist")},
         "envelope": {"ok": True}}
    d.update(kw)
    return json.dumps(d).encode()


def ref(pool="yam", uid="ds--ep0"):
    return EpisodeRef(uid=uid, dataset="ds", episode_id=uid.split("--")[1], pool=pool,
                      duration_s=10.0, oracle_eligible=True, envelope_ok=True, index_ref="idx")


def test_input_resolver_reads_nvidia_arguments(tmp_path):
    store = LocalStore(tmp_path / "s")
    store.put_bytes("idx", receipt())
    inp = InputResolver(store, media_root="/data")(ref())
    assert inp.paths == {r: f"/data/videos/{r}.mp4" for r in ("main", "left_wrist",
                                                                "right_wrist")}
    assert (inp.profile, inp.yam32, inp.instruction) == ("yam", False, "fold the towel")
    store.put_bytes("idx", receipt(instruction="none"))
    assert InputResolver(store, media_root="/d")(ref()).instruction == ""
    store.put_bytes("idx", receipt(instruction="fixed", task=None))
    assert InputResolver(store, media_root="/d", default_task="Do it.")(ref()).instruction \
        == "Do it."
    store.put_bytes("idx", receipt(profile=None, yam32=None, pool="yam32"))
    inp = InputResolver(store, media_root="/d")(ref("yam32"))
    assert (inp.profile, inp.yam32) == ("yam", True)


def test_input_resolver_refuses_unknown_roles(tmp_path):
    store = LocalStore(tmp_path / "s")
    store.put_bytes("idx", receipt(streams={"cam_high": {"source": {"uri": "a.mp4"}}}))
    with pytest.raises(ValueError):
        InputResolver(store, media_root="/d")(ref())


def prov(cfg):
    return Provenance(models={}, engine="e", preset="balanced", config_hash=cfg,
                      logic_version="l", framework="f")


def seg(uid, cfg="cfg", start=0.0):
    return S.segment_record([dict(start_s=start, end_s=start + 1, metadata={"k": uid})],
                            uid=uid, dataset="ds", episode_id=uid, provenance=prov(cfg))


def test_upstream_segments_reads_parent_and_bisection_children(tmp_path):
    store = LocalStore(tmp_path / "s")
    jf = A.JobFiles(store, "job", "cfg")
    uids = ("ds--a", "ds--b", "ds--c")
    P.write_part(jf, "segment", "segment-00000", "ta", 0, [seg("ds--a")])
    P.write_part(jf, "segment", A.child_shard_id("segment", "ds--b"), "tb", 0, [seg("ds--b")])
    P.write_part(jf, "segment", "segment-00000", "tc", 0, [replace(seg("ds--c"))])
    foreign = A.JobFiles(store, "job", "other")
    P.write_part(foreign, "segment", "segment-00000", "td", 0, [seg("ds--c", "other", 5.0)])
    shard = Shard("caption-00000", "caption", uids, 1.0, depends_on=("segment-00000",))
    ctx = ShardContext(jf, shard, None, "tx")  # type: ignore[arg-type]
    up = UpstreamSegments()
    assert {u: up(ref(uid=u), ctx).segments[0].nv_metadata["k"] for u in uids} == \
        {u: u for u in uids}
    assert up(ref(uid="ds--c"), ctx).segments[0].start_s == 0.0  # not the foreign config's
    with pytest.raises(MissingUpstream):
        up(ref(uid="ds--z"), ctx)


def test_upstream_segments_accepts_segment_stage_hash(tmp_path):
    """H1: real runs key segment parts and stamp records with the segment *stage* hash."""
    from flora_deploy.workers_impl.reward import ShardPartSink

    store = LocalStore(tmp_path / "s")
    jf = A.JobFiles(store, "job", "JOBHASH",
                    stage_hashes={"segment": "SEGHASH", "caption": "CAPHASH"})
    sink = ShardPartSink(jf, "segment", "segment-00000", "c1", tmp_path / "loc")
    sink.add(records=[seg("ds--a", "whatever")])
    sink.flush()
    shard = Shard("caption-00000", "caption", ("ds--a", "ds--b"), 1.0,
                  depends_on=("segment-00000",))
    # A record of another config under the same stage path is still refused.
    P.write_part(jf, "segment", "segment-00000", "c2", 0, [seg("ds--b", "FOREIGN")])
    loaded = UpstreamSegments.load(jf, shard)
    assert set(loaded) == {"ds--a"}
    assert loaded["ds--a"].provenance.config_hash == "SEGHASH"
