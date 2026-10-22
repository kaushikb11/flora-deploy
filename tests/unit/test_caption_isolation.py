"""Stage-isolated caption certification, the package side (caption pre-registration of
2026-10-23): ``pipeline.caption_segments``, its hash and key, the planner, the pinned segment
provider, ``provenance.segments_from``, the restated NVIDIA input digest, the worker env and
the cache-block facts in the shard audit row."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flora_deploy.config.hashing import config_hash, kind_modes
from flora_deploy.config.schema import ConfigError, parse_config, to_run_settings
from flora_deploy.jobs import attempts as A
from flora_deploy.jobs.stores.local import LocalStore
from flora_deploy.jobs.types import EpisodeRef, Shard
from flora_deploy.records import Provenance, Segment, StageRecord
from flora_deploy.workers_impl import caption as W
from flora_deploy.workers_impl.segment import ShardContext

SHA = "41d60b449629b2181ff5b735d31c2a2cf8b3cad8"
ORACLE = "a" * 64
RECEIPTS = {"captioner_bf16": "c" * 64, "tempseg_merged": "t" * 64}
SINK = "volume://flora-wave-h/run1"


def e2e() -> dict:
    return {"project": "demo", "license_ack": "research-only", "allow_uncertified": True,
            "modal": {"max_gpus": 4, "budget_usd": 20}, "preset": "balanced",
            "input": {"format": "lerobot_v3", "repo": "org/ds", "revision": SHA},
            "pipeline": {"stages": ["segment", "caption"], "embodiment": "human"},
            "output": {"sink": SINK}}


def iso(sha="b" * 64, **over) -> dict:
    d = copy.deepcopy(e2e())
    d["pipeline"] = {"stages": ["caption"], "embodiment": "human", "caption_segments": {
        "records": f"{SINK}/oracle-segments/{ORACLE}.jsonl", "sha256": sha,
        "oracle_stage_hash": ORACLE, **over}}
    return d


# ---------------------------------------------------------------------------------------------
# config


def test_caption_only_needs_pinned_segments():
    bare = copy.deepcopy(e2e())
    bare["pipeline"]["stages"] = ["caption"]
    with pytest.raises(ConfigError, match="caption_segments"):
        parse_config(bare)
    cfg = parse_config(iso())
    assert cfg.pipeline.stages == ("caption",)
    assert cfg.pipeline.caption_segments.as_hashed() == {"sha256": "b" * 64,
                                                         "oracle_stage_hash": ORACLE}


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(preset="exact"), "balanced caption stage only"),
    (lambda d: d["pipeline"].update(stages=["segment", "caption"]), "stages: \\[caption\\]"),
    (lambda d: d["pipeline"]["caption_segments"].update(sha256="XYZ"), "64 lowercase hex"),
    (lambda d: d["pipeline"]["caption_segments"].update(oracle_stage_hash="A" * 64),
     "64 lowercase hex"),
    (lambda d: d["pipeline"]["caption_segments"].update(records="volume://other/x.jsonl"),
     "under output.sink"),
])
def test_caption_segments_is_refused_outside_its_use(edit, match):
    d = iso()
    edit(d)
    with pytest.raises(ConfigError, match=match):
        parse_config(d)


def test_isolated_job_has_its_own_stage_hash_but_the_production_key():
    from flora_deploy.validate.certkey import key_document

    j1, j2 = parse_config(e2e()), parse_config(iso())
    h1, h2 = (config_hash(c, derived_receipts=RECEIPTS) for c in (j1, j2))
    assert h1.stage_hashes["caption"] != h2.stage_hashes["caption"]
    assert "segment" not in h2.stage_hashes
    assert h2.document["shard_kinds"]["caption"]["caption_segments"] == {
        "sha256": "b" * 64, "oracle_stage_hash": ORACLE}
    assert h2.document["shard_kinds"]["caption"]["depends_on"] == []
    other = config_hash(parse_config(iso("d" * 64)), derived_receipts=RECEIPTS)
    assert other.stage_hashes["caption"] != h2.stage_hashes["caption"]
    assert "caption_segments" not in h1.document["shard_kinds"]["caption"]
    # the certificate key: the same modes, so the same key document
    assert kind_modes(j1, "caption") == kind_modes(j2, "caption")
    fp = {"image_fingerprint": lambda name: f"fp-{name}"}
    doc = [key_document("balanced", "caption", c.pools()["caption"].gpu,
                        modes=kind_modes(c, "caption"), derived_content={"captioner_bf16":
                                                                          "c" * 64}, **fp)
           for c in (j1, j2)]
    assert doc[0] == doc[1]


def test_planner_gives_pinned_caption_shards_no_upstream():
    from tests.unit.test_jobs_manifest import make_index

    from flora_deploy.jobs.manifest import build_manifest

    rs = to_run_settings(parse_config(iso()), "job", derived_receipts=RECEIPTS)
    assert rs.caption_segments == {"sha256": "b" * 64, "oracle_stage_hash": ORACLE}
    idx = [make_index("org/ds", f"episode_{i:06d}", "/d/f0.mp4", 10.0) for i in range(3)]
    m = build_manifest(idx, rs, created_utc="t")
    assert {s.kind for s in m.shards} == {"caption"}
    assert all(s.depends_on == () for s in m.shards)
    with pytest.raises(ValueError, match="segment pool"):
        build_manifest(idx, replace(rs, caption_segments=None), created_utc="t")


# ---------------------------------------------------------------------------------------------
# records


def _prov(**kw) -> Provenance:
    return Provenance(models={}, engine="e", preset="exact", config_hash=ORACLE,
                      logic_version="l", framework="f", **kw)


def test_segments_from_is_absent_from_json_unless_set():
    r = StageRecord(uid="u", dataset="d", episode_id="u", stage="caption", provenance=_prov(),
                    certified=True)
    assert "segments_from" not in r.to_json()
    sf = {"sha256": "b" * 64, "oracle_stage_hash": ORACLE}
    r2 = replace(r, provenance=replace(r.provenance, segments_from=sf))
    assert StageRecord.from_json(r2.to_json()).provenance.segments_from == sf
    assert StageRecord.from_json(r.to_json()) == r


# ---------------------------------------------------------------------------------------------
# the pinned segment provider


def oracle_segment(uid: str, *, preset="exact", stage="segment", cfg=ORACLE) -> StageRecord:
    segs = (Segment("event_0000", 0.0, 2.0, segmentation_provenance="fine"),
            Segment("event_0001", 2.0, 5.0, segmentation_provenance="coarse",
                    fine_event_id="f1"))
    return StageRecord(uid=uid, dataset="ds", episode_id=uid, stage=stage,
                       provenance=replace(_prov(), preset=preset, config_hash=cfg),
                       certified=True, segments=segs)


def staged(tmp_path, records) -> tuple[LocalStore, W.PinnedSpec]:
    store = LocalStore(tmp_path / "s")
    data = "".join(r.to_json() + "\n" for r in records).encode()
    store.put_bytes("oracle-segments/x.jsonl", data)
    return store, W.PinnedSpec("oracle-segments/x.jsonl", hashlib.sha256(data).hexdigest(),
                               ORACLE)


def ctx_for(store, uids):
    jf = A.JobFiles(store, "job", "cfg")
    return ShardContext(jf, Shard("caption-0", "caption", tuple(uids), 1.0), None, "tx")  # type: ignore[arg-type]


def ref(uid):
    return EpisodeRef(uid=uid, dataset="ds", episode_id=uid, pool="human", duration_s=9.0,
                      oracle_eligible=True, envelope_ok=True, index_ref="idx")


def test_pinned_segments_read_the_oracle_records_and_rebuild_rows(tmp_path):
    store, spec = staged(tmp_path, [oracle_segment("ds--a"), oracle_segment("ds--b")])
    prov = W.PinnedSegments(spec)
    rec = prov(ref("ds--a"), ctx_for(store, ["ds--a", "ds--b"]))
    assert W.pinned_rows(rec) == [
        {"start_s": 0.0, "end_s": 2.0, "metadata": {"segmentation_provenance": "fine"}},
        {"start_s": 2.0, "end_s": 5.0, "metadata": {"segmentation_provenance": "coarse",
                                                    "fine_event_id": "f1"}}]
    assert prov.segments_from == {"sha256": spec.sha256, "oracle_stage_hash": ORACLE}
    with pytest.raises(W.MissingUpstream):
        prov(ref("ds--z"), ctx_for(store, ["ds--z"]))


@pytest.mark.parametrize("bad", [
    {"sha": "0" * 64},
    {"rec": dict(preset="balanced")},
    {"rec": dict(stage="caption")},
    {"rec": dict(cfg="f" * 64)},
])
def test_pinned_segments_refuse_an_altered_or_foreign_file(tmp_path, bad):
    store, spec = staged(tmp_path, [oracle_segment("ds--a", **bad.get("rec", {}))])
    if "sha" in bad:
        spec = replace(spec, sha256=bad["sha"])
    with pytest.raises(W.PinnedSegmentsError):
        W.PinnedSegments(spec)(ref("ds--a"), ctx_for(store, ["ds--a"]))


def test_provider_follows_the_worker_env():
    assert isinstance(W.segments_for({}), W.UpstreamSegments)
    env = {W.CAPTION_SEGMENTS_ENV: json.dumps({"key": "k/x.jsonl", "sha256": "b" * 64,
                                               "oracle_stage_hash": ORACLE})}
    p = W.segments_for(env)
    assert isinstance(p, W.PinnedSegments) and p.spec.key == "k/x.jsonl"
    with pytest.raises(ValueError):
        W.PinnedSpec("../x", "b" * 64, ORACLE)


def test_factory_hands_the_store_key_to_caption_workers():
    from flora_deploy.modal_app.factory import DeployError, caption_segments_env, parse_store

    cfg = parse_config(iso())
    env = caption_segments_env(cfg, parse_store(SINK))
    assert json.loads(env[W.CAPTION_SEGMENTS_ENV]) == {
        "key": f"oracle-segments/{ORACLE}.jsonl", "sha256": "b" * 64,
        "oracle_stage_hash": ORACLE}
    assert caption_segments_env(parse_config(e2e()), parse_store(SINK)) == {}
    with pytest.raises(DeployError, match="not a file in the job store"):
        caption_segments_env(cfg, parse_store("volume://flora-wave-h/other-run"))


# ---------------------------------------------------------------------------------------------
# the restated digest and the caption audit


class _Proc:
    """A processor returning fixed tensors (and metadata NVIDIA pops)."""

    def __call__(self, **kw):
        import torch

        return {"input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values_videos": torch.ones((2, 4), dtype=torch.bfloat16),
                "video_metadata": [object()], "note": "not a tensor"}


def _captured():
    import numpy as np

    from flora_deploy.nv.capture import CapturedRequest

    md = {"total_num_frames": 1, "fps": 8.0, "frames_indices": [0], "duration": 0.125,
          "do_sample_frames": False}
    return CapturedRequest(prompt="p", videos=[(np.zeros((1, 2, 2, 3), np.uint8), md)],
                           call_kwargs={"return_tensors": "pt", "return_metadata": True})


def test_nvidia_input_digest_restates_the_logged_rule():
    torch = pytest.importorskip("torch")  # in the worker images; not in every dev venv

    from flora_deploy.stages.segment import nvidia_input_digest

    out = _Proc()()
    tensors = {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": hashlib.sha256(
        v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for k, v in out.items() if torch.is_tensor(v)}
    want = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
                          ).hexdigest()
    assert nvidia_input_digest(_Proc(), _captured()) == want


def test_a_digest_failure_is_evidence_lost_never_an_episode_failure():
    from flora_deploy.stages.segment import InputDigests, TappedCall

    d = InputDigests(processor=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    d.add(TappedCall(_captured(), 512, "s0"))
    assert d.by_sentinel == {"s0": None} and len(d.errors) == 1  # boom (or no torch here)


def test_shard_audit_row_carries_the_cache_block_facts():
    from flora_deploy.workers_impl.segment import BalancedShardRunner

    blocks = {"num_gpu_blocks": 313, "block_size": 784, "blocks_per_seq": 26, "needed": 286,
              "max_num_seqs": 11, "worst_case_tokens": 17920, "gpu_memory_utilization": 0.9,
              "groups": [], "checked": True}
    r = BalancedShardRunner.__new__(BalancedShardRunner)
    r.engine = SimpleNamespace(cache_blocks=blocks)
    r.job = SimpleNamespace(stage="caption")
    r.policy = SimpleNamespace(as_json=lambda: {})
    r.settings = SimpleNamespace(phased=True)
    r.last_stats, r.last_gates, r.sampler, r.compile_cache = {}, {}, None, None
    row = r._shard_audit(Shard("caption-0", "caption", ("u",), 1.0), 1, 0.0)
    assert row["cache_blocks"] == blocks
