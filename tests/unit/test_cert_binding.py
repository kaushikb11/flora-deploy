"""``workers_impl.binding``: the certification binding every worker stamps (embodiment, cert_key,
gpu), through the shared ``ShardPartSink``, so ``flora certify`` accepts exact segcap and reward
records (``validate.certify.missing_provenance``)."""

from __future__ import annotations

from types import SimpleNamespace

from flora_deploy.jobs import attempts as A
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.records import Provenance, StageRecord
from flora_deploy.validate.certify import missing_provenance
from flora_deploy.workers_impl import binding as B
from flora_deploy.workers_impl.reward import ShardPartSink
from tests.fakes import certrecords as CR

CFG = SimpleNamespace(pipeline=SimpleNamespace(embodiment="yam"))
ENV = {"FLORA_PRESET": "exact", "FLORA_DETERMINISM": "nvidia",
       "FLORA_TERMINAL_MODE": "terminal",
       **B.binding_env(CFG, {"segcap": "H200", "reward.terminal": "H100!"},
                       derived_content=CR.CONTENT)}


def _record(stage: str, gpu: str | None = None) -> StageRecord:
    prov = Provenance(models={}, engine="nvidia-engine", preset="exact", config_hash="own",
                      logic_version="t", framework="flora-deploy test", gpu=gpu)
    return StageRecord(uid="u0", dataset="ds", episode_id="u0", stage=stage, provenance=prov,
                       certified=True)


def test_binding_from_env_is_the_key_document():
    b = B.binding_from_env(["segcap", "reward.terminal"], ENV, gpu="NVIDIA H200", **CR.FP)
    assert b["segcap"].cert_key == CR.key("exact", "segment", "H200")
    assert b["reward.terminal"].cert_key == CR.key("exact", "reward.terminal", "H100!")
    assert {x.embodiment for x in b.values()} == {"yam"}
    assert b["segcap"].problem is None and b["segcap"].gpu == "NVIDIA H200"


def test_fail_closed_without_content_or_pool():
    env = {**ENV, B.ENV_DERIVED_CONTENT: "{}"}
    b = B.binding_from_env(["segcap", "reward.dense"], env, **CR.FP)
    assert b["segcap"].cert_key is None and "captioner_bf16" in b["segcap"].problem
    assert b["reward.dense"].cert_key is None and "no pool GPU" in b["reward.dense"].problem


def test_sink_stamps_the_binding(tmp_path):
    jf = A.JobFiles(LocalStore(tmp_path / "store"), "job", "c" * 64, stage_hashes={})
    b = B.binding_from_env(["segcap"], ENV, gpu="NVIDIA H200", **CR.FP)["segcap"]
    sink = ShardPartSink(jf, "segcap", "s1", "ctr", tmp_path / "local", binding=b)
    out = sink._stamp(_record("segment"))
    assert out.provenance.config_hash == "c" * 64
    assert out.provenance.embodiment == "yam" and out.provenance.cert_key == b.cert_key
    assert out.provenance.gpu == "NVIDIA H200"
    assert missing_provenance(out.provenance) == []
    # a record that names its own GPU keeps it; no binding: unchanged binding fields
    assert sink._stamp(_record("segment", gpu="NVIDIA H200 NVL")).provenance.gpu == \
        "NVIDIA H200 NVL"
    bare = ShardPartSink(jf, "segcap", "s1", "ctr", tmp_path / "local2")._stamp(_record("segment"))
    assert bare.provenance.cert_key is None and "cert_key" in missing_provenance(bare.provenance)
