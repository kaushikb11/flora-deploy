"""Review 13 S3-1: the lab records the engine-settings hash of each engine it built, so its L0 /
Gate A evidence is tied to the engine settings it measured (``certkey.engine_settings_hash``,
the hash ``validate.certify.l0_summary`` and ``scripts/wave_h.py`` compare)."""

from __future__ import annotations

from types import SimpleNamespace

from flora_deploy.engines.offline_llm import EngineSpec
from flora_deploy.validate import certkey


def test_run_kind_records_each_engines_settings_hash(monkeypatch, tmp_path):
    from flora_deploy.modal_app import balanced_lab as lab
    from flora_deploy.modal_app import warm

    built = []

    def child(args, env):
        spec = EngineSpec.for_gpu(args["kind"], "/m", args["gpu"], dtbias_fp32=args["plugin"],
                                  logprobs_mode="raw_logits", max_logprobs=20)
        built.append(spec)
        out = {"kind": args["kind"], "plugin": args["plugin"], "gate_a": [],
               "engine_hash": lab.engine_hash(spec)}
        return {**out, "episodes": []} if not args["plugin"] else out

    monkeypatch.setattr(lab, "_run_child", child)
    monkeypatch.setattr(lab, "cache", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(lab, "cache_env", lambda kind, plugin: {})
    monkeypatch.setattr(lab, "compile_cache_state", lambda kind, plugin: {})
    monkeypatch.setattr(lab, "gate_a_compare", lambda a, b: {"requests": 1})
    monkeypatch.setattr(warm, "start_prefetch",
                        lambda *a, **k: SimpleNamespace(wait=lambda t: None))
    monkeypatch.setattr(lab, "Path", lambda p: tmp_path / str(p).strip("/"))
    res = lab.run_kind("tempseg", "wave-h-r", gate_a=True)
    want = {"baseline": certkey.engine_settings_hash(
                EngineSpec.for_gpu("tempseg", "", "L40S").hashed()),
            "plugin": certkey.engine_settings_hash(
                EngineSpec.for_gpu("tempseg", "", "L40S", dtbias_fp32=True).hashed())}
    assert res["engine_hashes"] == want
    assert [s.dtbias_fp32 for s in built] == [False, True]
