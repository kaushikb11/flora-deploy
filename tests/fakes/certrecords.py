"""Records that carry the certification binding (``provenance.cert_key`` etc.), for the
``validate/certify.py`` tests. ``prov`` stamps what a worker must stamp."""

from __future__ import annotations

import dataclasses

from flora_deploy import LOGIC_VERSION
from flora_deploy.config.presets import get_preset
from flora_deploy.records import DenseReward, Provenance, Segment, StageRecord, TerminalReward
from flora_deploy.validate import certkey

FP = {"image_fingerprint": lambda name: f"fp-{name}"}
CONTENT = {"captioner_bf16": "c" * 64, "tempseg_merged": "d" * 64}
ORACLE_HASH = "a" * 64
CAND_HASH = "b" * 64
GPU_NAME = {"L40S": "NVIDIA L40S", "H100!": "NVIDIA H100 80GB HBM3", "H100": "NVIDIA H100",
            "H200": "NVIDIA H200"}
# The pool GPU of each preset's kind, as the presets define it.
ORACLE_GPU = "H200"
BAL_GPU = {"segment": "L40S", "caption": "H100!", "reward.terminal": "H100!",
           "reward.dense": "H100!"}


VLLM_STAGES = ("segment", "caption")


def modes_for(preset: str, stage: str) -> dict:
    kind = get_preset(preset).stage_kinds[stage]
    if kind == "segcap":
        return {"determinism": "nvidia"}
    if kind == "reward.terminal":
        return {"terminal_mode": "terminal"}
    return {}


def key(preset: str, stage: str, gpu: str, *, fp=FP, content=CONTENT, modes=None, **kw) -> dict:
    kind = get_preset(preset).stage_kinds[stage]
    return certkey.key_document(preset, kind, gpu,
                                modes=modes_for(preset, stage) if modes is None else modes,
                                derived_content=content, **(fp or {}), **kw)


def default_gpu(preset: str, stage: str) -> str:
    return ORACLE_GPU if preset == "exact" and not stage.startswith("reward.") else (
        BAL_GPU[stage])


def prov(preset: str, stage: str, *, gpu: str | None = None, embodiment: str | None = "yam",
         config_hash: str | None = None, cert_key: dict | None = None, fp=FP, modes=None,
         engine: str | None = None, certificate: str | None = None, **kw) -> Provenance:
    g = gpu or default_gpu(preset, stage)
    doc = cert_key if cert_key is not None else key(preset, stage, g, fp=fp, modes=modes)
    if preset != "exact" and stage in VLLM_STAGES:  # what a balanced vLLM worker stamps
        kw.setdefault("compile_cache", "cold")
        kw.setdefault("engine_preemptions", 0)
    vllm = doc.get("vllm_engine") if isinstance(doc, dict) else None
    models = {k: v["revision"] for k, v in (doc.get("nvidia_pins") or {}).items()}
    return Provenance(
        models=kw.pop("models", models),
        engine=engine or (vllm["label"] if vllm else "nvidia-engine"),
        preset=preset,
        config_hash=config_hash or (ORACLE_HASH if preset == "exact" else CAND_HASH),
        logic_version=LOGIC_VERSION, framework=kw.pop("framework", "flora-deploy test"),
        gpu=GPU_NAME.get(g, g), certificate=certificate, embodiment=embodiment, cert_key=doc,
        **kw)


def rec(uid: str, stage: str, preset: str, *, segments=(), terminal=None, dense=None,
        audit_ref=None, **kw) -> StageRecord:
    return StageRecord(uid=uid, dataset="ds", episode_id=uid, stage=stage,
                       provenance=prov(preset, stage, **kw), certified=preset == "exact",
                       uncertified_reason=None if preset == "exact" else "uncertified_stack",
                       segments=tuple(segments), terminal=terminal, dense=dense,
                       audit_ref=audit_ref)


def segs(bounds, captions=None) -> list[Segment]:
    return [Segment(event_id=f"event_{i:04d}", start_s=s, end_s=e,
                    caption=(captions[i] if captions else None), segmentation_provenance="fine")
            for i, (s, e) in enumerate(bounds)]


BOUNDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 9.0)]


def cold_witness(records, n: int = 6) -> list[StageRecord]:
    """The cold witness: forced-cold copies (``compile_cache`` ``cold:forced``) of the first
    ``n`` candidate records. ``segment``/``caption`` certify only with one (``no_witness``)."""
    return [dataclasses.replace(r, provenance=dataclasses.replace(
        r.provenance, compile_cache="cold:forced")) for r in list(records)[:n]]


def segment_set(preset: str, n: int, shift: float = 0.0, *, start: int = 0, **kw):
    return [rec(f"u{i}", "segment", preset,
                segments=segs([(s + shift, e + shift) for s, e in BOUNDS]), **kw)
            for i in range(start, start + n)]


def terminal(score: int, es: float) -> TerminalReward:
    return TerminalReward(score=score, expected_score=es, probabilities=(0, 0, 0, 0.5, 0.5))


def terminal_set(preset: str, n: int, score: int = 4, es: float = 4.2, **kw):
    return [rec(f"u{i}", "reward.terminal", preset, terminal=terminal(score, es), **kw)
            for i in range(n)]


def dense_set(preset: str, n: int, steps=(3, 3, 3), **kw):
    return [rec(f"u{i}", "reward.dense", preset,
                dense=DenseReward(segments=(), per_step=tuple(steps)), **kw) for i in range(n)]


def caption_set(preset: str, n: int, captions=("a", "b", "c"), **kw):
    return [rec(f"u{i}", "caption", preset, segments=segs(BOUNDS, list(captions)), **kw)
            for i in range(n)]


STAGE_SETS = {"segment": segment_set, "caption": caption_set, "reward.terminal": terminal_set,
              "reward.dense": dense_set}

def lab_hashes(kind: str, gpu: str) -> dict[str, str]:
    """The engine-settings hashes a lab run on ``gpu`` records for the current limits."""
    from flora_deploy.engines.offline_llm import EngineSpec

    return {"baseline": certkey.engine_settings_hash(EngineSpec.for_gpu(kind, "", gpu).hashed()),
            "plugin": certkey.engine_settings_hash(
                EngineSpec.for_gpu(kind, "", gpu, dtbias_fp32=True).hashed())}


L0_PASS = {"kind": "tempseg", "baseline": {"l0": [
    {"key": "k", "ids_equal": True, "render_ids_equal": True, "grid_equal": True}] * 6},
    "gate_a": {"requests": 6, "greedy_identical": 6, "positions": 10, "max_abs_dlogit": 0.1,
               "mean_abs_dlogit": 0.01, "first_divergence": []},
    "engine_hashes": lab_hashes("tempseg", "L40S")}


def lab_for(stage: str) -> dict:
    """A passing L0 / Gate A result for ``stage``'s model kind, under the current limits."""
    from flora_deploy.validate.certify import L0_STAGES

    kind = L0_STAGES.get(stage, "x")
    hashes = lab_hashes(kind, BAL_GPU[stage]) if kind in ("tempseg", "captioner") else {}
    return {**L0_PASS, "kind": kind, "engine_hashes": hashes}


# ---------------------------------------------------------------------------------------------
# Stage-isolated caption evidence (certify v4)


def row_digest(uid: str, seg: Segment) -> str:
    """A stand-in for NVIDIA's input digest of a Captioner call: a function of the episode and
    the row's bounds, so two runs on the same segments agree and a moved boundary does not."""
    import hashlib

    return hashlib.sha256(f"{uid}|{seg.start_s:.6f}|{seg.end_s:.6f}".encode()).hexdigest()


def digests_of(records) -> dict[str, list[str | None]]:
    """``{uid: [digest | None per row]}``: a digest for every ``complete`` row."""
    return {r.uid: [row_digest(r.uid, s) if s.status == "complete" else None
                    for s in r.segments] for r in records}


def oracle_segment_records(exact) -> list[StageRecord]:
    """The oracle's ``segment`` records behind its caption records (same stage hash)."""
    return [dataclasses.replace(r, stage="segment", segments=tuple(
        dataclasses.replace(s, caption=None, status="complete") for s in r.segments))
        for r in exact]


def isolated(exact, balanced, witness=None, noise=None, *, directory=None):
    """Make caption ``balanced`` / ``witness`` records stage-isolated (``segments_from`` of a
    staged oracle segment file) and return ``(balanced, witness, certify_stage kwargs)`` with
    the oracle segment file and every side's digests."""
    import hashlib
    import tempfile
    from pathlib import Path

    exact = list(exact)
    d = Path(directory or tempfile.mkdtemp(prefix="oracle-segments-"))
    path = d / "oracle-segments.jsonl"
    data = "".join(r.to_json() + "\n" for r in oracle_segment_records(exact)).encode()
    path.write_bytes(data)
    sf = {"sha256": hashlib.sha256(data).hexdigest(),
          "oracle_stage_hash": exact[0].provenance.config_hash}

    def stamp(recs):
        return None if recs is None else [dataclasses.replace(r, provenance=dataclasses.replace(
            r.provenance, segments_from=dict(sf))) for r in recs]

    balanced, witness = stamp(list(balanced)), stamp(None if witness is None else list(witness))
    digests = {"exact": digests_of(exact), "balanced": digests_of(balanced)}
    if witness is not None:
        digests["witness"] = digests_of(witness)
    if noise is not None:
        digests["noise"] = digests_of(noise)
    return balanced, witness, {"digests": digests, "oracle_segments": path}
