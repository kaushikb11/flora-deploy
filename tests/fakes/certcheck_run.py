"""A fake certified run on disk for ``scripts/cert_check.py`` (``tests/unit/test_cert_check.py``).

:func:`build` writes what ``scripts/wave_h.py --plan certified`` and ``wave_h_finalize.py
--write`` leave behind, shaped like the real files:

- ``records/{exact,noise,balanced,witness}/human/{segment,caption}.jsonl``: ``flora_labels_v1``
  records with the binding provenance (the current ``cert_key``, stage hash, ``nvidia-smi``
  GPU name, embodiment, engine label, model revisions) and, for the balanced vLLM workers,
  ``compile_cache`` and ``engine_preemptions``. 12 oracle episodes (``n-`` noise subset of 6,
  ``x-`` the rest; uids ``cert-human--<episode id>``), the noise run on the 6, balanced (J1,
  end to end) on the 12, its cold witness (J1w, segment only) on the 6 (``cold:forced``);
- the stage-isolated caption evidence (generator v4): ``records/isolated/human/caption.jsonl``
  (J2: captions of the oracle's own segments, ``provenance.segments_from``),
  ``records/isolated_witness/human/caption.jsonl`` (J2w, forced cold), each caption side's
  ``digests.json`` (a stand-in input digest per captioned row), the staged oracle segment file
  ``work/oracle-segments/<oracle hash>.jsonl`` and ``families.json``;
- ``lab/balanced-{tempseg,captioner}.json``: L0 rows, Gate A and the engine-settings hashes;
- ``certs/<id>.json`` and ``certs/freeze/<id>.json``: written by the **real** generator
  (``validate.certify.certify_stage``, ``write_certificate``, ``certfreeze.write_freeze``), so a
  good run is also a cross-check of the generator against the independent checker;
- ``state.json`` (the runbook's state format) and ``report.md`` (the finalize report's spend
  section and certificate table);
- ``nv/<pin snapshot>/``: stand-in "NVIDIA" text for the shingle check (our own words).

Every text here is ours; nothing is NVIDIA's.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flora_deploy import LOGIC_VERSION
from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.engines.offline_llm import EngineSpec
from flora_deploy.layout import snapshot_dirname
from flora_deploy.records import Provenance, StageRecord
from flora_deploy.validate import certfreeze as F
from flora_deploy.validate import certify as CY
from flora_deploy.validate import certkey as K
from flora_deploy.workers_impl.certstamp import pinned_content

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import _runbook as rb  # noqa: E402
import cert_check as cc  # noqa: E402

STAMP = "2026-10-23T00:00:00Z"
DATA = "cert-human"
ORACLE_HASH, CAND_HASH = "a" * 64, "b" * 64
ISO_HASH = "c" * 64  # J2's caption stage hash (caption_segments is in it; the key is J1's)
GPU_NAME = {"L40S": "NVIDIA L40S", "H100!": "NVIDIA H100 80GB HBM3", "H200": "NVIDIA H200"}
POOL = {"segment": "L40S", "caption": "H100!"}
WARM = "warm:92c7ffabd3:0123456789abcdef"
NOISE_IDS = tuple(f"n-egodex_task_{i}-episode_{i:06d}" for i in range(6))
REST_IDS = tuple(f"x-wgo-homer_{40 + i}" for i in range(6))
EPISODES = NOISE_IDS + REST_IDS
BOUNDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 9.0)]
CAPTIONS = ("reach toward the blue mug", "lift the blue mug slowly", "set the mug on the tray")
FAKE_NV = ("the quick lantern keeper walks past seven quiet harbors while the copper gulls "
           "argue about tides and the baker counts every loaf twice before dawn breaks over "
           "the hills where nobody remembers the old road or the bridge that once crossed it")
VERSION = CY.CERTIFY_VERSION
SUPPORTS_WITNESS = "witness" in inspect.signature(CY.certify_stage).parameters


def uid(eid: str) -> str:
    return f"{DATA}--{eid}"


def oracle_key(stage: str) -> dict[str, Any]:
    return K.key_document("exact", "segcap", "H200", modes={"determinism": "nvidia"},
                          derived_content=pinned_content("exact", "segcap"))


def balanced_key(stage: str) -> dict[str, Any]:
    return K.key_document("balanced", stage, POOL[stage], modes={},
                          derived_content=pinned_content("balanced", stage))


_KEYS: dict[tuple[str, str], dict[str, Any]] = {}


def key(preset: str, stage: str) -> dict[str, Any]:
    if (preset, stage) not in _KEYS:
        _KEYS[(preset, stage)] = (oracle_key if preset == "exact" else balanced_key)(stage)
    return copy.deepcopy(_KEYS[(preset, stage)])


def provenance(preset: str, stage: str, eid: str, *, cache: str | None = None,
               preemptions: int | None = 0) -> dict[str, Any]:
    doc = key(preset, stage)
    vllm = doc.get("vllm_engine")
    gpu = "H200" if preset == "exact" else POOL[stage]
    p = {"models": {k: v["revision"] for k, v in doc["nvidia_pins"].items()},
         "engine": vllm["label"] if vllm else "hf-ref-5.8.1", "preset": preset,
         "config_hash": ORACLE_HASH if preset == "exact" else CAND_HASH,
         "logic_version": LOGIC_VERSION, "framework": "flora-deploy 0.0.1",
         "gpu": GPU_NAME[gpu], "certificate": None,
         "source": {"main": {"uri": f"{eid}.mp4", "range_s": [None, None]}},
         "engine_hash": "e" * 64, "embodiment": "human", "cert_key": doc}
    if preset != "exact":
        p["compile_cache"] = cache or WARM
        p["engine_preemptions"] = preemptions
    return p


def segment(i: int, s: float, e: float, caption: str | None, *, nv: bool) -> dict[str, Any]:
    return {"event_id": f"event_{i:04d}", "start_s": s, "end_s": e, "status": "complete",
            "caption": caption, "fine_event_id": None, "segmentation_provenance": "fine",
            "degradation": {},
            "nv_metadata": {"segment_topology": "concurrent_intervals",
                            "window_profile_id": "human_16s_4hz"} if nv else {}}


def record(preset: str, stage: str, eid: str, bounds, captions=None, **kw) -> dict[str, Any]:
    segs = [segment(i, s, e, (captions[i] if captions else None) if stage == "caption" else None,
                    nv=stage == "segment") for i, (s, e) in enumerate(bounds)]
    return {"schema": "flora_labels_v1", "uid": uid(eid), "dataset": DATA, "episode_id": eid,
            "stage": stage, "provenance": provenance(preset, stage, eid, **kw),
            "certified": preset == "exact",
            "uncertified_reason": None if preset == "exact" else "uncertified_stack",
            "segments": segs, "terminal": None, "dense": None, "rubric": None,
            "audit_ref": None}


@dataclass
class Options:
    """What :func:`build` varies. Defaults: a run that certifies under the current rule."""

    episodes: tuple[str, ...] = EPISODES
    noise: tuple[str, ...] = NOISE_IDS
    # balanced episodes (outside the noise subset) with one extra interval: 1 in 12 passes
    flips: int = 1
    shift: float = 0.05  # balanced boundary shift on every other episode (seconds)
    caption_typo: bool = True  # one balanced caption differs by one character
    witness: bool = True
    witness_cache: str = cc.FORCED_COLD
    witness_flips: int = 0  # the first noise episodes' witness records split one subtask
    witness_drift: str | None = None  # J1w (segment): "boundary" (+0.2 s) or "nan"
    iso_witness_drift: bool = False  # every J2w caption differs (caption witness fails)
    preemptions: int | None = 0  # on the first balanced record (J1 and J2)
    lab_hash: str | None = None  # None: the current engine-settings hash
    gate_requests: int = 6


def balanced_bounds(i: int, o: Options):
    b = [(s + (o.shift if i % 2 else 0.0), e + (o.shift if i % 2 else 0.0)) for s, e in BOUNDS]
    if i >= len(o.episodes) - o.flips:  # the last episodes: not in the noise subset
        b = b[:2] + [(5.0, 7.0), (7.0, 9.0)]
    return b


def balanced_captions(i: int, o: Options, n: int) -> list[str]:
    caps = list(CAPTIONS) + ["move the tray aside"] * (n - len(CAPTIONS))
    if o.caption_typo and i == 3:
        caps[1] = caps[1].replace("slowly", "slowlv")
    return caps[:n]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def lab_doc(kind: str, gpu: str, o: Options) -> dict[str, Any]:
    spec = EngineSpec.for_gpu(kind, "", gpu)
    hashed = spec.hashed()
    rows = [{"key": f"r{i}", "ids_equal": True, "render_ids_equal": True, "grid_equal": True}
            for i in range(6)]
    lab_kwargs = {**hashed["engine_kwargs"], "logprobs_mode": "raw_logits", "max_logprobs": 20}
    return {"run": "wave-h-test", "kind": kind,
            "baseline": {"kind": kind, "plugin": False, "gpu": GPU_NAME[gpu],
                         "spec": {**hashed, "gpu": gpu, "engine_kwargs": lab_kwargs}, "l0": rows},
            "gate_a": {"requests": o.gate_requests, "greedy_identical": o.gate_requests,
                       "positions": 40, "max_abs_dlogit": 0.125, "mean_abs_dlogit": 0.018,
                       "first_divergence": []},
            "engine_hashes": {"baseline": o.lab_hash or cc.engine_settings_hash(hashed),
                              "plugin": "f" * 64}}


def _stage_record(d: dict[str, Any]) -> StageRecord:
    """A record dict as the current ``records.StageRecord`` takes it (unknown provenance
    fields, e.g. ``compile_cache`` before the field exists, are dropped)."""
    names = {f.name for f in dataclasses.fields(Provenance)}
    d = copy.deepcopy(d)
    d["provenance"] = {k: v for k, v in d["provenance"].items() if k in names}
    return StageRecord.from_dict(d)


@dataclass
class Fake:
    root: Path
    inp: cc.Inputs
    certs: dict[str, dict[str, Any]]

    @property
    def cert_dir(self) -> Path:
        return self.inp.cert_dir

    def records(self, side: str, stage: str) -> Path:
        return self.inp.records / side / "human" / f"{stage}.jsonl"

    def edit_records(self, side: str, stage: str, fn: Callable[[list[dict]], None]) -> None:
        rows = read_jsonl(self.records(side, stage))
        fn(rows)
        write_jsonl(self.records(side, stage), rows)

    def digests(self, side: str) -> Path:
        """The digest JSON of a ``flora certify --digests`` side."""
        return self.inp.records / cc.DIGEST_DIRS[side] / "human" / cc.DIGESTS_FILE

    def edit_digests(self, side: str, fn: Callable[[dict], None]) -> None:
        d = json.loads(self.digests(side).read_text(encoding="utf-8"))
        fn(d)
        self.digests(side).write_text(json.dumps(d, sort_keys=True) + "\n", encoding="utf-8")

    @property
    def oracle_segments(self) -> Path:
        return self.root / "work" / "oracle-segments" / f"{ORACLE_HASH}.jsonl"

    def cert_path(self, stage: str) -> Path:
        return self.cert_dir / f"{self.certs[stage]['id']}.json"

    def edit_cert(self, stage: str, fn: Callable[[dict], None], *, readdress: bool = True,
                  move_lock: bool = True) -> dict[str, Any]:
        """Edit a certificate body; re-address it (a new content id, as a tampered but
        well-formed file would be) and move its freeze lock along."""
        old = self.certs[stage]
        body = copy.deepcopy(old)
        fn(body)
        if readdress:
            body["id"] = K.certificate_id(body)
        self.cert_path(stage).unlink()
        lock = F.freeze_path(self.cert_dir, old["id"])
        p = self.cert_dir / f"{body['id']}.json"
        p.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        if move_lock and lock.exists():
            doc = json.loads(lock.read_text(encoding="utf-8"))
            doc["certificate"] = body["id"]
            lock.unlink()
            F.freeze_path(self.cert_dir, body["id"]).write_text(
                json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.certs[stage] = body
        return body

    def lock_path(self, stage: str) -> Path:
        return F.freeze_path(self.cert_dir, self.certs[stage]["id"])

    def clone(self, dst: Path) -> Fake:
        """A copy of this run under ``dst`` (to mutate without rebuilding)."""
        import shutil

        shutil.copytree(self.root, dst, dirs_exist_ok=True)
        state = json.loads((dst / "state.json").read_text(encoding="utf-8"))
        iso = state["steps"].get("caption_iso", {}).get("result", {})
        for r in [iso.get("oracle_segments") or {},
                  *((j.get("oracle_segments") or {}) for j in (iso.get("jobs") or {}).values())]:
            if r.get("path"):
                r["path"] = str(dst / Path(r["path"]).relative_to(self.root))
        (dst / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
        inp = dataclasses.replace(self.inp, run_root=dst, cert_dir=dst / "certs",
                                  lab_dir=dst / "lab", report=dst / "report.md",
                                  records=dst / "records", state=dst / "state.json",
                                  nv_root=dst / "nv",
                                  families=dst / "families.json" if self.inp.families else None)
        return Fake(dst, inp, copy.deepcopy(self.certs))

    def run(self, **kw) -> list[cc.Row]:
        inp = dataclasses.replace(self.inp, **kw) if kw else self.inp
        return cc.check_all(inp)


def build(tmp: Path, opts: Options | None = None, *,
          drift_runner: Callable[[Path], tuple[int, str]] | None = None) -> Fake:
    o = opts or Options()
    recs = tmp / "records"
    sets: dict[str, dict[str, list[dict[str, Any]]]] = {s: {} for s in cc.SIDES}
    for stage in ("segment", "caption"):
        sets["exact"][stage] = [record("exact", stage, e, BOUNDS, CAPTIONS) for e in o.episodes]
        sets["noise"][stage] = [record("exact", stage, e, BOUNDS, CAPTIONS) for e in o.noise]
        bal = []
        for i, e in enumerate(o.episodes):
            b = balanced_bounds(i, o)
            bal.append(record("balanced", stage, e, b, balanced_captions(i, o, len(b)),
                              preemptions=o.preemptions if i == 0 else 0))
        sets["balanced"][stage] = bal
        if stage == "caption":  # J1w is segment only (v4: J2w is the caption witness)
            continue
        by_eid = {r["episode_id"]: r for r in bal}
        sets["witness"][stage] = []
        if o.witness:
            for j, e in enumerate(o.noise):
                r = copy.deepcopy(by_eid[e])
                if j < o.witness_flips or o.witness_drift:
                    i = o.episodes.index(e)
                    b = balanced_bounds(i, o)
                    if j < o.witness_flips:  # one subtask split in two
                        b = b[:2] + [(5.0, 7.0), (7.0, 9.0)]
                    if o.witness_drift == "boundary":
                        b = [(s + 0.2, t + 0.2) for s, t in b]
                    if o.witness_drift == "nan" and j == 0:  # one boundary not finite
                        b = [b[0], (b[1][0], float("nan")), *b[2:]]
                    r = record("balanced", stage, e, b, balanced_captions(i, o, len(b)))
                r["provenance"]["compile_cache"] = o.witness_cache
                r["provenance"]["engine_preemptions"] = 0
                sets["witness"][stage].append(r)
    for side in ("exact", "noise", "balanced", "witness"):
        for stage, rows in sets[side].items():
            if side != "witness" or o.witness:
                write_jsonl(recs / side / "human" / f"{stage}.jsonl", rows)

    # the stage-isolated caption evidence: the staged oracle segment file, J2 and J2w
    seg_file = tmp / "work" / "oracle-segments" / f"{ORACLE_HASH}.jsonl"
    seg_file.parent.mkdir(parents=True, exist_ok=True)
    seg_bytes = "".join(_stage_record(r).to_json() + "\n"
                        for r in sets["exact"]["segment"]).encode("utf-8")
    seg_file.write_bytes(seg_bytes)
    import hashlib

    segs = {"records": f"volume://flora-store/test/oracle-segments/{ORACLE_HASH}.jsonl",
            "sha256": hashlib.sha256(seg_bytes).hexdigest(), "oracle_stage_hash": ORACLE_HASH,
            "path": str(seg_file), "episodes": len(o.episodes), "oracle_job": "wave-h-test-x"}
    sf = {"sha256": segs["sha256"], "oracle_stage_hash": ORACLE_HASH}
    iso = []
    for i, e in enumerate(o.episodes):
        r = record("balanced", "caption", e, BOUNDS, balanced_captions(i, o, len(BOUNDS)),
                   preemptions=o.preemptions if i == 0 else 0)
        r["provenance"].update(config_hash=ISO_HASH, segments_from=dict(sf))
        iso.append(r)
    iso_wit = []
    if o.witness:
        by_eid = {r["episode_id"]: r for r in iso}
        for e in o.noise:
            r = copy.deepcopy(by_eid[e])
            if o.iso_witness_drift:
                for s in r["segments"]:
                    s["caption"] = "then " + s["caption"]
            r["provenance"]["compile_cache"] = o.witness_cache
            r["provenance"]["engine_preemptions"] = 0
            iso_wit.append(r)
        write_jsonl(recs / "isolated_witness" / "human" / "caption.jsonl", iso_wit)
    write_jsonl(recs / "isolated" / "human" / "caption.jsonl", iso)
    digests = {"exact": sets["exact"]["caption"], "noise": sets["noise"]["caption"],
               "balanced": iso, **({"witness": iso_wit} if o.witness else {})}
    for side, rows in digests.items():
        p = recs / cc.DIGEST_DIRS[side] / "human" / cc.DIGESTS_FILE
        p.write_text(json.dumps(digests_of(rows), sort_keys=True) + "\n", encoding="utf-8")
    families = {uid(e): ("n-egodex" if e.startswith("n-") else "x-wgo") for e in o.episodes}
    (tmp / "families.json").write_text(json.dumps(families, sort_keys=True), encoding="utf-8")

    labs = tmp / "lab"
    labs.mkdir(parents=True, exist_ok=True)
    for stage, kind in cc.LAB_KIND.items():
        (labs / f"balanced-{kind}.json").write_text(
            json.dumps(lab_doc(kind, POOL[stage], o), indent=1), encoding="utf-8")

    certs = tmp / "certs"
    written: dict[str, dict[str, Any]] = {}
    for stage in ("segment", "caption"):
        lab = json.loads((labs / f"balanced-{cc.LAB_KIND[stage]}.json").read_text())
        l0 = {"kind": lab["kind"], "baseline": {"gpu": POOL[stage], "l0": lab["baseline"]["l0"]},
              "gate_a": lab["gate_a"], "engine_hashes": lab["engine_hashes"]}
        kw: dict[str, Any] = {}
        cand = sets["balanced"][stage]
        if stage == "caption":  # v4: what wave_h_finalize.isolated_inputs passes
            cand = iso
            kw.update(digests={k: digests_of(v) for k, v in digests.items()},
                      oracle_segments=seg_file,
                      reported_e2e=[_stage_record(r) for r in sets["balanced"][stage]],
                      families=families)
            if o.witness:
                kw["witness"] = [_stage_record(r) for r in iso_wit]
        elif SUPPORTS_WITNESS and o.witness:
            kw["witness"] = [_stage_record(r) for r in sets["witness"][stage]]
        doc = key("balanced", stage)
        cert = CY.certify_stage(
            stage, exact=[_stage_record(r) for r in sets["exact"][stage]],
            balanced=[_stage_record(r) for r in cand],
            noise=[_stage_record(r) for r in sets["noise"][stage]], engine_spec=doc,
            gpu=POOL[stage], preset="balanced", kind=stage, l0=l0, generated_at=STAMP, **kw)
        CY.write_certificate(cert, certs)
        F.write_freeze(cert, doc, certs)
        written[stage] = cert

    state = state_doc(o, segs)
    (tmp / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    (tmp / "report.md").write_text(report_text(state, written), encoding="utf-8")

    nv = tmp / "nv"
    for k in cc.NV_PINS:
        d = nv / snapshot_dirname(PINS_BY_KEY[k]) / "batch_utils"
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompts.txt").write_text(f"{k}: {FAKE_NV}\n", encoding="utf-8")

    inp = cc.Inputs(run_root=tmp, cert_dir=certs, lab_dir=labs, report=tmp / "report.md",
                    nv_root=nv, drift_runner=drift_runner, families=tmp / "families.json")
    return Fake(tmp, inp, written)


def row_digest(u: str, seg: dict[str, Any]) -> str:
    """A stand-in for NVIDIA's input digest of a Captioner call: a function of the episode and
    the row's bounds (two runs on the same segments agree; a moved boundary does not)."""
    import hashlib

    return hashlib.sha256(f"{u}|{seg['start_s']:.6f}|{seg['end_s']:.6f}".encode()).hexdigest()


def digests_of(rows: list[dict[str, Any]]) -> dict[str, list[str | None]]:
    """``{uid: [digest | None per row]}``: a digest for every ``complete`` row."""
    return {r["uid"]: [row_digest(r["uid"], s) if s.get("status") == "complete" else None
                       for s in r["segments"]] for r in rows}


def attempt(tracked: float, *, budget: float = 1.0, status: str = "done",
            report_error: bool = False) -> dict[str, Any]:
    return {"started": 1792700000.0, "status": status, "finished": 1792700600.0,
            "heartbeat": 1792700590.0, "rate_usd_per_s": 0.002, "budget_usd": budget,
            "tracked_usd": tracked, "apps": ["flora-wave-h-test"], "report_error": report_error,
            "notes": []}


def state_doc(o: Options, segs: dict[str, Any]) -> dict[str, Any]:
    seg_c = {"kind": "segment", "gpu": "L40S", "startup_s": 300.0, "busy_s": 60.0,
             "seconds": 400.0}
    cap_c = {"kind": "caption", "gpu": "H100!", "startup_s": 320.0, "busy_s": 90.0,
             "seconds": 450.0}
    containers = [seg_c, cap_c]
    # J1w is segment only; J2w is the caption witness
    witness = {"human": {"job_id": "wave-h-test-cold-human", "state": "done",
                         "compile_cache": "off", "episodes": len(o.noise),
                         "containers": [seg_c]}} if o.witness else {}
    iso_witness = {"human": {"job_id": "wave-h-test-isocold-human", "state": "done",
                             "compile_cache": "off", "episodes": len(o.noise),
                             "stages": ["caption"], "containers": [cap_c]}} if o.witness else {}
    step = {"status": "done", "projected_usd": 1.0, "projected_upper_usd": 2.0,
            "billed_usd": None, "notes": [], "result": {}}
    return {"run_id": "102300000000", "created": 1792690000.0, "billing_since": "2026-10-22",
            "billing_baseline": {}, "billed_total": 3.1, "billed_at": 1792790000.0,
            "billed_through": 1792780000.0, "plan": "certified", "gate_cap": 9.5,
            "gate_cap_plan": "certified", "cap_override": None, "cap_override_plan": None,
            "steps": {
                "exact_e2e": {**step, "billed_usd": 1.9, "attempts": [attempt(1.85)]},
                "noise": {**step, "billed_usd": 0.7, "attempts": [attempt(0.72)]},
                "balanced": {**step, "attempts": [attempt(1.1)], "result": {
                    "jobs": {"human": {"cfg": None, "state": "done", "job_id": "wave-h-test-h",
                                       "stage_hashes": {"segment": CAND_HASH,
                                                        "caption": CAND_HASH},
                                       "containers": containers}},
                    "witness_jobs": witness}},
                "lab": {**step, "attempts": [attempt(0.3, budget=1.5, report_error=True,
                                                     status="failed"), attempt(0.6)]},
                "caption_iso": {**step, "attempts": [attempt(0.0)], "result": {
                    "jobs": {"human": {"cfg": None, "state": "done", "job_id": "wave-h-test-iso",
                                       "stages": ["caption"],
                                       "stage_hashes": {"caption": ISO_HASH},
                                       "oracle_segments": dict(segs), "containers": [cap_c]}},
                    "witness_jobs": iso_witness, "oracle_segments": dict(segs)}},
                "certify": {**step, "attempts": [attempt(0.0)]}}}


def report_text(state: dict[str, Any], certs: dict[str, dict[str, Any]]) -> str:
    """The finalize report's spend section and certificate table, from the state."""
    st = rb.State.from_json(state)
    wm = st.watermark
    usd = cc.usd
    lines = ["# Wave H-lite report", "", "Numbers only: no NVIDIA text, captions, prompts or "
             "episode ids.", "", f"Run `{st.run_id}`.", "", "## Spend per step", "",
             "| step | status | attempts | cap | tracked | charged | billed | billing |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in st.steps.items():
        lines.append(f"| {name} | {s.status} | {len(s.attempts)} | $5.00 | {usd(s.tracked_usd)} "
                     f"| {usd(s.charged(wm))} | {usd(s.billed_usd)} | reconciled |")
    tracked = sum(s.tracked_usd for s in st.steps.values())
    charged = sum(s.charged(wm) for s in st.steps.values())
    lines += ["", f"Total: **{usd(st.spent)}** of the $9.50 global cap: the larger of charged "
              "and billed, the figure the runbook gates on.", "",
              f"- tracked {usd(tracked)}: container-seconds x list $/h of every attempt;",
              f"- charged {usd(charged)}: tracked, with a killed or unreported attempt at its "
              "bound or budget until billing covers it;",
              f"- billed {usd(st.billed_total)}: Modal's billed delta of every app.", "",
              "## Certificate", "", "| stage | pool | GPU | verdict | id |",
              "|---|---|---|---|---|"]
    for stage, c in sorted(certs.items()):
        lines.append(f"| {stage} | human | {c['gpu']} | {c['verdict']} | `{c['id']}` |")
    return "\n".join(lines) + "\n"


def by_item(rows: list[cc.Row]) -> dict[str, cc.Row]:
    return {f"{r.check} | {r.item}": r for r in rows}


def failing(rows: list[cc.Row]) -> set[str]:
    return {f"{r.check} | {r.item}" for r in rows if r.status == cc.FAIL}
