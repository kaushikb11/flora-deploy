"""``scripts/wave_h_finalize.py`` for the v4 (stage-isolated) caption certificate: the digest
JSON ``--fetch`` exports per side (from the exact audits' ``model_calls`` and the balanced
caption audits' ``input_digests``; hex only), the per-job store it reads, and the
``flora certify`` arguments ``certify_all`` passes (caption: J2 + J2w + noise-12 with
``--digests``, ``--oracle-segments``, ``--reported-e2e``, ``--families``, the repeat; segment:
J1 + J1w + noise-12). Nothing here contacts Modal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.fakes.certrecords import BOUNDS, rec, segs
from tests.unit.test_wave_h_certified import (
    SPEC,
    _jsonl,
    _labs,
    _uid,
    certified_world,
    fin,
    wh,
)

SECRET = "SECRET-NV-CAPTION"


def _hex(i: int) -> str:
    return hashlib.sha256(str(i).encode()).hexdigest()


def _cap_rec(uid: str, preset: str, statuses, caps=None):
    rows = segs([(i, i + 1.0) for i in range(len(statuses))],
                caps or [f"{SECRET} {i}" for i in range(len(statuses))])
    import dataclasses

    rows = [dataclasses.replace(s, status=st) for s, st in zip(rows, statuses, strict=True)]
    return rec(uid, "caption", preset, fp=None, segments=rows)


def _segcap_audit(uid: str, calls, captions):
    return {"schema": "flora_segcap_audit_v1", "uid": uid, "stage": "segcap",
            "result": {"segments": [{"caption": c, "status": "complete"} for c in captions],
                       "flora_reference": {"model_calls": [
                           {"model": m, "input_digest": d, "output_limit": lim}
                           for m, d, lim in calls]}}}


def test_segcap_digests_pair_the_kth_captioner_call_with_the_kth_complete_row():
    r = _cap_rec("u1", "exact", ["complete", "unsupported_target_frames", "complete"])
    a = _segcap_audit("u1", [("tempseg", _hex(0), 4096), ("captioner", _hex(1), 512),
                             ("captioner", _hex(2), 512)], [f"{SECRET} 0", f"{SECRET} 2"])
    got = fin.segcap_digests([r], [a])
    assert got == {"u1": [_hex(1), None, _hex(2)]}
    # a complete row with no call left: None (certify refuses oracle_digests_missing)
    short = _segcap_audit("u1", [("captioner", _hex(1), 512)], [f"{SECRET} 0"])
    assert fin.segcap_digests([r], [short]) == {"u1": [_hex(1), None, None]}
    # a retried shard: the line whose captions are the record's wins
    stale = _segcap_audit("u1", [("captioner", _hex(7), 512), ("captioner", _hex(8), 512)],
                          ["other", "other"])
    assert fin.segcap_digests([r], [stale, a]) == got
    # two lines with the record's captions but other digests: all None, never a guess
    twin = _segcap_audit("u1", [("captioner", _hex(5), 512), ("captioner", _hex(6), 512)],
                         [f"{SECRET} 0", f"{SECRET} 2"])
    assert fin.segcap_digests([r], [a, twin]) == {"u1": [None, None, None]}
    # no text: only hex or None
    bad = _segcap_audit("u1", [("captioner", "not hex", 512), ("captioner", _hex(2), 512)],
                        [f"{SECRET} 0", f"{SECRET} 2"])
    assert fin.segcap_digests([r], [bad]) == {"u1": [None, None, _hex(2)]}


def test_caption_digests_come_from_the_caption_audit_rows():
    r = _cap_rec("u1", "balanced", ["complete", "unsupported_target_frames", "complete"])
    row = {"stage": "caption", "uid": "u1", "input_digests": [_hex(1), None, _hex(2)]}
    shard = {"stage": "caption", "shard_audit": True, "uid": "u1",
             "input_digests": [_hex(9)] * 3}
    assert fin.caption_digests([r], [shard, row]) == {"u1": [_hex(1), None, _hex(2)]}
    assert fin.caption_digests([r], [{**row, "input_digests": [_hex(1)]}]) == {
        "u1": [None, None, None]}  # not one per row
    assert fin.caption_digests([r], []) == {"u1": [None, None, None]}


def test_fetch_exports_records_and_digests_from_each_jobs_store(tmp_path):
    ctx, labs = certified_world(tmp_path)
    st = ctx.st
    st.steps["exact_e2e"].result["jobs"][wh.JOB]["store"] = "modal-volume://flora-wave-h/b1"
    st.steps["balanced"].result["witness_jobs"][wh.JOB]["stages"] = ["segment"]  # J1w
    paths = fin.Paths(root=wh.ROOT, state=ctx.state_path, records=ctx.spec.records, lab=labs)
    ran, asked = [], []

    def run(args, check):
        ran.append(args)

    def audits(job_id, store):
        asked.append((job_id, store))
        if "exact" in job_id or "noise" in job_id:
            return [_segcap_audit(_uid(g), [("captioner", _hex(i * 10 + j), 512)
                                            for j in range(len(BOUNDS))], [])
                    for i, g in enumerate(SPEC.episodes)]
        return [{"stage": "caption", "uid": _uid(g),
                 "input_digests": [_hex(i * 10 + j) for j in range(len(BOUNDS))]}
                for i, g in enumerate(SPEC.episodes)]

    fin.fetch_records(paths, st, run=run, audits=audits)
    stores = {a[a.index("--job") + 1]: a[a.index("--store") + 1] for a in ran}
    assert stores["wave-h-r-exact_e2e"] == "modal-volume://flora-wave-h/b1"
    assert stores["wave-h-r-iso"] == "modal-volume://flora-wave-h/r"
    stages = {(a[a.index("--job") + 1], a[a.index("--stage") + 1]) for a in ran}
    assert ("wave-h-r-cold", "segment") in stages
    assert ("wave-h-r-cold", "caption") not in stages  # J1w is segment-only
    for side in ("exact", "noise", "isolated", "isolated_witness"):
        d = json.loads(fin.digest_path(paths, side, wh.JOB).read_text())
        text = fin.digest_path(paths, side, wh.JOB).read_text()
        assert SECRET not in text and "c0-" not in text  # hex and null only
        for ds in d.values():
            assert all(x is None or len(x) == 64 for x in ds)
        i = SPEC.episodes.index(SPEC.witness[0])
        assert d[_uid(SPEC.witness[0])] == [_hex(i * 10 + j) for j in range(len(BOUNDS))]
    assert {s for _, s in asked} == {"modal-volume://flora-wave-h/b1",
                                     "modal-volume://flora-wave-h/r"}
    assert not fin.digest_path(paths, "balanced", wh.JOB).exists()  # J1: no pairing


def test_fetch_without_the_isolated_job_reads_no_audits(tmp_path):
    ctx, labs = certified_world(tmp_path, isolated=False)
    paths = fin.Paths(root=wh.ROOT, state=ctx.state_path, records=ctx.spec.records, lab=labs)
    fin.fetch_records(paths, ctx.st, run=lambda args, check: None,
                      audits=lambda *a: pytest.fail("audits read"))


def test_certify_all_passes_the_v4_caption_and_the_segment_arguments(tmp_path):
    ctx, labs = certified_world(tmp_path)
    rep = ctx.spec.records / "repeat" / wh.JOB
    _jsonl(rep / "caption.jsonl", [])
    (rep / fin.DIGESTS).write_text("{}")
    ctx.st.step("repeat").status = "done"
    ctx.st.step("repeat").result = {"jobs": {wh.JOB: {"job_id": "old-noise", "state": "done"}}}
    paths = fin.Paths(root=wh.ROOT, state=ctx.state_path, records=ctx.spec.records, lab=labs)
    seen = []

    def cli(args):
        seen.append(args)
        return 2, ""

    fin.certify_all(paths, ctx.st, tmp_path / "c", derived={}, cli=cli)
    by = {a[a.index("--stage") + 1]: a for a in seen}
    r = ctx.spec.records

    def val(a, flag):
        return a[a.index(flag) + 1]

    def vals(a, flag):
        return [a[i + 1] for i, x in enumerate(a) if x == flag]

    seg, cap = by["segment"], by["caption"]
    # segment: J1 + J1w + noise-12, unchanged
    assert val(seg, "--balanced") == str(r / "balanced" / wh.JOB / "segment.jsonl")
    assert val(seg, "--witness") == str(r / "witness" / wh.JOB / "segment.jsonl")
    assert val(seg, "--noise") == str(r / "noise" / wh.JOB / "segment.jsonl")
    assert "--digests" not in seg and "--oracle-segments" not in seg
    # caption: the isolated J2 is the candidate (its config), J2w the witness
    iso = ctx.st.steps["caption_iso"].result
    assert val(cap, "--balanced") == str(r / "isolated" / wh.JOB / "caption.jsonl")
    assert val(cap, "-c") == iso["jobs"][wh.JOB]["cfg"]
    assert val(cap, "--witness") == str(r / "isolated_witness" / wh.JOB / "caption.jsonl")
    assert val(cap, "--noise") == str(r / "noise" / wh.JOB / "caption.jsonl")
    assert val(cap, "--exact") == str(r / "exact" / wh.JOB / "caption.jsonl")
    d = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in vals(cap, "--digests")}
    assert d == {"exact": str(r / "exact" / wh.JOB / fin.DIGESTS),
                 "noise": str(r / "noise" / wh.JOB / fin.DIGESTS),
                 "balanced": str(r / "isolated" / wh.JOB / fin.DIGESTS),
                 "witness": str(r / "isolated_witness" / wh.JOB / fin.DIGESTS),
                 "repeat": str(rep / fin.DIGESTS)}
    assert val(cap, "--oracle-segments") == iso["oracle_segments"]["path"]
    assert val(cap, "--reported-e2e") == str(r / "balanced" / wh.JOB / "caption.jsonl")
    assert val(cap, "--reported-repeat") == str(rep / "caption.jsonl")
    fam = json.loads(Path(val(cap, "--families")).read_text())
    assert set(fam) == {_uid(g) for g in SPEC.episodes}
    assert {fam[_uid(g)] for g in SPEC.witness} <= {"n-egodex", "n-wgo"}
    assert {fam[_uid(g)] for g in SPEC.episodes if g not in SPEC.witness} <= {
        "x-egodex", "x-wgo"}
    # a staged file that is not the pinned one is a missing input, never certified
    Path(iso["oracle_segments"]["path"]).write_bytes(b"{}\n")
    seen.clear()
    out = fin.certify_all(paths, ctx.st, tmp_path / "c2", derived={}, cli=cli)
    caption = next(o for o in out if o["stage"] == "caption")
    assert "sha256 differs" in caption["refused"]
    assert [a[a.index("--stage") + 1] for a in seen] == ["segment"]


def test_family_labels():
    assert fin.family_of("cert-human--n-egodex_add_remove_lid-episode_000014") == "n-egodex"
    assert fin.family_of("cert-human--x-wgo-homer_53") == "x-wgo"
    assert fin.family_of("cert-yam--n-manirewardbench_yam-stack_plates_000003") == \
        "n-manirewardbench"
    assert fin.family_of("ds--u1") is None
    for uid in ("cert-yam--x-aloha_static_towel-episode_000000", "c--n-robocoin_x"):
        f = fin.family_of(uid)
        assert f and fin._FAMILY.fullmatch(f)


def test_the_certify_step_certifies_caption_stage_isolated(tmp_path, monkeypatch):
    """End to end on the package's v4 ``flora certify``: the stage-isolated caption
    certificate comes out ``certified`` when every Captioner input matched the oracle's."""
    monkeypatch.setattr(fin, "fetch_records", lambda paths, st, run=None: None)
    ctx, labs = certified_world(tmp_path)

    class Sh:
        def __call__(self, args, **kw):
            return type("P", (), {"returncode": 0})()

    res = wh.run_flora_certify(ctx, finalize=fin, sh=Sh(), lab_dir=labs)
    assert res["ok"], res
    certs = {json.loads(p.read_text())["stage"]: json.loads(p.read_text())
             for p in (ctx.work / "certify").glob("cert_*.json")}
    cap = certs["caption"]
    assert cap["verdict"] == "certified"
    assert (cap.get("scoring") or {}).get("caption") == "stage_isolated"
    assert (cap.get("segments_from") or {}).get("sha256") == \
        ctx.st.steps["caption_iso"].result["oracle_segments"]["sha256"]
    # inputs that differ from the oracle's are refused, not failed
    dpath = ctx.spec.records / "isolated" / wh.JOB / fin.DIGESTS
    d = json.loads(dpath.read_text())
    uid = next(iter(d))
    d[uid][0] = "0" * 64
    dpath.write_text(json.dumps(d))
    res = wh.run_flora_certify(ctx, finalize=fin, sh=Sh(), lab_dir=_labs(tmp_path / "lab2"))
    assert not res["ok"] and res["stages"]["caption"]["verdict"] is None
    assert res["stages"]["caption"]["refused"]
