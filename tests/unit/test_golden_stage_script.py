"""``scripts/golden_stage.py``'s verify pass on a synthetic manifest (no Volume)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "golden_stage.py"
spec = importlib.util.spec_from_file_location("golden_stage", SCRIPT)
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

DATA = {"golden/v2/human/ds/a/main.mp4": b"aaaa", "golden/v2/human/ds/b/main.mp4": b"bb"}


def _ep(slug: str, data: bytes) -> dict:
    return {"id": f"human/ds/{slug}", "path": f"golden/v2/human/ds/{slug}", "status": "staged",
            "oracle_eligible": True, "verify": {"ok": True}, "cut_check": {"main": "ok"},
            "selection_digest": f"d-{slug}",
            "files": {"main": {"file": "main.mp4", "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}}}


def _manifest() -> dict:
    return {"complete": True, "episodes": [_ep("a", b"aaaa"), _ep("b", b"bb")]}


def test_verify_reads_every_file_back():
    seen = []

    def read(path):
        seen.append(path)
        return DATA[path]

    assert gs.verify(_manifest(), read=read) == []
    assert sorted(seen) == sorted(DATA)


def test_verify_names_every_problem():
    m = _manifest()
    m["episodes"][0]["cut_check"] = {"main": ["rate"]}
    m["episodes"][1]["oracle_eligible"] = False
    bad = dict(DATA, **{"golden/v2/human/ds/b/main.mp4": b"bX"})
    probs = gs.verify(m, read=bad.__getitem__)
    assert any("cut_check" in p for p in probs)
    assert any("not oracle-eligible" in p for p in probs)
    assert any("sha256 differs" in p for p in probs)


def test_verify_compares_entries_shared_with_the_earlier_version():
    earlier = copy.deepcopy(_manifest())
    earlier["episodes"][0]["files"]["main"]["sha256"] = "0" * 64  # same entry, other bytes
    earlier["episodes"][1]["selection_digest"] = "changed"  # another entry: not compared
    earlier["episodes"][1]["files"]["main"]["sha256"] = "1" * 64
    probs = gs.verify(_manifest(), read=DATA.__getitem__, earlier=earlier)
    assert probs == ["human/ds/a: main differs from the earlier version's file"]
