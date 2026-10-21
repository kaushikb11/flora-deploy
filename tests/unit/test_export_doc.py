"""docs/export.md cannot drift: its field tables match what the LeRobot export writes."""

from __future__ import annotations

import json
import re
from pathlib import Path

from flora_deploy.export.lerobot import (
    EPISODE_FIELDS,
    FRAME_FEATURES,
    PROVENANCE_TYPE,
    export_lerobot,
)
from tests.unit.test_export_lerobot import LENGTHS, make_v3, records

DOC = Path(__file__).resolve().parents[2] / "docs" / "export.md"


def tables() -> dict[str, dict[str, str]]:
    text = DOC.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"<!-- fields: ([\w.]+) -->\n((?:\|.*\n)+)", text):
        rows = m.group(2).strip().splitlines()[2:]
        out[m.group(1)] = {r.strip("|").split("|")[0].strip().strip("`"):
                           r.strip("|").split("|")[1].strip() for r in rows}
    return out


T = tables()


def test_episode_fields():
    assert set(T["episode"]) == set(EPISODE_FIELDS)
    for name, (typ, _stage) in EPISODE_FIELDS.items():
        if name != "flora_provenance":
            assert T["episode"][name] == re.sub(r"(item|element): ", "", str(typ)), name


def test_provenance_fields():
    assert set(T["episode.provenance"]) == {f.name for f in PROVENANCE_TYPE.value_type}


def test_frame_columns():
    assert T["frame"] == {
        k: f["dtype"] for k, f in FRAME_FEATURES.items()}


def test_info_flora_block(tmp_path):
    src = make_v3(tmp_path / "src", LENGTHS)
    export_lerobot(src, tmp_path / "out", records(LENGTHS))
    info = json.loads((tmp_path / "out/meta/info.json").read_text())
    assert set(T["info.flora"]) == set(info["flora"])
