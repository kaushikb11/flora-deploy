"""docs/output-format.md cannot drift: every emitted field is documented, and nothing else is."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

from flora_deploy.export.compose import REWARD_COLS, SEGMENT_STAGES, compose, stage_schema
from flora_deploy.jobs.stores import LocalStore
from flora_deploy.jobs.types import EpisodeError
from flora_deploy.records import (
    DenseReward,
    Provenance,
    RubricReward,
    Segment,
    StageRecord,
    TerminalReward,
    UncertifiedReason,
)

DOC = Path(__file__).resolve().parents[2] / "docs" / "output-format.md"


def doc_tables() -> dict[str, dict[str, str]]:
    """``<!-- fields: name -->`` -> {field: type} of the table that follows it."""
    text = DOC.read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"<!-- fields: ([\w.]+) -->\n((?:\|.*\n)+)", text):
        rows = m.group(2).strip().splitlines()[2:]  # skip header + separator
        cols = {}
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            name = cells[0].strip("`")
            assert name not in cols, f"{m.group(1)}: {name} documented twice"
            cols[name] = cells[1]
        assert m.group(1) not in out
        out[m.group(1)] = cols
    return out


TABLES = doc_tables()


def names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


@pytest.mark.parametrize("section,cls", [
    ("record", StageRecord), ("provenance", Provenance), ("segment", Segment),
    ("terminal", TerminalReward), ("dense", DenseReward), ("rubric", RubricReward),
    ("error", EpisodeError),
])
def test_jsonl_fields(section, cls):
    assert set(TABLES[section]) == names(cls)


def test_parquet_columns_and_types():
    common = stage_schema("segment").names[: len(TABLES["parquet.common"])]
    for stage in ("segment", "caption", *REWARD_COLS):
        schema = stage_schema(stage)
        own = "parquet.segment" if stage in SEGMENT_STAGES else f"parquet.{stage}"
        documented = {**TABLES["parquet.common"], **TABLES[own]}
        assert set(schema.names) == set(documented), stage
        for f in schema:
            assert documented[f.name] == str(f.type), (stage, f.name)
    assert set(common) == set(TABLES["parquet.common"])


def test_uncertified_reasons_documented():
    text = DOC.read_text(encoding="utf-8")
    for reason in get_args(UncertifiedReason):
        assert f"`{reason}`" in text


def test_summary_fields(tmp_path):
    from tests.unit.test_export_compose import JOB, setup_job

    store = LocalStore(tmp_path / "store")
    setup_job(store)
    summary = compose(store, JOB).summary
    assert set(TABLES["summary"]) == set(summary)
    per_stage = {k for s in summary["stages"].values() for k in s}
    assert set(TABLES["summary.stage"]) == per_stage


def test_stage_files_in_layout():
    text = DOC.read_text(encoding="utf-8")
    for stage in ("segment", "caption", *REWARD_COLS):
        assert f"{stage}.jsonl" in text and f"{stage}.parquet" in text
