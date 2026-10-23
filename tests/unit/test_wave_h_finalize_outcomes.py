"""The generated certification and cost wording for the three re-certification outcomes:
segmentation and captioning certified, segmentation only (outcome (b): captioning failed), and
none. Outcome (b) must never advertise a certified ``balanced`` captioning stage. The README
note is one plain sentence (no embodiment lists, GPU names or ``exact`` workaround), and the
README cost block is the all-in table ``scripts/cost_table.py`` renders. Fake certificates only,
in a temporary registry (``certkey.CERT_DIR`` patched).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import cost_table as C
from scripts import limitations_status as L
from tests.unit.test_partial_certification import (  # noqa: F401 - registry is a fixture
    blocks,
    no_certificates_readme,
    registry,
    write_cert,
)
from tests.unit.test_wave_h_finalize import fin

NONE_NOTE = ("**Note:** `balanced` segment and caption records currently say `certified: false`, "
             "so they need\n`allow_uncertified: true` here and `--allow-uncertified` when you "
             "export.\n")


@pytest.fixture
def segment_only(registry):  # noqa: F811 - the imported fixture
    write_cert(registry, "segment")
    return registry


@pytest.fixture
def both(registry):  # noqa: F811 - the imported fixture
    for s in ("segment", "caption"):
        write_cert(registry, s)
    return registry


def test_the_exact_price_is_the_planners_figure():
    from flora_deploy.planner.rates import RateTable

    planned = fin.planned_usd_per_video_h(RateTable.load(), "exact")
    assert fin.exact_price() == f"${planned['usd_per_video_h']:.2f}"


# ---------------------------------------------------------------------------------------------
# README wave-h:allow-uncertified


def test_readme_note_segment_only(segment_only, tmp_path):
    readme = no_certificates_readme(tmp_path)
    b = blocks(fin.certification_blocks(readme, [segment_only]))
    assert b["allow-uncertified"] == (
        "**Note:** `balanced` caption records currently say `certified: false`, so they need\n"
        "`allow_uncertified: true` here and `--allow-uncertified` when you export.\n")
    assert "certified" not in b  # the status block is retired


def test_readme_note_both_and_none(both, tmp_path):
    readme = no_certificates_readme(tmp_path)
    b = blocks(fin.certification_blocks(readme, [both]))
    assert b["allow-uncertified"] == ("Every `balanced` stage is certified for `human` data; "
                                      "remove `allow_uncertified`.\n")
    assert blocks(readme)["allow-uncertified"] == NONE_NOTE
    for text in (readme, fin.certification_blocks(readme, [both])):
        note = blocks(text)["allow-uncertified"]
        assert "Captioning is not" not in text and "H100" not in note and "L40S" not in note


# ---------------------------------------------------------------------------------------------
# docs/limitations.md certification block


def test_limitations_segment_only(segment_only):
    text = L.block_certification(segment_only)
    assert "Certified today: `segment` for `human`." in text
    assert "Captioning is not certified" not in text and "preset: exact" not in text
    row = next(line for line in text.splitlines() if line.startswith("| `human`"))
    assert row.split(" | ")[1:3] == ["**certified**", "not yet"]


def test_limitations_both_and_none(both, tmp_path):
    assert "Captioning is not certified" not in L.block_certification(both)
    empty = tmp_path / "empty"
    empty.mkdir()
    none = L.block_certification(empty)
    assert "Nothing is certified yet" in none and "exact" not in none


# ---------------------------------------------------------------------------------------------
# README wave-h:cost


def test_cost_block_is_the_all_in_table_whatever_is_certified():
    block = fin.cost_block(SimpleNamespace(table=None))
    assert block == C.block_readme()
    lines = block.splitlines()
    assert lines[0] == "| Video | Segment + caption | + rewards | Wall-clock |"
    assert [ln.split(" | ")[0] for ln in lines[2:5]] == ["| 3 min (0.05 h)", "| 1 h", "| 10 h"]
    assert "All-in estimates at Modal list prices" in block
    # never the measured-vs-planner `exact` comparison
    assert "exact" not in block and "planner" not in block and "measured" not in block


def test_cost_block_rows_are_the_short_tables_prices():
    from flora_deploy.planner.rates import RateTable

    t = RateTable.load()
    rows = C.block_readme(t).splitlines()[2:5]
    for v, row in zip(C.README_SIZES, rows, strict=True):
        a, b = C.price(v, C.SEGCAP, table=t), C.price(v, C.SEGCAP + C.REWARDS, table=t)
        assert row.split(" | ")[1:] == [C.usd(a.cost), C.usd(b.cost), C.hours(a.wall_s) + " |"]


def test_committed_readme_and_limitations_reproduce():
    readme = (fin.ROOT / "README.md").read_text(encoding="utf-8")
    from flora_deploy.validate import certkey

    assert fin.certification_blocks(readme, [certkey.CERT_DIR]) == readme
    assert C.render_readme(readme) == readme
    assert L.render() == L.DOC.read_text(encoding="utf-8")
