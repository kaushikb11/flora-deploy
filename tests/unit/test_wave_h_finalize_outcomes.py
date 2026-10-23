"""The generated certification and cost wording for the three re-certification outcomes:
segmentation and captioning certified, segmentation only (outcome (b): captioning failed), and
none. Outcome (b) must never advertise a certified ``balanced`` captioning stage or price, and
must give the two honest ways to get captions (``allow_uncertified: true``, or a separate
``exact`` job). Fake certificates only, in a temporary registry (``certkey.CERT_DIR`` patched).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import limitations_status as L
from tests.unit.test_partial_certification import (  # noqa: F401 - registry is a fixture
    blocks,
    no_certificates_readme,
    registry,
    write_cert,
)
from tests.unit.test_wave_h_finalize import fin

EXACT = fin.exact_price()


@pytest.fixture
def segment_only(registry):  # noqa: F811 - the imported fixture
    write_cert(registry, "segment")
    return registry


@pytest.fixture
def both(registry):  # noqa: F811 - the imported fixture
    for s in ("segment", "caption"):
        write_cert(registry, s)
    return registry


def flat(s: str) -> str:
    return " ".join(s.split())


def test_the_exact_price_is_the_cost_blocks_figure():
    from flora_deploy.planner.rates import RateTable

    planned = fin.planned_usd_per_video_h(RateTable.load(), "exact")
    assert EXACT == f"${planned['usd_per_video_h']:.2f}"
    assert L.exact_price() == EXACT


# ---------------------------------------------------------------------------------------------
# README wave-h:certified / wave-h:allow-uncertified


def test_readme_blocks_segment_only(segment_only, tmp_path):
    readme = no_certificates_readme(tmp_path)
    b = blocks(fin.certification_blocks(readme, [segment_only]))
    st, cfg = flat(b["certified"]), flat(b["allow-uncertified"])
    assert st.startswith("Accuracy certificates: segmentation for `human` data (`L40S`). "
                         "Captioning is not certified:")
    assert "captioning for" not in st and "H100" not in st
    assert "`allow_uncertified: true`" in st and "separate `exact` job, which captions its own " \
        f"segments at the `exact` price (about {EXACT} per video-hour)" in st
    assert "Certified today: segmentation for `human` data (`L40S`). Captioning is not " \
        "certified." in cfg
    assert "one `balanced` job with `allow_uncertified: true`: its caption records say " \
        "`certified: false`" in cfg
    assert "`preset: exact, stages: [segment, caption]`: it captions `exact`'s own segments, " \
        f"at the `exact` price (about {EXACT} per video-hour)" in cfg
    assert "One config has one preset" in cfg
    assert all(len(line) <= 125 for v in b.values() for line in v.splitlines())


def test_readme_blocks_both_and_none_keep_their_wording(both, tmp_path):
    readme = no_certificates_readme(tmp_path)
    b = blocks(fin.certification_blocks(readme, [both]))
    assert b["certified"] == ("Accuracy certificates: segmentation and captioning for `human` "
                              "data (`L40S` / `H100!`).\nRewards run NVIDIA's own engine, as "
                              "`exact` does, and need none. Everything else runs uncertified.\n")
    assert "exact" not in b["allow-uncertified"] and "Captioning is not" not in b["certified"]
    none = blocks(readme)
    assert "none yet" in none["certified"] and "required" in none["allow-uncertified"]
    assert "Captioning is not" not in readme


# ---------------------------------------------------------------------------------------------
# docs/limitations.md certification block


def test_limitations_segment_only(segment_only):
    text = L.block_certification(segment_only)
    assert "Certified today: `segment` for `human`." in text
    t = flat(text)
    assert "Captioning is not certified for any embodiment." in t
    assert "`allow_uncertified: true` (its caption records say `certified: false`)" in t
    assert "`preset: exact, stages: [segment, caption]`: it captions `exact`'s own segments" in t
    assert f"(about {EXACT} per video-hour)" in t and "One config has one preset" in t
    row = next(line for line in text.splitlines() if line.startswith("| `human`"))
    assert row.split(" | ")[1:3] == ["**certified**", "not yet"]


def test_limitations_both_and_none_have_no_caption_options(both, tmp_path):
    assert "Captioning is not certified" not in L.block_certification(both)
    empty = tmp_path / "empty"
    empty.mkdir()
    none = L.block_certification(empty)
    assert "Nothing is certified yet" in none and "exact" not in none


# ---------------------------------------------------------------------------------------------
# README wave-h:cost


def findings(certified=()):
    return SimpleNamespace(
        date="2026-10-22", certified=frozenset(certified), ms={},
        cost={"balanced": {"usd_per_video_h": 5.28, "uncertainty": 0.5, "terms": [
            {"kind": "segment", "usd_per_video_h": 2.10},
            {"kind": "caption", "usd_per_video_h": 3.18}]}, "exact": None},
        planned={"exact": {"usd_per_video_h": 36.39}})


@pytest.mark.parametrize("certified", [(), ("segment", "caption")])
def test_cost_block_both_or_none_is_one_price(certified):
    block = fin.cost_block(findings(certified))
    assert block.startswith("`balanced` costs about **$5.28 per video-hour** of GPU work (±50%) "
                            "for segmentation + captioning (`exact`: $36.39),\n")
    assert len(block.splitlines()) == 3


def test_cost_block_segment_only_never_prices_certified_captioning():
    block = fin.cost_block(findings(("segment",)))
    lines = block.splitlines()
    assert lines[0] == ("`balanced` costs about **$2.10 per video-hour** of GPU work (±50%) for "
                        "segmentation, which is certified;")
    assert lines[1:3] == ["captioning adds about $3.18 but is not certified (its records say "
                          "`certified: false`),",
                          "or `exact` runs certified segmentation + captioning for $36.39,"]
    assert "$5.28" not in block and "segmentation + captioning (" not in block
    assert all(len(line) <= 125 for line in lines)
    assert lines[3].startswith("at Modal list prices; a projection from rates")
    assert "`flora plan` shows the all-in estimate" in block


def test_committed_readme_and_limitations_reproduce():
    readme = (fin.ROOT / "README.md").read_text(encoding="utf-8")
    from flora_deploy.validate import certkey

    assert fin.certification_blocks(readme, [certkey.CERT_DIR]) == readme
    assert L.render() == L.DOC.read_text(encoding="utf-8")
