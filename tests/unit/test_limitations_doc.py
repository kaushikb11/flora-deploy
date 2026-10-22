"""``docs/limitations.md`` stays true of the code: its status tables are generated from the
certificates and ``planner/rates.json`` (``scripts/limitations_status.py``), and each hand-written
claim is pinned to the code it describes, so a change that makes one false fails here."""

from __future__ import annotations

import json
from dataclasses import replace

from flora_deploy.planner.rates import RateTable
from scripts import limitations_status as L

TEXT = L.DOC.read_text(encoding="utf-8")


def test_limitations_md_is_up_to_date():
    assert TEXT == L.render(), (
        "docs/limitations.md is stale: run `uv run python scripts/limitations_status.py --write`")


def test_the_readme_and_the_docs_index_link_it():
    root = L.DOC.parents[1]
    assert "(docs/limitations.md)" in (root / "README.md").read_text(encoding="utf-8")
    assert "(limitations.md)" in (root / "docs" / "README.md").read_text(encoding="utf-8")


def test_the_table_is_certkeys_status_for_every_embodiment_and_stage():
    from flora_deploy.validate.certify import EMBODIMENTS

    st = L.statuses()
    assert list(st) == list(EMBODIMENTS)
    for emb, row in st.items():
        line = next(x for x in TEXT.splitlines() if x.startswith(f"| `{emb}` |"))
        assert line == f"| `{emb}` | " + " | ".join(L.WORDS[row[s]] for s in L.STAGES) + " |"
        for s in ("reward.terminal", "reward.dense", "reward.rubric"):
            assert row[s] == "reference"  # "Rewards ... need none"


def test_a_certificate_changes_the_table(monkeypatch):
    def certified(cert_dir=None):
        return {e: {s: "certified" if (e, s) == ("human", "segment") else "none"
                    for s in L.STAGES} for e in ("human", "yam")}

    monkeypatch.setattr(L, "statuses", certified)
    out = L.render(TEXT)
    assert "| `human` | **certified** | not yet |" in out
    assert "Certified today: `segment` for `human`." in out
    assert "Nothing is certified yet" not in out


def test_a_rate_change_changes_the_cost_table():
    data = json.loads((L.DOC.parents[1] / "src/flora_deploy/planner/rates.json").read_text())
    for r in data["rates"]:
        if r["kind"] == "reward.rubric":
            r.update(status="measured", n=3, uncertainty=0.3)
    table = replace(RateTable.from_dict(data, source="test"), source="planner/rates.json")
    out = L.render(TEXT, table=table)
    assert "| GPU work, `reward.rubric` | measured on 3 jobs, ±30% |" in out
    assert out != TEXT


def test_describe():
    assert L.describe("measured", 1, 0.5) == "measured on one job, ±50%"
    assert L.describe("measured", 1, 0.5, "container") == "measured on one container, ±50%"
    assert L.describe("measured", None, 1.0) == "measured, ±100%"
    assert L.describe("placeholder", None, 0.6) == "estimate (not measured), ±60%"


# ---------------------------------------------------------------------------------------------
# The hand-written claims


def test_what_certified_promises_matches_the_policy():
    from flora_deploy.validate.certify import Thresholds
    from flora_deploy.validate.certkey import metric_threshold

    tol = Thresholds().tolerance
    assert f"no more than {tol} times as much" in TEXT
    count = metric_threshold("count_mismatch", 0.0, tol, 0.0)  # quiet NVIDIA runs: the floor
    assert f"up to {count:.1%} of episodes" in TEXT
    # counted over the candidate and the cold witness together, to the nearest episode
    from flora_deploy.validate.certkey import count_flips_allowed

    assert count_flips_allowed(count, 18) == 2 and "2 of 18 pass" in TEXT
    assert "warm and\ncold containers together" in TEXT
    assert metric_threshold("count_mismatch", 0.1, tol, 0.0) > count  # "more when ... disagree"


def test_rubric_has_not_run_on_a_gpu_while_its_rate_is_unmeasured():
    if "`reward.rubric` has only run in CPU tests" in TEXT:
        assert RateTable.load().rate("balanced", "reward.rubric", 1).status != "measured"


def test_unsupported_items_are_refused_by_the_code(tmp_path):
    import pytest

    from flora_deploy.config.presets import PRESETS
    from flora_deploy.config.schema import UNSUPPORTED_SINK_SCHEMES, unsupported_sink_message
    from flora_deploy.engines.offline_llm import FORBIDDEN
    from flora_deploy.media.avsafe import UnsupportedContainer, open_input
    from flora_deploy.media.sources.mp4_dir import VIDEO_SUFFIXES

    assert set(VIDEO_SUFFIXES) == {".mp4", ".mov", ".m4v"}
    mkv = tmp_path / "a.mkv"
    mkv.write_bytes(b"\x1aE\xdf\xa3" + b"\0" * 60)  # an EBML (Matroska / WebM) header
    with pytest.raises(UnsupportedContainer):
        open_input(mkv)

    assert set(UNSUPPORTED_SINK_SCHEMES) == {"gs", "r2"}
    assert "AWS_ENDPOINT_URL" in unsupported_sink_message("r2://b")
    assert "quantization" in FORBIDDEN and "kv_cache_dtype" in FORBIDDEN
    assert not PRESETS["interactive"].available


def test_embodiments_without_a_certification_path():
    import sys

    sys.path.insert(0, str(L.DOC.parents[1] / "scripts"))
    import wave_h

    from flora_deploy.validate.certkey import MIN_SAMPLES_FLOOR
    from flora_deploy.validate.golden import POOL_TARGETS

    for emb in ("dex", "tianji", "yam32"):
        assert emb not in wave_h.CERT_SPECS
        assert POOL_TARGETS[emb] < MIN_SAMPLES_FLOOR  # "too few suitable public episodes"
    assert {"human", "yam"} <= set(wave_h.CERT_SPECS)


def test_known_approximations():
    from flora_deploy.config.presets import get_preset

    t = RateTable.load()
    for k in ("segment", "caption"):  # three-camera data is priced with the same rate
        assert t.rate("balanced", k, 3) == t.rate("balanced", k, 1)
    pools = get_preset("balanced").pools
    assert all(pools[k].gpu == "H100!" for k in pools if k != "segment")
