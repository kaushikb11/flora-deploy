"""The committed golden selection (validate/golden_selection_v1.json) and its validator."""

import copy

import pytest

from flora_deploy.validate.golden import (
    POOL_TARGETS,
    POOLS,
    SelectionError,
    entry_digest,
    entry_id,
    episode_relpath,
    load_selection,
    pool_counts,
    validate_selection,
)

SHA = "0" * 40


@pytest.fixture(scope="module")
def committed():
    return load_selection()


def test_committed_selection_is_valid(committed):
    validate_selection(committed)


def test_committed_selection_fits_the_budget(committed):
    counts = pool_counts(committed)
    tempseg = sum(n for p, n in counts.items() if p != "reward")
    assert 28 <= tempseg <= 32, counts
    assert 10 <= counts["reward"] <= 14, counts
    for pool in POOLS:  # every pool covered, each within one or two of its target
        assert abs(counts[pool] - POOL_TARGETS[pool]) <= 2, (pool, counts)


def test_committed_selection_pins_and_licenses(committed):
    for key, ds in committed["datasets"].items():
        assert len(ds["revision"]) == 40, key
        assert ds["license"].strip(), key
    for e in committed["episodes"]:
        assert e.get("expected_s", 1) <= 120
        assert episode_relpath(e).parts[:2] == ("golden", "v1")


def test_yam32_picks_long_three_view_yam(committed):
    longest = [e for e in committed["episodes"] if e["pool"] == "yam32"]
    assert longest and all(e["yam32"] and len(e["cameras"]) == 3 for e in longest)
    assert all(e["expected_s"] > 60 for e in longest)


def test_reward_set_has_both_outcomes(committed):
    reward = [e for e in committed["episodes"] if e["pool"] == "reward"]
    binary = [e["outcome"]["binary_success"] for e in reward if "binary_success" in e["outcome"]]
    assert 0 in binary and 1 in binary
    rubric = {e["outcome"]["reward"] for e in reward if "reward" in e["outcome"]}
    assert rubric == {1, 2, 3, 4, 5}


def _minimal():
    return {
        "version": "golden/v1",
        "datasets": {
            "ds": {"repo": "o/n", "revision": SHA, "format": "lerobot_v3", "license": "MIT",
                   "task_source": "meta"},
            "files": {"repo": "o/f", "revision": SHA, "format": "mp4_files", "license": "MIT",
                      "task_source": "jsonl", "annotations": "roboreward",
                      "annotation_file": "m.jsonl"},
        },
        "episodes": [
            {"pool": "yam", "dataset": "ds", "slug": "episode_000000", "episode": 0,
             "profile": "yam", "why": "x",
             "cameras": {"main": "a", "left_wrist": "b", "right_wrist": "c"}},
            {"pool": "reward", "dataset": "files", "slug": "r0", "episode": "x.mp4",
             "views": ["x.mp4"], "modes": ["terminal"], "outcome": {"reward": 3}, "why": "y"},
        ],
    }


def test_minimal_selection_is_valid():
    validate_selection(_minimal())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s["datasets"]["ds"].update(revision="main"), "40-hex"),
        (lambda s: s["datasets"]["ds"].update(license=""), "license is required"),
        (lambda s: s["datasets"]["ds"].update(format="zip"), "format must be"),
        (lambda s: s["episodes"][0].update(pool="human"), "single view"),
        (lambda s: s["episodes"][0].update(yam32=True), "yam32=true"),
        (lambda s: s["episodes"][0].update(profile="dex"), "uses profile 'yam'"),
        (lambda s: s["episodes"][0].update(episode="0"), "integer episode"),
        (lambda s: s["episodes"][0].update(why=""), "'why' is required"),
        (lambda s: s["episodes"][0].update(expected_s=130), "expected_s"),
        (lambda s: s["episodes"][0].update(cameras={"main": "a", "left_wrist": "b"}),
         "cameras must map"),
        (lambda s: s["episodes"][0].update(
            cameras={"main": "a", "left_wrist": "a", "right_wrist": "c"}), "two roles"),
        (lambda s: s["episodes"][0].update(dataset="nope"), "unknown dataset"),
        (lambda s: s["episodes"][0].update(slug="../x"), "slug must match"),
        (lambda s: s["episodes"][1].update(outcome={}), "known outcome"),
        (lambda s: s["episodes"][1].update(views=[]), "1-3 views"),
        (lambda s: s["episodes"][1].update(modes=["vibes"]), "modes must be"),
        (lambda s: s["episodes"].append(copy.deepcopy(s["episodes"][0])), "duplicate episode"),
        (lambda s: s["episodes"].append({**copy.deepcopy(s["episodes"][0]), "slug": "other"}),
         "selected more than once"),
        (lambda s: s["datasets"].update(spare=dict(s["datasets"]["ds"])), "not used"),
        (lambda s: s.update(version="golden/v0"), "version"),
    ],
)
def test_validator_rejects(mutate, message):
    sel = _minimal()
    mutate(sel)
    with pytest.raises(SelectionError, match=message):
        validate_selection(sel)


def test_validator_lists_every_problem():
    sel = _minimal()
    sel["datasets"]["ds"]["revision"] = "main"
    sel["episodes"][1]["outcome"] = {}
    with pytest.raises(SelectionError) as ei:
        validate_selection(sel)
    assert "40-hex" in str(ei.value) and "known outcome" in str(ei.value)


def test_entry_digest_tracks_entry_and_dataset():
    sel = _minimal()
    e, ds = sel["episodes"][0], sel["datasets"]["ds"]
    d0 = entry_digest(e, ds)
    assert entry_digest(copy.deepcopy(e), copy.deepcopy(ds)) == d0
    assert entry_digest({**e, "episode": 1}, ds) != d0
    assert entry_digest(e, {**ds, "revision": "1" * 40}) != d0
    assert entry_id(e) == "yam/ds/episode_000000"


def test_v2_is_the_human_certification_set_and_keeps_v1s_six():
    from flora_deploy.validate.golden import SELECTIONS, prefix_for

    v2 = validate_selection(load_selection(SELECTIONS["golden/v2"]))
    v1 = load_selection()
    assert v2["version"] == "golden/v2" and str(prefix_for("golden/v2")) == "golden/v2"
    eps = v2["episodes"]
    assert {e["pool"] for e in eps} == {"human"} and len(eps) >= 12  # 10 + spares
    assert all(e["profile"] == "human" and list(e["cameras"]) == ["main"] for e in eps)
    assert all(e["expected_s"] <= 60 for e in eps)
    by1 = {entry_id(e): e for e in v1["episodes"]}
    kept = [e for e in eps if entry_id(e) in by1]
    assert len(kept) == 6
    for e in kept:  # unchanged entries and datasets: the same staged bytes as v1
        assert entry_digest(e, v2["datasets"][e["dataset"]]) == entry_digest(
            by1[entry_id(e)], v1["datasets"][e["dataset"]])
    new = [e for e in eps if entry_id(e) not in by1]
    assert len({e["dataset"] for e in new}) == len(new)  # one per new task
    egodex = {ds["revision"] for k, ds in v2["datasets"].items() if k.startswith("egodex_")}
    assert egodex == {v1["datasets"]["egodex_make_sandwich"]["revision"]}
    ids = {entry_id(e) for e in eps}
    assert len(v2["noise_subset"]) == 5 and set(v2["noise_subset"]) <= ids


def test_noise_subset_must_name_selected_episodes():
    sel = _minimal()
    sel["noise_subset"] = ["yam/ds/nope"]
    with pytest.raises(SelectionError, match="noise_subset"):
        validate_selection(sel)
