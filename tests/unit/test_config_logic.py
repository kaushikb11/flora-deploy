"""config/logic.py: every module of the package is classified, and the digest is sound."""

from __future__ import annotations

import pytest

from flora_deploy.config import logic
from flora_deploy.config.presets import PRESETS


def _output() -> set[str]:
    return {m for mods in logic.OUTPUT_MODULES.values() for m in mods}


def _package_dirs() -> list[str]:
    root = logic.package_root()
    return sorted({p.parent.relative_to(root).as_posix()
                   for p in root.rglob("*.py") if "__pycache__" not in p.parts})


@pytest.mark.parametrize("rel_dir", _package_dirs())
def test_every_module_is_classified(rel_dir):
    """Review 4 M2: walk **every** package directory (found on disk, not listed by hand), so a
    new module anywhere must be declared output-affecting or not before the tests pass."""
    prefix = "" if rel_dir == "." else f"{rel_dir}/"
    mods = {m for m in logic.package_modules()
            if m.startswith(prefix) and "/" not in m[len(prefix):]}
    assert mods
    unclassified = sorted(mods - _output() - set(logic.NON_OUTPUT_MODULES))
    assert not unclassified, (f"classify {unclassified} in config/logic.py (OUTPUT_MODULES or "
                              "NON_OUTPUT_MODULES)")


def test_walk_covers_every_module():
    assert logic.package_modules() <= _output() | set(logic.NON_OUTPUT_MODULES)
    dirs = _package_dirs()
    for d in ("workers_impl", "jobs", "jobs/stores", "media/sources", "modal_app", "nv",
              "accel", "stages", "export", "prep", "observe", "validate", "cli", "planner"):
        assert d in dirs, d


def test_lists_are_consistent():
    out, non = _output(), set(logic.NON_OUTPUT_MODULES)
    assert not out & non, out & non
    root = logic.package_root()
    for rel in out | non:
        assert (root / rel).is_file(), rel
    for group, mods in logic.OUTPUT_MODULES.items():
        assert len(set(mods)) == len(mods), group
    assert len(set(logic.NON_OUTPUT_MODULES)) == len(logic.NON_OUTPUT_MODULES)
    assert set(logic.KIND_GROUPS.values()) | {"common"} == set(logic.OUTPUT_MODULES)


def test_every_shard_kind_has_a_logic_group():
    kinds = {k for spec in PRESETS.values() for k in spec.pools}
    assert kinds <= set(logic.KIND_GROUPS)
    with pytest.raises(KeyError, match="no logic group"):
        logic.modules_for("segment.v2")


def test_named_output_modules_are_hashed():
    """The modules reviews 3 (A1) and 4 (M2) name, for the kinds that run them."""
    seg = set(logic.modules_for("segcap"))
    for rel in ("nv/capture.py", "nv/session.py", "nv/segment_glue.py", "nv/budget.py",
                "media/index.py", "media/rangedecode.py", "media/cuts.py",
                "stages/reference.py", "records.py", "config/hashing.py",
                "jobs/indexing.py", "media/sources/lerobot_v3.py"):
        assert rel in seg, rel
    accel = {m for m in logic.package_modules() if m.startswith("accel/")} - set(
        logic.NON_OUTPUT_MODULES)
    assert accel <= seg and accel <= set(logic.modules_for("caption"))
    for kind in ("reward.terminal", "reward.dense"):
        mods = set(logic.modules_for(kind))
        for rel in ("stages/reward.py", "workers_impl/reward.py", "modal_app/workers_app.py",
                    "jobs/indexing.py", "jobs/manifest.py", "media/sources/mp4_dir.py"):
            assert rel in mods, (kind, rel)
    assert "stages/reward.py" not in seg
    assert "workers_impl/reward.py" not in seg
    assert "cli/init.py" not in set(logic.modules_for("segcap")) | set(
        logic.modules_for("reward.dense"))


def test_digest_is_deterministic_and_path_bound(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    d = logic.source_digest(("a.py", "b.py"), tmp_path)
    assert d == logic.source_digest(("b.py", "a.py"), tmp_path)
    (tmp_path / "b.py").rename(tmp_path / "c.py")
    (tmp_path / "b.py").write_text("y = 2\n")
    assert logic.source_digest(("a.py", "c.py"), tmp_path) != d  # a rename changes it
    # moving bytes across the file boundary changes it too
    (tmp_path / "a.py").write_text("x = 1\ny")
    (tmp_path / "b.py").write_text(" = 2\n")
    assert logic.source_digest(("a.py", "b.py"), tmp_path) != d
