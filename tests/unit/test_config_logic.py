"""config/logic.py: every module on the output path is classified, and the digest is sound."""

from __future__ import annotations

import pytest

from flora_deploy.config import logic
from flora_deploy.config.presets import PRESETS


def _modules_in(rel_dir: str) -> set[str]:
    base = logic.package_root() / rel_dir if rel_dir else logic.package_root()
    prefix = f"{rel_dir}/" if rel_dir else ""
    return {prefix + p.name for p in base.glob("*.py")}


def _output() -> set[str]:
    return {m for mods in logic.OUTPUT_MODULES.values() for m in mods}


@pytest.mark.parametrize("rel_dir", logic.CLASSIFIED_DIRS)
def test_every_module_is_classified(rel_dir):
    """A new module under nv/, accel/, stages/, media/ (or the package root, config/) must be
    declared output-affecting or not in config/logic.py before the tests pass."""
    unclassified = sorted(_modules_in(rel_dir) - _output() - set(logic.NON_OUTPUT_MODULES))
    assert not unclassified, (f"classify {unclassified} in config/logic.py (OUTPUT_MODULES or "
                              "NON_OUTPUT_MODULES)")


def test_classified_dirs_cover_the_output_path():
    for d in ("nv", "accel", "stages", "media"):
        assert d in logic.CLASSIFIED_DIRS
    subdirs = {p.relative_to(logic.package_root()).as_posix()
               for p in (logic.package_root() / "media").rglob("*") if p.is_dir()
               and p.name != "__pycache__"}
    assert subdirs <= set(logic.CLASSIFIED_DIRS)


def test_lists_are_consistent():
    out, non = _output(), set(logic.NON_OUTPUT_MODULES)
    assert not out & non, out & non
    root = logic.package_root()
    for rel in out | non:
        assert (root / rel).is_file(), rel
    for group, mods in logic.OUTPUT_MODULES.items():
        assert len(set(mods)) == len(mods), group
    assert set(logic.KIND_GROUPS.values()) | {"common"} == set(logic.OUTPUT_MODULES)


def test_every_shard_kind_has_a_logic_group():
    kinds = {k for spec in PRESETS.values() for k in spec.pools}
    assert kinds <= set(logic.KIND_GROUPS)
    with pytest.raises(KeyError, match="no logic group"):
        logic.modules_for("segment.v2")


def test_named_output_modules_are_hashed():
    """The modules review A1 names, for the kinds that run them."""
    seg = set(logic.modules_for("segcap"))
    for rel in ("nv/capture.py", "nv/session.py", "nv/segment_glue.py", "nv/budget.py",
                "media/index.py", "media/rangedecode.py", "media/cuts.py",
                "stages/reference.py", "records.py", "config/hashing.py"):
        assert rel in seg, rel
    accel = {p for p in _modules_in("accel")} - set(logic.NON_OUTPUT_MODULES)
    assert accel <= seg and accel <= set(logic.modules_for("caption"))
    assert "stages/reward.py" in logic.modules_for("reward.dense")
    assert "stages/reward.py" not in seg
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
