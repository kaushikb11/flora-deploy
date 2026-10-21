"""Linker logic against a fake ``batch_utils`` tree (no NVIDIA code)."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import NV_CODE_CACHE_ENV, snapshot_dirname
from flora_deploy.nv import Family, FamilyConflictError, linker


@pytest.fixture
def isolated(monkeypatch):
    """Hide any family loaded earlier in this process; restore it afterwards."""
    monkeypatch.setattr(linker, "_LOADED", None)
    for name in list(sys.modules):
        top = name.split(".")[0]
        if top in linker.FOREIGN_SENTINELS or top.startswith(linker.ALIAS_PREFIX):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name in list(sys.modules):
        if name.startswith(linker.ALIAS_PREFIX + "fake") or name.startswith("fakevendor"):
            del sys.modules[name]


def make_tree(root: Path, name: str, *, bootstrap_ok=True, manifest=True) -> Path:
    code = root / name
    pkg = code / "batch_utils"
    (pkg / "vendor").mkdir(parents=True)
    files = {
        "__init__.py": "from .api import VERSION\n",
        "api.py": "from .config import VALUE\nVERSION = VALUE + 1\n",
        "config.py": "from pathlib import Path\nROOT = Path(__file__).parent\nVALUE = 41\n",
        "bootstrap.py": textwrap.dedent(
            f"""
            import sys
            from .config import ROOT
            if not {bootstrap_ok!r}:
                raise RuntimeError('Production code changed: x')
            sys.path.insert(0, str(ROOT / 'vendor'))
            import fakevendor_mod
            production = fakevendor_mod
            """
        ),
        "models.py": "from .bootstrap import production as a\nKIND = a.NAME\n",
        "core.py": "from . import engine\nCORE = engine.ENGINE * 2\n",
        "engine.py": "ENGINE = 21\n",
        "vendor/fakevendor_mod.py": "NAME = 'fake-adapter'\n",
    }
    for rel, text in files.items():
        (pkg / rel).write_text(text)
    if manifest:
        entries = {
            f"batch_utils/{rel}": {
                "bytes": len(text.encode()),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for rel, text in files.items()
        }
        entries["infer.py"] = {"bytes": 1, "sha256": "0" * 64}  # outside batch_utils: skipped
        (pkg / "CODE_MANIFEST.json").write_text(json.dumps({"files": entries}))
    return code


def test_alias_and_relative_imports(isolated, tmp_path):
    code = make_tree(tmp_path, "acme--Fake@0123456789abcdef0123")
    nv = linker.load_family(Family.SEGCAP, code)
    assert nv.alias == "flora_nv_segcap_01234567"
    assert nv.package.VERSION == 42
    assert nv.config.__name__ == "flora_nv_segcap_01234567.config"
    assert nv.production.NAME == "fake-adapter"
    assert nv.models.KIND == "fake-adapter"
    assert "batch_utils" not in sys.modules
    assert linker.load_family("segcap", code) is nv
    assert linker.loaded_family() is nv


def test_alias_without_revision_dir_name(tmp_path):
    a = linker.alias_for(Family.REWARDER, tmp_path / "plain")
    assert a.startswith("flora_nv_rewarder_") and len(a) == len("flora_nv_rewarder_") + 8


def test_unknown_module_attribute(isolated, tmp_path):
    nv = linker.load_family(Family.REWARDER, make_tree(tmp_path, "r@abcdef0123456789"))
    assert nv.core.CORE == 42
    with pytest.raises(AttributeError, match="no module 'pipeline'"):
        nv.pipeline  # noqa: B018
    with pytest.raises(AttributeError):
        nv.vendor("anything")


def test_second_family_conflicts(isolated, tmp_path):
    linker.load_family(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    with pytest.raises(FamilyConflictError, match="already linked segcap"):
        linker.load_family(Family.REWARDER, make_tree(tmp_path, "b@11111111bbbbbbbb"))


def test_same_family_other_dir_conflicts(isolated, tmp_path):
    linker.load_family(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    with pytest.raises(FamilyConflictError):
        linker.load_family(Family.SEGCAP, make_tree(tmp_path, "c@22222222cccccccc"))


@pytest.mark.parametrize("name", ["batch_utils", "adapter", "flora_runtime"])
def test_foreign_module_conflicts(isolated, tmp_path, monkeypatch, name):
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    module = type(sys)(name)
    module.__file__ = str(foreign / f"{name}.py")
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(FamilyConflictError, match=name):
        linker.load_family(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    assert linker.loaded_family() is None


def test_other_alias_conflicts(isolated, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "flora_nv_rewarder_99999999", type(sys)("x"))
    with pytest.raises(FamilyConflictError, match="flora_nv_rewarder_99999999"):
        linker.load_family(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))


def test_failed_bootstrap_leaves_nothing_behind(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa", bootstrap_ok=False)
    path_before = list(sys.path)
    with pytest.raises(RuntimeError, match="Production code changed"):
        linker.load_family(Family.SEGCAP, code)
    assert not [m for m in sys.modules if m.startswith("flora_nv_")]
    assert sys.path == path_before
    assert linker.loaded_family() is None


def test_code_manifest(tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    assert linker.verify_code_manifest(code) == 8
    (code / "batch_utils" / "engine.py").write_text("ENGINE = 22\n")
    with pytest.raises(linker.CodeManifestError, match="engine.py"):
        linker.verify_code_manifest(code)
    (code / "batch_utils" / "engine.py").unlink()
    with pytest.raises(linker.CodeManifestError, match="missing"):
        linker.verify_code_manifest(code)
    assert linker.verify_code_manifest(make_tree(tmp_path, "b@1", manifest=False)) == 0


def test_tampered_manifest_blocks_load(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    (code / "batch_utils" / "api.py").write_text("VERSION = 0\n")
    with pytest.raises(linker.CodeManifestError):
        linker.load_family(Family.SEGCAP, code)
    assert linker.loaded_family() is None


def test_missing_package(isolated, tmp_path):
    with pytest.raises(FileNotFoundError):
        linker.load_family(Family.SEGCAP, tmp_path)


def test_code_dir_for(monkeypatch, tmp_path):
    monkeypatch.setenv(NV_CODE_CACHE_ENV, str(tmp_path))
    assert linker.code_dir_for(Family.SEGCAP) == tmp_path / snapshot_dirname(
        PINS_BY_KEY["tempseg_code"]
    )
    assert linker.code_dir_for(Family.REWARDER, "/weights/snapshots") == Path(
        "/weights/snapshots"
    ) / snapshot_dirname(PINS_BY_KEY["rewarder"])
