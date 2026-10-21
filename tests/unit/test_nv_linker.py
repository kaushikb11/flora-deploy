"""Linker logic against a fake ``batch_utils`` tree (no NVIDIA code)."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import textwrap
from pathlib import Path

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import NV_CODE_CACHE_ENV, snapshot_dirname
from flora_deploy.nv import Family, FamilyConflictError, linker


def _family_module(name: str) -> bool:
    top = name.split(".")[0]
    return (top in linker.FOREIGN_SENTINELS or top.startswith(linker.ALIAS_PREFIX)
            or top.startswith("fakevendor") or top in ("transformers_planted", "lazy_mod"))


@pytest.fixture
def isolated(monkeypatch):
    """Hide any family loaded earlier in this process (e.g. by nvlink tests) and restore it
    afterwards, leaving no trace of the fake families: sys.modules, sys.path, sys.meta_path,
    sys.path_importer_cache and the linker state are all put back."""
    monkeypatch.setattr(linker, "_LOADED", None)
    hidden = {name: mod for name, mod in sys.modules.items() if _family_module(name)}
    for name in hidden:
        del sys.modules[name]
    path, meta_path = list(sys.path), list(sys.meta_path)
    importer_cache = dict(sys.path_importer_cache)
    sys.meta_path[:] = [f for f in meta_path if not isinstance(f, linker._FamilyFinder)]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if _family_module(n)]:
            del sys.modules[name]
        sys.modules.update(hidden)
        sys.path[:] = path
        sys.meta_path[:] = meta_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(importer_cache)


FILES = {
    "__init__.py": "from .api import VERSION\n",
    "api.py": "from .config import VALUE\nVERSION = VALUE + 1\n",
    "config.py": "from pathlib import Path\nROOT = Path(__file__).parent\nVALUE = 41\n",
    "models.py": "from .bootstrap import production as a\nKIND = a.NAME\n",
    "lazy.py": "def get():\n    import lazy_mod\n    return lazy_mod\n",
    "core.py": "from . import engine\nCORE = engine.ENGINE * 2\n",
    "engine.py": "ENGINE = 21\n",
    "vendor/fakevendor_mod.py": "NAME = 'fake-adapter'\ndef double(x):\n    return 2 * x\n",
    "vendor/lazy_mod.py": "WHERE = 'vendor'\n",
}


def bootstrap_src(ok: bool) -> str:
    return textwrap.dedent(
        f"""
        import sys
        from .config import ROOT
        if not {ok!r}:
            raise RuntimeError('Production code changed: x')
        sys.path.insert(0, str(ROOT / 'vendor'))
        import fakevendor_mod
        production = fakevendor_mod
        """
    )


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def write_manifests(code: Path) -> None:
    """(Re)write vendor/MANIFEST.json and CODE_MANIFEST.json for the files now on disk."""
    pkg = code / "batch_utils"
    vendor = {p.relative_to(pkg / "vendor").as_posix(): _sha(p.read_bytes())
              for p in sorted((pkg / "vendor").rglob("*.py"))}
    (pkg / "vendor" / "MANIFEST.json").write_text(json.dumps(vendor))
    entries = {
        f"batch_utils/{rel}": {"bytes": len(data), "sha256": _sha(data)}
        for rel, data in (
            (p.relative_to(pkg).as_posix(), p.read_bytes()) for p in sorted(pkg.rglob("*"))
            if p.is_file() and p.name != "CODE_MANIFEST.json" and "__pycache__" not in p.parts
        )
    }
    entries["infer.py"] = {"bytes": 1, "sha256": "0" * 64}  # outside batch_utils: skipped
    (pkg / "CODE_MANIFEST.json").write_text(json.dumps({"files": entries}))


def make_tree(root: Path, name: str, *, bootstrap_ok=True, manifest=True) -> Path:
    code = root / name
    pkg = code / "batch_utils"
    (pkg / "vendor").mkdir(parents=True)
    for rel, text in {**FILES, "bootstrap.py": bootstrap_src(bootstrap_ok)}.items():
        (pkg / rel).write_text(text)
    if manifest:
        write_manifests(code)
    return code


def pins_for(code: Path) -> dict[str, str]:
    """What ``RepoPin.code_digests`` would pin for this fake tree, as it is now."""
    pkg = code / "batch_utils"
    out = {"code_tree_sha256": linker.code_tree_sha256(code)}
    for key, rel in (("code_manifest_sha256", "CODE_MANIFEST.json"),
                     ("vendor_manifest_sha256", "vendor/MANIFEST.json")):
        if (pkg / rel).is_file():
            out[key] = _sha((pkg / rel).read_bytes())
    return out


def load(family, code, **kw):
    return linker.load_family(family, code, pinned_digests=pins_for(code), **kw)


def test_alias_and_relative_imports(isolated, tmp_path):
    code = make_tree(tmp_path, "acme--Fake@0123456789abcdef0123")
    nv = load(Family.SEGCAP, code)
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
    nv = load(Family.REWARDER, make_tree(tmp_path, "r@abcdef0123456789"))
    assert nv.core.CORE == 42
    with pytest.raises(AttributeError, match="no module 'pipeline'"):
        nv.pipeline  # noqa: B018
    with pytest.raises(AttributeError):
        nv.vendor("anything")


def test_second_family_conflicts(isolated, tmp_path):
    load(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    with pytest.raises(FamilyConflictError, match="already linked segcap"):
        load(Family.REWARDER, make_tree(tmp_path, "b@11111111bbbbbbbb"))


def test_same_family_other_dir_conflicts(isolated, tmp_path):
    load(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    with pytest.raises(FamilyConflictError):
        load(Family.SEGCAP, make_tree(tmp_path, "c@22222222cccccccc"))


@pytest.mark.parametrize("name", ["batch_utils", "adapter", "flora_runtime"])
def test_foreign_module_conflicts(isolated, tmp_path, monkeypatch, name):
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    module = type(sys)(name)
    module.__file__ = str(foreign / f"{name}.py")
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(FamilyConflictError, match=name):
        load(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    assert linker.loaded_family() is None


def test_other_alias_conflicts(isolated, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "flora_nv_rewarder_99999999", type(sys)("x"))
    with pytest.raises(FamilyConflictError, match="flora_nv_rewarder_99999999"):
        load(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))


def test_failed_bootstrap_leaves_nothing_behind(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa", bootstrap_ok=False)
    path_before, meta_before = list(sys.path), list(sys.meta_path)
    with pytest.raises(RuntimeError, match="Production code changed"):
        load(Family.SEGCAP, code)
    assert not [m for m in sys.modules if m.startswith("flora_nv_")]
    assert sys.path == path_before
    assert sys.meta_path == meta_before
    assert linker.loaded_family() is None


def test_rollback_keeps_unrelated_modules(isolated, tmp_path, monkeypatch):
    """M5: a failed load removes only the family's modules, never stdlib/third-party ones that
    happened to be imported for the first time during the load, nor concurrent sys.path edits."""
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    pkg = code / "batch_utils"
    (pkg / "bootstrap.py").write_text(textwrap.dedent(
        """
        import sys
        import colorsys  # stdlib, first imported here
        from .config import ROOT
        sys.path.insert(0, str(ROOT / 'vendor'))
        sys.path.append('/somewhere/else')  # someone else's edit
        import fakevendor_mod
        raise RuntimeError('Production code changed: late')
        """
    ))
    write_manifests(code)
    monkeypatch.delitem(sys.modules, "colorsys", raising=False)
    with pytest.raises(RuntimeError, match="late"):
        load(Family.SEGCAP, code)
    assert "colorsys" in sys.modules
    assert "fakevendor_mod" not in sys.modules
    assert not [m for m in sys.modules if m.startswith("flora_nv_")]
    assert "/somewhere/else" in sys.path
    assert not [p for p in sys.path if str(code) in p]


def test_code_manifest(tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    assert linker.verify_code_manifest(code) == 11
    (code / "batch_utils" / "engine.py").write_text("ENGINE = 22\n")
    with pytest.raises(linker.CodeManifestError, match="engine.py"):
        linker.verify_code_manifest(code)
    (code / "batch_utils" / "engine.py").unlink()
    with pytest.raises(linker.CodeManifestError, match="missing"):
        linker.verify_code_manifest(code)
    assert linker.verify_code_manifest(make_tree(tmp_path, "b@1", manifest=False)) == 0


def test_manifest_entry_without_sha(tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    path = code / "batch_utils" / "CODE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["batch_utils/api.py"]["sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(linker.CodeManifestError, match="no sha256"):
        linker.verify_code_manifest(code)


def test_tampered_manifest_blocks_load(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    pinned = pins_for(code)
    (code / "batch_utils" / "api.py").write_text("VERSION = 0\n")
    with pytest.raises(linker.CodeManifestError):
        linker.load_family(Family.SEGCAP, code, pinned_digests=pinned)
    assert linker.loaded_family() is None


def test_rewritten_manifests_are_refused(isolated, tmp_path):
    """Editing a file *and* its manifest entries changes the manifests' pinned sha256."""
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    pinned = pins_for(code)
    (code / "batch_utils" / "vendor" / "fakevendor_mod.py").write_text("NAME = 'evil'\n")
    write_manifests(code)
    with pytest.raises(linker.CodeManifestError, match="CODE_MANIFEST.json"):
        linker.load_family(Family.SEGCAP, code, pinned_digests=pinned)
    with pytest.raises(linker.CodeManifestError, match="vendor/MANIFEST.json"):
        linker.load_family(Family.SEGCAP, code, pinned_digests=dict(
            pinned, code_manifest_sha256=pins_for(code)["code_manifest_sha256"]))
    assert linker.loaded_family() is None


@pytest.mark.parametrize("planted", [
    "vendor/transformers_planted.py", "vendor/sub/__init__.py", "evil.pth",
    "vendor/native.cpython-312-darwin.so", "vendor/legacy.pyc", "notes.txt",
])
def test_unlisted_files_are_refused(isolated, tmp_path, planted):
    """M3: a file the code manifest does not list, anywhere under batch_utils/ (which holds every
    sys.path entry bootstrap adds), is refused, even though all listed files are intact."""
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    pinned = pins_for(code)
    target = code / "batch_utils" / planted
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("HIJACKED = True\n")
    with pytest.raises(linker.CodeManifestError, match="not listed"):
        linker.load_family(Family.SEGCAP, code, pinned_digests=pinned)
    assert linker.loaded_family() is None and "transformers_planted" not in sys.modules


def test_pycache_is_ignored_and_never_written(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    pinned = pins_for(code)
    stale = code / "batch_utils" / "vendor" / "__pycache__"
    stale.mkdir()
    (stale / "fakevendor_mod.cpython-312.pyc").write_bytes(b"not bytecode")
    nv = linker.load_family(Family.SEGCAP, code, pinned_digests=pinned)
    assert nv.production.NAME == "fake-adapter"  # compiled from the verified source
    written = [p for p in code.rglob("*.pyc") if p.parent != stale]
    assert written == []
    assert [p.name for p in stale.iterdir()] == ["fakevendor_mod.cpython-312.pyc"]


def test_lazy_vendor_import_resolves_to_snapshot(isolated, tmp_path):
    """L1: a module NVIDIA imports lazily is resolved from the vendor dir even when a directory
    ahead of it on sys.path has a module of the same name."""
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    nv = load(Family.SEGCAP, code)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "lazy_mod.py").write_text("WHERE = 'shadow'\n")
    sys.path.insert(0, str(shadow))
    mod = nv.module("lazy").get()
    assert mod.WHERE == "vendor"
    assert Path(mod.__file__).resolve().is_relative_to(code.resolve())
    assert "lazy_mod" in nv.audit()


def test_audit_rejects_foreign_module(isolated, tmp_path, monkeypatch):
    nv = load(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    foreign = type(sys)("lazy_mod")
    foreign.__file__ = str(tmp_path / "elsewhere" / "lazy_mod.py")
    monkeypatch.setitem(sys.modules, "lazy_mod", foreign)
    with pytest.raises(linker.CodeProvenanceError, match="outside"):
        nv.audit()


def test_file_planted_after_load_is_not_imported(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    load(Family.SEGCAP, code)
    (code / "batch_utils" / "vendor" / "late_planted.py").write_text("X = 1\n")
    with pytest.raises(linker.CodeProvenanceError, match="not a verified"):
        __import__("late_planted")
    sys.modules.pop("late_planted", None)


def test_file_changed_after_verification_is_not_executed(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    nv = load(Family.SEGCAP, code)
    (code / "batch_utils" / "engine.py").write_text("ENGINE = 666\n")
    with pytest.raises(linker.CodeProvenanceError, match="changed after"):
        nv.module("engine")


def test_rewarder_tree_digest(isolated, tmp_path):
    code = make_tree(tmp_path, "r@abcdef0123456789", manifest=False)
    pinned = pins_for(code)
    assert linker.verify_family_code(Family.REWARDER, code, pinned).keys() == {
        f"batch_utils/{rel}" for rel in [*FILES, "bootstrap.py"]
    }
    (code / "batch_utils" / "extra.py").write_text("")
    with pytest.raises(linker.CodeManifestError, match="code_tree_sha256"):
        linker.load_family(Family.REWARDER, code, pinned_digests=pinned)


def test_missing_pins_fail_closed(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    with pytest.raises(linker.CodeManifestError, match="no pinned"):
        linker.load_family(Family.SEGCAP, code, pinned_digests={})
    # a directory not named after a pin is checked against the family's default pin
    with pytest.raises(linker.CodeManifestError, match="pinned"):
        linker.load_family(Family.SEGCAP, code)


def test_every_code_pin_has_its_digests():
    for family, keys in linker.FAMILY_PINS.items():
        for key in keys:
            digests = dict(PINS_BY_KEY[key].code_digests)
            assert set(digests) == set(linker.REQUIRED_DIGESTS[family]), key
            assert all(len(v) == 64 for v in digests.values())
    assert linker.pin_for_code_dir(
        Family.SEGCAP, Path("/x") / snapshot_dirname(PINS_BY_KEY["captioner_code"])
    ).key == "captioner_code"
    assert linker.pin_for_code_dir(Family.SEGCAP, Path("/x/other")).key == "tempseg_code"


def test_verify_requested_later_runs(isolated, tmp_path):
    code = make_tree(tmp_path, "a@00000000aaaaaaaa")
    pinned = pins_for(code)
    nv = linker.load_family(Family.SEGCAP, code, verify_manifest=False)
    assert not nv.verified
    (code / "batch_utils" / "notes.txt").write_text("x")
    with pytest.raises(linker.CodeManifestError):
        linker.load_family(Family.SEGCAP, code, pinned_digests=pinned)
    (code / "batch_utils" / "notes.txt").unlink()
    assert linker.load_family(Family.SEGCAP, code, pinned_digests=pinned).verified


def test_nvref_pickles_and_resolves(isolated, tmp_path):
    nv = load(Family.SEGCAP, make_tree(tmp_path, "a@00000000aaaaaaaa"))
    ref = nv.ref_for(nv.production.double)
    assert ref == linker.NvRef(Family.SEGCAP, "fakevendor_mod", "double", vendor=True)
    assert pickle.loads(pickle.dumps(ref))(21) == 42
    api_ref = nv.ref("api", "VERSION")
    assert api_ref.resolve() == 42
    with pytest.raises(ValueError):
        nv.ref_for(json.dumps)
    kwargs = nv.pool_initializer()
    assert kwargs["initializer"] is linker.worker_init
    assert kwargs["initargs"][:2] == ("segcap", str(nv.code_dir))
    assert pickle.loads(pickle.dumps(kwargs["initargs"])) == kwargs["initargs"]


def test_nvref_without_linked_family(isolated):
    with pytest.raises(FamilyConflictError, match="worker_init"):
        linker.NvRef(Family.SEGCAP, "media", "frames_for").resolve()


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
