"""Import one NVIDIA code family into this process under a unique module name (PLAN.md D1).

NVIDIA ships three unrelated packages that are all called ``batch_utils``. The TempSeg /
Captioner one runs ``bootstrap.initialize()`` at import time. That call SHA-checks
``vendor/MANIFEST.json``, puts ``vendor``, ``vendor/support`` and ``vendor/production_adapter`` at
the front of ``sys.path``, and imports the top-level module ``adapter``, which in turn adds
``.../core`` so ``flora_runtime`` becomes importable. We never edit or re-implement that. We load
the package from the user's snapshot under an alias such as ``flora_nv_segcap_78a0e4ac``, so its
relative imports work, and let NVIDIA's bootstrap run untouched.

Isolation rules enforced here:

- One family per process. Loading a second family, or the same family from a different
  directory, raises ``FamilyConflictError``.
- The top-level names NVIDIA's bootstrap claims (``adapter``, ``flora_runtime``, ...) and the bare
  name ``batch_utils`` must not already belong to someone else. If they did, NVIDIA's
  ``import adapter`` would silently bind to foreign code, so we refuse.

Integrity (before any NVIDIA code runs; ``verify_family_code``):

- SEGCAP: the sha256 of ``batch_utils/CODE_MANIFEST.json`` and of
  ``batch_utils/vendor/MANIFEST.json`` must equal the values pinned in ``config/pins.py``
  (``RepoPin.code_digests``). Every
  ``batch_utils/`` entry of the code manifest is then checked, and **any file under
  ``batch_utils/`` that the manifest does not list is refused** (``__pycache__`` aside). The
  paths bootstrap puts on ``sys.path`` are all inside ``batch_utils/vendor``, so a planted
  ``vendor/support/transformers.py`` (or a ``.pth`` / ``.so``) cannot shadow a real package.
- REWARDER: NVIDIA ships no manifest, so we pin ``code_tree_sha256``: the sha256 of the sorted
  ``"<sha256>  <path>\\n"`` lines of every file under ``batch_utils/`` (``code_tree_lines``).
  Extra, missing or changed files all change it.

Import-time provenance (``_FamilyFinder``, installed at the front of ``sys.meta_path``):

- Every module of the family (the alias package and its submodules, and the top-level vendor
  modules that live on the ``sys.path`` entries bootstrap added) is resolved from the snapshot and
  must be one of the files verified above. Its source is hashed again when it is executed, and
  the code is compiled from exactly those bytes. This covers modules NVIDIA imports lazily much
  later (``plans``, ``verify_gdn``, ``production_gdn_fix``, ``tempseg_fp32_loader``, ...): a
  directory placed ahead of the vendor dirs on ``sys.path`` cannot capture them.
- No bytecode is read from or written to the snapshot. The family's loader always compiles from
  the verified source and never writes ``__pycache__``. This is stronger than toggling
  ``sys.dont_write_bytecode`` around the import (which is process-global and racy, does not cover
  lazy imports, and would still *read* a planted, timestamp-valid ``.pyc``). Snapshots on a
  shared or read-only Volume are therefore never written to.

Process pools (PLAN.md D12): the alias exists only in the process that linked it. Pass
``worker_init`` as the pool initializer (``NvFamily.pool_initializer()`` returns the arguments),
and submit NVIDIA functions as ``NvRef`` (family + module + qualname). The ref is resolved in the
worker after the worker has re-linked the family with all of the checks above.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from flora_deploy.config.pins import PINS_BY_KEY, RepoPin
from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname
from flora_deploy.nv import FAMILY_CODE_PIN, Family, FamilyConflictError

PACKAGE = "batch_utils"
ALIAS_PREFIX = "flora_nv_"
CODE_MANIFEST = "CODE_MANIFEST.json"
VENDOR_MANIFEST = "vendor/MANIFEST.json"

# Top-level module names that SEGCAP's bootstrap (and adapter.py's own sys.path insert) make
# importable. Checked before linking: if any is already imported from elsewhere, NVIDIA's code
# would bind to it. After linking, the finder derives the real list from the snapshot.
SEGCAP_TOPLEVEL = (
    "adapter",
    "flora_runtime",
    "flora_unified",
    "plans",
    "composers",
    "canonical_rgb",
    "u2_contract",
    "pts_input_adapter",
    "robot_caption_geometry",
    "tempseg_fp32_loader",
    "verify_gdn",
    "production_gdn_fix",
)
FOREIGN_SENTINELS = ("batch_utils", *SEGCAP_TOPLEVEL)

# Submodules exposed as attributes on the handle, per family. Imported lazily on first access.
FAMILY_MODULES: dict[Family, tuple[str, ...]] = {
    Family.SEGCAP: ("config", "bootstrap", "models", "media", "preprocessing", "pipeline", "api"),
    Family.REWARDER: ("core", "engine", "terminal_reward", "dense_reward", "rubric_reward"),
}

# Code pins per family; a snapshot directory named after one of them uses that pin's digests.
FAMILY_PINS: dict[Family, tuple[str, ...]] = {
    Family.SEGCAP: ("tempseg_code", "captioner_code"),
    Family.REWARDER: ("rewarder",),
}
REQUIRED_DIGESTS: dict[Family, tuple[str, ...]] = {
    Family.SEGCAP: ("code_manifest_sha256", "vendor_manifest_sha256"),
    Family.REWARDER: ("code_tree_sha256",),
}


class CodeManifestError(RuntimeError):
    """NVIDIA's code on disk does not match the pinned manifests (or digest)."""


class CodeProvenanceError(ImportError):
    """A module of the linked family resolved to a file that was not verified."""


# --- integrity ----------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_files(code_dir: Path) -> list[str]:
    """Every regular file under ``batch_utils/`` (``__pycache__`` excluded), as sorted paths
    relative to ``code_dir``. Symlinks are refused."""
    root = Path(code_dir) / PACKAGE
    out = []
    for path in root.rglob("*"):
        rel = path.relative_to(code_dir)
        if "__pycache__" in rel.parts:
            continue
        if path.is_symlink():
            raise CodeManifestError(f"{rel.as_posix()}: symlinks are not allowed in NVIDIA code")
        if path.is_file():
            out.append(rel.as_posix())
    return sorted(out)


def code_tree_lines(code_dir: Path) -> str:
    """``"<sha256>  <path>\\n"`` for every file of ``package_files``, sorted by path."""
    code_dir = Path(code_dir)
    return "".join(f"{_sha256(code_dir / rel)}  {rel}\n" for rel in package_files(code_dir))


def code_tree_sha256(code_dir: Path) -> str:
    return hashlib.sha256(code_tree_lines(code_dir).encode()).hexdigest()


def _code_manifest_entries(code_dir: Path) -> dict[str, dict] | None:
    path = code_dir / PACKAGE / CODE_MANIFEST
    if not path.is_file():
        return None
    files = json.loads(path.read_text()).get("files")
    if not isinstance(files, dict):
        raise CodeManifestError(f"{path}: no 'files' mapping")
    return {rel: entry for rel, entry in files.items() if rel.startswith(PACKAGE + "/")}


def _check_manifest_files(code_dir: Path, entries: Mapping[str, dict]) -> dict[str, str]:
    """Check each manifest entry against the file on disk; return ``{rel: sha256}``."""
    digests = {}
    for rel, entry in entries.items():
        target = code_dir / rel
        if not target.resolve().is_relative_to(code_dir.resolve()):
            raise CodeManifestError(f"{rel}: path escapes the snapshot")
        if not target.is_file():
            raise CodeManifestError(f"{rel}: missing")
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise CodeManifestError(f"{rel}: manifest entry has no sha256")
        data = target.read_bytes()
        if "bytes" in entry and len(data) != entry["bytes"]:
            raise CodeManifestError(f"{rel}: size {len(data)} != {entry['bytes']}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise CodeManifestError(f"{rel}: sha256 mismatch")
        digests[rel] = digest
    return digests


def verify_code_manifest(code_dir: Path) -> int:
    """Check every ``batch_utils/`` entry of NVIDIA's ``CODE_MANIFEST.json``; return the count.

    Entries outside ``batch_utils/`` (e.g. the repo-root ``infer.py``) are not part of the code
    pin we fetch and are skipped. Returns 0 when the family ships no manifest. This checks the
    listed files only; ``verify_family_code`` adds the pinned manifest digests and refuses
    unlisted files.
    """
    code_dir = Path(code_dir)
    entries = _code_manifest_entries(code_dir)
    return 0 if entries is None else len(_check_manifest_files(code_dir, entries))


def pin_for_code_dir(family: Family, code_dir: Path) -> RepoPin:
    """The code pin whose snapshot directory name ``code_dir`` has; else the family's default."""
    for key in FAMILY_PINS[family]:
        if snapshot_dirname(PINS_BY_KEY[key]) == Path(code_dir).name:
            return PINS_BY_KEY[key]
    return PINS_BY_KEY[FAMILY_CODE_PIN[family]]


def pinned_digests_for(family: Family, code_dir: Path) -> dict[str, str]:
    return dict(pin_for_code_dir(family, code_dir).code_digests)


def verify_family_code(
    family: Family, code_dir: Path, pinned: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Verify ``family``'s code in ``code_dir`` against ``pinned`` digests (default: from
    ``pins.py``). Returns ``{path relative to code_dir: sha256}`` of every verified file."""
    family = Family(family)
    code_dir = Path(code_dir)
    pinned = pinned_digests_for(family, code_dir) if pinned is None else dict(pinned)
    missing = [k for k in REQUIRED_DIGESTS[family] if not pinned.get(k)]
    if missing:
        raise CodeManifestError(f"no pinned {missing} for {family.value} code in {code_dir}")
    present = package_files(code_dir)
    if family is Family.REWARDER:
        lines = code_tree_lines(code_dir)
        if hashlib.sha256(lines.encode()).hexdigest() != pinned["code_tree_sha256"]:
            raise CodeManifestError(
                f"{code_dir}: Rewarder code tree differs from the pinned code_tree_sha256 "
                f"({len(present)} files under {PACKAGE}/)"
            )
        return {rel: digest for digest, rel in (ln.split("  ", 1) for ln in lines.splitlines())}

    pkg = code_dir / PACKAGE
    for name, key in ((CODE_MANIFEST, "code_manifest_sha256"),
                      (VENDOR_MANIFEST, "vendor_manifest_sha256")):
        path = pkg / name
        if not path.is_file():
            raise CodeManifestError(f"{PACKAGE}/{name}: missing")
        if _sha256(path) != pinned[key]:
            raise CodeManifestError(f"{PACKAGE}/{name}: sha256 differs from the pinned {key}")
    entries = _code_manifest_entries(code_dir) or {}
    digests = _check_manifest_files(code_dir, entries)
    vendor = json.loads((pkg / VENDOR_MANIFEST).read_text())
    for rel, digest in vendor.items():
        if digests.get(f"{PACKAGE}/vendor/{rel}") != digest:
            raise CodeManifestError(f"vendor manifest entry {rel!r} disagrees with {CODE_MANIFEST}")
    unlisted = sorted(set(present) - set(digests) - {f"{PACKAGE}/{CODE_MANIFEST}"})
    if unlisted:
        raise CodeManifestError(f"files not listed in {CODE_MANIFEST}: {unlisted}")
    digests[f"{PACKAGE}/{CODE_MANIFEST}"] = _sha256(pkg / CODE_MANIFEST)
    return digests


# --- import-time provenance ---------------------------------------------------------------------


class _VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile from the verified source bytes; never read or write bytecode."""

    def __init__(self, fullname: str, path: str, sha256: str | None):
        super().__init__(fullname, path)
        self._sha256 = sha256

    def get_code(self, fullname):
        data = self.get_data(self.path)
        if self._sha256 is not None and hashlib.sha256(data).hexdigest() != self._sha256:
            raise CodeProvenanceError(f"{self.path} changed after it was verified", name=fullname)
        return self.source_to_code(data, self.path)

    def set_data(self, *args, **kwargs):  # bytecode cache writes
        return None


class _FamilyFinder(importlib.abc.MetaPathFinder):
    """Resolve the linked family's modules from its snapshot, and only to verified files."""

    def __init__(self, alias: str, code_dir: Path, digests: Mapping[str, str] | None):
        self.alias = alias
        self.code_dir = code_dir
        self.pkg_dir = code_dir / PACKAGE
        # absolute file path -> sha256 (None: provenance only, verify_manifest=False)
        self.digests = (
            None if digests is None else {str(code_dir / rel): d for rel, d in digests.items()}
        )
        self.tops: set[str] = set()  # top-level vendor names this finder has resolved
        self._path_key: tuple | None = None
        self._index: dict[str, Path] = {}

    # sys.path entries bootstrap (and adapter.py) added, in sys.path order
    def roots(self) -> list[Path]:
        key = tuple(sys.path)
        if key != self._path_key:
            roots, index = [], {}
            for entry in sys.path:
                try:
                    p = Path(entry).resolve() if entry else None
                except OSError:
                    continue
                if p is not None and p.is_relative_to(self.code_dir) and p not in roots:
                    roots.append(p)
            for root in roots:
                for child in sorted(root.iterdir()) if root.is_dir() else ():
                    name = child.stem if child.suffix == ".py" else child.name
                    is_mod = child.suffix == ".py" or (child / "__init__.py").is_file()
                    if is_mod and name.isidentifier() and name not in index:
                        index[name] = root
            self._path_key, self._index = key, index
            self._roots = roots
        return self._roots

    def toplevel_names(self) -> set[str]:
        self.roots()
        return set(self._index)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top == self.alias:
            if fullname == self.alias:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        elif "." not in fullname:
            roots = self.roots()
            if not roots:
                return None
            # Resolve exactly as sys.path order would among the family's own entries. A hit
            # that is not a verified file (e.g. planted after loading) is refused below.
            spec = importlib.machinery.PathFinder.find_spec(fullname, [str(r) for r in roots])
            if spec is None or spec.origin is None or spec.loader is None:
                return None  # not ours, or a namespace portion (a regular package wins elsewhere)
            self.tops.add(fullname)
        elif top in self.tops:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        else:
            return None
        if spec is None:
            return None
        return self.verified(spec)

    def check_file(self, name: str, origin: str | None) -> tuple[str, str | None]:
        if not origin or not origin.endswith(".py"):
            raise CodeProvenanceError(f"{name}: resolved to {origin!r}, not a .py source",
                                      name=name)
        resolved = Path(origin).resolve()
        if not resolved.is_relative_to(self.code_dir):
            raise CodeProvenanceError(f"{name}: resolved to {origin}, outside {self.code_dir}",
                                      name=name)
        if self.digests is None:
            return str(resolved), None
        if str(resolved) not in self.digests:
            raise CodeProvenanceError(f"{name}: {origin} is not a verified NVIDIA file", name=name)
        return str(resolved), self.digests[str(resolved)]

    def verified(self, spec):
        origin, digest = self.check_file(spec.name, spec.origin)
        spec.loader = _VerifiedSourceLoader(spec.name, origin, digest)
        spec.origin = origin
        return spec

    def owns(self, name: str) -> bool:
        top = name.partition(".")[0]
        return top == self.alias or top in self.tops or (
            "." not in name and name in self.toplevel_names()
        )


# --- handle -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NvRef:
    """Picklable reference to an NVIDIA function or class of a linked family.

    ``module`` is a ``batch_utils`` submodule (``"preprocessing"``) or, with ``vendor=True``, a
    top-level vendor module (``"composers"``). Resolution happens in the calling process, which
    must have linked the same family (in a pool worker: via ``worker_init``).
    """

    family: Family
    module: str
    qualname: str
    vendor: bool = False

    def resolve(self) -> Any:
        nv = loaded_family()
        if nv is None or nv.family is not self.family:
            raise FamilyConflictError(
                f"NvRef {self}: this process has not linked {self.family.value}; use "
                "flora_deploy.nv.linker.worker_init as the process-pool initializer"
            )
        obj: Any = nv.vendor(self.module) if self.vendor else nv.module(self.module)
        for part in self.qualname.split("."):
            obj = getattr(obj, part)
        return obj

    def __call__(self, *args, **kwargs):
        return self.resolve()(*args, **kwargs)


@dataclass
class NvFamily:
    """Handle to one linked NVIDIA family. Attribute access imports submodules lazily."""

    family: Family
    code_dir: Path
    alias: str
    package: ModuleType
    pinned: dict | None = None
    verified: bool = False
    _finder: _FamilyFinder | None = field(default=None, repr=False)
    _cache: dict[str, ModuleType] = field(default_factory=dict, repr=False)

    @property
    def package_dir(self) -> Path:
        return self.code_dir / PACKAGE

    def module(self, name: str) -> ModuleType:
        """Import ``batch_utils.<name>`` of this family (relative to the alias)."""
        if name not in self._cache:
            self._cache[name] = importlib.import_module(f"{self.alias}.{name}")
        return self._cache[name]

    def vendor(self, dotted: str) -> ModuleType:
        """Import a top-level vendor module NVIDIA's bootstrap made importable (SEGCAP only),
        e.g. ``vendor("flora_unified.unified_input")``, and check it comes from this snapshot."""
        if self.family is not Family.SEGCAP:
            raise AttributeError(f"{self.family.value} has no vendor tree")
        self.module("bootstrap")
        mod = importlib.import_module(dotted)
        origin = getattr(mod, "__file__", None)
        root = (self.package_dir / "vendor").resolve()
        if origin is None or not Path(origin).resolve().is_relative_to(root):
            raise FamilyConflictError(f"{dotted!r} resolved to {origin}, outside {root}")
        return mod

    def ref(self, module: str, qualname: str, *, vendor: bool = False) -> NvRef:
        return NvRef(self.family, module, qualname, vendor)

    def ref_for(self, obj: Any) -> NvRef:
        """``NvRef`` for a function/class object of this family (e.g. ``nv.media.frames_for``)."""
        mod, qualname = obj.__module__, obj.__qualname__
        if mod.startswith(self.alias + "."):
            return self.ref(mod[len(self.alias) + 1:], qualname)
        if self._finder is not None and self._finder.owns(mod):
            return self.ref(mod, qualname, vendor=True)
        raise ValueError(f"{mod}.{qualname} does not belong to {self.alias}")

    def pool_initializer(self) -> dict:
        """``ProcessPoolExecutor(**nv.pool_initializer(), mp_context=...)`` keyword arguments."""
        return dict(initializer=worker_init,
                    initargs=(self.family.value, str(self.code_dir), self.pinned))

    def audit(self) -> list[str]:
        """Check that every imported module of this family comes from a verified snapshot file.

        Returns the checked module names; raises ``CodeProvenanceError`` on the first foreign one.
        Call it after NVIDIA code that imports lazily (stages may call it per episode).
        """
        finder = self._finder
        names = []
        for name, mod in list(sys.modules.items()):
            if finder is None or not finder.owns(name) or mod is None:
                continue
            origin = getattr(mod, "__file__", None)
            if origin is None and name.partition(".")[0] in finder.toplevel_names() | finder.tops:
                # a namespace package can only come from somewhere else
                raise CodeProvenanceError(f"{name}: no __file__ (namespace package?)", name=name)
            if origin is not None:
                finder.check_file(name, origin)
            names.append(name)
        return sorted(names)

    @property
    def production(self) -> ModuleType:
        """NVIDIA's ``adapter`` module as returned by ``bootstrap.initialize()`` (SEGCAP)."""
        return self.module("bootstrap").production

    adapter = production

    def __getattr__(self, name: str) -> ModuleType:
        # Only reached for names that are not regular attributes.
        family = self.__dict__.get("family")
        if name.startswith("_") or family is None:
            raise AttributeError(name)
        if name in FAMILY_MODULES[family]:
            return self.module(name)
        raise AttributeError(f"NVIDIA family {family.value!r} exposes no module {name!r}")


_LOCK = threading.RLock()
_LOADED: NvFamily | None = None


def code_dir_for(family: Family, root: Path | str | None = None) -> Path:
    """Directory holding ``family``'s code pin: ``<root>/<repo>@<rev>``.

    ``root`` defaults to ``$FLORA_NV_CODE_CACHE`` or ``~/.cache/flora-deploy/nv`` (local tests);
    on Modal pass the weights Volume's ``snapshots`` directory.
    """
    if root is None:
        root = os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)
    return Path(root).expanduser() / snapshot_dirname(PINS_BY_KEY[FAMILY_CODE_PIN[family]])


def alias_for(family: Family, code_dir: Path) -> str:
    """``flora_nv_<family>_<rev8>``; the revision comes from the ``<repo>@<rev>`` dir name, or
    from a hash of the path when the directory is not named that way."""
    match = re.search(r"@([0-9a-f]{8,40})$", code_dir.name)
    tag = match[1][:8] if match else hashlib.sha256(str(code_dir).encode()).hexdigest()[:8]
    return f"{ALIAS_PREFIX}{family.value}_{tag}"


def _foreign_modules(code_dir: Path, names) -> list[str]:
    """Already-imported sentinel modules that do not come from ``code_dir``."""
    root = code_dir.resolve()
    bad = []
    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        origin = getattr(mod, "__file__", None)
        paths = [origin] if origin else list(getattr(mod, "__path__", []) or [])
        if name == PACKAGE or not paths or not all(
            Path(p).resolve().is_relative_to(root) for p in paths
        ):
            bad.append(f"{name} ({origin or paths or 'no file'})")
    return bad


def _other_aliases(alias: str) -> list[str]:
    return sorted(
        n for n in sys.modules if n.startswith(ALIAS_PREFIX) and "." not in n and n != alias
    )


def loaded_family() -> NvFamily | None:
    return _LOADED


def _rollback(alias: str, code_dir: Path, finder: _FamilyFinder, path_before: list[str]) -> None:
    """Undo a failed load: only the family's own modules, finder and ``sys.path`` entries."""
    for name, mod in list(sys.modules.items()):
        origin = getattr(mod, "__file__", None) if mod is not None else None
        own = name == alias or name.startswith(alias + ".")
        if not own and origin:
            try:
                own = Path(origin).resolve().is_relative_to(code_dir)
            except OSError:
                own = False
        if own:
            sys.modules.pop(name, None)
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)
    before = set(path_before)
    for entry in list(sys.path):
        if entry in before or not entry:
            continue
        if Path(entry).resolve().is_relative_to(code_dir):
            sys.path.remove(entry)
            sys.path_importer_cache.pop(entry, None)


def load_family(
    family: Family | str,
    code_dir: Path | str | None = None,
    *,
    verify_manifest: bool = True,
    pinned_digests: Mapping[str, str] | None = None,
) -> NvFamily:
    """Link NVIDIA's ``batch_utils`` for ``family`` from ``code_dir`` (see module docstring).

    Idempotent for the same family and directory. For SEGCAP this runs NVIDIA's bootstrap
    immediately (vendor SHA check + ``sys.path`` inserts); other submodules load on first use.
    ``pinned_digests`` overrides the ``RepoPin.code_digests`` looked up for ``code_dir`` (tests).
    ``verify_manifest=False`` skips our integrity checks (NVIDIA's bootstrap check still runs).
    """
    global _LOADED
    family = Family(family)
    code_dir = Path(code_dir).expanduser() if code_dir is not None else code_dir_for(family)
    code_dir = code_dir.resolve()
    with _LOCK:
        if _LOADED is not None:
            if _LOADED.family is family and _LOADED.code_dir == code_dir:
                if verify_manifest and not _LOADED.verified:
                    verify_family_code(family, code_dir, pinned_digests)
                    _LOADED.verified = True
                return _LOADED
            raise FamilyConflictError(
                f"cannot load {family.value} from {code_dir}: this process already linked "
                f"{_LOADED.family.value} from {_LOADED.code_dir}; use a separate process"
            )
        alias = alias_for(family, code_dir)
        others = _other_aliases(alias)
        if others:
            raise FamilyConflictError(f"another NVIDIA family is already imported: {others}")
        foreign = _foreign_modules(code_dir, FOREIGN_SENTINELS)
        if foreign:
            raise FamilyConflictError(
                "modules NVIDIA's code expects to own are already imported from elsewhere: "
                + ", ".join(foreign)
            )
        pkg_dir = code_dir / PACKAGE
        init = pkg_dir / "__init__.py"
        if not init.is_file():
            raise FileNotFoundError(f"no NVIDIA {PACKAGE} package at {pkg_dir}")
        digests = verify_family_code(family, code_dir, pinned_digests) if verify_manifest else None

        finder = _FamilyFinder(alias, code_dir, digests)
        path_before = list(sys.path)
        sys.meta_path.insert(0, finder)
        try:
            origin, digest = finder.check_file(alias, str(init))
            spec = importlib.util.spec_from_file_location(
                alias, origin, loader=_VerifiedSourceLoader(alias, origin, digest),
                submodule_search_locations=[str(pkg_dir)],
            )
            assert spec is not None and spec.loader is not None
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            spec.loader.exec_module(package)
            handle = NvFamily(
                family=family, code_dir=code_dir, alias=alias, package=package,
                pinned=None if pinned_digests is None else dict(pinned_digests),
                verified=verify_manifest, _finder=finder,
            )
            if family is Family.SEGCAP:
                handle.module("bootstrap")  # NVIDIA's vendor SHA check + sys.path inserts
                outside = [r for r in finder.roots() if not r.is_relative_to(pkg_dir)]
                if outside:
                    raise CodeProvenanceError(f"sys.path entries outside {PACKAGE}/: {outside}")
                foreign = _foreign_modules(code_dir, sorted(finder.toplevel_names()))
                if foreign:
                    raise FamilyConflictError("foreign modules shadow NVIDIA's: "
                                              + ", ".join(foreign))
            handle.audit()
        except BaseException:
            _rollback(alias, code_dir, finder, path_before)
            raise
        _LOADED = handle
        return handle


def worker_init(family: Family | str, code_dir: Path | str,
                pinned_digests: Mapping[str, str] | None = None) -> None:
    """Process-pool initializer: link ``family`` from ``code_dir`` in the worker, with every
    check ``load_family`` runs in the parent (spawn and forkserver children start empty)."""
    load_family(family, code_dir, pinned_digests=pinned_digests)


def _reset_for_tests() -> None:
    """Forget the loaded family (unit tests only; does not undo NVIDIA's side effects)."""
    global _LOADED
    with _LOCK:
        if _LOADED is not None and _LOADED._finder in sys.meta_path:
            sys.meta_path.remove(_LOADED._finder)
        _LOADED = None
