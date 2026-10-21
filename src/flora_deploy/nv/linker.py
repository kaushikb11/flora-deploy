"""Import one NVIDIA code family into this process under a unique module name (PLAN.md D1).

NVIDIA ships three unrelated packages that are all called ``batch_utils``, and the TempSeg /
Captioner one runs ``bootstrap.initialize()`` at import time, which SHA-checks
``vendor/MANIFEST.json``, puts ``vendor``, ``vendor/support`` and ``vendor/production_adapter`` at
the front of ``sys.path`` and imports the top-level module ``adapter`` (which in turn adds
``.../core`` so ``flora_runtime`` becomes importable). We never edit or re-implement that: we load
the package from the user's snapshot under an alias such as ``flora_nv_segcap_78a0e4ac`` so its
relative imports work, and let NVIDIA's bootstrap run untouched.

Isolation rules enforced here:

- One family per process. Loading a second family, or the same family from a different
  directory, raises ``FamilyConflictError``.
- The top-level names NVIDIA's bootstrap claims (``adapter``, ``flora_runtime``, ...) and the bare
  name ``batch_utils`` must not already belong to someone else; if they do, NVIDIA's
  ``import adapter`` would silently bind to foreign code, so we refuse.
- Besides NVIDIA's own vendor SHA check we verify ``batch_utils/CODE_MANIFEST.json`` (when the
  family ships one), because nothing in NVIDIA's runtime checks ``pipeline.py`` & co.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname
from flora_deploy.nv import FAMILY_CODE_PIN, Family, FamilyConflictError

PACKAGE = "batch_utils"
ALIAS_PREFIX = "flora_nv_"

# Top-level module names that SEGCAP's bootstrap (and adapter.py's own sys.path insert) make
# importable. If any of these is already imported from elsewhere, NVIDIA's code would bind to it.
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


class CodeManifestError(RuntimeError):
    """NVIDIA's ``CODE_MANIFEST.json`` does not match the files on disk."""


@dataclass
class NvFamily:
    """Handle to one linked NVIDIA family. Attribute access imports submodules lazily."""

    family: Family
    code_dir: Path
    alias: str
    package: ModuleType
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


def verify_code_manifest(code_dir: Path) -> int:
    """Check every ``batch_utils/`` entry of NVIDIA's ``CODE_MANIFEST.json``; return the count.

    Entries outside ``batch_utils/`` (e.g. the repo-root ``infer.py``) are not part of the code
    pin we fetch and are skipped. Returns 0 when the family ships no manifest.
    """
    path = code_dir / PACKAGE / "CODE_MANIFEST.json"
    if not path.is_file():
        return 0
    manifest = json.loads(path.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CodeManifestError(f"{path}: no 'files' mapping")
    checked = 0
    for rel, entry in files.items():
        if not rel.startswith(PACKAGE + "/"):
            continue
        target = code_dir / rel
        if not target.resolve().is_relative_to(code_dir.resolve()):
            raise CodeManifestError(f"{rel}: path escapes the snapshot")
        if not target.is_file():
            raise CodeManifestError(f"{rel}: missing")
        data = target.read_bytes()
        if "bytes" in entry and len(data) != entry["bytes"]:
            raise CodeManifestError(f"{rel}: size {len(data)} != {entry['bytes']}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise CodeManifestError(f"{rel}: sha256 mismatch")
        checked += 1
    return checked


def _foreign_modules(code_dir: Path) -> list[str]:
    """Already-imported sentinel modules that do not come from ``code_dir``."""
    root = code_dir.resolve()
    bad = []
    for name in FOREIGN_SENTINELS:
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


def load_family(
    family: Family | str,
    code_dir: Path | str | None = None,
    *,
    verify_manifest: bool = True,
) -> NvFamily:
    """Link NVIDIA's ``batch_utils`` for ``family`` from ``code_dir`` (see module docstring).

    Idempotent for the same family and directory. For SEGCAP this runs NVIDIA's bootstrap
    immediately (vendor SHA check + ``sys.path`` inserts); other submodules load on first use.
    """
    global _LOADED
    family = Family(family)
    code_dir = Path(code_dir).expanduser() if code_dir is not None else code_dir_for(family)
    code_dir = code_dir.resolve()
    with _LOCK:
        if _LOADED is not None:
            if _LOADED.family is family and _LOADED.code_dir == code_dir:
                return _LOADED
            raise FamilyConflictError(
                f"cannot load {family.value} from {code_dir}: this process already linked "
                f"{_LOADED.family.value} from {_LOADED.code_dir}; use a separate process"
            )
        alias = alias_for(family, code_dir)
        others = _other_aliases(alias)
        if others:
            raise FamilyConflictError(f"another NVIDIA family is already imported: {others}")
        foreign = _foreign_modules(code_dir)
        if foreign:
            raise FamilyConflictError(
                "modules NVIDIA's code expects to own are already imported from elsewhere: "
                + ", ".join(foreign)
            )
        pkg_dir = code_dir / PACKAGE
        init = pkg_dir / "__init__.py"
        if not init.is_file():
            raise FileNotFoundError(f"no NVIDIA {PACKAGE} package at {pkg_dir}")
        if verify_manifest:
            verify_code_manifest(code_dir)

        path_before = list(sys.path)
        modules_before = set(sys.modules)
        try:
            spec = importlib.util.spec_from_file_location(
                alias, init, submodule_search_locations=[str(pkg_dir)]
            )
            assert spec is not None and spec.loader is not None
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            spec.loader.exec_module(package)
            handle = NvFamily(family=family, code_dir=code_dir, alias=alias, package=package)
            if family is Family.SEGCAP:
                handle.module("bootstrap")  # NVIDIA's vendor SHA check + sys.path inserts
        except BaseException:
            for name in set(sys.modules) - modules_before:
                sys.modules.pop(name, None)
            sys.path[:] = path_before
            raise
        _LOADED = handle
        return handle


def _reset_for_tests() -> None:
    """Forget the loaded family (unit tests only; does not undo NVIDIA's side effects)."""
    global _LOADED
    with _LOCK:
        _LOADED = None
