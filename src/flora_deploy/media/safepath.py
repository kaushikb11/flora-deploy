"""Dataset metadata is untrusted: it never chooses a file outside the dataset (review 11b F1).

A third-party dataset's ``meta/info.json`` names its video files through a ``video_path``
template, and its episode rows and camera keys fill that template in. Readers and the worker's
:class:`~flora_deploy.workers_impl.reward.VideoResolver` turn those values into files the decoder
opens, in a container that holds the bucket credentials. So:

* :func:`format_template` formats a template with a fixed set of named fields only. Attribute or
  index access, conversions, positional or unknown fields and nested specs are refused, and a
  format spec is a short integer spec (``03d``) or empty.
* :func:`check_relative` accepts a relative POSIX path without ``.``/``..``/empty components,
  backslashes or control characters.
* :func:`contain` joins such a path to a root and requires the result, with symlinks resolved, to
  stay under the resolved root.

:class:`UnsafePath` is a ``ValueError``, so callers that treat ``ValueError`` as a per-episode
data error (a ``SourceError`` at index time, an excluded episode in a worker) keep doing so.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path


class UnsafePath(ValueError):
    """A dataset-controlled value would name a file outside its root, or is malformed."""


_INT_SPEC = re.compile(r"0?[1-9]?[0-9]?d?")  # "", "d", "03d", "06d", "6", ...
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Camera keys (LeRobot feature names such as ``observation.images.top``); as ``safe_key``.
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


def format_template(template: object, **fields: str | int) -> str:
    """``template.format(**fields)`` restricted to plain ``{name}`` / ``{name:03d}`` fields, then
    :func:`check_relative`. Raises :class:`UnsafePath`."""
    if not isinstance(template, str):
        raise UnsafePath(f"video_path template {template!r} is not a string")
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as e:
        raise UnsafePath(f"video_path template {template!r}: {e}") from None
    out = []
    for literal, name, spec, conversion in parts:
        out.append(literal)
        if name is None:
            continue
        if name not in fields:
            raise UnsafePath(f"video_path template {template!r}: field {{{name}}} is not one of "
                             f"{sorted(fields)}")
        if conversion is not None:
            raise UnsafePath(f"video_path template {template!r}: conversion !{conversion}")
        value = fields[name]
        if spec:
            if not isinstance(value, int) or not _INT_SPEC.fullmatch(spec):
                raise UnsafePath(f"video_path template {template!r}: format spec {spec!r} for "
                                 f"{{{name}}}")
            out.append(format(value, spec))
        else:
            out.append(str(value))
    return check_relative("".join(out))


def check_relative(rel: object) -> str:
    """A relative POSIX path with no ``.``, ``..`` or empty components. Raises UnsafePath."""
    if not isinstance(rel, str) or not rel:
        raise UnsafePath(f"path {rel!r}: empty or not a string")
    if (rel.startswith("/") or "\\" in rel or _CONTROL.search(rel)
            or any(p in ("", ".", "..") for p in rel.split("/"))):
        raise UnsafePath(f"path {rel!r}: must be relative, without '.', '..' or empty components")
    return rel


def contain(root: str | os.PathLike[str], rel: str) -> Path:
    """``root / rel`` (not resolved) after checking that it resolves under ``root`` resolved.
    Raises UnsafePath."""
    root = Path(root)
    path = root / check_relative(rel)
    if not _under_any(path.resolve(), _allowed_roots(root)):
        raise UnsafePath(f"path {rel!r} resolves outside {root}")
    return path


def is_within(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> bool:
    """``path`` (absolute or relative to the cwd), symlinks resolved, is under ``root``."""
    return _under_any(Path(path).resolve(), _allowed_roots(Path(root)))


def _allowed_roots(root: Path) -> list[Path]:
    """``root`` resolved, plus, when ``root`` is (inside) a Hugging Face cache snapshot
    (``<cache>/datasets--o--n/snapshots/<rev>/...``), that repo's own ``blobs/`` directory:
    snapshot files are symlinks into it, so a dataset fetched with ``huggingface-cli download``
    (no ``--local-dir``) must still resolve. Only the same repo's blobs, never the wider cache."""
    real = root.resolve()
    roots = [real]
    for base in (root.absolute(), real):
        parts = base.parts
        if "snapshots" in parts:
            i = len(parts) - 1 - parts[::-1].index("snapshots")
            repo = Path(*parts[:i])
            if repo.name.startswith(("datasets--", "models--")) and (repo / "blobs").is_dir():
                roots.append((repo / "blobs").resolve())
    return roots


def _under_any(path: Path, roots: list[Path]) -> bool:
    return any(path.is_relative_to(r) for r in roots)


def check_key(key: object) -> str:
    """A dataset camera key that is safe as one path component. Raises UnsafePath."""
    if not isinstance(key, str) or not _KEY.fullmatch(key) or ".." in key:
        raise UnsafePath(f"camera key {key!r}: use [A-Za-z0-9._-], max 200 chars, no '..'")
    return key
