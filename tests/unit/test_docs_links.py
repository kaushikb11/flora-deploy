"""Every relative link in ``README.md`` and ``docs/**/*.md`` points at an existing file or dir.

Checked: inline links and images (``[text](target)``, ``![alt](target)``), including ``<target>``
and ``"title"`` forms, and reference definitions (``[id]: target``). Not checked: URLs with a
scheme (``https:``, ``mailto:``, ...), same-page anchors (``#section``), and anything inside fenced
code blocks or inline code spans (paths in backticks are prose, not links). A ``#fragment`` on a
relative link is dropped; only the file is checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1[`~]*[ \t]*$", re.M | re.S)
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1", re.S)
_INLINE = re.compile(
    r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]"  # [text], one level of nested brackets
    r"\(\s*(<[^>]*>|[^)\s]+)"  # (target or (<target>
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"  # optional title, then )
)
_REFDEF = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)", re.M)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def markdown_files(root: Path = REPO) -> list[Path]:
    return [root / "README.md", *sorted((root / "docs").rglob("*.md"))]


def link_targets(text: str) -> list[str]:
    """Link targets of a Markdown text, outside code blocks and code spans."""
    text = _FENCE.sub("", text)
    text = _CODE_SPAN.sub("", text)
    found = [m.group(1) for m in _INLINE.finditer(text)]
    found += [m.group(1) for m in _REFDEF.finditer(text)]
    return [t[1:-1] if t.startswith("<") and t.endswith(">") else t for t in found]


def broken_links(path: Path) -> list[str]:
    bad = []
    for target in link_targets(path.read_text(encoding="utf-8")):
        if not target or target.startswith("#") or _SCHEME.match(target) or target.startswith("//"):
            continue
        rel = target.split("#", 1)[0].split("?", 1)[0]
        if not rel:
            continue
        resolved = (REPO / rel.lstrip("/")) if rel.startswith("/") else (path.parent / rel)
        if not resolved.exists():
            bad.append(target)
    return bad


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(path: Path) -> None:
    bad = broken_links(path)
    assert not bad, f"{path.relative_to(REPO)}: broken relative links: {bad}"


def test_checker_finds_broken_links_and_ignores_code(tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    (tmp_path / "ok.md").write_text("x")
    doc.write_text(
        "[ok](ok.md) [ok anchor](ok.md#a) [gone](gone.md) ![img](<pic one.png> \"t\")\n"
        "[web](https://example.com) [anchor](#here) `[span](span.md)`\n"
        "```\n[fenced](fenced.md)\n```\n"
        "[ref]: missing/ref.md\n"
    )
    assert broken_links(doc) == ["gone.md", "pic one.png", "missing/ref.md"]
