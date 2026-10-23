"""Fenced code blocks of README.md, for the README-as-test checks (tests/unit/test_readme.py).

    python scripts/readme_blocks.py try-it [README.md]

prints the quickstart ("Try it in a few minutes") bash block as a shell script without its install
line (``uv tool install ...`` / ``pip install ...``: CI installs the built wheel instead), so CI
runs exactly the commands the README tells a new user to type (``mkdir``, ``cd``, every
``flora`` line) in an empty directory.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
TRY_IT_HEADING = "Try it in a few minutes"  # the quickstart block
_FENCE = re.compile(r"^```(\w*)\s*$")
_SPLIT = {"&&", "||", ";", "|"}


@dataclass(frozen=True)
class Block:
    heading: str  # the nearest "## " / "### " heading above the block
    lang: str
    code: str


def blocks(text: str) -> list[Block]:
    out: list[Block] = []
    heading, lang, buf = "", None, []
    for line in text.splitlines():
        m = _FENCE.match(line)
        if lang is None:
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
            elif m:
                lang, buf = m.group(1), []
        elif line.strip() == "```":
            out.append(Block(heading, lang, "\n".join(buf)))
            lang = None
        else:
            buf.append(line)
    return out


def logical_lines(code: str) -> list[str]:
    """Lines with backslash continuations joined."""
    out, cur = [], ""
    for line in code.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    if cur:
        out.append(cur)
    return [ln for ln in out if ln.strip()]


def commands(code: str) -> list[list[str]]:
    """Each simple command of a bash block as argv (comments dropped; split on && ; | ||)."""
    out: list[list[str]] = []
    for line in logical_lines(code):
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = "#"
        cur: list[str] = []
        for tok in lex:
            if tok in _SPLIT:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(tok)
        if cur:
            out.append(cur)
    return out


def block_under(text: str, heading: str, lang: str = "bash") -> Block:
    """The one ``lang`` block under the heading containing ``heading``; raises ``LookupError``
    naming the headings present when there is not exactly one."""
    bs = blocks(text)
    found = [b for b in bs if b.lang == lang and heading in b.heading]
    if len(found) != 1:
        heads = sorted({b.heading for b in bs if b.lang == lang})
        raise LookupError(f"README: expected one {lang} block under a heading containing "
                          f"{heading!r}, found {len(found)} (headings with {lang} blocks: {heads})")
    return found[0]


def is_install(line: str) -> bool:
    """An install command of the quickstart (replaced by the built wheel in CI)."""
    return any(argv[:3] == ["uv", "tool", "install"] or argv[:2] in (["pip", "install"],
                                                                     ["pipx", "install"])
               for argv in commands(line))


def try_it_script(text: str) -> str:
    block = block_under(text, TRY_IT_HEADING)
    lines = [ln for ln in logical_lines(block.code) if not is_install(ln)]
    if not any(ln.lstrip().startswith("flora ") for ln in lines):
        raise LookupError("README quickstart block runs no flora command")
    return "\n".join(["set -euo pipefail", *lines]) + "\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "try-it":
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1]) if len(argv) > 1 else README
    sys.stdout.write(try_it_script(path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
