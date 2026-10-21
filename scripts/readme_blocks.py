"""Fenced code blocks of README.md, for the README-as-test checks (tests/unit/test_readme.py).

    python scripts/readme_blocks.py try-it [README.md]

prints the "Try it locally" bash block as a shell script without its setup lines (clone and
install: everything before the first ``flora`` command), so CI can run exactly the commands the
README tells a new user to type, in a fresh directory with the package installed.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
TRY_IT_HEADING = "Try it locally"
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


def try_it_script(text: str) -> str:
    (block,) = [b for b in blocks(text)
                if b.lang == "bash" and b.heading.startswith(TRY_IT_HEADING)]
    lines = logical_lines(block.code)
    first = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("flora "))
    return "\n".join(["set -euo pipefail", *lines[first:]]) + "\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "try-it":
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1]) if len(argv) > 1 else README
    sys.stdout.write(try_it_script(path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
