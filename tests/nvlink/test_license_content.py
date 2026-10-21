"""Content half of the license guard (NOTICE.md; review 7 G4-4).

``tests/unit/test_license_guard.py`` matches *paths*. This one matches *content*: no tracked file
may contain NVIDIA's code, prompts, chat templates, model cards or license text, under any name.
The fingerprints are computed at test time from the fetched NVIDIA files (``nv_code``) and only
compared by hash or in memory; no NVIDIA text is stored in this repo.

* **Blocks**: every run of ``SHINGLE`` consecutive non-trivial lines (whitespace-normalized) of
  every NVIDIA text file. Short mirrored API calls and import lines are fine; a copied function,
  template or README section is not.
* **Prose**: every long natural-language string (prompts, docstrings, JSON values) in NVIDIA's
  ``.py`` / ``.json`` files, matched against our string literals and our whitespace-normalized
  text, so a prompt pasted into a fixture is caught however it is wrapped.
"""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.nvlink

ROOT = Path(__file__).resolve().parents[2]
PINS = ("tempseg_code", "captioner_code", "rewarder")
SKIP_NV = {"tokenizer.json", "LICENSE.pdf"}  # vocabularies (not text) and the PDF
SHINGLE, MIN_LINE, MIN_BLOCK = 6, 12, 200
WS = re.compile(r"\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _norm(s: str) -> str:
    return WS.sub(" ", s).strip()


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def blocks(text: str) -> set[str]:
    lines = [x for x in (_norm(x) for x in text.splitlines()) if len(x) >= MIN_LINE]
    out = set()
    for i in range(len(lines) - SHINGLE + 1):
        s = "\n".join(lines[i:i + SHINGLE])
        if len(s) >= MIN_BLOCK:
            out.add(_h(s))
    return out


def _prose(s: str) -> str | None:
    s = _norm(s)
    return s if len(s) >= 80 and len(WORD.findall(s)) >= 12 else None


def prose_strings(name: str, text: str) -> set[str]:
    raw: list[str] = []
    if name.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return set()
        raw = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    elif name.endswith(".json"):
        def walk(o):
            if isinstance(o, str):
                raw.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        try:
            walk(json.loads(text))
        except ValueError:
            pass
    return {p for p in map(_prose, raw) if p}


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


@pytest.fixture(scope="module")
def nv_fingerprints(nv_code):
    block_src: dict[str, str] = {}
    prose: dict[str, str] = {}
    nfiles = 0
    for key in PINS:
        root = nv_code(key)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name in SKIP_NV or ".cache" in p.relative_to(root).parts:
                continue
            text = _text(p)
            if text is None:
                continue
            nfiles += 1
            rel = f"{key}/{p.relative_to(root)}"
            for b in blocks(text):
                block_src.setdefault(b, rel)
            for s in prose_strings(p.name, text):
                prose.setdefault(s, rel)
    assert nfiles >= 20 and len(block_src) > 1000, (nfiles, len(block_src))  # not vacuous
    return block_src, prose


def tracked_text_files() -> list[tuple[str, str]]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                         check=True)
    files = []
    for f in out.stdout.splitlines():
        if f and f != "uv.lock" and (text := _text(ROOT / f)) is not None:
            files.append((f, text))
    return files


def test_no_nvidia_content_tracked(nv_fingerprints):
    block_src, prose = nv_fingerprints
    hits = []
    for f, text in tracked_text_files():
        for b in blocks(text) & block_src.keys():
            hits.append(f"{f}: a {SHINGLE}-line block of {block_src[b]}")
        mine, flat = prose_strings(f, text), _norm(text)
        for s, src in prose.items():
            if s in mine or s in flat:
                hits.append(f"{f}: prose from {src} ({s[:40]}...)")
    assert not hits, "NVIDIA material in tracked files (NOTICE.md):\n" + "\n".join(sorted(hits))


def test_detector_catches_copies(nv_fingerprints, nv_code):
    """The guard must not be vacuous: a relocated NVIDIA file and a re-wrapped prose string
    are both caught."""
    block_src, prose = nv_fingerprints
    code = next((nv_code("tempseg_code") / "batch_utils").glob("*.py"))
    relocated = "# vendor/whatever.py\n" + "\n".join(
        "    " + line for line in code.read_text(encoding="utf-8").splitlines())
    assert blocks(relocated) & block_src.keys()
    s = next(iter(prose))
    words = s.split(" ")
    pasted = 'PROMPT = (\n    "' + " ".join(words[: len(words) // 2]) + ' "\n    "' + \
        " ".join(words[len(words) // 2:]) + '"\n)\n'
    assert s in prose_strings("fixture.py", pasted)
