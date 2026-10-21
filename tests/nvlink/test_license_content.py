"""Content half of the license guard (NOTICE.md; review 7 G4-4, review 9a M1).

``tests/unit/test_license_guard.py`` matches *paths*. This one matches *content*: no tracked file
may contain NVIDIA's code, prompts, chat templates or model cards, under any name or wrapping.
Detection is ``scripts/license_shingles.py``: 8-word shingles of every fetched NVIDIA text file
(and of each long string value in it) against the shingles of each tracked file's text and, for
Python, of its string constants. That catches a template pasted into a string literal, a prompt
split into a list of lines or joined with ``+``, and a copy with words changed.

The reference is computed here, in memory, from the fetched files (``nv_code``); no NVIDIA text
or hash of it is stored in this repo, and no NVIDIA text is printed: a failure names only our
file, a line and a match ratio. Without the fetched files the tests skip as "not fetched",
which ``FLORA_FAIL_ON_SKIP=1`` (CI's nvlink job) turns into a failure (tests/conftest.py).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import license_shingles as L

pytestmark = pytest.mark.nvlink

ROOT = Path(__file__).resolve().parents[2]
PINS = ("tempseg_code", "captioner_code", "rewarder")
# Vocabularies (not text), the PDF, and dependency pin lists (facts we mirror on purpose in
# pyproject.toml and modal_app/locks/*.in).
SKIP_NV = {"tokenizer.json", "LICENSE.pdf"}


def _skip_nv(p: Path) -> bool:
    return p.name in SKIP_NV or (p.name.startswith("requirements") and p.suffix == ".txt")


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def nv_files(nv_code):
    for key in PINS:
        root = nv_code(key)
        for p in sorted(root.rglob("*")):
            if p.is_file() and ".cache" not in p.relative_to(root).parts and not _skip_nv(p):
                if (text := _text(p)) is not None:
                    yield p, text


@pytest.fixture(scope="module")
def nv_reference(nv_code):
    apache = (ROOT / "LICENSE").read_text(encoding="utf-8")  # ours to ship, whoever else does
    ref = L.Reference(L.shingles(L.tokens(apache)))
    nfiles = 0
    for p, text in nv_files(nv_code):
        nfiles += 1
        ref.add(p.name, text)
    assert nfiles >= 20 and len(ref) >= 50 and len(ref.index) > 10_000, \
        (nfiles, len(ref), len(ref.index))  # not vacuous
    return ref


def tracked_text_files() -> list[tuple[str, str]]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                         check=True)
    return [(f, text) for f in out.stdout.splitlines()
            if f and f != "uv.lock" and (text := _text(ROOT / f)) is not None]


# Flagged files a person has looked at and accepted, each with the highest match ratio accepted.
# A hit above that ratio, or anywhere else, still fails.
REVIEWED = {
    # ``_asset`` re-expresses robot_caption_geometry.py:23-35 (one packed pair) so it can run on
    # a thread pool; about 10 lines of the same statements. Flagged by review 9a's shingle
    # guard; kept pending a licensing decision (rewrite it, or call NVIDIA's code instead).
    "src/flora_deploy/accel/geometry.py": 0.15,
}


def test_no_nvidia_content_tracked(nv_reference):
    hits = [str(hit) for f, text in tracked_text_files()
            if (hit := nv_reference.check(f, text)) is not None
            and hit.ratio > REVIEWED.get(f, -1.0)]
    assert not hits, "NVIDIA material in tracked files (NOTICE.md):\n" + "\n".join(sorted(hits))


def _longest_json_string(nv_code) -> str:
    best = ""
    for p, text in nv_files(nv_code):
        if p.suffix == ".json":
            stack = [json.loads(text)]
            while stack:
                o = stack.pop()
                if isinstance(o, str) and len(o) > len(best):
                    best = o
                elif isinstance(o, dict):
                    stack += o.values()
                elif isinstance(o, list):
                    stack += o
    return best


def _pastes(nv_code):
    """Ways of pasting NVIDIA text into one of our files, by label only (never printed)."""
    for key in PINS:
        t = (nv_code(key) / "chat_template.jinja").read_text(encoding="utf-8")
        yield f"{key} template as one literal", f"T = {t!r}\n"
        yield f"{key} half the template", f"T = {t[: len(t) // 2]!r}\n"
        yield f"{key} template as a list of lines", "T = [\n" + "".join(
            f"    {line!r},\n" for line in t.splitlines()) + "]\n"
    prompt = _longest_json_string(nv_code)
    assert len(prompt.split()) >= 40
    words = prompt.split(" ")
    yield "prompt as a list of lines", "P = [\n" + "".join(
        f"    {line!r},\n" for line in prompt.splitlines()) + "]\n"
    yield "prompt joined with +", "P = (\n    " + "\n    + ".join(
        repr(" ".join(words[i:i + 6]) + " ") for i in range(0, len(words), 6)) + "\n)\n"
    mid = len(words) // 2
    yield "prompt with one word changed", repr(" ".join(words[:mid] + ["zzz"] + words[mid + 1:]))
    code = next((nv_code("tempseg_code") / "batch_utils").glob("*.py"))
    yield "relocated code file", "# vendor/x.py\n" + "\n".join(
        "    " + line for line in code.read_text(encoding="utf-8").splitlines())


def test_detector_catches_copies(nv_reference, nv_code):
    """The guard must not be vacuous: each way of pasting NVIDIA text is caught."""
    missed = []
    for label, pasted in _pastes(nv_code):
        hit = nv_reference.check("fixture.py", pasted)
        if hit is None:
            missed.append(label)
    assert not missed, missed  # labels only: the pasted text is NVIDIA's


def test_missing_nvidia_files_skip_and_fail_under_fail_on_skip(tmp_path):
    """Without the fetched files this module skips, and ``FLORA_FAIL_ON_SKIP=1`` makes that
    skip fail the run, so CI's nvlink job cannot go green with the guard unrun."""
    def run(**env):
        e = {k: v for k, v in os.environ.items() if k != "FLORA_FAIL_ON_SKIP"}
        e.update(FLORA_NV_CODE_CACHE=str(tmp_path), **env)
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
             "-p", "no:xdist", "-m", "nvlink", f"{__file__}::test_no_nvidia_content_tracked"],
            cwd=ROOT, env=e, capture_output=True, text=True, timeout=300)

    quiet = run()
    assert quiet.returncode == 0 and "1 skipped" in quiet.stdout, quiet.stdout[-500:]
    strict = run(FLORA_FAIL_ON_SKIP="1")
    assert strict.returncode == 1 and "not fetched" in strict.stdout, strict.stdout[-500:]
