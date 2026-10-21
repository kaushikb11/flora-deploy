"""The near-copy detector behind the content license guard (scripts/license_shingles.py).

The reference text here is made up for the test; the real reference (NVIDIA's fetched files)
is only used in ``tests/nvlink/test_license_content.py``. Review 9a M1: a template pasted into
one string literal, a prompt split into a list of lines or joined with ``+``, and a copy with
one word changed must all be caught.
"""

import json
from pathlib import Path

import pytest

from scripts import license_shingles as L

ROOT = Path(__file__).resolve().parents[2]

TEMPLATE = """{%- for message in messages %}
{%- if message.role == 'system' %}<|start|>system
You are a careful annotator of robot videos. Describe every manipulation the arm performs,
name each object it touches, and say where the object ends up relative to the table edge.
{%- elif message.role == 'user' %}<|start|>user
{%- for part in message.content %}{%- if part.type == 'video' %}<|video_pad|>{%- else %}
{{ part.text }}{%- endif %}{%- endfor %}
{%- endif %}<|end|>
{%- endfor %}
{%- if add_generation_prompt %}<|start|>assistant
Think about the order of the subtasks first, then list them with start and end timestamps,
one per line, and never invent a subtask that the video does not show clearly.
{%- endif %}"""

PROMPT = ("Split the episode into the smallest subtasks a person would name, such as reaching "
          "for the cup, grasping its handle, lifting it off the saucer and placing it on the "
          "shelf, and give each subtask a short imperative caption that a new operator could "
          "follow without seeing the video.")


@pytest.fixture(scope="module")
def ref():
    r = L.Reference(L.shingles(L.tokens((ROOT / "LICENSE").read_text(encoding="utf-8"))))
    r.add("chat_template.jinja", TEMPLATE)
    r.add("PROMPTS.json", json.dumps({"segment": PROMPT, "n": 3}))
    return r


def _lines_list(text: str) -> str:
    return "PARTS = [\n" + "".join(f"    {line!r},\n" for line in text.splitlines()) + "]\n"


def _plus_joined(text: str) -> str:
    words = text.split(" ")
    chunks = [" ".join(words[i:i + 5]) + " " for i in range(0, len(words), 5)]
    return "P = (\n    " + "\n    + ".join(repr(c) for c in chunks) + "\n)\n"


CAUGHT = {
    "template as one literal": lambda: f"x = 1\nTEMPLATE = {TEMPLATE!r}\n",
    "half the template": lambda: f"T = {TEMPLATE[: len(TEMPLATE) // 2]!r}\n",
    "template as a list of lines": lambda: _lines_list(TEMPLATE),
    "template lines with comments between": lambda: "PARTS = [\n" + "".join(
        f"    # part {i}\n    {line!r},\n" for i, line in enumerate(TEMPLATE.splitlines())) + "]\n",
    "prompt joined with +": lambda: _plus_joined(PROMPT),
    "prompt as adjacent literals": lambda: "P = (\n" + "".join(
        f"    {line!r}\n" for line in PROMPT.split(", ")) + ")\n",
    "prompt with one word changed": lambda: f"P = {PROMPT.replace('shelf', 'counter')!r}\n",
    "template with one word changed":
        lambda: f"T = {TEMPLATE.replace('careful', 'precise')!r}\n",
    "prompt re-wrapped in a markdown doc": lambda: "# Notes\n\n> " + PROMPT.upper().replace(
        " a ", "\n> a ") + "\n",
}


@pytest.mark.parametrize("name", CAUGHT)
def test_near_copies_are_caught(ref, name):
    hit = ref.check("fixture.py" if "markdown" not in name else "notes.md", CAUGHT[name]())
    assert hit is not None, name


def test_hit_names_only_our_file_line_and_ratio(ref):
    hit = ref.check("pkg/mod.py", "import os\n\n\n" + _lines_list(TEMPLATE))
    assert hit is not None and hit.line == 5  # the first copied line, not the file's start
    msg = str(hit)
    assert msg.startswith("pkg/mod.py:5 (match ratio ") and msg.endswith(")")
    assert not set(L.tokens(msg)) & (set(L.tokens(TEMPLATE)) - {"5", "match", "ratio"})


NOT_CAUGHT = {
    "a short shared phrase": "# Describe every manipulation the arm performs.\n",
    "the Apache-2.0 text we ship": (ROOT / "LICENSE").read_text(encoding="utf-8"),
    "scattered short phrases": "\n\n".join(
        " ".join(TEMPLATE.split()[i:i + L.SHINGLE + 1]) + " zz" * 30 for i in range(0, 140, 40)),
    "unrelated text": (ROOT / "README.md").read_text(encoding="utf-8"),
}


@pytest.mark.parametrize("name", NOT_CAUGHT)
def test_unrelated_or_short_overlaps_pass(ref, name):
    assert ref.check("x.md", NOT_CAUGHT[name]) is None


def test_a_long_passage_of_a_long_doc_is_caught_by_run_length():
    long_doc = " ".join(f"{w}{i}" for i in range(2000) for w in ("alpha",))
    r = L.Reference()
    r.add("README.md", long_doc)
    part = " ".join(long_doc.split()[500:500 + L.RUN_MIN + L.SHINGLE])  # ~2% of the doc
    hit = r.check("docs/x.md", "intro\n" + part)
    assert hit is not None and hit.line == 2 and hit.ratio < L.RATIO
    short = " ".join(long_doc.split()[500:500 + L.RUN_MIN // 2])
    assert r.check("docs/x.md", short) is None


def test_no_reference_text_or_hashes_are_stored():
    """The module holds thresholds and code only (NOTICE.md)."""
    src = (ROOT / "scripts" / "license_shingles.py").read_text(encoding="utf-8")
    assert "frozenset({b" not in src and not any(
        len(tok) >= 16 and all(c in "0123456789abcdef" for c in tok) for tok in L.tokens(src))
