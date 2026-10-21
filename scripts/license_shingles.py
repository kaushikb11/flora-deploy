"""Word-shingle near-copy detection for the content license guard (NOTICE.md; review 9a M1).

``tests/nvlink/test_license_content.py`` uses this to check that no tracked file contains
NVIDIA's code, prompts, chat templates or model cards. Reference shingles are computed at test
time from the fetched NVIDIA files and are held in memory only. This module stores no NVIDIA
text and no hashes of it, and its reports name only *our* file, a line and a match ratio.

Text is reduced to lowercase word tokens (punctuation, quotes, ``+``, brackets and whitespace
are all dropped), so these all reduce to the same token stream:

* a template pasted into one string literal (``T = '...\\n...'``);
* the same text split into a list of lines, or joined with ``+``, or as adjacent literals;
* the same text re-wrapped or re-indented.

Python files also contribute the stream of their string constants in source order, which
skips comments and code interleaved between the pieces. A copy with a word changed still
shares every shingle that does not span the change, so the match ratio stays high.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

SHINGLE = 8  # words per shingle
RATIO = 0.5  # flag when this share of one reference doc's shingles is in one of our files
RUN_MIN = 40  # ... or on a run of this many consecutive reference shingles (a partial copy)
MIN_DOC = 8  # reference docs with fewer shingles are too short to judge
MIN_STRING_WORDS = 20  # long string values of a reference file are also docs of their own

_ESCAPE = re.compile(r"\\[nrtfv]")  # a literal "\n" in source text separates words
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(_ESCAPE.sub(" ", text.lower()))


def _hash(words: Iterable[str]) -> bytes:
    return hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest()


def shingles(words: list[str]) -> set[bytes]:
    return {_hash(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


Stream = list[tuple[bytes, int]]  # (shingle, line it starts on), in text order


def _stream(pieces: Iterable[tuple[int, str]]) -> Stream:
    """Shingles of the concatenated ``(line, text)`` pieces, in order."""
    words: list[str] = []
    lines: list[int] = []
    for line, text in pieces:
        t = tokens(text)
        words += t
        lines += [line] * len(t)
    return [(_hash(words[i:i + SHINGLE]), lines[i]) for i in range(len(words) - SHINGLE + 1)]


def _py_strings(text: str) -> list[tuple[int, int, str]] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    return sorted((n.lineno, n.col_offset, n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _json_strings(text: str) -> list[str]:
    out: list[str] = []

    def walk(o) -> None:
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    try:
        walk(json.loads(text))
    except ValueError:
        pass
    return out


def file_streams(name: str, text: str) -> list[Stream]:
    """Shingle streams of one of our files: its text, and for Python its string constants."""
    out = [_stream(enumerate(text.splitlines(), 1))]
    if name.endswith(".py") and (strings := _py_strings(text)):
        out.append(_stream((ln, s) for ln, _, s in strings))
    return out


def reference_docs(name: str, text: str) -> Iterator[set[bytes]]:
    """Shingle sets of one reference file: the whole file, plus each long string value in it
    (so a single prompt copied out of a large JSON or ``.py`` file is judged on its own)."""
    yield shingles(tokens(text))
    strings: list[str] = []
    if name.endswith(".py"):
        strings = [s for _, _, s in _py_strings(text) or ()]
    elif name.endswith(".json"):
        strings = _json_strings(text)
    for s in strings:
        if len(t := tokens(s)) >= MIN_STRING_WORDS:
            yield shingles(t)


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    ratio: float

    def __str__(self) -> str:  # our file only: never the reference text or its source
        return f"{self.path}:{self.line} (match ratio {self.ratio:.2f})"


class Reference:
    """Reference shingle sets, in memory only."""

    def __init__(self, allow: set[bytes] = frozenset()):
        self.allow = set(allow)  # shingles of text we may share (e.g. the Apache-2.0 text)
        self.docs: list[set[bytes]] = []
        self.index: dict[bytes, list[int]] = {}

    def add(self, name: str, text: str) -> None:
        for doc in reference_docs(name, text):
            doc -= self.allow
            if len(doc) < MIN_DOC:
                continue
            i = len(self.docs)
            self.docs.append(doc)
            for h in doc:
                self.index.setdefault(h, []).append(i)

    def __len__(self) -> int:
        return len(self.docs)

    def _ratio(self, run: set[bytes]) -> float:
        """The largest share of one reference doc that this run of shingles covers."""
        counts: dict[int, int] = {}
        for h in run:
            for i in self.index[h]:
                counts[i] = counts.get(i, 0) + 1
        return max(m / len(self.docs[i]) for i, m in counts.items())

    def check(self, path: str, text: str) -> Hit | None:
        """The worst near-copy of reference text in this file, or None.

        Two tests: one of our files holds at least ``RATIO`` of one reference doc's shingles
        (a whole doc copied, even with words changed), or a run of ``RUN_MIN`` consecutive
        shingles that all occur in the reference (a long passage copied). Code we wrote to do
        what NVIDIA's code does shares scattered short phrases, which trip neither."""
        streams = file_streams(path, text)
        first: dict[bytes, int] = {}
        for stream in streams:
            for h, line in stream:
                first.setdefault(h, line)
        common = first.keys() & self.index.keys()
        hits: list[Hit] = []
        counts: dict[int, int] = {}
        for h in common:
            for i in self.index[h]:
                counts[i] = counts.get(i, 0) + 1
        for i, m in counts.items():
            doc = self.docs[i]
            if m / len(doc) >= RATIO:
                hits.append(Hit(path, min(first[h] for h in doc & common), m / len(doc)))
        for stream in streams:
            run = 0
            for k, (h, _) in enumerate(stream + [(b"", 0)]):
                if h in common:
                    run += 1
                    continue
                if run >= RUN_MIN:
                    hits.append(Hit(path, stream[k - run][1],
                                    self._ratio({h for h, _ in stream[k - run:k]})))
                run = 0
        return max(hits, key=lambda hit: hit.ratio, default=None)
