"""Gate C invariant: a ``<t seconds>`` string tokenizes the same alone and after a newline.

vLLM 0.24.0 tokenizes each per-frame timestamp string on its own (``get_video_repl``), HF
tokenizes it inside the expanded prompt. ``nv.capture.wrap_bare_video_tokens`` therefore requires
a newline before every video token, and between frames the previous token is ``<|vision_end|>``.
The property holds because of the pre-tokenizer's Split regex (no alternative continues past
``\\n`` into ``<``) and because the vision tokens are added specials without lstrip/rstrip. This
test locks it for the pinned Flora tokenizer files, so a tokenizer change cannot silently break
the wrap. It also documents why the guard is needed: after ``"A:"`` the property fails.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.nvlink

tokenizers = pytest.importorskip("tokenizers")

PREFIXES = ("\n", ":\n", "].\n", "\n\n", " \n", "}\n", "<|im_start|>user\n", "<|vision_end|>")
LARGE = (119.875, 999.95, 4096.0, 19999.9, 1e5, 1e7, 1.2e8)


def _tokenizer_files() -> list[Path]:
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    files = [root / snapshot_dirname(PINS_BY_KEY[f"{k}_code"]) / "tokenizer.json"
             for k in ("tempseg", "captioner")]
    if not all(f.is_file() for f in files):
        pytest.skip("Flora tokenizer files not fetched (fetch_nv_code.py --with-processor)")
    return files


def _stamps():
    vals = {f"{i / 10:.1f}" for i in range(0, 50001)}  # 0.0 .. 5000.0 s
    vals |= {f"{x:.1f}" for x in LARGE}
    return [f"<{v} seconds>" for v in sorted(vals)]


def test_tempseg_and_captioner_share_the_tokenizer():
    a, b = (hashlib.sha256(f.read_bytes()).hexdigest() for f in _tokenizer_files())
    assert a == b


def test_timestamp_tokenizes_the_same_alone_and_after_newline():
    tk = tokenizers.Tokenizer.from_file(str(_tokenizer_files()[0]))
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids  # noqa: E731
    prefix_ids = {p: enc(p) for p in PREFIXES}
    bad = []
    for ts in _stamps():
        alone = enc(ts)
        for p in PREFIXES:
            if enc(p + ts) != prefix_ids[p] + alone:
                bad.append((p, ts))
    assert not bad, bad[:5]


def test_property_fails_without_newline():
    tk = tokenizers.Tokenizer.from_file(str(_tokenizer_files()[0]))
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids  # noqa: E731
    ts = "<0.2 seconds>"
    assert enc("A:" + ts) != enc("A:") + enc(ts)
