"""Download NVIDIA's *code* pins (no weights) for `pytest -m nvlink`.

Files land in $FLORA_NV_CODE_CACHE (default ~/.cache/flora-deploy/nv/<repo>@<rev>/) and are never
committed: they are NVIDIA OneWay Noncommercial material (see NOTICE.md).

Usage: uv run python scripts/fetch_nv_code.py [--with-processor]
  --with-processor also fetches the small tokenizer/processor/chat-template files that the
  capture-processor parity tests need (still no weights).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

PROCESSOR_FILES = [
    "processor_config.json",
    "video_preprocessor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
]
CODE_FILES = ["batch_utils/*", "LICENSE*", "README.md"]


def cache_root() -> Path:
    return Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-processor", action="store_true")
    args = ap.parse_args()
    root = cache_root()
    patterns = CODE_FILES + (PROCESSOR_FILES if args.with_processor else [])
    for key in ("tempseg_code", "captioner_code", "rewarder"):
        pin = PINS_BY_KEY[key]
        dest = root / snapshot_dirname(pin)
        snapshot_download(pin.repo, revision=pin.revision, allow_patterns=patterns,
                          local_dir=dest)
        print(f"{key}: {dest}")


if __name__ == "__main__":
    main()
