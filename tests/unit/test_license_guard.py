"""The public repo must never contain NVIDIA code, prompts or weights (NOTICE.md)."""

import subprocess
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = [
    "*batch_utils/*",
    "*TEMPSEG_PROMPTS.json",
    "*.safetensors",
    "*LICENSE.pdf",
    "research_notes/*",
    "reports/*",
]
MAX_BYTES = 1_000_000
ALLOW_LARGE = {"uv.lock"}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def test_no_forbidden_paths_tracked():
    bad = [f for f in tracked_files() if any(fnmatch(f, pat) for pat in FORBIDDEN)]
    assert not bad, f"NVIDIA/local-only material tracked in git: {bad}"


def test_no_large_files_tracked():
    big = [
        f for f in tracked_files()
        if f not in ALLOW_LARGE and (ROOT / f).is_file() and (ROOT / f).stat().st_size > MAX_BYTES
    ]
    assert not big, f"unexpectedly large tracked files: {big}"
