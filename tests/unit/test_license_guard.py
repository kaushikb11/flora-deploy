"""The public repo must never contain NVIDIA code, prompts or weights (NOTICE.md)."""

import subprocess
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = [
    "*batch_utils/*",
    "*TEMPSEG_PROMPTS.json",
    "*LICENSE.pdf",
    "research_notes/*",
    "reports/*",
    "*/research_notes/*",
    "*/reports/*",
    # NVIDIA's chat templates and HF processor/tokenizer/generation configs (prompt material;
    # exactly what fetch_nv_code.py --with-processor downloads), under any directory
    "*.jinja",
    "*chat_template*.json",
    "*tokenizer.json",
    "*tokenizer_config.json",
    "*processor_config.json",  # also preprocessor_ / video_preprocessor_config.json
    "*generation_config.json",
    # weights, of any size
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
]
# Content (relocated code, pasted prompts, model cards) is checked against the fetched NVIDIA
# files in tests/nvlink/test_license_content.py.
MAX_BYTES = 1_000_000
# The README figure (scripts/render_hero.py: frames + captions of one public EgoDex episode).
ALLOW_LARGE = {"uv.lock", "docs/img/hero.png", "docs/img/hero.svg"}


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


def test_forbidden_patterns_catch_relocated_nvidia_files():
    """The patterns match wherever the files land, not only at NVIDIA's paths."""
    for f in ("vendor/x/chat_template.jinja", "tests/fakes/tokenizer_config.json",
              "a/video_preprocessor_config.json", "b/generation_config.json",
              "lora/adapter_model.bin", "w.pt", "docs/reports/r.md", "x/research_notes/n.md",
              "tests/fakes/batch_utils/models.py"):
        assert any(fnmatch(f, pat) for pat in FORBIDDEN), f
    for f in ("tests/nvlink/test_nv_link_capture_vllm_prompt_tokenizer.py",
              "src/flora_deploy/config/templates/local_demo.yaml", "docs/ci.md"):
        assert not any(fnmatch(f, pat) for pat in FORBIDDEN), f
