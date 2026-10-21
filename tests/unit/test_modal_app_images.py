"""The image locks match the RUNTIME pins, are fully hashed, and feed a stable fingerprint."""

import os
import re

import pytest

from flora_deploy.config.pins import RUNTIME
from flora_deploy.modal_app import images

NAMES = ("img_cpu", "img_hfref", "img_vllm")
REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(name: str) -> dict[str, tuple[str, int]]:
    """{normalized name: (version, number of hashes)} for every requirement in the lock."""
    out: dict[str, tuple[str, int]] = {}
    current = None
    for line in images.lock_path(name).read_text().splitlines():
        if m := REQ.match(line):
            current = _norm(m.group(1))
            assert current not in out, f"{name}: {current} listed twice"
            out[current] = (m.group(2), 0)
        elif line.strip().startswith("--hash=sha256:") and current:
            ver, n = out[current]
            out[current] = (ver, n + 1)
        elif line and not line.startswith((" ", "#")):
            pytest.fail(f"{name}: unexpected lock line {line!r}")
    return out


# The RUNTIME pins each image must carry exactly.
EXPECTED = {
    "img_cpu": {
        "av": "av", "numpy": "numpy", "pillow": "pillow", "pyarrow": "pyarrow",
        "huggingface-hub": "huggingface_hub_reference", "hf-xet": "hf_xet",
    },
    "img_hfref": {
        "torch": "torch_reference", "torchvision": "torchvision_reference",
        "huggingface-hub": "huggingface_hub_reference", "transformers": "transformers",
        "peft": "peft", "accelerate": "accelerate", "safetensors": "safetensors", "av": "av",
        "numpy": "numpy", "pillow": "pillow", "hf-xet": "hf_xet",
    },
    "img_vllm": {
        "vllm": "vllm", "transformers": "transformers", "qwen-vl-utils": "qwen_vl_utils",
        "decord": "decord", "av": "av", "pillow": "pillow",
        "huggingface-hub": "huggingface_hub_reference", "hf-xet": "hf_xet",
        "torch": "torch_reference",  # vLLM 0.24.0's own torch pin equals NVIDIA's
    },
}


@pytest.mark.parametrize("name", NAMES)
def test_lock_carries_runtime_pins_exactly(name):
    lock = parse_lock(name)
    for pkg, key in EXPECTED[name].items():
        assert pkg in lock, f"{name}: {pkg} missing from lock"
        assert lock[pkg][0] == RUNTIME[key], f"{name}: {pkg}=={lock[pkg][0]} != {RUNTIME[key]}"


@pytest.mark.parametrize("name", NAMES)
def test_lock_is_fully_hashed(name):
    lock = parse_lock(name)
    assert len(lock) >= len(EXPECTED[name])
    unhashed = [pkg for pkg, (_, n) in lock.items() if n == 0]
    assert not unhashed, f"{name}: requirements without hashes: {unhashed}"


@pytest.mark.parametrize("name", NAMES)
def test_lock_input_matches_packages(name):
    """``locks/<image>.in`` is what scripts/lock_images.py writes from the package tuples, so a
    changed tuple without a re-lock fails here."""
    src = images.lock_path(name).with_suffix(".in").read_text().splitlines()
    assert [ln for ln in src if not ln.startswith("#")] == list(images.PACKAGES[name])
    lock = parse_lock(name)
    for req in images.PACKAGES[name]:
        pkg, ver = req.split("==")
        assert lock[_norm(pkg)][0] == ver, req


def test_hfref_and_vllm_get_cuda13_torch_from_pypi():
    for name in ("img_hfref", "img_vllm"):
        lock = parse_lock(name)
        assert lock["nvidia-cuda-runtime"][0].startswith("13.0."), name
        assert "--index-url" not in images.lock_path(name).read_text()


def test_bases_are_pinned():
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", RUNTIME["cuda_base_vllm_digest"])
    assert images.CUDA_BASE == (
        f"{RUNTIME['cuda_base_vllm']}@{RUNTIME['cuda_base_vllm_digest']}"
    )
    assert re.fullmatch(r"\d{4}\.\d{2}", RUNTIME["modal_image_builder_version"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", RUNTIME["uv"])


def test_builder_version_is_enforced(monkeypatch):
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", "placeholder")  # so undo restores it
    monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION")
    images._pin_builder_version()
    assert os.environ["MODAL_IMAGE_BUILDER_VERSION"] == RUNTIME["modal_image_builder_version"]
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", "2023.12")
    with pytest.raises(RuntimeError):
        images._pin_builder_version()


@pytest.mark.parametrize("name", NAMES)
def test_every_image_sets_env(name):
    assert images.ENV[name]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert images.ENV[name]["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert ("VLLM_CACHE_ROOT" in images.ENV[name]) == (name == "img_vllm")


def test_fingerprint_is_stable_and_distinct():
    fps = images.image_fingerprints()
    assert set(fps) == set(NAMES)
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in fps.values())
    assert len(set(fps.values())) == 3
    assert fps == images.image_fingerprints()


def test_fingerprint_tracks_lock_bytes_and_bases(tmp_path, monkeypatch):
    before = images.image_fingerprint("img_cpu")
    other = tmp_path / "cpu.txt"
    other.write_bytes(images.lock_path("img_cpu").read_bytes() + b"# changed\n")
    monkeypatch.setattr(images, "LOCK_DIR", tmp_path)
    changed_lock = images.image_fingerprint("img_cpu")
    assert changed_lock != before

    monkeypatch.undo()
    assert images.image_fingerprint("img_cpu") == before
    monkeypatch.setitem(RUNTIME, "modal_image_builder_version", "2099.01")
    assert images.image_fingerprint("img_cpu") != before
