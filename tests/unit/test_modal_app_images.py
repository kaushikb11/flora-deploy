"""The image locks match the RUNTIME pins, are fully hashed, and feed a stable fingerprint."""

import inspect
import os
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

from flora_deploy.config.pins import RUNTIME
from flora_deploy.modal_app import images

NAMES = ("img_cpu", "img_hfref", "img_vllm")
REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
# ``name @ https://.../name-<version>-<tags>.whl#sha256=...`` (img_hfref's cu128 torch wheels)
URL_REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*) @ https://\S+/[A-Za-z0-9_]+-([^-/]+)-\S+\.whl")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(name: str) -> dict[str, tuple[str, int]]:
    """{normalized name: (version, number of hashes)} for every requirement in the lock."""
    out: dict[str, tuple[str, int]] = {}
    current = None
    for line in images.lock_path(name).read_text().splitlines():
        if (m := REQ.match(line)) or (m := URL_REQ.match(line)):
            current = _norm(m.group(1))
            assert current not in out, f"{name}: {current} listed twice"
            out[current] = (unquote(m.group(2)), 0)
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
        "torch": "torch_hfref", "torchvision": "torchvision_hfref",
        "huggingface-hub": "huggingface_hub_reference", "transformers": "transformers",
        "peft": "peft", "accelerate": "accelerate", "safetensors": "safetensors", "av": "av",
        "numpy": "numpy", "pillow": "pillow", "hf-xet": "hf_xet",
    },
    "img_vllm": {
        "vllm": "vllm", "transformers": "transformers", "qwen-vl-utils": "qwen_vl_utils",
        "decord": "decord", "av": "av", "numpy": "numpy", "pillow": "pillow",
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
        # img_hfref's torch/torchvision resolve to the backend's local build (2.11.0+cu128)
        assert lock[_norm(pkg)][0].split("+")[0] == ver, req


def test_numpy_is_one_version_everywhere():
    assert {parse_lock(name)["numpy"][0] for name in NAMES} == {RUNTIME["numpy"]}
    assert int(RUNTIME["numpy"].split(".")[0]) < 3  # NVIDIA's "numpy<3"


def test_hfref_gets_nvidia_cu128_torch_from_pytorch_index():
    """NVIDIA's batch_utils/README.md: torch==2.11.0 torchvision==0.26.0 from the cu128 index."""
    lock = parse_lock("img_hfref")
    assert lock["torch"][0] == f"{RUNTIME['torch_reference']}+cu128" == RUNTIME["torch_hfref"]
    tv = f"{RUNTIME['torchvision_reference']}+cu128"
    assert lock["torchvision"][0] == tv == RUNTIME["torchvision_hfref"]
    assert lock["nvidia-cuda-runtime-cu12"][0].startswith("12.8.")
    assert not any(pkg.endswith("-cu13") or pkg == "nvidia-cuda-runtime" for pkg in lock)
    # exactly the two wheels from download.pytorch.org/whl/cu128, hash-locked; the rest is PyPI
    index = "https://download.pytorch.org/whl/cu128/"
    assert RUNTIME["torch_index_hfref"] + "/" == index
    text = images.lock_path("img_hfref").read_text()
    url_lines = [ln for ln in text.splitlines() if " @ " in ln]
    assert len(url_lines) == 2
    for pkg in ("torch", "torchvision"):
        sha = RUNTIME[f"{pkg}_hfref_sha256"]
        line = next(ln for ln in url_lines if ln.startswith(f"{pkg} @ "))
        assert line.startswith(f"{pkg} @ {index}{pkg}-") and f"#sha256={sha}" in line
        assert "cp312-cp312-manylinux_2_28_x86_64.whl" in line
        assert lock[pkg][1] == 1, pkg  # one hash: that wheel's
        assert f"{line}\n    --hash=sha256:{sha}\n" in text
    assert images.OVERRIDES["img_hfref"] == tuple(ln.removesuffix(" \\") for ln in url_lines)
    override = images.lock_path("img_hfref").with_name("hfref.overrides.txt").read_text()
    assert [ln for ln in override.splitlines() if not ln.startswith("#")] == list(
        images.OVERRIDES["img_hfref"]
    )
    assert "--index-url" not in text and "--extra-index-url" not in text


def test_vllm_gets_cuda13_torch_from_pypi():
    lock = parse_lock("img_vllm")
    assert lock["torch"][0] == RUNTIME["torch_reference"]
    assert lock["nvidia-cuda-runtime"][0].startswith("13.0.")
    for name in ("img_cpu", "img_vllm"):
        assert images.OVERRIDES[name] == ()
        text = images.lock_path(name).read_text()
        assert "--index-url" not in text and " @ " not in text


@pytest.mark.parametrize("name", NAMES)
def test_install_steps_are_pinned(name):
    assert images.INSTALL_OPTIONS[name].startswith("--require-hashes --no-build")
    copy_uv, copy_lock, run = images.install_commands(name)
    assert copy_uv == f"COPY --from={images.UV_IMAGE} /uv /.uv/uv"
    assert "@sha256:" in copy_uv and ":0." not in copy_uv  # digest only (no tag)
    assert copy_lock.endswith(f"/.uv/{images.LOCK_FILES[name]}")
    assert run.startswith("RUN /.uv/uv pip install ")
    assert images.INSTALL_OPTIONS[name] in run


def test_bases_are_pinned():
    digest = re.compile(r"sha256:[0-9a-f]{64}")
    for key in ("cuda_base_vllm_digest", "python_base_digest", "uv_image_digest"):
        assert digest.fullmatch(RUNTIME[key]), key
    assert images.CUDA_BASE == (
        f"{RUNTIME['cuda_base_vllm']}@{RUNTIME['cuda_base_vllm_digest']}"
    )
    assert images.PY_BASE == f"{RUNTIME['python_base']}@{RUNTIME['python_base_digest']}"
    assert images.UV_IMAGE == f"ghcr.io/astral-sh/uv@{RUNTIME['uv_image_digest']}"
    for name in NAMES:
        assert "@sha256:" in images.BASES[name]["from"], name
    assert re.fullmatch(r"\d{4}\.\d{2}", RUNTIME["modal_image_builder_version"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", RUNTIME["uv"])
    assert images.LOCK_PYTHON == {
        "img_cpu": "3.12.10", "img_hfref": "3.12.10", "img_vllm": RUNTIME["python_standalone"],
    }


def test_modal_client_matches_pin():
    """uv.lock pins the client; the image builders refuse any other (it renders every step)."""
    import modal
    from modal.mount import PYTHON_STANDALONE_VERSIONS

    assert modal.__version__ == RUNTIME["modal_client"]
    assert PYTHON_STANDALONE_VERSIONS[images.PY] == (
        RUNTIME["python_standalone_release"], RUNTIME["python_standalone"],
    )
    images._check_modal_client()


def test_modal_client_mismatch_refused(monkeypatch):
    monkeypatch.setitem(RUNTIME, "modal_client", "0.0.1")
    with pytest.raises(RuntimeError, match="modal client"):
        images._check_modal_client()


def test_builders_use_bases_and_no_debian_slim():
    """One code path: every builder goes through _build, which reads the fingerprinted BASES."""
    body = inspect.getsource(images).split('"""', 2)[2]  # skip the module docstring
    assert "debian_slim(" not in body
    build = inspect.getsource(images._build)
    for needle in ("BASES[name]", 'base["from"]', 'base["add_python"]', 'base["entrypoint"]',
                   "install_commands(name)", "lock_path(name)", "ENV[name]"):
        assert needle in build, needle
    for name, builder in images._BUILDERS.items():
        assert f'_build("{name}")' in inspect.getsource(builder)


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


def _with_lock(monkeypatch, tmp_path, name, text):
    (tmp_path / images.LOCK_FILES[name]).write_text(text)
    monkeypatch.setattr(images, "LOCK_DIR", tmp_path)
    return images.image_fingerprint(name)


@pytest.mark.parametrize("name", NAMES)
def test_locks_have_no_cwd_dependent_paths(name):
    text = images.lock_path(name).read_text()
    assert not re.search(r"(^|\s)/(Users|home|private|tmp|var)/", text), name
    assert "#    uv run python scripts/lock_images.py\n" in text
    assert f"-r src/flora_deploy/modal_app/locks/{images.LOCK_FILES[name][:-4]}.in" in text


def test_fingerprint_ignores_lock_comments_and_cwd(tmp_path, monkeypatch):
    """A lock compiled elsewhere (absolute ``# via`` paths, another header) fingerprints the same,
    and so does computing it from another working directory."""
    before = images.image_fingerprint("img_cpu")
    text = images.lock_path("img_cpu").read_text()
    moved = text.replace("-r src/flora_deploy", "-r /somewhere/else/src/flora_deploy").replace(
        "#    uv run python scripts/lock_images.py", "#    uv pip compile /abs/cpu.in -o x.txt"
    )
    assert moved != text
    monkeypatch.chdir(tmp_path)
    assert images.image_fingerprint("img_cpu") == before
    assert _with_lock(monkeypatch, tmp_path, "img_cpu", moved + "\n# trailing\n\n") == before


def test_fingerprint_tracks_lock_requirements_and_bases(tmp_path, monkeypatch):
    before = images.image_fingerprint("img_cpu")
    text = images.lock_path("img_cpu").read_text()
    changed = text.replace(f"numpy=={RUNTIME['numpy']}", "numpy==9.9.9")
    assert changed != text
    assert _with_lock(monkeypatch, tmp_path, "img_cpu", changed) != before
    monkeypatch.undo()
    assert images.image_fingerprint("img_cpu") == before
    for key, value in (("modal_image_builder_version", "2099.01"), ("modal_client", "9.9.9")):
        with monkeypatch.context() as m:
            m.setitem(RUNTIME, key, value)
            assert images.image_fingerprint("img_cpu") != before, key
    for attr, value in (
        ("UV_IMAGE", "ghcr.io/astral-sh/uv@sha256:" + "0" * 64),
        ("BASES", {**images.BASES, "img_cpu": {"from": "x"}}),
        ("INSTALL_OPTIONS", {**images.INSTALL_OPTIONS, "img_cpu": "--x"}),
    ):
        with monkeypatch.context() as m:
            m.setattr(images, attr, value)
            assert images.image_fingerprint("img_cpu") != before, attr


def test_lock_digest_normalizes_only_comments():
    a = "foo==1 \\\n    --hash=sha256:aa\n    # via bar\n"
    assert images.lock_digest(a) == images.lock_digest("# header\n" + a.replace("bar", "baz"))
    assert images.lock_digest(a) != images.lock_digest(a.replace("aa", "ab"))


def test_locks_are_shipped_with_source():
    """image_fingerprint() reads the locks, so the source mount must include them."""
    from modal import FilePatternMatcher

    ignore = ~FilePatternMatcher(*images.SOURCE_PATTERNS)
    pkg_root = Path(images.__file__).parents[2]  # .../src
    for name in NAMES:
        assert not ignore(images.lock_path(name)), name
        assert not ignore(images.lock_path(name).relative_to(pkg_root)), name
    assert ignore(images.lock_path("img_cpu").with_suffix(".in"))
    assert not ignore(Path(images.__file__))
