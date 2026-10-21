"""Pinned upstream revisions and runtime versions.

Every pin is a full commit SHA, never a branch name. Changing any value here changes the config
hash (PLAN.md D11) and requires re-certification.

Two pins per NVIDIA repo family:
- weights: the revision NVIDIA's own ``batch_utils/models.json`` validates against.
- code: the revision that ships ``batch_utils/`` (added after the weights were staged).
The Rewarder loads weights from the same snapshot as its code, so it has one pin.

Checked 2026-10-21: every safetensors LFS hash at the TempSeg and Captioner weights pins equals
the one at their code pins, and the other files' sizes match. The two pins carry the same weights.

Trust anchors (PLAN.md D11). Each pin carries values pinned in this repo, so a changed Hub tree or
a changed NVIDIA manifest fails closed instead of being trusted from the same API that serves it:
- ``plan_digest``: ``prep.verify.plan_digest`` of the resolved file list (repo, revision,
  allow_patterns and, per file, path, size, LFS sha256 and git blob id). Checked before anything
  is downloaded, and again in the receipt.
- ``expected["manifest_sha256"]``: sha256 of the bytes of NVIDIA's own manifest
  (``SOURCE_MANIFEST.sha256`` for TempSeg/Captioner weights, ``SHA256SUMS`` for the Rewarder).
  Every per-file manifest line is then checked against the downloaded files.

How the values were obtained (2026-10-21, read-only Hub calls): ``plan_snapshot`` +
``plan_digest`` against ``HfApi().list_repo_tree`` at each pinned revision, and
``hf_hub_download`` of each manifest at its pinned revision, hashed with sha256. The three
manifest digests agree with an independent computation in the WS-A review. The fetched NVIDIA code
pins (``~/.cache/flora-deploy/nv``) contain no weight manifests, so they gave no second source.

``code_digests`` (2026-10-21): computed from the fetched code pins, as sha256 of the two manifest
files (TempSeg and Captioner) and with ``nv.linker.code_tree_sha256`` for the Rewarder (11 files).
Before hashing, every fetched ``batch_utils/`` file was compared with the object id the Hub sent
for it at download time (the ETag in ``.cache/huggingface/download/*.metadata``, recomputed as
sha1 of ``b"blob <len>\\0" + bytes``): 43 + 43 + 11 files, all equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["weights", "code", "weights+code", "base"]


@dataclass(frozen=True)
class RepoPin:
    key: str
    repo: str
    revision: str
    role: Role
    # Passed to snapshot_download; None downloads the whole snapshot.
    allow_patterns: tuple[str, ...] | None = None
    # Expected digests (keys derived as documented in ``prep/verify.py``), verified after download.
    expected: tuple[tuple[str, str], ...] = ()
    # Pinned ``prep.verify.plan_digest`` of the resolved file list; None fails closed at prepare.
    plan_digest: str | None = None
    # Pinned digests of NVIDIA's *code* (``batch_utils/``), checked by ``nv.linker.load_family``
    # before any NVIDIA module runs. Keys: ``code_manifest_sha256`` / ``vendor_manifest_sha256``
    # (sha256 of the bytes of ``batch_utils/CODE_MANIFEST.json`` / ``batch_utils/vendor/
    # MANIFEST.json``; TempSeg and Captioner code pins) and ``code_tree_sha256`` (Rewarder, which
    # ships no manifest: ``nv.linker.code_tree_sha256``). Not part of ``prep.verify.pin_record``.
    code_digests: tuple[tuple[str, str], ...] = ()


# File list NVIDIA's reference loader downloads for TempSeg/Captioner weights.
_NV_WEIGHT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
    "MODEL_PROVENANCE.json",
    "SOURCE_MANIFEST.sha256",
)

_NV_CODE_FILES = ("batch_utils/*", "third_party/*", "LICENSE*", "README.md")

# TempSeg and Captioner code pins ship byte-identical batch_utils/, hence identical manifests.
_SEGCAP_CODE_DIGESTS = (
    ("code_manifest_sha256", "802568893397ea470156ce67b64a9beecb3d9e06f53ca8914f8c92278221681d"),
    ("vendor_manifest_sha256", "91ce5d8a5f13a583d48384c5de7f46696f5cd896ad2c1f3ce0eead9a1acae6d8"),
)

PINS: tuple[RepoPin, ...] = (
    RepoPin(
        key="tempseg",
        repo="nvidia/Flora-TempSeg",
        revision="114a1076f70eb53f4a58ac045505db3087e79347",
        role="weights",
        allow_patterns=_NV_WEIGHT_FILES,
        expected=(
            ("adapter_sha256", "1d80fe544e4cd1a7ff6b0ea4be71e988972408dab81d4c8af72e3a628739d6c3"),
            ("manifest_sha256", "df85ebc4be7e56bffe1100b8b623498864a6d93201282176ff47bcab66acfd0b"),
        ),
        plan_digest="d2ca1fa2dd54a1fa38b7cec2a5f0fa7ccab541ae1660ce9a62e7d15fba59381d",
    ),
    RepoPin(
        key="tempseg_code",
        repo="nvidia/Flora-TempSeg",
        revision="78a0e4ac75d33509f077fdf985ebb51153920e7c",
        role="code",
        allow_patterns=_NV_CODE_FILES,
        plan_digest="d409703421ecbd1510e4fd5e0148de1d4320d7c06984ff9a9dec3fc6734b0c46",
        code_digests=_SEGCAP_CODE_DIGESTS,
    ),
    RepoPin(
        key="tempseg_base",
        repo="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        role="base",
        allow_patterns=("*.safetensors", "*.json", "*.jinja"),
        # No NVIDIA manifest covers the base model: this is its only repo-pinned content anchor.
        plan_digest="199dd29d2572997cc501d0ab4ffd95c4970bfed5dde2faad52b20fc3a5c4e4f6",
    ),
    RepoPin(
        key="captioner",
        repo="nvidia/Flora-Captioner",
        revision="fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6",
        role="weights",
        allow_patterns=_NV_WEIGHT_FILES,
        expected=(
            ("manifest_sha256", "75241af6f42d26f955f207f8cfb98ba6e6d05c303f74a7c85bde30a6cf997ad8"),
            # Secondary consistency check only (see prep/verify.py): a field of
            # MODEL_PROVENANCE.json, whose bytes the pinned manifest above already binds.
            (
                "source_manifest_identity",
                "2ab6cafd286df7f446bfb84fbc4af7c54773d5e890dd277ac5a0453d0d4fb778",
            ),
        ),
        plan_digest="c5ae91aadfde343cc550279a3a05a7c491f647fea424f5ba0bc6d886ba68655b",
    ),
    RepoPin(
        key="captioner_code",
        repo="nvidia/Flora-Captioner",
        revision="f736ca36925ae55d2f7915993ed1bc9f87288bb3",
        role="code",
        allow_patterns=_NV_CODE_FILES,
        plan_digest="f40914ffff9509998c443095ca0f66a80cdc581922074f4ee50c1482755dfc28",
        code_digests=_SEGCAP_CODE_DIGESTS,
    ),
    RepoPin(
        key="rewarder",
        repo="nvidia/Flora-Rewarder",
        revision="7de68d22c4a9decec5ddb7f86ddcaee8f82d3367",
        role="weights+code",
        allow_patterns=None,
        expected=(
            ("manifest_sha256", "b20a073e6d792365248aeb29f3a1b339a434ee1309ef26a5cd2df683941ebe56"),
        ),
        plan_digest="1640693217ef3937b81868bff1cc7edad3e86de74e1510fb238663a26a7a6160",
        code_digests=(
            ("code_tree_sha256",
             "a0659828417584abe63dea21df28c81e394fff16058418f17e8fc09de225629d"),
        ),
    ),
)

PINS_BY_KEY: dict[str, RepoPin] = {p.key: p for p in PINS}

# Runtime versions for the Modal images (PLAN.md §7). Not used locally yet.
RUNTIME = {
    "vllm": "0.24.0",
    "transformers": "5.8.1",
    "torch_reference": "2.11.0",
    "peft": "0.18.1",
    "qwen_vl_utils": "0.0.14",
    "decord": "0.6.0",
    "av": "16.0.1",
    "python_image": "3.12",
    # Added for the Modal images (WS-A). The *_reference pins mirror NVIDIA's TempSeg/Captioner
    # batch_utils/requirements.txt exactly; Pillow/pyarrow are what uv.lock resolved.
    "torchvision_reference": "0.26.0",
    "huggingface_hub_reference": "1.33.0",
    "accelerate": "1.12.0",
    "safetensors": "0.7.0",
    "pillow": "12.3.0",
    # One numpy in all three images: vLLM's mistral-common caps numpy<2.4 on CPython 3.12, and
    # 2.3.5 is inside NVIDIA's "numpy<3" (review A6).
    "numpy": "2.3.5",
    "pyarrow": "26.0.0",
    "huggingface_hub": ">=1.29,<2.0",
    # CUDA 13.0 matches the torch 2.11.0 PyPI wheels (cu130); Modal's host driver is CUDA 13.3.
    "cuda_base_vllm": "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    # Image reproducibility (review M3; see modal_app/images.py). Digest of the tag's multi-arch
    # index on Docker Hub, read 2026-10-21 (its linux/amd64 manifest is
    # sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c).
    "cuda_base_vllm_digest": (
        "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
    ),
    # Fixes debian_slim's Debian release and Python micro version and add_python's build.
    "modal_image_builder_version": "2025.06",
    # uv binary used both to compile the image locks and to install them in the image.
    "uv": "0.12.22",
    "hf_xet": "1.7.0",
    # img_hfref follows NVIDIA's TempSeg/Captioner install recipe (batch_utils/README.md): torch and
    # torchvision from https://download.pytorch.org/whl/cu128, then requirements.txt. The lock
    # takes exactly these two cp312 x86_64 wheels by URL and sha256 (an override); everything
    # else, including their nvidia-*-cu12 CUDA 12.8 and triton deps, comes from PyPI (review A1).
    # (uv's --torch-backend took triton from another index at install than at lock time, so the
    # hashes did not match.)
    "torch_index_hfref": "https://download.pytorch.org/whl/cu128",
    "torch_hfref": "2.11.0+cu128",
    "torch_hfref_sha256": "d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997",
    "torchvision_hfref": "0.26.0+cu128",
    "torchvision_hfref_sha256": (
        "ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d"
    ),
    # Bases pinned by digest (multi-arch index digests, read 2026-10-21; review A2). The Python base
    # replaces Modal's debian_slim, whose layer adds unpinned apt and pip steps.
    "python_base": "python:3.12.10-slim-bookworm",
    "python_base_digest": (
        "sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db"
    ),
    "uv_image_digest": (
        "sha256:f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc"
    ),
    # The Modal client renders every builder step and picks add_python's standalone build
    # (modal/mount.py PYTHON_STANDALONE_VERSIONS); the image builders refuse another client.
    "modal_client": "1.6.2",
    "python_standalone_release": "20240107",
    "python_standalone": "3.12.1",
}


# Derived artifacts we build from a pinned snapshot (OPT-2, ``docs/captioner-bf16.md``). The
# runtime refuses a derived copy whose receipt bytes do not hash to ``receipt_sha256`` (``None``
# fails closed: nothing is trusted until a receipt is pinned here), whose converter version
# differs, or whose recorded toolchain differs from ``proven_with`` (the versions the GPU
# identity proof ran with; local build tags such as ``+cu130`` are not compared).
@dataclass(frozen=True)
class DerivedPin:
    key: str
    source_key: str  # PINS_BY_KEY key of the snapshot it is derived from
    converter_version: int
    receipt_sha256: str | None
    proven_with: tuple[tuple[str, str], ...] = ()


DERIVED_PINS: tuple[DerivedPin, ...] = (
    # Receipt written 2026-10-21 by ``convert_app.py --step convert`` to
    # flora-weights:/derived/captioner-bf16@fef0bbfe.../derived_receipt.json (value recorded in
    # docs/captioner-bf16.md); identity proven on an H200 with torch 2.11.0+cu130,
    # transformers 5.8.1 and safetensors 0.7.0.
    DerivedPin(
        key="captioner-bf16",
        source_key="captioner",
        converter_version=1,
        receipt_sha256="0e91e467c7ffdb636573e779672d11b203571f7869ca94317f1005eb700d9c65",
        proven_with=(("torch", "2.11.0"), ("transformers", "5.8.1"), ("safetensors", "0.7.0")),
    ),
)

DERIVED_PINS_BY_KEY: dict[str, DerivedPin] = {p.key: p for p in DERIVED_PINS}
