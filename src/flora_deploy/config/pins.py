"""Pinned upstream revisions and runtime versions.

Every pin is a full commit SHA, never a branch name. Changing any value here changes the config
hash (PLAN.md D11) and requires re-certification.

Two pins per NVIDIA repo family:
- weights: the revision NVIDIA's own ``batch_utils/models.json`` validates against.
- code: the revision that ships ``batch_utils/`` (added after the weights were staged).
The Rewarder loads weights from the same snapshot as its code, so it has one pin.

Checked 2026-10-21: every safetensors LFS hash at the TempSeg and Captioner weights pins equals
the one at their code pins, and the other files' sizes match. The two pins carry the same weights.
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
    # Expected digests published by NVIDIA, verified after download.
    expected: tuple[tuple[str, str], ...] = ()


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

PINS: tuple[RepoPin, ...] = (
    RepoPin(
        key="tempseg",
        repo="nvidia/Flora-TempSeg",
        revision="114a1076f70eb53f4a58ac045505db3087e79347",
        role="weights",
        allow_patterns=_NV_WEIGHT_FILES,
        expected=(
            ("adapter_sha256", "1d80fe544e4cd1a7ff6b0ea4be71e988972408dab81d4c8af72e3a628739d6c3"),
        ),
    ),
    RepoPin(
        key="tempseg_code",
        repo="nvidia/Flora-TempSeg",
        revision="78a0e4ac75d33509f077fdf985ebb51153920e7c",
        role="code",
        allow_patterns=_NV_CODE_FILES,
    ),
    RepoPin(
        key="tempseg_base",
        repo="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        role="base",
        allow_patterns=("*.safetensors", "*.json", "*.jinja"),
    ),
    RepoPin(
        key="captioner",
        repo="nvidia/Flora-Captioner",
        revision="fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6",
        role="weights",
        allow_patterns=_NV_WEIGHT_FILES,
        expected=(
            (
                "source_manifest_identity",
                "2ab6cafd286df7f446bfb84fbc4af7c54773d5e890dd277ac5a0453d0d4fb778",
            ),
        ),
    ),
    RepoPin(
        key="captioner_code",
        repo="nvidia/Flora-Captioner",
        revision="f736ca36925ae55d2f7915993ed1bc9f87288bb3",
        role="code",
        allow_patterns=_NV_CODE_FILES,
    ),
    RepoPin(
        key="rewarder",
        repo="nvidia/Flora-Rewarder",
        revision="7de68d22c4a9decec5ddb7f86ddcaee8f82d3367",
        role="weights+code",
        allow_patterns=None,
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
}
