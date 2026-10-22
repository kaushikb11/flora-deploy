"""The packaged templates name only GPUs that a deploy accepts for each shard kind.

A GPU is accepted for (preset, kind) when it has the memory the preset needs
(``config.presets.gpu_problem``) and, for the ``balanced`` vLLM kinds, the engine has scheduler
limits for it (``engines.offline_llm.gpu_limits``, which ``flora deploy`` builds the engine
spec from). The template comments list GPUs per kind as `` `kind` (Engine: A, B or C) ``.
"""

from __future__ import annotations

import re

import pytest

from flora_deploy.cli.init import TEMPLATES, template_text
from flora_deploy.config.presets import GPU_MEMORY_GIB, gpu_problem
from flora_deploy.config.schema import loads_config
from flora_deploy.engines.offline_llm import SHARD_ENGINE_KINDS, EngineArgsError, gpu_limits

_KINDS = ("segcap", "segment", "caption", "reward.terminal", "reward.dense", "reward.rubric")
# Longest names first, so "H100!" and "A100-80GB" win over "H100" and "A100".
_GPU = "|".join(re.escape(g) for g in sorted(GPU_MEMORY_GIB, key=len, reverse=True))
_GPU_RE = re.compile(rf"(?<![\w-])({_GPU})(?![\w-])")
_KIND_LIST = re.compile(r"`(" + "|".join(re.escape(k) for k in _KINDS) + r")` \(([^)]*)\)")


def accepted(preset: str, kind: str, gpu: str) -> bool:
    if gpu_problem(preset, kind, gpu, 1) is not None:
        return False
    if preset == "balanced" and kind in SHARD_ENGINE_KINDS:
        try:
            gpu_limits(SHARD_ENGINE_KINDS[kind], gpu)
        except EngineArgsError:
            return False
    return True


def comment_text(text: str) -> str:
    """The template's comments joined into one string (``#`` and line breaks removed)."""
    parts = [line.split("#", 1)[1] for line in text.splitlines() if "#" in line]
    return " ".join(p.strip() for p in parts)


def listed_gpus(text: str) -> list[tuple[str, str]]:
    """(kind, GPU) for every GPU a comment lists for a shard kind."""
    return [(kind, g) for kind, inner in _KIND_LIST.findall(comment_text(text))
            for g in _GPU_RE.findall(inner)]


def test_accepted_matches_the_deploy_refusals():
    assert accepted("balanced", "caption", "H100!") and accepted("balanced", "caption", "H200")
    assert not accepted("balanced", "caption", "A100-80GB")  # no Captioner engine limits
    assert not accepted("balanced", "caption", "B200")
    assert accepted("balanced", "segment", "A100-80GB")
    assert not accepted("balanced", "segment", "B200")
    assert not accepted("exact", "segcap", "H100")  # 81 GiB minimum


def test_the_balanced_template_lists_gpus_per_kind():
    found = listed_gpus(template_text("balanced_lerobot"))
    assert {k for k, _ in found} >= {"segment", "caption"}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_list_only_accepted_gpus(name):
    text = template_text(name)
    preset = loads_config(text).preset
    bad = [(k, g) for k, g in listed_gpus(text) if not accepted(preset, k, g)]
    assert not bad, f"{name}: lists GPUs {preset} cannot deploy: {bad}"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_resources_use_accepted_gpus(name):
    cfg = loads_config(template_text(name))
    for kind, res in (cfg.resources or {}).items():
        if res.gpu is not None:
            assert accepted(cfg.preset, kind, res.gpu), (name, kind, res.gpu)
