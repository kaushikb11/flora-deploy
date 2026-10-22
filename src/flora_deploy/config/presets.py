"""Requirement presets (PLAN.md §2, D9, D16): which shard kinds run which stage, on which GPU.

A preset maps each pipeline stage to a shard kind and gives every shard kind a default pool
(GPU, count, cores, memory). ``flora.yaml`` ``stages:`` overrides any of these per shard kind.

- ``exact``: NVIDIA's own pipeline. Segment and caption run together in one ``segcap`` worker on
  an ``H200`` (TempSeg and Captioner both resident; ``docs/reference-smoke.md``: 8 cores, 96 GiB,
  80.95 GiB HBM high-water mark). Rewards run NVIDIA's offline vLLM engine on ``H100!``
  (``docs/cold-start.md``: 8 cores, >= 72 GiB so the 52 GiB snapshot stays in the page cache).
- ``balanced``: vLLM 0.24.0 engines, stage-separated pools (D8). TempSeg on ``L40S`` (6 cores,
  §6.7), Captioner on ``H100!`` (8 cores; memory >= BF16 snapshot ≈ 19 GiB + 16 GiB, plus
  decode/caption-cache headroom), rewards as in ``exact``.
- ``budget`` / ``interactive``: declared (D9) but not available yet; selecting one is an error.

Certification (D9, D16): ``exact`` is the reference by construction. Every other preset needs a
Gate certificate matching (shard kind, GPU, engine spec) before it may run (``find_certificates``;
the registry is ``config/certificates/``, the match rule ``validate/certkey.py``). Without a
``certified`` one (``provisional`` does not count), ``balanced`` is refused unless the config
opts in with ``allow_uncertified: true`` (its records are then an uncertified stack). A config
whose every stage is certified for its embodiment needs no opt-in; otherwise the refusal names
each stage that is not covered and why (``coverage`` / ``refusal_message``, in plain words).

GPU overrides are checked against ``GPU_MEMORY_GIB`` (Modal GPU types) and a per-shard-kind
minimum (``min_gpu_memory_gib``): exact ``segcap`` keeps TempSeg and Captioner resident (80.95 GiB
HBM high-water), so it needs an H200/B200-class GPU; rewards need an 80 GB-class GPU.

Memory values are the pre-P0 defaults; P0 measurements replace them (§6.7).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace

from flora_deploy.jobs.settings import PoolSettings, Preset
from flora_deploy.jobs.types import ShardKind
from flora_deploy.records import Stage

_REWARDS: dict[ShardKind, PoolSettings] = {
    "reward.terminal": PoolSettings(gpu="H100!", cpu_cores=8.0, memory_gib=80.0),
    "reward.dense": PoolSettings(gpu="H100!", cpu_cores=8.0, memory_gib=80.0),
    # Rubric runs NVIDIA's settings at TP=2 (PLAN.md §7).
    "reward.rubric": PoolSettings(gpu="H100!", gpu_count=2, cpu_cores=8.0, memory_gib=96.0),
}


@dataclass(frozen=True)
class EngineIdentity:
    """What produces a shard kind's outputs; hashed (D11). GPU is deliberately not here.

    ``patches`` lists the output-identical runtime wraps / engine patches applied (D17); it and
    the arg/sampling/mm_kwargs placeholders are filled in as S1 and P1b-1 land."""

    name: str
    version: dict[str, str]
    image: str  # modal_app.images name, e.g. "img_hfref"
    patches: tuple[str, ...] = ()
    engine_args: dict = field(default_factory=dict)  # placeholder until P1b-1
    sampling: dict = field(default_factory=dict)  # placeholder until P1b-1
    mm_kwargs: dict = field(default_factory=dict)  # placeholder until P1b-1


@dataclass(frozen=True)
class PresetSpec:
    name: Preset
    available: bool
    stage_kinds: dict[Stage, ShardKind]  # pipeline stage -> shard kind that produces it
    pools: dict[ShardKind, PoolSettings]
    engines: dict[ShardKind, EngineIdentity]
    # NVIDIA pin keys (config/pins.py) each shard kind depends on.
    pins: dict[ShardKind, tuple[str, ...]]
    # Derived-weight receipts (prep receipts) each shard kind needs, by name.
    derived_weights: dict[ShardKind, tuple[str, ...]]
    note: str = ""


def _engines() -> tuple[EngineIdentity, EngineIdentity, EngineIdentity]:
    from flora_deploy.config.pins import RUNTIME
    from flora_deploy.nv.capture import REQUIRED_VLLM_ENGINE_KWARGS

    # Review 3 A9: the versions img_hfref actually installs (locks/hfref.txt): torch is the
    # cu128 wheel, and qwen_vl_utils / decord are not in that image (NVIDIA's HF path does not
    # use them), so they are not part of this identity. The image fingerprint covers the rest.
    hf = EngineIdentity(
        name="nvidia-hf",
        version={"transformers": RUNTIME["transformers"], "torch": RUNTIME["torch_hfref"],
                 "peft": RUNTIME["peft"]},
        image="img_hfref",
    )
    nv_vllm = EngineIdentity(name="nvidia-vllm-offline", version={"vllm": RUNTIME["vllm"]},
                             image="img_vllm")
    # Every vLLM engine that carries Flora video must use these (Gate C), so they are hashed.
    vllm = EngineIdentity(name="vllm-offline", version={"vllm": RUNTIME["vllm"]},
                          image="img_vllm", engine_args=dict(REQUIRED_VLLM_ENGINE_KWARGS))
    return hf, nv_vllm, vllm


_REWARD_PINS: tuple[str, ...] = ("rewarder",)
_SEG_PINS: tuple[str, ...] = ("tempseg", "tempseg_code", "tempseg_base")
_CAP_PINS: tuple[str, ...] = ("captioner", "captioner_code")


def _build() -> dict[str, PresetSpec]:
    hf, nv_vllm, vllm = _engines()
    reward_kinds: tuple[ShardKind, ...] = ("reward.terminal", "reward.dense", "reward.rubric")
    reward_stage_kinds: dict[Stage, ShardKind] = {k: k for k in reward_kinds}  # type: ignore[misc]
    reward_engines = dict.fromkeys(reward_kinds, nv_vllm)
    reward_pins = dict.fromkeys(reward_kinds, _REWARD_PINS)
    exact = PresetSpec(
        name="exact",
        available=True,
        stage_kinds={"segment": "segcap", "caption": "segcap", **reward_stage_kinds},
        pools={"segcap": PoolSettings(gpu="H200", cpu_cores=8.0, memory_gib=96.0), **_REWARDS},
        engines={"segcap": hf, **reward_engines},
        pins={"segcap": _SEG_PINS + _CAP_PINS, **reward_pins},
        # OPT-2: NVIDIA's loader reads our BF16 Captioner (proven identical, captioner-bf16.md).
        derived_weights={"segcap": ("captioner_bf16",)},
        note="reference by construction: NVIDIA HF code on H200, NVIDIA's vLLM Rewarder",
    )
    balanced = PresetSpec(
        name="balanced",
        available=True,
        stage_kinds={"segment": "segment", "caption": "caption", **reward_stage_kinds},
        pools={
            # the shapes balanced_lab validated (review 10 M2): TempSeg L40S 8 cores 48 GiB,
            # Captioner H100 8 cores 96 GiB (55 GB weights hashed in full while vLLM loads)
            "segment": PoolSettings(gpu="L40S", cpu_cores=8.0, memory_gib=48.0),
            "caption": PoolSettings(gpu="H100!", cpu_cores=8.0, memory_gib=96.0),
            **_REWARDS,
        },
        engines={"segment": vllm, "caption": vllm, **reward_engines},
        pins={"segment": _SEG_PINS, "caption": _CAP_PINS, **reward_pins},
        derived_weights={"segment": ("tempseg_merged",), "caption": ("captioner_bf16",)},
        note="vLLM 0.24.0, stage-separated pools; needs Gate certificates (config/certificates/)",
    )
    unavailable = {
        "budget": "planned for P2 (FP8 / B200 kernels; uncertified until validated)",
        "interactive": "planned for v0.4 (online API, AsyncLLM servers)",
    }
    out: dict[str, PresetSpec] = {"exact": exact, "balanced": balanced}
    for name, note in unavailable.items():
        out[name] = PresetSpec(name=name, available=False, stage_kinds={}, pools={}, engines={},
                               pins={}, derived_weights={}, note=note)  # type: ignore[arg-type]
    return out


PRESETS: dict[str, PresetSpec] = _build()
AVAILABLE_PRESETS: tuple[str, ...] = tuple(n for n, p in PRESETS.items() if p.available)


class PresetUnavailableError(ValueError):
    pass


def get_preset(name: str) -> PresetSpec:
    """The preset spec, or a clear error for unknown / not-yet-available presets."""
    spec = PRESETS.get(name)
    if spec is None:
        raise PresetUnavailableError(
            f"unknown preset {name!r}; choose one of {', '.join(AVAILABLE_PRESETS)}")
    if not spec.available:
        raise PresetUnavailableError(
            f"preset {name!r} is not available yet ({spec.note}); "
            f"use one of {', '.join(AVAILABLE_PRESETS)}")
    return spec


def shard_kinds_for(preset: PresetSpec, stages: tuple[Stage, ...] | list[Stage]) -> list[ShardKind]:
    """Shard kinds a pipeline needs, deduplicated, in first-use order."""
    out: list[ShardKind] = []
    for s in stages:
        k = preset.stage_kinds[s]
        if k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------------------------------------
# GPUs and certificates

# Modal GPU type -> usable memory per GPU in GiB (approximate, as the driver reports it).
GPU_MEMORY_GIB: dict[str, float] = {
    "T4": 15.0,
    "L4": 22.0,
    "A10": 22.0,
    "A10G": 22.0,
    "A100": 39.5,
    "A100-40GB": 39.5,
    "A100-80GB": 79.2,
    "L40S": 44.5,
    "H100": 79.6,
    "H100!": 79.6,
    "H200": 139.8,
    "B200": 178.0,
}

# Minimum GPU memory per container (GiB, summed over its GPUs), per (preset, shard kind).
# exact segcap: 80.95 GiB HBM high-water (docs/reference-smoke.md), above an H100's 79.6 GiB.
# Rewards: NVIDIA's vLLM recipe is sized for 80 GB GPUs (README, docs/cold-start.md).
_REWARD_MIN_GIB = 79.0
_MIN_GPU_GIB: dict[tuple[str, str], float] = {
    ("exact", "segcap"): 81.0,
    ("balanced", "segment"): 44.0,  # L40S class (§6.7)
    ("balanced", "caption"): 79.0,  # H100 class
}


def min_gpu_memory_gib(preset: str, kind: str) -> float:
    if kind.startswith("reward."):
        return _REWARD_MIN_GIB
    return _MIN_GPU_GIB[(preset, kind)]


def gpu_problem(preset: str, kind: str, gpu: str, gpu_count: int) -> str | None:
    """Why ``gpu_count`` x ``gpu`` cannot run ``kind`` under ``preset``, or None if it can."""
    mem = GPU_MEMORY_GIB.get(gpu)
    if mem is None:
        return f"unknown GPU {gpu!r}; use one of {', '.join(GPU_MEMORY_GIB)}"
    need = min_gpu_memory_gib(preset, kind)
    if mem * gpu_count < need:
        fits = [g for g, m in GPU_MEMORY_GIB.items() if m * gpu_count >= need]
        return (f"{kind} needs >= {need:g} GiB of GPU memory per container, but {gpu_count} x "
                f"{gpu} has {mem * gpu_count:g} GiB (use one of {', '.join(fits)})")
    return None


@dataclass(frozen=True)
class Certificate:
    """A Gate certificate (D16): shard kind ``kind`` on ``gpu`` under ``preset`` is faithful."""

    id: str
    preset: str
    kind: str
    gpu: str
    verdict: str = "certified"  # certified | provisional | fail (validate/certify.py)


def needs_certificate(preset: str) -> bool:
    """Everything but ``exact`` (the reference by construction) needs certificates."""
    return preset != "exact"


def find_certificates(preset: str, pools: Mapping[str, PoolSettings], *,
                      modes: Mapping[str, Mapping] | None = None,
                      **kwargs) -> dict[str, Certificate]:
    """The ``certified`` certificate matching each (shard kind, GPU, engine spec) of ``pools``,
    by shard kind (``validate.certkey.matching``; pass ``**certkey.lookup_context(cfg, pools,
    derived_content=...)``: embodiment, modes, derived content digests). The deciding (newest)
    certificate must be ``certified``: provisional ones never count, and a newer ``fail``
    revokes an older ``certified``.

    A kind that is the reference by identity (``certkey.reference_identical``: its live key
    equals ``exact``'s but for the preset name, e.g. ``balanced``'s rewards, which run NVIDIA's
    own engine) needs no certificate: it is returned as ``Certificate(REFERENCE_ID, ...,
    verdict="reference")``, so every gate that asks this function (config validation, the
    deploy factory) applies the same rule."""
    from flora_deploy.validate.certkey import matching, reference_kinds

    out: dict[str, Certificate] = {}
    for kind, certs in matching(preset, pools, modes=modes, **kwargs).items():
        best = certs[0]
        if best["verdict"] == "certified":
            out[kind] = Certificate(best["id"], preset, kind, best["gpu"], best["verdict"])
    if preset != "exact":
        for kind in reference_kinds(preset, pools, modes=modes, **kwargs):
            out[kind] = Certificate(REFERENCE_ID, preset, kind, pools[kind].gpu, "reference")
    return out


# ``Certificate.id`` of a kind that is the reference by identity (no certificate file).
REFERENCE_ID = "reference:exact-identical"


# ---------------------------------------------------------------------------------------------
# Coverage in plain words: which stages of a config a certificate covers, and why not


@dataclass(frozen=True)
class Coverage:
    """How far the certificates cover one shard kind's pool for the config's embodiment.

    ``state``: ``reference`` (``exact``), ``certified``, ``other_embodiment`` (certified on this
    GPU and stack for ``embodiments``, not for this one), ``other_build`` (a certificate exists
    for this kind and GPU, but for another stack: engine, image, pins, modes, code or weights),
    ``provisional``, ``fail``, ``unverified_weights`` (the derived weights' content digests are
    unknown, so a certificate cannot be checked) or ``none``. ``status`` is ``certkey.status``'s
    raw value."""

    kind: str
    gpu: str
    state: str
    status: str
    certificate: str | None = None
    embodiments: tuple[str, ...] = ()

    @property
    def unlocks(self) -> bool:
        return self.state in ("reference", "certified")


def coverage(preset: str, pools: Mapping[str, object], *, embodiment: str | None,
             **match_kw) -> dict[str, Coverage]:
    """Shard kind -> :class:`Coverage` of ``pools`` (kind -> anything with ``.gpu``) under
    ``preset`` for ``embodiment``. ``match_kw``: as for ``certkey.matching`` (modes,
    derived_content, cert_dir, ...)."""
    from flora_deploy.validate import certkey

    raw = certkey.status(preset, pools, embodiment=embodiment, **match_kw)
    certs = None
    out: dict[str, Coverage] = {}
    for kind, r in raw.items():
        gpu, st, cid = str(r.get("gpu")), str(r.get("status", "none")), r.get("certificate")
        if st in ("reference", "certified", "provisional", "fail"):
            out[kind] = Coverage(kind, gpu, st, st, cid,
                                 (embodiment,) if st == "certified" and embodiment else ())
            continue
        if certs is None:
            certs = certkey.load_certificates(match_kw.get("cert_dir"))
        near = [c for c in certs if c.get("preset") == preset and c.get("kind") == kind
                and c.get("gpu") == gpu]
        if not near:
            out[kind] = Coverage(kind, gpu, "none", st)
            continue
        if st == "unverified_weights":
            out[kind] = Coverage(kind, gpu, st, st)
            continue
        others = sorted({str(e) for c in near if c.get("verdict") == "certified"
                         for e in c.get("covered_pools") or ()} - {embodiment})
        one = {kind: pools[kind]}
        ok = tuple(e for e in others
                   if kind in find_certificates(preset, one, embodiment=e,  # type: ignore[arg-type]
                                                **match_kw))
        out[kind] = (Coverage(kind, gpu, "other_embodiment", st, embodiments=ok) if ok
                     else Coverage(kind, gpu, "other_build", st))
    return out


_NOT_COVERED = {
    "none": "no certificate",
    "provisional": "only a provisional certificate",
    "fail": "its certificate failed",
    "other_build": "certified for another build (engine, image, models, run modes, code or "
                   "weights changed)",
    "unverified_weights": "certificate not checked without the prepared weights (run `flora "
                          "prepare`)",
}


def coverage_label(c: Coverage, embodiment: str | None) -> str:
    """The status of one stage in plain words, e.g. ``certified (human, L40S)``, ``not
    certified`` or ``certified for human, not for yam``."""
    if c.state == "reference":
        return "reference (NVIDIA's own engine, no certificate needed)"
    if c.state == "certified":
        return f"certified ({embodiment}, {c.gpu})"
    if c.state == "other_embodiment":
        return f"certified for {', '.join(c.embodiments)}, not for {embodiment}"
    if c.state == "none":
        return "not certified"
    return f"not certified ({_NOT_COVERED.get(c.state, c.state)})"


def uncovered_parts(cov: Mapping[str, Coverage], embodiment: str | None) -> list[str]:
    """What keeps a config from being fully certified, one phrase per problem, e.g.
    ``reward.terminal on H100!: no certificate`` and ``embodiment yam: segment/caption certified
    for human only`` (kinds certified for the same other embodiments are named together)."""
    parts: list[str] = []
    by_emb: dict[tuple[str, ...], list[str]] = {}
    for c in cov.values():
        if c.unlocks:
            continue
        if c.state == "other_embodiment":
            by_emb.setdefault(c.embodiments, []).append(c.kind)
        else:
            parts.append(f"{c.kind} on {c.gpu}: {_NOT_COVERED.get(c.state, 'not certified')}")
    for embs, kinds in by_emb.items():
        parts.append(f"embodiment {embodiment}: {'/'.join(kinds)} certified for "
                     f"{', '.join(embs)} only")
    return parts


def uncertified_message(preset: str, cov: Mapping[str, Coverage],
                        embodiment: str | None) -> str | None:
    """Why a config of ``preset`` may not run without ``allow_uncertified`` (``None`` when every
    stage is covered). When no stage has any certificate, the message is the one used before
    certificates existed."""
    missing = [c for c in cov.values() if not c.unlocks]
    if not missing:
        return None
    if all(c.state in ("none", "reference") for c in cov.values()):
        return (f"preset {preset!r} has no accuracy certificate for "
                f"{', '.join(f'{c.kind} on {c.gpu}' for c in missing)} (a provisional "
                "certificate does not count; config/certificates/), so its outputs are not "
                "certified faithful to NVIDIA's. Use preset 'exact', or set "
                "`allow_uncertified: true` (or pass --allow-uncertified) to run it as an "
                "uncertified stack")
    ok = [c.kind for c in cov.values() if c.state == "certified"]
    ref = [c.kind for c in cov.values() if c.state == "reference"]
    have = "; ".join(f"{w}: {', '.join(ks)}" for w, ks in (("certified", ok), ("reference", ref))
                     if ks)
    have = f" ({have})" if have else ""
    return (f"preset {preset!r} is not certified for every stage of this config{have}: "
            f"{'; '.join(uncovered_parts(cov, embodiment))}. Add `allow_uncertified: true` "
            "to flora.yaml (or pass --allow-uncertified) to run it anyway: those stages' "
            "records then say certified: false. Or use preset 'exact'")


def refusal_message(preset: str, pools: Mapping[str, object], certified: Collection[str], *,
                    embodiment: str | None, **match_kw) -> str:
    """Config validation's refusal (``allow_uncertified`` unset, a kind of ``pools`` not in
    ``certified``, the kinds ``find_certificates`` found): :func:`uncertified_message` of the
    :func:`coverage` of ``pools``, with ``certified`` taken as given."""
    cov = coverage(preset, pools, embodiment=embodiment, **match_kw)
    for k in certified:
        if k in cov and not cov[k].unlocks:
            cov[k] = replace(cov[k], state="certified", embodiments=(str(embodiment),))
    msg = uncertified_message(preset, cov, embodiment)
    return msg or f"preset {preset!r}: not every stage is certified"
