"""vLLM 0.24.0 offline ``LLM`` for TempSeg (merged) and the Captioner (BF16), PLAN.md D2/D14.

Every argument that can change outputs is explicit, because vLLM's defaults depend on the GPU
and on the entry point (``research_notes/.../vllm_0240_verification.md``):

- ``dtype="bfloat16"``: NVIDIA's reference dtype (``auto`` would give bf16 too; say it).
- ``gdn_prefill_backend="triton"``: ``auto`` is FlashInfer on H100 and Triton on L40S, and a
  certificate must not depend on the GPU.
- ``enable_prefix_caching=False``: off by default for hybrid models; on is the experimental
  ``align`` mode where cache hit or miss changes numerics. TempSeg/Captioner share no prefixes.
- ``generation_config="vllm"``: the Captioner's ``generation_config.json`` adds EOS 248044,
  which NVIDIA's HF ``generate`` (explicit ``eos_token_id``) never stops on.
- ``max_num_seqs`` / ``max_num_batched_tokens`` from the per-GPU table ``GPU_LIMITS``: the
  offline ``LLM`` would take 1024/16384 on an H100 and AsyncLLM 128/2048; #49064 (Mamba
  blocks) bounds ``max_num_seqs``.
- ``max_model_len=32768``: NVIDIA's context limit (``nv.budget``).
- ``limit_mm_per_prompt={"image": 0, "video": 3}``: up to 3 videos (3-view captions), no images.
- ``mm_processor_kwargs=NV_VIDEO_KWARGS[kind]``: NVIDIA's ``models.processor`` builds
  ``Qwen3VLVideoProcessor(fps=4|8, max_frames=128|40)``.
- ``mm_processor_cache_gb=0``: Gate C certifies the uncached path only
  (``nv.capture.assert_engine_kwargs``).
- ``seed=0``: recorded; greedy decoding does not use it.

Sampling (``nvidia_sampling``) is NVIDIA's ``models.generate``: greedy, ``max_new_tokens`` =
NVIDIA's budget for the kind (1024 TempSeg, 512 Captioner, parsed by ``nv.budget``), stop on the
tokenizer's EOS (``<|im_end|>``) only. NVIDIA's acceptance rules (EOS reached, non-empty
``decode(ids, skip_special_tokens=True).strip()``, the 32k guard) are applied by the stages, not
here.

``OfflineEngine.generate_batch`` runs one ``llm.generate`` per batch. If the batch raises, it is
**bisected** (review 6 M5): each half is retried as one batch, recursively, so one bad request
(e.g. a prompt over ``max_model_len``) costs O(log n) generates instead of n serial ones and its
batch-mates stay batched; a request that still fails alone gets ``Completion.error`` (and
``too_long`` when vLLM rejected its length). A **dead engine** (vLLM's ``EngineDeadError``, or a
CUDA fault: ``workers_impl.reward.cuda_fatal``) is never retried: ``generate_batch`` raises
:class:`EngineDead` at once and on every later call, so the worker fails fast and exits.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from flora_deploy.engines.base import Completion, Sampling, VllmRequest
from flora_deploy.nv.capture import REQUIRED_VLLM_ENGINE_KWARGS, assert_engine_kwargs

VLLM_VERSION = "0.24.0"
KINDS = ("tempseg", "captioner")
MAX_MODEL_LEN = 32768
LIMIT_MM_PER_PROMPT = {"image": 0, "video": 3}
# NVIDIA ``models.processor`` (batch_utils/models.py:48-52 at nvidia/Flora-TempSeg@78a0e4ac):
# ``Qwen3VLVideoProcessor.from_pretrained(path, fps=4 if kind == 'tempseg' else 8,
# max_frames=128 if kind == 'tempseg' else 40)``. Restated values (license review flag, as in
# ``modal_app/gatec_app.NV_VIDEO_KWARGS``); ``tests/nvlink/test_balanced_nvlink.py`` checks them
# against the processor NVIDIA's own function builds.
NV_VIDEO_KWARGS: Mapping[str, Mapping[str, int]] = {
    "tempseg": {"fps": 4, "max_frames": 128},
    "captioner": {"fps": 8, "max_frames": 40},
}


class EngineDead(RuntimeError):
    """The vLLM engine is unusable (``EngineDeadError`` / CUDA fault): never retried."""


def engine_dead(message: str) -> bool:
    from flora_deploy.workers_impl.reward import cuda_fatal

    return "EngineDead" in message or "EngineCore encountered" in message or \
        cuda_fatal(message)


class EngineVerifyError(RuntimeError):
    """The built engine is not the one its spec describes (review 6 M6)."""


class EngineArgsError(ValueError):
    """Engine keyword arguments that would leave an output-relevant default implicit."""


@dataclass(frozen=True)
class GpuLimits:
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float


# Per (kind, GPU) scheduler limits. TempSeg is 9B (19.3 GB merged), the Captioner 27B (54.7 GB
# BF16). ``max_num_batched_tokens`` 16384 equals the offline LLM's H100 default. It holds every
# human prompt (at most ~7k tokens) and the 1-view golden worst case (14.7k) in one prefill
# chunk; 3-view (yam) Captioner prompts reach ~16.8k tokens and are **always** prefilled in two
# chunks, and that chunked path is what a yam certificate measures. It stays 16384 for every
# kind, because changing it moves chunk boundaries, and chunk boundaries change the GDN prefill
# numerics. ``max_num_seqs`` must fit a full batch of **worst-case** requests
# (``WORST_CASE_TOKENS``) in the KV / Mamba cache, so that vLLM never preempts: a preempted GDN
# sequence recomputes its generated tokens through the chunked-prefill kernel, a path no
# certificate measured (review 13 B1). :func:`worst_case_fit` is the arithmetic;
# :func:`check_mamba_blocks` checks the same bound on the built engine against the blocks vLLM
# actually allocated, and fails the build otherwise.
# TempSeg/L40S: 64 needs about 19.5 GiB at the worst case and has about 17: 48 (fit 55).
# Captioner/H100: 11 (caption pre-registration, 2026-10-23). At the 3-view worst case (17,920
# tokens) one sequence holds 26 cache blocks of 784 tokens; 11 need 286 of the ~313 estimated
# (12 would need 312, a 1-block margin; 16, the earlier value, 416). One limit for every
# embodiment: the certificate key has no embodiment.
GPU_LIMITS: Mapping[tuple[str, str], GpuLimits] = {
    ("tempseg", "L40S"): GpuLimits(48, 16384, 0.90),
    ("tempseg", "A100-80GB"): GpuLimits(48, 16384, 0.90),
    ("tempseg", "H100"): GpuLimits(48, 16384, 0.90),
    ("tempseg", "H200"): GpuLimits(64, 16384, 0.90),
    # raise it only after a measured H100 run (``num_gpu_blocks`` from the shard audit's
    # ``cache_blocks``, peak KV use, zero preemptions) shows it holds
    ("captioner", "H100"): GpuLimits(11, 16384, 0.90),
    ("captioner", "H200"): GpuLimits(32, 16384, 0.90),
}
# The longest request (prompt + NVIDIA's output budget) a kind sends, in tokens. TempSeg: one
# window per request; the 64-frame window measures 1.84-1.86k prompt tokens on every lab call
# (reports/wave-h/balanced-tempseg.json), the 128-frame ``yam32`` window at most twice that:
# 4096 + 1024 budget. Captioner: the 3-view (yam) bound, BEFORE/TARGET/AFTER blocks of at most
# 12/40/12 fixed-size mosaic frames: 16,811 prompt tokens (local tokenizer count), plus up to
# ~600 for an episode instruction, plus the 512-token budget. Human prompts average ~5-6k.
WORST_CASE_TOKENS: Mapping[str, int] = {"tempseg": 4096 + 1024, "captioner": 17920}


@dataclass(frozen=True)
class BlockFit:
    """The cache layout of one (kind, GPU), estimated (vLLM 0.24 hybrid cache, prefix caching
    off): one block id holds ``block_tokens`` tokens of every attention layer and one GDN state
    per Mamba group, so a sequence of L tokens holds ``ceil(L / block_tokens) + mamba_groups``
    blocks (:func:`blocks_per_sequence`) and the pool ``floor(pool / block)`` blocks."""

    pool_gib: float  # KV / Mamba cache left after the weights at gpu_memory_utilization 0.90
    block_mib: float  # one block id across all groups
    block_tokens: int
    mamba_groups: int

    @property
    def pool_blocks(self) -> int:
        return int(self.pool_gib * 1024 // self.block_mib)

    def blocks_per_seq(self, tokens: int) -> int:
        return -(-int(tokens) // self.block_tokens) + self.mamba_groups


# Captioner/H100 (Qwen3.5-27B: 16 attention + 48 GDN layers, 1 + 3 groups of 16): 784-token
# blocks of 49 MiB in ~15 GiB, 313 blocks. Estimates: the built engine's own count is checked by
# :func:`check_mamba_blocks` and recorded in the shard audit row (``cache_blocks``).
BLOCK_FIT: Mapping[tuple[str, str], BlockFit] = {
    ("captioner", "H100"): BlockFit(15.0, 49.0, 784, 3),
}
# Linear estimate where no block layout was derived: (KV GiB left after the weights at
# gpu_memory_utilization 0.90, GiB per 1k tokens of one sequence incl. its Mamba state), the
# Batch fill (WP1c) estimate, TempSeg 0.11 GiB at 1.85k tokens.
KV_FIT: Mapping[tuple[str, str], tuple[float, float]] = {
    ("tempseg", "L40S"): (17.0, 0.11 / 1.85),
}


def worst_case_fit(kind: str, gpu: str) -> int:
    """How many worst-case sequences (``WORST_CASE_TOKENS``) the cache holds: block-granular
    (``BLOCK_FIT``: ``floor(pool_blocks / blocks_per_seq)``) where the layout is known, else
    the linear ``KV_FIT`` estimate."""
    key = (kind, normalize_gpu(gpu))
    if key in BLOCK_FIT:
        fit = BLOCK_FIT[key]
        return fit.pool_blocks // fit.blocks_per_seq(WORST_CASE_TOKENS[kind])
    kv_gib, per_1k = KV_FIT[key]
    return int(kv_gib // (per_1k * WORST_CASE_TOKENS[kind] / 1000.0))


_GPU_PATTERNS = (("H200", r"H200"), ("H100", r"H100"), ("L40S", r"L40S"),
                 ("A100-80GB", r"A100.*80"))


def normalize_gpu(name: str) -> str:
    """``"H100!"`` / ``"NVIDIA H100 80GB HBM3"`` -> ``"H100"``; unknown names raise."""
    for short, pat in _GPU_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return short
    raise EngineArgsError(f"no engine limits for GPU {name!r}")


def gpu_limits(kind: str, gpu: str) -> GpuLimits:
    key = (kind, normalize_gpu(gpu))
    if key not in GPU_LIMITS:
        raise EngineArgsError(f"no engine limits for {kind} on {key[1]} (GPU_LIMITS)")
    return GPU_LIMITS[key]


# What every TempSeg / Captioner engine must set, with the required value (``None``: must be
# present, any value). ``validate_engine_kwargs`` enforces it.
REQUIRED: Mapping[str, Any] = {
    "dtype": "bfloat16",
    "gdn_prefill_backend": "triton",
    "enable_prefix_caching": False,
    "generation_config": "vllm",
    "max_model_len": MAX_MODEL_LEN,
    "limit_mm_per_prompt": LIMIT_MM_PER_PROMPT,
    "max_num_seqs": None,
    "max_num_batched_tokens": None,
    "seed": None,
    **REQUIRED_VLLM_ENGINE_KWARGS,
}
# Arguments that would change numerics or inputs in ways no certificate covers.
FORBIDDEN = ("quantization", "enable_lora", "kv_cache_dtype", "mamba_ssm_cache_dtype",
             "mamba_cache_dtype", "renderer_num_workers", "tokenizer_mode")


def validate_engine_kwargs(kind: str, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise ``EngineArgsError`` unless ``kwargs`` set every ``REQUIRED`` value explicitly, the
    video kwargs mirror NVIDIA's processor for ``kind``, and nothing in ``FORBIDDEN`` is set.
    Also applies Gate C's ``assert_engine_kwargs``. Returns ``kwargs``."""
    if kind not in KINDS:
        raise EngineArgsError(f"unknown kind {kind!r}")
    assert_engine_kwargs(kwargs)
    for key, want in REQUIRED.items():
        if key not in kwargs:
            raise EngineArgsError(f"{kind}: engine kwargs must set {key} explicitly")
        if want is not None and kwargs[key] != want:
            raise EngineArgsError(f"{kind}: {key}={kwargs[key]!r}, required {want!r}")
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        if type(kwargs[key]) is not int or kwargs[key] <= 0:
            raise EngineArgsError(f"{kind}: {key} must be a positive int")
    if dict(kwargs.get("mm_processor_kwargs") or {}) != dict(NV_VIDEO_KWARGS[kind]):
        raise EngineArgsError(f"{kind}: mm_processor_kwargs {kwargs.get('mm_processor_kwargs')!r}"
                              f" must equal NVIDIA's {dict(NV_VIDEO_KWARGS[kind])!r}")
    bad = [k for k in FORBIDDEN if kwargs.get(k) not in (None, "auto", False)]
    if bad:
        raise EngineArgsError(f"{kind}: engine kwargs set uncertified options {bad}")
    if kwargs.get("max_num_batched_tokens", 0) < 2048:
        raise EngineArgsError(f"{kind}: max_num_batched_tokens below 2048")
    return kwargs


@dataclass(frozen=True)
class EngineSpec:
    """Everything that defines one TempSeg / Captioner engine (hashed into the config, D11).

    ``dtbias_fp32``: run with the gated ``engines.patches.dt_bias_fp32`` plugin (Gate A). Off by
    default. ``logprobs_mode`` / ``max_logprobs``: only what the engine *returns* for logprob
    requests (Gate A measurements); greedy outputs do not depend on them. ``extra``: arguments
    that change I/O only (e.g. ``safetensors_load_strategy``), recorded but never required.
    """

    kind: str
    model: str
    gpu: str
    limits: GpuLimits
    dtbias_fp32: bool = False
    enforce_eager: bool = False
    seed: int = 0
    logprobs_mode: str | None = None
    max_logprobs: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def for_gpu(cls, kind: str, model: str, gpu: str, **kw) -> EngineSpec:
        return cls(kind=kind, model=model, gpu=normalize_gpu(gpu), limits=gpu_limits(kind, gpu),
                   **kw)

    @property
    def label(self) -> str:
        return f"vllm-{VLLM_VERSION}-offline" + ("+dtbias" if self.dtbias_fp32 else "")

    def engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            tokenizer=self.model,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            limit_mm_per_prompt=dict(LIMIT_MM_PER_PROMPT),
            mm_processor_kwargs=dict(NV_VIDEO_KWARGS[self.kind]),
            mm_processor_cache_gb=0,
            generation_config="vllm",
            gdn_prefill_backend="triton",
            enable_prefix_caching=False,
            max_num_seqs=self.limits.max_num_seqs,
            max_num_batched_tokens=self.limits.max_num_batched_tokens,
            gpu_memory_utilization=self.limits.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            seed=self.seed,
        )
        if self.logprobs_mode is not None:
            kwargs["logprobs_mode"] = self.logprobs_mode
        if self.max_logprobs is not None:
            kwargs["max_logprobs"] = self.max_logprobs
        for k, v in self.extra.items():
            if k in kwargs:
                raise EngineArgsError(f"extra engine arg {k!r} would override a pinned value")
            kwargs[k] = v
        return dict(validate_engine_kwargs(self.kind, kwargs))

    def as_json(self) -> dict:
        """Hashable description (no local model path: the model is pinned separately)."""
        kwargs = {k: v for k, v in self.engine_kwargs().items() if k not in ("model", "tokenizer")}
        return {"label": self.label, "kind": self.kind, "gpu": self.gpu,
                "dtbias_fp32": self.dtbias_fp32, "engine_kwargs": kwargs}

    def hashed(self) -> dict:
        """``as_json`` minus the GPU (D11: GPU goes to provenance, not the config hash). The
        per-GPU scheduler limits stay: they change batching, hence numerics."""
        out = self.as_json()
        out.pop("gpu")
        return out


# Balanced shard kind -> engine kind.
SHARD_ENGINE_KINDS: Mapping[str, str] = {"segment": "tempseg", "caption": "captioner"}


def spec_for(shard_kind: str, model: str, gpu: str, *, dtbias_fp32: bool = False,
             **kw) -> EngineSpec:
    """The engine a balanced ``segment`` / ``caption`` worker runs on ``gpu``. The config hash
    (``config.hashing``) hashes :func:`hashed_spec_for` of the same arguments, so a worker that
    builds its engine here runs exactly what keys its shards."""
    return EngineSpec.for_gpu(SHARD_ENGINE_KINDS[shard_kind], model, gpu,
                              dtbias_fp32=dtbias_fp32, **kw)


def hashed_spec_for(shard_kind: str, gpu: str, *, dtbias_fp32: bool = False) -> dict:
    """``spec_for(...).hashed()`` (the model path is not hashed: the model is pinned)."""
    return spec_for(shard_kind, "<pinned>", gpu, dtbias_fp32=dtbias_fp32).hashed()


def nvidia_sampling(budget: int, eos_token_id: int, *, logprobs: int | None = None) -> Sampling:
    """NVIDIA's ``models.generate``: greedy, ``max_new_tokens=budget``, stop on EOS only."""
    if type(eos_token_id) is not int:
        raise EngineArgsError(f"EOS must be one token id, got {eos_token_id!r}")
    return Sampling(max_tokens=int(budget), temperature=0.0, stop_token_ids=(eos_token_id,),
                    logprobs=logprobs)


def to_vllm_sampling(sp: Sampling, params_cls: Callable[..., Any] | None = None) -> Any:
    if params_cls is None:
        from vllm import SamplingParams as params_cls  # noqa: N813
    kwargs: dict[str, Any] = dict(
        n=sp.n, temperature=sp.temperature, top_p=sp.top_p, top_k=sp.top_k,
        max_tokens=sp.max_tokens, seed=sp.seed, skip_special_tokens=True,
        stop_token_ids=list(sp.stop_token_ids) if sp.stop_token_ids else None,
    )
    if sp.logprobs is not None:
        kwargs["logprobs"] = sp.logprobs
    if sp.regex is not None:
        from vllm.sampling_params import StructuredOutputsParams

        kwargs["structured_outputs"] = StructuredOutputsParams(regex=sp.regex)
    return params_cls(**kwargs)


_TOO_LONG = re.compile(r"(longer than the maximum model length|max_model_len|too long)", re.I)


def _logprobs(out) -> tuple[dict[int, float], ...] | None:
    lp = getattr(out, "logprobs", None)
    if not lp:
        return None
    return tuple({int(t): float(getattr(v, "logprob", v)) for t, v in pos.items()} for pos in lp)


def completion_from_output(key: str, ro, *, keep_prompt_ids: bool = False) -> Completion:
    """``vllm.RequestOutput`` -> ``Completion`` (n=1)."""
    if len(ro.outputs) != 1:
        raise RuntimeError(f"{key}: expected one completion, got {len(ro.outputs)}")
    out = ro.outputs[0]
    prompt_ids = list(ro.prompt_token_ids or [])
    return Completion(
        key=key, token_ids=tuple(int(t) for t in out.token_ids),
        finish_reason=out.finish_reason, stop_reason=out.stop_reason,
        prompt_tokens=len(prompt_ids),
        prompt_token_ids=tuple(prompt_ids) if keep_prompt_ids else None,
        logprobs=_logprobs(out),
    )


class OfflineEngine:
    """``BatchEngine`` over a ``vllm.LLM`` (built from ``spec`` unless ``llm`` is given)."""

    def __init__(self, spec: EngineSpec, llm: Any = None, *,
                 params_cls: Callable[..., Any] | None = None, keep_prompt_ids: bool = False,
                 verify: bool | None = None):
        """``verify`` (default: when the engine is built here): after the build, check the
        GDN parameter dtypes against the spec (:func:`verify_gdn_dtypes`); vLLM only logs a
        plugin that failed to load, so a ``+dtbias`` spec could otherwise run bf16."""
        self.spec = spec
        self.kwargs = spec.engine_kwargs()  # validated before anything is loaded
        self.name = spec.label
        self.keep_prompt_ids = keep_prompt_ids
        self._params_cls = params_cls
        self.stats: dict[str, float] = {"batches": 0, "requests": 0, "isolated_retries": 0,
                                        "bisections": 0, "errors": 0, "generate_s": 0.0}
        self.dead: str | None = None
        if llm is None:
            from flora_deploy.engines.patches import dt_bias_fp32

            dt_bias_fp32.check_activation(spec.dtbias_fp32)
            from vllm import LLM

            from flora_deploy.engines.worker_ext import WORKER_EXTENSION

            if self.kwargs.get("worker_extension_cls", WORKER_EXTENSION) != WORKER_EXTENSION:
                raise EngineArgsError("worker_extension_cls is flora's (engines.worker_ext)")
            t0 = time.perf_counter()
            # the worker extension only adds read-only flora_* rpc methods (not hashed: it
            # changes no numerics); it is how gdn_param_dtypes runs in vLLM's engine process
            # disable_log_stats=False: vLLM's ``vllm:num_preemptions`` counter (``preemptions``);
            # stats change no numerics, so like the extension it is not hashed
            llm = LLM(**{**self.kwargs, "worker_extension_cls": WORKER_EXTENSION,
                         "disable_log_stats": False})
            self.stats["build_s"] = round(time.perf_counter() - t0, 3)
            verify = True if verify is None else verify
        self.llm = llm
        self.gdn_dtypes: dict[str, Any] | None = None
        self.cache_blocks: dict[str, Any] | None = None
        if verify:
            self.cache_blocks = check_mamba_blocks(spec, llm)
            self.gdn_dtypes = verify_gdn_dtypes(spec, gdn_param_dtypes(llm))

    def _params(self, sp: Sampling):
        return to_vllm_sampling(sp, self._params_cls)

    def _generate(self, requests: Sequence[VllmRequest],
                  samplings: Sequence[Sampling]) -> list[Completion]:
        outs = self.llm.generate([r.prompt_dict() for r in requests],
                                 [self._params(s) for s in samplings], use_tqdm=False)
        if len(outs) != len(requests):
            raise RuntimeError(f"generate returned {len(outs)} outputs for {len(requests)}")
        return [completion_from_output(r.key, o, keep_prompt_ids=self.keep_prompt_ids)
                for r, o in zip(requests, outs, strict=True)]

    def _check_dead(self, exc: BaseException) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        if engine_dead(msg):
            self.dead = msg[:2000]
            raise EngineDead(self.dead) from exc

    def _bisect(self, requests: Sequence[VllmRequest],
                samplings: Sequence[Sampling]) -> list[Completion]:
        try:
            return self._generate(requests, samplings)
        except Exception as exc:  # noqa: BLE001 - isolate by bisection
            self._check_dead(exc)
            if len(requests) == 1:
                self.stats["errors"] += 1
                msg = f"{type(exc).__name__}: {exc}"
                return [Completion(key=requests[0].key, error=msg[:2000],
                                   too_long=bool(_TOO_LONG.search(msg)))]
        self.stats["bisections"] += 1
        mid = len(requests) // 2
        return self._bisect(requests[:mid], samplings[:mid]) + \
            self._bisect(requests[mid:], samplings[mid:])

    def generate_batch(self, requests: Sequence[VllmRequest],
                       samplings: Sequence[Sampling]) -> list[Completion]:
        if len(requests) != len(samplings):
            raise ValueError("one Sampling per request")
        if not requests:
            return []
        if self.dead is not None:  # fail fast: a dead engine never recovers in-process
            raise EngineDead(self.dead)
        t0 = time.perf_counter()
        self.stats["batches"] += 1
        self.stats["requests"] += len(requests)
        try:
            try:
                result = self._generate(requests, samplings)
            except Exception as exc:  # noqa: BLE001 - isolate: bisect the batch
                self._check_dead(exc)
                if len(requests) > 1:
                    self.stats["isolated_retries"] += 1
                    self.stats["bisections"] += 1
                    mid = len(requests) // 2
                    result = self._bisect(requests[:mid], samplings[:mid]) + \
                        self._bisect(requests[mid:], samplings[mid:])
                else:
                    self.stats["errors"] += 1
                    msg = f"{type(exc).__name__}: {exc}"
                    result = [Completion(key=requests[0].key, error=msg[:2000],
                                         too_long=bool(_TOO_LONG.search(msg)))]
        finally:
            self.stats["generate_s"] += time.perf_counter() - t0
        return result

    def preemptions(self) -> int | None:
        """vLLM's ``vllm:num_preemptions`` counter so far (summed over engines), from
        ``LLM.get_metrics()`` (needs ``disable_log_stats=False``, which the build sets). None when
        it cannot be read: unknown, never 0."""
        get = getattr(self.llm, "get_metrics", None)
        if not callable(get):
            return None
        try:
            metrics = list(get())
        except Exception:  # noqa: BLE001 - stats off or another vLLM: unknown
            return None
        vals = [getattr(m, "value", None) for m in metrics
                if getattr(m, "name", None) == PREEMPTION_METRIC]
        if not vals or not all(isinstance(v, int | float) and not isinstance(v, bool)
                               for v in vals):
            return None
        return int(sum(vals))

    def audit(self) -> dict:
        out: dict[str, Any] = {"label": self.name, "spec": self.spec.as_json(),
                               "stats": dict(self.stats)}
        out["resolved"] = resolved_facts(self.llm)
        out["cache_blocks"] = self.cache_blocks
        out["preemptions"] = self.preemptions()
        return out


PREEMPTION_METRIC = "vllm:num_preemptions"


def cache_blocks(llm) -> int | None:
    """The cache blocks the frontend's config says vLLM allocated
    (``cache_config.num_gpu_blocks``, which the engine core reports back at start); None when
    the attribute path is not there."""
    try:
        n = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    except Exception:  # noqa: BLE001 - internal attribute path
        return None
    return int(n) if isinstance(n, int) and not isinstance(n, bool) else None


def kv_cache_layout(llm) -> list[Any] | dict[str, Any]:
    """Each worker's ``KVCacheConfig`` as plain data (``worker_ext.KV_LAYOUT_RPC``), or
    ``{"error": ...}``."""
    from flora_deploy.engines.worker_ext import KV_LAYOUT_RPC

    try:
        return list(llm.collective_rpc(KV_LAYOUT_RPC))
    except Exception as exc:  # noqa: BLE001 - reported, and the check fails closed
        return {"error": repr(exc)[:500]}


def _group_kind(spec_name: str) -> str | None:
    if "Mamba" in spec_name:
        return "mamba"
    if any(k in spec_name for k in ("Attention", "SlidingWindow", "ChunkedLocal")):
        return "attention"  # a sliding window needs at most as many blocks as full attention
    return None


def blocks_per_sequence(groups: Sequence[Mapping[str, Any]], tokens: int) -> int:
    """Blocks one running sequence of ``tokens`` tokens holds (no prefix caching): one per
    Mamba group (its state), ``ceil(tokens / block_size)`` per attention group. Raises
    ``ValueError`` on a group it cannot price."""
    if not groups:
        raise ValueError("no KV cache groups")
    need = 0
    for g in groups:
        kind, bs = _group_kind(str(g.get("spec"))), g.get("block_size")
        if kind is None or type(bs) is not int or bs <= 0:
            raise ValueError(f"unpriced KV cache group {dict(g)!r}")
        need += 1 if kind == "mamba" else -(-int(tokens) // bs)
    return need


def check_mamba_blocks(spec: EngineSpec, llm, *, blocks: int | None = None) -> dict[str, Any]:
    """Fail the build (:class:`EngineVerifyError`) unless the cache blocks vLLM allocated hold
    ``max_num_seqs`` sequences of the kind's worst-case length (``WORST_CASE_TOKENS``): a
    hybrid sequence holds one block per Mamba group plus ``ceil(L / block_size)`` per attention
    group (vLLM #49064; with fewer, the scheduler preempts or stalls mid-run instead of failing
    here). **Fail closed**: a block count or a KV cache layout that cannot be read fails too.
    ``blocks``: a count to add to the ones read (the smallest decides). Returns the inputs and
    ``{"num_gpu_blocks", "blocks_per_seq", "needed", "checked": True}``."""
    seqs, tokens = spec.limits.max_num_seqs, WORST_CASE_TOKENS[spec.kind]
    where = f"{spec.kind} on {spec.gpu}"
    layout = kv_cache_layout(llm)
    try:
        if isinstance(layout, dict) or not layout:
            raise ValueError(f"no KV cache layout from the workers: {layout!r}")
        counts = [n for n in (blocks, cache_blocks(llm), *(w.get("num_blocks") for w in layout))
                  if type(n) is int]
        if not counts:
            raise ValueError("no cache block count")
        per_seq = max(blocks_per_sequence(list(w.get("groups") or ()), tokens) for w in layout)
    except (ValueError, TypeError, AttributeError) as exc:
        raise EngineVerifyError(f"{where}: cannot check max_num_seqs={seqs} against the cache "
                                f"blocks vLLM allocated ({exc}); refusing to run (#49064)"
                                ) from None
    n = min(counts)
    needed = seqs * per_seq
    groups = [dict(g) for g in layout[0].get("groups") or ()]
    attn = [g.get("block_size") for g in groups if _group_kind(str(g.get("spec"))) == "attention"]
    out = {"num_gpu_blocks": n, "max_num_seqs": seqs, "worst_case_tokens": tokens,
           "blocks_per_seq": per_seq, "needed": needed, "checked": True,
           "block_size": attn[0] if attn else None,
           "gpu_memory_utilization": spec.limits.gpu_memory_utilization,
           "groups": groups}
    if n < needed:
        raise EngineVerifyError(
            f"{where}: max_num_seqs={seqs} worst-case sequences ({tokens} tokens, {per_seq} "
            f"blocks each) need {needed} cache blocks; vLLM allocated {n} at "
            f"gpu_memory_utilization={spec.limits.gpu_memory_utilization} (#49064); lower "
            "max_num_seqs in GPU_LIMITS")
    return out


def resolved_facts(llm) -> dict[str, Any]:
    """What vLLM resolved (best effort: attribute paths are internal)."""
    facts: dict[str, Any] = {}
    try:
        import vllm

        facts["vllm"] = vllm.__version__
        config = llm.llm_engine.vllm_config
        facts["max_num_batched_tokens"] = config.scheduler_config.max_num_batched_tokens
        facts["max_num_seqs"] = config.scheduler_config.max_num_seqs
        facts["dtype"] = str(config.model_config.dtype)
        facts["enable_prefix_caching"] = config.cache_config.enable_prefix_caching
        facts["mamba_ssm_cache_dtype"] = str(getattr(config.cache_config,
                                                     "mamba_ssm_cache_dtype", None))
        facts["mm_processor_cache_gb"] = getattr(config.model_config.multimodal_config,
                                                 "mm_processor_cache_gb", None)
        facts["additional_config"] = dict(getattr(config, "additional_config", {}) or {})
        facts["generation_config"] = getattr(config.model_config, "generation_config", None)
    except Exception as exc:  # pragma: no cover - diagnostics only
        facts["error"] = repr(exc)
    return facts


def gdn_param_dtypes(llm) -> dict[str, Any]:
    """dtype counts of every GDN ``dt_bias`` / ``A_log`` parameter in the engine's workers
    (L1 / Gate A evidence).

    Calls the worker extension's method **by name** (``collective_rpc("flora_gdn_dtypes")``,
    :mod:`flora_deploy.engines.worker_ext`): vLLM 0.24 refuses to send a function to its engine
    process without ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (review 8a R5), which flora never
    sets. The engine must be built with ``worker_extension_cls`` (:class:`OfflineEngine` does)."""
    from flora_deploy.engines.worker_ext import GDN_DTYPES_RPC

    try:
        return {"workers": llm.collective_rpc(GDN_DTYPES_RPC)}
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": repr(exc)}


def verify_gdn_dtypes(spec: EngineSpec, report: Mapping[str, Any]) -> dict[str, Any]:
    """Raise :class:`EngineVerifyError` unless every worker of the built engine has GDN
    parameters and every ``dt_bias`` is FP32 with ``dtbias_fp32`` (bf16, vLLM's own, without)
    and every ``A_log`` FP32 (vLLM creates it FP32). ``report``: :func:`gdn_param_dtypes`.
    Returns the report."""
    if "error" in report or not report.get("workers"):
        raise EngineVerifyError(f"cannot read the engine's GDN parameter dtypes: {report!r}")
    want = "float32" if spec.dtbias_fp32 else "bfloat16"
    for i, counts in enumerate(report["workers"]):
        counts = dict(counts or {})
        dt = {k.split(":", 1)[1]: n for k, n in counts.items() if k.startswith("dt_bias:")}
        alog = {k.split(":", 1)[1]: n for k, n in counts.items() if k.startswith("A_log:")}
        if not dt or not alog:
            raise EngineVerifyError(f"worker {i}: no GDN dt_bias / A_log parameters: {counts}")
        if set(dt) != {want}:
            raise EngineVerifyError(
                f"worker {i}: dt_bias dtypes {dt}, spec {spec.label!r} needs all {want}"
                + (" (the dt_bias plugin did not load: vLLM only logs plugin errors)"
                   if spec.dtbias_fp32 else " (a dt_bias patch is active but not in the spec)"))
        if set(alog) != {"float32"}:
            raise EngineVerifyError(f"worker {i}: A_log dtypes {alog}, expected all float32")
    return dict(report)


def extra_state_prefixes(weight_map: Mapping[str, Any]) -> list[str]:
    """Module prefixes (before ``.layers.`` / ``.blocks.``) of ``*._extra_state`` keys."""
    out = set()
    for k in weight_map:
        if "_extra_state" in k:
            out.add(re.split(r"\.(?:layers|blocks)\.", k, maxsplit=1)[0])
    return sorted(out)


def check_extra_state(weight_map: Mapping[str, Any]) -> list[str]:
    """Preflight (review 6 Q5): vLLM's language-model loader skips unknown ``_extra_state``
    keys, but ``Qwen3_VisionTransformer.load_weights`` does a bare ``params_dict[name]``, so
    any such key under ``model.visual`` fails the load. Raises :class:`EngineVerifyError`
    then; returns the prefixes otherwise."""
    prefixes = extra_state_prefixes(weight_map)
    bad = sorted(k for k in weight_map if "_extra_state" in k
                 and (k.startswith("model.visual") or k.startswith("visual.")))
    if bad:
        raise EngineVerifyError(f"{len(bad)} _extra_state key(s) under model.visual would "
                                f"KeyError in vLLM's vision loader, e.g. {bad[:3]}")
    return prefixes
