"""One-time FP32 merge of the TempSeg LoRA into Qwen3.5-9B (P1b-0, PLAN.md D3,
``docs/tempseg-merge.md``).

NVIDIA serves TempSeg unmerged (``batch_utils/vendor/tempseg_fp32_loader.py``): the base is built
with ``AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)`` after
``production_gdn_fix._keep_native_gdn_fp32`` (so ``A_log`` / ``dt_bias`` stay FP32), the FP32
PEFT LoRA (r=32, alpha=64) is attached with FP32 holders, and ``generate`` runs under BF16
autocast. D3 serves a plain model instead, so this module bakes the adapter in:

    W_merged = bf16( fp32( fp64(W_base) + (alpha / r) * sum_k fp64(B[:, k]) * fp64(A[k, :]) ) )

for every adapter target: the rank sum in a fixed order in FP64 (exact products, so the result
is byte-identical on every CPU and thread count; :func:`merge_weight`), then rounded to BF16 at
the end (PEFT's own ``merge_and_unload`` on a BF16 base would round the delta to BF16 first,
then round again).
Every other tensor gets exactly the dtype NVIDIA's loader gives it, by the same rule
``convert_captioner`` uses: transformers' ``dtype_plan`` (with NVIDIA's own
``_keep_native_gdn_fp32`` applied) and the model's meta dtypes, so ``A_log`` / ``dt_bias`` are
written as FP32 (exact upcasts of the base values) and nothing is hard-coded. Checkpoint keys
the model ignores on load (``_keys_to_ignore_on_load_unexpected``, i.e. ``mtp.*``) are copied
unchanged.

The result is not bit-identical to NVIDIA's runtime (that is impossible: merging moves the
rounding). ``modal_app/convert_app.py::tempseg_equivalence`` measures how far it moves.

Layout written to ``layout.derived_path("tempseg-merged", PINS_BY_KEY["tempseg"])``:

- the base shards (same names, same key -> shard map) with merged targets;
- ``model.safetensors.index.json`` (same ``weight_map``, new ``total_size``);
- the base ``config.json`` (the model NVIDIA builds is the base's);
- **Flora's** ``processor_config.json``, ``tokenizer.json``, ``tokenizer_config.json``,
  ``chat_template.jinja`` (what NVIDIA's ``models.processor('tempseg', path)`` reads) plus
  Flora's ``README.md`` and ``MODEL_PROVENANCE.json`` (notices). Never ``adapter_config.json``
  (transformers would treat the directory as an adapter) and never the base's processor files;
- ``derived_receipt.json`` (written last): sha256/size of every file, source pins, the sha256 of
  every source file as hashed while it was read, the LoRA spec, the dtype policy, the cast and
  merge counts, merge code version + module sha256 + git commit, tool versions.

Memory is bounded: the adapter (0.35 GB) plus ``workers`` base shards (<= 5.4 GB each) and one
FP32 target tensor per worker at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from flora_deploy.config.pins import PINS_BY_KEY, RepoPin
from flora_deploy.prep.convert_captioner import (
    ContentDigestError,
    DtypePolicy,
    _read_whole,
    _sha256_file,
    checked_content_digest,
    content_digest,
    encode_header,
    read_header,
    st_dtype,
    tool_versions,
    torch_to_st,
)
from flora_deploy.prep.verify import parse_sha256_manifest, pin_record

DERIVED_NAME = "tempseg-merged"
DERIVED_RECEIPT = "derived_receipt.json"
RECEIPT_SCHEMA = 1
# Bump whenever the merge rule or the output format changes. v2: fixed-order FP64 rank sum
# (byte-deterministic across machines, so the content digest can be pinned); v1 used BLAS.
MERGER_VERSION = 2
SOURCE_KEY = "tempseg"  # nvidia/Flora-TempSeg weights pin (adapter + processor files)
BASE_KEY = "tempseg_base"  # Qwen/Qwen3.5-9B
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "SOURCE_MANIFEST.sha256"
PREP_RECEIPT = "prep_receipt.json"
ADAPTER_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
BASE_FILES = ("config.json",)
FLORA_FILES = (
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
    "MODEL_PROVENANCE.json",
)
# What the TempSeg adapter must contain (the task brief; NVIDIA's loader asserts the same counts).
EXPECTED_TARGETS = 248
EXPECTED_TENSORS = 496
PEFT_PREFIX = "base_model.model."
LORA_KEY_RE = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")
LOAD_DTYPE = "bfloat16"


class MergeError(RuntimeError):
    pass


# --- adapter spec and key mapping -------------------------------------------------------------


@dataclass(frozen=True)
class LoraSpec:
    r: int
    alpha: float
    targets: frozenset[str]

    @property
    def scale(self) -> float:
        # PEFT LoraLayer.update_layer: lora_alpha / r (lora_alpha / sqrt(r) only with rsLoRA,
        # which ``from_config`` rejects).
        return self.alpha / self.r

    def record(self) -> dict:
        return {"r": self.r, "lora_alpha": self.alpha, "scale": self.scale,
                "target_modules": len(self.targets),
                "targets_sha256": hashlib.sha256("\n".join(sorted(self.targets)).encode())
                .hexdigest(),
                # merger v2 (``merge_weight``); enters the content digest (review 8b L2)
                "rule": "bf16(fp32(fp64(W) + scale * sum_{k=0..r-1} fp64(B[:,k]) * "
                        "fp64(A[k,:]))), k in fixed order"}

    def matches(self, module: str) -> bool:
        """PEFT's rule for a list of names: exact name or a ``.``-separated suffix (PEFT may
        shorten a long explicit list to suffixes when it saves the config)."""
        return any(module == t or module.endswith("." + t) for t in self.targets)

    @classmethod
    def from_config(cls, cfg: Mapping, base_revision: str | None = None) -> LoraSpec:
        """Accept only the plain LoRA the merge formula is exact for; anything else fails."""
        problems = []
        if cfg.get("peft_type") != "LORA":
            problems.append(f"peft_type={cfg.get('peft_type')!r}")
        for key, ok in (("use_dora", False), ("use_rslora", False), ("fan_in_fan_out", False)):
            if bool(cfg.get(key, False)) != ok:
                problems.append(f"{key}={cfg.get(key)!r}")
        if cfg.get("bias", "none") != "none":
            problems.append(f"bias={cfg.get('bias')!r}")
        for key in ("rank_pattern", "alpha_pattern"):
            if cfg.get(key):
                problems.append(f"{key} is set")
        for key in ("modules_to_save", "layers_to_transform", "trainable_token_indices",
                    "target_parameters"):
            if cfg.get(key):
                problems.append(f"{key} is set")
        if cfg.get("lora_bias"):
            problems.append("lora_bias is set")
        r, alpha = cfg.get("r"), cfg.get("lora_alpha")
        if not isinstance(r, int) or r <= 0 or not isinstance(alpha, int | float):
            problems.append(f"r={r!r} lora_alpha={alpha!r}")
        targets = cfg.get("target_modules")
        if not isinstance(targets, list) or not targets:
            problems.append("target_modules must be an explicit list of module names")
        if base_revision is not None and cfg.get("revision") not in (None, base_revision):
            problems.append(f"adapter revision {cfg.get('revision')} != base pin {base_revision}")
        if problems:
            raise MergeError(f"unsupported adapter config: {problems}")
        return cls(r=r, alpha=float(alpha), targets=frozenset(targets))


@dataclass(frozen=True)
class Target:
    module: str
    weight: str  # base checkpoint key, ``<module>.weight``
    lora_a: str  # adapter key
    lora_b: str


def map_adapter_keys(adapter_keys: Iterable[str], spec: LoraSpec) -> dict[str, Target]:
    """Adapter keys -> base ``.weight`` keys. Every adapter key must be ``lora_A`` / ``lora_B`` of
    a declared target; every target must have both; nothing else may be in the file."""
    found: dict[str, dict[str, str]] = {}
    bad = []
    for key in adapter_keys:
        m = LORA_KEY_RE.match(key)
        if m is None:
            bad.append(key)
            continue
        found.setdefault(m["module"], {})
        if m["ab"] in found[m["module"]]:
            bad.append(key)
        found[m["module"]][m["ab"]] = key
    if bad:
        raise MergeError(f"adapter keys that are not plain LoRA A/B weights: {sorted(bad)[:5]}")
    incomplete = sorted(mod for mod, ab in found.items() if set(ab) != {"A", "B"})
    if incomplete:
        raise MergeError(f"targets without both lora_A and lora_B: {incomplete[:5]}")
    extra = sorted(m for m in found if not spec.matches(m))
    unused = sorted(t for t in spec.targets
                    if not any(m == t or m.endswith("." + t) for m in found))
    if extra or unused:
        raise MergeError(f"adapter modules != target_modules: not targeted {extra[:5]}, "
                         f"targets without weights {unused[:5]}")
    return {f"{mod}.weight": Target(mod, f"{mod}.weight", ab["A"], ab["B"])
            for mod, ab in sorted(found.items())}


def check_shapes(targets: Mapping[str, Target], base_shapes: Mapping[str, list[int]],
                 adapter_shapes: Mapping[str, list[int]], r: int) -> None:
    """A is [r, in], B is [out, r], W is [out, in], for every target."""
    problems = []
    for key, t in targets.items():
        w = base_shapes.get(key)
        if w is None:
            problems.append(f"{key}: not in the base checkpoint")
            continue
        a, b = list(adapter_shapes[t.lora_a]), list(adapter_shapes[t.lora_b])
        if len(w) != 2 or a != [r, w[1]] or b != [w[0], r]:
            problems.append(f"{key}: W{w} A{a} B{b}")
    if problems:
        raise MergeError(f"shape mismatch: {problems[:5]}")


def check_target_coverage(targets: Mapping[str, Target], base_keys: Iterable[str],
                          spec: LoraSpec) -> None:
    """Every base linear weight PEFT would wrap (its module matches ``target_modules``) has an
    adapter, and nothing else does."""
    wrapped = {k for k in base_keys if k.endswith(".weight") and spec.matches(k[:-7])}
    if wrapped != set(targets):
        raise MergeError(f"targeted base weights != adapter targets: no adapter "
                         f"{sorted(wrapped - set(targets))[:5]}, adapter without base weight "
                         f"{sorted(set(targets) - wrapped)[:5]}")


MERGE_TILE_ELEMENTS = 1 << 20  # fp64 accumulator tile (8 MiB): bounded memory, cache-sized


def merge_weight(w, a, b, scale: float, *, tile_elements: int = MERGE_TILE_ELEMENTS):
    """Merge one target, byte-deterministically on any CPU and thread count (merger v2)::

        fp32( fp64(W) + scale * sum_{k=0..r-1} fp64(B[:, k]) * fp64(A[k, :]) )

    Why it is deterministic (``docs/tempseg-merge.md``): ``A``/``B`` are FP32, so each product
    ``B[i,k] * A[k,j]`` (24 + 24 significand bits) is **exact** in FP64, and fused and unfused
    multiply-adds give the same result. The rank sum runs in a fixed order (k = 0, 1, ...) as
    element-wise FP64 adds, each correctly rounded (IEEE 754), so no BLAS kernel, SIMD width or
    thread split can reorder it. The scale, the add of ``W`` and the FP64 -> FP32 cast are single
    correctly rounded operations. (A BLAS ``matmul`` chooses its summation order by CPU and
    thread count, so v1's FP32 ``B @ A`` was not reproducible across machines.) The caller's
    FP32 -> BF16 cast is torch's integer round-to-nearest-even, identical in scalar and SIMD
    code. Returns FP32 (not yet rounded to the target dtype)."""
    import torch

    rows_total, cols = w.shape
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    out = torch.empty((rows_total, cols), dtype=torch.float32)
    step = max(1, tile_elements // max(1, cols))
    for i in range(0, rows_total, step):
        j = min(rows_total, i + step)
        acc = torch.zeros((j - i, cols), dtype=torch.float64)
        bt = b64[i:j]
        for k in range(a64.shape[0]):
            acc.addcmul_(bt[:, k:k + 1], a64[k:k + 1, :])
        acc.mul_(scale)
        acc.add_(w[i:j].to(torch.float64))
        out[i:j] = acc
    return out


# --- dtype policy ---------------------------------------------------------------------------


def nvidia_merge_policy(base: Path, keep_fp32_patch: Callable[[type], None]
                        ) -> tuple[DtypePolicy, tuple[str, ...]]:
    """Per-key target dtypes of NVIDIA's ``from_pretrained(base, dtype=bf16)`` call (see
    ``convert_captioner.nvidia_dtype_policy``) and the model's ignore-on-load key patterns.
    ``keep_fp32_patch`` must be NVIDIA's ``production_gdn_fix._keep_native_gdn_fp32``."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    keep_fp32_patch(Qwen3_5PreTrainedModel)
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    plan = {k: str(v).removeprefix("torch.")
            for k, v in model._get_dtype_plan(torch.bfloat16).items()}
    dtypes = {k: str(v.dtype).removeprefix("torch.") for k, v in model.state_dict().items()}
    ignored = tuple(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ())
    return DtypePolicy.build(LOAD_DTYPE, plan, dtypes), ignored


def check_key_coverage(weight_map: Mapping[str, str], policy: DtypePolicy,
                       ignored: Iterable[str], targets: Mapping[str, Target]) -> list[str]:
    """Checkpoint keys vs model keys. Returns the ignored (copied-unchanged) keys."""
    patterns = [re.compile(p) for p in ignored]
    unknown = [k for k in weight_map if policy.target(k) is None]
    unignored = [k for k in unknown if not any(p.search(k) for p in patterns)]
    if unignored:
        raise MergeError(f"checkpoint keys the model neither has nor ignores: {unignored[:5]}")
    absent = [k for k in policy.model_dtypes if k not in weight_map]
    if absent:
        raise MergeError(f"model keys missing from the base checkpoint: {absent[:5]}")
    for key in targets:
        if policy.target(key) != LOAD_DTYPE:
            raise MergeError(f"{key}: target dtype {policy.target(key)} (only BF16 linear "
                             "weights may be merged)")
    return sorted(unknown)


# --- merge ----------------------------------------------------------------------------------


def merge_shard(src: Path, dest: Path, policy: DtypePolicy, targets: Mapping[str, Target],
                adapter: Mapping, scale: float, expected_sha256: str) -> dict:
    """Merge one base shard; raises if the source does not hash to ``expected_sha256``."""
    import torch

    t0 = time.monotonic()
    buf, src_sha = _read_whole(src)
    if src_sha != expected_sha256:
        raise MergeError(f"{src.name}: sha256 {src_sha} != expected {expected_sha256}")
    t_read = time.monotonic() - t0
    (n,) = struct.unpack_from("<Q", buf, 0)
    start = 8 + n
    header = json.loads(bytes(buf[8:start]))
    metadata = header.pop("__metadata__", None)
    names = sorted(header, key=lambda k: header[k]["data_offsets"][0])

    def source_tensor(name):
        info = header[name]
        b, e = info["data_offsets"]
        dtype = st_dtype(info["dtype"])
        count = (e - b) // torch.empty((), dtype=dtype).element_size()
        if count == 0:
            return torch.empty(info["shape"], dtype=dtype)
        return torch.frombuffer(buf, dtype=dtype, count=count, offset=start + b).reshape(
            info["shape"])

    plan_out, casts = [], Counter()
    for name in names:
        info = header[name]
        target = policy.target(name)
        out_st = info["dtype"] if target is None else torch_to_st(getattr(torch, target))
        itemsize = torch.empty((), dtype=st_dtype(out_st)).element_size()
        numel = 1
        for d in info["shape"]:
            numel *= d
        plan_out.append((name, out_st, list(info["shape"]), numel * itemsize))
        tag = ("merged" if name in targets else
               "ignored_copied" if target is None else "loaded")
        casts[f"{info['dtype']}->{out_st}:{tag}"] += 1

    head = encode_header(plan_out, metadata)
    h = hashlib.sha256(head)
    stats = {"merged": 0, "elements": 0, "elements_changed": 0, "max_abs_change": 0.0,
             "sum_abs_change": 0.0}
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with partial.open("wb") as f:
        f.write(head)
        for name, out_st, _, nbytes in plan_out:
            t = source_tensor(name)
            if name in targets:
                tg = targets[name]
                merged = merge_weight(t, adapter[tg.lora_a], adapter[tg.lora_b], scale)
                out = merged.to(st_dtype(out_st))
                base_cast = t.to(st_dtype(out_st))
                diff = (out.to(torch.float32) - base_cast.to(torch.float32)).abs_()
                stats["merged"] += 1
                stats["elements"] += out.numel()
                stats["elements_changed"] += int(torch.count_nonzero(diff))
                stats["max_abs_change"] = max(stats["max_abs_change"], float(diff.max()))
                stats["sum_abs_change"] += float(diff.sum(dtype=torch.float64))
                del merged, base_cast, diff
            else:
                out = t if torch_to_st(t.dtype) == out_st else t.to(st_dtype(out_st))
            raw = out.contiguous().reshape(-1).view(torch.uint8).numpy().data
            if len(raw) != nbytes:
                raise MergeError(f"{name}: wrote {len(raw)} bytes, planned {nbytes}")
            h.update(raw)
            f.write(raw)
            del t, out, raw
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, dest)
    return {
        "path": dest.name,
        "source_sha256": src_sha,
        "written_sha256": h.hexdigest(),
        "header_sha256": hashlib.sha256(head).hexdigest(),
        "tensors": len(plan_out),
        "casts": dict(casts),
        "merge": stats,
        "read_s": round(t_read, 2),
        "seconds": round(time.monotonic() - t0, 2),
    }


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def receipt_path(derived_dir: Path) -> Path:
    return Path(derived_dir) / DERIVED_RECEIPT


def load_receipt(derived_dir: Path) -> dict | None:
    try:
        return json.loads(receipt_path(derived_dir).read_text())
    except (OSError, ValueError):
        return None


def flora_manifest(flora: Path, pin: RepoPin) -> dict[str, str]:
    """NVIDIA's ``SOURCE_MANIFEST.sha256`` entries, after checking the manifest's own digest
    against ``pins.py``."""
    raw = (flora / MANIFEST_NAME).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    want = dict(pin.expected).get("manifest_sha256")
    if got != want:
        raise MergeError(f"{MANIFEST_NAME} sha256 {got} != pinned {want}")
    return {name: digest for digest, name in parse_sha256_manifest(raw.decode())}


def base_digests(base: Path, pin: RepoPin) -> tuple[dict[str, str], str]:
    """Per-file sha256 from the base snapshot's passing prep receipt, which is bound to the
    pinned ``plan_digest`` (the base has no NVIDIA manifest). Returns (digests, receipt sha256)."""
    raw = (base / PREP_RECEIPT).read_bytes()
    receipt = json.loads(raw)
    if (receipt.get("ok") is not True or receipt.get("plan_digest") != pin.plan_digest
            or receipt.get("pin", {}).get("revision") != pin.revision):
        raise MergeError(f"{base / PREP_RECEIPT} is not a passing receipt of {pin.key}")
    return ({d["path"]: d["sha256"] for d in receipt["digests"]},
            hashlib.sha256(raw).hexdigest())


def load_adapter(path: Path, expected_sha256: str) -> tuple[dict, dict[str, list[int]]]:
    """Adapter tensors (hashed while read) and their shapes."""
    from safetensors.torch import load as st_load

    buf, sha = _read_whole(path)
    if sha != expected_sha256:
        raise MergeError(f"{path.name}: sha256 {sha} != pinned {expected_sha256}")
    tensors = st_load(bytes(buf))
    del buf
    return tensors, {k: list(v.shape) for k, v in tensors.items()}


def merge(
    flora: Path,
    base: Path,
    dest: Path,
    keep_fp32_patch: Callable[[type], None] | None = None,
    *,
    policy: tuple[DtypePolicy, tuple[str, ...]] | None = None,
    workers: int = 2,
    threads_per_worker: int = 4,
    force: bool = False,
    provenance: dict | None = None,
    commit: Callable[[], None] | None = None,
    pin: RepoPin | None = None,
    base_pin: RepoPin | None = None,
    expected_counts: tuple[int, int] | None = (EXPECTED_TARGETS, EXPECTED_TENSORS),
) -> dict:
    """Merge ``flora``'s adapter into ``base``, writing ``dest`` (module docstring); returns the
    receipt. Idempotent unless ``force``. ``commit`` runs after the receipt is written."""
    import torch

    flora, base, dest = Path(flora), Path(base), Path(dest)
    pin = pin or PINS_BY_KEY[SOURCE_KEY]
    base_pin = base_pin or PINS_BY_KEY[BASE_KEY]
    existing = load_receipt(dest)
    if (not force and existing and existing.get("ok") is True
            and existing.get("merger", {}).get("version") == MERGER_VERSION
            and existing.get("source", {}).get("pin", {}).get("revision") == pin.revision
            and existing.get("base", {}).get("pin", {}).get("revision") == base_pin.revision):
        return {**existing, "reused": True}

    t0 = time.monotonic()
    manifest = flora_manifest(flora, pin)
    adapter_sha = dict(pin.expected).get("adapter_sha256")
    if manifest.get(ADAPTER_FILE) != adapter_sha:
        raise MergeError("NVIDIA manifest and pins.py disagree on the adapter sha256")
    missing = [n for n in (ADAPTER_FILE, ADAPTER_CONFIG, *FLORA_FILES) if n not in manifest]
    if missing:
        raise MergeError(f"not covered by NVIDIA's manifest: {missing}")
    cfg_raw = (flora / ADAPTER_CONFIG).read_bytes()
    if hashlib.sha256(cfg_raw).hexdigest() != manifest[ADAPTER_CONFIG]:
        raise MergeError(f"{ADAPTER_CONFIG}: sha256 != NVIDIA manifest")
    spec = LoraSpec.from_config(json.loads(cfg_raw), base_pin.revision)

    base_sha, prep_sha = base_digests(base, base_pin)
    index = json.loads((base / INDEX_NAME).read_text())
    if _sha256_file(base / INDEX_NAME) != base_sha.get(INDEX_NAME):
        raise MergeError(f"base {INDEX_NAME}: sha256 != prep receipt")
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    on_disk = sorted(p.name for p in base.glob("*.safetensors"))
    if on_disk != shards:
        raise MergeError(f"base shards on disk {on_disk} != index {shards}")
    unbound = [s for s in [*shards, *BASE_FILES] if s not in base_sha]
    if unbound:
        raise MergeError(f"base files not in its prep receipt: {unbound}")

    adapter, adapter_shapes = load_adapter(flora / ADAPTER_FILE, adapter_sha)
    bad_dtypes = Counter(str(v.dtype) for v in adapter.values() if v.dtype != torch.float32)
    if bad_dtypes:
        raise MergeError(f"adapter tensors are not all FP32: {dict(bad_dtypes)}")
    targets = map_adapter_keys(adapter, spec)
    if expected_counts is not None and (len(targets), len(adapter)) != expected_counts:
        raise MergeError(f"adapter has {len(targets)} targets / {len(adapter)} tensors, "
                         f"expected {expected_counts}")
    base_shapes: dict[str, list[int]] = {}
    for s in shards:
        with (base / s).open("rb") as f:
            _, hdr, _ = read_header(f)
        base_shapes.update({k: v["shape"] for k, v in hdr.items() if k != "__metadata__"})
    if set(base_shapes) != set(weight_map):
        raise MergeError("base shard headers disagree with the index weight_map")
    check_shapes(targets, base_shapes, adapter_shapes, spec.r)
    check_target_coverage(targets, base_shapes, spec)

    if policy is None:
        if keep_fp32_patch is None:
            raise MergeError("need NVIDIA's _keep_native_gdn_fp32 (or an explicit policy)")
        policy = nvidia_merge_policy(base, keep_fp32_patch)
    dtype_policy, ignored_patterns = policy
    ignored = check_key_coverage(weight_map, dtype_policy, ignored_patterns, targets)

    if dest.exists() and force and receipt_path(dest).exists():
        receipt_path(dest).unlink()
    dest.mkdir(parents=True, exist_ok=True)
    t_prep = time.monotonic() - t0

    # 1. small files, verbatim: base config, Flora's processor/tokenizer/template + notices
    copied = []
    for src_dir, name, want, origin in (
        *((base, n, base_sha[n], "base") for n in BASE_FILES),
        *((flora, n, manifest[n], "flora") for n in FLORA_FILES),
    ):
        sha = _sha256_file(src_dir / name)
        if sha != want:
            raise MergeError(f"{origin} {name}: sha256 {sha} != recorded {want}")
        shutil.copyfile(src_dir / name, dest / name)
        copied.append((name, sha, origin))

    # 2. shards (bounded: ``workers`` base shards in memory at once)
    torch.set_num_threads(max(1, threads_per_worker))
    by_shard: dict[str, dict[str, Target]] = {s: {} for s in shards}
    for key, t in targets.items():
        by_shard[weight_map[key]][key] = t
    t1 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = {s: pool.submit(merge_shard, base / s, dest / s, dtype_policy, by_shard[s],
                               adapter, spec.scale, base_sha[s]) for s in shards}
        shard_results = {s: j.result() for s, j in jobs.items()}
    t_merge = time.monotonic() - t1
    del adapter
    merged_total = sum(r["merge"]["merged"] for r in shard_results.values())
    if merged_total != len(targets):
        raise MergeError(f"merged {merged_total} targets, expected {len(targets)}")

    # 3. index: same weight_map, new total_size
    total = 0
    for s in shards:
        with (dest / s).open("rb") as f:
            _, hdr, _ = read_header(f)
        total += sum(v["data_offsets"][1] - v["data_offsets"][0]
                     for k, v in hdr.items() if k != "__metadata__")
    new_index = {**index, "metadata": {**index.get("metadata", {}), "total_size": total}}
    (dest / INDEX_NAME).write_text(json.dumps(new_index, indent=2, sort_keys=True) + "\n")

    # 4. re-read everything from the destination
    t2 = time.monotonic()
    names = sorted([*shards, INDEX_NAME, *BASE_FILES, *FLORA_FILES])
    with ThreadPoolExecutor(max_workers=8) as pool:
        stored = dict(zip(names, pool.map(lambda n: _sha256_file(dest / n), names), strict=True))
    for s in shards:
        if stored[s] != shard_results[s]["written_sha256"]:
            raise MergeError(f"{s}: stored bytes differ from the bytes written")
    for name, sha, _ in copied:
        if stored[name] != sha:
            raise MergeError(f"{name}: copy differs from source")
    t_reread = time.monotonic() - t2
    extra = sorted(p.name for p in dest.iterdir()
                   if p.name not in stored and p.name != DERIVED_RECEIPT)
    if extra:
        raise MergeError(f"unexpected files in {dest}: {extra}")

    origin = {name: o for name, _, o in copied}
    files = []
    for name in names:
        entry = {"path": name, "size": (dest / name).stat().st_size, "sha256": stored[name]}
        if name in shard_results:
            r = shard_results[name]
            entry.update(kind="merged_shard", header_sha256=r["header_sha256"],
                         source_sha256=r["source_sha256"], tensors=r["tensors"],
                         merged_targets=r["merge"]["merged"])
        elif name == INDEX_NAME:
            entry.update(kind="rewritten_index", source_sha256=base_sha[INDEX_NAME])
        else:
            entry.update(kind=f"copied_from_{origin[name]}",
                         source_sha256=base_sha[name] if origin[name] == "base"
                         else manifest[name])
        files.append(entry)
    casts: Counter = Counter()
    stats = Counter()
    max_change = 0.0
    for r in shard_results.values():
        casts.update(r["casts"])
        m = r["merge"]
        stats.update({k: m[k] for k in ("merged", "elements", "elements_changed")})
        stats["sum_abs_change"] += m["sum_abs_change"]
        max_change = max(max_change, m["max_abs_change"])
    merge_stats = {
        "targets": int(stats["merged"]), "elements": int(stats["elements"]),
        "elements_changed_vs_base": int(stats["elements_changed"]),
        "fraction_changed": (stats["elements_changed"] / stats["elements"]
                             if stats["elements"] else 0.0),
        "max_abs_change": max_change,
        "mean_abs_change": (stats["sum_abs_change"] / stats["elements"]
                            if stats["elements"] else 0.0),
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "ok": True,
        "kind": DERIVED_NAME,
        "merger": {"version": MERGER_VERSION, "module_sha256": module_sha256(),
                   "git_commit": (provenance or {}).get("git_commit"),
                   "git_dirty": (provenance or {}).get("git_dirty"),
                   "workers": workers, "torch_threads": threads_per_worker},
        "source": {"pin": pin_record(pin), "manifest_sha256": dict(pin.expected)["manifest_sha256"],
                   "adapter_sha256": adapter_sha, "adapter_config_sha256": manifest[ADAPTER_CONFIG],
                   "adapter_tensors": len(adapter_shapes)},
        "base": {"pin": pin_record(base_pin), "plan_digest": base_pin.plan_digest,
                 "prep_receipt_sha256": prep_sha,
                 "shard_bytes": sum((base / s).stat().st_size for s in shards)},
        "lora": spec.record(),
        "dtype_policy": {**dtype_policy.record(), "ignored_on_load": list(ignored_patterns),
                         "ignored_keys_copied": len(ignored)},
        "casts": dict(sorted(casts.items())),
        "merge_stats": merge_stats,
        "files": files,
        "bytes": sum(f["size"] for f in files),
        "tools": tool_versions(),
        "timings_s": {"prepare": round(t_prep, 2), "merge": round(t_merge, 2),
                      "reread": round(t_reread, 2), "total": round(time.monotonic() - t0, 2),
                      "per_shard": {s: r["seconds"] for s, r in shard_results.items()}},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt["content_sha256"] = content_digest(receipt)
    tmp = receipt_path(dest).with_name(DERIVED_RECEIPT + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, receipt_path(dest))
    if commit is not None:
        commit()
    return receipt


def verify_merged(derived_dir: Path, *, full: bool = False) -> dict:
    """Check ``derived_dir`` against its receipt: exact file set and sizes, full sha256 of small
    files, header sha256 of shards (``full`` re-hashes the shards too)."""
    from flora_deploy.prep.convert_captioner import header_sha256

    derived_dir = Path(derived_dir)
    receipt = load_receipt(derived_dir)
    if not receipt or receipt.get("ok") is not True or receipt.get("kind") != DERIVED_NAME:
        raise MergeError(f"{derived_dir}: no passing {DERIVED_NAME} receipt")
    if receipt.get("merger", {}).get("version") != MERGER_VERSION:
        raise MergeError("receipt written by another merger version")
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY

    try:
        content = checked_content_digest(receipt)
    except ContentDigestError as e:
        raise MergeError(f"{derived_dir}: {e}") from None
    dpin = DERIVED_PINS_BY_KEY.get(DERIVED_NAME)
    if dpin is not None and dpin.content_sha256 is not None and content != dpin.content_sha256:
        raise MergeError(f"{derived_dir}: content digest {content} != pinned "
                         f"{dpin.content_sha256}")
    want = {f["path"]: f for f in receipt["files"]}
    have = {p.name for p in derived_dir.iterdir() if p.name != DERIVED_RECEIPT}
    if have != set(want):
        raise MergeError(f"file set differs: extra {sorted(have - set(want))}, "
                         f"missing {sorted(set(want) - have)}")
    for name, f in want.items():
        path = derived_dir / name
        if path.stat().st_size != f["size"]:
            raise MergeError(f"{name}: size differs from receipt")
        if f["kind"] == "merged_shard" and not full:
            if header_sha256(path) != f["header_sha256"]:
                raise MergeError(f"{name}: header differs from receipt")
        elif _sha256_file(path) != f["sha256"]:
            raise MergeError(f"{name}: sha256 differs from receipt")
    return {"files": len(want), "full": full, "content_sha256": content,
            "receipt_sha256": hashlib.sha256(receipt_path(derived_dir).read_bytes()).hexdigest()}


# --- equivalence metrics (used by modal_app/convert_app.py::tempseg_equivalence) -------------


def logit_diff(a, b) -> dict:
    """max / mean |a - b| over all logits (FP32), top-1 agreement per position, and whether the
    last position's argmax (the greedy next token) agrees. ``a``/``b``: [..., T, V]."""
    import torch

    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    d = (a32 - b32).abs()
    ta, tb = a32.argmax(-1), b32.argmax(-1)
    return {
        "positions": int(ta.numel()),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean(dtype=torch.float64)),
        "logit_abs_mean": float(a32.abs().mean(dtype=torch.float64)),
        "top1_agree": float((ta == tb).to(torch.float64).mean()),
        "next_token_equal": bool((ta[..., -1] == tb[..., -1]).all()),
    }


def teacher_forced_inputs(inputs: Mapping, generated, pad: Mapping[str, int] | None = None
                          ) -> tuple[dict, int]:
    """Prompt inputs extended by ``generated[:-1]`` so one forward pass yields the logits that
    predicted each generated token (pass ``logits_to_keep=len(generated)``). Per-token tensors
    (shape ``[1, n]``) are extended: ``attention_mask`` with 1, anything else with ``pad`` or 0."""
    import torch

    ids = inputs["input_ids"]
    n = ids.shape[1]
    gen = torch.as_tensor(generated, dtype=ids.dtype).reshape(1, -1)
    if gen.shape[1] == 0:
        raise ValueError("nothing generated")
    ext = gen[:, :-1]
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n):
            if k == "input_ids":
                fill = ext.to(v.dtype)
            else:
                value = 1 if k == "attention_mask" else (pad or {}).get(k, 0)
                fill = torch.full((1, ext.shape[1]), value, dtype=v.dtype)
            out[k] = torch.cat([v, fill.to(v.device)], dim=1)
        else:
            out[k] = v
    return out, gen.shape[1]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token (or the shorter length), ``None`` if identical."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def tensor_digests(named: Iterable[tuple[str, object]], skip: Callable[[str], bool]
                   ) -> dict[str, str]:
    """sha256 of each tensor's raw bytes (dtype + shape included), for names not ``skip``ped."""
    import torch

    out = {}
    for name, t in named:
        if skip(name):
            continue
        v = t.detach().contiguous().cpu()
        h = hashlib.sha256(f"{v.dtype}{list(v.shape)}".encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        out[name] = h.hexdigest()
    return out
