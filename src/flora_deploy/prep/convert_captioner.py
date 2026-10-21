"""One-time FP32 -> BF16 conversion of the Captioner checkpoint (OPT-2, ``docs/captioner-bf16.md``).

NVIDIA's Captioner snapshot stores every tensor in FP32 (109.4 GB). NVIDIA's loader
(``batch_utils/models.py`` ``Session.load``) builds it with
``AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, ...)`` after
``production_gdn_fix._keep_native_gdn_fp32`` has added ``A_log`` / ``dt_bias`` to
``Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict``. Transformers then materialises each
checkpoint tensor with ``tensor.to(device, dtype=<target>)`` (``core_model_loading``), where
``<target>`` is, in order:

1. the ``dtype_plan`` entry (``model._get_dtype_plan(bf16)``) whose glob matches the key: FP32
   for ``A_log`` / ``dt_bias``;
2. otherwise the dtype of the model's own (meta) parameter, if it differs from the load dtype;
3. otherwise the load dtype, BF16.

Keys the model does not have (``*._extra_state``) are "unexpected" and never loaded.

``convert`` applies exactly that rule ahead of time, with the same torch cast, and writes a
checkpoint whose every tensor already has its target dtype. Loading it with NVIDIA's unchanged
call is then a no-op cast, so the in-memory model is bit-identical (proved on GPU by
``modal_app/convert_app.py::identity``). The rule is not hand-written: the dtype plan and the meta
dtypes come from transformers itself, with NVIDIA's own ``_keep_native_gdn_fp32`` applied.

Layout written to ``layout.derived_path("captioner-bf16", PINS_BY_KEY["captioner"])``:

- every non-weight file of the snapshot, byte-for-byte (``config.json`` keeps its
  ``"dtype": "float32"``; NVIDIA passes ``dtype`` explicitly, so it is never read);
- ``model.safetensors.index.json`` with the same ``weight_map`` (same shard names, same
  key -> shard mapping, which ``verify_gdn`` reads) and the new ``total_size``;
- ``model-XXXXX-of-XXXXX.safetensors`` in BF16 (plus FP32 ``A_log`` / ``dt_bias`` and the
  unexpected ``_extra_state`` tensors copied unchanged), same tensor names and shapes;
- ``derived_receipt.json`` (written last): sha256, size and header sha256 of every file, the
  source pin and NVIDIA manifest digest, each source shard's sha256 as hashed while converting,
  the dtype policy, the cast counts, converter version + module sha256, torch/transformers/
  safetensors versions.

Memory is bounded by one source shard per worker (read once into a buffer, sha256'd while read,
tensors viewed in place, each cast tensor streamed to the output file).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import struct
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from flora_deploy.config.pins import PINS_BY_KEY, RepoPin
from flora_deploy.prep.verify import parse_sha256_manifest, pin_record

DERIVED_NAME = "captioner-bf16"
DERIVED_RECEIPT = "derived_receipt.json"
RECEIPT_SCHEMA = 1
# Bump whenever the conversion rule or the output format changes; the runtime patch refuses a
# receipt written by another version.
CONVERTER_VERSION = 1
SOURCE_KEY = "captioner"
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "SOURCE_MANIFEST.sha256"
SHARD_RE = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
# Snapshot files that are not weights; copied verbatim (each checked against NVIDIA's manifest).
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
    "MODEL_PROVENANCE.json",
    MANIFEST_NAME,
)
CHUNK = 64 * 2**20
ALIGN = 8  # safetensors pads the header so tensor data starts 8-byte aligned


class ConversionError(RuntimeError):
    pass


# --- safetensors (format: u64le header length, JSON header, raw little-endian tensor bytes) ----

_ST_TO_TORCH = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
}


def st_dtype(name: str):
    import torch

    return getattr(torch, _ST_TO_TORCH[name])


def torch_to_st(dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    for st, tn in _ST_TO_TORCH.items():
        if tn == name:
            return st
    raise ConversionError(f"unsupported dtype {dtype}")


def read_header(f) -> tuple[int, dict, bytes]:
    """(data start offset, header dict, raw header bytes incl. the length prefix)."""
    prefix = f.read(8)
    if len(prefix) != 8:
        raise ConversionError("truncated safetensors file")
    (n,) = struct.unpack("<Q", prefix)
    if n > 100 * 2**20:
        raise ConversionError(f"implausible safetensors header length {n}")
    raw = f.read(n)
    if len(raw) != n:
        raise ConversionError("truncated safetensors header")
    return 8 + n, json.loads(raw), prefix + raw


def header_sha256(path: Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.sha256(read_header(f)[2]).hexdigest()


def encode_header(entries: list[tuple[str, str, list[int], int]], metadata: dict | None) -> bytes:
    """Header bytes (length prefix included) for ``entries`` = [(name, st dtype, shape, nbytes)]
    laid out back to back in that order."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % ALIGN)
    return struct.pack("<Q", len(raw)) + raw


# --- dtype policy --------------------------------------------------------------------------


@dataclass(frozen=True)
class DtypePolicy:
    """Target dtype per checkpoint key, mirroring transformers' ``core_model_loading`` step 3."""

    load_dtype: str  # "bfloat16"
    plan: dict[str, str]  # dtype_plan glob -> dtype name (from transformers + NVIDIA's patch)
    model_dtypes: dict[str, str]  # meta state_dict key -> dtype name
    _regex: re.Pattern | None = field(default=None, compare=False, repr=False)

    @classmethod
    def build(cls, load_dtype: str, plan: dict[str, str], model_dtypes: dict[str, str]):
        regex = None
        if plan:
            # Same construction as transformers.core_model_loading.build_glob_alternation.
            branches = [f"(?P<g{i}>{g.replace('*', '.*')})" for i, g in enumerate(plan)]
            regex = re.compile("|".join(branches))
        return cls(load_dtype, dict(plan), dict(model_dtypes), regex)

    def target(self, key: str) -> str | None:
        """Dtype name the loader gives ``key``; ``None`` if the model never loads it."""
        if key not in self.model_dtypes:
            return None
        if self._regex is not None and (m := self._regex.search(key)) is not None:
            return self.plan[list(self.plan)[int(m.lastgroup[1:])]]
        # Rule 2 (own dtype if it differs from the load dtype) and rule 3 (the load dtype)
        # both come down to the model's own parameter dtype.
        return self.model_dtypes[key]

    def record(self) -> dict:
        kept = sorted(k for k in self.model_dtypes if self.target(k) != self.load_dtype)
        return {
            "load_dtype": self.load_dtype,
            "dtype_plan": self.plan,
            "non_load_dtype_keys": len(kept),
            "non_load_dtype_key_sha256": hashlib.sha256("\n".join(kept).encode()).hexdigest(),
        }


def nvidia_dtype_policy(source: Path, keep_fp32_patch: Callable[[type], None]) -> DtypePolicy:
    """Ask transformers what NVIDIA's ``from_pretrained(dtype=bf16)`` call does per key.

    ``keep_fp32_patch`` must be NVIDIA's own ``production_gdn_fix._keep_native_gdn_fp32``
    (from the linked SEGCAP family); it is applied to ``Qwen3_5PreTrainedModel`` exactly as
    ``Session.load`` does before loading. The model is built on the meta device (no memory)."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    keep_fp32_patch(Qwen3_5PreTrainedModel)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    plan = {
        k: str(v).removeprefix("torch.")
        for k, v in model._get_dtype_plan(torch.bfloat16).items()
    }
    dtypes = {k: str(v.dtype).removeprefix("torch.") for k, v in model.state_dict().items()}
    return DtypePolicy.build("bfloat16", plan, dtypes)


# --- conversion ----------------------------------------------------------------------------


def _sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    buf = bytearray(chunk)
    view = memoryview(buf)
    with Path(path).open("rb", buffering=0) as f:
        while n := f.readinto(buf):
            h.update(view[:n])
    return h.hexdigest()


def _read_whole(path: Path, chunk: int = CHUNK) -> tuple[bytearray, str]:
    """Read ``path`` into one buffer, hashing as it is read: the bytes converted are the bytes
    hashed."""
    size = path.stat().st_size
    buf = bytearray(size)
    view = memoryview(buf)
    h = hashlib.sha256()
    off = 0
    with path.open("rb", buffering=0) as f:
        while off < size:
            n = f.readinto(view[off : off + chunk])
            if not n:
                raise ConversionError(f"{path}: short read at {off}/{size}")
            h.update(view[off : off + n])
            off += n
        if f.read(1):
            raise ConversionError(f"{path}: grew while reading")
    return buf, h.hexdigest()


def convert_shard(src: Path, dest: Path, policy: DtypePolicy, expected_sha256: str) -> dict:
    """Convert one shard. Raises if the source does not hash to NVIDIA's manifest value."""
    import torch

    t0 = time.monotonic()
    buf, src_sha = _read_whole(src)
    if src_sha != expected_sha256:
        raise ConversionError(f"{src.name}: sha256 {src_sha} != NVIDIA manifest {expected_sha256}")
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
            info["shape"]
        )

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
        tag = "unexpected_copied" if target is None else "loaded"
        casts[f"{info['dtype']}->{out_st}:{tag}"] += 1

    head = encode_header(plan_out, metadata)
    h = hashlib.sha256(head)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with partial.open("wb") as f:
        f.write(head)
        for name, out_st, _, nbytes in plan_out:
            t = source_tensor(name)
            out = t if torch_to_st(t.dtype) == out_st else t.to(st_dtype(out_st))
            raw = out.contiguous().reshape(-1).view(torch.uint8).numpy().data
            if len(raw) != nbytes:
                raise ConversionError(f"{name}: wrote {len(raw)} bytes, planned {nbytes}")
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
        "read_s": round(t_read, 2),
        "seconds": round(time.monotonic() - t0, 2),
    }


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def tool_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for dist in ("torch", "transformers", "safetensors"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def receipt_path(derived_dir: Path) -> Path:
    return Path(derived_dir) / DERIVED_RECEIPT


def load_receipt(derived_dir: Path) -> dict | None:
    try:
        return json.loads(receipt_path(derived_dir).read_text())
    except (OSError, ValueError):
        return None


def _source_manifest(source: Path, pin) -> dict[str, str]:
    raw = (source / MANIFEST_NAME).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    want = dict(pin.expected).get("manifest_sha256")
    if got != want:
        raise ConversionError(f"{MANIFEST_NAME} sha256 {got} != pinned {want}")
    entries = {name: digest for digest, name in parse_sha256_manifest(raw.decode())}
    # The manifest does not list itself; its own bytes are bound by the pinned digest.
    return {**entries, MANIFEST_NAME: got}


def convert(
    source: Path,
    dest: Path,
    keep_fp32_patch: Callable[[type], None] | None = None,
    *,
    policy: DtypePolicy | None = None,
    workers: int = 3,
    threads_per_worker: int = 2,
    force: bool = False,
    provenance: dict | None = None,
    commit: Callable[[], None] | None = None,
    pin: RepoPin | None = None,
) -> dict:
    """Convert the snapshot at ``source`` into ``dest`` (see module docstring); returns the receipt.

    Idempotent: an existing passing receipt of this converter version is returned as is unless
    ``force``. ``commit`` (e.g. ``volume.commit``) is called after the receipt is written."""
    import torch

    source, dest = Path(source), Path(dest)
    pin = pin or PINS_BY_KEY[SOURCE_KEY]
    existing = load_receipt(dest)
    if (not force and existing and existing.get("ok") is True
            and existing.get("converter", {}).get("version") == CONVERTER_VERSION
            and existing.get("source", {}).get("pin", {}).get("revision") == pin.revision):
        return {**existing, "reused": True}

    t0 = time.monotonic()
    manifest = _source_manifest(source, pin)
    index = json.loads((source / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if not shards or not all(SHARD_RE.fullmatch(s) for s in shards):
        raise ConversionError(f"unexpected shard names {shards[:3]}")
    on_disk = sorted(p.name for p in source.glob("model-*.safetensors"))
    if on_disk != shards:
        raise ConversionError(f"shards on disk {on_disk} != index {shards}")
    missing = [s for s in [*shards, INDEX_NAME, *COPIED_FILES] if s not in manifest]
    if missing:
        raise ConversionError(f"not covered by NVIDIA's manifest: {missing}")
    if policy is None:
        if keep_fp32_patch is None:
            raise ConversionError("need NVIDIA's _keep_native_gdn_fp32 (or an explicit policy)")
        policy = nvidia_dtype_policy(source, keep_fp32_patch)
    unknown = [k for k in weight_map if policy.target(k) is None and "_extra_state" not in k]
    if unknown:
        raise ConversionError(f"checkpoint keys the model does not have: {unknown[:5]}")
    absent = [k for k in policy.model_dtypes if k not in weight_map]
    if absent:
        raise ConversionError(f"model keys missing from the checkpoint: {absent[:5]}")

    if dest.exists() and force:
        old = receipt_path(dest)
        if old.exists():
            old.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    t_policy = time.monotonic() - t0

    # 1. small files, verbatim
    copied = []
    for name in (INDEX_NAME, *COPIED_FILES):
        sha = _sha256_file(source / name)
        if sha != manifest[name]:
            raise ConversionError(f"{name}: sha256 {sha} != NVIDIA manifest")
        if name != INDEX_NAME:
            shutil.copyfile(source / name, dest / name)
        copied.append((name, sha))

    # 2. shards (bounded: ``workers`` source shards in memory at once)
    # Containers may report the host's CPU count, so the thread count is explicit.
    torch.set_num_threads(max(1, threads_per_worker))
    t1 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = {s: pool.submit(convert_shard, source / s, dest / s, policy, manifest[s])
                for s in shards}
        shard_results = {s: j.result() for s, j in jobs.items()}
    t_convert = time.monotonic() - t1

    # 3. index: same weight_map, new total_size (sum of tensor bytes, as transformers writes it)
    total = 0
    for s in shards:
        with (dest / s).open("rb") as f:
            _, hdr, _ = read_header(f)
        total += sum(v["data_offsets"][1] - v["data_offsets"][0]
                     for k, v in hdr.items() if k != "__metadata__")
    new_index = {**index, "metadata": {**index.get("metadata", {}), "total_size": total}}
    (dest / INDEX_NAME).write_text(json.dumps(new_index, indent=2, sort_keys=True) + "\n")

    # 4. re-read everything from the destination (what is stored is what the receipt binds)
    t2 = time.monotonic()
    names = sorted([*shards, INDEX_NAME, *COPIED_FILES])
    with ThreadPoolExecutor(max_workers=8) as pool:
        stored = dict(zip(names, pool.map(lambda n: _sha256_file(dest / n), names), strict=True))
    for s in shards:
        if stored[s] != shard_results[s]["written_sha256"]:
            raise ConversionError(f"{s}: stored bytes differ from the bytes written")
    for name, sha in copied:
        if name != INDEX_NAME and stored[name] != sha:
            raise ConversionError(f"{name}: copy differs from source")
    t_reread = time.monotonic() - t2

    extra = sorted(p.name for p in dest.iterdir()
                   if p.name not in stored and p.name != DERIVED_RECEIPT)
    if extra:
        raise ConversionError(f"unexpected files in {dest}: {extra}")

    files = []
    for name in names:
        entry = {"path": name, "size": (dest / name).stat().st_size, "sha256": stored[name]}
        if name in shard_results:
            r = shard_results[name]
            entry.update(kind="converted", header_sha256=r["header_sha256"],
                         source_sha256=r["source_sha256"], tensors=r["tensors"])
        elif name == INDEX_NAME:
            entry.update(kind="rewritten_index", source_sha256=manifest[name])
        else:
            entry.update(kind="copied", source_sha256=manifest[name])
        files.append(entry)
    casts: Counter = Counter()
    for r in shard_results.values():
        casts.update(r["casts"])
    src_bytes = sum((source / s).stat().st_size for s in shards)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "ok": True,
        "kind": DERIVED_NAME,
        "converter": {"version": CONVERTER_VERSION, "module_sha256": module_sha256(),
                      "git_commit": (provenance or {}).get("git_commit"),
                      "git_dirty": (provenance or {}).get("git_dirty")},
        "source": {"pin": pin_record(pin), "manifest_sha256": dict(pin.expected)["manifest_sha256"],
                   "shard_bytes": src_bytes},
        "dtype_policy": policy.record(),
        "casts": dict(sorted(casts.items())),
        "files": files,
        "bytes": sum(f["size"] for f in files),
        "tools": tool_versions(),
        "timings_s": {"policy": round(t_policy, 2), "convert": round(t_convert, 2),
                      "reread": round(t_reread, 2), "total": round(time.monotonic() - t0, 2),
                      "per_shard": {s: r["seconds"] for s, r in shard_results.items()}},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = receipt_path(dest).with_name(DERIVED_RECEIPT + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, receipt_path(dest))
    if commit is not None:
        commit()
    return receipt
