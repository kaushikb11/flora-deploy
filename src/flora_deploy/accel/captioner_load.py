"""Make NVIDIA's ``Session.load('captioner')`` load our BF16 Captioner (OPT-2).

``prep/convert_captioner.py`` writes a BF16 copy of the FP32 Captioner whose tensors already
carry the dtype NVIDIA's loader would give them. This module redirects NVIDIA's loader to it,
by replacing two attributes of NVIDIA's ``batch_utils.models`` module at runtime (NVIDIA's files
are never touched):

- ``models.snapshot``: ``snapshot('captioner')`` returns the derived directory on the weights
  Volume (read-only is fine: nothing is written, so no local copy is needed). Every other kind
  goes to NVIDIA's original function.
- ``models.validate_manifest``: for the derived directory, NVIDIA's manifest re-hash (whose
  ``SOURCE_MANIFEST.sha256`` describes the FP32 files) is replaced by a check against our
  ``derived_receipt.json``. Any other path goes to NVIDIA's original function.

Everything else in ``Session.load`` runs unchanged: ``_keep_native_gdn_fp32``, the same
``from_pretrained(dtype=torch.bfloat16, device_map='balanced', ...)`` call, the loading-info
check, ``verify_cuda_residency`` and ``verify_gdn`` (which re-reads ``A_log`` / ``dt_bias`` from
the derived shards and asserts FP32 and exact values after placement).

Receipt check (``verify_derived``):

- fast (default): schema, ``ok``, converter version, source pin revision and NVIDIA manifest
  digest equal ``pins.py``; the files on disk are exactly the receipt's files with the recorded
  sizes; every non-shard file (config, tokenizer, index, ...) and any shard under 64 MiB is fully
  sha256'd; every other shard's safetensors header (names, dtypes, shapes, offsets) is
  sha256'd. Reads ~20 MB.
- full (``full=True`` or ``FLORA_CAPTIONER_FULL_VERIFY=1``): sha256 of every byte.

The identity proof is ``modal_app/convert_app.py::identity`` (``docs/captioner-bf16.md``); the
comparison helpers it uses (``compare_models``, ``fixed_forward``) live here.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from flora_deploy.config.pins import PINS_BY_KEY, RepoPin
from flora_deploy.layout import WEIGHTS_MOUNT, derived_path
from flora_deploy.prep.convert_captioner import (
    CONVERTER_VERSION,
    DERIVED_NAME,
    DERIVED_RECEIPT,
    RECEIPT_SCHEMA,
    SOURCE_KEY,
    header_sha256,
)

KIND = "captioner"
FULL_VERIFY_ENV = "FLORA_CAPTIONER_FULL_VERIFY"
SMALL_FILE = 64 * 2**20
_MARK = "__flora_accel_captioner_bf16__"


class DerivedReceiptError(RuntimeError):
    """The derived BF16 Captioner does not match its receipt."""


def default_derived_dir(root=WEIGHTS_MOUNT) -> Path:
    return Path(derived_path(DERIVED_NAME, PINS_BY_KEY[SOURCE_KEY], root))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    buf = bytearray(16 * 2**20)
    view = memoryview(buf)
    with path.open("rb", buffering=0) as f:
        while n := f.readinto(buf):
            h.update(view[:n])
    return h.hexdigest()


def verify_derived(
    derived_dir: Path, *, full: bool = False, workers: int = 8, pin: RepoPin | None = None
) -> dict:
    """Check ``derived_dir`` against its ``derived_receipt.json`` (see module docstring).
    ``pin`` defaults to ``PINS_BY_KEY["captioner"]`` (tests pass a synthetic one)."""
    t0 = time.monotonic()
    derived_dir = Path(derived_dir)
    pin = pin or PINS_BY_KEY[SOURCE_KEY]
    try:
        receipt = json.loads((derived_dir / DERIVED_RECEIPT).read_text())
    except (OSError, ValueError) as e:
        raise DerivedReceiptError(f"no readable receipt in {derived_dir}: {e}") from e
    problems = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        problems.append(f"schema {receipt.get('schema')} != {RECEIPT_SCHEMA}")
    if receipt.get("ok") is not True:
        problems.append("receipt is not passing")
    if receipt.get("kind") != DERIVED_NAME:
        problems.append(f"kind {receipt.get('kind')!r}")
    if receipt.get("converter", {}).get("version") != CONVERTER_VERSION:
        problems.append(f"converter version {receipt.get('converter', {}).get('version')}")
    src = receipt.get("source", {})
    if src.get("pin", {}).get("revision") != pin.revision:
        problems.append(f"source revision {src.get('pin', {}).get('revision')} != {pin.revision}")
    if src.get("manifest_sha256") != dict(pin.expected).get("manifest_sha256"):
        problems.append("source manifest digest differs from pins.py")
    files = {f["path"]: f for f in receipt.get("files", [])}
    if not files:
        problems.append("receipt lists no files")
    if problems:
        raise DerivedReceiptError(f"{derived_dir}: " + "; ".join(problems))

    on_disk = {p.name for p in derived_dir.iterdir() if p.name != DERIVED_RECEIPT}
    if on_disk != set(files):
        raise DerivedReceiptError(
            f"{derived_dir}: files differ from receipt: missing {sorted(set(files) - on_disk)}, "
            f"extra {sorted(on_disk - set(files))}"
        )

    def check(entry: dict) -> tuple[str, int]:
        path = derived_dir / entry["path"]
        size = path.stat().st_size
        if size != entry["size"]:
            return f"{entry['path']}: size {size} != {entry['size']}", 0
        if full or size < SMALL_FILE or "header_sha256" not in entry:
            if _sha256(path) != entry["sha256"]:
                return f"{entry['path']}: sha256 mismatch", size
            return "", size
        if header_sha256(path) != entry["header_sha256"]:
            return f"{entry['path']}: safetensors header mismatch", 0
        return "", 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(check, files.values()))
    bad = [msg for msg, _ in results if msg]
    if bad:
        raise DerivedReceiptError(f"{derived_dir}: " + "; ".join(bad))
    return {
        "mode": "full" if full else "fast",
        "files": len(files),
        "bytes": sum(f["size"] for f in files.values()),
        "bytes_hashed": sum(n for _, n in results),
        "receipt_sha256": hashlib.sha256((derived_dir / DERIVED_RECEIPT).read_bytes()).hexdigest(),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


@dataclass
class CaptionerBf16Patch:
    """Installed redirection; ``uninstall()`` restores NVIDIA's original attributes."""

    models: ModuleType
    derived_dir: Path
    full_verify: bool
    original_snapshot: Callable
    original_validate: Callable
    pin: RepoPin | None = None
    verifications: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self, kind):
        if kind == KIND:
            return self.derived_dir
        return self.original_snapshot(kind)

    def validate_manifest(self, path):
        if Path(path) != self.derived_dir:
            return self.original_validate(path)
        with self._lock:
            # A full re-hash is done once per patch; the fast check runs on every load.
            full = self.full_verify and not any(v["mode"] == "full" for v in self.verifications)
            report = verify_derived(self.derived_dir, full=full, pin=self.pin)
            self.verifications.append(report)
        runtime_event = getattr(self.models, "runtime_event", None)
        if runtime_event is not None:
            runtime_event("manifest_replaced_by_derived_receipt", kind=KIND,
                          directory=str(self.derived_dir), **report)
        return None

    def uninstall(self) -> None:
        if getattr(self.models.snapshot, _MARK, None) is self:
            self.models.snapshot = self.original_snapshot
        if getattr(self.models.validate_manifest, _MARK, None) is self:
            self.models.validate_manifest = self.original_validate

    def __enter__(self) -> CaptionerBf16Patch:
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()


def install(
    nv_or_models: Any,
    derived_dir: Path | str | None = None,
    *,
    full_verify: bool | None = None,
    verify_now: bool = True,
    pin: RepoPin | None = None,
) -> CaptionerBf16Patch:
    """Redirect NVIDIA's ``models`` module (or a linked family handle's ``.models``) to the BF16
    Captioner. ``verify_now`` runs the receipt check immediately, so a bad derived copy fails
    before anything is loaded (it runs again inside ``Session.load``)."""
    models = getattr(nv_or_models, "models", nv_or_models)
    if not hasattr(models, "Session") or not hasattr(models, "snapshot"):
        raise TypeError(f"{models!r} is not NVIDIA's batch_utils.models")
    if getattr(models.snapshot, _MARK, None) is not None:
        raise RuntimeError("the BF16 Captioner patch is already installed")
    derived = Path(derived_dir) if derived_dir is not None else default_derived_dir()
    if full_verify is None:
        full_verify = os.environ.get(FULL_VERIFY_ENV, "0") == "1"
    patch = CaptionerBf16Patch(models, derived, bool(full_verify), models.snapshot,
                               models.validate_manifest, pin)
    if verify_now:
        patch.verifications.append(verify_derived(derived, full=False, pin=pin))

    def snapshot(kind):
        return patch.snapshot(kind)

    def validate_manifest(path):
        return patch.validate_manifest(path)

    setattr(snapshot, _MARK, patch)
    setattr(validate_manifest, _MARK, patch)
    models.snapshot = snapshot
    models.validate_manifest = validate_manifest
    return patch


# --- identity proof helpers ------------------------------------------------------------------


def _bits(t):
    import torch

    t = t.detach()
    if t.dim() == 0:
        t = t.reshape(1)
    return t.contiguous().reshape(-1).view(torch.uint8)


def tensors_equal_bits(a, b) -> bool:
    import torch

    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and a.device == b.device and torch.equal(_bits(a), _bits(b)))


def compare_models(a, b) -> dict:
    """Every parameter and buffer of ``a`` vs ``b``: same names, dtype, shape, device, bytes."""
    out: dict = {}
    for what, get in (("parameters", lambda m: m.named_parameters(remove_duplicate=False)),
                      ("buffers", lambda m: m.named_buffers(remove_duplicate=False))):
        ta, tb = dict(get(a)), dict(get(b))
        diffs = []
        names_equal = set(ta) == set(tb)
        nbytes = 0
        for name in sorted(set(ta) & set(tb)):
            x, y = ta[name], tb[name]
            if not tensors_equal_bits(x, y):
                diffs.append({"name": name, "a": [str(x.dtype), list(x.shape), str(x.device)],
                              "b": [str(y.dtype), list(y.shape), str(y.device)]})
            nbytes += x.numel() * x.element_size()
        out[what] = {
            "count": len(ta),
            "names_equal": names_equal,
            "only_a": sorted(set(ta) - set(tb))[:10],
            "only_b": sorted(set(tb) - set(ta))[:10],
            "bytes_compared": nbytes,
            "mismatches": len(diffs),
            "first_mismatches": diffs[:10],
            "dtypes": dict(sorted(Counter(str(t.dtype) for t in ta.values()).items())),
        }
    out["identical"] = all(out[w]["names_equal"] and out[w]["mismatches"] == 0
                           for w in ("parameters", "buffers"))
    return out


def fixed_forward(model, *, seq_len: int = 32, seed: int = 1234, vocab: int | None = None):
    """Logits of one forward pass on a fixed, seeded text-only input."""
    import torch

    vocab = vocab or model.config.get_text_config().vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, seq_len), generator=g)
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(input_ids=ids.to(device), attention_mask=torch.ones_like(ids).to(device),
                    use_cache=False)
    return out.logits.detach()


def logits_digest(logits) -> str:
    return hashlib.sha256(_bits(logits).cpu().numpy().tobytes()).hexdigest()
