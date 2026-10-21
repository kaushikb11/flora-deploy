"""Gate C comparison and report logic (PLAN.md §9): vLLM 0.24.0 vs NVIDIA's HF processor call.

Compared per request: prompt token ids, per-video ``video_grid_thw`` and per-video
``pixel_values_videos`` (as sha256 digests in fp32 and bf16). M-RoPE position ids are not
compared here: both engines derive them from exactly these ids and grids, and their
implementations are left to Gate D (logits parity).

Pure functions over token-id lists, grids and digests, so they are unit-testable without vLLM or
NVIDIA code (only the digest helpers need torch). Results carry counts, positions and token ids
of *special* tokens only, never prompt text (NVIDIA's prompts must not end up in the repository).

Token layout of one video in both HF and vLLM (Qwen3-VL), for each temporal group ``t``::

    <timestamp text tokens> <|vision_start|> <|video_pad|> * (H*W/merge^2) <|vision_end|>
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpecialIds:
    vision_start: int
    vision_end: int
    video_pad: int


@dataclass(frozen=True)
class FrameRun:
    """One ``vision_start, video_pad * n, vision_end`` run."""

    start: int  # index of vision_start
    pads: int


def frame_runs(ids: Sequence[int], sp: SpecialIds) -> list[FrameRun]:
    """Every vision_start ... vision_end run. Raises if the structure is malformed."""
    runs: list[FrameRun] = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == sp.vision_start:
            j = i + 1
            while j < n and ids[j] == sp.video_pad:
                j += 1
            if j >= n or ids[j] != sp.vision_end:
                raise ValueError(f"vision_start at {i} is not closed by vision_end after pads")
            runs.append(FrameRun(i, j - i - 1))
            i = j + 1
        elif ids[i] == sp.vision_end:
            raise ValueError(f"vision_end at {i} without vision_start")
        else:
            i += 1
    return runs


def bare_pad_positions(ids: Sequence[int], sp: SpecialIds) -> list[int]:
    """Positions of video_pad tokens not inside a vision_start ... vision_end run."""
    out, inside = [], False
    for i, t in enumerate(ids):
        if t == sp.vision_start:
            inside = True
        elif t == sp.vision_end:
            inside = False
        elif t == sp.video_pad and not inside:
            out.append(i)
    return out


def summarize_ids(ids: Sequence[int], sp: SpecialIds) -> dict:
    """Structure counts of a token sequence (no text)."""
    summary = dict(
        length=len(ids),
        vision_start=sum(t == sp.vision_start for t in ids),
        vision_end=sum(t == sp.vision_end for t in ids),
        video_pad=sum(t == sp.video_pad for t in ids),
    )
    try:
        runs = frame_runs(ids, sp)
        summary["frame_groups"] = len(runs)
        summary["pads_per_group"] = sorted({r.pads for r in runs})
    except ValueError as exc:
        summary["structure_error"] = str(exc)
    bare = bare_pad_positions(ids, sp)
    if bare:
        summary["bare_video_pad_positions"] = bare[:16]
    return summary


def expected_runs(grids: Sequence[Sequence[int]], merge_size: int) -> list[int]:
    """Pad count of every frame group, in order, implied by per-video ``(T, H, W)`` grids."""
    out: list[int] = []
    for t, h, w in grids:
        out += [h * w // (merge_size * merge_size)] * t
    return out


def grids_match_runs(ids: Sequence[int], grids: Sequence[Sequence[int]], sp: SpecialIds,
                     merge_size: int) -> bool:
    """Do the token runs in ``ids`` have exactly the frame groups and pads that ``grids`` imply?"""
    try:
        return [r.pads for r in frame_runs(ids, sp)] == expected_runs(grids, merge_size)
    except ValueError:
        return False


def first_diff(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def vllm_timestamps(frames_indices: Sequence[int], fps: float,
                    temporal_patch_size: int) -> list[float]:
    """vLLM 0.24.0 ``Qwen3VLProcessingInfo._calculate_timestamps`` (same as HF's): pad the
    indices to a multiple of ``temporal_patch_size`` with the last index, then average the
    first and last time of each group."""
    idx = list(frames_indices)
    m = temporal_patch_size
    if len(idx) % m:
        idx += [idx[-1]] * (m - len(idx) % m)
    t = [i / fps for i in idx]
    return [(t[i] + t[i + m - 1]) / 2 for i in range(0, len(t), m)]


def simulate_vllm_expansion(prompt_ids: Sequence[int], grids: Sequence[Sequence[int]],
                            timestamps: Sequence[Sequence[float]],
                            encode: Callable[[str], list[int]], sp: SpecialIds,
                            merge_size: int) -> list[int]:
    """Model of vLLM 0.24.0's video prompt update (``_replace_video_token_placeholders`` +
    ``get_video_repl``): each ``vision_start, video_pad, vision_end`` triplet of the tokenized
    prompt becomes, per temporal group, ``encode(f"<{t:.1f} seconds>")`` + vision_start +
    pads + vision_end. A bare video_pad is not a target. Raises like vLLM when the number of
    triplets differs from the number of videos."""
    out: list[int] = []
    k, i, n = 0, 0, len(prompt_ids)
    triplet = [sp.vision_start, sp.video_pad, sp.vision_end]
    while i < n:
        if list(prompt_ids[i:i + 3]) == triplet:
            if k >= len(grids):
                raise RuntimeError("more video placeholders than videos")
            t, h, w = grids[k]
            if len(timestamps[k]) != t:
                raise RuntimeError(f"video {k}: {len(timestamps[k])} timestamps for T={t}")
            for stamp in timestamps[k]:
                out += encode(f"<{stamp:.1f} seconds>")
                out += [sp.vision_start] + [sp.video_pad] * (h * w // merge_size**2)
                out.append(sp.vision_end)
            k += 1
            i += 3
        else:
            out.append(prompt_ids[i])
            i += 1
    if k != len(grids):
        raise RuntimeError(f"Found {k} video placeholders but expected {len(grids)}")
    return out


PIXEL_DTYPES = ("float32", "bfloat16")


def tensor_digest(t: Any, dtype: str) -> str:
    """sha256 of a torch tensor's values cast to ``dtype`` (shape included), on CPU."""
    import torch

    x = t.detach().to("cpu", getattr(torch, dtype)).contiguous()
    raw = x.view(torch.int16) if dtype == "bfloat16" else x
    h = hashlib.sha256(repr(tuple(x.shape)).encode())
    h.update(raw.numpy().tobytes())
    return h.hexdigest()


def video_pixel_digests(pixel_values: Any, grids: Sequence[Sequence[int]]) -> list[dict]:
    """Per-video digests of HF's concatenated ``pixel_values_videos`` (``T*H*W`` rows each)."""
    out, start = [], 0
    for t, h, w in grids:
        n = int(t) * int(h) * int(w)
        part = pixel_values[start:start + n]
        out.append({"rows": int(part.shape[0]), **{d: tensor_digest(part, d)
                                                   for d in PIXEL_DTYPES}})
        start += n
    if start != pixel_values.shape[0]:
        raise ValueError(f"grids cover {start} pixel rows, tensor has {pixel_values.shape[0]}")
    return out


def pixels_equal_per_video(hf_digests: Sequence[dict], vllm_parts: Sequence[Any]) -> list[bool]:
    """Per video: does vLLM's ``pixel_values_videos`` item equal HF's? Compared in vLLM's own
    dtype (bf16: HF cast to bf16; fp32: exact). Any other dtype or a count mismatch is unequal."""
    if len(hf_digests) != len(vllm_parts):
        return [False] * max(len(hf_digests), len(vllm_parts))
    out = []
    for ref, part in zip(hf_digests, vllm_parts, strict=True):
        dt = str(part.dtype).removeprefix("torch.")
        rows = part.reshape(-1, part.shape[-1])
        out.append(dt in PIXEL_DTYPES and int(rows.shape[0]) == ref["rows"]
                   and tensor_digest(rows, dt) == ref[dt])
    return out


@dataclass
class Comparison:
    case_id: str
    index: int
    source: str
    variant: str
    ids_equal: bool
    grids_equal: bool
    runs_match_grid: bool
    hf: dict
    vllm: dict
    hf_grids: list
    vllm_grids: list
    first_diff: int | None = None
    error: str | None = None
    # one entry per video; None = not compared (counts as a failure)
    pixels_equal: list[bool] | None = None
    # LLM.generate saw the rendered ids; None = generate not run on this variant
    generate_ids_equal: bool | None = None
    extras: dict = field(default_factory=dict)

    @property
    def pixels_ok(self) -> bool:
        p = self.pixels_equal
        return bool(p) and len(p) == len(self.hf_grids) and all(p)

    @property
    def passed(self) -> bool:
        return (self.error is None and self.ids_equal and self.grids_equal
                and self.runs_match_grid and self.pixels_ok
                and self.generate_ids_equal is not False
                and "bare_video_pad_positions" not in self.hf
                and "bare_video_pad_positions" not in self.vllm)

    def to_dict(self) -> dict:
        return {**asdict(self), "pixels_ok": self.pixels_ok, "passed": self.passed}


def compare(*, case_id: str, index: int, source: str, variant: str, hf_ids: Sequence[int],
            hf_grids: Sequence[Sequence[int]], vllm_ids: Sequence[int] | None,
            vllm_grids: Sequence[Sequence[int]] | None, sp: SpecialIds, merge_size: int,
            pixels_equal: Sequence[bool] | None = None, generate_ids_equal: bool | None = None,
            error: str | None = None, extras: dict | None = None) -> Comparison:
    """Compare one request's vLLM model inputs with NVIDIA's HF call.

    PASS needs: no error, identical ids, identical grids, vLLM's frame runs as the grids imply,
    pixels equal for every video, ``generate`` (if run) saw the same ids, and no stray bare
    ``video_pad`` in either sequence."""
    hf_ids, hf_grids = list(hf_ids), [list(g) for g in hf_grids]
    pix = None if pixels_equal is None else [bool(p) for p in pixels_equal]
    if vllm_ids is None:
        return Comparison(case_id, index, source, variant, False, False, False,
                          summarize_ids(hf_ids, sp), {}, hf_grids, [], None,
                          error or "no vLLM output", pix, generate_ids_equal,
                          dict(extras or {}))
    vllm_ids = list(vllm_ids)
    vgrids = [list(g) for g in (vllm_grids or [])]
    return Comparison(
        case_id=case_id, index=index, source=source, variant=variant,
        ids_equal=hf_ids == vllm_ids,
        grids_equal=hf_grids == vgrids,
        runs_match_grid=grids_match_runs(vllm_ids, vgrids, sp, merge_size),
        hf=summarize_ids(hf_ids, sp), vllm=summarize_ids(vllm_ids, sp),
        hf_grids=hf_grids, vllm_grids=vgrids,
        first_diff=first_diff(hf_ids, vllm_ids), error=error, pixels_equal=pix,
        generate_ids_equal=generate_ids_equal, extras=dict(extras or {}),
    )


# Error messages that may reach the report verbatim: the matched text only (fixed words,
# numbers, identifiers). Anything else is reduced to its exception class.
_SAFE_ERRORS = (
    re.compile(r"Failed to apply prompt replacement for mm_items\['video'\]\[\d+\]"),
    re.compile(r"Found \d+ video placeholders but expected \d+"),
    re.compile(r"Expected there to be \d+ prompt updates corresponding to \d+ video items, "
               r"but instead found \d+ prompt updates!"),
    re.compile(r"Failed to apply \w+"),
    re.compile(r"The decoder prompt \(length \d+\) is longer than the maximum model length of "
               r"\d+"),
    re.compile(r"CUDA out of memory"),
)


def sanitize_error(exc: BaseException) -> str:
    """Exception class plus, only if its first line starts with one of ``_SAFE_ERRORS``, the
    matched text. An allowlist, because vLLM echoes prompts in several error formats and NVIDIA's
    prompt text must not reach the report."""
    msg = str(exc).splitlines()[0] if str(exc) else ""
    for pattern in _SAFE_ERRORS:
        m = pattern.match(msg)
        if m:
            tail = "" if m.end() == len(msg) else " [rest elided]"
            return f"{type(exc).__name__}: {m.group(0)}{tail}"
    return f"{type(exc).__name__}: [message elided]" if msg else type(exc).__name__


def case_verdicts(rows: Sequence[dict], variant: str) -> dict[str, bool]:
    """Per case id: True iff every request of that case passed in ``variant``."""
    out: dict[str, bool] = {}
    for r in rows:
        if r["variant"] != variant:
            continue
        out[r["case_id"]] = out.get(r["case_id"], True) and bool(r["passed"])
    return out


def render_report(rows: Sequence[dict], *, primary: str = "wrapped") -> str:
    """Markdown tables: one line per case and variant, then one line per request."""
    variants = sorted({r["variant"] for r in rows}, key=lambda v: (v != primary, v))
    lines = ["| case | " + " | ".join(variants) + " |",
             "|---|" + "---|" * len(variants)]
    cases = sorted({r["case_id"] for r in rows})
    verdicts = {v: case_verdicts(rows, v) for v in variants}
    for c in cases:
        cells = []
        for v in variants:
            n = [r for r in rows if r["case_id"] == c and r["variant"] == v]
            ok = verdicts[v].get(c)
            cells.append("n/a" if ok is None else
                         f"{'PASS' if ok else 'FAIL'} ({sum(r['passed'] for r in n)}/{len(n)})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "| case | # | source | variant | ids | grid | runs | pixels | gen | HF len | "
              "vLLM len | first diff | grids (T,H,W) | error |", "|" + "---|" * 14]
    yn = lambda b: "=" if b else "≠"  # noqa: E731
    for r in sorted(rows, key=lambda r: (r["case_id"], r["source"], r["index"], r["variant"])):
        pix = r.get("pixels_equal")
        gen = r.get("generate_ids_equal")
        lines.append(
            f"| {r['case_id']} | {r['index']} | {r['source']} | {r['variant']} | "
            f"{yn(r['ids_equal'])} | {yn(r['grids_equal'])} | {yn(r['runs_match_grid'])} | "
            f"{'-' if pix is None else ''.join(yn(p) for p in pix)} | "
            f"{'-' if gen is None else yn(gen)} | "
            f"{r['hf'].get('length', '')} | {r['vllm'].get('length', '')} | "
            f"{'' if r['first_diff'] is None else r['first_diff']} | "
            f"{';'.join(','.join(map(str, g)) for g in r['hf_grids'])} | "
            f"{r['error'] or ''} |")
    return "\n".join(lines) + "\n"
