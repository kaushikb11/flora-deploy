"""Gate C comparison and report logic (PLAN.md §8): vLLM 0.24.0 vs NVIDIA's HF processor call.

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
from collections.abc import Callable, Mapping, Sequence
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


def placeholder_problem(ids: Sequence[int], placeholders: Sequence[Sequence[int]],
                        grids: Sequence[Sequence[int]], sp: SpecialIds,
                        merge_size: int) -> str | None:
    """Do vLLM's ``mm_placeholders['video']`` ranges (``[offset, length]`` per video) cover
    exactly where each video's embeddings go? None if so, else why not.

    vLLM 0.24.0 marks a video's whole prompt replacement as its placeholder: from the first
    group's timestamp text to the last group's ``vision_end``. So, per video ``k`` (in order):
    the range lies inside the prompt, starts at or before its first ``vision_start`` with no
    special token in between, ends right after its ``T``-th ``vision_end``, holds exactly
    ``T*H*W/merge^2`` pads, and does not overlap the previous range. Every pad must be inside a
    range. (Where the timestamp text starts is not checked: it needs the tokenizer, and only the
    pads inside a range receive embeddings.)"""
    if len(placeholders) != len(grids):
        return f"{len(placeholders)} placeholder ranges for {len(grids)} videos"
    try:
        runs = frame_runs(ids, sp)
    except ValueError as exc:
        return str(exc)
    specials = {sp.vision_start, sp.vision_end, sp.video_pad}
    i, prev_end, covered = 0, 0, 0
    for k, (rng, (t, h, w)) in enumerate(zip(placeholders, grids, strict=True)):
        off, length = int(rng[0]), int(rng[1])
        mine = runs[i:i + int(t)]
        i += int(t)
        if len(mine) != t:
            return f"video {k}: {len(mine)} frame groups for T={t}"
        first, last_end = mine[0].start, mine[-1].start + mine[-1].pads + 1
        if off < 0 or length < 0 or off + length > len(ids):
            return f"video {k}: range [{off}, {off + length}) outside the prompt ({len(ids)})"
        if off < prev_end:
            return f"video {k}: range starts at {off}, inside the previous one (ends {prev_end})"
        if off > first or any(ids[j] in specials for j in range(off, first)):
            return f"video {k}: range starts at {off}, not at its timestamp before {first}"
        if off + length != last_end + 1:
            return f"video {k}: range ends at {off + length}, its last vision_end is {last_end}"
        pads = sum(ids[j] == sp.video_pad for j in range(off, off + length))
        want = int(t) * int(h) * int(w) // (merge_size * merge_size)
        if pads != want:
            return f"video {k}: {pads} pads in its range, grid implies {want}"
        covered += pads
        prev_end = off + length
    if i != len(runs) or covered != sum(t == sp.video_pad for t in ids):
        return "pads outside every placeholder range"
    return None


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


PRIMARY_VARIANT = "wrapped"


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
    # vLLM's placeholder ranges sit on the frame runs (``placeholder_problem``); None = not checked
    placeholders_ok: bool | None = None
    placeholder_error: str | None = None

    @property
    def pixels_ok(self) -> bool:
        p = self.pixels_equal
        return bool(p) and len(p) == len(self.hf_grids) and all(p)

    @property
    def passed(self) -> bool:
        """The primary ``wrapped`` variant must have run ``generate`` and the placeholder check
        (not run = FAIL); diagnostic variants fail only on a negative result."""
        if self.variant == PRIMARY_VARIANT:
            checks_ok = self.generate_ids_equal is True and self.placeholders_ok is True
        else:
            checks_ok = self.generate_ids_equal is not False and self.placeholders_ok is not False
        return (self.error is None and self.ids_equal and self.grids_equal
                and self.runs_match_grid and self.pixels_ok and checks_ok
                and "bare_video_pad_positions" not in self.hf
                and "bare_video_pad_positions" not in self.vllm)

    def to_dict(self) -> dict:
        return {**asdict(self), "pixels_ok": self.pixels_ok, "passed": self.passed}


def compare(*, case_id: str, index: int, source: str, variant: str, hf_ids: Sequence[int],
            hf_grids: Sequence[Sequence[int]], vllm_ids: Sequence[int] | None,
            vllm_grids: Sequence[Sequence[int]] | None, sp: SpecialIds, merge_size: int,
            pixels_equal: Sequence[bool] | None = None, generate_ids_equal: bool | None = None,
            error: str | None = None, extras: dict | None = None,
            placeholders: Sequence[Sequence[int]] | None = None) -> Comparison:
    """Compare one request's vLLM model inputs with NVIDIA's HF call.

    PASS needs: no error, identical ids, identical grids, vLLM's frame runs as the grids imply,
    vLLM's placeholder ranges on those runs, pixels equal for every video, ``generate`` saw the
    same ids, and no stray bare ``video_pad`` in either sequence. For the ``wrapped`` variant the
    placeholder and ``generate`` checks must have run. ``placeholders`` (vLLM's
    ``mm_placeholders['video']`` as ``[offset, length]``) defaults to ``extras['placeholders']``."""
    hf_ids, hf_grids = list(hf_ids), [list(g) for g in hf_grids]
    pix = None if pixels_equal is None else [bool(p) for p in pixels_equal]
    extras = dict(extras or {})
    if placeholders is None:
        placeholders = extras.get("placeholders")
    if vllm_ids is None:
        return Comparison(case_id, index, source, variant, False, False, False,
                          summarize_ids(hf_ids, sp), {}, hf_grids, [], None,
                          error or "no vLLM output", pix, generate_ids_equal, extras)
    vllm_ids = list(vllm_ids)
    vgrids = [list(g) for g in (vllm_grids or [])]
    ph_error = (None if placeholders is None else
                placeholder_problem(vllm_ids, placeholders, vgrids, sp, merge_size))
    return Comparison(
        case_id=case_id, index=index, source=source, variant=variant,
        ids_equal=hf_ids == vllm_ids,
        grids_equal=hf_grids == vgrids,
        runs_match_grid=grids_match_runs(vllm_ids, vgrids, sp, merge_size),
        hf=summarize_ids(hf_ids, sp), vllm=summarize_ids(vllm_ids, sp),
        hf_grids=hf_grids, vllm_grids=vgrids,
        first_diff=first_diff(hf_ids, vllm_ids), error=error, pixels_equal=pix,
        generate_ids_equal=generate_ids_equal, extras=extras,
        placeholders_ok=None if placeholders is None else ph_error is None,
        placeholder_error=ph_error,
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


# The raw fields ``Comparison.passed`` reads. A recorded row without one of them cannot show
# that criterion holds: insufficient evidence, the criterion FAILS (review 6 B4).
EVIDENCE_FIELDS = ("variant", "error", "ids_equal", "grids_equal", "runs_match_grid",
                   "pixels_equal", "hf_grids", "generate_ids_equal", "placeholders_ok", "hf",
                   "vllm")


def rescore_row(row: Mapping[str, Any]) -> dict:
    """``row`` with ``passed`` / ``pixels_ok`` recomputed from its raw fields under the current
    criteria (``Comparison.passed``); a stored ``passed`` is never trusted. Missing fields are
    listed in ``insufficient`` and make the row FAIL."""
    missing = [f for f in EVIDENCE_FIELDS if f not in row]
    out = dict(row)
    out["insufficient"] = missing
    if missing:
        out["passed"] = False
        out["pixels_ok"] = False if "pixels_equal" in missing else out.get("pixels_ok")
        return out
    c = Comparison(case_id=str(row.get("case_id")), index=int(row.get("index") or 0),
                   source=str(row.get("source")), variant=row["variant"],
                   ids_equal=bool(row["ids_equal"]), grids_equal=bool(row["grids_equal"]),
                   runs_match_grid=bool(row["runs_match_grid"]), hf=dict(row["hf"] or {}),
                   vllm=dict(row["vllm"] or {}), hf_grids=list(row["hf_grids"] or []),
                   vllm_grids=list(row.get("vllm_grids") or []), error=row["error"],
                   pixels_equal=row["pixels_equal"],
                   generate_ids_equal=row["generate_ids_equal"],
                   placeholders_ok=row["placeholders_ok"])
    out["passed"], out["pixels_ok"] = c.passed, c.pixels_ok
    return out


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
    lines += ["", "| case | # | source | variant | ids | grid | runs | pixels | gen | ph | "
              "HF len | vLLM len | first diff | grids (T,H,W) | error |", "|" + "---|" * 15]
    yn = lambda b: "=" if b else "≠"  # noqa: E731
    for r in sorted(rows, key=lambda r: (r["case_id"], r["source"], r["index"], r["variant"])):
        pix = r.get("pixels_equal")
        gen = r.get("generate_ids_equal")
        ph = r.get("placeholders_ok")
        lines.append(
            f"| {r['case_id']} | {r['index']} | {r['source']} | {r['variant']} | "
            f"{yn(r['ids_equal'])} | {yn(r['grids_equal'])} | {yn(r['runs_match_grid'])} | "
            f"{'-' if pix is None else ''.join(yn(p) for p in pix)} | "
            f"{'-' if gen is None else yn(gen)} | {'-' if ph is None else yn(ph)} | "
            f"{r['hf'].get('length', '')} | {r['vllm'].get('length', '')} | "
            f"{'' if r['first_diff'] is None else r['first_diff']} | "
            f"{';'.join(','.join(map(str, g)) for g in r['hf_grids'])} | "
            f"{r['error'] or ''} |")
    return "\n".join(lines) + "\n"


REFERENCE_IMAGE = "img_hfref"  # where NVIDIA's HF reference runs in production (`exact`)


def reference_cross_check(primary: Sequence[dict], reference: Sequence[dict]) -> dict:
    """Review 3 C2: stage-1 records built in two images (e.g. ``img_vllm`` and ``img_hfref``)
    must give the same HF reference. Each record is ``{case_id, index, hf_ids_digest,
    hf_pixels_digest}`` (digests only); returns ``{passed, compared, mismatches, missing}``."""
    def key(r: dict) -> tuple[str, int]:
        return str(r["case_id"]), int(r["index"])

    a = {key(r): r for r in primary}
    b = {key(r): r for r in reference}
    fields = ("hf_ids_digest", "hf_pixels_digest")
    mismatches = sorted(f"{k[0]}#{k[1]}" for k in set(a) & set(b)
                        if any(a[k].get(f) != b[k].get(f) or a[k].get(f) is None
                               for f in fields))
    missing = sorted(f"{k[0]}#{k[1]}" for k in set(a) ^ set(b))
    compared = len(set(a) & set(b))
    return {"passed": compared > 0 and not mismatches and not missing, "compared": compared,
            "mismatches": mismatches, "missing": missing}


def gate_verdict(builds: Sequence[dict], results: dict[str, dict], *,
                 expected_parts: Sequence[tuple[str, str]], expected_cases: Sequence[str],
                 variant: str = PRIMARY_VARIANT, quota: int | None = None,
                 reference: dict | None = None, capacity: dict[str, dict] | None = None,
                 allow_unverified_reference: bool = False) -> dict:
    """The Gate C verdict over a whole run, not just the rows that exist.

    ``builds``: ``build_part`` results (``kind``, ``part``, ``n``, ``errors``, optionally
    ``quota``); ``results``: kind -> ``run_gate_c`` result (``rows``). PASS needs every expected
    (kind, part) built with no error and at least one request, every ``golden:<pool>`` part at
    its request quota (a shortfall fails, review 4 L3), a ``variant`` row for every built
    request, every expected case present, and every ``variant`` row passed. A pool whose builds
    all failed is a FAIL, never a silent ``n/a``.

    Quota (review 5 M-A): the required quota is ``max(quota, build's own quota)``: a build that
    reports a lower quota never lowers the caller's floor. ``capacity`` (from
    ``gatec_driver.golden_capacity``): per pool, how many requests the golden selection can
    give at most; a shortfall names that reason (add episodes to the golden selection), and a
    pool that cannot reach the quota at all is listed in ``unsatisfiable``.

    ``reference``: ``{"image": <where the HF reference ran>, "cross_check":
    reference_cross_check(...)}`` (review 3 C2). If the HF side ran anywhere but ``img_hfref``,
    a passing cross-check against ``img_hfref`` is required. Without ``reference`` the verdict
    FAILS (review 5 M-A) unless ``allow_unverified_reference`` (then a warning).

    Every row is re-scored from its raw fields (:func:`rescore_row`, review 6 B4), so a live run
    and ``--rescore`` of a recorded file apply exactly the same criteria; a ``variant`` row that
    lacks a criterion's data fails it (insufficient evidence)."""
    problems: list[str] = []
    warnings: list[str] = []
    results = {kind: {**(res or {}), "rows": [rescore_row(r) for r in
                                              (res or {}).get("rows", [])]}
               for kind, res in results.items()}
    for kind, res in sorted(results.items()):
        for r in res["rows"]:
            if r["insufficient"] and r.get("variant") == variant:
                problems.append(f"{kind} {r.get('case_id')}#{r.get('index')} {variant}: "
                                f"insufficient evidence (no {', '.join(r['insufficient'])})")
    by_part = {(b["kind"], b["part"]): b for b in builds}
    for kind, part in expected_parts:
        b = by_part.get((kind, part))
        if b is None:
            problems.append(f"{kind} {part}: not built")
        elif b.get("errors"):
            classes = sorted({e.get("error", "?") for e in b["errors"]})
            problems.append(f"{kind} {part}: {len(b['errors'])} build error(s) {classes}")
        elif not b.get("n"):
            problems.append(f"{kind} {part}: 0 requests")
        elif part.startswith("golden:"):
            wants = [q for q in (quota, b.get("quota")) if q is not None]
            want = max(wants) if wants else None
            if want is not None and b["n"] < want:
                cap = (capacity or {}).get(part.split(":", 1)[1])
                why = ""
                if cap is not None and cap["max_requests"] < want:
                    why = (f": the golden selection has {cap['episodes']} episode(s), at most "
                           f"{cap['max_requests']} requests; add episodes to reach the quota")
                elif cap is not None:
                    why = (f": the golden selection's {cap['episodes']} episode(s) yielded "
                           f"fewer calls than their {cap['max_requests']} maximum; add episodes")
                problems.append(f"{kind} {part}: {b['n']} of {want} requests "
                                f"(quota shortfall{why})")
            elif want is None:
                warnings.append(f"{kind} {part}: no quota given; shortfalls not checked")
    unsatisfiable = sorted(p for p, c in (capacity or {}).items()
                           if quota is not None and c["max_requests"] < quota
                           and any(part == f"golden:{p}" for _, part in expected_parts))
    for pool in unsatisfiable:
        warnings.append(f"golden:{pool}: quota {quota} is unsatisfiable from the golden "
                        f"selection ({capacity[pool]['episodes']} episode(s), at most "
                        f"{capacity[pool]['max_requests']} requests)")
    if reference is None:
        msg = "HF reference image not recorded (review 3 C2)"
        if allow_unverified_reference:
            warnings.append(msg)
        else:
            problems.append(msg)
    elif reference.get("image") != REFERENCE_IMAGE:
        check = reference.get("cross_check") or {}
        if not check.get("passed"):
            problems.append(f"HF reference ran in {reference.get('image')!r}, not "
                            f"{REFERENCE_IMAGE!r}, and no passing cross-check against "
                            f"{REFERENCE_IMAGE!r} (review 3 C2)")
    coverage: dict[str, dict[str, int]] = {}
    for kind in sorted({k for k, _ in expected_parts} | set(results)):
        built = sum(b.get("n", 0) for b in builds if b["kind"] == kind)
        rows = [r for r in (results.get(kind) or {}).get("rows", []) if r["variant"] == variant]
        coverage[kind] = {"built": built, "compared": len(rows),
                          "passed": sum(bool(r["passed"]) for r in rows)}
        if len(rows) != built:
            problems.append(f"{kind}: {built} requests built but {len(rows)} {variant} rows")
    all_rows = [r for res in results.values() for r in res.get("rows", [])]
    verdicts = case_verdicts(all_rows, variant)
    missing = sorted(set(expected_cases) - set(verdicts))
    if missing:
        problems.append(f"no {variant} rows for case(s) {missing}")
    failed = sorted(c for c, ok in verdicts.items() if not ok)
    return {"passed": not problems and not failed and bool(verdicts), "variant": variant,
            "failed_cases": failed, "problems": problems, "coverage": coverage,
            "cases": verdicts, "warnings": warnings, "quota": quota,
            "unsatisfiable": unsatisfiable}


def _git(args: Sequence[str], cwd: str | None) -> str | None:
    import subprocess

    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def results_provenance(*, cwd: str | None = None,
                       git: Callable[[Sequence[str], str | None], str | None] = _git,
                       fingerprint: Callable[[str], str] | None = None,
                       images: Sequence[str] = ("img_vllm",)) -> dict:
    """What ties a ``results.json`` to the code and image that produced it: the git commit
    (``git_dirty`` if the tree had uncommitted changes, so the commit alone does not describe
    it) and the ``image_fingerprint`` of every image the run used."""
    if fingerprint is None:
        from flora_deploy.modal_app.images import image_fingerprint as fingerprint
    status = git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return {
        "git_sha": git(["rev-parse", "HEAD"], cwd),
        "git_dirty": None if status is None else bool(status),
        "image_fingerprints": {name: fingerprint(name) for name in images},
    }
