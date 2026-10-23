"""Render the README hero figure: one episode's subtask timeline with its captions.

    uv run python scripts/render_hero.py --captions caption.jsonl [--segments segment.jsonl] \\
        [--uid UID] [--video episode.mp4] [--title TEXT] --out docs/img/hero

The inputs are what ``flora results <stage> --format jsonl`` writes (``docs/output-format.md``).
Writes ``<out>.png`` and ``<out>.svg``: a timeline bar of the episode's subtasks, numbered and
colored, and below it one card per subtask with its time range, caption and (with ``--video``)
the frame at the subtask's midpoint.

Only the subtask times, the captions and the optional frames are drawn. Nothing else from the
records reaches the figure: no uid, dataset, episode id, file path, provenance, GPU or hash,
and the PNG/SVG carry no creator, date or path metadata. A ``--title`` is drawn as given.

Without ``--uid`` the first episode (by uid) with 3-6 captioned subtasks is drawn, else the
first with any. ``--video`` is the episode's video file; ``--video-start`` is where the
episode starts in it, default the record's ``provenance.source.<camera>.range_s[0]`` (LeRobot
v3 packs several episodes per file). Needs matplotlib (the dev group); frames are decoded with
PyAV, a core dependency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

# Categorical slots in fixed order (adjacent pairs validated for colour-vision deficiency).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
           "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
RULE = "#e4e3df"

WIDTH_IN = 12.0
MARGIN_IN = 0.45
GUTTER_IN = 0.22
MAX_COLS = 4
CAPTION_PT = 10.5
DPI = 200


@dataclass(frozen=True)
class Subtask:
    start_s: float
    end_s: float
    caption: str


@dataclass(frozen=True)
class Episode:
    uid: str  # used to pick the episode and the video offset; never drawn
    duration_s: float
    subtasks: tuple[Subtask, ...]
    video_start_s: float | None  # where the episode starts in its source video, if recorded


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: not JSON ({e.msg})") from None
    return out


def clean_caption(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _by_uid(records: list[dict], stage: str) -> dict[str, dict]:
    out = {}
    for r in records:
        if r.get("stage", stage) != stage:
            raise SystemExit(f"expected {stage} records, found a {r.get('stage')!r} record")
        out[r["uid"]] = r
    return out


def episodes(captions: list[dict], segments: list[dict] | None = None,
             camera: str = "main") -> list[Episode]:
    """One ``Episode`` per captioned uid, sorted by uid. Times come from the segment record
    when given (matched on ``event_id``), else from the caption record's own segments."""
    caps = _by_uid(captions, "caption")
    segs = _by_uid(segments or [], "segment")
    out = []
    for uid in sorted(caps):
        rec = caps[uid]
        times = {s["event_id"]: (s["start_s"], s["end_s"])
                 for s in segs.get(uid, {}).get("segments", [])}
        subtasks = []
        for s in rec.get("segments", []):
            if s.get("status", "complete") != "complete":
                continue
            start, end = times.get(s.get("event_id"), (s["start_s"], s["end_s"]))
            if end > start:
                subtasks.append(Subtask(float(start), float(end), clean_caption(s.get("caption"))))
        if not subtasks:
            continue
        subtasks.sort(key=lambda t: t.start_s)
        source = (rec.get("provenance") or {}).get("source") or {}
        cam = source.get(camera) or next(iter(source.values()), None) or {}
        rng = cam.get("range_s")
        duration = max(t.end_s for t in subtasks)
        start = None
        if rng and len(rng) == 2:
            start = float(rng[0])
            duration = max(duration, float(rng[1]) - float(rng[0]))
        out.append(Episode(uid, duration, tuple(subtasks), start))
    return out


def choose(eps: list[Episode], uid: str | None = None) -> Episode:
    if not eps:
        raise SystemExit("no captioned episode with a complete subtask in the records")
    if uid is not None:
        for e in eps:
            if e.uid == uid:
                return e
        raise SystemExit("--uid: no captioned episode with that uid")
    good = [e for e in eps if 3 <= len(e.subtasks) <= 6 and all(t.caption for t in e.subtasks)]
    return (good or eps)[0]


def keyframes(video: Path, times_s: list[float], height_px: int = 360) -> list:
    """RGB arrays of the first frame at or after each time (seconds into the file)."""
    import av

    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        base = stream.time_base
        for t in times_s:
            container.seek(int(max(t, 0) / base), stream=stream, backward=True, any_frame=False)
            picked = None
            for frame in container.decode(stream):
                picked = frame
                if frame.pts is not None and frame.pts * base >= t:
                    break
            if picked is None:
                raise SystemExit(f"--video: no frame at {t:.2f} s")
            w = max(2, round(picked.width * height_px / picked.height))
            frames.append(picked.to_ndarray(format="rgb24", width=w, height=height_px))
    return frames


def _wrap(text: str, width_in: float, pt: float) -> list[str]:
    chars = max(12, int(width_in * 72 / (pt * 0.49)))
    return textwrap.wrap(text, chars) or [""]


def _fmt_s(s: float) -> str:
    return f"{s:.1f}"


def render(ep: Episode, out: Path, title: str | None = None,
           frames: list | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    matplotlib.rcParams.update({
        "svg.hashsalt": "flora-hero",  # stable element ids, so reruns diff cleanly
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    })
    n = len(ep.subtasks)
    rows = -(-n // MAX_COLS)
    cols = -(-n // rows)  # balance the rows: 5 subtasks are 3 + 2, 6 are 3 + 3
    inner = WIDTH_IN - 2 * MARGIN_IN
    card_w = (inner - GUTTER_IN * (cols - 1)) / cols
    line_h = CAPTION_PT * 1.45 / 72
    wrapped = [_wrap(t.caption or "(no caption)", card_w - 0.3, CAPTION_PT) for t in ep.subtasks]
    thumb_h = 0.0
    if frames:
        aspect = max(f.shape[1] / f.shape[0] for f in frames)
        thumb_h = min((card_w - 0.3) / aspect, 1.9)

    # Vertical layout in inches from the top.
    y = MARGIN_IN - 0.1
    title_y = None
    if title:
        title_y = y
        y += 0.42
    meta_y = y + 0.08
    y += 0.38
    bar_y, bar_h = y, 0.42
    y += bar_h + 0.42  # bar, then its time axis
    row_tops = []
    for r in range(rows):
        lines = max(len(w) for w in wrapped[r * cols:(r + 1) * cols])
        h = 0.56 + (thumb_h + 0.14 if frames else 0) + lines * line_h + 0.1
        row_tops.append((y, h))
        y += h + GUTTER_IN
    height = y - GUTTER_IN + MARGIN_IN

    fig = plt.figure(figsize=(WIDTH_IN, height), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    if title_y is not None:
        ax.text(MARGIN_IN, title_y, title, fontsize=17, fontweight="bold", color=INK, va="top")
    noun = "subtask" if n == 1 else "subtasks"
    ax.text(MARGIN_IN, meta_y, f"{n} {noun} over {_fmt_s(ep.duration_s)} s, segmented and "
            "captioned by NVIDIA Flora", fontsize=10.5, color=INK_2, va="top")

    def x_of(s: float) -> float:
        return MARGIN_IN + inner * s / ep.duration_s

    ax.add_patch(Rectangle((MARGIN_IN, bar_y), inner, bar_h, color=RULE, lw=0))
    for i, t in enumerate(ep.subtasks):
        x0, x1 = x_of(t.start_s), x_of(t.end_s)
        ax.add_patch(FancyBboxPatch((x0, bar_y), x1 - x0, bar_h,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=PALETTE[i % len(PALETTE)], ec=SURFACE, lw=2.2))
        if x1 - x0 >= 0.28:
            ax.text((x0 + x1) / 2, bar_y + bar_h / 2, str(i + 1), color="white", fontsize=11,
                    fontweight="bold", ha="center", va="center")
    step = _tick_step(ep.duration_s)
    s = 0.0
    while s <= ep.duration_s + 1e-9:
        ax.plot([x_of(s)] * 2, [bar_y + bar_h + 0.04, bar_y + bar_h + 0.1], color=INK_3, lw=0.8)
        ax.text(x_of(s), bar_y + bar_h + 0.14, f"{s:g} s", fontsize=8.5, color=INK_3,
                ha="center", va="top")
        s += step

    for i, (t, lines) in enumerate(zip(ep.subtasks, wrapped, strict=True)):
        r, c = divmod(i, cols)
        top, h = row_tops[r]
        x = MARGIN_IN + c * (card_w + GUTTER_IN)
        color = PALETTE[i % len(PALETTE)]
        ax.add_patch(FancyBboxPatch((x, top), card_w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec=RULE, lw=1))
        ax.add_patch(Rectangle((x + 0.15, top + 0.2), 0.2, 0.2, color=color, lw=0))
        ax.text(x + 0.25, top + 0.3, str(i + 1), color="white", fontsize=8.5, fontweight="bold",
                ha="center", va="center")
        ax.text(x + 0.45, top + 0.3, f"{_fmt_s(t.start_s)}–{_fmt_s(t.end_s)} s", fontsize=9.5,
                color=INK_2, va="center")
        cy = top + 0.56
        if frames:
            f = frames[i]
            w = thumb_h * f.shape[1] / f.shape[0]
            ax.imshow(f, extent=(x + 0.15, x + 0.15 + w, cy + thumb_h, cy),
                      interpolation="lanczos", zorder=3)  # above the card patch
            cy += thumb_h + 0.14
        for k, line in enumerate(lines):  # one text per line: even spacing whatever the glyphs
            ax.text(x + 0.15, cy + k * line_h, line, fontsize=CAPTION_PT, color=INK, va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_suffix(".png"), out.with_suffix(".svg")]
    fig.savefig(paths[0], facecolor=SURFACE, metadata={"Software": None})
    fig.savefig(paths[1], facecolor=SURFACE,
                metadata={"Creator": None, "Date": None, "Format": None, "Type": None})
    plt.close(fig)
    return paths


def _tick_step(duration: float) -> float:
    for step in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
        if duration / step <= 12:
            return float(step)
    return 600.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--captions", type=Path, required=True, help="caption.jsonl")
    p.add_argument("--segments", type=Path, help="segment.jsonl (optional; times from here)")
    p.add_argument("--uid", help="episode to draw (default: first with 3-6 captioned subtasks)")
    p.add_argument("--video", type=Path, help="the episode's video file, for keyframes")
    p.add_argument("--video-start", type=float, help="episode start in --video, seconds")
    p.add_argument("--camera", default="main", help="camera role for the default video start")
    p.add_argument("--title", help="drawn as given (default: no title)")
    p.add_argument("--out", type=Path, default=Path("docs/img/hero"),
                   help="output path without suffix; writes .png and .svg")
    a = p.parse_args(argv)
    caps = load_jsonl(a.captions)
    segs = load_jsonl(a.segments) if a.segments else None
    ep = choose(episodes(caps, segs, a.camera), a.uid)
    frames = None
    if a.video:
        start = a.video_start if a.video_start is not None else (ep.video_start_s or 0.0)
        frames = keyframes(a.video, [start + (t.start_s + t.end_s) / 2 for t in ep.subtasks])
    for path in render(ep, a.out.with_suffix(""), a.title, frames):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
