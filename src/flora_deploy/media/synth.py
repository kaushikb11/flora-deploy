"""Tiny synthetic videos and dataset trees, generated on demand (tests and self-checks).

Nothing here is committed media: every file is produced at call time with PyAV's bundled
encoders (libx264, libsvtav1/libaom-av1, mpeg4). Sizes are a few kB.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class Spec:
    frames: int
    rate: int = 30
    codec: str = "libx264"
    size: tuple[int, int] = (64, 48)
    seed: int = 0
    # Per-frame pts in units of ``time_base`` (VFR); default i * (1/rate).
    pts: Sequence[int] | None = None
    time_base: Fraction | None = None
    gop: int | None = None  # keyframe interval (None = encoder default)
    bframes: int | None = None  # libx264 only
    colorspace: int | None = None  # AVColorSpace, e.g. 1 = BT.709 (non-canonical branch)
    color_range: int | None = None  # AVColorRange: 1 = MPEG/limited, 2 = JPEG/full
    x264_params: str | None = None  # extra libx264 params, e.g. "intra-refresh=1:keyint=30"
    track_timescale: int | None = None  # movenc video_track_timescale (e.g. 30: below 10000)
    codec_options: Mapping[str, str] | None = None  # extra encoder options (e.g. libx265)


def _frame_image(spec: Spec, i: int) -> np.ndarray:
    w, h = spec.size
    rng = np.random.default_rng(spec.seed * 100_003 + i)
    x = np.arange(w, dtype=np.int32)[None, :]
    y = np.arange(h, dtype=np.int32)[:, None]
    base = (x * 3 + y * 2 + i * 7 + spec.seed * 31) % 256
    img = np.stack([base, (base + 85) % 256, (255 - base)], axis=-1)
    noise = rng.integers(-24, 24, size=img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def encode(path: Path, spec: Spec) -> Path:
    """Encode ``spec`` into an MP4 (or MOV, by suffix) at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opts: dict[str, str] = {}
    if spec.codec == "libx264":
        if spec.bframes is not None:
            opts["bf"] = str(spec.bframes)
        opts["preset"] = "veryfast"
        params = []
        if spec.gop is not None:
            params.append("scenecut=0")  # keyframes only every `gop` frames
        if spec.x264_params:
            params.append(spec.x264_params)
        if params:
            opts["x264-params"] = ":".join(params)
    if spec.codec == "libsvtav1":
        opts["preset"] = "12"
    if spec.codec == "libaom-av1":
        opts["cpu-used"] = "8"
    opts.update(spec.codec_options or {})
    fmt_opts = {}
    if spec.track_timescale is not None:
        fmt_opts["video_track_timescale"] = str(spec.track_timescale)
    with av.open(str(path), "w", options=fmt_opts) as out:
        stream = out.add_stream(spec.codec, rate=spec.rate, options=opts)
        stream.width, stream.height = spec.size
        stream.pix_fmt = "yuv420p"
        if spec.gop is not None:
            stream.codec_context.gop_size = spec.gop
        tb = spec.time_base or Fraction(1, spec.rate)
        stream.codec_context.time_base = tb
        stream.time_base = tb
        if spec.colorspace is not None:
            stream.codec_context.colorspace = spec.colorspace
        if spec.color_range is not None:
            stream.codec_context.color_range = spec.color_range
        pts = list(spec.pts) if spec.pts is not None else None
        for i in range(spec.frames):
            frame = av.VideoFrame.from_ndarray(_frame_image(spec, i), format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = pts[i] if pts is not None else round(i / (tb * spec.rate))
            frame.time_base = tb
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return path


def concat_like_lerobot(inputs: Sequence[Path], output: Path) -> Path:
    """Concatenate per-episode MP4s the way LeRobot v3 does: ffconcat demuxer + stream copy.

    Same approach as ``lerobot.datasets.video_utils.concatenate_video_files`` (huggingface/lerobot
    @afbf3f14, Apache-2.0): concat demuxer with ``safe=0``, ``add_stream_from_template(...,
    opaque=True)``, output ``time_base`` = input ``time_base``, packets without dts skipped,
    ``movflags=faststart``.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    listing = output.with_suffix(".ffconcat")
    listing.write_text(
        "ffconcat version 1.0\n" + "".join(f"file '{Path(p).resolve()}'\n" for p in inputs)
    )
    try:
        with av.open(str(listing), "r", format="concat", options={"safe": "0"}) as ic:
            with av.open(str(output), "w", options={"movflags": "faststart"}) as oc:
                ist = ic.streams.video[0]
                ost = oc.add_stream_from_template(template=ist, opaque=True)
                ost.time_base = ist.time_base
                for pkt in ic.demux(ist):
                    if pkt.dts is None:
                        continue
                    pkt.stream = ost
                    oc.mux(pkt)
    finally:
        listing.unlink(missing_ok=True)
    return output


def video_duration_s(path: Path) -> float:
    """LeRobot's ``get_video_duration_in_s``: ``stream.duration * time_base``."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.duration is not None:
            return float(s.duration * s.time_base)
        return float(c.duration / av.time_base)


def remux_with_last_duration(src: Path, dst: Path, last_duration: int) -> Path:
    """Copy ``src`` with the last packet's duration replaced (non-standard last ``stts``)."""
    with av.open(str(src)) as ic:
        ist = ic.streams.video[0]
        pkts = [p for p in ic.demux(ist) if p.size and p.dts is not None]
        with av.open(str(dst), "w") as oc:
            ost = oc.add_stream_from_template(template=ist, opaque=True)
            ost.time_base = ist.time_base
            for k, p in enumerate(pkts):
                if k == len(pkts) - 1:
                    p.duration = last_duration
                p.stream = ost
                oc.mux(p)
    return dst


def _remux(src: Path, dst: Path, edit) -> Path:
    with av.open(str(src)) as ic:
        ist = ic.streams.video[0]
        pkts = [p for p in ic.demux(ist) if p.size and p.dts is not None]
        with av.open(str(dst), "w") as oc:
            ost = oc.add_stream_from_template(template=ist, opaque=True)
            ost.time_base = ist.time_base
            for p in edit(pkts):
                p.stream = ost
                oc.mux(p)
    return Path(dst)


def remux_trimmed(src: Path, dst: Path, from_keyframe: int, trim_ticks: int) -> Path:
    """What ``ffmpeg -ss T -i src -c copy dst`` produces: packets from the ``from_keyframe``-th
    keyframe on, shifted by ``-trim_ticks``. Pre-roll packets get negative pts, movenc writes an
    edit list, and the mov demuxer flags them ``AV_PKT_FLAG_DISCARD``."""

    def edit(pkts):
        start = [i for i, p in enumerate(pkts) if p.is_keyframe][from_keyframe]
        for p in pkts[start:]:
            p.pts -= trim_ticks
            p.dts -= trim_ticks
            yield p

    return _remux(src, dst, edit)


def remux_drop_head(src: Path, dst: Path, n: int) -> Path:
    """Copy ``src`` without its first ``n`` packets: a file that starts mid-GOP (segment split)."""
    return _remux(src, dst, lambda pkts: pkts[n:])


# ---------------------------------------------------------------------------------------------
# LeRobot trees


def lerobot_v21_tree(
    root: Path,
    episodes: Sequence[Mapping[str, Spec]],
    tasks: Sequence[str],
    *,
    fps: int = 30,
    chunks_size: int = 1000,
) -> Path:
    """``codebase_version`` v2.1 layout: one MP4 per episode and camera.

    ``meta/info.json`` (``video_path`` template), ``meta/episodes.jsonl`` and
    ``meta/tasks.jsonl``, as documented in LeRobot's
    ``scripts/convert_dataset_v21_to_v30.py`` header.
    """
    root = Path(root)
    keys = list(episodes[0])
    template = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "chunks_size": chunks_size,
        "total_episodes": len(episodes),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": template,
        "features": {k: {"dtype": "video", "shape": [48, 64, 3]} for k in keys},
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    lines, task_lines = [], []
    for t, task in enumerate(tasks):
        task_lines.append(json.dumps({"task_index": t, "task": task}))
    for e, cams in enumerate(episodes):
        for key, spec in cams.items():
            rel = template.format(episode_chunk=e // chunks_size, video_key=key, episode_index=e)
            encode(root / rel, spec)
        n = next(iter(cams.values())).frames
        lines.append(
            json.dumps({"episode_index": e, "tasks": [tasks[e % len(tasks)]], "length": n})
        )
    (root / "meta/episodes.jsonl").write_text("\n".join(lines) + "\n")
    (root / "meta/tasks.jsonl").write_text("\n".join(task_lines) + "\n")
    return root


def lerobot_v3_tree(
    root: Path,
    episodes: Sequence[Mapping[str, Spec]],
    tasks: Sequence[str],
    *,
    fps: int = 30,
    episodes_per_file: int = 1000,
    scratch: Path | None = None,
) -> Path:
    """``codebase_version`` v3.0 layout: per-camera concatenated MP4s + episodes parquet.

    Mirrors ``convert_dataset_v21_to_v30.convert_videos_of_camera``: episode files are encoded
    separately, concatenated by stream copy, and ``videos/<key>/from_timestamp`` /
    ``to_timestamp`` are float sums of the per-episode ``get_video_duration_in_s``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    scratch = Path(scratch) if scratch else root / ".scratch"
    keys = list(episodes[0])
    video_template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "chunks_size": 1000,
        "total_episodes": len(episodes),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": video_template,
        "features": {k: {"dtype": "video", "shape": [48, 64, 3]} for k in keys},
    }
    (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    cols: dict[str, list] = {"episode_index": [], "tasks": [], "length": []}
    for key in keys:
        for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            cols[f"videos/{key}/{c}"] = []
    for e, cams in enumerate(episodes):
        cols["episode_index"].append(e)
        cols["tasks"].append([tasks[e % len(tasks)]])
        cols["length"].append(next(iter(cams.values())).frames)
    for key in keys:
        groups = [
            list(range(i, min(i + episodes_per_file, len(episodes))))
            for i in range(0, len(episodes), episodes_per_file)
        ]
        for file_index, group in enumerate(groups):
            parts, t = [], 0.0
            for e in group:
                part = encode(scratch / key / f"ep{e:06d}.mp4", episodes[e][key])
                d = video_duration_s(part)
                cols[f"videos/{key}/chunk_index"].append(0)
                cols[f"videos/{key}/file_index"].append(file_index)
                cols[f"videos/{key}/from_timestamp"].append(t)
                cols[f"videos/{key}/to_timestamp"].append(t + d)
                t += d
                parts.append(part)
            out = root / video_template.format(video_key=key, chunk_index=0, file_index=file_index)
            concat_like_lerobot(parts, out)
    pq.write_table(pa.Table.from_pydict(cols), root / "meta/episodes/chunk-000/file-000.parquet")
    task_table = pa.Table.from_pydict({"task_index": list(range(len(tasks))), "task": list(tasks)})
    pq.write_table(task_table, root / "meta/tasks.parquet")
    return root
