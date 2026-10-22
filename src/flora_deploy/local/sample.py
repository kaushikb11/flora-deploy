"""Tiny synthetic datasets for the local demos (``config/templates/local_demo*.yaml``).

Generated on demand with ``media.synth`` (PyAV's bundled libx264), deterministic for a given
``seed``, a few kB per episode:

* :func:`sample_dataset`: ``mp4_dir``, ``<path>/ep000.mp4`` ... with a task text next to each
  (``ep000.txt``, the ``dataset_task`` instruction source);
* :func:`sample_lerobot`: a LeRobot v3.0 dataset (one camera, ``main``) with per-frame data
  parquet, so ``flora export`` has something to write labels into.
"""

from __future__ import annotations

from pathlib import Path

TASKS = ("pick up the red block", "fold the towel", "open the drawer", "stack the cups",
         "pour water into the bowl", "close the lid")


def sample_dataset(path: str | Path, *, episodes: int = 6, seed: int = 0, fps: int = 30,
                   min_s: float = 3.0, max_s: float = 6.0, overwrite: bool = False) -> Path:
    """Write ``episodes`` single-camera MP4 episodes (``min_s``..``max_s`` long, 64x48) to
    ``path`` and return it. Existing episode files are kept unless ``overwrite``."""
    from flora_deploy.media import synth

    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    span = max(max_s - min_s, 0.0)
    for i in range(episodes):
        video = root / f"ep{i:03d}.mp4"
        task = root / f"ep{i:03d}.txt"
        if overwrite or not video.exists():
            seconds = min_s + (span * ((i * 7 + seed) % 5) / 4 if span else 0.0)
            synth.encode(video, synth.Spec(frames=max(int(round(seconds * fps)), 2), rate=fps,
                                           seed=seed * 1000 + i, gop=fps))
        if overwrite or not task.exists():
            task.write_text(TASKS[(i + seed) % len(TASKS)] + "\n", encoding="utf-8")
    return root


def sample_lerobot(path: str | Path, *, episodes: int = 6, seed: int = 0, fps: int = 30) -> Path:
    """Write a LeRobot v3.0 dataset of ``episodes`` 3-6 s single-camera episodes to ``path``:
    ``meta/`` and ``videos/`` (``media.synth.lerobot_v3_tree``) plus the per-frame
    ``data/chunk-000/file-000.parquet`` that real LeRobot datasets carry."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from flora_deploy.media import synth

    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    root = Path(path)
    lengths = [int(round((3.0 + 3.0 * ((i * 7 + seed) % 5) / 4) * fps)) for i in range(episodes)]
    specs = [{"main": synth.Spec(frames=n, rate=fps, seed=seed * 1000 + i, gop=fps)}
             for i, n in enumerate(lengths)]
    synth.lerobot_v3_tree(root, specs, TASKS, fps=fps)
    shutil.rmtree(root / ".scratch", ignore_errors=True)
    frames: dict[str, list] = {"timestamp": [], "frame_index": [], "episode_index": [],
                               "index": [], "task_index": []}
    for e, n in enumerate(lengths):
        start = len(frames["index"])
        frames["timestamp"] += [i / fps for i in range(n)]
        frames["frame_index"] += list(range(n))
        frames["episode_index"] += [e] * n
        frames["index"] += list(range(start, start + n))
        frames["task_index"] += [e % len(TASKS)] * n
    data = root / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({**frames, "timestamp": pa.array(frames["timestamp"], pa.float32())}),
                   data)
    meta = root / "meta/episodes/chunk-000/file-000.parquet"
    t = pq.read_table(meta)
    for col in ("data/chunk_index", "data/file_index"):
        t = t.append_column(col, pa.array([0] * episodes, pa.int64()))
    pq.write_table(t, meta)
    return root
