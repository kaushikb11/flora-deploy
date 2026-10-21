"""A tiny synthetic ``mp4_dir`` dataset for the local demo (``config/templates/local_demo.yaml``).

Generated on demand with ``media.synth`` (PyAV's bundled libx264): ``<path>/ep000.mp4`` ... with
a task text next to each (``ep000.txt``, the ``dataset_task`` instruction source). A few kB per
episode, deterministic for a given ``seed``.
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
