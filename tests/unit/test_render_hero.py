"""scripts/render_hero.py: the README figure, from synthetic segment and caption records.

The records carry private-looking values in every field the figure must not draw (uid,
dataset, episode id, video path, GPU, hashes); the outputs are checked for none of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import render_hero as rh

pytest.importorskip("matplotlib")

PRIVATE = ("acme-private", "episode_000042", "secret/videos/cam_high.mp4", "H100 80GB",
           "deadbeefcafe", "flora-deploy 0.0.1", "vllm-0.24.0-offline", "/Users/")
CAPTIONS = ("The right hand picks up the blue mug from the table.",
            "The right hand carries the mug to the sink and tilts it to pour out the water.",
            "The left hand turns the tap on.",
            "The right hand holds the mug under the running water and rinses it.")
TIMES = ((0.0, 2.4), (2.4, 6.1), (6.1, 7.0), (7.0, 11.8))


def _record(stage: str, uid: str, times, captions=None, offset: float = 30.0) -> dict:
    segs = [{"caption": None if captions is None else captions[i], "degradation": {},
             "end_s": e, "event_id": f"event_{i:04d}", "fine_event_id": None,
             "nv_metadata": {"note": "deadbeefcafe"}, "segmentation_provenance": None,
             "start_s": s, "status": "complete"} for i, (s, e) in enumerate(times)]
    return {
        "audit_ref": None, "certified": False, "dataset": "acme-private/teleop",
        "dense": None, "episode_id": uid.split("--")[1],
        "provenance": {"certificate": None, "config_hash": "deadbeefcafe", "embodiment": "human",
                       "engine": "vllm-0.24.0-offline", "engine_hash": "deadbeefcafe",
                       "framework": "flora-deploy 0.0.1", "gpu": "NVIDIA H100 80GB HBM3",
                       "logic_version": "0.0.1", "models": {"captioner": "deadbeefcafe"},
                       "preset": "balanced",
                       "source": {"main": {"range_s": [offset, offset + 12.0],
                                           "uri": "secret/videos/cam_high.mp4"}}},
        "rubric": None, "schema": "flora_labels_v1", "segments": segs, "stage": stage,
        "terminal": None, "uid": uid, "uncertified_reason": "uncertified_stack",
    }


def synthetic(tmp: Path) -> tuple[Path, Path]:
    one = "acme-private-teleop--episode_000001"  # only 2 subtasks: skipped by default
    hero = "acme-private-teleop--episode_000042"
    caps = [_record("caption", one, TIMES[:2], CAPTIONS[:2]),
            _record("caption", hero, TIMES, CAPTIONS)]
    segs = [_record("segment", one, TIMES[:2]), _record("segment", hero, TIMES)]
    paths = (tmp / "caption.jsonl", tmp / "segment.jsonl")
    for path, recs in zip(paths, (caps, segs), strict=True):
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    return paths


def test_episode_choice_and_times(tmp_path):
    caps, segs = synthetic(tmp_path)
    eps = rh.episodes(rh.load_jsonl(caps), rh.load_jsonl(segs))
    assert [len(e.subtasks) for e in eps] == [2, 4]
    ep = rh.choose(eps)
    assert ep.uid.endswith("episode_000042")
    assert [(t.start_s, t.end_s) for t in ep.subtasks] == list(TIMES)
    assert [t.caption for t in ep.subtasks] == list(CAPTIONS)
    assert ep.duration_s == 12.0 and ep.video_start_s == 30.0
    assert rh.choose(eps, eps[0].uid) is eps[0]
    with pytest.raises(SystemExit):
        rh.choose(eps, "nope")


def test_segment_times_win_and_incomplete_segments_drop(tmp_path):
    cap = _record("caption", "d--e", TIMES, CAPTIONS)
    cap["segments"][2]["status"] = "unsupported_target_frames"
    seg = _record("segment", "d--e", [(0.0, 2.5), *TIMES[1:]])
    (ep,) = rh.episodes([cap], [seg])
    assert ep.subtasks[0].end_s == 2.5
    assert len(ep.subtasks) == 3


def test_wrong_stage_is_refused():
    with pytest.raises(SystemExit, match="caption"):
        rh.episodes([_record("segment", "d--e", TIMES)])


def _assert_clean(paths: list[Path]) -> None:
    for path in paths:
        data = path.read_bytes()
        for s in PRIVATE:
            assert s.encode() not in data, f"{path.name} contains {s!r}"


def test_render_png_and_svg(tmp_path):
    caps, segs = synthetic(tmp_path)
    out = tmp_path / "img" / "hero"
    assert rh.main(["--captions", str(caps), "--segments", str(segs), "--out", str(out),
                    "--title", "One episode, labelled"]) == 0
    png, svg = out.with_suffix(".png"), out.with_suffix(".svg")
    from PIL import Image

    with Image.open(png) as im:
        assert im.format == "PNG" and im.width == rh.WIDTH_IN * rh.DPI
        assert "Software" not in im.info
    text = svg.read_text()
    for c in ("One episode, labelled", "4 subtasks", "turns the", "rinses it."):
        assert c in text
    assert "<dc:date>" not in text
    _assert_clean([png, svg])
    first = svg.read_bytes()
    rh.main(["--captions", str(caps), "--segments", str(segs), "--out", str(out),
             "--title", "One episode, labelled"])
    assert svg.read_bytes() == first  # deterministic


def test_render_with_keyframes(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    video = tmp_path / "secret.mp4"
    with av.open(str(video), "w") as c:
        st = c.add_stream("mpeg4", rate=10)
        st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
        for i in range(10 * 44):  # 44 s: the episode starts at 30 s in this file
            img = np.full((48, 64, 3), (i * 5) % 255, dtype=np.uint8)
            for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    frames = rh.keyframes(video, [31.2, 40.0], height_px=48)
    assert [f.shape for f in frames] == [(48, 64, 3)] * 2
    assert abs(int(frames[1].mean()) - int(frames[0].mean())) > 10  # different moments

    caps, segs = synthetic(tmp_path)
    out = tmp_path / "hero"
    assert rh.main(["--captions", str(caps), "--segments", str(segs), "--video", str(video),
                    "--out", str(out)]) == 0
    _assert_clean([out.with_suffix(".png"), out.with_suffix(".svg")])
    assert "<image" in out.with_suffix(".svg").read_text()
