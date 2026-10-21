"""``flora-rewarder`` Modal app: imports lazily, staging helpers work offline."""

from __future__ import annotations

import numpy as np


def test_rewarder_app_imports_without_network():
    from flora_deploy.modal_app import rewarder_app as ra

    assert ra.app.name == "flora-rewarder"
    assert ra.GPU == "H100!" and ra.TIMEOUT_S <= 45 * 60
    assert str(ra.SNAPSHOT) == (
        "/weights/snapshots/nvidia--Flora-Rewarder@7de68d22c4a9decec5ddb7f86ddcaee8f82d3367"
    )
    assert str(ra.SAMPLES_DIR) == "/golden/samples/wsh"
    ids = [s["id"] for s in ra.SAMPLE_EPISODES]
    assert len(ids) == len(set(ids)) == 3
    assert sorted(len(s["cameras"]) for s in ra.SAMPLE_EPISODES) == [1, 2, 3]
    for spec in ra.SAMPLE_EPISODES:
        assert len(spec["revision"]) == 40 and spec["license"]


def test_probe_stream_reports_cfr(tmp_path):
    import av

    from flora_deploy.modal_app.rewarder_app import probe_stream

    path = tmp_path / "v.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 32, 32, "yuv420p"
        for i in range(25):
            img = np.full((32, 32, 3), i * 9, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    info = probe_stream(path)
    assert info["frames"] == 25 and info["fps"] == 10.0 and info["cfr"]
    assert info["duration_s"] == 2.5 and (info["width"], info["height"]) == (32, 32)
