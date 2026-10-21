"""Unit tests for the Gate C request driver's plumbing (no NVIDIA code, no media)."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from flora_deploy.validate import gatec_driver as D


def fake_nv(n_calls=5):
    pipeline = SimpleNamespace(generate=None)
    seen = {}

    def segment(videos, pool, session, instruction, progress):
        seen["videos"] = videos
        for _ in range(n_calls):
            pipeline.generate(None, None, None, 1024)

    pipeline.segment = segment
    media = SimpleNamespace(decode_selected="real", Video=lambda **kw: SimpleNamespace(**kw))
    return SimpleNamespace(pipeline=pipeline, media=media), seen


def test_run_pipeline_stops_after_max_calls_and_restores_patches():
    nv, _ = fake_nv(5)
    assert D.run_pipeline(nv, "segment", "yam", {}, [], "", {}, max_calls=None) == 5
    assert D.run_pipeline(nv, "segment", "yam", {}, [], "", {}, decode=len, max_calls=2) == 2
    assert nv.pipeline.generate is None and nv.media.decode_selected == "real"
    with pytest.raises(ValueError, match="unknown pipeline path"):
        D.run_pipeline(nv, "nope", "yam", {}, [], "", {})


def test_make_videos_keeps_non_integer_rates_exact():
    nv, _ = fake_nv()
    src = D.FrameSource("ntsc", (64, 48), Fraction(30000, 1001), 2.0, lambda r, i: None)
    (v,) = D.make_videos(nv, ("main",), src).values()
    assert v.rate == Fraction(30000, 1001)
    assert len(v.pts_us) == 59  # int(2.0 * 29.97...)
    assert v.pts_us[1] == Fraction(1_001_000_000, 30000)


def test_golden_episodes_and_segments():
    manifest = {"episodes": [
        dict(id="human/a/x", pool="human", path="golden/v1/human/a/x", status="staged",
             oracle_eligible=True, task="t"),
        dict(id="human/a/y", pool="human", path="golden/v1/human/a/y", status="source_error",
             oracle_eligible=False),
        dict(id="reward/r/z", pool="reward", path="golden/v1/reward/r/z", status="staged",
             oracle_eligible=True),
        dict(id="tianji/t/e", pool="tianji", path="golden/v1/tianji/t/e", status="staged",
             oracle_eligible=True, task=None),
    ]}
    eps = D.golden_episodes("/vol", manifest)
    assert set(eps) == set(D.GOLDEN_POOLS)
    assert [e.id for e in eps["human"]] == ["human/a/x"]
    assert eps["human"][0].dir == Path("/vol/golden/v1/human/a/x")
    assert eps["tianji"][0].task == "" and eps["yam"] == []
    segs = D.golden_segments(10.0)
    assert [(s["start_s"], s["end_s"]) for s in segs] == [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0)]
    assert all(s["end_s"] <= 0.98 * 2.0 for s in D.golden_segments(2.0))


def test_memo_decode_decodes_once_and_returns_copies():
    calls = []

    class Im:
        def __init__(self, i):
            self.i = i

        def copy(self):
            return Im(self.i)

    def real(video, wanted, canonical=False):
        calls.append((video.path, sorted(wanted), canonical))
        return {i: Im(i) for i in wanted}

    memo = D.MemoDecode(real)
    v = SimpleNamespace(path=Path("main.mp4"))
    a = memo(v, {1, 2})
    b = memo(v, [2, 1])
    memo(v, {1, 2}, canonical=True)
    assert len(calls) == 2 and a[1] is not b[1] and a[1].i == b[1].i == 1


def test_build_golden_spreads_quota_and_records_errors(monkeypatch):
    eps = {"yam": [D.GoldenEpisode(f"yam/d/{i}", "yam", Path(f"/v/{i}"), "task")
                   for i in range(4)],
           "tianji": [D.GoldenEpisode("tianji/d/0", "tianji", Path("/v/t"), "")],
           "dex": []}
    asked = []

    def fake_build(nv, procs, kind, ep, max_requests, use_instruction, videos=None):
        asked.append((ep.id, max_requests, use_instruction))
        if ep.id == "yam/d/1":
            raise OSError("secret path text")
        return [ep.id] * max_requests

    monkeypatch.setattr(D, "build_golden_requests", fake_build)
    monkeypatch.setattr(D, "compact", lambda b: b)
    records, errors = D.build_golden(None, {}, "tempseg", eps, quota=5)
    assert asked == [("yam/d/0", 2, True), ("yam/d/1", 2, False), ("yam/d/2", 2, True),
                     ("yam/d/3", 1, False), ("tianji/d/0", 3, True)]
    assert errors == [dict(pool="yam", episode="yam/d/1", error="OSError")]
    assert records.count("tianji/d/0") == 3 and len(records) == 8
