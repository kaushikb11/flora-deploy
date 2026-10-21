"""Reference-stage helpers: runtime-event capture, orchestration, persistence, sample choice."""

import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from flora_deploy.stages import reference as ref


def test_runtime_event_tee_parses_nvidia_lines(capsys):
    with ref.capture_runtime_events() as events:
        print("50% Finding actions")
        print("FLORA_RUNTIME " + json.dumps({"event": "generation_end", "input_tokens": 100,
                                             "generated_tokens": 7, "complete": True,
                                             "elapsed_s": 1.5, "peak_allocated_gib": 30.0}),
              flush=True)
        sys.stdout.write("FLORA_RUNTIME {broken\nFLORA_RUNTIME ")
        sys.stdout.write(json.dumps({"event": "generation_end", "input_tokens": 50,
                                     "generated_tokens": 3, "elapsed_s": 0.5}))
    assert [e["generated_tokens"] for e in events] == [7, 3]
    assert "Finding actions" in capsys.readouterr().out  # still forwarded
    stats = ref.generation_stats(events)
    assert stats == {
        "calls": 2, "input_tokens": 150, "max_input_tokens": 100, "generated_tokens": 10,
        "max_generated_tokens": 7, "incomplete": 1, "generation_s": 2.0,
        "peak_allocated_gib": 30.0,
    }
    assert ref.generation_stats([])["calls"] == 0


def test_persist_result_writes_whole_file(tmp_path):
    commits = []
    result = {"segments": [{"caption": "pick up the cup ☕"}], "duration_s": 1.0}
    path = ref.persist_result(result, golden_root=tmp_path, run_id="wsd-1", episode="homer_3",
                              commit=lambda: commits.append(1))
    assert path == tmp_path / "reference_runs" / "wsd-1" / "homer_3.json"
    assert json.loads(path.read_text()) == result
    assert "☕" in path.read_text()  # ensure_ascii=False, like NVIDIA's CLI
    assert commits == [1]
    assert not list(path.parent.glob("*.partial"))


@pytest.mark.parametrize("bad", ["../x", "", "a/b", ".hidden"])
def test_persist_result_rejects_unsafe_names(tmp_path, bad):
    with pytest.raises(ValueError):
        ref.persist_result({}, golden_root=tmp_path, run_id="r", episode=bad)


def test_persist_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        ref.persist_result({"x": float("nan")}, golden_root=tmp_path, run_id="r", episode="e")


def test_choose_human():
    c = [
        {"id": "a", "duration_s": 5.0, "error": None},
        {"id": "b", "duration_s": 40.0, "error": None},
        {"id": "c", "duration_s": 25.0, "error": None},
        {"id": "d", "duration_s": 30.0, "error": "InputError"},
        {"id": "e", "duration_s": 100.0, "error": None},
        {"id": "f", "duration_s": 60.0, "error": None},
    ]
    assert [x["id"] for x in ref.choose_human(c)] == ["c", "b", "f"]
    assert [x["id"] for x in ref.choose_human(c[:3])] == ["c", "b", "a"]
    with pytest.raises(RuntimeError):
        ref.choose_human(c[:2])


def test_aloha_video_file():
    assert ref.aloha_video_file("cam_high", 0) == (
        "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    )


# --- orchestration with a fake NVIDIA family -------------------------------------------------


class InputError(ValueError):
    pass


def _video(path, n=40):
    rate = Fraction(4)
    pts = tuple(Fraction(i * 250000) for i in range(n))
    return SimpleNamespace(path=Path(path), sha256="ab" * 32, rate=rate, pts_us=pts,
                           duration_us=n * 250000, duration_s=n / 4, size=(64, 48), cfr=True)


def fake_nv(calls):
    profiles = {"human": "Human", "yam": "YAM"}

    def _inputs(video, profile, left, right, yam32, instruction):
        calls.append(("inputs", profile, yam32))
        paths = {"main": video} | ({"left_wrist": left, "right_wrist": right} if left else {})
        return {r: _video(p) for r, p in paths.items()}, profiles[profile]

    def segment(videos, pool, session, instruction, progress):
        calls.append(("segment", pool, session.load("tempseg")))
        print("FLORA_RUNTIME " + json.dumps({"event": "generation_end", "input_tokens": 9,
                                             "generated_tokens": 4, "complete": True}))
        return ([{"start_s": 0.0, "end_s": 5.0, "metadata": {"fine_event_id": "e0"}},
                 {"start_s": 5.0, "end_s": 10.0,
                  "metadata": {"segmentation_provenance": "fine_repaired_by_fine_retry"}}],
                [{"window": 1}])

    def caption_single(videos, segments, profile, session, instruction, progress):
        calls.append(("caption_single", profile, session.load("captioner")))
        return ([dict(start_s=s["start_s"], end_s=s["end_s"], caption="c", status="complete")
                 for s in segments], [{"event_id": "event_0000"}, {"event_id": "event_0001"}])

    def caption_three(videos, segments, session, instruction, progress, profile="YAM"):
        calls.append(("caption_three", profile))
        return caption_single(videos, segments, profile, session, instruction, progress)

    return SimpleNamespace(
        api=SimpleNamespace(DataEnginePipeline=SimpleNamespace(_inputs=_inputs)),
        media=SimpleNamespace(check_synchronized=lambda v: calls.append(("sync", len(v))),
                              InputError=InputError),
        preprocessing=SimpleNamespace(instruction_suffix=lambda s: ""),
        config=SimpleNamespace(PROFILE_POOLS={"Human": "human", "YAM": "yam"},
                               MODELS={"tempseg": {"revision": "r"}}),
        models=SimpleNamespace(runtime_event=lambda e, **kw: calls.append(("event", e))),
        pipeline=SimpleNamespace(segment=segment, caption_single=caption_single,
                                 caption_three=caption_three),
    )


class FakeSession:
    def __init__(self):
        self.loads = []

    def load(self, kind):
        self.loads.append(kind)
        return kind


def test_annotate_reference_single_view():
    calls, session = [], FakeSession()
    out = ref.annotate_reference(fake_nv(calls), session, {"main": "/v/h.mp4"}, "human")
    assert [c[0] for c in calls] == ["inputs", "sync", "segment", "event", "caption_single",
                                     "event"]
    assert ("segment", "human", "tempseg") in calls
    assert out["schema_version"] == "flora_demo_annotation_v1"
    assert out["profile"] == "Human" and out["camera_roles"] == ["main"]
    assert [s["segmentation_provenance"] for s in out["segments"]] == [
        "fine", "fine_repaired_by_fine_retry"]
    assert out["pipeline"] == {"temporal_windows": [{"window": 1}],
                               "caption_inputs": [{"event_id": "event_0000"},
                                                  {"event_id": "event_0001"}]}
    fr = out["flora_reference"]
    assert fr["counts"] == {"temporal_windows": 1, "segments": 2, "caption_inputs": 2,
                            "captions_complete": 2}
    assert fr["generation"]["segment"]["generated_tokens"] == 4
    assert fr["generation"]["caption"]["calls"] == 0
    assert set(fr["timings_s"]) == {"probe", "segment", "caption", "total"}
    assert "flora_reference" not in ref.nvidia_result(out)
    row = ref.summary_row("h", out)
    assert row["windows"] == 1 and row["captions"] == 2 and row["views"] == 1


def test_annotate_reference_three_view_and_checks():
    calls = []
    videos = {"main": "/m.mp4", "left_wrist": "/l.mp4", "right_wrist": "/r.mp4"}
    out = ref.annotate_reference(fake_nv(calls), FakeSession(), videos, "yam")
    assert ("caption_three", "YAM") in calls and ("sync", 3) in calls
    assert out["camera_roles"] == ["main", "left_wrist", "right_wrist"]
    with pytest.raises(ValueError, match="camera roles"):
        ref.annotate_reference(fake_nv([]), FakeSession(), {"main": "/m", "left_wrist": "/l"},
                               "yam")
    with pytest.raises(InputError, match="32-second"):
        ref.annotate_reference(fake_nv([]), FakeSession(), {"main": "/m"}, "human",
                               long_windows=True)
