"""Reference-stage helpers: runtime-event capture, orchestration, persistence, sample choice."""

import json
import sys

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


# --- NVIDIA's own entry point with the session injected -------------------------------------

from tests.unit.test_reference_concurrency import (  # noqa: E402 - shared fake family
    FakeNvSession,
    GpuProbe,
    fake_nv,
)
from tests.unit.test_reference_concurrency import InputError as FakeInputError  # noqa: E402


def _entry(calls=None):
    from flora_deploy.nv.session import CachedSession
    from flora_deploy.observe import Patcher

    nv = fake_nv(GpuProbe(), calls)
    patcher = Patcher()
    ref.install_nvidia_entry(nv, patcher)
    return nv, CachedSession(FakeNvSession), patcher


def test_annotate_nvidia_single_view():
    calls = []
    nv, session, patcher = _entry(calls)
    out = ref.annotate_nvidia(nv, session, {"main": "/v/h.mp4"}, "human")
    assert ("api.annotate", "human", False) in calls  # NVIDIA's entry point ran
    assert not [c for c in calls if c[0] == "fresh_session"]  # with our session
    assert out["profile"] == "Human" and out["camera_roles"] == ["main"]
    assert "pipeline" in out  # include_audit=True
    fr = out["flora_reference"]
    assert fr["entry"] == "api.DataEnginePipeline.annotate"
    n = fr["counts"]["temporal_windows"]
    assert fr["counts"]["segments"] == fr["counts"]["captions_complete"] == n
    assert fr["generation"]["segment"]["calls"] == n  # split at segmentation_complete
    assert fr["generation"]["caption"]["calls"] == n
    assert [c["model"] for c in fr["model_calls"]] == ["tempseg"] * n + ["captioner"] * n
    assert set(fr["timings_s"]) == {"probe", "segment", "caption", "total"}
    assert "flora_reference" not in ref.nvidia_result(out)
    row = ref.summary_row("h", out)
    assert row["windows"] == n and row["views"] == 1
    patcher.restore()


def test_annotate_nvidia_three_view_and_nvidia_errors():
    calls = []
    nv, session, _ = _entry(calls)
    videos = {"main": "/m.mp4", "left_wrist": "/l.mp4", "right_wrist": "/r.mp4"}
    out = ref.annotate_nvidia(nv, session, videos, "yam")
    assert ("caption_three", "YAM") in calls
    assert out["camera_roles"] == ["main", "left_wrist", "right_wrist"]
    with pytest.raises(ValueError, match="camera roles"):
        ref.annotate_nvidia(nv, session, {"main": "/m", "left_wrist": "/l"}, "yam")
    with pytest.raises(FakeInputError, match="yam32 requires"):  # NVIDIA's own check
        ref.annotate_nvidia(nv, session, {"main": "/m"}, "human", long_windows=True)


def test_model_calls_pairs_outputs_and_skips_incomplete():
    events = [
        {"event": "generation_start", "output_limit": 1024, "input_tensor_digest": "a"},
        {"event": "generation_end", "complete": False, "generated_tokens": 1024},
        {"event": "generation_start", "output_limit": 1024, "input_tensor_digest": "b"},
        {"event": "generation_end", "complete": True, "generated_tokens": 5},
        {"event": "segmentation_complete"},
        {"event": "generation_start", "output_limit": 512, "input_tensor_digest": "c"},
        {"event": "generation_end", "complete": True, "generated_tokens": 3},
    ]
    result = {"pipeline": {"temporal_windows": [{"output": "W"}]},
              "segments": [{"status": "unsupported_target_frames", "caption": ""},
                           {"status": "complete", "caption": "C"}]}
    calls = ref.model_calls(events, result)
    assert [c["input_digest"] for c in calls] == ["a", "b", "c"]
    assert calls[0]["output_sha256"] is None and calls[0]["complete"] is False
    assert calls[1]["output_sha256"] == ref._sha("W") and calls[1]["model"] == "tempseg"
    assert calls[2]["output_sha256"] == ref._sha("C") and calls[2]["model"] == "captioner"


def test_nvidia_errors_pickle_with_type_name():
    import pickle

    class InputError(ValueError):  # stands in for NVIDIA's aliased media.InputError
        pass

    err = ref.transport_error(InputError("Choose a supported action profile."))
    assert isinstance(err, ref.NvidiaInputError) and isinstance(err, RuntimeError)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is ref.NvidiaInputError
    assert back.type_name.endswith("InputError") and back.message.startswith("Choose")
    other = ref.transport_error(KeyError("x"))
    assert type(other) is ref.NvidiaError and "KeyError" in str(other)
    assert ref.transport_error(other) is other


def test_run_ids_are_unique_and_safe():
    ids = {ref.new_run_id("s1") for _ in range(50)}
    assert len(ids) == 50 and all(ref.SAFE_NAME.fullmatch(i) for i in ids)


def test_persist_refuses_overwrite(tmp_path):
    ref.persist_result({"a": 1}, golden_root=tmp_path, run_id="r", episode="e")
    with pytest.raises(FileExistsError):
        ref.persist_result({"a": 2}, golden_root=tmp_path, run_id="r", episode="e")
    path = ref.persist_result({"a": 2}, golden_root=tmp_path, run_id="r", episode="e",
                              overwrite=True)
    assert json.loads(path.read_text()) == {"a": 2}


def test_exact_profile_toggles():
    p = ref.ExactProfile.from_spec()
    assert p.captioner_bf16 and p.linked_layout and p.prefetch and p.concurrency == 4
    assert len(p.accel) == 4
    prov = p.provenance()
    assert prov["overlap_hash"] is False  # irrelevant with the BF16 Captioner
    q = ref.ExactProfile.from_spec("captioner_bf16, png_inmem", concurrency=1)
    assert not q.captioner_bf16 and "png_inmem" not in q.accel and q.concurrency == 1
    assert q.provenance()["overlap_hash"] is True
    with pytest.raises(ValueError, match="unknown"):
        ref.ExactProfile.from_spec("turbo")


def test_stage_records_from_result():
    from flora_deploy.records import Provenance, StageRecord

    prov = Provenance(models={"tempseg": "r"}, engine="hf-ref", preset="exact",
                      config_hash="c", logic_version="1", framework="flora-deploy 0")
    result = {"segments": [
        {"start_s": 0.0, "end_s": 1.5, "caption": "pick", "status": "complete",
         "segmentation_provenance": "fine"},
        {"start_s": 1.5, "end_s": 2.0, "caption": "", "status": "unsupported_target_frames",
         "segmentation_provenance": "fine_repaired_by_fine_retry"}]}
    seg, cap = ref.stage_records(result, uid="ds--e1", dataset="ds", episode_id="e1",
                                 provenance=prov, audit_ref="runs/x.json")
    assert (seg.stage, cap.stage) == ("segment", "caption")
    assert seg.segments[0].caption is None and cap.segments[0].caption == "pick"
    assert cap.segments[1].status == "unsupported_target_frames"
    assert StageRecord.from_json(cap.to_json()) == cap


def test_golden_episode_inputs():
    manifest = {"episodes": [
        {"id": "yam/ds/ep1", "pool": "yam", "profile": "yam", "yam32": False,
         "oracle_eligible": True, "duration_s": 8.0, "task": " fold ",
         "path": "golden/v1/yam/ds/ep1",
         "files": {"main": {"file": "main.mp4"}, "left_wrist": {"file": "left_wrist.mp4"},
                   "right_wrist": {"file": "right_wrist.mp4"}}},
        {"id": "reward/x/y", "pool": "reward", "oracle_eligible": True, "files": {}},
    ]}
    eps = ref.golden_episode_inputs(manifest, ["yam/ds/ep1"], "/golden")
    assert eps[0]["episode"] == "yam__ds__ep1" and eps[0]["instruction"] == "fold"
    assert eps[0]["videos"]["left_wrist"] == "/golden/golden/v1/yam/ds/ep1/left_wrist.mp4"
    assert ref.golden_episode_inputs(manifest, ["yam/ds/ep1"], "/g",
                                     instruction=False)[0]["instruction"] == ""
    with pytest.raises(ValueError):
        ref.golden_episode_inputs(manifest, ["reward/x/y"], "/g")
    with pytest.raises(KeyError):
        ref.golden_episode_inputs(manifest, ["nope"], "/g")
    assert all(i.count("/") == 2 for i in ref.IDENTITY_EPISODES)
