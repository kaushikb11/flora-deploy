"""``stages.reward`` with a fake NVIDIA core and a fake LLM (no NVIDIA code, no GPU)."""

from __future__ import annotations

import argparse
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from flora_deploy.stages import reward


class Params:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = type("SamplingParams", (Params,), {})
    sp = types.ModuleType("vllm.sampling_params")
    sp.StructuredOutputsParams = type("StructuredOutputsParams", (Params,), {})
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)


class FakeCollage:
    def __init__(self, paths):
        self.paths = list(paths)
        self.fps, self.num_frames = 10.0, 10 * len(paths[0])  # length from the file name
        self.duration_s = self.num_frames / self.fps
        self.released = 0

    def frames(self, indices):
        return np.zeros((len(indices), 2, 2, 3), np.uint8) + len(self.paths)

    def release(self):
        self.released += 1


class FakeWindow(SimpleNamespace):
    pass


def make_core():
    core = SimpleNamespace()
    core.Collage = FakeCollage
    core.TERMINAL_TOP_LOGPROBS = 20
    core.DENSE_MAX_TOKENS = 640
    core.DENSE_TAIL_TRIM_S = 4.0
    core.terminal_frame_indices = lambda n, fps: [0, n - 1]
    core.terminal_messages = lambda task, r: [("terminal", task, r)]
    core.terminal_regex = lambda r: "R" if r else "S"
    core.terminal_max_tokens = lambda r: 160 if r else 12
    core.dense_regex = lambda: "D"
    core.video_metadata = lambda idx, fps, n: {"frames_indices": list(idx), "fps": fps}
    core.dense_windows = lambda n, fps: [
        FakeWindow(index=i, frame_indices=(i, i + 1)) for i in range(max(1, n // 20))
    ]
    core.dense_messages = lambda task, w: [("dense", task, w.index)]

    def parse_terminal(text, r):
        score = int(text[-1])
        return score, ("why" if r else None)

    def score_probabilities(positions, score):
        p = {str(k): (0.6 if k == score else 0.1) for k in range(1, 6)}
        return p, sum(k * v for k, v in zip(range(1, 6), p.values(), strict=True))

    core.parse_terminal = parse_terminal
    core.score_probabilities = score_probabilities
    core.parse_dense = lambda text: SimpleNamespace(text=text)
    return core


def make_nv():
    core = make_core()
    engine = SimpleNamespace(
        render=lambda proc, messages, thinking: f"{proc}|{messages}|{thinking}",
        COLLAGE_VIDEO_SIZE={"shortest_edge": 1, "longest_edge": 2},
    )

    def summarize(job, collage, windows, answers, texts, trim, frames):
        assert frames is True
        return {"id": job["id"], "fps": collage.fps, "num_frames": collage.num_frames,
                "segments": [{"n": len(texts)}], "frame_scores": [3] * collage.num_frames,
                "invalid_clips": 0, "clips": [{"output": t} for t in texts], "trim": trim}

    return SimpleNamespace(core=core, engine=engine,
                           dense_reward=SimpleNamespace(summarize=summarize))


class Tok:
    def decode(self, ids):
        return str(ids[0])


class FakeLLM:
    def __init__(self):
        self.calls = []

    def get_tokenizer(self):
        return Tok()

    def generate(self, requests, params, use_tqdm=True):
        self.calls.append((requests, params, use_tqdm))
        outs = []
        for r in requests:
            text = "Score: 4"
            lp = [{4: SimpleNamespace(logprob=-0.1), 3: SimpleNamespace(logprob=-2.0)}]
            outs.append(SimpleNamespace(prompt_token_ids=[0] * len(r["prompt"]),
                                        outputs=[SimpleNamespace(text=text, token_ids=[4],
                                                                 logprobs=lp)]))
        return outs


EPISODES = [
    reward.Episode("a", "task a", ("x" * 4, "y" * 4, "z" * 4)),  # 40 frames -> 2 windows
    reward.Episode("b", "task b", ("x" * 6,)),  # 60 frames -> 3 windows
]


def test_modes():
    assert reward.with_rationale("terminal_rationale") and not reward.with_rationale("dense")
    with pytest.raises(ValueError):
        reward.check_mode("rubric")


def test_terminal_one_generate_call_and_record_schema():
    nv, llm = make_nv(), FakeLLM()
    run = reward.run(llm, nv, "PROC", EPISODES, "terminal_rationale")
    assert len(llm.calls) == 1
    requests, params, use_tqdm = llm.calls[0]
    assert use_tqdm is False and len(requests) == 2
    assert requests[0]["prompt"] == "PROC|[('terminal', 'task a', True)]|False"
    assert requests[0]["mm_processor_kwargs"] == {"do_sample_frames": False}
    frames, meta = requests[0]["multi_modal_data"]["video"]
    assert frames.shape == (2, 2, 2, 3) and meta["frames_indices"] == [0, 39]
    assert params.kw["max_tokens"] == 160 and params.kw["logprobs"] == 20
    assert params.kw["structured_outputs"].kw == {"regex": "R"}
    assert params.kw["temperature"] == 0.0
    rec = run.records[0]
    assert rec["schema"] == "flora_labels_v1" and rec["episode_id"] == "a"
    t = rec["terminal"]
    assert t["score"] == 4 and len(t["probabilities"]) == 5 and t["rationale"] == "why"
    assert math.isclose(t["expected_score"], 0.1 * (1 + 2 + 3 + 5) + 0.6 * 4)
    assert rec["nv"]["probabilities"]["4"] == 0.6 and rec["nv"]["sampled_frames"] == 2
    assert rec["nv"]["progress"] == round((t["expected_score"] - 1) / 4, 4)
    assert run.stats["generate_calls"] == 1 and run.stats["num_requests"] == 2
    assert run.stats["prompt_tokens"] == sum(len(r["prompt"]) for r in requests)


def test_terminal_without_rationale_omits_it():
    run = reward.run(FakeLLM(), make_nv(), "P", EPISODES[:1], "terminal")
    assert "rationale" not in run.records[0]["terminal"]
    assert run.records[0]["nv"]["rationale"] is None


def test_dense_batches_all_windows_of_all_episodes():
    nv, llm = make_nv(), FakeLLM()
    run = reward.run(llm, nv, "P", EPISODES, "dense", tail_trim_s=2.0)
    (requests, params, _), = llm.calls
    assert [r["prompt"].split("|")[1] for r in requests] == [
        "[('dense', 'task a', 0)]", "[('dense', 'task a', 1)]",
        "[('dense', 'task b', 0)]", "[('dense', 'task b', 1)]", "[('dense', 'task b', 2)]",
    ]
    assert params.kw["max_tokens"] == 640 and "logprobs" not in params.kw
    a, b = run.records
    assert a["dense"]["segments"] == [{"n": 2}] and b["dense"]["segments"] == [{"n": 3}]
    assert len(b["dense"]["per_step"]) == 60 and b["nv"]["trim"] == 2.0
    assert [u["window"] for u in b["requests"]] == [0, 1, 2]
    assert reward.run(FakeLLM(), nv, "P", EPISODES, "dense").records[0]["nv"]["trim"] == 4.0


def test_shared_engine_checks_args_and_restores():
    nv = make_nv()
    nv.engine.base_parser = lambda desc: _parser()
    spec = reward.engine_spec(nv, "/m")
    module = SimpleNamespace(load_llm="orig")
    args = argparse.Namespace(**spec.args)
    llm = object()
    with reward.shared_engine(module, llm, spec, {"max_logprobs": 20}) as calls:
        kw = dict(spec.kwargs)
        assert module.load_llm("/m", args, **kw) is llm
        kw.pop("max_logprobs")  # the dense script relies on vLLM's default
        assert module.load_llm("/m", args, **kw) is llm
        with pytest.raises(reward.EngineMismatchError, match="max_model_len"):
            module.load_llm("/m", args, **{**kw, "max_model_len": 8192})
        with pytest.raises(reward.EngineMismatchError, match="max_num_seqs"):
            module.load_llm("/m", argparse.Namespace(**{**spec.args, "max_num_seqs": 4}), **kw)
    assert module.load_llm == "orig" and len(calls) == 4
    with reward.shared_engine(module, llm, spec, {"max_logprobs": 5}):
        with pytest.raises(reward.EngineMismatchError, match="max_logprobs"):
            module.load_llm("/m", args, **kw)


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--video", nargs="+")
    p.add_argument("--output")
    p.add_argument("--model")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p


def test_build_engine_uses_nvidia_load_llm():
    nv = make_nv()
    nv.engine.base_parser = lambda desc: _parser()
    seen = {}

    def load_llm(path, args, **kw):
        seen.update(path=path, args=vars(args), kw=kw)
        return "LLM"

    nv.engine.load_llm = load_llm
    llm, spec = reward.build_engine(nv, "/m")
    assert llm == "LLM" and seen["path"] == "/m"
    assert seen["args"] == {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.88,
                            "max_num_seqs": 8, "seed": 0}
    assert seen["kw"] == {"max_model_len": 16384, "max_logprobs": 20,
                          "mm_processor_kwargs": {"do_sample_frames": False,
                                                  "size": {"shortest_edge": 1,
                                                           "longest_edge": 2}}}
    assert spec.as_json()["model"] == "/m"


def test_tap_and_recording_llm():
    core = make_core()
    original = core.score_probabilities
    with reward.tap_score_probabilities(core) as seen:
        core.score_probabilities([], 2)
    assert core.score_probabilities is original and seen == [[0.1, 0.6, 0.1, 0.1, 0.1]]

    ticks = iter([0.0, 2.0])
    rec = reward.RecordingLLM(FakeLLM(), clock=lambda: next(ticks))
    rec.generate([{"prompt": "abcd"}, {"prompt": "ab"}], None)
    stats = rec.stats("terminal", 2)
    assert stats["generate_calls"] == 1 and stats["prompt_tokens"] == 6
    assert stats["throughput"]["requests_per_s"] == 1.0
    assert isinstance(rec.get_tokenizer(), Tok)


def test_reference_argv():
    argv = reward.reference_argv("terminal_rationale", "j.jsonl", "o.jsonl", "/m")
    assert argv[-1] == "--with-rationale" and "--overwrite" in argv
    assert reward.reference_argv("dense", "j", "o", "/m")[-1] == "--frame-scores"


def test_compare_terminal_and_dense():
    nv_line = {"id": "a", "score": 4, "output": "Score: 4", "probabilities": {"4": 0.6}}
    rec = {"episode_id": "a", "nv": dict(nv_line), "terminal": {"probabilities": [0.1] * 5}}
    ok = reward.compare("terminal", [nv_line], [rec], [[0.1] * 5])
    assert ok["identical"] and ok["episodes"][0]["max_abs_prob_diff"] == 0.0
    off = reward.compare("terminal", [nv_line], [rec], [[0.1 + 2e-6] + [0.1] * 4])
    assert not off["identical"]
    changed = dict(rec, nv={**nv_line, "output": "Score: 3"})
    res = reward.compare("terminal", [nv_line], [changed])
    assert not res["identical"] and res["episodes"][0]["diff_keys"] == ["output"]
    assert not res["episodes"][0]["text_equal"]
    assert not reward.compare("terminal", [nv_line, {**nv_line, "id": "b"}], [rec])["identical"]

    dense_line = {"id": "a", "clips": [{"output": "x"}, {"output": "y"}], "frame_scores": [3]}
    drec = {"episode_id": "a", "nv": {**dense_line, "clips": [{"output": "x"}, {"output": "z"}]}}
    res = reward.compare("dense", [dense_line], [drec])
    assert not res["identical"] and res["episodes"][0]["clip_text_mismatches"] == [1]
    assert res["episodes"][0]["frame_scores_equal"]


def test_read_jobs_uses_nvidia_reader(tmp_path):
    seen = {}

    def read_jobs(args):
        seen["jobs"] = args.jobs
        return [{"id": "e1", "task": "t", "videos": ["/a.mp4", "/b.mp4"]}]

    nv = SimpleNamespace(engine=SimpleNamespace(read_jobs=read_jobs))
    (ep,) = reward.read_jobs(nv, tmp_path / "jobs.jsonl")
    assert ep == reward.Episode("e1", "t", ("/a.mp4", "/b.mp4"))
    assert ep.as_job()["videos"] == ["/a.mp4", "/b.mp4"]
    assert seen["jobs"] == tmp_path / "jobs.jsonl"
