"""``stages.reward`` with a fake NVIDIA core and a fake LLM (no NVIDIA code, no GPU)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from flora_deploy.records import StageRecord
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
        name = paths[0].rsplit("/", 1)[-1]
        if name.startswith("bad"):
            raise ValueError(f"{paths[0]}: decoded fewer frames than its container reports")
        if name.startswith("oserr"):
            raise OSError("volume read failed")
        self.fps, self.num_frames = 10.0, 10 * len(name)  # length from the file name
        self.duration_s = self.num_frames / self.fps
        self.width, self.height = 2 * len(paths), 2

    def frames(self, indices):
        return np.zeros((len(indices), 2, 2, 3), np.uint8) + len(self.paths)

    def release(self):
        pass


class FakeWindow(SimpleNamespace):
    pass


def make_core():
    core = SimpleNamespace()
    core.Collage = FakeCollage
    core.TERMINAL_TOP_LOGPROBS = 20
    core.DENSE_MAX_TOKENS = 640
    core.DENSE_TAIL_TRIM_S = 4.0
    core.DENSE_FPS = 4.0
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

    def parse_terminal(text, r):  # NVIDIA's semantics: ValueError on anything unexpected
        if not text.startswith("Score: ") or not text[-1].isdigit():
            raise ValueError(f"unexpected terminal answer: {text!r}")
        return int(text[-1]), ("why" if r else None)

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
        if any("STITCH_FAIL" in t for t in texts):
            raise ValueError("no kept groups cover the episode")
        scores = [1 + (i // 3) % 3 for i in range(collage.num_frames)]
        return {"id": job["id"], "fps": collage.fps, "num_frames": collage.num_frames,
                "segments": [{"n": len(texts)}], "frame_scores": scores,
                "invalid_clips": sum("BAD" in t for t in texts),
                "clips": [{"output": t} for t in texts], "trim": trim}

    return SimpleNamespace(core=core, engine=engine,
                           dense_reward=SimpleNamespace(summarize=summarize))


class Tok:
    def decode(self, ids):
        return str(ids[0])


class FakeLLM:
    """Answers depend only on the prompt; ``answers`` maps a prompt substring to a text."""

    def __init__(self, answers=None, raise_on=None, reset=True):
        self.calls, self.resets = [], 0
        self.answers, self.raise_on, self.reset = answers or {}, raise_on, reset

    def get_tokenizer(self):
        return Tok()

    def reset_prefix_cache(self):
        self.resets += 1
        return self.reset

    def generate(self, requests, params, use_tqdm=True):
        self.calls.append((requests, params, use_tqdm))
        if self.raise_on and any(self.raise_on in r["prompt"] for r in requests):
            raise ValueError("prompt too long")
        outs = []
        for r in requests:
            text = next((t for key, t in self.answers.items() if key in r["prompt"]),
                        f"Score: {1 + len(r['prompt']) % 5}")
            finish = "length" if "TRUNC" in text else "stop"
            score = int(text[-1]) if text[-1].isdigit() else 4
            lp = [{score: SimpleNamespace(logprob=-0.1), 9: SimpleNamespace(logprob=-2.0)}]
            outs.append(SimpleNamespace(
                prompt=r["prompt"], prompt_token_ids=[0] * len(r["prompt"]),
                outputs=[SimpleNamespace(text=text, token_ids=[score], logprobs=lp,
                                         finish_reason=finish)]))
        return outs


EPISODES = [
    reward.Episode("a", "task a", ("/x/" + "x" * 4, "/y/" + "y" * 4, "/z/" + "z" * 4)),
    reward.Episode("b", "task b", ("/x/" + "x" * 6,)),  # 60 frames -> 3 windows
]


def many(n):
    return [reward.Episode(f"e{i}", f"task {i}", ("/v/" + "v" * (2 + i % 5),) * (1 + i % 3))
            for i in range(n)]


def test_modes_and_rubric_deferred():
    assert reward.with_rationale("terminal_rationale") and not reward.with_rationale("dense")
    with pytest.raises(NotImplementedError, match="deferred"):
        reward.check_mode("rubric")
    with pytest.raises(ValueError):
        reward.check_mode("nope")


def test_terminal_record_schema_provenance_and_certification():
    nv, llm = make_nv(), FakeLLM()
    eps = [reward.Episode("a", "task a", EPISODES[0].videos, dataset="lerobot/x",
                          envelope_ok=True), EPISODES[1]]
    run = reward.run(llm, nv, "PROC", eps, "terminal_rationale", gpu="H100", models={"r": "7d"})
    assert len(llm.calls) == 1
    requests, params, use_tqdm = llm.calls[0]
    assert use_tqdm is False and len(requests) == 2
    assert requests[0]["prompt"] == "PROC|[('terminal', 'task a', True)]|False"
    assert requests[0]["mm_processor_kwargs"] == {"do_sample_frames": False}
    frames, meta = requests[0]["multi_modal_data"]["video"]
    assert frames.shape == (2, 2, 2, 3) and meta["frames_indices"] == [0, 39]
    assert params.kw["max_tokens"] == 160 and params.kw["logprobs"] == 20
    assert params.kw["structured_outputs"].kw == {"regex": "R"}
    rec, rec_b = run.records
    assert isinstance(rec, StageRecord) and rec.stage == "reward.terminal"
    assert rec.uid == "lerobot-x--a" and rec.dataset == "lerobot/x" and rec.episode_id == "a"
    assert rec.certified and rec.uncertified_reason is None
    assert not rec_b.certified and rec_b.uncertified_reason == "oracle_ineligible_source"
    t = rec.terminal
    assert t.rationale == "why" and len(t.probabilities) == 5
    score = t.score
    assert math.isclose(t.expected_score, 0.6 * score + 0.1 * (15 - score))
    p = rec.provenance
    assert p.engine == reward.ENGINE_LABEL and p.preset == "exact" and p.gpu == "H100"
    assert p.models == {"r": "7d"} and len(p.config_hash) == 64 and p.logic_version
    assert p.framework.startswith("flora-deploy ")
    assert p.source["camera_2"] == {"uri": "/z/zzzz", "range_s": [0.0, 4.0]}
    assert StageRecord.from_json(rec.to_json()) == rec
    audit = run.audits[0]
    assert audit["nv"]["probabilities"][str(score)] == 0.6 and audit["nv"]["sampled_frames"] == 2
    assert audit["requests"][0]["finish_reason"] == "stop"
    assert run.stats["generate_calls"] == 1 and run.stats["num_requests"] == 2
    assert run.stats["prompt_tokens"] == sum(len(r["prompt"]) for r in requests)
    assert llm.resets == 1


def test_envelope_false_and_config_hash_depends_on_mode():
    ep = reward.Episode("a", "t", EPISODES[1].videos, envelope_ok=False)
    rec = reward.run(FakeLLM(), make_nv(), "P", [ep], "terminal").records[0]
    assert rec.uncertified_reason == "outside_reference_envelope" and rec.terminal.rationale is None
    rec2 = reward.run(FakeLLM(), make_nv(), "P", [ep], "terminal_rationale").records[0]
    assert rec.provenance.config_hash != rec2.provenance.config_hash


def test_dense_per_step_is_per_half_second_and_truncation_flagged():
    nv = make_nv()
    llm = FakeLLM(answers={"'task b', 1)": "Worst: 3 TRUNC BAD"})
    run = reward.run(llm, nv, "P", EPISODES, "dense", tail_trim_s=2.0,
                     limits=reward.UNBOUNDED)
    (requests, params, _), = llm.calls
    assert [r["prompt"].split("|")[1] for r in requests] == [
        "[('dense', 'task a', 0)]", "[('dense', 'task a', 1)]",
        "[('dense', 'task b', 0)]", "[('dense', 'task b', 1)]", "[('dense', 'task b', 2)]",
    ]
    assert params.kw["max_tokens"] == 640 and "logprobs" not in params.kw
    a, b = run.records
    assert a.stage == "reward.dense" and a.dense.segments == ({"n": 2},)
    # 60 frames at 10 fps = 6 s -> 12 steps of 0.5 s, read at frame int((k + 0.5) * 5)
    frame_scores = run.audits[1]["nv"]["frame_scores"]
    assert len(b.dense.per_step) == 12
    assert list(b.dense.per_step) == [frame_scores[int((k + 0.5) * 5)] for k in range(12)]
    assert run.audits[1]["nv"]["trim"] == 2.0
    assert [u["window"] for u in run.audits[1]["requests"]] == [0, 1, 2]
    assert run.audits[1]["flags"] == {"truncated_clips": [1], "invalid_clips": 1}
    assert run.stats["flagged"] == {b.uid: run.audits[1]["flags"]}
    assert run.audits[0]["flags"] == {}
    assert reward.run(FakeLLM(), nv, "P", EPISODES, "dense").audits[0]["nv"]["trim"] == 4.0


def test_per_step_lengths():
    nv = make_nv()
    assert len(reward.per_step(nv, [3] * 1100, 50.0, 1100)) == 44  # 22.00 s
    assert len(reward.per_step(nv, [3] * 81, 5.0, 81)) == 33  # 16.2 s -> ceil(32.4)
    assert reward.per_step(nv, [1, 2, 3], 1.0, 3) == [1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("mode", ["terminal", "terminal_rationale", "dense"])
@pytest.mark.parametrize("decoder", ["inline", "thread"])
def test_rounds_equal_unbounded(mode, decoder):
    eps = many(23)
    one = FakeLLM()
    unbounded = reward.run(one, make_nv(), "P", eps, mode, limits=reward.UNBOUNDED,
                           decoder="inline")
    assert len(one.calls) == 1
    llm = FakeLLM()
    limits = reward.RoundLimits(max_requests=4, max_bytes=300, processed_factor=4.0)
    rounds = reward.run(llm, make_nv(), "P", eps, mode, limits=limits, decoder=decoder,
                        workers=3, lookahead=2)
    assert len(llm.calls) > 3
    assert all(len(reqs) <= 4 for reqs, _, _ in llm.calls)
    for reqs, _, _ in llm.calls:  # byte budget (one oversize request may run alone)
        sizes = [limits.estimate(r["multi_modal_data"]["video"][0]) for r in reqs]
        assert len(reqs) == 1 or sum(sizes) <= limits.max_bytes
    flat = [r["prompt"] for reqs, _, _ in llm.calls for r in reqs]
    assert flat == [r["prompt"] for r in one.calls[0][0]]
    same = reward.same_outputs(unbounded.as_json(), rounds.as_json())
    assert same["identical"], same
    assert [r.episode_id for r in rounds.records] == [e.id for e in eps]
    assert llm.resets == len(llm.calls)


def test_nvidia_aligned_rounds_mirror_nvidia_generate_calls():
    eps = many(19)
    llm = FakeLLM()
    reward.run(llm, make_nv(), "P", eps, "terminal")  # default: align="nvidia", group 8
    assert [len(reqs) for reqs, _, _ in llm.calls] == [8, 8, 3]
    # dense: never across episodes, at most 8 clips per call (NVIDIA's per-episode chunks)
    long = [reward.Episode("long", "t", ("/v/" + "v" * 19,)), EPISODES[0], EPISODES[1]]
    llm = FakeLLM()
    run = reward.run(llm, make_nv(), "P", long, "dense")
    groups = [[r["prompt"].split("|")[1].split(",")[1] for r in reqs] for reqs, _, _ in llm.calls]
    assert [len(g) for g in groups] == [8, 1, 2, 3] and all(len(set(g)) == 1 for g in groups)
    assert run.stats["budget_splits"] == 0
    # the byte budget still bounds memory, and says when it broke NVIDIA's grouping
    llm = FakeLLM()
    run = reward.run(llm, make_nv(), "P", eps, "terminal",
                     limits=reward.RoundLimits(max_bytes=200))
    assert run.stats["budget_splits"] > 0 and all(len(r) <= 8 for r, _, _ in llm.calls)
    # packed ignores episode boundaries
    llm = FakeLLM()
    reward.run(llm, make_nv(), "P", long, "dense",
               limits=reward.RoundLimits(max_requests=8, align="packed"))
    assert [len(reqs) for reqs, _, _ in llm.calls] == [8, 6]


def test_bad_output_is_isolated_with_nvidia_error():
    eps = many(6)
    llm = FakeLLM(answers={"task 2'": "Rationale: cut off by max_tok"})
    run = reward.run(llm, make_nv(), "P", eps, "terminal_rationale",
                     limits=reward.RoundLimits(max_requests=2), shard_id="s1", container_id="c1")
    assert [r.episode_id for r in run.records] == ["e0", "e1", "e3", "e4", "e5"]
    (err,) = run.errors
    assert err.kind == "nv_failure" and err.uid == "local--e2" and err.stage == "reward.terminal"
    assert err.message.startswith("unexpected terminal answer")
    assert err.shard_id == "s1" and err.container_id == "c1"
    assert err.diagnostics["outputs"] == ["Rationale: cut off by max_tok"]
    assert err.diagnostics["finish_reasons"] == ["stop"]


def test_bad_video_is_isolated_before_generate():
    eps = [EPISODES[0], reward.Episode("bad", "t", ("/v/bad.mp4",)),
           reward.Episode("io", "t", ("/v/oserr.mp4",)), EPISODES[1]]
    for decoder in ("inline", "thread"):
        llm = FakeLLM()
        run = reward.run(llm, make_nv(), "P", eps, "dense", decoder=decoder)
        assert [r.episode_id for r in run.records] == ["a", "b"]
        assert [(e.episode_id if hasattr(e, "episode_id") else e.diagnostics["episode_id"],
                 e.kind, e.diagnostics["phase"]) for e in run.errors] == [
            ("bad", "nv_failure", "decode"), ("io", "infra", "decode")]
        prompts = [r["prompt"] for reqs, _, _ in llm.calls for r in reqs]
        assert len(prompts) == 5 and not any("'t'" in p for p in prompts)


def test_dense_stitch_failure_is_isolated():
    llm = FakeLLM(answers={"'task a', 1)": "STITCH_FAIL"})
    run = reward.run(llm, make_nv(), "P", EPISODES, "dense")
    assert [r.episode_id for r in run.records] == ["b"]
    assert run.errors[0].kind == "nv_failure" and "no kept groups" in run.errors[0].message


def test_generate_error_retried_per_episode():
    eps = many(5)
    llm = FakeLLM(raise_on="task 3'")
    run = reward.run(llm, make_nv(), "P", eps, "terminal", limits=reward.UNBOUNDED)
    assert [r.episode_id for r in run.records] == ["e0", "e1", "e2", "e4"]
    (err,) = run.errors
    assert err.kind == "infra" and "prompt too long" in err.message
    assert run.stats["retries"] == 1 and len(llm.calls) == 1 + 5
    # identical outputs to a run without the poison episode
    clean = reward.run(FakeLLM(), make_nv(), "P", eps[:3] + eps[4:], "terminal")
    assert [r.to_json() for r in clean.records] == [r.to_json() for r in run.records]


def test_systemic_generate_failure_raises():
    class Dead(FakeLLM):
        def generate(self, requests, params, use_tqdm=True):
            raise RuntimeError("EngineCore died")

    with pytest.raises(RuntimeError, match="EngineCore"):
        reward.run(Dead(), make_nv(), "P", many(3), "terminal")


def test_prefix_cache_reset_must_succeed():
    with pytest.raises(RuntimeError, match="reset_prefix_cache"):
        reward.run(FakeLLM(reset=False), make_nv(), "P", many(1), "terminal")


def test_input_validation_and_empty():
    llm = FakeLLM()
    run = reward.run(llm, make_nv(), "P", [], "terminal")
    assert run.records == [] and llm.calls == [] and run.stats["num_episodes"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        reward.run(llm, make_nv(), "P", [EPISODES[0], EPISODES[0]], "terminal")
    with pytest.raises(ValueError, match="relative"):
        reward.run(llm, make_nv(), "P", [reward.Episode("r", "t", ("cam.mp4",))], "terminal")
    with pytest.raises(ValueError, match="empty task"):
        reward.validate_episodes([reward.Episode("r", " ", ("/cam.mp4",))])
    ep = reward.Episode.from_job({"id": "e", "task": "t", "videos": ["d/cam.mp4"]}, root="/j")
    assert ep.videos == ("/j/d/cam.mp4",)
    # the same source id in two datasets is two episodes
    two = [reward.Episode("e", "t", ("/a.mp4",), dataset="ds1"),
           reward.Episode("e", "t", ("/a.mp4",), dataset="ds2")]
    assert len(reward.validate_episodes(two)) == 2


def test_parts_written_per_round_and_survive_a_crash(tmp_path):
    eps = many(7)

    class Preempted(FakeLLM):
        def generate(self, requests, params, use_tqdm=True):
            if len(self.calls) == 2:
                raise SystemExit("preempted")  # not an Exception: no retry, the run dies
            return super().generate(requests, params, use_tqdm)

    sink = reward.LocalPartSink(tmp_path / "local", tmp_path / "pub")
    with pytest.raises(SystemExit):
        reward.run(Preempted(), make_nv(), "P", eps, "terminal_rationale",
                   limits=reward.RoundLimits(max_requests=3), sink=sink)
    names = sorted(p.name for p in (tmp_path / "pub").iterdir())
    assert names == ["part-00000.audit.jsonl", "part-00000.jsonl",
                     "part-00001.audit.jsonl", "part-00001.jsonl"]
    lines = (tmp_path / "pub" / "part-00001.jsonl").read_text().splitlines()
    assert [StageRecord.from_json(x).episode_id for x in lines] == ["e3", "e4", "e5"]

    sink = reward.LocalPartSink(tmp_path / "l2")
    llm = FakeLLM(answers={"task 1'": "garbage"})
    run = reward.run(llm, make_nv(), "P", eps, "terminal",
                     limits=reward.RoundLimits(max_requests=3), sink=sink)
    assert len(run.parts) == 3 * 2 + 1  # 3 rounds of records + audits, and one errors file
    err = json.loads((tmp_path / "l2" / "part-00000.errors.jsonl").read_text())
    assert err["uid"] == "local--e1" and err["kind"] == "nv_failure"


def test_store_part_sink(tmp_path):
    from flora_deploy.jobs.store import JobPaths

    class MemStore:
        def __init__(self):
            self.files = {}

        def put_bytes(self, path, data):
            self.files[path] = data

    store = MemStore()
    sink = reward.StorePartSink(store, JobPaths("job1"), kind="reward.terminal", shard_id="s-1",
                                container_id="ct", local_dir=tmp_path)
    run = reward.run(FakeLLM(), make_nv(), "P", many(4), "terminal",
                     limits=reward.RoundLimits(max_requests=2), sink=sink)
    cfg = run.stats["config_hash"]
    paths = JobPaths("job1")
    assert paths.part(cfg, "reward.terminal", "s-1", "ct", 1) in store.files
    assert paths.audit(cfg, "reward.terminal", "s-1", "ct", 0) in store.files
    assert sorted(run.parts) == sorted(store.files)


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
    rec.generate([{"prompt": "abcd", "multi_modal_data": {}},
                  {"prompt": "ab", "multi_modal_data": {}}], None)
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
    bat = {"records": [{"episode_id": "a", "terminal": {"probabilities": [0.1] * 5}}],
           "audits": [{"episode_id": "a", "nv": dict(nv_line)}], "errors": []}
    ok = reward.compare("terminal", [nv_line], bat, [[0.1] * 5])
    assert ok["identical"] and ok["episodes"][0]["max_abs_prob_diff"] == 0.0
    off = reward.compare("terminal", [nv_line], bat, [[0.1 + 2e-6] + [0.1] * 4])
    assert not off["identical"]
    changed = dict(bat, audits=[{"episode_id": "a", "nv": {**nv_line, "output": "Score: 3"}}])
    res = reward.compare("terminal", [nv_line], changed)
    assert not res["identical"] and res["episodes"][0]["diff_keys"] == ["output"]
    assert not res["episodes"][0]["text_equal"]
    assert not reward.compare("terminal", [nv_line, {**nv_line, "id": "b"}], bat)["identical"]
    errs = dict(bat, errors=[{"uid": "x", "kind": "infra", "message": "m",
                              "diagnostics": {"episode_id": "b"}}])
    assert not reward.compare("terminal", [nv_line], errs)["identical"]

    dense_line = {"id": "a", "clips": [{"output": "x"}, {"output": "y"}], "frame_scores": [3]}
    dbat = {"records": [], "errors": [], "audits": [
        {"episode_id": "a", "nv": {**dense_line, "clips": [{"output": "x"}, {"output": "z"}]}}]}
    res = reward.compare("dense", [dense_line], dbat)
    assert not res["identical"] and res["episodes"][0]["clip_text_mismatches"] == [1]
    assert res["episodes"][0]["frame_scores_equal"]


def test_read_jobs_uses_nvidia_reader(tmp_path):
    seen = {}

    def read_jobs(args):
        seen["jobs"] = args.jobs
        return [{"id": "e1", "task": "t", "videos": ["/a.mp4", "/b.mp4"]}]

    nv = SimpleNamespace(engine=SimpleNamespace(read_jobs=read_jobs))
    (ep,) = reward.read_jobs(nv, tmp_path / "jobs.jsonl", dataset="lerobot/x")
    assert ep == reward.Episode("e1", "t", ("/a.mp4", "/b.mp4"), dataset="lerobot/x")
    assert ep.as_job()["videos"] == ["/a.mp4", "/b.mp4"] and ep.uid == "lerobot-x--e1"
    assert seen["jobs"] == tmp_path / "jobs.jsonl"
