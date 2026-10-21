"""``stages.rubric`` (reward.rubric) on the fake Rewarder family, the ``pipeline.rubric`` config,
its hashing, and the reward worker's rubric path (stamp, crash classification).

The model inputs are proven identical to NVIDIA's unmodified script in
``tests/nvlink/test_rubric_identity.py`` (with NVIDIA's real ``core.medoid`` on ties)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from flora_deploy.config.hashing import stage_hashes
from flora_deploy.config.schema import ConfigError, parse_config, rubric_spec
from flora_deploy.jobs import attempts as A
from flora_deploy.local.engines import FakeLLM, fake_rewarder_family, vllm_shim
from flora_deploy.nv.errors import NvidiaValueError
from flora_deploy.stages import reward as R
from flora_deploy.stages import rubric as RB
from flora_deploy.workers_impl.certstamp import CertStamp
from flora_deploy.workers_impl.reward import RewardEngine, RewardShardRunner, RewardWorkerSettings

SHA = "a" * 40
RUBRIC = {"robot": "bimanual robot",
          "cameras": [{"role": "main", "description": "a top (scene) camera"},
                      {"role": "left_wrist", "description": "a left-wrist camera"},
                      {"role": "right_wrist", "description": "a right-wrist camera"}],
          "steps": [{"action": "Grasp the cup",
                     "grading": [{"label": "Not attempted", "points": 0.0},
                                 {"label": "Clean grasp", "points": 0.2}]},
                    {"action": "Set the cup on the tray", "max_score": 0.8,
                     "grading": [{"label": "Not attempted", "points": 0.0},
                                 {"label": "Placed upright", "points": 0.8}]}]}


def cfg_dict(rubric=RUBRIC, stages=("reward.rubric",)) -> dict:
    d = {"project": "demo", "license_ack": "research-only",
         "modal": {"max_gpus": 8, "budget_usd": 20}, "preset": "exact",
         "input": {"format": "lerobot_v3", "repo": "org/ds", "revision": SHA,
                   "cameras": {"top": "main", "l": "left_wrist", "r": "right_wrist"}},
         "pipeline": {"stages": list(stages), "embodiment": "yam"},
         "output": {"sink": "s3://b/p"}}
    if rubric is not None:
        d["pipeline"]["rubric"] = copy.deepcopy(rubric)
    return d


@pytest.fixture(autouse=True)
def shim():
    with vllm_shim():
        yield


# --- config ----------------------------------------------------------------------------------


def test_rubric_config_round_trips_in_order():
    cfg = parse_config(cfg_dict())
    spec = rubric_spec(cfg)
    assert spec == RUBRIC  # None keys dropped (NVIDIA's max_score default), order kept
    rs = RB.RubricSpec.from_json(spec)
    assert rs.roles == ("main", "left_wrist", "right_wrist") and rs.robot == "bimanual robot"
    assert rs.steps[1]["max_score"] == 0.8 and "max_score" not in rs.steps[0]


@pytest.mark.parametrize("change,match", [
    (lambda d: d["pipeline"].pop("rubric"), "needs `pipeline.rubric`"),
    (lambda d: d["pipeline"].update(stages=["reward.terminal"]), "only applies"),
    (lambda d: d["pipeline"]["rubric"].update(header="Header.\n\n"), "not both"),
    (lambda d: d["pipeline"]["rubric"].pop("robot"), "needs `robot`"),
    (lambda d: d["pipeline"]["rubric"]["cameras"].append({"role": "main",
                                                         "description": "again"}), "more than"),
    (lambda d: d["pipeline"]["rubric"].update(cameras=RUBRIC["cameras"][:1]), "input camera"),
    (lambda d: d["pipeline"]["rubric"]["steps"][0].update(grading=[]), "grading"),
    (lambda d: d["pipeline"]["rubric"]["steps"][0].update(action="see <|vision_start|>"),
     "special token"),
])
def test_rubric_config_refusals(change, match):
    d = cfg_dict()
    change(d)
    with pytest.raises(ConfigError, match=match):
        parse_config(d)


def test_header_form_needs_no_descriptions():
    r = {**RUBRIC, "robot": None, "header": "You see a robot.\n\n",
         "cameras": [{"role": c["role"]} for c in RUBRIC["cameras"]]}
    r.pop("robot")
    spec = RB.RubricSpec.from_json(rubric_spec(parse_config(cfg_dict(r))))
    assert spec.header == "You see a robot.\n\n" and spec.cameras is None
    job = spec.job(R.Episode("e", "Put the cup.", ("/a", "/b", "/c")))
    assert job["header"] == spec.header and "robot" not in job and "cameras" not in job


def test_rubric_is_hashed_into_its_own_stage_only():
    both = ("reward.terminal", "reward.rubric")
    a = stage_hashes(parse_config(cfg_dict(stages=both)))
    r2 = copy.deepcopy(RUBRIC)
    r2["steps"][0]["grading"][1]["points"] = 0.25
    b = stage_hashes(parse_config(cfg_dict(r2, stages=both)))
    assert a["reward.terminal"] == b["reward.terminal"]
    assert a["reward.rubric"] != b["reward.rubric"]
    r3 = copy.deepcopy(RUBRIC)
    r3["cameras"] = r3["cameras"][::-1]  # video order is the prompt order: hashed
    c = stage_hashes(parse_config(cfg_dict(r3, stages=both)))
    assert c["reward.rubric"] != a["reward.rubric"]


# --- stage -----------------------------------------------------------------------------------


def nv_with(video_inputs=None):
    nv = fake_rewarder_family()
    calls = []

    def default(messages):
        calls.append(messages)
        paths = [c["video"] for c in messages[-1]["content"] if c.get("type") == "video"]
        return [(np.full((2, 4, 4, 3), sum(map(ord, p)) % 255, np.uint8),
                 {"fps": 2.0, "duration": 3.5, "total_num_frames": 7}) for p in paths]

    nv.rubric_reward.video_inputs = video_inputs or default
    nv.calls = calls
    return nv


def episodes(n=3):
    return [R.Episode(f"e{i}", f"Put cup {i} on the tray.",
                      tuple(f"/v/e{i}-{r}.mp4" for r in ("m", "l", "r")), dataset="org/ds",
                      envelope_ok=True,
                      sources=tuple((r, f"videos/{r}/e{i}.mp4", 1.0 * i, None)
                                    for r in ("main", "left_wrist", "right_wrist")))
            for i in range(n)]


SPEC = RB.RubricSpec.from_json(RUBRIC)


def test_recipe_comes_from_nvidias_parser():
    rec = RB.nvidia_recipe(nv_with(), "/m", 3)
    assert rec.num_samples == 9 and rec.temperature == 1.0
    assert rec.engine.args["tensor_parallel_size"] == 2
    assert rec.engine.args["gpu_memory_utilization"] == 0.85
    assert rec.engine.kwargs == {"max_model_len": 4096, "videos_per_prompt": 3,
                                 "mm_processor_kwargs": {"do_resize": False},
                                 "max_num_batched_tokens": 1024}
    p = RB.sampling_params(rec)
    assert (p.n, p.temperature, p.top_p, p.top_k) == (9, 1.0, 1.0, 0)


def test_run_records_audits_and_one_generate_per_episode():
    nv, llm = nv_with(), CountingLLM()
    out = RB.run(llm, nv, "PROC", episodes(), SPEC, models={"rewarder": "7d"}, gpu="H100")
    assert llm.calls == 3 and [len(r) for r in llm.batches] == [1, 1, 1]
    assert len(out.records) == 3 and not out.errors
    for rec, audit in zip(out.records, out.audits, strict=True):
        rb, line = rec.rubric, audit["nv"]
        assert rec.stage == "reward.rubric" and rec.certified
        assert rb.samples == 9 and rb.medoid_index == line["medoid_sample"]
        assert list(rb.scores) == line["scores"] and rb.total == line["total"]
        assert rb.parsed_samples == line["parsed_samples"] == sum(
            s["parsed"] for s in line["samples"])
        assert line["steps"] == ["Grasp the cup", "Set the cup on the tray"]
        assert rec.provenance.source["main"]["range_s"][1] == pytest.approx(
            rec.provenance.source["main"]["range_s"][0] + 3.5)
    # NVIDIA's prompt pieces: thinking + vision ids on, the rubric's camera order
    assert all("thinking=True" in p and "vision_ids=True" in p for p in llm.prompts)
    msgs = nv.calls[0]
    assert [c["video"] for c in msgs[-1]["content"][:-1]] == list(episodes()[0].videos)


def test_medoid_sample_maps_back_through_unparsed_samples():
    core = fake_rewarder_family().core
    s = [{"parsed": False, "scores": [9.0]}, {"parsed": True, "scores": [0.2]},
         {"parsed": True, "scores": [0.4]}, {"parsed": False, "scores": [0.0]},
         {"parsed": True, "scores": [0.6]}]
    assert RB.medoid_sample(core, s) == 2  # parsed [0.2, 0.4, 0.6]: the middle one
    tie = [{"parsed": True, "scores": [0.0]}, {"parsed": True, "scores": [1.0]}]
    assert RB.medoid_sample(core, tie) == 0  # tie: lowest index
    with pytest.raises(ValueError, match="no parsed rubric answers"):
        RB.medoid_sample(core, [{"parsed": False, "scores": [0.0]}])


class CountingLLM(FakeLLM):
    def __init__(self, fail=None, unparsed=False):
        super().__init__()
        self.batches, self.prompts, self.fail, self.unparsed = [], [], fail, unparsed
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1
        return True

    def generate(self, requests, params, use_tqdm=True):
        self.batches.append(requests)
        self.prompts += [r["prompt"] for r in requests]
        if self.fail is not None and len(self.batches) == 2:
            self.calls += 1
            raise self.fail
        outs = super().generate(requests, params, use_tqdm)
        if self.unparsed and len(self.batches) == 2:
            for c in outs[0].outputs:
                c.text = "<think>never closed"
        return outs


def test_no_parsed_sample_is_nvidias_verdict_with_its_line():
    out = RB.run(CountingLLM(unparsed=True), nv_with(), "PROC", episodes(), SPEC)
    assert len(out.records) == 2
    (err,) = out.errors
    assert err.kind == "nv_failure" and err.message == "no parsed rubric answers"
    assert err.diagnostics["nv"]["scores"] is None
    assert err.diagnostics["nv"]["medoid_sample"] is None


@pytest.mark.parametrize("exc,kind", [
    (OSError("connection reset"), "infra"),
    (RuntimeError("engine core died"), "infra"),
])
def test_generate_failures_are_retried_infra(exc, kind):
    out = RB.run(CountingLLM(fail=exc), nv_with(), "PROC", episodes(), SPEC)
    assert len(out.records) == 2 and [e.kind for e in out.errors] == [kind]
    assert out.errors[0].diagnostics["phase"] == "generate"


def test_cuda_fault_is_a_crash():
    with pytest.raises(RB.RubricCrash) as ei:
        RB.run(CountingLLM(fail=RuntimeError("CUDA error: an illegal memory access")),
               nv_with(), "PROC", episodes(), SPEC)
    assert ei.value.crash_kind == "cuda_fatal"


def test_decode_rejection_is_nvidias_verdict_and_isolated():
    def video_inputs(messages):
        if "e1" in messages[-1]["content"][0]["video"]:
            raise NvidiaValueError("cannot decode e1")
        return nv_with().rubric_reward.video_inputs(messages)

    out = RB.run(CountingLLM(), nv_with(video_inputs), "PROC", episodes(), SPEC)
    assert [r.episode_id for r in out.records] == ["e0", "e2"]
    (err,) = out.errors
    assert err.kind == "nv_failure" and err.diagnostics["phase"] == "inputs"


def test_sink_gets_one_part_per_episode_and_prefix_cache_resets(tmp_path):
    llm = CountingLLM()
    sink = R.LocalPartSink(tmp_path)
    out = RB.run(llm, nv_with(), "PROC", episodes(), SPEC, sink=sink, prefetch=False)
    names = sorted(p.rsplit("/", 1)[-1] for p in out.parts)
    assert names == [f"part-{k:05d}{s}.jsonl" for k in range(3) for s in (".audit", "")]
    assert llm.resets == 3


def test_camera_mismatch_is_a_config_error():
    with pytest.raises(RB.RubricCamerasMismatch):
        RB.episode_for_roles(SPEC, ("main",))
    assert RB.episode_for_roles(SPEC, ("right_wrist", "main", "left_wrist")) == SPEC.roles


# --- worker ----------------------------------------------------------------------------------


def test_worker_runs_rubric_shards_stamped_and_crashes_are_marked(tmp_path):
    from tests.fakes.modal import FakeStore
    from tests.unit.test_reward_worker import CFG, JOB, Clock, ep, jf_of, setup, tshard

    cams = {"main": "/v/m.mp4", "left_wrist": "/v/l.mp4", "right_wrist": "/v/r.mp4"}
    store = FakeStore()
    shard = tshard(["a", "b"], sid="reward.rubric-00000", kind="reward.rubric")
    setup(store, [ep("a", cams=cams), ep("b", cams=cams)], [shard])

    def runner(llm, cid, fatal=None):
        s = RewardWorkerSettings(local_dir=str(tmp_path / cid), container_id=cid, gpu="H100",
                                 models={"rewarder": "7d"}, rubric=RUBRIC, media_root="/v",
                                 cert_stamps={"reward.rubric": CertStamp("yam", {"k": 1},
                                                                         "NVIDIA H100")})
        return RewardShardRunner(store, RewardEngine(llm, "PROC"), nv_with(), s,
                                 call_id=lambda: "fc", clock=Clock(), fatal=fatal)

    died = []
    with pytest.raises(RB.RubricCrash):
        runner(CountingLLM(fail=RuntimeError("CUDA error: device-side assert")), "c1",
               died.append).run_shard(JOB, shard.shard_id, CFG)
    marker = A.read_attempt(jf_of(store), shard.kind, shard.shard_id, "c1")
    assert marker.crashed and marker.error.startswith("cuda_fatal") and len(died) == 1

    receipt = runner(CountingLLM(), "c2").run_shard(JOB, shard.shard_id, CFG)
    assert set(receipt.episodes_done) == set(shard.episodes) and receipt.kind == "reward.rubric"
    recs = [json.loads(line) for p in receipt.parts
            for line in store.get_bytes(p).decode().splitlines()]
    assert len(recs) == 2 and all(r["stage"] == "reward.rubric" for r in recs)
    assert all(r["provenance"]["embodiment"] == "yam" for r in recs)
    assert all(r["provenance"]["cert_key"] == {"k": 1} for r in recs)


def test_rubric_runner_refuses_other_kinds(tmp_path):
    from tests.fakes.modal import FakeStore
    from tests.unit.test_reward_worker import CFG, JOB, Clock, ep, setup, tshard

    store = FakeStore()
    shard = tshard(["a"])
    setup(store, [ep("a")], [shard])
    s = RewardWorkerSettings(local_dir=str(tmp_path), container_id="c", rubric=RUBRIC)
    r = RewardShardRunner(store, RewardEngine(CountingLLM(), "PROC"), nv_with(), s,
                          clock=Clock())
    with pytest.raises(RuntimeError, match="rubric worker runs reward.rubric shards only"):
        r.run_shard(JOB, shard.shard_id, CFG)


def test_fake_llm_answers_n_samples():
    nv = nv_with()
    req = RB.episode_request(nv, "PROC", SPEC, RB.nvidia_recipe(nv, "", 3), episodes(1)[0])
    (out,) = FakeLLM().generate([req], SimpleNamespace(n=9))
    assert len(out.outputs) == 9


def test_worker_settings_read_the_rubric_from_the_factory_env():
    from flora_deploy.modal_app.workers_app import reward_settings_from_env
    from flora_deploy.workers_impl.certstamp import cert_env
    from flora_deploy.workers_impl.reward import PreflightError

    cfg = parse_config(cfg_dict())
    env = {**cert_env(cfg, cfg.pools()),
           "FLORA_SHARD_KINDS": "reward.rubric"}
    assert reward_settings_from_env(env).rubric == RUBRIC
    assert reward_settings_from_env({"FLORA_SHARD_KINDS": "reward.terminal"}).rubric is None
    with pytest.raises(PreflightError, match="reward.rubric"):
        reward_settings_from_env({"FLORA_SHARD_KINDS": "reward.rubric"})
