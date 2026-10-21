import pytest

from flora_deploy.validate import gatec as G

SP = G.SpecialIds(vision_start=1, vision_end=2, video_pad=3)


def encode(s: str) -> list[int]:
    # fake tokenizer: one id per character, offset past the special ids
    return [100 + ord(c) for c in s]


def expanded(grids, stamps, prefix=(10, 11), suffix=(12,)):
    out = list(prefix)
    for (_t, h, w), ts in zip(grids, stamps, strict=True):
        for s in ts:
            out += encode(f"<{s:.1f} seconds>") + [1] + [3] * (h * w // 4) + [2]
        out.append(50)
    return out + list(suffix)


def ranges(grids, stamps, prefix_len=2):
    """vLLM-style placeholder ranges for ``expanded``: each video's whole replacement."""
    out, pos = [], prefix_len
    for (_t, h, w), ts in zip(grids, stamps, strict=True):
        length = sum(len(encode(f"<{s:.1f} seconds>")) + h * w // 4 + 2 for s in ts)
        out.append([pos, length])
        pos += length + 1  # the 50 separator
    return out


def test_frame_runs_and_summary():
    ids = [9, 1, 3, 3, 2, 8, 1, 3, 3, 2, 3]
    runs = G.frame_runs(ids, SP)
    assert runs == [G.FrameRun(1, 2), G.FrameRun(6, 2)]
    s = G.summarize_ids(ids, SP)
    assert s["frame_groups"] == 2 and s["pads_per_group"] == [2]
    assert s["vision_start"] == 2 and s["video_pad"] == 5
    assert s["bare_video_pad_positions"] == [10]


def test_frame_runs_malformed():
    with pytest.raises(ValueError):
        G.frame_runs([1, 3, 3], SP)
    with pytest.raises(ValueError):
        G.frame_runs([3, 2], SP)
    assert "structure_error" in G.summarize_ids([1, 3], SP)


def test_vllm_timestamps_pad_and_average():
    assert G.vllm_timestamps([0, 1, 2, 3], 2.0, 2) == [0.25, 1.25]
    # odd length: last index repeated
    assert G.vllm_timestamps([0, 2, 4], 4, 2) == [0.25, 1.0]


def test_simulate_expansion_matches_layout():
    grids = [[2, 4, 4], [1, 2, 2]]
    stamps = [[0.25, 1.25], [3.0]]
    prompt = [10, 11, 1, 3, 2, 50, 1, 3, 2, 50, 12]
    out = G.simulate_vllm_expansion(prompt, grids, stamps, encode, SP, 2)
    assert out == expanded(grids, stamps)
    assert G.grids_match_runs(out, grids, SP, 2)
    assert not G.grids_match_runs(out, [[2, 4, 4], [1, 4, 4]], SP, 2)


def test_simulate_expansion_bare_pad_is_not_a_target():
    with pytest.raises(RuntimeError, match="Found 0 video placeholders but expected 1"):
        G.simulate_vllm_expansion([10, 3, 12], [[1, 2, 2]], [[0.0]], encode, SP, 2)
    with pytest.raises(RuntimeError, match="timestamps"):
        G.simulate_vllm_expansion([1, 3, 2], [[2, 2, 2]], [[0.0]], encode, SP, 2)


def test_compare_pass_and_fail_details():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    ok = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf,
                   hf_grids=grids, vllm_ids=list(hf), vllm_grids=grids, sp=SP, merge_size=2,
                   pixels_equal=[True], generate_ids_equal=True,
                   extras={"placeholders": ranges(grids, [[0.25, 1.25]])})
    assert ok.placeholders_ok is True and ok.placeholder_error is None
    assert ok.passed and ok.first_diff is None
    bad_ids = list(hf)
    bad_ids[5] = 999
    bad = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf,
                    hf_grids=grids, vllm_ids=bad_ids, vllm_grids=grids, sp=SP, merge_size=2)
    assert not bad.passed and bad.first_diff == 5 and bad.grids_equal
    short = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf,
                      hf_grids=grids, vllm_ids=hf[:-1], vllm_grids=[[2, 2, 4]], sp=SP,
                      merge_size=2)
    assert short.first_diff == len(hf) - 1 and not short.grids_equal
    err = G.compare(case_id="c", index=0, source="s", variant="bare", hf_ids=hf,
                    hf_grids=grids, vllm_ids=None, vllm_grids=None, sp=SP, merge_size=2,
                    error="RuntimeError: x")
    d = err.to_dict()
    assert d["passed"] is False and d["error"] == "RuntimeError: x" and d["vllm"] == {}


def test_sanitize_error_is_an_allowlist():
    exc = ValueError("Failed to apply Qwen3VLProcessor on data={'text': 'SECRET'} with kwargs")
    assert G.sanitize_error(exc) == "ValueError: Failed to apply Qwen3VLProcessor [rest elided]"
    bare = AssertionError("Failed to apply prompt replacement for mm_items['video'][0]")
    assert G.sanitize_error(bare) == ("AssertionError: Failed to apply prompt replacement for "
                                      "mm_items['video'][0]")
    # unknown formats lose the whole message, whatever payload shape they use
    for msg in ("prompt text SECRET", "Bad input: {'prompt': 'SECRET'}", "a\nSECRET",
                "SECRET Failed to apply x"):
        out = G.sanitize_error(RuntimeError(msg))
        assert out == "RuntimeError: [message elided]" and "SECRET" not in out
    assert G.sanitize_error(KeyError()) == "KeyError"


def test_pass_requires_pixels_generate_and_no_bare_pads():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    ph = ranges(grids, [[0.25, 1.25]])
    kw = dict(case_id="c", index=0, source="s", variant="wrapped", hf_ids=hf, hf_grids=grids,
              vllm_ids=list(hf), vllm_grids=grids, sp=SP, merge_size=2, placeholders=ph)
    ok = dict(pixels_equal=[True], generate_ids_equal=True)
    assert G.compare(**kw, **ok).passed
    # ids and grids equal, pixels not: FAIL (e.g. frame order or resample changed)
    bad = G.compare(**kw, pixels_equal=[False], generate_ids_equal=True)
    assert bad.ids_equal and bad.grids_equal and not bad.passed
    assert bad.to_dict()["pixels_ok"] is False and bad.to_dict()["passed"] is False
    assert not G.compare(**kw, generate_ids_equal=True).passed  # pixels not compared
    assert not G.compare(**kw, pixels_equal=[], generate_ids_equal=True).passed
    assert not G.compare(**kw, pixels_equal=[True, True], generate_ids_equal=True).passed
    assert not G.compare(**kw, pixels_equal=[True], generate_ids_equal=False).passed
    stray = hf[:3] + [3] + hf[3:]  # a bare video_pad outside any frame run, on both sides
    s = G.compare(**{**kw, "hf_ids": stray, "vllm_ids": stray}, **ok)
    assert s.ids_equal and s.runs_match_grid and not s.passed


def test_wrapped_needs_generate_and_placeholder_checks_to_have_run():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    ph = ranges(grids, [[0.25, 1.25]])
    kw = dict(case_id="c", index=0, source="s", hf_ids=hf, hf_grids=grids, vllm_ids=list(hf),
              vllm_grids=grids, sp=SP, merge_size=2, pixels_equal=[True])
    assert G.compare(**kw, variant="wrapped", generate_ids_equal=True, placeholders=ph).passed
    skipped_gen = G.compare(**kw, variant="wrapped", generate_ids_equal=None, placeholders=ph)
    assert not skipped_gen.passed  # a skipped generate is not a pass
    no_ph = G.compare(**kw, variant="wrapped", generate_ids_equal=True)
    assert no_ph.placeholders_ok is None and not no_ph.passed
    # diagnostic variants: not run is fine, a negative result is not
    assert G.compare(**kw, variant="wrapped-no-duration").passed
    assert not G.compare(**kw, variant="wrapped-no-duration", generate_ids_equal=False).passed
    shifted = [[ph[0][0], ph[0][1] - 1]]
    assert not G.compare(**kw, variant="wrapped-no-duration", placeholders=shifted).passed


@pytest.mark.parametrize("mutate,needle", [
    (lambda ph: ph[:-1], "placeholder ranges for"),
    (lambda ph: [[ph[0][0], ph[0][1] - 1], ph[1]], "range ends"),
    (lambda ph: [[ph[0][0], ph[0][1] + 1], ph[1]], "range ends"),
    (lambda ph: [[ph[0][0] - 3, ph[0][1] + 3], ph[1]], "outside the prompt"),
    (lambda ph: [ph[0], [ph[1][0], ph[1][1] + 99]], "outside the prompt"),
    (lambda ph: [ph[0], [ph[1][0] - 3, ph[1][1] + 3]], "inside the previous one"),
    (lambda ph: [[ph[0][0] + 14, ph[0][1] - 14], ph[1]], "not at its timestamp"),
    (lambda ph: [ph[1], ph[0]], "range"),
    (lambda ph: [ph[0], [ph[0][0], ph[1][0] + ph[1][1] - ph[0][0]]], "range"),
])
def test_placeholder_ranges_must_sit_on_the_frame_runs(mutate, needle):
    grids, stamps = [[2, 4, 4], [1, 2, 2]], [[0.25, 1.25], [3.0]]
    ids = expanded(grids, stamps)
    ph = ranges(grids, stamps)
    assert G.placeholder_problem(ids, ph, grids, SP, 2) is None
    problem = G.placeholder_problem(ids, mutate(ph), grids, SP, 2)
    assert problem is not None and needle in problem
    c = G.compare(case_id="c", index=0, source="s", variant="wrapped", hf_ids=ids,
                  hf_grids=grids, vllm_ids=ids, vllm_grids=grids, sp=SP, merge_size=2,
                  pixels_equal=[True, True], generate_ids_equal=True, placeholders=mutate(ph))
    assert c.ids_equal and c.grids_equal and c.runs_match_grid and not c.passed
    assert c.to_dict()["placeholder_error"] == problem


def test_gate_verdict_needs_clean_builds_and_full_coverage():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    good = dict(hf_ids=hf, hf_grids=grids, vllm_ids=hf, vllm_grids=grids, sp=SP, merge_size=2,
                pixels_equal=[True], generate_ids_equal=True,
                placeholders=ranges(grids, [[0.25, 1.25]]))

    def row(case, i=0, variant="wrapped", **kw):
        return G.compare(case_id=case, index=i, source="s", variant=variant,
                         **{**good, **kw}).to_dict()

    parts = [("tempseg", "cases:synthetic"), ("tempseg", "golden:human"),
             ("captioner", "golden:human")]
    builds = [dict(kind="tempseg", part="cases:synthetic", n=2, errors=[]),
              dict(kind="tempseg", part="golden:human", n=1, errors=[]),
              dict(kind="captioner", part="golden:human", n=1, errors=[])]
    results = {"tempseg": {"rows": [row("temporal-human"), row("temporal-human", 1),
                                    row("temporal-human", 0, "bare", vllm_ids=None),
                                    row("golden-human")]},
               "captioner": {"rows": [row("golden-human")]}}
    cases = ["temporal-human", "golden-human"]
    v = G.gate_verdict(builds, results, expected_parts=parts, expected_cases=cases)
    assert v["passed"], v
    assert v["coverage"] == {"captioner": {"built": 1, "compared": 1, "passed": 1},
                             "tempseg": {"built": 3, "compared": 3, "passed": 3}}

    def verdict(builds=builds, results=results, cases=cases):
        return G.gate_verdict(builds, results, expected_parts=parts, expected_cases=cases)

    # a build error fails the gate, even though every compared row passed
    errs = [dict(b) for b in builds]
    errs[2]["errors"] = [dict(pool="human", episode="human/a/x", error="KeyError")]
    v = verdict(builds=errs)
    assert not v["passed"] and "1 build error(s) ['KeyError']" in v["problems"][0]
    # a pool whose builds produced nothing fails (and is not hidden as n/a)
    zero = [dict(b) for b in builds]
    zero[2]["n"] = 0
    v = verdict(builds=zero, results={**results, "captioner": {"rows": []}})
    assert not v["passed"] and "captioner golden:human: 0 requests" in v["problems"]
    assert not verdict(builds=builds[:2])["passed"]  # a part never built
    # built requests without a wrapped row
    short = {**results, "captioner": {"rows": []}}
    assert any("1 requests built but 0" in p for p in verdict(results=short)["problems"])
    # an expected case with no rows
    v = verdict(cases=[*cases, "caption-three-yam"])
    assert not v["passed"] and "caption-three-yam" in v["problems"][0]
    # one failing row
    bad = {**results, "captioner": {"rows": [row("golden-human", generate_ids_equal=None)]}}
    v = verdict(results=bad)
    assert not v["passed"] and v["failed_cases"] == ["golden-human"]

    # review 4 L3: a golden pool below its request quota fails (the recorded run's tianji
    # TempSeg had 5 of 6)
    v = G.gate_verdict(builds, results, expected_parts=parts, expected_cases=cases, quota=2)
    assert not v["passed"]
    assert "tempseg golden:human: 1 of 2 requests (quota shortfall)" in v["problems"]
    assert G.gate_verdict(builds, results, expected_parts=parts, expected_cases=cases,
                          quota=1)["passed"]
    own = [dict(b, quota=3) if b["part"].startswith("golden:") else b for b in builds]
    assert not G.gate_verdict(own, results, expected_parts=parts, expected_cases=cases)["passed"]
    assert any("no quota" in w for w in verdict()["warnings"])

    # review 3 C2: an HF reference outside img_hfref needs a passing cross-check
    def ref(image, check=None):
        return G.gate_verdict(builds, results, expected_parts=parts, expected_cases=cases,
                              quota=1, reference={"image": image, "cross_check": check})

    assert ref("img_hfref")["passed"]
    assert not ref("img_vllm")["passed"]
    rec = [{"case_id": "c", "index": 0, "hf_ids_digest": "a", "hf_pixels_digest": "p"}]
    good_check = G.reference_cross_check(rec, rec)
    assert good_check == {"passed": True, "compared": 1, "mismatches": [], "missing": []}
    assert ref("img_vllm", good_check)["passed"]
    drift = G.reference_cross_check(rec, [{**rec[0], "hf_pixels_digest": "q"}])
    assert drift["mismatches"] == ["c#0"] and not ref("img_vllm", drift)["passed"]
    assert not G.reference_cross_check(rec, [])["passed"]
    assert any("C2" in w for w in verdict()["warnings"])


def test_results_provenance():
    calls = []

    def git(args, cwd):
        calls.append((tuple(args), cwd))
        return {"rev-parse": "a" * 40, "status": " M src/x.py"}[args[0]]

    p = G.results_provenance(cwd="/repo", git=git, fingerprint=lambda n: f"fp-{n}")
    assert p == {"git_sha": "a" * 40, "git_dirty": True,
                 "image_fingerprints": {"img_vllm": "fp-img_vllm"}}
    assert all(cwd == "/repo" for _, cwd in calls)
    clean = G.results_provenance(git=lambda a, c: "" if a[0] == "status" else "b" * 40,
                                 fingerprint=lambda n: "x")
    assert clean["git_dirty"] is False
    nogit = G.results_provenance(git=lambda a, c: None, fingerprint=lambda n: "x")
    assert nogit["git_sha"] is None and nogit["git_dirty"] is None
    real = G.results_provenance(git=lambda a, c: None)  # the real image fingerprint
    from flora_deploy.modal_app.images import image_fingerprint

    assert real["image_fingerprints"] == {"img_vllm": image_fingerprint("img_vllm")}


def test_pixel_digests_catch_frame_order_and_resample():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    # two videos: (T=2, H=2, W=2) and (T=1, H=2, W=2) -> 8 + 4 rows of patch vectors
    grids = [[2, 2, 2], [1, 2, 2]]
    hf = torch.rand(12, 6, generator=g) * 2 - 1
    digests = G.video_pixel_digests(hf, grids)
    assert [d["rows"] for d in digests] == [8, 4]
    v1, v2 = hf[:8].clone(), hf[8:].clone()
    assert G.pixels_equal_per_video(digests, [v1, v2]) == [True, True]
    assert G.pixels_equal_per_video(digests, [v1.bfloat16(), v2.bfloat16()]) == [True, True]
    # frames in the wrong order (temporal groups swapped): same shape, same grid, FAIL
    swapped = torch.cat([v1[4:], v1[:4]])
    assert G.pixels_equal_per_video(digests, [swapped, v2]) == [False, True]
    # videos swapped across items
    assert G.pixels_equal_per_video(digests, [v2, v1]) == [False, False]
    # a different resample filter: values move by more than a bf16 step
    assert G.pixels_equal_per_video(digests, [v1, (v2 + 0.02).bfloat16()]) == [True, False]
    assert G.pixels_equal_per_video(digests, [v1.half(), v2]) == [False, True]  # other dtype
    assert G.pixels_equal_per_video(digests, [v1]) == [False, False]
    with pytest.raises(ValueError, match="pixel rows"):
        G.video_pixel_digests(hf[:10], grids)


def test_report_and_verdicts():
    grids = [[2, 4, 4]]
    hf = expanded(grids, [[0.25, 1.25]])
    rows = []
    for case, ids in (("a", hf), ("b", hf[:-1])):
        for idx in (0, 1):
            rows.append(G.compare(case_id=case, index=idx, source="syn", variant="wrapped",
                                  hf_ids=hf, hf_grids=grids,
                                  vllm_ids=ids if idx else hf, vllm_grids=grids, sp=SP,
                                  merge_size=2, pixels_equal=[True]).to_dict())
        rows.append(G.compare(case_id=case, index=0, source="syn", variant="bare", hf_ids=hf,
                              hf_grids=grids, vllm_ids=None, vllm_grids=None, sp=SP,
                              merge_size=2, error="RuntimeError: 0 placeholders").to_dict())
    for r in rows:  # stand-ins for the generate and placeholder checks
        if r["variant"] == "wrapped" and r["ids_equal"]:
            r["passed"] = True
    assert G.case_verdicts(rows, "wrapped") == {"a": True, "b": False}
    assert G.case_verdicts(rows, "bare") == {"a": False, "b": False}
    md = G.render_report(rows)
    head = md.splitlines()[0]
    assert head == "| case | wrapped | bare |"
    assert "| a | PASS (2/2) | FAIL (0/1) |" in md
    assert "| b | FAIL (1/2) | FAIL (0/1) |" in md
    assert "RuntimeError: 0 placeholders" in md


def test_detail_table_is_well_formed():
    grids = [[1, 2, 2]]
    hf = expanded(grids, [[0.0]])
    row = G.compare(case_id="a", index=0, source="syn", variant="wrapped", hf_ids=hf,
                    hf_grids=grids, vllm_ids=hf, vllm_grids=grids, sp=SP,
                    merge_size=2).to_dict()
    detail = G.render_report([row]).split("\n\n", 1)[1].splitlines()
    assert detail[1] == "|" + "---|" * 15
    assert {line.count("|") for line in detail} == {16}
    assert "| = | = | = | - | - | - |" in detail[2]  # pixels/gen/placeholders not given
    row = G.compare(case_id="a", index=0, source="syn", variant="wrapped", hf_ids=hf,
                    hf_grids=grids, vllm_ids=hf, vllm_grids=grids, sp=SP, merge_size=2,
                    pixels_equal=[False], generate_ids_equal=True,
                    placeholders=ranges(grids, [[0.0]])).to_dict()
    assert "| = | = | = | ≠ | = | = |" in G.render_report([row])
