"""``scripts/cert_preflight.py``: every check against fakes (``gh``, git, Modal, the Volume, the
runbook's CLI). Nothing here reaches the network or Modal."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cert_preflight.py"
spec = importlib.util.spec_from_file_location("cert_preflight", SCRIPT)
cp = importlib.util.module_from_spec(spec)
sys.modules["cert_preflight"] = cp  # dataclasses look the module up
spec.loader.exec_module(cp)

SHA = "a" * 40
EPS = ["human/ds/a", "human/ds/b"]

DRY_RUN = """\
Wave H-lite run 1022130531d5 (human only), plan certified (b -> f -> d -> d2 -> e): spent $0.00 of $10.60 (billing read never)
  episodes: 12 human, 235.9 video s (golden/v2 manifest)
  b. exact oracle, 12 human episodes (golden/v2), nvidia mode, H200  [pending] cap $4.00  nominal $2.59  upper $3.97  rule: ok
      segcap human (12 ep, 235.9 s, cold start)                     H200   1643 s /   2514 s x $5.69/h = $2.595 (upper $3.971)
  f. noise: 5 of (b)'s episodes, same config/GPU/mode                [pending] cap $1.95  nominal $1.06  upper $1.90  rule: ok
  d. balanced via flora deploy/run, 12 episodes                      [pending] cap $2.40  nominal $1.35  upper $2.39  rule: ok
  d2. lab L0 / Gate A (skipped when already done)                    [pending] cap $2.35  nominal $1.74  upper $2.31  rule: ok
  e. flora certify, then wave_h_finalize.py                          [pending] cap $0.00  nominal $0.00  upper $0.00  rule: ok
  lab (d2): runs (tempseg: no balanced-tempseg.json; captioner: no balanced-captioner.json)
  total: nominal $6.75, upper $10.58 (incl. 240 s client slack per app); global cap $10.60 (the sum)
dry run: nothing started (run with --yes-spend after the user approves)
"""  # noqa: E501


@dataclass
class Entry:
    path: str
    size: int


class FakeVolume:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def listdir(self, path, *, recursive=False):
        hits = [Entry(p, len(b)) for p, b in self.files.items() if p.startswith(path + "/")]
        if not hits:
            raise FileNotFoundError(path)
        return hits


def manifest(sizes=None) -> dict:
    sizes = sizes or {}
    return {"episodes": [{"id": e, "path": f"golden/v2/{e}", "status": "staged",
                          "files": {"main": {"file": "main.mp4",
                                             "bytes": sizes.get(e, 4)}}} for e in EPS]}


def golden_volume(**kw) -> FakeVolume:
    files = {f"golden/v2/{e}/main.mp4": b"xxxx" for e in EPS}
    files["golden/v2/manifest.json"] = json.dumps(manifest(**kw)).encode()
    return FakeVolume(files)


class Fake:
    """A command table: the longest key found in the joined argv wins."""

    def __init__(self, table: dict[str, cp.Result]):
        self.table = table
        self.calls: list[str] = []

    def __call__(self, cmd, *, cwd, timeout=300.0, env=None):
        line = " ".join(cmd)
        self.calls.append(line)
        hits = [k for k in self.table if k in line]
        if hits:
            return self.table[max(hits, key=len)]
        return cp.Result(127, "", f"no fake for {line}")


def ok(out="") -> cp.Result:
    return cp.Result(0, out, "")


RECEIPTS = [{"key": "tempseg_merged", "content_sha256": "1" * 64,
             "content_sha256_pinned": "1" * 64, "matches_pin": True, "stale": None},
            {"key": "captioner_bf16", "content_sha256": "2" * 64,
             "content_sha256_pinned": "2" * 64, "matches_pin": True, "stale": None}]


def good_table(**over) -> dict[str, cp.Result]:
    t = {
        "git status --porcelain": ok(""),
        "git rev-parse HEAD": ok(SHA + "\n"),
        "git ls-remote origin": ok(f"{SHA}\trefs/heads/main\n"),
        "gh run list": ok(json.dumps([{"databaseId": 1, "name": "ci", "status": "completed",
                                       "conclusion": "success", "headSha": SHA}])),
        "gh run view": ok(json.dumps({"jobs": [
            {"name": "test", "status": "completed", "conclusion": "success"},
            {"name": "lint", "status": "completed", "conclusion": "success"}]})),
        "wave_h.py --status --plan certified --json": cp.Result(2, "", "unrecognized: --json"),
        "wave_h.py --plan certified --dry-run": ok(DRY_RUN),
        "wave_h.py --status": cp.Result(2, "", "unrecognized: --status"),
        "load_selection": ok(json.dumps(EPS) + "\n"),
        "flora prepare --show-receipt": ok(json.dumps(RECEIPTS)),
        "lock_images.py --check": ok("locks current: img_cpu, img_hfref, img_vllm\n"),
        "modal app list": ok(json.dumps([
            {"app_id": "ap-1", "description": "flora-balanced-lab", "state": "stopped"},
            {"app_id": "ap-2", "description": "plumbline-llm", "state": "deployed"}])),
        "lab_engine_hashes": ok(json.dumps({"tempseg": "t" * 64, "captioner": "c" * 64})),
        "cert_freeze.py --check": ok("ok"),
        "pytest": ok("1 passed"),
        "spend.py": ok("flora-x  $ 20.00\nTOTAL                    $  26.10  of $50.00  "
                       "(left $23.90)\n"),
    }
    t.update(over)
    return t


def make_env(tmp_path, vol=None, **over) -> tuple[cp.Env, Fake]:
    fake = Fake(good_table(**over))
    v = vol or golden_volume()
    return cp.Env(repo=tmp_path, run=fake, volume=lambda name: v), fake


def by_name(checks):
    return {c.name: c for c in checks}


def test_all_good_is_go(tmp_path, capsys):
    env, _ = make_env(tmp_path)
    assert cp.main([], env=env) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "GO"
    assert "[PASS] images" in out


def test_json_output_and_nogo_exit(tmp_path, capsys):
    env, _ = make_env(tmp_path, **{"git status --porcelain": ok(" M x.py\n")})
    assert cp.main(["--json"], env=env) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == "NO-GO"
    git = next(c for c in doc["checks"] if c["name"] == "git")
    assert git["status"] == "fail" and git["fix"]


def test_usage_error_exits_2(tmp_path):
    with pytest.raises(SystemExit) as e:
        cp.main(["--embodiment", "dog"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cp.main(["--repo", str(tmp_path)])
    assert e.value.code == 2


def test_every_fail_and_warn_has_a_fix(tmp_path):
    env, _ = make_env(tmp_path, **{"git status --porcelain": ok(" M x\n"),
                                   "modal app list": cp.Result(1, "", "no token")})
    for c in cp.preflight(env):
        if c.status != cp.PASS:
            assert c.fix, c


# 1. git ----------------------------------------------------------------------------------------

def test_git_not_pushed(tmp_path):
    env, _ = make_env(tmp_path, **{"git ls-remote origin": ok("b" * 40 + "\trefs/heads/main"),
                                   "merge-base": cp.Result(1)})
    c = cp.check_git(env)
    assert c.status == cp.FAIL and "not on origin/main" in c.message


def test_git_behind_origin(tmp_path):
    env, _ = make_env(tmp_path, **{"git ls-remote origin": ok("b" * 40 + "\trefs/heads/main"),
                                   "merge-base": ok()})
    assert "behind" in cp.check_git(env).message


def test_git_ci_failed_job(tmp_path):
    env, _ = make_env(tmp_path, **{"gh run view": ok(json.dumps({"jobs": [
        {"name": "test", "status": "completed", "conclusion": "failure"}]}))})
    c = cp.check_git(env)
    assert c.status == cp.FAIL and "ci/test: failure" in c.message


def test_git_ci_in_progress(tmp_path):
    env, _ = make_env(tmp_path, **{"gh run view": ok(json.dumps({"jobs": [
        {"name": "test", "status": "in_progress", "conclusion": ""}]}))})
    assert "still in_progress" in cp.check_git(env).message


def test_git_ci_other_sha_ignored(tmp_path):
    env, _ = make_env(tmp_path, **{"gh run list": ok(json.dumps([
        {"databaseId": 1, "name": "ci", "status": "completed", "conclusion": "success",
         "headSha": "c" * 40}]))})
    c = cp.check_git(env)
    assert c.status == cp.FAIL and "no CI run" in c.message


def test_git_old_gh_without_commit_flag(tmp_path):
    fake_runs = ok(json.dumps([{"databaseId": 1, "name": "ci", "status": "completed",
                                "conclusion": "success", "headSha": SHA}]))
    env, fake = make_env(tmp_path, **{"gh run list --commit": cp.Result(1, "", "unknown flag"),
                                      "gh run list --limit": fake_runs})
    assert cp.check_git(env).status == cp.PASS
    assert any("--limit" in c for c in fake.calls)


def test_git_without_gh_warns(tmp_path):
    env, _ = make_env(tmp_path, **{"gh run list": cp.Result(127, "", "gh: not found")})
    c = cp.check_git(env)
    assert c.status == cp.WARN and "gh" in c.message


# 2. plan ---------------------------------------------------------------------------------------

def test_dry_run_parse():
    p = cp.parse_dry_run(DRY_RUN)
    assert [s["name"] for s in p.steps] == ["exact_e2e", "noise", "balanced", "lab", "certify"]
    assert (p.upper, p.global_cap, p.embodiment) == (10.58, 10.6, "human")
    assert (p.next_step, p.next_action, p.lab_runs) == ("exact_e2e", "run", True)


def test_plan_stop_rule_is_nogo(tmp_path):
    text = DRY_RUN.replace("[pending] cap $4.00  nominal $2.59  upper $3.97  rule: ok",
                           "[failed] cap $4.00  nominal $2.59  upper $3.97  rule: STOP: "
                           "upper $3.97 > $1.00 left")
    env, _ = make_env(tmp_path, **{"wave_h.py --plan certified --dry-run": ok(text)})
    plan, why = cp.load_plan(env)
    c = cp.check_plan(env, plan, why, "human")
    assert c.status == cp.FAIL and "STOP" in c.message


def test_plan_status_text_decision_wins(tmp_path):
    status = ("Wave H-lite run x, plan certified\n  held: exact_e2e attempt 1 charged $4.00\n"
              "  next: exact_e2e: wait: does not fit; it fits once billing covers the held "
              "attempts, about 2026-10-22 14:15 UTC\n    because exact_e2e attempt 1 killed\n")
    env, _ = make_env(tmp_path, **{"wave_h.py --status": ok(status)})
    plan, why = cp.load_plan(env)
    assert plan.source == "status+dry-run" and plan.next_action == "wait"
    c = cp.check_plan(env, plan, why, "human")
    assert c.status == cp.FAIL and "wait until billing" in c.fix and "held" in c.message


def test_plan_status_json_preferred(tmp_path):
    doc = {"embodiment": "human", "upper": 10.58, "global_cap": 10.6,
           "steps": [{"name": "exact_e2e", "status": "pending", "cap": 4.0, "upper": 3.97}],
           "next": {"step": "exact_e2e", "action": "run", "message": "runs with $4.00"},
           "lab": {"runs": False, "why": "done", "engine_settings_hash": {"tempseg": "t" * 64}}}
    env, fake = make_env(tmp_path, **{
        "wave_h.py --status --plan certified --json": ok(json.dumps(doc))})
    plan, why = cp.load_plan(env)
    assert plan.source == "status-json" and plan.lab_hashes == {"tempseg": "t" * 64}
    assert cp.check_plan(env, plan, why, "human").status == cp.PASS
    assert not any("--dry-run" in c for c in fake.calls)


def test_plan_is_read_for_the_preflights_embodiment(tmp_path):
    env, fake = make_env(tmp_path)
    cp.load_plan(env, "yam")
    wave = [c for c in fake.calls if "wave_h.py" in c]
    assert wave and all(c.endswith("--embodiment yam") for c in wave)


def test_plan_unreadable_and_wrong_embodiment(tmp_path):
    env, _ = make_env(tmp_path, **{
        "wave_h.py --plan certified --dry-run": cp.Result(1, "", "boom")})
    plan, why = cp.load_plan(env)
    assert plan is None and cp.check_plan(env, plan, why, "human").status == cp.FAIL
    p = cp.parse_dry_run(DRY_RUN)
    assert "not yam" in cp.check_plan(env, p, "", "yam").message


def test_plan_upper_over_global_cap(tmp_path):
    env, _ = make_env(tmp_path)
    p = cp.parse_dry_run(DRY_RUN)
    p.global_cap = 9.0
    assert "exceeds the global cap" in cp.check_plan(env, p, "", "human").message


# 3. episodes -----------------------------------------------------------------------------------

def test_episodes_ok(tmp_path):
    env, _ = make_env(tmp_path)
    assert cp.check_episodes(env, "human").status == cp.PASS


def test_episodes_size_mismatch_and_missing(tmp_path):
    vol = golden_volume(sizes={"human/ds/a": 5})
    del vol.files["golden/v2/human/ds/b/main.mp4"]
    env, _ = make_env(tmp_path, vol=vol)
    c = cp.check_episodes(env, "human")
    assert c.status == cp.FAIL and len(c.details["problems"]) == 2


def test_episodes_not_in_manifest(tmp_path):
    env, _ = make_env(tmp_path, **{"load_selection": ok(json.dumps(EPS + ["human/ds/z"]))})
    c = cp.check_episodes(env, "human")
    assert "human/ds/z: not in the manifest" in c.message


def test_episodes_yam_selection_missing(tmp_path):
    env, _ = make_env(tmp_path, **{"load_selection": ok("null\n")})
    c = cp.check_episodes(env, "yam")
    assert c.status == cp.FAIL and "golden/v3-yam" in c.message


def test_episodes_volume_without_version(tmp_path):
    env, _ = make_env(tmp_path, vol=FakeVolume({}))
    assert "no golden/v2" in cp.check_episodes(env, "human").message


# 4. weights ------------------------------------------------------------------------------------

def test_weights_ok_and_mismatch(tmp_path):
    env, _ = make_env(tmp_path)
    assert cp.check_weights(env).status == cp.PASS
    bad = [dict(RECEIPTS[0], matches_pin=False, content_sha256="9" * 64), RECEIPTS[1]]
    env, _ = make_env(tmp_path, **{"flora prepare --show-receipt": ok(json.dumps(bad))})
    c = cp.check_weights(env)
    assert c.status == cp.FAIL and "!= pin" in c.message


def test_weights_unpinned_or_missing(tmp_path):
    rows = [dict(RECEIPTS[0], content_sha256_pinned=None, matches_pin=None),
            {"key": "captioner_bf16", "error": "no receipt on the Volume"}]
    env, _ = make_env(tmp_path, **{"flora prepare --show-receipt": ok(json.dumps(rows))})
    c = cp.check_weights(env)
    assert "not pinned" in c.message and "no receipt" in c.message


# 5. images -------------------------------------------------------------------------------------

def test_images_current_passes(tmp_path):
    env, fake = make_env(tmp_path)
    c = cp.check_images(env)
    assert c.status == cp.PASS and "current" in c.message
    assert [ln for ln in fake.calls if "lock_images" in ln] == [
        "uv run python scripts/lock_images.py --check"]


def test_images_stale_fails(tmp_path):
    env, _ = make_env(tmp_path, **{"lock_images.py --check": cp.Result(
        1, "", "stale lock(s): img_vllm; run uv run python scripts/lock_images.py")})
    c = cp.check_images(env)
    assert c.status == cp.FAIL and "stale" in c.message and "img_vllm" in c.message
    assert "commit the locks" in c.fix


@pytest.mark.parametrize("rc", [2, 124, 127])
def test_images_unresolvable_fails(tmp_path, rc):
    """A compile error, timeout or missing uv is a FAIL (never a silent pass or a warning)."""
    env, _ = make_env(tmp_path, **{"lock_images.py --check": cp.Result(rc, "", "boom")})
    c = cp.check_images(env)
    assert c.status == cp.FAIL and f"exit {rc}" in c.message and "network" in c.fix


# 6. apps ---------------------------------------------------------------------------------------

def test_apps_stray_flora_app_only(tmp_path):
    apps = [{"app_id": "ap-1", "description": "flora-wh-exact", "state": "deployed"},
            {"app_id": "ap-2", "description": "other", "state": "running"}]
    env, _ = make_env(tmp_path, **{"modal app list": ok(json.dumps(apps))})
    c = cp.check_apps(env)
    assert c.status == cp.FAIL and "ap-1" in c.message and "ap-2" not in c.message
    env, _ = make_env(tmp_path)
    assert cp.check_apps(env).status == cp.PASS


def test_apps_reads_modals_real_column_keys(tmp_path):
    apps = [{"App ID": "ap-x", "Description": "flora-st-run", "State": "deployed", "Tasks": "1",
             "Created at": "2026-10-22 10:00:00+00:00", "Stopped at": None}]
    env, _ = make_env(tmp_path, **{"modal app list": ok(json.dumps(apps))})
    c = cp.check_apps(env)
    assert c.status == cp.FAIL and "ap-x" in c.message and "flora-st-run" in c.message
    env, _ = make_env(tmp_path, **{"modal app list": ok(json.dumps([{"Name": "flora-x"}]))})
    assert cp.check_apps(env).status == cp.FAIL  # unreadable rows: never a silent pass


# 7. lab ----------------------------------------------------------------------------------------

def _lab_files(tmp_path, hashes):
    d = tmp_path / "reports" / "wave-h"
    d.mkdir(parents=True, exist_ok=True)
    for k in cp.LAB_KINDS:
        res = {"baseline": {"l0": []}}
        if hashes.get(k):  # the lab's own shape: engine_hashes {"baseline", "plugin"}
            res[cp.LAB_HASHES] = {"baseline": hashes[k], "plugin": "p" * 64}
        (d / f"balanced-{k}.json").write_text(json.dumps(res))


def _skipped_plan():
    p = cp.parse_dry_run(DRY_RUN)
    p.lab_runs, p.lab_why = False, "already done"
    return p


def test_lab_scheduled_passes(tmp_path):
    env, _ = make_env(tmp_path)
    assert cp.check_lab(env, cp.parse_dry_run(DRY_RUN)).status == cp.PASS


def test_lab_hash_equal_passes_and_stale_fails(tmp_path):
    env, _ = make_env(tmp_path)
    _lab_files(tmp_path, {"tempseg": "t" * 64, "captioner": "c" * 64})
    assert cp.check_lab(env, _skipped_plan()).status == cp.PASS
    _lab_files(tmp_path, {"tempseg": "t" * 64, "captioner": "x" * 64})
    c = cp.check_lab(env, _skipped_plan())
    assert c.status == cp.FAIL and "stale" in c.message


def test_lab_hash_absent_is_will_rerun(tmp_path):
    env, _ = make_env(tmp_path)
    _lab_files(tmp_path, {})
    c = cp.check_lab(env, _skipped_plan())
    assert c.status == cp.WARN and "will re-run" in c.message


# 8. freeze -------------------------------------------------------------------------------------

def test_freeze(tmp_path):
    env, fake = make_env(tmp_path)
    assert cp.check_freeze(env).status == cp.PASS
    assert any(c.startswith("nice -n 10") and "test_certificate_drift" in c for c in fake.calls)
    env, _ = make_env(tmp_path, **{"cert_freeze.py --check": cp.Result(1, "drift: x")})
    assert cp.check_freeze(env).status == cp.FAIL
    env, _ = make_env(tmp_path, **{"pytest": cp.Result(1, "1 failed")})
    assert "failed" in cp.check_freeze(env).message


# 9. budget -------------------------------------------------------------------------------------

def test_budget_remaining_and_margin(tmp_path):
    env, _ = make_env(tmp_path)
    p = cp.parse_dry_run(DRY_RUN)
    c = cp.check_budget(env, p, budget=50.0, margin=2.0)
    assert c.status == cp.PASS and c.details["remaining_after_upper"] == 13.32
    c = cp.check_budget(env, p, budget=38.0, margin=2.0)
    assert c.status == cp.FAIL and "remaining" in c.message


def test_budget_billing_unreadable(tmp_path):
    env, _ = make_env(tmp_path, **{"spend.py": cp.Result(1, "", "no token")})
    assert cp.check_budget(env, cp.parse_dry_run(DRY_RUN), budget=50, margin=2).status == cp.FAIL


def test_a_crashing_check_is_a_fail(tmp_path):
    env, _ = make_env(tmp_path)
    env.volume = lambda name: (_ for _ in ()).throw(RuntimeError("x"))
    checks = by_name(cp.preflight(env))
    assert checks["episodes"].status == cp.FAIL
