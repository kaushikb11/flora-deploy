"""``flora prepare`` builds the derived weights the config's preset needs (``prep/derived.py``),
with the Volume and ``flora-convert`` faked out: nothing runs on Modal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib import resources
from pathlib import Path

import modal
import pytest
import yaml
from typer.testing import CliRunner

from flora_deploy.cli import prepare as cli
from flora_deploy.config import receipts as rc
from flora_deploy.config.pins import DERIVED_PINS_BY_KEY, PINS_BY_KEY
from flora_deploy.config.schema import loads_yaml
from flora_deploy.prep import derived as dv

runner = CliRunner()
TEMPLATE = "exact_mp4_dir.yaml"


def receipt_bytes(name: str, **changes) -> bytes:
    b = dv.DERIVED_BUILDS[name]
    doc: dict = {"ok": True, "kind": b.dirname, "created_utc": "t"}
    for path, want in b.reuse_checks:
        node = doc
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = want
    doc.update(changes)
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


def balanced_cfg(tmp_path: Path) -> Path:
    """The shipped template text (comments and all) switched to the balanced preset."""
    text = (resources.files("flora_deploy.config") / "templates" / TEMPLATE).read_text()
    text = text.replace("preset: exact", "preset: balanced", 1)
    text = text.replace("  segcap: {max_containers: 2}", "  segment: {max_containers: 1}", 1)
    p = tmp_path / "flora.yaml"
    p.write_text(text)
    return p


class Volume:
    def __init__(self, files: dict[str, bytes] | None = None):
        self.files = dict(files or {})
        self.reads: list[str] = []

    def read(self, path: str) -> bytes | None:
        self.reads.append(path)
        return self.files.get(path)


class Builder:
    """Stands in for ``_run_derived_remote``: 'builds' by writing a receipt to the Volume."""

    def __init__(self, vol: Volume, receipts: dict[str, bytes]):
        self.vol, self.receipts, self.calls = vol, receipts, []

    def __call__(self, names, *, force=False, quiet=False):
        self.calls.append((list(names), force))
        out = {}
        for n in names:
            raw = self.receipts[n]
            self.vol.files[dv.DERIVED_BUILDS[n].receipt_path()] = raw
            out[n] = json.loads(raw)
        return out


@pytest.fixture
def fakes(monkeypatch):
    vol = Volume()
    built = Builder(vol, {n: receipt_bytes(n) for n in dv.DERIVED_BUILDS})
    monkeypatch.setattr(cli, "_read_weights_file", vol.read)
    monkeypatch.setattr(cli, "_run_derived_remote", built)

    def no_snapshots(*a, **k):  # pragma: no cover - a test failure if reached
        raise AssertionError("snapshots must not be prepared with --only-derived")

    monkeypatch.setattr(cli, "_run_prepare_remote", no_snapshots)
    monkeypatch.setattr(cli, "_hf_api", no_snapshots)
    return vol, built


def pin_to(monkeypatch, name: str, raw: bytes) -> None:
    """Pin ``name``'s receipt to ``raw`` (the real pin is the maintainer's own build)."""
    key = rc.pin_key(name)
    pins = dict(DERIVED_PINS_BY_KEY)
    pins[key] = replace(pins[key], receipt_sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(rc, "DERIVED_PINS_BY_KEY", pins)


# --- the plan ------------------------------------------------------------------------------


def test_builds_match_convert_app():
    from flora_deploy.modal_app import convert_app as ca

    cap, ts = dv.DERIVED_BUILDS["captioner_bf16"], dv.DERIVED_BUILDS["tempseg_merged"]
    assert (cap.cpu, cap.memory_mib) == (ca.CONVERT_CPU, ca.CONVERT_MEMORY_MIB)
    assert (ts.cpu, ts.memory_mib) == (ca.MERGE_CPU, ca.MERGE_MEMORY_MIB)
    for b in (cap, ts):
        assert isinstance(getattr(ca, b.function), modal.Function), b.function
    rev = PINS_BY_KEY["captioner"].revision
    assert cap.receipt_path() == f"/derived/captioner-bf16@{rev}/derived_receipt.json"
    assert "/" + cap.receipt_path().lstrip("/") == str(ca._derived()).removeprefix(
        "/weights") + "/derived_receipt.json"


def test_plan_reuses_passing_receipts_and_estimates_the_rest():
    raw = receipt_bytes("captioner_bf16")
    vol = Volume({dv.DERIVED_BUILDS["captioner_bf16"].receipt_path(): raw})
    steps = dv.plan_derived(["captioner_bf16", "tempseg_merged"], vol.read)
    cap, ts = steps
    assert (cap.action, cap.receipt_sha256, cap.est_usd) == (
        "reuse", hashlib.sha256(raw).hexdigest(), 0.0)
    assert ts.action == "build" and ts.reason == "no receipt on the Volume"
    assert 0.01 < ts.est_usd < 0.2  # an 8-core CPU container for a few minutes
    assert dv.summarize(steps)["build"] == ["tempseg_merged"]

    forced = dv.plan_derived(["captioner_bf16"], vol.read, force=True)[0]
    assert forced.action == "rebuild" and forced.est_usd > 0


def test_plan_rebuilds_stale_or_unreadable_receipts():
    path = dv.DERIVED_BUILDS["captioner_bf16"].receipt_path()
    stale = receipt_bytes("captioner_bf16", converter={"version": 0})
    step = dv.plan_derived(["captioner_bf16"], Volume({path: stale}).read)[0]
    assert step.action == "build" and "converter.version" in step.reason

    def boom(path):
        raise RuntimeError("no Modal token")

    step = dv.plan_derived(["captioner_bf16"], boom)[0]
    assert step.action == "build" and "no Modal token" in step.reason


# --- flora prepare --dry-run ---------------------------------------------------------------


def test_dry_run_prints_what_would_be_built_and_the_cost(tmp_path, fakes):
    vol, built = fakes
    cfg = balanced_cfg(tmp_path)
    vol.files[dv.DERIVED_BUILDS["captioner_bf16"].receipt_path()] = receipt_bytes(
        "captioner_bf16")
    res = runner.invoke(cli.app, ["prepare", "--dry-run", "--only-derived", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    out = " ".join(res.output.split())
    assert "Would build: tempseg_merged; estimated cost $0." in out
    assert "reuse" in out and built.calls == []

    res = runner.invoke(cli.app, ["prepare", "--dry-run", "--only-derived", "--json",
                                  "-c", str(cfg)])
    rows = json.loads(res.stdout)
    assert [(r["key"], r["action"]) for r in rows] == [("captioner_bf16", "reuse"),
                                                       ("tempseg_merged", "build")]
    assert rows[1]["est_usd"] > 0 and rows[0]["est_usd"] == 0
    assert "weights" not in loads_yaml(cfg.read_text())  # a dry run writes nothing


def test_dry_run_lists_snapshots_and_derived(tmp_path, monkeypatch, fakes):
    class Api:
        def list_repo_tree(self, repo_id, *, revision, recursive):
            return []

    monkeypatch.setattr(cli, "_hf_api", Api)
    cfg = balanced_cfg(tmp_path)
    res = runner.invoke(cli.app, ["prepare", "--dry-run", "--json", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    keys = [r["key"] for r in json.loads(res.stdout)]
    assert keys[:len(PINS_BY_KEY)] == list(PINS_BY_KEY)
    assert keys[len(PINS_BY_KEY):] == ["captioner_bf16", "tempseg_merged"]


# --- flora prepare -------------------------------------------------------------------------


def test_prepare_builds_and_records_receipts(tmp_path, monkeypatch, fakes):
    vol, built = fakes
    cfg = balanced_cfg(tmp_path)
    before = cfg.read_text()
    pin_to(monkeypatch, "captioner_bf16", built.receipts["captioner_bf16"])
    res = runner.invoke(cli.app, ["prepare", "--only-derived", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    assert built.calls == [(["captioner_bf16", "tempseg_merged"], False)]
    after = cfg.read_text()
    assert after.startswith(before)  # comments and every other line kept
    sha = {n: hashlib.sha256(r).hexdigest() for n, r in built.receipts.items()}
    assert loads_yaml(after)["weights"] == {"derived": sha}
    assert "recorded in" in res.output

    # second run: both receipts pass on the Volume, nothing is built, the file is unchanged
    res = runner.invoke(cli.app, ["prepare", "--only-derived", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    assert len(built.calls) == 1 and cfg.read_text() == after
    assert "already recorded" in res.output

    # --force-derived rebuilds
    runner.invoke(cli.app, ["prepare", "--only-derived", "--force-derived", "-c", str(cfg)])
    assert built.calls[-1] == (["captioner_bf16", "tempseg_merged"], True)


def test_prepare_refuses_a_receipt_that_does_not_match_its_pin(tmp_path, fakes):
    vol, built = fakes
    cfg = balanced_cfg(tmp_path)
    res = runner.invoke(cli.app, ["prepare", "--only-derived", "--json", "-c", str(cfg)])
    assert res.exit_code == 1
    rows = {r["key"]: r for r in json.loads(res.stdout)}
    assert rows["captioner_bf16"]["ok"] is False
    assert "pins.DERIVED_PINS['captioner-bf16']" in rows["captioner_bf16"]["error"]
    assert rows["tempseg_merged"]["ok"] is True  # unpinned: recorded
    assert loads_yaml(cfg.read_text())["weights"]["derived"] == {
        "tempseg_merged": rows["tempseg_merged"]["receipt_sha256"]}


def test_prepare_reports_a_failed_build(tmp_path, monkeypatch, fakes):
    cfg = balanced_cfg(tmp_path)
    monkeypatch.setattr(cli, "_run_derived_remote", lambda names, **kw: {
        n: RuntimeError("container OOM") for n in names})
    res = runner.invoke(cli.app, ["prepare", "--only-derived", "-c", str(cfg)])
    assert res.exit_code == 1
    assert "RuntimeError: container OOM" in res.output
    assert "weights" not in loads_yaml(cfg.read_text())


def test_prepare_skips_derived_after_a_snapshot_failure(tmp_path, monkeypatch, fakes):
    vol, built = fakes
    cfg = balanced_cfg(tmp_path)
    monkeypatch.setattr(cli, "_run_prepare_remote", lambda keys, **kw: [
        {"key": k, "ok": False, "status": "failed", "files": 0, "bytes": 0, "receipt": "",
         "failures": [], "timings_s": {}} for k in keys])
    res = runner.invoke(cli.app, ["prepare", "-c", str(cfg)])
    assert res.exit_code == 1 and built.calls == []
    assert "not building the derived weights" in res.output


@pytest.mark.parametrize("args", [["--skip-derived", "--only-derived"],
                                  ["--skip-derived", "--force-derived"],
                                  ["--only", "tempseg", "--only-derived"]])
def test_conflicting_flags(tmp_path, fakes, args):
    cfg = balanced_cfg(tmp_path)
    res = runner.invoke(cli.app, ["prepare", "-c", str(cfg), *args])
    assert res.exit_code == 2


def test_only_derived_needs_a_config(tmp_path, fakes):
    res = runner.invoke(cli.app, ["prepare", "--only-derived", "-c", str(tmp_path / "no.yaml")])
    assert res.exit_code == 2 and "not found" in res.output


def test_preset_without_derived_weights_builds_nothing(tmp_path, fakes):
    vol, built = fakes
    d = yaml.safe_load((resources.files("flora_deploy.config") / "templates" /
                        "rewards_only.yaml").read_text())
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(yaml.safe_dump(d))
    res = runner.invoke(cli.app, ["prepare", "--only-derived", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    assert built.calls == [] and "needs no derived weights" in res.output
