"""The second cold README walk (newcomer report 2, 2026-10-22): each friction point, as a check.

`flora index --local` before `flora deploy` (no Volume yet), raw tracebacks, `flora export` from
the demo, the stale `examples/lerobot_small`, silent local runs on real data, and the small
items (`--version`, hash-heavy paths, run-output jargon, the boto3 hint, the pandas setup)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app, main

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
EXAMPLE = ROOT / "examples" / "lerobot_small"
runner = CliRunner()

# run-output words a newcomer cannot read (report 2, F2)
RUN_JARGON = re.compile(r"dead=|problems=|excused|duplicates resolved|python failures|"
                        r"\{'success'|oracle")


@contextlib.contextmanager
def in_dir(path: Path) -> Iterator[Path]:
    from flora_deploy import sdk
    from tests.fakes.flora_backend import FakeBackend

    old, env = Path.cwd(), os.environ.pop("FLORA_STORE", None)
    sdk.set_backend(FakeBackend(path / "remote-stores"))
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old)
        sdk.set_backend(None)
        if env is not None:
            os.environ["FLORA_STORE"] = env


def _ok(argv, columns="100"):
    r = runner.invoke(app, argv, env={"COLUMNS": columns})
    assert r.exit_code == 0, r.output
    return r.output


# ---------------------------------------------------------------------------------------------
# F-N1 / F-N2: `flora index --local` before `flora deploy`


class _Volumes:
    """``modal.Volume.from_name`` over in-memory fakes; records each call's kwargs."""

    def __init__(self):
        from tests.fakes.client_volume import FakeClientVolume

        self.calls: list[tuple[str, dict]] = []
        self.vols: dict[str, FakeClientVolume] = {}
        self._cls = FakeClientVolume

    def from_name(self, name, **kw):
        self.calls.append((name, kw))
        return self.vols.setdefault(name, self._cls())


def test_index_local_creates_the_sink_volume(tmp_path, monkeypatch):
    import modal

    from flora_deploy.cli.init import template_text
    from flora_deploy.local import sample_dataset

    vols = _Volumes()
    monkeypatch.setattr(modal.Volume, "from_name", vols.from_name)
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(template_text("local_demo"))  # sink: volume://flora-results/local-demo
    sample_dataset(tmp_path / "videos", episodes=2)
    out = _ok(["index", "--local", "-c", str(cfg)])
    assert "next: flora plan" in out
    assert vols.calls and all(kw.get("create_if_missing") for _n, kw in vols.calls), vols.calls
    assert any(p.endswith("index/summary.json") for p in vols.vols["flora-results"].files)


def test_index_local_resolves_a_relative_work_dir(tmp_path, monkeypatch):
    """A relative --work reached the Hub download as a relative path (SameFileError)."""
    from flora_deploy.cli.init import template_text
    from flora_deploy.jobs import indexing

    seen = {}

    def stop(spec, *, work=None, **kw):
        seen["work"] = work
        raise ValueError("stop here")

    monkeypatch.setattr(indexing, "open_dataset", stop)
    cfg = tmp_path / "flora.yaml"
    cfg.write_text(template_text("balanced_lerobot").replace(
        "volume://flora-results/", f"file://{tmp_path}/store/"))
    monkeypatch.chdir(tmp_path)
    with contextlib.suppress(ValueError):
        runner.invoke(app, ["index", "--local", "-c", str(cfg), "--work", "./work"],
                      catch_exceptions=False)
    assert Path(seen["work"]).is_absolute() and seen["work"] == tmp_path.resolve() / "work"


def _main(monkeypatch, capsys, exc):
    import flora_deploy.cli as cli

    def boom(**kw):
        raise exc

    monkeypatch.setattr(cli, "app", boom)
    monkeypatch.delenv("FLORA_DEBUG", raising=False)
    with pytest.raises(SystemExit) as e:
        main([])
    return e.value.code, capsys.readouterr().err


def test_expected_conditions_print_one_line_not_a_traceback(monkeypatch, capsys):
    from modal.exception import AuthError, ConnectionError, NotFoundError

    code, err = _main(monkeypatch, capsys, NotFoundError(
        "Volume 'flora-results' not found in environment 'main'"))
    assert code == 2 and "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert "modal volume create flora-results" in err and "internal error" not in err
    code, err = _main(monkeypatch, capsys, AuthError("Token missing"))
    assert code == 2 and "Traceback" not in err and "modal setup" in err
    code, err = _main(monkeypatch, capsys, ConnectionError("timed out"))
    assert code == 2 and "Traceback" not in err and "network" in err
    code, err = _main(monkeypatch, capsys, RuntimeError("a bug"))  # bugs keep the traceback
    assert code == 2 and "Traceback" in err and "internal error" in err


# ---------------------------------------------------------------------------------------------
# F-N3 / F-N4: `flora export` from the demo


def test_export_on_the_mp4_demo_says_what_it_supports(tmp_path):
    with in_dir(tmp_path):
        _ok(["init", "--template", "local_demo"])
        _ok(["run", "--local", "--fake-models", "--sample-data"])
        r = runner.invoke(app, ["export", "--to", "lerobot", "--out", "labeled", "--allow-fake"],
                          env={"COLUMNS": "200"})
    assert r.exit_code == 2
    assert "LeRobot datasets (v2.1 and v3.0) only" in r.output
    assert "flora init -t local_demo_lerobot" in r.output and "--format parquet" in r.output


def test_lerobot_demo_can_be_exported(tmp_path):
    with in_dir(tmp_path):
        out = _ok(["init", "--template", "local_demo_lerobot"])
        assert "flora run --local --fake-models --sample-data" in out
        _ok(["run", "--local", "--fake-models", "--sample-data"])
        r = runner.invoke(app, ["export", "--to", "lerobot", "--out", "labeled"],
                          env={"COLUMNS": "200"})
        assert r.exit_code == 2 and "--allow-fake" in r.output
        out = _ok(["export", "--to", "lerobot", "--out", "labeled", "--allow-fake"])
    assert "6/6 episodes labeled" in out
    info = json.loads((tmp_path / "labeled/meta/info.json").read_text())
    assert info["flora"]["stages"]


def test_readme_export_flags_are_real_and_name_allow_fake():
    help_ = runner.invoke(app, ["export", "--help"], env={"COLUMNS": "200"}).output
    sec = README.split("Training on LeRobot?", 1)[1].split("\n\n", 1)[0]
    spans = re.findall(r"`(flora export [^`]+|--[a-z-]+)`", sec)  # export's own flags
    for flag in {f for sp in spans for f in re.findall(r"--[a-z][a-z-]+", sp)}:
        assert flag in help_, f"README names `flora export {flag}`, which does not exist"
    assert "--allow-fake" in sec and "local_demo_lerobot" in sec
    assert "allow_uncertified` doesn't apply" in sec


# ---------------------------------------------------------------------------------------------
# F-N5: examples/lerobot_small


def test_lerobot_small_cost_is_the_planners_and_its_sink_a_volume():
    from flora_deploy.cli.init import DEFAULT_TEMPLATE, template_text
    from flora_deploy.config.schema import load_config, loads_config

    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Estimated cost: \$(\d+\.\d\d) \(\$(\d+\.\d\d)-\$(\d+\.\d\d)\)\*\*.*?"
                  r"`flora plan (--video-hours \S+ --episodes \d+)`", readme, re.DOTALL)
    assert m, "examples/lerobot_small/README.md: no generated cost line"
    r = runner.invoke(app, ["plan", "-c", str(EXAMPLE / "flora.yaml"), *m.group(4).split(),
                            "--json"])
    d = json.loads(r.output)
    o = d["options"][d["recommended"] - 1]
    assert (f"{o['cost_usd']:.2f}", f"{o['cost_low_usd']:.2f}", f"{o['cost_high_usd']:.2f}") \
        == m.groups()[:3], "stale: rerun the README's `flora plan` and paste its numbers"
    cfg = load_config(EXAMPLE / "flora.yaml")
    default = loads_config(template_text(DEFAULT_TEMPLATE), source="t").output.sink
    assert cfg.output.sink.split("://")[0] == default.split("://")[0] == "volume"
    assert "my-bucket" not in (EXAMPLE / "flora.yaml").read_text()


def test_doctor_does_not_fail_on_the_example(tempseg_pinned):
    from flora_deploy.cli import doctor as d
    from tests.unit.test_doctor import FakeProbes

    checks = d.run_checks(EXAMPLE / "flora.yaml", FakeProbes(volume="missing"))
    assert not [c for c in checks if c.status == "fail"], checks


def test_plan_camera_flags_do_not_contradict_or_cite_files():
    r = runner.invoke(app, ["plan", "-c", str(EXAMPLE / "flora.yaml"), "--video-hours", "0.11",
                            "--episodes", "50", "--json"])
    flags = json.loads(r.output)["options"][0]["flags"]
    scaled = next(f for f in flags if f.startswith("camera scaling"))
    assert "scripts/" not in scaled and ".json" not in scaled and "docs/" not in scaled
    for f in flags:
        if f.startswith("no camera scaling"):
            assert "reward.terminal" in f and "segment" not in f, f


# ---------------------------------------------------------------------------------------------
# F-N6: local runs say what they are doing


def test_local_run_prints_progress_and_no_jargon(tmp_path, monkeypatch):
    from flora_deploy.local.runner import LocalRunner

    monkeypatch.setattr(LocalRunner, "progress_every_s", 0.0)
    with in_dir(tmp_path):
        _ok(["init", "--template", "local_demo"])
        out = _ok(["run", "--local", "--fake-models", "--sample-data"], columns="200")
    assert "index: 6 episodes in 6 source files" in out
    assert re.search(r"started segcap shard \(\d episodes\)", out)
    assert re.search(r"finished reward\.dense shard \(6 episodes\): \d/5 shards done", out)
    assert re.search(r"running \dm\d\ds: segcap \d/6, .*episodes started; \d/5 shards done", out)
    assert "all 5 shards finished" in out and "6/6 episodes labelled" in out
    hits = sorted({m.group(0) for m in RUN_JARGON.finditer(out)})
    assert not hits, f"run output jargon {hits}:\n{out}"


def test_fake_decode_cache_is_bounded_by_bytes(tmp_path, monkeypatch):
    """By count alone, 64 decoded real episodes (220 MB each) got the run killed."""
    from flora_deploy.local import engines as E
    from flora_deploy.local import sample_dataset

    root = sample_dataset(tmp_path / "v", episodes=4)
    one = E.decode(str(root / "ep000.mp4")).frames.nbytes
    monkeypatch.setattr(E, "_CACHE", type(E._CACHE)())
    monkeypatch.setattr(E, "_CACHE_MAX_BYTES", int(one * 1.5))
    for i in range(4):
        E.decode(str(root / f"ep{i:03d}.mp4"))
    assert sum(d.frames.nbytes for d in E._CACHE.values()) <= max(one * 1.5, one * 2)
    assert len(E._CACHE) <= 2


# ---------------------------------------------------------------------------------------------
# small items


def test_version_flag():
    from flora_deploy import __version__

    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and r.output.strip() == f"flora-deploy {__version__}"


def test_report_shortens_hashes_and_json_keeps_them(tmp_path):
    with in_dir(tmp_path):
        _ok(["init", "--template", "local_demo"])
        _ok(["run", "--local", "--fake-models", "--sample-data"])
        text = _ok(["report"], columns="300")
        js = _ok(["report", "--json"])
    assert not re.search(r"[0-9a-f]{13,}", text), text
    assert re.search(r"[0-9a-f]{12}…", text) and "--json" in text
    assert "errors: none" in text and "certification: none (fake models)" in text
    assert re.search(r"[0-9a-f]{64}", js)
    assert not RUN_JARGON.search(text), text


def test_boto3_hint_matches_the_install(monkeypatch):
    from flora_deploy.cli import doctor as d

    monkeypatch.setitem(sys.modules, "boto3", None)  # import boto3 -> ImportError
    msg = d._probe_bucket("s3", "b", "", "k", b"x")
    assert "uv tool install" in msg and "--with boto3" in msg
    assert "uv tool install" in README  # the install it refers to


def test_readme_pandas_snippet_says_what_to_write_first():
    sec = README.split("## Using the labels", 1)[1]
    before, code = sec.split("```python", 1)
    assert "flora results caption --format parquet --out\ncaption.parquet" in before
    assert "pip install pandas pyarrow" in before
    assert 'read_parquet("caption.parquet")' in code
