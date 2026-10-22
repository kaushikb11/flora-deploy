"""`flora doctor`: every check's pass / warn / fail path, with fake probes (no Modal, no Hub)."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from flora_deploy.cli import app
from flora_deploy.cli import doctor as d
from flora_deploy.config.pins import PINS_BY_KEY, RUNTIME
from flora_deploy.config.schema import load_config
from flora_deploy.layout import RECEIPT_NAME, snapshot_dirname

EXAMPLE_FILE = Path(__file__).resolve().parents[2] / "examples" / "lerobot_small" / "flora.yaml"
ALL_OK_SINK = "s3://my-bucket/flora/lerobot-small"
# The suite runs on the example with an S3 sink, so the bucket checks are exercised (the example
# itself ships a Volume sink: test_newcomer2_dx).
EXAMPLE = Path(tempfile.mkdtemp(prefix="doctor-example-")) / "flora.yaml"
EXAMPLE.write_text(EXAMPLE_FILE.read_text().replace("volume://flora-results/lerobot-small",
                                                    ALL_OK_SINK, 1))
SNAPSHOTS = [snapshot_dirname(p) for p in PINS_BY_KEY.values()]


@dataclass
class FakeProbes:
    version: str | None = RUNTIME["modal_client"]
    token: bool = True
    environment: str = "ok"
    secret: str = "ok"
    volume: str = "ok"
    snapshots: list[str] | None = field(default_factory=lambda: list(SNAPSHOTS))
    receipts: bool = True
    hf: str | None = "hf_x"
    repos: dict[str, str] = field(default_factory=dict)  # repo -> outcome (default ok)
    write: str = "ok"
    files: int | None = 3
    certs: dict[str, str] | None = None
    calls: list[tuple] = field(default_factory=list)

    def modal_version(self):
        return self.version

    def modal_token(self):
        return self.token

    def modal_environment(self, name):
        return self.environment

    def modal_secret(self, name, environment):
        self.calls.append(("secret", name, environment))
        return self.secret

    def modal_volume(self, name, environment):
        return self.volume

    def volume_listdir(self, name, environment, path):
        if path == "/snapshots":
            return self.snapshots
        return [RECEIPT_NAME] if self.receipts else []

    def hf_token(self):
        return self.hf

    def hf_repo(self, repo, revision, token, repo_type):
        self.calls.append(("hf", repo, revision, repo_type))
        return self.repos.get(repo, "ok")

    def store_write(self, sink, environment):
        self.calls.append(("write", sink))
        return self.write

    def local_files(self, path, pattern):
        return self.files

    def certificate_status(self, cfg):
        return self.certs


def by_name(checks):
    return {c.name: c for c in checks}


def run(probes, config=EXAMPLE):
    return by_name(d.run_checks(config, probes))


def write_cfg(tmp_path: Path, **edits) -> Path:
    text = EXAMPLE.read_text()
    for old, new in edits.items():
        assert old in text, old
        text = text.replace(old, new)
    p = tmp_path / "flora.yaml"
    p.write_text(text)
    return p



def test_all_pass_on_the_example(tempseg_pinned):
    p = FakeProbes()
    checks = d.run_checks(EXAMPLE, p)
    fails = [c for c in checks if c.status == "fail"]
    assert not fails, fails
    c = by_name(checks)
    assert c["config"].status == "pass"
    assert c["modal.client"].status == c["modal.auth"].status == "pass"
    assert c["store.secret"].status == "pass" and c["store.write"].status == "pass"
    assert c["weights.snapshots"].status == "pass"
    assert c["input"].status == "pass"
    # every pinned nvidia/Flora-* repo at its pinned revision, as a model; the input as a dataset
    flora = {(pin.repo, pin.revision) for pin in PINS_BY_KEY.values()
             if pin.repo.startswith("nvidia/Flora-")}
    assert {(r, v) for k, r, v, t in (x for x in p.calls if x[0] == "hf") if t == "model"} == flora
    assert ("hf", "lerobot/aloha_static_cups_open", "d793c969cf716001dcca18a0842c3d7e9de9e41b",
            "dataset") in p.calls
    assert ("secret", "flora-bucket", "main") in p.calls


def test_config_missing_invalid_and_license(tmp_path):
    c = run(FakeProbes(), tmp_path / "nope.yaml")
    assert c["config"].status == "fail" and "flora init" in c["config"].fix
    assert c["rest"].status == "warn"  # dependent checks skipped, Modal/HF still checked
    assert "modal.client" in c and "hf.token" in c

    p = write_cfg(tmp_path, **{"license_ack: research-only": ""})
    c = run(FakeProbes(), p)
    assert c["config"].status == "fail" and "license_ack" in c["config"].fix

    p = write_cfg(tmp_path, **{"allow_uncertified: true": "allow_uncertified: false"})
    c = run(FakeProbes(), p)
    assert c["config"].status == "fail" and "allow_uncertified" in c["config"].fix


@pytest.mark.parametrize("probes,name,status,fix", [
    (FakeProbes(version=None), "modal.client", "fail", "pip install modal==1.6.2"),
    (FakeProbes(version="1.5.0"), "modal.client", "fail", "pip install modal==1.6.2"),
    (FakeProbes(token=False), "modal.auth", "fail", "modal setup"),
    (FakeProbes(environment="unauthenticated"), "modal.auth", "fail", "modal token new"),
    (FakeProbes(environment="missing"), "modal.environment", "fail", "modal environment create"),
    (FakeProbes(environment="error:Timeout"), "modal.auth", "warn", "network"),
])
def test_modal_paths(probes, name, status, fix):
    c = run(probes)[name]
    assert c.status == status and fix in c.fix


@pytest.mark.parametrize("outcome,status,word", [
    ("gated", "fail", "request access"),
    ("revision_missing", "fail", "update flora-deploy"),
    ("repo_missing", "fail", "access"),
    ("unauthorized", "fail", "token"),
    ("error:ConnectionError", "warn", "network"),
])
def test_hf_repo_paths(outcome, status, word):
    c = run(FakeProbes(repos={"nvidia/Flora-Captioner": outcome}))
    assert c["hf.captioner"].status == status and word in c["hf.captioner"].fix
    assert c["hf.tempseg"].status == "pass"


def test_hf_no_token_is_a_warning_and_probes_anonymously():
    # review 8b M1: the nvidia/Flora-* repos are not gated, so no token is not a failure.
    p = FakeProbes(hf=None)
    c = run(p)
    assert c["hf.token"].status == "warn" and "rate limits" in c["hf.token"].message
    assert c["hf.captioner"].status == "pass"  # still checked, anonymously
    c = run(FakeProbes(hf=None, repos={"nvidia/Flora-Captioner": "no_token"}))
    assert c["hf.captioner"].status == "fail" and "HF_TOKEN" in c["hf.captioner"].fix


class _FakeApi:
    def __init__(self, auth_exc=None, info_exc=None):
        self.auth_exc, self.info_exc, self.calls = auth_exc, info_exc, []

    def model_info(self, repo, revision=None, token=None):
        self.calls.append(("info", repo, revision, token))
        if self.info_exc:
            raise self.info_exc

    dataset_info = model_info

    def auth_check(self, repo, repo_type=None, token=None):
        self.calls.append(("auth", repo, repo_type, token))
        if self.auth_exc:
            raise self.auth_exc


def _gated_error():
    import httpx
    from huggingface_hub import errors as hfe

    req = httpx.Request("GET", "https://huggingface.co/api/models/x/auth-check")
    return hfe.GatedRepoError("gated", response=httpx.Response(403, request=req))


@pytest.mark.parametrize("token,want", [(None, "no_token"), ("hf_x", "gated")])
def test_real_hf_probe_detects_gated_repos(monkeypatch, token, want):
    # review 8b M1: ``model_info`` succeeds on a gated repo without access; ``auth_check``
    # is what tests access.
    import huggingface_hub

    api = _FakeApi(auth_exc=_gated_error())
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    assert d.RealProbes().hf_repo("meta-llama/x", "a" * 40, token, "model") == want
    assert any(c[0] == "auth" for c in api.calls)


def test_real_hf_probe_ok_when_both_pass(monkeypatch):
    import huggingface_hub

    api = _FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    assert d.RealProbes().hf_repo("nvidia/Flora-X", "a" * 40, None, "model") == "ok"
    assert ("info", "nvidia/Flora-X", "a" * 40, False) in api.calls
    assert ("auth", "nvidia/Flora-X", "model", False) in api.calls


def test_real_certificate_status_for_balanced():
    # review 8b M2: certkey.status needs (preset, pools, **lookup_context); it crashed.
    cfg = load_config(EXAMPLE)
    st = d.RealProbes().certificate_status(cfg)
    assert st is not None and set(st) == set(cfg.pools())
    assert all(isinstance(v, str) and "{" not in v for v in st.values())
    c = by_name(d.check_certificates(d.RealProbes(), cfg))
    assert not any(x.name == "doctor" for x in c.values())
    named = {k for x in c for k in x.removeprefix("certificates.").split(",")}
    assert named == set(cfg.pools())  # every kind, one line per distinct status


@pytest.mark.parametrize("probes,name,status", [
    (FakeProbes(secret="missing"), "store.secret", "fail"),
    (FakeProbes(secret="error:x"), "store.secret", "warn"),
    (FakeProbes(write="skip:no bucket credentials on this machine"), "store.write", "warn"),
    (FakeProbes(write="error:AccessDenied"), "store.write", "fail"),
    (FakeProbes(write="missing"), "store.write", "fail"),
])
def test_store_bucket_paths(probes, name, status):
    assert run(probes)[name].status == status


def test_store_volume_paths(tmp_path):
    p = write_cfg(tmp_path, **{ALL_OK_SINK: "volume://flora-results/lerobot-small"})
    probes = FakeProbes()
    c = run(probes, p)
    assert c["store.volume"].status == "pass" and "store.secret" not in c
    assert ("write", "volume://flora-results/lerobot-small") in probes.calls
    c = run(FakeProbes(volume="missing"), p)  # `flora deploy` creates it: nothing to do
    assert c["store.volume"].status == "pass" and "flora deploy" in c["store.volume"].message
    assert "modal volume create" in c["store.volume"].fix
    assert "store.write" not in c


def test_store_local_probe_writes_and_deletes(tmp_path):
    out = tmp_path / "sink"
    out.mkdir()
    real = d.RealProbes()
    assert real.store_write(f"file://{out}", "main") == "ok"
    assert [p for p in out.rglob("*") if p.is_file()] == []
    assert real.store_write(f"file://{tmp_path / 'absent'}", "main") == "missing"


def test_weights_paths():
    c = run(FakeProbes(snapshots=None))
    assert c["weights.snapshots"].status == "fail" and "flora prepare" in c["weights.snapshots"].fix
    c = run(FakeProbes(snapshots=SNAPSHOTS[:1]))
    assert c["weights.snapshots"].status == "fail"
    c = run(FakeProbes(receipts=False))
    assert c["weights.snapshots"].status == "fail"


def _cfg_with_weights(derived):
    cfg = load_config(EXAMPLE)
    return SimpleNamespace(**{k: getattr(cfg, k) for k in ("modal", "preset", "pipeline")},
                           weights=SimpleNamespace(derived=derived))


def test_weights_derived_paths():
    c = run(FakeProbes())  # the example records no weights.derived yet
    assert c["weights.derived"].status == "warn" and "flora prepare" in c["weights.derived"].fix
    ok = by_name(d.check_weights(FakeProbes(), _cfg_with_weights(
        {"tempseg_merged": "a" * 64, "captioner_bf16": "b" * 64})))
    assert ok["weights.derived"].status == "pass"
    bad = by_name(d.check_weights(FakeProbes(), _cfg_with_weights({"captioner_bf16": "b" * 64})))
    assert bad["weights.derived"].status == "fail" and "tempseg_merged" in \
        bad["weights.derived"].message


def test_certificates_paths(tmp_path):
    # balanced, no certificates exist, allow_uncertified: true (the README's config): the
    # expected state, said once, not a warning per shard kind
    c = run(FakeProbes())
    certs = [k for k in c if k.startswith("certificates") and "reward" not in k]
    assert len(certs) == 1 and c[certs[0]].status == "pass"
    assert "certified: false" in c[certs[0]].message
    # the rewards run NVIDIA's own engine, as exact does: the reference, no certificate needed
    assert c["certificates.reward.terminal"].status == "pass"
    c = run(FakeProbes(certs={"segment": "certified", "caption": "missing"}))
    assert c["certificates.segment"].status == "pass"
    assert c["certificates.caption"].status == "pass"  # missing, but allowed
    exact = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/" \
        "exact_lerobot.yaml"
    assert run(FakeProbes(), exact)["certificates"].status == "pass"


def test_budget_prints_environment_budget():
    c = run(FakeProbes())["budget"]
    assert c.status == "pass" and "$15.00" in c.message
    assert "$18.00" in c.fix and "'main'" in c.fix  # 1.2 x budget_usd


@pytest.mark.parametrize("outcome,status", [
    ("ok", "pass"), ("revision_missing", "fail"), ("gated", "fail"), ("repo_missing", "fail"),
    ("error:Timeout", "warn")])
def test_input_dataset_paths(outcome, status):
    c = run(FakeProbes(repos={"lerobot/aloha_static_cups_open": outcome}))["input"]
    assert c.status == status


def test_input_local_mp4_dir(tmp_path):
    tpl = Path(__file__).resolve().parents[2] / "src/flora_deploy/config/templates/" \
        "exact_mp4_dir.yaml"
    cfg = load_config(tpl)
    assert cfg.input.repo is None
    assert d.check_input(FakeProbes(files=2), cfg, None).status == "pass"
    assert d.check_input(FakeProbes(files=0), cfg, None).status == "fail"
    assert d.check_input(FakeProbes(files=None), cfg, None).status == "fail"
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert d.RealProbes().local_files(str(tmp_path), "*.mp4") == 1
    assert d.RealProbes().local_files(str(tmp_path / "nope"), "*.mp4") is None


def test_a_crashing_probe_is_a_warning():
    class Boom(FakeProbes):
        def volume_listdir(self, *a):
            raise RuntimeError("boom")

    c = run(Boom())
    assert c["doctor"].status == "warn" and "boom" in c["doctor"].message


def test_cli_exit_codes_and_json(monkeypatch, tempseg_pinned):
    cli = CliRunner()
    monkeypatch.setattr(d, "_probes_factory", FakeProbes)
    r = cli.invoke(app, ["doctor", "--config", str(EXAMPLE)])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output and "note:" in r.output
    r = cli.invoke(app, ["doctor", "--config", str(EXAMPLE), "--json"])
    doc = json.loads(r.output)
    assert r.exit_code == 0 and doc["ok"] is True
    assert {"name", "status", "message", "fix"} <= set(doc["checks"][0])

    monkeypatch.setattr(d, "_probes_factory",
                        lambda: FakeProbes(repos={"nvidia/Flora-Captioner": "gated"}))
    r = cli.invoke(app, ["doctor", "--config", str(EXAMPLE), "--json"])
    assert r.exit_code == 2 and json.loads(r.output)["ok"] is False
