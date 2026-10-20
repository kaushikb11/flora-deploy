"""Our digest derivations agree with NVIDIA's own code (loaded from the fetched code pins)."""

import hashlib
import importlib
import json
import sys
import types

import pytest

from flora_deploy.config.pins import PINS_BY_KEY
from flora_deploy.prep.download import FileEntry, SnapshotPlan
from flora_deploy.prep.verify import hash_file, verify_snapshot

pytestmark = pytest.mark.nvlink


@pytest.fixture
def nv_pkg(nv_code):
    """Import ``batch_utils`` submodules under a unique name without running its __init__
    (which would bootstrap the whole pipeline); removed from sys.modules afterwards."""
    before = set(sys.modules)

    def load(key: str, sub: str):
        name = f"_flora_prep_test_{key}"
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(nv_code(key) / "batch_utils")]
            sys.modules[name] = pkg
        return importlib.import_module(f"{name}.{sub}")

    yield load
    for m in set(sys.modules) - before:
        del sys.modules[m]


@pytest.mark.parametrize("key", ["tempseg_code", "captioner_code"])
def test_pins_match_nvidia_models_json(nv_code, key):
    models = json.loads((nv_code(key) / "batch_utils" / "models.json").read_text())
    tempseg, captioner = PINS_BY_KEY["tempseg"], PINS_BY_KEY["captioner"]
    assert models["tempseg"]["revision"] == tempseg.revision
    assert models["captioner"]["revision"] == captioner.revision
    assert models["tempseg_base"]["revision"] == PINS_BY_KEY["tempseg_base"].revision
    assert dict(tempseg.expected)["adapter_sha256"] == models["tempseg"]["sha256"]
    assert (dict(captioner.expected)["source_manifest_identity"]
            == models["captioner"]["source_manifest_identity"])


def test_source_manifest_identity_is_not_computed_by_nvidia_code(nv_code):
    # verify.py checks MODEL_PROVENANCE.json because NVIDIA's code never derives this value.
    code = nv_code("tempseg_code") / "batch_utils"
    assert not [p for p in code.rglob("*.py") if "source_manifest_identity" in p.read_text()]


def test_adapter_digest_matches_nvidia_loader(nv_code, nv_pkg, tmp_path):
    loader = nv_pkg("tempseg_code", "vendor.tempseg_fp32_loader")
    p = tmp_path / "adapter_model.safetensors"
    p.write_bytes(bytes(range(256)) * 40_000)
    assert hash_file(p, git_blob=False, chunk=4096)[0] == loader.digest(p)


def test_manifest_semantics_match_nvidia_validate_manifest(nv_pkg, tmp_path, capsys):
    models = nv_pkg("tempseg_code", "models")
    snap = tmp_path / "captioner"
    snap.mkdir()
    files = {"config.json": b"{}\n", "model-00001-of-00001.safetensors": b"w" * 10_000}
    for n, b in files.items():
        (snap / n).write_bytes(b)
    manifest = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in files.items())
    manifest = manifest.replace("  model-", " *model-")  # binary-mode marker, both accept it
    (snap / "SOURCE_MANIFEST.sha256").write_text(manifest)
    plan_files = [
        FileEntry(n, (snap / n).stat().st_size, None,
                  hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())
        for n in sorted(p.name for p in snap.iterdir())
        for d in [(snap / n).read_bytes()]
    ]
    plan = SnapshotPlan(PINS_BY_KEY["captioner"], tuple(plan_files))
    models._validate_manifest(snap)  # NVIDIA: passes
    ours = {c.name: c.ok for c in verify_snapshot(plan, snap).checks}
    assert ours["manifest:SOURCE_MANIFEST.sha256"]

    (snap / "config.json").write_bytes(b"[]\n")
    with pytest.raises(RuntimeError):
        models._validate_manifest(snap)
    ours = {c.name: c.ok for c in verify_snapshot(plan, snap).checks}
    assert not ours["manifest:SOURCE_MANIFEST.sha256"]
    capsys.readouterr()
