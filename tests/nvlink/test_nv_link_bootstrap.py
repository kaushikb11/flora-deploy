"""Linking NVIDIA's real code: bootstrap SHA check, isolation, prompt SHA assertion per pool."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from flora_deploy.nv import Family
from flora_deploy.nv.linker import load_family, verify_code_manifest
from tests.nvtext import nv_output

pytestmark = pytest.mark.nvlink


def run_isolated(code: str, *args: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter so families never leak between tests. The CI
    workflow sets ``PYTHONDONTWRITEBYTECODE=1``; drop it so the bytecode assertions test what
    *our* code does to ``sys.dont_write_bytecode``, not the runner's environment."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *map(str, args)],
        capture_output=True, text=True, timeout=300, env=env,
    )


def test_segcap_bootstrap_passes(nv_code):
    code_dir = nv_code("tempseg_code")
    nv = load_family(Family.SEGCAP, code_dir)
    assert nv.alias == "flora_nv_segcap_78a0e4ac"
    assert load_family(Family.SEGCAP, code_dir) is nv  # idempotent
    vendor = (code_dir / "batch_utils" / "vendor").resolve()
    # bootstrap.initialize() returned NVIDIA's adapter module, imported from this snapshot
    assert Path(nv.production.__file__).resolve() == vendor / "production_adapter" / "adapter.py"
    assert sys.modules["adapter"] is nv.production
    assert Path(sys.modules["flora_runtime"].__file__).resolve().is_relative_to(vendor)
    assert "batch_utils" not in sys.modules  # only the alias exists
    assert nv.pipeline.__name__ == f"{nv.alias}.pipeline"
    assert nv.vendor("flora_unified.unified_input").__name__ == "flora_unified.unified_input"
    # NVIDIA's vendor manifest is what bootstrap verified; CODE_MANIFEST covers the rest
    manifest = json.loads((vendor / "MANIFEST.json").read_text())
    for rel, digest in manifest.items():
        assert hashlib.sha256((vendor / rel).read_bytes()).hexdigest() == digest
    assert verify_code_manifest(code_dir) > len(manifest)


def test_captioner_batch_utils_is_byte_identical(nv_code):
    ts, cap = nv_code("tempseg_code") / "batch_utils", nv_code("captioner_code") / "batch_utils"

    def digests(root):
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        }

    assert digests(ts) == digests(cap)


def test_rewarder_conflicts_with_segcap(nv_code):
    seg, rew = nv_code("tempseg_code"), nv_code("rewarder")
    out = run_isolated(
        """
        import sys
        from flora_deploy.nv import Family, FamilyConflictError
        from flora_deploy.nv.linker import load_family
        first, second = sys.argv[1], sys.argv[3]
        load_family(Family(first), sys.argv[2])
        try:
            load_family(Family(second), sys.argv[4])
        except FamilyConflictError as exc:
            print("CONFLICT", exc)
        else:
            print("LOADED BOTH")
        """,
        "segcap", seg, "rewarder", rew,
    )
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    assert out.stdout.startswith("CONFLICT"), out.stdout

    out = run_isolated(
        """
        import sys
        from flora_deploy.nv import Family, FamilyConflictError
        from flora_deploy.nv.linker import load_family
        nv = load_family(Family.REWARDER, sys.argv[1])
        assert nv.core.__name__ == nv.alias + ".core", nv.core.__name__
        assert nv.engine.__name__ == nv.alias + ".engine"
        assert "adapter" not in sys.modules and "batch_utils" not in sys.modules
        for other in (sys.argv[2], sys.argv[3]):
            try:
                load_family(Family.SEGCAP, other)
            except FamilyConflictError:
                print("CONFLICT")
        """,
        rew, seg, nv_code("captioner_code"),
    )
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    assert out.stdout.split() == ["CONFLICT", "CONFLICT"]


def test_same_family_from_two_snapshots_conflicts(nv_code):
    out = run_isolated(
        """
        import sys
        from flora_deploy.nv import Family, FamilyConflictError
        from flora_deploy.nv.linker import load_family
        load_family(Family.SEGCAP, sys.argv[1])
        try:
            load_family(Family.SEGCAP, sys.argv[2])
        except FamilyConflictError:
            print("CONFLICT")
        """,
        nv_code("tempseg_code"), nv_code("captioner_code"),
    )
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    assert out.stdout.strip() == "CONFLICT"


def test_foreign_adapter_module_conflicts(nv_code, tmp_path):
    (tmp_path / "adapter.py").write_text("X = 1\n")
    out = run_isolated(
        """
        import sys
        sys.path.insert(0, sys.argv[2])
        import adapter
        from flora_deploy.nv import Family, FamilyConflictError
        from flora_deploy.nv.linker import load_family
        try:
            load_family(Family.SEGCAP, sys.argv[1])
        except FamilyConflictError as exc:
            print("CONFLICT", "adapter" in str(exc))
        """,
        nv_code("tempseg_code"), tmp_path,
    )
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    assert out.stdout.strip() == "CONFLICT True"


def test_tampered_vendor_fails_nvidia_bootstrap(nv_code, tmp_path):
    """NVIDIA's own SHA check is live: a modified vendor file stops the import (we skip our
    CODE_MANIFEST check here to reach NVIDIA's), and the failed load leaves no alias behind."""
    copy = tmp_path / "nvidia--Flora-TempSeg@deadbeef00000000"
    shutil.copytree(nv_code("tempseg_code") / "batch_utils", copy / "batch_utils")
    target = copy / "batch_utils" / "vendor" / "support" / "plans.py"
    target.write_text(target.read_text() + "\n# local edit\n")
    out = run_isolated(
        """
        import sys
        from flora_deploy.nv import Family
        from flora_deploy.nv.linker import CodeManifestError, load_family
        try:
            load_family(Family.SEGCAP, sys.argv[1])
        except CodeManifestError:
            print("MANIFEST")
        try:
            load_family(Family.SEGCAP, sys.argv[1], verify_manifest=False)
        except RuntimeError as exc:
            print("BOOTSTRAP", type(exc).__name__, str(exc).startswith("Production code changed"))
        print(sorted(m for m in sys.modules if m.startswith("flora_nv_")))
        """,
        copy,
    )
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    assert out.stdout.splitlines() == ["MANIFEST", "BOOTSTRAP RuntimeError True", "[]"]


def _copy_code(nv_code, key, tmp_path, name=None) -> Path:
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_dirname

    src = nv_code(key)
    dst = tmp_path / (name or snapshot_dirname(PINS_BY_KEY[key]))
    shutil.copytree(src / "batch_utils", dst / "batch_utils",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_pinned_code_digests_match_fetched_code(nv_code):
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.nv.linker import code_tree_sha256, verify_family_code

    for key in ("tempseg_code", "captioner_code"):
        pinned = dict(PINS_BY_KEY[key].code_digests)
        pkg = nv_code(key) / "batch_utils"
        assert hashlib.sha256((pkg / "CODE_MANIFEST.json").read_bytes()).hexdigest() == pinned[
            "code_manifest_sha256"]
        assert hashlib.sha256((pkg / "vendor" / "MANIFEST.json").read_bytes()).hexdigest() == (
            pinned["vendor_manifest_sha256"])
        assert len(verify_family_code(Family.SEGCAP, nv_code(key))) == 43
    rewarder = nv_code("rewarder")
    assert code_tree_sha256(rewarder) == dict(PINS_BY_KEY["rewarder"].code_digests)[
        "code_tree_sha256"]
    assert {Path(p).name for p in verify_family_code(Family.REWARDER, rewarder)} >= {
        "core.py", "engine.py", "terminal_reward.py", "dense_reward.py", "rubric_reward.py"}


PLANTED = '''
import sys
from flora_deploy.nv import Family
from flora_deploy.nv.linker import CodeManifestError, load_family
try:
    load_family(Family(sys.argv[2]), sys.argv[1])
except CodeManifestError as exc:
    print("REFUSED", exc)
else:
    import transformers
    print("LOADED", getattr(transformers, "HIJACKED", False))
print(sorted(m for m in sys.modules if m.startswith("flora_nv_")))
'''


@pytest.mark.parametrize("planted", [
    "vendor/support/transformers.py",
    "vendor/production_adapter/core/transformers.py",
    "vendor/hook.pth",
    "vendor/support/_native.cpython-312-darwin.so",
    "models_extra.py",
])
def test_planted_file_in_segcap_snapshot_is_refused(nv_code, tmp_path, planted):
    """M3: every listed file intact, one unlisted file added on (or next to) a sys.path entry
    bootstrap adds -> refused before any NVIDIA code runs."""
    copy = _copy_code(nv_code, "tempseg_code", tmp_path)
    target = copy / "batch_utils" / planted
    target.write_text("HIJACKED = True\n")
    out = run_isolated(PLANTED, copy, "segcap")
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    lines = out.stdout.splitlines()
    assert lines[0].startswith("REFUSED") and "not listed" in lines[0], lines
    assert planted in lines[0] and lines[1] == "[]"


def test_planted_file_in_rewarder_snapshot_is_refused(nv_code, tmp_path):
    copy = _copy_code(nv_code, "rewarder", tmp_path)
    (copy / "batch_utils" / "transformers.py").write_text("HIJACKED = True\n")
    out = run_isolated(PLANTED, copy, "rewarder")
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    assert out.stdout.startswith("REFUSED") and "code_tree_sha256" in out.stdout
    edited = _copy_code(nv_code, "rewarder", tmp_path / "b")
    core = edited / "batch_utils" / "core.py"
    core.write_text(core.read_text() + "\n# local edit\n")
    out = run_isolated(PLANTED, edited, "rewarder")
    assert out.stdout.startswith("REFUSED"), nv_output(out.stdout + out.stderr)


def test_rewritten_manifests_are_refused(nv_code, tmp_path):
    """Changing a vendor file *and* both manifests consistently still fails: the manifests'
    sha256 is pinned in pins.py."""
    copy = _copy_code(nv_code, "tempseg_code", tmp_path)
    pkg = copy / "batch_utils"
    target = pkg / "vendor" / "support" / "plans.py"
    target.write_text(target.read_text() + "\n# local edit\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    vendor = json.loads((pkg / "vendor" / "MANIFEST.json").read_text())
    vendor["support/plans.py"] = digest
    (pkg / "vendor" / "MANIFEST.json").write_text(json.dumps(vendor, indent=2))
    manifest = json.loads((pkg / "CODE_MANIFEST.json").read_text())
    for rel in ("batch_utils/vendor/support/plans.py", "batch_utils/vendor/MANIFEST.json"):
        data = (copy / rel).read_bytes()
        manifest["files"][rel] = dict(manifest["files"][rel], sha256=hashlib.sha256(
            data).hexdigest(), bytes=len(data))
    (pkg / "CODE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    out = run_isolated(PLANTED, copy, "segcap")
    assert out.stdout.startswith("REFUSED") and "pinned code_manifest_sha256" in out.stdout


def test_lazy_vendor_imports_are_verified_and_write_no_bytecode(nv_code, tmp_path):
    """L1 + L2: modules NVIDIA imports lazily (plans, verify_gdn, production_gdn_fix, ...) come
    from the snapshot even with same-named modules ahead on sys.path; nothing is written into the
    snapshot (no __pycache__; the unit tests check that a stale .pyc is never executed)."""
    copy = _copy_code(nv_code, "tempseg_code", tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    names = ["plans", "verify_gdn", "production_gdn_fix", "tempseg_fp32_loader",
             "robot_caption_geometry", "pts_input_adapter", "canonical_rgb", "composers"]
    for name in names:
        (shadow / f"{name}.py").write_text("SHADOW = True\n")
    out = run_isolated(
        """
        import json, sys
        from pathlib import Path
        from flora_deploy.nv import Family
        from flora_deploy.nv.linker import load_family
        nv = load_family(Family.SEGCAP, sys.argv[1])
        sys.path.insert(0, sys.argv[2])  # e.g. a later sys.path edit by someone else
        names = json.loads(sys.argv[3])
        mods = {n: __import__(n) for n in names}
        nv.pipeline, nv.api, nv.models, nv.media, nv.preprocessing  # all of batch_utils
        import flora_runtime.video_segmentation.fine_worker
        root = Path(sys.argv[1]).resolve()
        print(json.dumps({
            "shadowed": [n for n, m in mods.items() if hasattr(m, "SHADOW")],
            "outside": [n for n, m in mods.items()
                        if not Path(m.__file__).resolve().is_relative_to(root)],
            "audited": len(nv.audit()),
            "dont_write_bytecode": sys.dont_write_bytecode,
        }))
        """,
        copy, shadow, json.dumps(names),
    )
    assert out.returncode == 0, nv_output(out.stderr, "subprocess stderr")
    result = json.loads(out.stdout)
    assert result["shadowed"] == [] and result["outside"] == []
    assert result["audited"] > 30 and result["dont_write_bytecode"] is False
    assert not list(copy.rglob("__pycache__")) and not list(copy.rglob("*.pyc"))


DURATIONS_S = [4.0, 7.3, 16.0, 30.0, 61.25, 120.0]


def test_temporal_prompt_every_pool(nv_code):
    """``adapter.temporal_prompt`` (which asserts NVIDIA's prompt-template SHA) succeeds for every
    pool, on windows from NVIDIA's own scheduler, and substitutes each window's envelope."""
    nv = load_family(Family.SEGCAP, nv_code("tempseg_code"))
    a = nv.production
    from flora_runtime.video_segmentation.fine_temporal import window_from_scope
    from flora_runtime.video_segmentation.scope_scheduler import schedule_fine_scopes

    assert set(a.PROFILES) == set(a.TEMPLATES)
    for pool, profile_id in a.PROFILES.items():
        prompts = []
        for duration_s in DURATIONS_S:
            scopes = schedule_fine_scopes(
                source_id="upload_" + "0" * 24, duration_s=duration_s, window_profile_id=profile_id
            )
            assert scopes
            for scope in scopes:
                window = window_from_scope(scope)
                prompt = a.temporal_prompt(window, pool)
                for key in ("valid_start_tick", "valid_end_tick", "core_start_tick",
                            "core_end_tick"):
                    assert f'"{key}":{getattr(window, key)}' in prompt
                prompts.append(prompt)
        assert len(set(prompts)) > 1, pool
