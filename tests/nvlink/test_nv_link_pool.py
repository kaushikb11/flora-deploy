"""NVIDIA CPU functions in a spawn process pool (PLAN.md D12), via ``worker_init`` + ``NvRef``.

The pool's workers start empty (spawn): ``worker_init`` re-links the family in each one with all
of the linker's checks, and ``NvRef`` objects (family + module + qualname) are resolved there.
Results must equal the same calls made in this process.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from PIL import Image

from flora_deploy.media import synth
from flora_deploy.nv import Family, FamilyConflictError
from flora_deploy.nv.linker import NvRef, load_family, worker_init

pytestmark = pytest.mark.nvlink


def _worker_state(_):
    import sys

    from flora_deploy.nv.linker import loaded_family

    nv = loaded_family()
    return (nv.alias if nv else None, nv.verified if nv else None, "adapter" in sys.modules,
            sorted(nv.audit()) if nv else [])


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("pool_videos")
    out = {}
    for seed, role in enumerate(("main", "left_wrist", "right_wrist")):
        out[role] = synth.encode(root / f"{role}.mp4", synth.Spec(24, size=(128, 96), seed=seed))
    return out


def test_nvidia_functions_in_spawn_pool(nv_code, videos):
    code_dir = nv_code("tempseg_code")
    nv = load_family(Family.SEGCAP, code_dir)
    probed = {role: nv.media.probe(path) for role, path in videos.items()}
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8), "RGB")
    wanted = {0, 5, 23}
    calls = [
        (nv.ref("preprocessing", "instruction_suffix"), ("Fold the towel.",)),
        (nv.ref("preprocessing", "human_transport"), ({0: image, 1: image},)),
        (nv.ref("composers", "compose_top_dominant_mosaic", vendor=True), (image, image, image)),
        (nv.ref("media", "probe"), (videos["main"],)),
        (nv.ref("media", "decode_selected"), (probed["main"], wanted)),
        # three views: NVIDIA's canonical_rgb (lazily imported vendor module) + YAM mosaic
        (nv.ref_for(nv.media.frames_for), (probed, wanted)),
    ]
    local = [ref(*args) for ref, args in calls]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(2, mp_context=ctx, **nv.pool_initializer()) as pool:
        alias, verified, has_adapter, audited = pool.submit(_worker_state, 0).result()
        assert (alias, verified, has_adapter) == (nv.alias, True, True)
        assert "canonical_rgb" not in audited  # not imported yet in the worker
        remote = [pool.submit(ref, *args).result() for ref, args in calls]
        assert "canonical_rgb" in pool.submit(_worker_state, 0).result()[3]
        # a plain alias-module function pickles by name and works too, once workers are linked
        plain = nv.preprocessing.instruction_suffix
        assert pool.submit(plain, "x").result() == plain("x")

    assert remote[0] == local[0]
    assert remote[1].keys() == local[1].keys()
    for i in local[1]:
        assert remote[1][i].mode == "RGB"
        assert np.array_equal(np.asarray(remote[1][i]), np.asarray(local[1][i]))
    assert np.array_equal(np.asarray(remote[2]), np.asarray(local[2]))
    assert remote[3] == local[3] and type(remote[3]) is type(local[3])
    for got, want in ((remote[4], local[4]), (remote[5], local[5])):
        assert got.keys() == want.keys() == wanted
        for i in wanted:
            assert np.array_equal(np.asarray(got[i]), np.asarray(want[i]))


def _unlinked_call(ref):
    try:
        ref.resolve()
    except FamilyConflictError as exc:
        return str(exc)
    return "resolved"


def test_ref_in_unlinked_worker_fails_cleanly(nv_code):
    """Without ``worker_init`` the worker has no family: the ref says so (no broken pool)."""
    ref = NvRef(Family.SEGCAP, "preprocessing", "instruction_suffix")
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        assert "worker_init" in pool.submit(_unlinked_call, ref).result()


def _rewarder_state(_):
    from flora_deploy.nv.linker import loaded_family

    nv = loaded_family()
    return nv.family.value, nv.core.__name__


def test_worker_init_rewarder(nv_code):
    """The initializer also links the Rewarder family (in its own pool)."""
    code_dir = nv_code("rewarder")
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=worker_init,
                             initargs=("rewarder", str(code_dir))) as pool:
        family, core = pool.submit(_rewarder_state, 0).result()
    assert family == "rewarder" and core.endswith(".core")
