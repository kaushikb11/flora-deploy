"""NVIDIA's real input builders + real Flora processors, run concurrently under our GPU lock.

CPU only, no weights: frames are synthetic (``media.decode_selected`` replaced) and the model call
is a fake that hashes every tensor NVIDIA hands to ``generate``. Concurrent episode threads with
per-thread processor clones must hand ``generate`` byte-identical tensors to a sequential run;
torch's intra-op thread count must not change them either.
"""

import contextvars
import hashlib
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flora_deploy.nv import Family, load_family
from flora_deploy.nv.session import CachedSession
from flora_deploy.observe import Patcher, TimedLock
from flora_deploy.stages import reference as ref

pytestmark = pytest.mark.nvlink
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROLES3 = ("main", "left_wrist", "right_wrist")


@pytest.fixture(scope="module")
def nv_env():
    import os

    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import NV_CODE_CACHE_DEFAULT, NV_CODE_CACHE_ENV, snapshot_dirname

    root = Path(os.environ.get(NV_CODE_CACHE_ENV, NV_CODE_CACHE_DEFAULT)).expanduser()
    dirs = {k: root / snapshot_dirname(PINS_BY_KEY[k]) for k in ("tempseg_code", "captioner_code")}
    for key, path in dirs.items():
        if not (path / "processor_config.json").is_file():
            pytest.skip(f"{key} processor files not fetched (fetch_nv_code.py --with-processor)")
    nv = load_family(Family.SEGCAP, dirs["tempseg_code"])
    procs = {"tempseg": nv.models.processor("tempseg", dirs["tempseg_code"]),
             "captioner": nv.models.processor("captioner", dirs["captioner_code"])}
    return nv, procs


def frame(role, i, size=(640, 480)):
    from PIL import Image

    w, h = size
    rng = np.random.default_rng((ROLES3.index(role) if role in ROLES3 else 7) * 100_003 + i)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x + 3 * i) % 256, (y + 5 * i) % 256, (x + y) % 256], axis=-1)
    noise = rng.integers(-40, 41, size=(h, w, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


def videos(nv, roles, n=600):
    return {r: nv.media.Video(path=Path(f"{r}.mp4"), sha256=f"{k:02d}" * 32, rate=Fraction(30),
                              pts_us=tuple(Fraction(j * 1_000_000, 30) for j in range(n)),
                              duration_us=20_000_000, size=(640, 480), cfr=True)
            for k, r in enumerate(roles)}


SEGS = [dict(start_s=2.0, end_s=5.0, metadata={}), dict(start_s=6.0, end_s=9.5, metadata={})]
CASES = [("caption_three", "YAM", 3), ("caption_single", "Human", 1), ("segment", "human", 1),
         ("caption_single", "YAM", 1)]


class FakeNvSession:
    def __init__(self, procs):
        self.procs, self.audit = procs, {}

    def load(self, kind):
        return object(), self.procs[kind]

    def close(self):
        pass


@pytest.fixture
def harness(nv_env, monkeypatch):
    nv, procs = nv_env
    seen = contextvars.ContextVar("seen")  # follows asyncio.to_thread (NVIDIA predictor)

    def fake_decode(video, wanted, canonical=False):
        return {i: frame(video.path.stem, i) for i in wanted}

    def fake_generate(model, proc, inputs, budget):
        d = hashlib.sha256()
        for k in sorted(inputs):
            v = inputs[k]
            if torch.is_tensor(v):
                d.update(k.encode() + str(v.dtype).encode() + str(list(v.shape)).encode())
                d.update(v.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        seen.get().append(d.hexdigest())
        return "pick up the cup" if budget == 512 else "not json"

    monkeypatch.setattr(nv.media, "decode_selected", fake_decode)
    monkeypatch.setattr(nv.pipeline, "generate", fake_generate)
    patcher, lock = Patcher(), TimedLock("gpu")
    ref.install_gpu_lock(nv, lock, patcher)
    base = CachedSession(lambda: FakeNvSession(procs))

    def run(case, clones=None):
        out = []
        seen.set(out)
        session = ref.EpisodeSession(base, clones)
        kind, prof, n = case
        v = videos(nv, ROLES3[:n])

        def progress(*a, **k):
            return None

        if kind == "segment":
            with pytest.raises(Exception):  # noqa: B017 - the fake output is not valid JSON
                nv.pipeline.segment(v, prof, session, "", progress)
        elif kind == "caption_three":
            nv.pipeline.caption_three(v, SEGS, session, "", progress, profile=prof)
        else:
            nv.pipeline.caption_single(v, SEGS, prof, session, "", progress)
        return out

    yield run, lock
    patcher.restore()


def test_concurrent_inputs_identical_to_sequential(harness):
    run, lock = harness
    reference = [run(c) for c in CASES]
    assert all(reference) and lock.stats()["acquisitions"] == sum(map(len, reference))
    clones = ref.ProcessorClones()
    with ThreadPoolExecutor(len(CASES)) as pool:
        got = list(pool.map(lambda c: run(c, clones), CASES))
    assert got == reference
    assert clones.clones >= len(CASES)


def test_torch_threads_do_not_change_inputs(harness):
    run, _ = harness
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        four = [run(c) for c in CASES[:2]]
        torch.set_num_threads(1)
        one = [run(c) for c in CASES[:2]]
        torch.set_num_threads(max(8, before))
        many = [run(c) for c in CASES[:2]]
    finally:
        torch.set_num_threads(before)
    assert four == one == many
