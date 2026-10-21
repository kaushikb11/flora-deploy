"""The built-in patches, in application order (see ``docs/accel.md``)."""

from __future__ import annotations

from flora_deploy.accel.framecache import decode_once_factory, frames_for_targets
from flora_deploy.accel.geometry import GEOMETRY_TARGETS, geometry_factory
from flora_deploy.accel.pngmem import PNG_TARGETS, png_inmem_factory
from flora_deploy.accel.registry import Patch, register
from flora_deploy.accel.transport import transport_factory
from flora_deploy.nv import Family

PARITY = "tests/nvlink/test_accel_parity.py"

DECODE_ONCE = register(
    Patch(
        name="decode_once",
        family=Family.SEGCAP,
        targets=frames_for_targets(),
        factory=decode_once_factory,
        parity_test=f"{PARITY}::test_decode_once_frames_identical",
        description=(
            "frames_for decodes each source frame at most once per episode: persistent decoder "
            "cursor, verified random-access restarts, LRU of converted frames / mosaics"
        ),
    )
)

TRANSPORT_POOL = register(
    Patch(
        name="transport_pool",
        family=Family.SEGCAP,
        targets=("batch_utils.pipeline.human_transport",),
        factory=transport_factory,
        parity_test=f"{PARITY}::test_transport_pool_identical",
        description=(
            "human_transport's JPEG q98 round trip in a thread pool, memoized by input content"
        ),
    )
)

PNG_INMEM = register(
    Patch(
        name="png_inmem",
        family=Family.SEGCAP,
        targets=PNG_TARGETS,
        factory=png_inmem_factory,
        parity_test=f"{PARITY}::test_png_inmem_receipts_identical",
        description=(
            "caption_single's PNG frames encoded ahead in a thread pool with Pillow's identical "
            "encoder call; load_frames decodes each PNG once (SHA-keyed)"
        ),
    )
)

GEOMETRY_POOL = register(
    Patch(
        name="geometry_pool",
        family=Family.SEGCAP,
        targets=GEOMETRY_TARGETS,
        factory=geometry_factory,
        parity_test=f"{PARITY}::test_geometry_pool_identical",
        description=(
            "caption_three's pair packing / PNG level-1 encode (NVIDIA's own png()) and the "
            "decode-back in a thread pool"
        ),
    )
)

ALL = (DECODE_ONCE.name, TRANSPORT_POOL.name, PNG_INMEM.name, GEOMETRY_POOL.name)
