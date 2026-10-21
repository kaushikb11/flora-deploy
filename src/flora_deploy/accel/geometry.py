"""``geometry_pool``: ``caption_three``'s caption-geometry normalization on a thread pool.

NVIDIA's ``robot_caption_geometry.normalized`` (``vendor/robot_caption_geometry.py:14-46``)
packs each pair of 3-view mosaic frames side by side (LANCZOS-resized when the scale factor is
below 1), PNG-encodes the pair with ``png()`` (``compress_level=1``), sums the encoded sizes and
retries with the next smaller factor while the total exceeds ``PACKED_IMAGE_LIMIT_BYTES``.
``source_frames`` (``:48-60``) then decodes every PNG back and crops the frames out. All of it
runs serially and is the dominant CPU cost of a 3-view caption.

The replacement re-expresses both functions with the per-pair work submitted to a thread pool
(Pillow releases the GIL in resize, zlib encode and decode). It calls NVIDIA's own ``png`` and
``sha`` helpers and reads ``SCALES`` / ``PACKED_IMAGE_LIMIT_BYTES`` from NVIDIA's module, so the
encoded bytes - whose sizes decide the chosen scale - come from the identical encoder call. The
result dicts are assembled in NVIDIA's order with NVIDIA's keys; errors (missing frames, no
fitting geometry, the ``expected`` assertions, the crop SHA assertion) are raised in NVIDIA's
order. A factor whose running byte total already exceeds the limit stops early: NVIDIA discards
that factor's assets, so the remaining ones are never observable.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from typing import Any

from PIL import Image


def _asset(rcg, indices: list, originals: list, factor) -> dict:
    # robot_caption_geometry.py:23-35, per pair
    w, h = originals[0].size
    nw = round(w * factor)
    nh = round(h * factor)
    packed = Image.new("RGB", (nw * len(indices), nh))
    rgb = []
    for pos, image in enumerate(originals):
        if factor != 1:
            image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        packed.paste(image, (pos * nw, 0))
        rgb.append(rcg.sha(image.tobytes()))
    blob = rcg.png(packed)
    return dict(
        indices=indices,
        size=list(packed.size),
        sha256=rcg.sha(blob),
        rgb_sha256=rgb,
        bytes=len(blob),
        _blob=blob,
    )


def make_normalized(nv, original: Callable, state) -> Callable:
    def normalized(plan, frames, expected=None):
        rcg = nv.vendor("robot_caption_geometry")
        limit = rcg.PACKED_IMAGE_LIMIT_BYTES
        factors = [expected["common_geometry"]["scale"]] if expected else rcg.SCALES
        blocks = None
        count = 0
        for factor in factors:
            jobs = []  # (block index, indices, originals) in NVIDIA's order
            for b, block in enumerate(plan["blocks"]):
                for start in range(0, len(block["indices"]), 2):
                    indices = block["indices"][start : start + 2]
                    originals = [frames[i] for i in indices]
                    jobs.append((b, indices, originals))
            futures = [state.submit(_asset, rcg, ix, ims, factor) for _, ix, ims in jobs]
            assets: list[dict] = []
            count = 0
            try:
                for fut in futures:
                    asset = fut.result()
                    count += len(asset["_blob"]) + (-len(asset["_blob"]) % 3)
                    assets.append(asset)
                    if count > limit:
                        break  # NVIDIA discards this factor; the rest is unobservable
            finally:
                for fut in futures:
                    fut.cancel()
            state.count("geometry_assets", len(assets))
            if count <= limit:
                blocks = [dict(role=block["role"], images=[]) for block in plan["blocks"]]
                for (b, _, _), asset in zip(jobs, assets, strict=True):
                    blocks[b]["images"].append(asset)
                break
        if blocks is None:
            raise RuntimeError("No common caption input geometry fits")
        if expected:
            for old, new in zip(expected["frame_blocks"], blocks, strict=False):
                assert old["role"] == new["role"] and len(old["images"]) == len(new["images"])
                for a, b in zip(old["images"], new["images"], strict=False):
                    assert (
                        a["indices"] == b["indices"]
                        and a["rgb_sha256"] == b["rgb_sha256"]
                        and (a["size"] == b["size"])
                    )
        return dict(
            plan,
            frame_blocks=blocks,
            common_geometry=dict(
                scale=factor,
                source_pixel_unchanged=factor == 1,
                aligned_image_bytes=count,
                image_size_limit_bytes=limit,
                sampling_frames_removed=0,
            ),
        )

    normalized.__wrapped__ = original  # type: ignore[attr-defined]
    return normalized


def _crops(rcg, asset: dict) -> list:
    # robot_caption_geometry.py:50-59, per packed image
    image = Image.open(io.BytesIO(asset["_blob"])).convert("RGB")
    n = len(asset["indices"])
    w = image.width // n
    out = []
    for pos, i in enumerate(asset["indices"]):
        frame = image.crop((pos * w, 0, (pos + 1) * w, image.height))
        ok = rcg.sha(frame.tobytes()) == asset["rgb_sha256"][pos]
        out.append((i, frame, ok))
    return out


def make_source_frames(nv, original: Callable, state) -> Callable:
    def source_frames(normalized):
        rcg = nv.vendor("robot_caption_geometry")
        assets = [a for block in normalized["frame_blocks"] for a in block["images"]]
        futures = [state.submit(_crops, rcg, a) for a in assets]
        out = {}
        for fut in futures:
            for i, frame, ok in fut.result():
                assert ok
                out[i] = frame
        return out

    source_frames.__wrapped__ = original  # type: ignore[attr-defined]
    return source_frames


GEOMETRY_TARGETS = (
    "vendor.robot_caption_geometry.normalized",
    "vendor.robot_caption_geometry.source_frames",
)


def geometry_factory(nv, current: Mapping[str, Any], state) -> dict[str, Any]:
    return {
        GEOMETRY_TARGETS[0]: make_normalized(nv, current[GEOMETRY_TARGETS[0]], state),
        GEOMETRY_TARGETS[1]: make_source_frames(nv, current[GEOMETRY_TARGETS[1]], state),
    }
