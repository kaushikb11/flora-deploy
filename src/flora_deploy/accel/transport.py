"""``transport_pool``: NVIDIA's ``human_transport`` JPEG round trip, memoized and parallel.

NVIDIA's ``preprocessing.human_transport`` (``batch_utils/preprocessing.py:14-21``) encodes every
frame of every TempSeg window with ``encode_regularized_image(image, encoding='jpeg_q98_444')``
and decodes it back, one frame at a time. Frames shared by overlapping windows are re-encoded.

The replacement calls NVIDIA's own ``encode_regularized_image`` and the same decode
(``Image.open(BytesIO(payload))`` + ``convert('RGB')``) per frame, but

- runs the frames of one call concurrently in a thread pool (Pillow releases the GIL inside its
  JPEG encoder/decoder), and
- memoizes the result by a content key of the input image (mode, size, pixel digest, palette,
  info): JPEG encode/decode is deterministic for identical input, so a hit is the value NVIDIA
  would recompute.

Output dict keys and order equal the input's; values are fresh copies; an exception raised for a
frame is re-raised in input order, as NVIDIA's sequential loop would raise it first.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from typing import Any

from PIL import Image

from flora_deploy.accel.state import image_key


def make_human_transport(nv, original: Callable, state) -> Callable:
    def one(image):
        # preprocessing.py:17-20, with NVIDIA's encoder looked up at call time like NVIDIA does
        efs = nv.vendor("flora_runtime.exact_frame_serving")
        payload = efs.encode_regularized_image(image, encoding="jpeg_q98_444")["payload"]
        with Image.open(io.BytesIO(payload)) as decoded:
            return decoded.convert("RGB")

    def keyed(image):
        if not isinstance(image, Image.Image):
            return one(image)  # NVIDIA's own error for a non-image
        key = ("jpeg_q98_444", image_key(image))
        hit = state.transport.get(key)
        if hit is not None:
            state.count("transport_cache_hits")
            return hit
        out = one(image)
        state.transport.put(key, out)
        state.count("transport_encoded")
        return out

    def human_transport(frames):
        items = list(frames.items())
        futures = [state.submit(keyed, image) for _, image in items]
        output = {}
        for (index, _), fut in zip(items, futures, strict=True):
            output[index] = fut.result().copy()
        return output

    human_transport.__wrapped__ = original  # type: ignore[attr-defined]
    return human_transport


def transport_factory(nv, current: Mapping[str, Any], state) -> dict[str, Any]:
    return {t: make_human_transport(nv, fn, state) for t, fn in current.items()}
