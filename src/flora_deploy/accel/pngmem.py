"""``png_inmem``: ``caption_single``'s PNG round trip without serial encode / repeated decode.

NVIDIA's ``pipeline.caption_single`` (``batch_utils/pipeline.py:85-115``) saves every caption
frame with ``image.save(path, format='PNG')`` into a temporary directory, records
``png_sha256`` of the file bytes and the image size, and ``unified_input.load_frames``
(``vendor/flora_unified/unified_input.py:56-83``) then re-reads, SHA-checks and PNG-decodes a
frame once **per block it appears in**. The encode (zlib level 6) is the single largest CPU cost
of the caption stage, and it runs one frame at a time.

The patch has three parts; NVIDIA's ``caption_single`` itself runs unmodified:

1. ``pipeline.caption_single`` is wrapped so the frames ``frames_for`` returns *inside it* are
   tagged for prefetch: each frame's PNG is encoded **in a thread pool** as soon as the frames are
   available, by calling Pillow's ``Image.save(buffer, format='PNG')`` - the same encoder with
   the same parameters as NVIDIA's ``image.save(path, format='PNG')`` (the PNG writer emits
   identical bytes for a path and a buffer; the parity test checks the bytes and receipts).
   When NVIDIA's loop then calls ``image.save(path, format='PNG')`` the prefetched bytes are
   written to ``path``; any other ``save`` call goes to Pillow unchanged.
2. The worker also decodes those bytes (``Image.open`` + ``load`` + ``convert('RGB')``, as
   NVIDIA does) and stores the frame under the bytes' SHA-256.
3. ``unified_input.load_frames`` is replaced by a re-expression with NVIDIA's validation
   unchanged: ``validate(sample)``, the root-ownership check, reading the file and comparing its
   SHA-256 to the receipt, the size check, and the same ``resize`` for oversized frames. Only the
   PNG decode is served from the SHA-keyed cache (content-addressed: equal bytes decode to the
   same pixels), so each frame is decoded once instead of once per block.

The files are still written and read (NVIDIA hashes what is on disk); what disappears is the
serial encode and the repeated decode.
"""

from __future__ import annotations

import contextvars
import hashlib
import io
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image

_PREFETCH: contextvars.ContextVar[Any] = contextvars.ContextVar("flora_accel_png", default=None)


def _decode_png(data: bytes) -> Image.Image:
    # unified_input.py:69-71 (from bytes instead of the path: the same decoder on the same bytes)
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return im.convert("RGB")


class PrefetchedPNG(Image.Image):
    """A frame whose ``save(path, format='PNG')`` writes bytes encoded ahead of time."""

    _flora_png: Any

    def save(self, fp, format=None, **params):  # noqa: A002 - Pillow's signature
        fut = self.__dict__.get("_flora_png")
        if (
            fut is not None
            and format == "PNG"
            and not params
            and isinstance(fp, (str, os.PathLike))
        ):
            data = fut.result()
            with open(fp, "w+b") as f:  # Image.save opens paths with "w+b"
                f.write(data)
            return None
        return Image.Image.save(self, fp, format, **params)


SINGLE_VIEW_MAX_PIXELS = 262144  # unified_input.py:61 for camera_count 1


def regularized_size(frame: Image.Image, max_pixels: int) -> tuple[int, int] | None:
    """``load_frames``' downscale target (unified_input.py:76-79), or ``None`` to keep."""
    area = frame.width * frame.height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        return (int(frame.width * scale), int(frame.height * scale))
    return None


def _resized(state, digest: str, frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    key = ("resized", digest, size)
    hit = state.png.get(key)
    if hit is not None:
        state.count("png_resize_cache_hits")
        return hit[1]
    out = frame.resize(size)  # unified_input.py:79, same call (Pillow's default resample)
    state.png.put(key, (None, out))
    return out


def _encode_job(state, image: Image.Image) -> bytes:
    buf = io.BytesIO()
    Image.Image.save(image, buf, format="PNG")  # pipeline.py:99's encoder and parameters
    data = buf.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if state.png.get(digest) is None:
        frame = _decode_png(data)
        state.png.put(digest, (None, frame))
        size = regularized_size(frame, SINGLE_VIEW_MAX_PIXELS)
        if size is not None:  # caption_single is single-view: precompute load_frames' resize
            _resized(state, digest, frame, size)
    state.count("png_prefetched")
    return data


def _tag(state, images: dict) -> dict:
    for image in images.values():
        if type(image) is not Image.Image:
            continue
        # NVIDIA only reads ``size`` and calls ``save`` (which waits for this job) on it
        image.__class__ = PrefetchedPNG
        image._flora_png = state.submit(_encode_job, state, image)
    return images


def make_caption_single(original: Callable, state) -> Callable:
    def caption_single(*args, **kwargs):
        token = _PREFETCH.set(state)
        try:
            return original(*args, **kwargs)
        finally:
            _PREFETCH.reset(token)

    caption_single.__wrapped__ = original  # type: ignore[attr-defined]
    return caption_single


def make_frames_for(original: Callable, state) -> Callable:
    def frames_for(videos, wanted, mosaic="YAM"):
        images = original(videos, wanted, mosaic=mosaic)
        if _PREFETCH.get() is state:
            _tag(state, images)
        return images

    frames_for.__wrapped__ = original  # type: ignore[attr-defined]
    return frames_for


def make_load_frames(nv, original: Callable, state) -> Callable:
    def decoded(digest: str, raw: bytes) -> Image.Image:
        """The decoded frame (shared, read-only by contract: copy before handing it out)."""
        hit = state.png.get(digest)
        if hit is not None:
            state.count("png_decode_cache_hits")
            return hit[1]
        frame = _decode_png(raw)
        state.png.put(digest, (None, frame))
        state.count("png_decoded")
        return frame

    def load_frames(sample, root, preserve_canvas=True):
        ui = nv.vendor("flora_unified.unified_input")
        ui.validate(sample)  # unified_input.py:57
        videos = []
        sizes = []
        max_pixels = 524288 if preserve_canvas and sample["camera_count"] == 3 else 262144
        for b in sample["blocks"]:
            frames = []
            shapes = []
            for path, receipt in zip(b["paths"], b["frames"], strict=True):
                p = Path(root) / path
                if not p.resolve().is_relative_to(Path(root).resolve()):
                    raise ValueError("Unowned frame path")
                raw = p.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != receipt["png_sha256"]:
                    raise ValueError("PNG changed")
                frame = decoded(digest, raw)
                if list(frame.size) != receipt["size"]:
                    raise ValueError("Frame dimensions changed")
                original_size = list(frame.size)
                size = regularized_size(frame, max_pixels)
                if size is not None:
                    frame = _resized(state, digest, frame, size)
                frame = frame.copy()  # never hand out a cached object
                frames.append(frame)
                shapes.append(dict(original=original_size, regularized=list(frame.size)))
            videos.append(frames)
            sizes.append(shapes)
        return (videos, sizes)

    load_frames.__wrapped__ = original  # type: ignore[attr-defined]
    return load_frames


PNG_TARGETS = (
    "batch_utils.pipeline.caption_single",
    "batch_utils.pipeline.frames_for",
    "vendor.flora_unified.unified_input.load_frames",
)


def png_inmem_factory(nv, current: Mapping[str, Any], state) -> dict[str, Any]:
    return {
        PNG_TARGETS[0]: make_caption_single(current[PNG_TARGETS[0]], state),
        PNG_TARGETS[1]: make_frames_for(current[PNG_TARGETS[1]], state),
        PNG_TARGETS[2]: make_load_frames(nv, current[PNG_TARGETS[2]], state),
    }
