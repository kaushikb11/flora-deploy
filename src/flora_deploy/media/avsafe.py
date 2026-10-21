"""Open untrusted dataset media with FFmpeg restricted to local MP4/MOV (review 11b F5).

FFmpeg picks a demuxer from the file's content, not its name, and several demuxers open other
files or URLs (``concat``, ``hls``, ``mov`` data references). A dataset file named ``.mp4`` must
not make the decoder read ``/proc/self/environ`` or fetch a URL. :func:`open_input` passes, as
container (format-context) options only, so codec options and decode semantics are unchanged:

* ``protocol_whitelist=file``: no ``concat:``, ``http:``, ``pipe:`` ... protocol, also for any
  nested open a demuxer makes;
* ``format_whitelist`` = the ``mov`` demuxer's names: the only container the sources accept
  (``.mp4``/``.mov``/``.m4v``; ``index.scan_stream`` already refuses non-MP4/MOV time bases);
* ``enable_drefs=0``: the mov demuxer's external data references stay off (FFmpeg's default,
  pinned here).

Verified against the locked PyAV 16.0.1 / FFmpeg 8.0 (``tests/unit/test_untrusted_inputs.py``):
an ``ffconcat`` or HLS playlist named ``.mp4`` and the ``concat:`` / ``http:`` protocols are
refused. NVIDIA's own code (``batch_utils``) opens the videos we hand it with its own
``av.open``; those calls are not wrapped (we do not modify NVIDIA's files), but they only ever
see files this module and ``media.safepath`` already admitted.
"""

from __future__ import annotations

import os
from typing import Any

import av

MOV_DEMUXER = "mov,mp4,m4a,3gp,3g2,mj2"  # FFmpeg's name list for the mov demuxer
INPUT_CONTAINER_OPTIONS: dict[str, str] = {
    "protocol_whitelist": "file",
    "format_whitelist": MOV_DEMUXER,
    "enable_drefs": "0",
}


def open_input(path: str | os.PathLike[str], *, options: dict[str, str] | None = None,
               **kwargs: Any) -> Any:
    """``av.open(path, "r", options=options, container_options=INPUT_CONTAINER_OPTIONS)``.
    ``options`` (e.g. NVIDIA's ``err_detect``) apply to the container and codecs as before."""
    return av.open(str(path), "r", options=options,
                   container_options=dict(INPUT_CONTAINER_OPTIONS), **kwargs)
