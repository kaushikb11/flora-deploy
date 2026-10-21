"""Which side failed: the one failure classifier for every worker that runs NVIDIA's code.

``classify(exc)`` returns one of:

- ``nv_failure``: NVIDIA's own verdict on the episode. Terminal and excused (``EpisodeError``
  ``kind="nv_failure"``): re-running the same input on the same code gives the same verdict.
- ``cuda_fatal``: a CUDA fault (illegal access, device-side assert, cuBLAS / cuDNN / NCCL
  status). The CUDA context is unusable: the container must be replaced, and the start counts
  toward D15 quarantine, so an input that faults every time ends isolated (bisected, then
  poison-pilled) instead of crash-looping.
- ``oom``: GPU memory ran out for this input (``torch.OutOfMemoryError`` / "out of memory").
  Handled like ``cuda_fatal`` (PLAN.md D15: a deterministic per-episode OOM is a poison pill).
- ``infra``: the environment failed (I/O, the store, the GPU runtime, a model that is not
  loaded, NVIDIA's own integrity / environment checks). Retried, never excused.
- ``python``: a bug in code that is not NVIDIA's (ours). Retried like ``infra``, never excused.

**NVIDIA's code is identified by where it was loaded from**, never by a module-name prefix: the
linked family's snapshot directory (``nv.linker.loaded_family().code_dir``: the ``flora_nv_*``
alias package *and* the top-level vendor modules its bootstrap puts on ``sys.path``, such as
``flora_runtime``, ``adapter`` or ``plans``), plus any root registered with
:func:`register_nv_root` (fakes that stand in for NVIDIA's code in tests and local runs). A class
is NVIDIA's when its defining module's ``__file__`` is under such a root (or, when the module is
not importable here, the linked family's finder owns its name). A frame is NVIDIA's when its
code file is.

In order:

1. "out of memory" / ``OutOfMemoryError`` → ``oom``; a CUDA fault message → ``cuda_fatal``;
2. ``OSError`` (I/O, timeouts, connections), torch runtime classes, NVIDIA's ``NotReady`` and
   NVIDIA's environment checks (integrity, "Production code changed", "CUDA is required", ...)
   → ``infra``;
3. a class NVIDIA's code defines (anywhere in the MRO: ``FineSegmentationOutputError``,
   ``U2WorkerError``, ``InputError``, ...), our re-implementation of NVIDIA's context guard
   (``nv.budget.ContextBudgetError``) and every :class:`NvidiaSemanticsError` (an NVIDIA
   rejection reproduced by our certified re-implementations: ``NvidiaRuntimeError``, ...)
   → ``nv_failure``;
4. a builtin-family exception (``AssertionError``, ``TypeError``, ``LookupError``,
   ``StopIteration``, ``ValueError``, ``RuntimeError``, ``ArithmeticError``) whose innermost
   non-library frame is NVIDIA's → ``nv_failure`` (library frames, i.e. the standard library and
   site-packages, are skipped: ``json.loads`` or a torch shape check called by NVIDIA is NVIDIA's
   verdict; one called by us is ours);
5. otherwise ``python`` if that frame is ours (or the test's), else ``infra``.

Errors carried across a process boundary (``stages.reference.NvidiaError``) are classified from
the original exception when it is still attached (``__cause__``), else as the ``kind`` the sender
computed with this same function while the traceback was still available (so a pickled error
classifies exactly like the in-process one), else (an error built from strings only) from the
``origin`` the sender computed (:func:`origin`) and the original type name.
"""

from __future__ import annotations

import builtins
import sys
import sysconfig
import threading
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Literal

Kind = Literal["nv_failure", "infra", "python", "cuda_fatal", "oom"]
Origin = Literal["nv", "ours", "lib", "other"]
KINDS: tuple[Kind, ...] = ("nv_failure", "infra", "python", "cuda_fatal", "oom")
# kinds after which the container must be replaced and the start counts toward D15
CRASH_KINDS = frozenset({"cuda_fatal", "oom"})

# Same markers as ``workers_impl.reward.CUDA_FATAL_MARKERS`` (tests keep them equal).
CUDA_FATAL_MARKERS = ("CUDA error", "cudaError", "CUBLAS_STATUS", "CUDNN_STATUS",
                      "illegal memory access", "device-side assert", "NCCL error")
OOM_MARKERS = ("out of memory", "OutOfMemoryError")
# Never an episode verdict, whoever defines them (the GPU or its runtime failed, or NVIDIA's
# model is not loaded): by class name, anywhere in the MRO.
INFRA_TYPE_NAMES = frozenset({"OutOfMemoryError", "AcceleratorError", "NotReady", "CudaError",
                              "CUDAError"})
INFRA_MODULE_ROOTS = frozenset({"torch", "modal", "grpclib", "botocore", "aiohttp"})
# NVIDIA's own environment / integrity checks (builtin RuntimeError in batch_utils): the snapshot,
# the weights or the machine are wrong, not the episode.
NVIDIA_ENVIRONMENT_FAILURES = (
    "Invalid runtime manifest path",  # bootstrap.py
    "Production code changed",  # bootstrap.py
    "CUDA is required",  # api.py
    "Production request binding changed",  # pipeline.py
    "Invalid model manifest",  # models.py
    "A model asset failed its integrity check",  # models.py
    "A model asset changed during its integrity check",  # models.py
    "The released Captioner did not load completely",  # models.py
)
# NVIDIA modules that only run while loading weights / verifying the install: their asserts are
# environment checks (file stems).
NVIDIA_LOAD_MODULES = frozenset({"bootstrap", "verify_gdn", "tempseg_fp32_loader",
                                 "production_gdn_fix"})
# Builtin exception families that are NVIDIA's verdict when NVIDIA's own frame raised them.
VERDICT_BUILTINS: tuple[type[BaseException], ...] = (
    AssertionError, TypeError, LookupError, StopIteration, ValueError, RuntimeError,
    ArithmeticError)
# NVIDIA's input-rejection class names (``media.InputError``, ``models.InputErrorForModel``):
# only used for a transported error whose class cannot be resolved in this process.
NVIDIA_INPUT_ERRORS = frozenset({"InputError", "InputErrorForModel"})
# NVIDIA rejections raised as builtin RuntimeError: kept for transported errors (no frames).
NVIDIA_EPISODE_REJECTIONS = (
    "The pipeline returned an invalid caption interval.",  # api.py
    "No common caption input geometry fits",  # vendor/robot_caption_geometry.py
)

_PKG_DIR = Path(__file__).resolve().parents[1]  # src/flora_deploy


# --- our copies of NVIDIA's rejections ----------------------------------------------------------


class NvidiaSemanticsError(Exception):
    """Marker: one of NVIDIA's own episode rejections, raised by our certified re-implementation
    of the NVIDIA function that raises it (an ``accel`` patch, a balanced stage). Classified
    ``nv_failure`` (NVIDIA's verdict) although our frame raised it.

    Never raised directly: raise the concrete class for the builtin NVIDIA raises
    (``NvidiaRuntimeError``, ``NvidiaValueError``, ``NvidiaAssertionError``) with NVIDIA's exact
    message. Each subclasses that builtin (``except RuntimeError`` still catches it) and keeps
    the builtin's ``__name__``, so ``f"{type(e).__name__}: {e}"`` records exactly what NVIDIA's
    own error records; its ``__qualname__`` (``NvidiaRuntimeError``, ...) is the module
    attribute pickle resolves. :func:`nvidia_type_name` is the qualified name NVIDIA's error
    has (``builtins.RuntimeError``)."""


_MARKERS: dict[type[BaseException], type[Exception]] = {}


def _semantics(base: type[Exception]) -> type[Exception]:
    cls = type(base.__name__, (NvidiaSemanticsError, base), {
        "__module__": __name__, "__qualname__": f"Nvidia{base.__name__}",
        "__doc__": f"NVIDIA's ``{base.__name__}`` rejection, raised by our re-implementation "
                   "(:class:`NvidiaSemanticsError`)."})
    _MARKERS[base] = cls
    return cls


NvidiaRuntimeError = _semantics(RuntimeError)
NvidiaValueError = _semantics(ValueError)
NvidiaAssertionError = _semantics(AssertionError)
NvidiaKeyError = _semantics(KeyError)
NvidiaIndexError = _semantics(IndexError)


def as_nvidia(exc: BaseException) -> BaseException:
    """``exc`` (a builtin our re-implementation hit exactly where NVIDIA's code raises the same
    builtin, e.g. a missing frame lookup) as NVIDIA's verdict: the marker class for its type, same
    args (so the same message) and traceback. Any other exception is returned unchanged."""
    cls = _MARKERS.get(type(exc))
    if cls is None:
        return exc
    return cls(*exc.args).with_traceback(exc.__traceback__)


def nvidia_type_name(exc: BaseException) -> str:
    """The qualified type name of the error NVIDIA's own code raises: the builtin for an
    :class:`NvidiaSemanticsError` (``builtins.RuntimeError``), else ``exc``'s class."""
    cls = type(exc)
    if isinstance(exc, NvidiaSemanticsError):
        base = next(c for c in cls.__mro__ if c.__module__ == "builtins")
        return f"builtins.{base.__qualname__}"
    return f"{cls.__module__}.{cls.__qualname__}"
_extra_roots: list[Path] = []
_roots_lock = threading.Lock()


# --- where code comes from ----------------------------------------------------------------------


def register_nv_root(path: str | Path) -> None:
    """Treat ``path`` (a directory, or one ``.py`` file) as NVIDIA's code: for fakes that stand
    in for NVIDIA's family in tests and local runs. Idempotent."""
    p = Path(path).resolve()
    with _roots_lock:
        if p not in _extra_roots:
            _extra_roots.append(p)
        _file_origin.cache_clear()


def nv_roots() -> tuple[Path, ...]:
    """Directories (and files) holding NVIDIA's code in this process."""
    from flora_deploy.nv import linker

    fam = linker.loaded_family()
    roots = [Path(fam.code_dir).resolve()] if fam is not None else []
    with _roots_lock:
        return (*roots, *_extra_roots)


@lru_cache(maxsize=1)
def _lib_dirs() -> tuple[Path, ...]:
    out: list[Path] = []
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        if paths.get(key):
            out.append(Path(paths[key]).resolve())
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        out.append((Path(prefix) / "lib").resolve())
    return tuple(dict.fromkeys(out))


@lru_cache(maxsize=4096)
def _file_origin(filename: str, roots: tuple[Path, ...]) -> Origin:
    if not filename or filename.startswith("<"):  # <frozen ...>, <string>
        return "lib"
    try:
        p = Path(filename).resolve()
    except OSError:
        return "other"
    if any(p == r or p.is_relative_to(r) for r in roots):
        return "nv"
    if p.is_relative_to(_PKG_DIR):
        return "ours"
    if any(p.is_relative_to(d) for d in _lib_dirs()) or "site-packages" in p.parts:
        return "lib"
    return "other"


def file_origin(filename: str) -> Origin:
    """``nv`` (NVIDIA's code), ``ours`` (flora_deploy), ``lib`` (stdlib / site-packages) or
    ``other`` (tests, scripts) for a code file."""
    return _file_origin(filename, nv_roots())


def is_nv_class(cls: type) -> bool:
    """Whether NVIDIA's code defines ``cls`` (by its module's file, see the module docstring)."""
    name = getattr(cls, "__module__", "") or ""
    mod = sys.modules.get(name)
    file = getattr(mod, "__file__", None)
    if file:
        return file_origin(file) == "nv"
    # a live class whose module is not loaded (or has no file) was not loaded from NVIDIA's
    # snapshot, whatever its name says; names alone only decide for transported errors
    return mod is not None and getattr(mod, cls.__name__, None) is cls and \
        is_nv_module_name(name)


def is_nv_module_name(name: str) -> bool:
    """A module name the linked family's finder resolves (alias, vendor top-level names)."""
    if not name:
        return False
    from flora_deploy.nv import linker

    fam = linker.loaded_family()
    finder = getattr(fam, "_finder", None)
    if finder is None:
        return False
    try:
        return bool(finder.owns(name))
    except Exception:  # noqa: BLE001 - a probe; never let classification raise
        return False


def _frames(tb: TracebackType | None) -> Iterator[str]:
    files = []
    while tb is not None:
        files.append(tb.tb_frame.f_code.co_filename)
        tb = tb.tb_next
    return reversed(files)  # innermost first


def raising_frame(exc: BaseException) -> tuple[Origin | None, str | None]:
    """``(origin, file)`` of the innermost non-library frame of ``exc``'s traceback; ``(lib,
    None)`` when every frame is library code, ``(None, None)`` without a traceback."""
    tb = exc.__traceback__
    if tb is None:
        return None, None
    for f in _frames(tb):
        o = file_origin(f)
        if o != "lib":
            return o, f
    return "lib", None


def origin(exc: BaseException) -> Origin | None:
    """Who raised ``exc``: the class's origin when NVIDIA defines it, else the innermost
    non-library frame's (what :class:`~flora_deploy.stages.reference.NvidiaError` carries)."""
    if any(is_nv_class(c) for c in type(exc).__mro__ if c.__module__ != "builtins"):
        return "nv"
    return raising_frame(exc)[0]


# --- classification -----------------------------------------------------------------------------


def _unwrap(exc: BaseException) -> tuple[BaseException, str, str, Origin | None, bool]:
    """``(original, type_name, message, carried origin, transported)``."""
    type_name = getattr(exc, "type_name", None)
    if isinstance(type_name, str) and type_name:  # stages.reference.NvidiaError
        cause = exc.__cause__
        if cause is not None and nvidia_type_name(cause) == type_name:
            return cause, type_name, str(cause), None, False
        message = str(getattr(exc, "message", None) or exc)
        return exc, type_name, message, getattr(exc, "origin", None), True
    cls = type(exc)
    return exc, f"{cls.__module__}.{cls.__qualname__}", str(exc), None, False


def _env_failure(message: str) -> bool:
    return any(m in message for m in NVIDIA_ENVIRONMENT_FAILURES)


def classify(exc: BaseException) -> Kind:
    """``nv_failure`` | ``infra`` | ``python`` | ``cuda_fatal`` | ``oom`` (module docstring)."""
    err, type_name, message, carried, transported = _unwrap(exc)
    sent = getattr(exc, "kind", None) if transported else None
    if sent in KINDS:  # classified by the sender, with the original traceback
        return sent  # type: ignore[return-value]
    module, _, name = type_name.rpartition(".")
    if transported:
        names, roots, mro = {name}, {module.partition(".")[0]}, ()
    else:
        mro = tuple(c for c in type(err).__mro__ if c not in (BaseException, Exception, object))
        names = {c.__name__ for c in mro}
        roots = {(c.__module__ or "").partition(".")[0] for c in mro}
    low = message.lower()
    # 1. the GPU
    if any(m.lower() in low for m in OOM_MARKERS) or "OutOfMemoryError" in names:
        return "oom"
    if any(m in message for m in CUDA_FATAL_MARKERS):
        return "cuda_fatal"
    # 2. the environment
    if names & INFRA_TYPE_NAMES or roots & INFRA_MODULE_ROOTS or _env_failure(message):
        return "infra"
    if transported:
        bcls = getattr(builtins, name, None) if module in ("builtins", "") else None
        if isinstance(bcls, type) and issubclass(bcls, OSError):
            return "infra"
    elif isinstance(err, OSError | MemoryError | ConnectionError | TimeoutError):
        return "infra"
    # 3. NVIDIA's classes (and our copy of NVIDIA's context guard)
    if transported:
        if carried == "nv" or is_nv_module_name(module):
            return "nv_failure"
        if carried is None and name in NVIDIA_INPUT_ERRORS:
            return "nv_failure"
        if module in ("builtins", "") and message.strip() in NVIDIA_EPISODE_REJECTIONS:
            return "nv_failure"
        return "python" if carried in ("ours", "other") else "infra"
    from flora_deploy.nv.budget import ContextBudgetError

    if isinstance(err, ContextBudgetError | NvidiaSemanticsError) or any(
            is_nv_class(c) for c in mro if c.__module__ != "builtins"):
        return "nv_failure"
    # 4. builtins raised by NVIDIA's own frames
    where, file = raising_frame(err)
    if where == "nv":
        if file is not None and Path(file).stem in NVIDIA_LOAD_MODULES:
            return "infra"
        if isinstance(err, VERDICT_BUILTINS):
            return "nv_failure"
        return "infra"
    if where is None and type(err).__module__ == "builtins" and \
            message.strip() in NVIDIA_EPISODE_REJECTIONS:
        return "nv_failure"
    # 5. ours, or unknown
    return "python" if where in ("ours", "other") else "infra"


def terminal(kind: str) -> bool:
    """Only NVIDIA's verdict is terminal (excused); everything else is retried."""
    return kind == "nv_failure"


def replaces_container(kind: str) -> bool:
    """The container must be replaced and the start counts toward D15 quarantine."""
    return kind in CRASH_KINDS
