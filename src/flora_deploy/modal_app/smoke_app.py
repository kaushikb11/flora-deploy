"""``flora-image-smoke``: CPU-only checks of every image the GPU session uses, for cents.

What fails on a billed GPU cold start but needs no GPU to find: a module that does not import in
the hash-locked image, a lock that did not install as written, a vLLM engine argument vLLM 0.24.0
refuses, a ``vllm.general_plugins`` entry point that does not resolve, a derived-weight receipt
that does not verify, an env var the app factory forgot. One **CPU** function (no ``gpu=``) per
worker class, each on the exact image that worker runs on (``images.*_image()``, so the same
image fingerprint):

========================  ===========  ==================================================
function                  image        worker class it stands for (``factory.WORKERS``)
========================  ===========  ==================================================
``smoke_segment``         img_vllm     ``balanced_app.SegmentWorker`` (balanced preset)
``smoke_caption``         img_vllm     ``balanced_app.CaptionWorker`` (balanced preset)
``smoke_reward``          img_vllm     ``workers_app.RewardWorker`` (the Rewarder image)
``smoke_segcap``          img_hfref    ``reference_app.ReferenceWorker`` (exact preset)
``smoke_orchestrator``    img_cpu      ``factory.orchestrator_run``
========================  ===========  ==================================================

Each function gets, as its Modal ``env``, exactly the env ``factory.plan_app`` plans for that
object (from the shipped ``balanced_lerobot.yaml`` / ``exact_lerobot.yaml`` templates and the
derived-weight receipts in :data:`RECEIPTS`), and the same read-only weights / cache Volumes.
Store and media Volumes are not mounted and nothing is written anywhere. Checks:

1. **imports**: every module of the worker's logic group (``config.logic.modules_for``) and its
   startup imports, plus vLLM / transformers / torch; installed versions against
   ``config.pins.RUNTIME`` and against every line of the image's lock;
2. **extensions** (balanced): ``engines.worker_ext.FloraWorkerExtension`` resolves the way vLLM
   resolves it; the FP32 ``dt_bias`` ``vllm.general_plugins`` entry point resolves, loads and
   patches its target in a child process; the exact engine kwargs (``spec_for`` on the pool GPU)
   are vLLM ``EngineArgs`` fields of the right types, construct an ``EngineArgs``, and
   ``create_engine_config()`` is attempted (reported, not required: it may need CUDA);
   ``VLLM_ALLOW_INSECURE_SERIALIZATION`` is unset. The Rewarder: NVIDIA's own ``load_llm`` call
   is captured (``vllm.LLM`` replaced by a recorder) and checked the same way;
3. **weights**: ``balanced_app.verify_derived_weights`` for ``captioner_bf16`` (receipt hash vs
   the recorded receipt, content digest vs ``pins.DERIVED_PINS``, listing, sizes, safetensors
   headers, small files hashed) with ``full=False``: the ~55 GB byte hash is skipped on purpose
   (CPU seconds cost money; the GPU worker runs it by default). ``tempseg_merged`` must fail
   closed with the "not pinned" message. The exact worker: ``captioner_load.verify_derived``
   (fast) and the toolchain check against this image's torch / transformers / safetensors;
4. **env**: the planned env is present and equal in the container, and the code that reads it
   (stamp, determinism, cache, receipts, image fingerprint) accepts it;
5. **accounting**: each function returns its seconds and a list-price cost estimate.

``scripts/image_smoke.py`` runs all five in one ephemeral app (``app.run()``), prints the table
and stops the app; ``--dry-run`` contacts nothing. Never deployed; never requests a GPU.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

from flora_deploy.layout import CACHE_MOUNT, CACHE_VOLUME, WEIGHTS_MOUNT, WEIGHTS_VOLUME
from flora_deploy.modal_app.images import cpu_image, hfref_image, vllm_image

IMPORTED_AT = time.time()

APP_NAME = "flora-image-smoke"
CPU = 2.0
MEMORY_MIB = 8 * 1024
TIMEOUT_S = 15 * 60
# Modal list prices (CPU per physical core, memory per GiB), as rewarder_app.USD_PER_H.
USD_PER_CORE_S = 0.0472 / 3600
USD_PER_GIB_S = 0.008 / 3600
USD_PER_S = CPU * USD_PER_CORE_S + MEMORY_MIB / 1024 * USD_PER_GIB_S

# Derived-weight receipts (``weights.derived``) the planned env carries. captioner_bf16: the
# receipt file of the pinned build (docs/captioner-bf16.md). tempseg_merged: the merger-v1 receipt
# file now on flora-weights (no content digest is pinned for it yet, so it must be refused).
# ``FLORA_SMOKE_RECEIPTS`` (JSON) overrides them; scripts/image_smoke.py sets it from --receipt.
RECEIPTS_ENV = "FLORA_SMOKE_RECEIPTS"
RECEIPTS = {
    "captioner_bf16": "0e91e467c7ffdb636573e779672d11b203571f7869ca94317f1005eb700d9c65",
    "tempseg_merged": "f4a6fd272469559aa2017745025cb649d6f1da0c248730c283ef323bceae1e72",
}
# A GPU name as nvidia-smi prints it, per pool GPU (the stamp records it; no GPU here).
SMI_NAMES = {"L40S": "NVIDIA L40S", "H100": "NVIDIA H100 80GB HBM3",
             "H200": "NVIDIA H200"}


@dataclass(frozen=True)
class Target:
    name: str
    image: str  # images key
    preset: str  # template whose plan supplies the env
    obj: str  # ObjectSpec name in the plan
    kind: str | None  # shard kind (logic group); None: orchestrator
    weights: bool
    cache: bool


TARGETS: dict[str, Target] = {
    "segment": Target("segment", "img_vllm", "balanced", "SegmentWorker", "segment", True, True),
    "caption": Target("caption", "img_vllm", "balanced", "CaptionWorker", "caption", True, True),
    "reward": Target("reward", "img_vllm", "balanced", "RewardWorker", "reward.terminal", True,
                     True),
    "segcap": Target("segcap", "img_hfref", "exact", "ReferenceWorker", "segcap", True, False),
    "orchestrator": Target("orchestrator", "img_cpu", "balanced", "orchestrator_run", None,
                           False, False),
}
TEMPLATES = {"balanced": "balanced_lerobot.yaml", "exact": "exact_lerobot.yaml"}


# ---------------------------------------------------------------------------------------------
# Local: the plans (what the factory would deploy)


def receipts() -> dict[str, str]:
    over = json.loads(os.environ.get(RECEIPTS_ENV) or "{}")
    return {**RECEIPTS, **over}


def plans() -> dict[str, Any]:
    """``factory.plan_app`` of each template, with :func:`receipts`. Contacts nothing."""
    from importlib import resources

    import yaml

    from flora_deploy.config.schema import parse_config
    from flora_deploy.modal_app.factory import plan_app

    out = {}
    for preset, name in TEMPLATES.items():
        text = (resources.files("flora_deploy.config") / "templates" / name).read_text()
        cfg = parse_config(yaml.safe_load(text))
        out[preset] = plan_app(cfg, derived_receipts=receipts())
    return out


def expected() -> dict[str, dict[str, Any]]:
    """Per target: the planned env, image fingerprint and Volumes (JSON-able)."""
    ps = plans()
    out = {}
    for t in TARGETS.values():
        plan = ps[t.preset]
        o = plan.object(t.obj)
        if o.image != t.image:
            raise RuntimeError(f"{t.obj} runs on {o.image} in the plan, smoke uses {t.image}")
        if o.gpu is None and o.kind == "cls":
            raise RuntimeError(f"{t.obj}: the plan has no GPU (fake-models plan?)")
        out[t.name] = {"env": dict(o.env), "image": o.image,
                       "fingerprint": plan.image_fingerprints[o.image],
                       "volumes": {p: {"name": m.name, "read_only": m.read_only}
                                   for p, m in o.volumes.items()},
                       "gpu_in_plan": o.gpu, "app_name": plan.app_name}
    return out


PLAN_ERROR: str | None = None
EXPECTED: dict[str, dict[str, Any]] = {}
if modal.is_local():
    try:
        EXPECTED = expected()
    except Exception as e:  # noqa: BLE001 - reported by scripts/image_smoke.py
        PLAN_ERROR = f"{type(e).__name__}: {e}"


SMOKE_ENV = "FLORA_IMAGE_SMOKE"


def _env_for(name: str) -> dict[str, str]:
    """The planned env plus a marker, so the dict is never empty: Modal turns a non-empty
    ``env=`` into a Secret dependency, and the container (where nothing is planned) must
    declare the same number of dependencies as the local app or it refuses to start."""
    return {**EXPECTED.get(name, {}).get("env", {}), SMOKE_ENV: "1"}


# ---------------------------------------------------------------------------------------------
# In the container: checks


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        t = time.monotonic()
        try:
            detail = fn()
            ok = True
        except Exception as e:  # noqa: BLE001 - every check reports
            detail = {"error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc()[-3000:]}
            ok = False
        self.rows.append({"check": name, "ok": ok, "detail": detail,
                          "s": round(time.monotonic() - t, 2)})
        return detail if ok else None


def _all_dists() -> dict[str, list[tuple[str, str]]]:
    """name -> [(version, location), ...] in sys.path order (the first one is imported)."""
    import importlib.metadata as md

    out: dict[str, list[tuple[str, str]]] = {}
    for d in md.distributions():
        name = d.metadata["Name"]
        if name:
            out.setdefault(_norm(name), []).append((d.version, str(getattr(d, "_path", "?"))))
    return out


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _lock_pins(image: str) -> dict[str, str]:
    """``name -> version`` of every requirement in the image's lock (also ``pkg @ url`` lines,
    versioned by their wheel file name)."""
    from urllib.parse import unquote

    from flora_deploy.modal_app.images import lock_path

    pins = {}
    for raw in lock_path(image).read_text().splitlines():
        line = raw.split(" #", 1)[0].strip().removesuffix("\\").strip()
        if not line or line.startswith(("#", "--")):
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[_norm(name.split("[")[0])] = ver.split(";")[0].strip()
        elif " @ " in line:
            name, url = line.split(" @ ", 1)
            wheel = unquote(url.split("#", 1)[0].rsplit("/", 1)[-1])
            pins[_norm(name.strip())] = wheel.split("-")[1]
    return pins


def _runtime_pins(image: str) -> dict[str, str]:
    from flora_deploy.config.pins import RUNTIME as R

    common = {"numpy": R["numpy"], "pillow": R["pillow"], "av": R["av"],
              "huggingface-hub": R["huggingface_hub_reference"], "hf-xet": R["hf_xet"]}
    if image == "img_vllm":
        return {**common, "vllm": R["vllm"], "transformers": R["transformers"],
                "torch": R["torch_reference"], "qwen-vl-utils": R["qwen_vl_utils"],
                "decord": R["decord"]}
    if image == "img_hfref":
        return {**common, "torch": R["torch_hfref"], "torchvision": R["torchvision_hfref"],
                "transformers": R["transformers"], "peft": R["peft"],
                "accelerate": R["accelerate"], "safetensors": R["safetensors"]}
    return {**common, "pyarrow": R["pyarrow"], "boto3": "1.40.0"}


def check_versions(image: str) -> dict[str, Any]:
    from flora_deploy.config.pins import RUNTIME
    from flora_deploy.modal_app.images import LOCK_PYTHON

    every = _all_dists()
    have = {k: found[0][0] for k, found in every.items()}
    runtime = {k: {"pin": v, "installed": have.get(k)} for k, v in _runtime_pins(image).items()}
    bad_runtime = {k: v for k, v in runtime.items() if v["installed"] != v["pin"]}
    lock = _lock_pins(image)
    bad_lock, shadowed = {}, {}
    for k, v in lock.items():
        found = every.get(k, [])
        if have.get(k) == v:
            continue
        # Modal's container runtime ships its own client dependencies under ``/pkg`` (older
        # aiohttp, certifi, protobuf, ...); site-packages comes first on sys.path (2026-10-22),
        # so the locked wheel is what imports. Should ``/pkg`` ever come first, the locked one
        # is still installed and this holds in every Modal container: reported, not failed.
        if found and "/pkg" in found[0][1] and any(ver == v for ver, _ in found):
            shadowed[k] = {"lock": v, "found_first": found[0][0], "from": found[0][1]}
        else:
            bad_lock[k] = {"lock": v, "installed": found}
    py = ".".join(map(str, sys.version_info[:3]))
    problems = []
    if bad_runtime:
        problems.append(f"RUNTIME pins differ: {bad_runtime}")
    if bad_lock:
        problems.append(f"{len(bad_lock)} lock lines not installed as locked: "
                        f"{dict(list(bad_lock.items())[:12])}")
    if py != LOCK_PYTHON[image]:
        problems.append(f"python {py} != lock python {LOCK_PYTHON[image]}")
    out = {"python": py, "modal_client_in_container": modal.__version__,
           "modal_client_pin": RUNTIME["modal_client"],
           "runtime": {k: v["installed"] for k, v in runtime.items()},
           "lock_lines": len(lock), "lock_mismatches": len(bad_lock),
           "shadowed_by_modal_runtime": shadowed}
    if problems:
        raise RuntimeError("; ".join(problems) + f" ({json.dumps(out)})")
    return out


def _module_name(rel: str) -> str:
    mod = rel.removesuffix(".py").replace("/", ".")
    mod = mod.removesuffix(".__init__") if mod != "__init__" else ""
    return "flora_deploy" + (f".{mod}" if mod else "")


# Startup imports of each worker class beyond its logic group (read from the startup code:
# balanced_app.build_runner, workers_app.RewardWorker.start / build_reward_engine,
# reference_app.ReferenceWorker.load, factory.orchestrator_run).
STARTUP_IMPORTS: dict[str, tuple[str, ...]] = {
    "balanced": ("flora_deploy.modal_app.balanced_app", "flora_deploy.modal_app.workers_app",
                 "flora_deploy.modal_app.warm", "flora_deploy.workers_impl.reward",
                 "flora_deploy.workers_impl.certstamp", "flora_deploy.config.presets",
                 "flora_deploy.config.receipts", "flora_deploy.prep.derived",
                 "flora_deploy.prep.merge_tempseg", "flora_deploy.jobs.stores.bucket",
                 "flora_deploy.jobs.stores.volume", "flora_deploy.validate.certkey",
                 "vllm", "vllm.engine.arg_utils", "torch", "transformers", "qwen_vl_utils",
                 "boto3"),
    "reward": ("flora_deploy.modal_app.workers_app", "flora_deploy.modal_app.rewarder_app",
               "flora_deploy.modal_app.warm", "flora_deploy.jobs.stores.bucket",
               "flora_deploy.jobs.stores.volume", "vllm", "torch", "transformers",
               "qwen_vl_utils", "decord", "boto3"),
    "segcap": ("flora_deploy.modal_app.reference_app", "flora_deploy.modal_app.workers_app",
               "flora_deploy.accel.identity", "flora_deploy.observe",
               "flora_deploy.workers_impl.reward", "flora_deploy.jobs.stores.volume",
               "flora_deploy.jobs.stores.bucket", "huggingface_hub", "torch", "torchvision",
               "transformers", "peft", "accelerate", "safetensors", "av"),
    "orchestrator": ("flora_deploy.modal_app.factory", "flora_deploy.modal_app.orchestrator_app",
                     "flora_deploy.modal_app.index_app", "flora_deploy.jobs.orchestrator",
                     "flora_deploy.jobs.driver", "flora_deploy.jobs.reconciler",
                     "flora_deploy.jobs.spawner", "flora_deploy.jobs.stores.bucket",
                     "flora_deploy.jobs.stores.volume", "flora_deploy.sdk", "pyarrow", "av",
                     "boto3"),
}


# Client-only (config parsing on pydantic): every runtime import of it is lazy and on a CLI /
# submit path, so the images without pydantic (img_cpu, img_hfref) never load it.
CLIENT_ONLY = ("flora_deploy.config.schema",)


def check_imports(t: Target) -> dict[str, Any]:
    import importlib.util

    from flora_deploy.config.logic import modules_for

    group = "balanced" if t.kind in ("segment", "caption") else (
        "reward" if t.kind and t.kind.startswith("reward.") else t.kind or "orchestrator")
    mods = [_module_name(m) for m in modules_for(t.kind)] + list(STARTUP_IMPORTS[group])
    skipped = [] if importlib.util.find_spec("pydantic") else [m for m in mods
                                                              if m in CLIENT_ONLY]
    mods = [m for m in mods if m not in skipped]
    failed, versions = {}, {}
    t0 = time.monotonic()
    for m in dict.fromkeys(mods):
        try:
            mod = importlib.import_module(m)
            if "." not in m and hasattr(mod, "__version__"):
                versions[m] = str(mod.__version__)
        except Exception as e:  # noqa: BLE001
            failed[m] = f"{type(e).__name__}: {e}"
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(set(mods))} modules failed to import: {failed}")
    return {"modules": len(set(mods)), "versions": versions, "client_only_skipped": skipped,
            "import_s": round(time.monotonic() - t0, 1)}


def check_env(expected_env: dict[str, str]) -> dict[str, Any]:
    missing = sorted(k for k in expected_env if k not in os.environ)
    differ = sorted(k for k, v in expected_env.items()
                    if k in os.environ and os.environ[k] != v)
    if missing or differ:
        raise RuntimeError(f"planned env not in the container: missing {missing}, "
                           f"different {differ}")
    return {"keys": len(expected_env)}


def check_fingerprint(image: str, want: str) -> dict[str, Any]:
    from flora_deploy.modal_app.images import image_fingerprint

    got = image_fingerprint(image)
    if got != want:
        raise RuntimeError(f"{image} fingerprint in the container {got} != planned {want}")
    return {"fingerprint": got[:16]}


def check_insecure_serialization() -> dict[str, Any]:
    if "VLLM_ALLOW_INSECURE_SERIALIZATION" in os.environ:
        raise RuntimeError("VLLM_ALLOW_INSECURE_SERIALIZATION is set: "
                           f"{os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']!r}")
    import vllm.envs as envs

    if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
        raise RuntimeError("vllm.envs.VLLM_ALLOW_INSECURE_SERIALIZATION is true")
    return {"unset": True}


# -- vLLM extensions and engine arguments ----------------------------------------------------


def check_worker_extension() -> dict[str, Any]:
    from flora_deploy.engines.worker_ext import GDN_DTYPES_RPC, WORKER_EXTENSION

    resolver = None
    for where in ("vllm.utils.import_utils", "vllm.utils"):
        try:
            resolver = importlib.import_module(where).resolve_obj_by_qualname
            break
        except (ImportError, AttributeError):
            continue
    if resolver is None:
        mod, _, name = WORKER_EXTENSION.rpartition(".")
        cls = getattr(importlib.import_module(mod), name)
        how = "importlib (vLLM's resolve_obj_by_qualname not found)"
    else:
        cls = resolver(WORKER_EXTENSION)
        how = f"{resolver.__module__}.resolve_obj_by_qualname"
    if not callable(getattr(cls, GDN_DTYPES_RPC, None)):
        raise RuntimeError(f"{WORKER_EXTENSION} has no {GDN_DTYPES_RPC}")
    # vLLM refuses extension attributes the worker already has (worker_base.py)
    out: dict[str, Any] = {"resolved_by": how, "rpc": GDN_DTYPES_RPC}
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as e:  # noqa: BLE001 - CUDA-only import: say what was not checked
        out["clash_check"] = f"not checked (gpu_worker import: {type(e).__name__}: {e})"
        return out
    clash = [a for a in vars(cls) if not a.startswith("__") and hasattr(Worker, a)]
    if clash:
        raise RuntimeError(f"extension attributes clash with vLLM's Worker: {clash}")
    out["clash_check"] = "no clash with vllm.v1.worker.gpu_worker.Worker"
    return out


_PLUGIN_CHILD = r"""
import importlib, importlib.metadata as md, json
from flora_deploy.engines.patches import dt_bias_fp32 as d
eps = [e for e in md.entry_points(group=d.ENTRY_POINT_GROUP) if e.name == d.ENTRY_POINT_NAME]
loaded = eps[0].load() if eps else None
from vllm.plugins import load_general_plugins
load_general_plugins()
mod, _, name = d.TARGET.rpartition(".")
cls = getattr(importlib.import_module(mod), name)
print("FLORA_CHILD " + json.dumps({
    "entry_points": [f"{e.name} = {e.value}" for e in eps],
    "loads_register": loaded is d.register,
    "patched": bool(getattr(cls, d.PATCH_MARK, False))}))
"""


def _child(code: str, env: dict[str, str], timeout: int = 300, args: tuple[str, ...] = ()
           ) -> dict[str, Any]:
    p = subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True,
                       text=True, timeout=timeout)
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith("FLORA_CHILD "):
            return json.loads(line[len("FLORA_CHILD "):])
    raise RuntimeError(f"child exit {p.returncode}: {p.stderr[-2500:]}")


def check_dtbias_plugin() -> dict[str, Any]:
    from flora_deploy.engines.patches import dt_bias_fp32 as d

    if d.enabled():
        raise RuntimeError(f"{d.ENV_FLAG}=1 in the worker env (the plugin is off by default)")
    root = Path(tempfile.mkdtemp(prefix="flora-plugin-"))
    env = {**os.environ, **d.install_entry_point(root)}
    d.check_activation(True, env)  # what OfflineEngine checks before a +dtbias build
    out = _child(_PLUGIN_CHILD, env)
    if not out["entry_points"] or not out["loads_register"] or not out["patched"]:
        raise RuntimeError(f"dt_bias plugin did not resolve/load/patch: {out}")
    off = _child(_PLUGIN_CHILD, {k: v for k, v in env.items() if k != d.ENV_FLAG})
    if off["patched"]:
        raise RuntimeError(f"dt_bias plugin patched without {d.ENV_FLAG}=1: {off}")
    return {**out, "patched_without_flag": off["patched"]}


_ENGINE_ARGS_CHILD = r"""
import dataclasses, json, sys, typing, traceback
kwargs = json.loads(sys.argv[1])
from vllm.engine.arg_utils import EngineArgs
fields = {f.name: f for f in dataclasses.fields(EngineArgs)}
try:
    hints = typing.get_type_hints(EngineArgs)
except Exception:
    hints = {}
unknown = sorted(k for k in kwargs if k not in fields)

def ok_type(value, hint):
    if hint is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    simple = (bool, int, float, str, dict, list)
    if hint in simple:
        return type(value) is hint or (hint is float and type(value) is int)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        res = [ok_type(value, a) for a in args if a is not type(None)]
        if value is None:
            return type(None) in args
        return True if None in res else any(res)
    if origin in (dict, list):
        return isinstance(value, origin)
    if origin is typing.Literal:
        return value in args
    return None

types = {}
bad_types = {}
for k, v in kwargs.items():
    if k in fields:
        h = hints.get(k)
        r = ok_type(v, h)
        types[k] = str(h if h is not None else fields[k].type)
        if r is False:
            bad_types[k] = {"value": repr(v), "type": types[k]}
out = {"unknown": unknown, "bad_types": bad_types, "fields_checked": len(kwargs) - len(unknown)}
try:
    ea = EngineArgs(**kwargs)
    out["engine_args"] = "constructed"
except Exception as e:
    out["engine_args"] = f"{type(e).__name__}: {e}"
    ea = None
if ea is not None:
    try:
        cfg = ea.create_engine_config()
        out["engine_config"] = "created"
        out["resolved"] = {
            "dtype": str(cfg.model_config.dtype),
            "architectures": list(getattr(cfg.model_config, "architectures", []) or []),
            "max_model_len": cfg.model_config.max_model_len,
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
            "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
            "worker_extension_cls": cfg.parallel_config.worker_extension_cls,
        }
    except Exception as e:
        out["engine_config"] = f"{type(e).__name__}: {str(e)[:800]}"
        out["engine_config_trace"] = traceback.format_exc()[-1500:]
print("FLORA_CHILD " + json.dumps(out, default=str))
"""


def _engine_args_report(kwargs: dict[str, Any], *, require_config: bool = False
                        ) -> dict[str, Any]:
    out = _child(_ENGINE_ARGS_CHILD, dict(os.environ), args=(json.dumps(kwargs),))
    problems = []
    if out["unknown"]:
        problems.append(f"not vLLM EngineArgs fields: {out['unknown']}")
    if out["bad_types"]:
        problems.append(f"wrong types: {out['bad_types']}")
    if out["engine_args"] != "constructed":
        problems.append(f"EngineArgs(**kwargs) raised {out['engine_args']}")
    if require_config and out.get("engine_config") != "created":
        problems.append(f"create_engine_config: {out.get('engine_config')}")
    if problems:
        raise RuntimeError("; ".join(problems) + f" ({json.dumps(out, default=str)[:3000]})")
    if out.get("engine_config") != "created":
        out["not_checked"] = ("create_engine_config() needs what a CPU container lacks (see "
                              "engine_config); field names, types and EngineArgs construction "
                              "were checked")
    out.pop("engine_config_trace", None)
    return out


def check_engine_args(kind: str, model_dir: Path) -> dict[str, Any]:
    from flora_deploy.engines.offline_llm import spec_for
    from flora_deploy.engines.worker_ext import WORKER_EXTENSION
    from flora_deploy.modal_app.balanced_app import pool_gpu

    spec = spec_for(kind, str(model_dir), pool_gpu(kind))
    kwargs = {**spec.engine_kwargs(), "worker_extension_cls": WORKER_EXTENSION}
    out = _engine_args_report(kwargs)
    out["spec_label"] = spec.label
    out["gpu"] = spec.gpu
    return out


def check_reward_engine_args() -> dict[str, Any]:
    """NVIDIA's ``engine.load_llm`` call (``stages.reward.build_engine``), captured: ``vllm.LLM``
    is replaced by a recorder for the call, so no engine is built."""
    import vllm

    from flora_deploy.modal_app import rewarder_app as ra
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.stages import reward

    nv = load_family(Family.REWARDER, str(ra.SNAPSHOT))
    seen: list[dict[str, Any]] = []

    class Recorder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if args:
                kwargs = {"model": args[0], **kwargs}
            seen.append(kwargs)

    patched = []
    for mod in {vllm, nv.engine}:
        if getattr(mod, "LLM", None) is not None:
            patched.append((mod, mod.LLM))
            mod.LLM = Recorder
    try:
        _llm, spec = reward.build_engine(nv, str(ra.SNAPSHOT))
    finally:
        for mod, orig in patched:
            mod.LLM = orig
    if len(seen) != 1:
        raise RuntimeError(f"NVIDIA's load_llm made {len(seen)} LLM(...) calls")
    kwargs = json.loads(json.dumps(seen[0], default=str))
    out = _engine_args_report(kwargs)
    out["load_llm_kwargs"] = sorted(kwargs)
    out["spec"] = spec.as_json()
    return out


# -- weights -----------------------------------------------------------------------------------


def check_captioner_bf16() -> dict[str, Any]:
    from flora_deploy.modal_app.balanced_app import (
        check_extra_state,
        derived_dir,
        recorded_receipts,
        verify_derived_weights,
    )

    d = derived_dir("captioner_bf16")
    out = verify_derived_weights("captioner_bf16", d, recorded_receipts().get("captioner_bf16"),
                                 full=False)
    bad = [p for p in check_extra_state(d) if p.startswith("model.visual")]
    if bad:
        raise RuntimeError(f"_extra_state under {bad}")
    out["full_byte_hash"] = "skipped (CPU seconds; the GPU worker runs it by default)"
    return out


def check_tempseg_merged_refused() -> dict[str, Any]:
    from flora_deploy.modal_app.balanced_app import (
        DerivedWeightsError,
        derived_dir,
        recorded_receipts,
        verify_derived_weights,
    )

    d = derived_dir("tempseg_merged")
    recorded = recorded_receipts().get("tempseg_merged")
    out: dict[str, Any] = {"dir": str(d), "recorded": recorded}
    receipt = d / "derived_receipt.json"
    if receipt.is_file():  # reach the pin check even when the recorded receipt is stale
        actual = hashlib.sha256(receipt.read_bytes()).hexdigest()
        out["receipt_on_volume"] = actual
        out["recorded_matches_volume"] = actual == recorded
        recorded = actual
    try:
        verify_derived_weights("tempseg_merged", d, recorded, full=False)
    except DerivedWeightsError as e:
        msg = str(e)
        out["refused"] = msg
        if "pins.DERIVED_PINS" not in msg:
            raise RuntimeError(f"refused, but not with the 'not pinned' message: {msg}") from e
        return out
    raise RuntimeError("tempseg_merged verified, but no content digest is pinned for it")


def check_segcap_weights() -> dict[str, Any]:
    from flora_deploy.accel.captioner_load import (
        check_toolchain,
        default_derived_dir,
        verify_derived,
    )

    d = default_derived_dir(WEIGHTS_MOUNT)
    report = verify_derived(d, full=False)
    receipt = json.loads((d / "derived_receipt.json").read_text())
    tools = check_toolchain(receipt)  # what captioner_load.install checks at load
    return {"verify": report, "toolchain": tools,
            "full_byte_hash": "skipped (CPU seconds; the GPU worker runs it by default)"}


def check_segcap_layout() -> dict[str, Any]:
    """``ReferenceWorker.load`` steps 1 and 3 that need no GPU: the link-farm layout with its
    prep-receipt checks (the base's full re-hash skipped), NVIDIA's code and ``models.json``."""
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_path
    from flora_deploy.modal_app.reference_app import HUB_ROOT
    from flora_deploy.modal_app.warm import build_linked_layout
    from flora_deploy.nv import Family, load_family
    from flora_deploy.stages.reference import check_nv_models, check_prep_receipt

    layout = build_linked_layout(weights_root=WEIGHTS_MOUNT, hub_root=HUB_ROOT)
    receipts = {k: bool(check_prep_receipt(PINS_BY_KEY[k], Path(src)))
                for k, src in layout.sources.items()}
    import huggingface_hub.constants as hfc

    if not hfc.HF_HUB_OFFLINE:
        raise RuntimeError("HF_HUB_OFFLINE is off in the container")
    if Path(hfc.HF_HUB_CACHE) != layout.hub_cache:
        raise RuntimeError(f"HF_HUB_CACHE {hfc.HF_HUB_CACHE} != {layout.hub_cache}")
    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"],
                                                       WEIGHTS_MOUNT)))
    check_nv_models(nv.config.MODELS)
    if Path(nv.config.MODEL_CACHE) != layout.model_cache:
        raise RuntimeError(f"MODEL_CACHE {nv.config.MODEL_CACHE} != {layout.model_cache}")
    return {"receipts": receipts, "hub_cache": str(layout.hub_cache)}


def check_balanced_nv(kind: str, model_dir: Path) -> dict[str, Any]:
    """``build_runner`` step 4 up to the engine: NVIDIA's code, the processor, the accel set."""
    from flora_deploy.config.pins import PINS_BY_KEY
    from flora_deploy.layout import snapshot_path
    from flora_deploy.modal_app.balanced_app import ENGINE_KINDS
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family
    from flora_deploy.workers_impl.balanced import apply_accel

    nv = load_family(Family.SEGCAP, Path(snapshot_path(PINS_BY_KEY["tempseg_code"],
                                                       WEIGHTS_MOUNT)))
    processor = nv.models.processor(ENGINE_KINDS[kind], model_dir)
    active = apply_accel(nv)
    return {"processor": type(processor).__name__,
            "accel": active.provenance() if active else None}


def check_reward_nv() -> dict[str, Any]:
    from flora_deploy.modal_app import rewarder_app as ra
    from flora_deploy.nv import Family
    from flora_deploy.nv.linker import load_family

    nv = load_family(Family.REWARDER, str(ra.SNAPSHOT))
    processor = nv.engine.load_processor(str(ra.SNAPSHOT))
    return {"snapshot": str(ra.SNAPSHOT), "processor": type(processor).__name__}


# -- env consumers -----------------------------------------------------------------------------


def check_balanced_env(kind: str) -> dict[str, Any]:
    from flora_deploy.config.pins import DERIVED_PINS_BY_KEY
    from flora_deploy.config.presets import get_preset
    from flora_deploy.config.receipts import pin_key
    from flora_deploy.engines.offline_llm import normalize_gpu
    from flora_deploy.modal_app.balanced_app import (
        ENGINE_KINDS,
        cache_env,
        pool_gpu,
        recorded_receipts,
    )
    from flora_deploy.modal_app.factory import CACHE_MODE_ENV
    from flora_deploy.workers_impl.certstamp import stamp_from_env

    preset = get_preset(os.environ.get("FLORA_PRESET", "balanced"))
    names = preset.derived_weights.get(kind, ())  # type: ignore[call-overload]
    rec = recorded_receipts()
    missing = [n for n in names if not rec.get(n)]
    if missing:
        raise RuntimeError(f"FLORA_DERIVED_RECEIPTS lacks {missing}")
    gpu = pool_gpu(kind)
    if os.environ.get(CACHE_MODE_ENV) != "ro":
        raise RuntimeError(f"{CACHE_MODE_ENV}={os.environ.get(CACHE_MODE_ENV)!r}, want ro")
    # stamp env: what the worker stamps (content digests: the pinned ones; unpinned -> absent)
    # (an unpinned one never reaches the stamp in build_runner: a placeholder stands in)
    content = {n: DERIVED_PINS_BY_KEY[pin_key(n)].content_sha256 or "0" * 64 for n in names}
    stamp = stamp_from_env(kind, gpu=SMI_NAMES.get(normalize_gpu(gpu), gpu),
                           derived_content=content)
    cenv = cache_env(ENGINE_KINDS[kind])
    os.environ.update(cenv)  # as build_runner, before vLLM is imported
    seeded = (Path(str(CACHE_MOUNT)) / "balanced" / ENGINE_KINDS[kind]).is_dir()
    return {"receipts": sorted(rec), "pool_gpu": gpu, "stamp": repr(stamp)[:300],
            "cache_env": cenv, "cache_seed_on_volume": seeded}


def check_reward_env() -> dict[str, Any]:
    from flora_deploy.modal_app import rewarder_app as ra
    from flora_deploy.modal_app.factory import CACHE_MODE_ENV
    from flora_deploy.modal_app.workers_app import cache_env_for, reward_settings_from_env

    want = os.environ.get(ra.IMAGE_FP_ENV)
    if not want or want != ra.IMAGE_FP:
        raise RuntimeError(f"{ra.IMAGE_FP_ENV}={want!r} != rewarder_app.IMAGE_FP {ra.IMAGE_FP}")
    mode = os.environ.get(CACHE_MODE_ENV, "")
    cenv = cache_env_for(mode)
    planned = {k: os.environ.get(k) for k in cenv}
    if planned != cenv:
        raise RuntimeError(f"cache env: planned {planned} != cache_env_for({mode!r}) {cenv}")
    settings = reward_settings_from_env(gpu=None)
    return {"cache_mode": mode, "cache_key": ra.cache_key(),
            "seeded": Path(ra.cache_env()["VLLM_CACHE_ROOT"]).parent.is_dir(),
            "terminal_mode": settings.terminal_mode}


def check_segcap_env() -> dict[str, Any]:
    from flora_deploy.accel import determinism as det
    from flora_deploy.modal_app.reference_app import (
        configured_determinism,
        episode_concurrency,
        job_worker_env,
    )
    from flora_deploy.modal_app.warm import linked_offline_env
    from flora_deploy.workers_impl.segcap import scheduling_from_env

    mode = os.environ.get("FLORA_DETERMINISM", "")
    want = {**linked_offline_env(), **det.container_env(mode)}
    bad = {k: os.environ.get(k) for k, v in want.items() if os.environ.get(k) != v}
    if bad:
        raise RuntimeError(f"offline/determinism env differs: {bad} (want {want})")
    if configured_determinism() != mode:
        raise RuntimeError(f"hashed determinism {configured_determinism()!r} != {mode!r}")
    if not job_worker_env():
        raise RuntimeError("job_worker_env() is false: the store env is not set")
    return {"determinism": mode, "episode_concurrency": episode_concurrency(),
            "scheduling": repr(scheduling_from_env(os.environ))[:300],
            "offline_keys": sorted(want)}


def check_orchestrator_env() -> dict[str, Any]:
    from flora_deploy.modal_app.factory import APP_ENV, WATCH_ENV

    missing = [k for k in (APP_ENV, WATCH_ENV, "FLORA_STORE", "FLORA_CONFIG_HASH",
                           "FLORA_SCALEDOWN_S") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"orchestrator env lacks {missing}")
    return {"app": os.environ[APP_ENV], "watch": os.environ[WATCH_ENV]}


# -- driver ------------------------------------------------------------------------------------


def run_target(name: str, exp: dict[str, Any]) -> dict[str, Any]:
    t_start = time.time()
    t = TARGETS[name]
    c = Checks()
    c.run("env: planned env present", lambda: check_env(exp["env"]))
    c.run("env: image fingerprint", lambda: check_fingerprint(t.image, exp["fingerprint"]))
    if t.kind in ("segment", "caption"):
        c.run("env: receipts/stamp/cache", lambda: check_balanced_env(t.kind))
    elif t.name == "reward":
        c.run("env: image fp/cache/settings", check_reward_env)
    elif t.name == "segcap":
        c.run("env: offline/determinism", check_segcap_env)
    else:
        c.run("env: orchestrator", check_orchestrator_env)
    c.run("imports", lambda: check_imports(t))
    c.run("versions vs pins + lock", lambda: check_versions(t.image))
    if t.image == "img_vllm":
        c.run("ext: no insecure serialization", check_insecure_serialization)
    if t.kind in ("segment", "caption"):
        from flora_deploy.modal_app.balanced_app import DERIVED_FOR_KIND, derived_dir

        model_dir = derived_dir(DERIVED_FOR_KIND[t.kind])
        c.run("ext: FloraWorkerExtension", check_worker_extension)
        c.run("ext: dt_bias plugin entry point", check_dtbias_plugin)
        c.run("ext: EngineArgs", lambda: check_engine_args(t.kind, model_dir))
        if t.kind == "caption":
            c.run("weights: captioner_bf16 (fast)", check_captioner_bf16)
        else:
            c.run("weights: tempseg_merged refused", check_tempseg_merged_refused)
        c.run("nv: code + processor + accel", lambda: check_balanced_nv(t.kind, model_dir))
    elif t.name == "reward":
        c.run("nv: rewarder code + processor", check_reward_nv)
        c.run("ext: NVIDIA load_llm EngineArgs", check_reward_engine_args)
    elif t.name == "segcap":
        c.run("weights: captioner_bf16 + toolchain", check_segcap_weights)
        c.run("nv: linked layout + models.json", check_segcap_layout)
    seconds = round(time.time() - t_start, 1)
    since_import = round(time.time() - IMPORTED_AT, 1)
    return {"target": name, "image": t.image, "ok": all(r["ok"] for r in c.rows),
            "checks": c.rows, "function_s": seconds, "since_import_s": since_import,
            "usd_estimate": round(since_import * USD_PER_S, 5)}


# ---------------------------------------------------------------------------------------------
# Modal objects (CPU only: no gpu= anywhere in this module)

app = modal.App(APP_NAME)
_weights = modal.Volume.from_name(WEIGHTS_VOLUME, version=2).read_only()
_cache = modal.Volume.from_name(CACHE_VOLUME, version=2).read_only()


def _volumes(t: Target) -> dict[str, Any]:
    v: dict[str, Any] = {}
    if t.weights:
        v[str(WEIGHTS_MOUNT)] = _weights
    if t.cache:
        v[str(CACHE_MOUNT)] = _cache
    return v


_IMAGES = {"img_vllm": vllm_image, "img_hfref": hfref_image, "img_cpu": cpu_image}


def _function(name: str):
    t = TARGETS[name]
    return dict(image=_IMAGES[t.image](), cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_S,
                max_containers=1, volumes=_volumes(t), env=_env_for(name))


@app.function(**_function("segment"))
def smoke_segment(exp: dict) -> dict:
    return run_target("segment", exp)


@app.function(**_function("caption"))
def smoke_caption(exp: dict) -> dict:
    return run_target("caption", exp)


@app.function(**_function("reward"))
def smoke_reward(exp: dict) -> dict:
    return run_target("reward", exp)


@app.function(**_function("segcap"))
def smoke_segcap(exp: dict) -> dict:
    return run_target("segcap", exp)


@app.function(**_function("orchestrator"))
def smoke_orchestrator(exp: dict) -> dict:
    return run_target("orchestrator", exp)


FUNCTIONS = {"segment": smoke_segment, "caption": smoke_caption, "reward": smoke_reward,
             "segcap": smoke_segcap, "orchestrator": smoke_orchestrator}
