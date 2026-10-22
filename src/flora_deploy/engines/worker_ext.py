"""vLLM worker extension (``worker_extension_cls``): flora's checks that must run inside vLLM's
workers, called by name through ``LLM.collective_rpc("flora_...")`` (review 8a R5).

vLLM 0.24's V1 engine runs in its own process by default (``VLLM_ENABLE_V1_MULTIPROCESSING=1``)
and ``SyncMPClient.collective_rpc`` sends the method through ``call_utility``, whose msgpack
encoder refuses a function unless ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``
(``vllm/v1/serial_utils.py:221-231``). A method *name* is a plain string; vLLM injects this
class into the worker class's bases (``vllm/v1/worker/worker_base.py:261-285``) so the name
resolves on every worker. Results must be msgpack-native (dicts of str -> int here).

Importable without torch or vLLM: vLLM resolves the class by its qualified name
(:data:`WORKER_EXTENSION`) inside the engine process. Read-only: no method changes the model or
its numerics."""

from __future__ import annotations

WORKER_EXTENSION = "flora_deploy.engines.worker_ext.FloraWorkerExtension"
GDN_DTYPES_RPC = "flora_gdn_dtypes"
KV_LAYOUT_RPC = "flora_kv_cache_layout"
GDN_PARAMS = ("dt_bias", "A_log")


def gdn_dtype_counts(model) -> dict[str, int]:
    """``{"dt_bias:float32": n, "A_log:float32": m, ...}`` over ``model.named_parameters()``."""
    counts: dict[str, int] = {}
    for name, p in model.named_parameters():
        tail = name.rsplit(".", 1)[-1]
        if tail in GDN_PARAMS:
            k = f"{tail}:{str(p.dtype).replace('torch.', '')}"
            counts[k] = counts.get(k, 0) + 1
    return counts


def kv_layout(kv_cache_config) -> dict:
    """vLLM 0.24's ``KVCacheConfig`` (``model_runner.kv_cache_config``) as msgpack-native data:
    ``{"num_blocks", "groups": [{"spec": <KVCacheSpec class name>, "block_size", "layers"}]}``."""
    groups = [{"spec": type(g.kv_cache_spec).__name__,
               "block_size": int(g.kv_cache_spec.block_size),
               "layers": len(g.layer_names)} for g in kv_cache_config.kv_cache_groups]
    return {"num_blocks": int(kv_cache_config.num_blocks), "groups": groups}


class FloraWorkerExtension:
    """Mixed into vLLM's GPU worker; methods are prefixed ``flora_`` (vLLM refuses an extension
    attribute the worker class already has)."""

    def flora_gdn_dtypes(self) -> dict[str, int]:
        return gdn_dtype_counts(self.model_runner.model)  # type: ignore[attr-defined]

    def flora_kv_cache_layout(self) -> dict:
        return kv_layout(self.model_runner.kv_cache_config)  # type: ignore[attr-defined]
