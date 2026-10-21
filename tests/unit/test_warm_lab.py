"""Log parsing in the cold-start lab (no Modal calls: importing the app is lazy)."""

from flora_deploy.modal_app import coldstart_lab as lab

LOG = [
    "(EngineCore pid=67) INFO [weight_utils.py:849] Filesystem type for checkpoints: 9P. "
    "Checkpoint size: 51.75 GiB. Available RAM: 1016.74 GiB.",
    "(EngineCore pid=67) INFO [default_loader.py:430] Loading weights took 35.46 seconds",
    "INFO [gpu_model_runner.py:5255] Model loading took 51.1 GiB memory and 36.595761 seconds",
    "INFO [backends.py:394] Compiling a graph for compile range (1, 16384) takes 20.10 s",
    "INFO [backends.py:394] Compiling a graph for compile range (1, 16384) takes 25.66 s",
    "INFO [monitor.py:53] torch.compile took 65.97 s in total",
    "INFO [monitor.py:81] Initial profiling/warmup run took 81.49 s",
    "INFO [gpu_model_runner.py:6656] Graph capturing finished in 3 secs, took 0.09 GiB",
    "INFO [core.py:337] init engine (profile, create kv cache, warmup model) took 239.17 s "
    "(compilation: 65.97 s)",
    "INFO Using cache directory: /cache/x/torch_compile_cache/1f/rank_0_0/backbone for vLLM's "
    "torch.compile",
]


def test_parse_vllm_timings():
    t = lab.parse_vllm_timings(LOG)
    assert t["fs_type"] == "9P"
    assert t["loading_weights_s"] == 35.46 and t["model_loading_s"] == 36.595761
    assert t["compile_graph_s"] == {"values": [20.1, 25.66], "sum": 45.76}
    assert t["torch_compile_total_s"] == 65.97 and t["profiling_warmup_s"] == 81.49
    assert t["graph_capture_s"] == 3.0 and t["init_engine_s"] == 239.17
    assert t["cache_dir"].startswith("/cache/x/")
    assert "compile_cache_load_s" not in t

