"""Inference engines for the ``balanced`` preset (PLAN.md §6.1, D2, D14).

- ``base``: the request / sampling / completion types and the ``Engine`` / ``BatchEngine``
  protocols;
- ``offline_llm``: vLLM 0.24.0's offline ``LLM`` with every output-relevant argument explicit
  (``EngineSpec``), Gate C's processor-cache rule enforced;
- ``microbatch``: the round-based micro-batcher that turns many episodes' awaits into a few
  ``llm.generate`` calls;
- ``patches.dt_bias_fp32``: the gated FP32 ``dt_bias`` ``vllm.general_plugins`` plugin (off by
  default; Gate A).

Nothing here imports ``modal`` (PLAN.md §5), and ``vllm`` is imported only when an engine is
built, so the package imports on a laptop.
"""
