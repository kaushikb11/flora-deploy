"""Compose and export (``output-format.md``): receipts' parts → deduped ``flora_labels_v1`` outputs.

``compose`` writes per-stage JSONL + Parquet and a ``summary.json`` under
``JobPaths.output(<config_hash>, ...)``; ``results`` reads them back (``flora results``).
"""

from flora_deploy.export.compose import ComposeError, ComposeResult, compose
from flora_deploy.export.results import iter_records, load_parquet, load_summary, store_from_uri

__all__ = ["ComposeError", "ComposeResult", "compose", "iter_records", "load_parquet",
           "load_summary", "store_from_uri"]
