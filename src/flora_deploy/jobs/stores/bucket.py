"""``BucketStore``: an S3-compatible bucket (AWS S3, Cloudflare R2, GCS via its S3 interop with
HMAC keys) as the job's state of record (PLAN.md D6).

``boto3`` is optional and imported only when no ``client`` is passed (:data:`BOTO3_INSTALL`).
R2 and GCS need ``endpoint_url`` (``https://<account>.r2.cloudflarestorage.com``,
``https://storage.googleapis.com``).

A PUT replaces the whole object atomically, so whole-file semantics come for free. S3, GCS and R2
are read-after-write consistent for GET/HEAD/LIST, but callers still treat listings as possibly
stale (``jobs.store``), so S3-compatible backends with lagging listings stay safe.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from flora_deploy.jobs.stores.local import check_list_prefix, check_rel_path

# How to add boto3 to the environment flora runs in (the README installs it with `uv tool`).
BOTO3_INSTALL = ("uv tool install --force --with boto3 git+https://github.com/kaushikb11/"
                 "flora-deploy (or `pip install boto3` in flora's environment)")

_MISSING = {"404", "NoSuchKey", "NotFound"}


def _error_code(exc: Exception) -> str | None:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return str(resp.get("Error", {}).get("Code"))
    return None


class BucketStore:
    def __init__(self, bucket: str, *, prefix: str = "", client: Any | None = None,
                 endpoint_url: str | None = None, region_name: str | None = None,
                 page_size: int = 1000):
        self.bucket = bucket
        p = prefix.strip("/")
        self.prefix = f"{p}/" if p else ""
        self.page_size = page_size
        if client is None:
            try:
                import boto3  # optional dependency
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise ImportError(f"an s3:// store needs boto3: {BOTO3_INSTALL}") from e
            client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)
        self.client = client

    def _key(self, path: str) -> str:
        return self.prefix + check_rel_path(path)

    def put_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes | bytearray):
            raise TypeError("whole-file bytes only")
        self.client.put_object(Bucket=self.bucket, Key=self._key(path), Body=bytes(data))

    def get_bytes(self, path: str) -> bytes:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=self._key(path))
        except Exception as e:
            if _error_code(e) in _MISSING:
                raise FileNotFoundError(path) from e
            raise
        body = resp["Body"]
        try:
            return body.read()
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    def exists(self, path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(path))
        except Exception as e:
            if _error_code(e) in _MISSING:
                return False
            raise
        return True

    def list(self, prefix: str) -> Iterator[str]:
        """Paginated ``list_objects_v2``; yields paths relative to the store prefix."""
        check_list_prefix(prefix)
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": self.prefix + prefix,
                                  "MaxKeys": self.page_size}
        n = len(self.prefix)
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", ()):
                yield obj["Key"][n:]
            if not resp.get("IsTruncated"):
                return
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
