from __future__ import annotations

import io
import os

import pytest

from flora_deploy.jobs.store import JobPaths
from flora_deploy.jobs.stores import BucketStore, LocalStore, VolumeStore


class FakeClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """Minimal boto3-shaped S3 client: whole-object PUT/GET/HEAD and paginated LIST."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.list_calls = 0

    def put_object(self, *, Bucket, Key, Body):  # noqa: N803 - boto3 names
        assert isinstance(Body, bytes)
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys, ContinuationToken=None):  # noqa: N803
        self.list_calls += 1
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + MaxKeys]
        resp = {"Contents": [{"Key": k} for k in page], "IsTruncated": start + MaxKeys < len(keys)}
        if resp["IsTruncated"]:
            resp["NextContinuationToken"] = str(start + MaxKeys)
        return resp


class FakeVolume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def _stores(tmp_path):
    return {
        "local": LocalStore(tmp_path / "local"),
        "volume": VolumeStore(tmp_path / "vol", FakeVolume(), tmp_dir=tmp_path),
        "bucket": BucketStore("b", prefix="pre/fix", client=FakeS3(), page_size=2),
    }


@pytest.mark.parametrize("name", ["local", "volume", "bucket"])
def test_store_contract(tmp_path, name):
    s = _stores(tmp_path)[name]
    p = JobPaths("job")
    rp = p.receipt("cfg", "segcap", "segcap-00000")
    assert not s.exists(rp)
    s.put_bytes(rp, b"one")
    s.put_bytes(rp, b"two")  # whole-file replace
    assert s.exists(rp) and s.get_bytes(rp) == b"two"
    parts = [p.part("cfg", "segcap", "segcap-00000", "ta-1", k) for k in range(5)]
    for q in parts:
        s.put_bytes(q, q.encode())
    s.put_bytes("flora/job/other.json", b"x")
    prefix = parts[0].rsplit("/", 1)[0] + "/"
    assert list(s.list(prefix)) == sorted(parts)
    assert list(s.list(prefix + "ta-1-3")) == [parts[3]]  # string prefix, not just dirs
    assert list(s.list("flora/nope/")) == []
    assert set(s.list("flora/job/")) == set(parts) | {rp, "flora/job/other.json"}
    with pytest.raises(TypeError):
        s.put_bytes("a/b", "text")  # type: ignore[arg-type]
    for bad in ("/abs", "a/../b", "a//b", "", "a/./b"):
        with pytest.raises(ValueError):
            s.put_bytes(bad, b"x")
    with pytest.raises((FileNotFoundError, KeyError)):
        s.get_bytes("flora/job/missing.json")


def test_local_store_leaves_no_temp_files(tmp_path):
    s = LocalStore(tmp_path)
    for i in range(10):
        s.put_bytes("a/b/c.json", str(i).encode())
    assert os.listdir(tmp_path / "a/b") == ["c.json"]


def test_bucket_store_paginates_and_prefixes():
    c = FakeS3()
    s = BucketStore("b", prefix="/root/", client=c, page_size=3)
    for i in range(10):
        s.put_bytes(f"x/{i:02d}", b"1")
    assert ("b", "root/x/00") in c.objects
    assert list(s.list("x/")) == [f"x/{i:02d}" for i in range(10)]
    assert c.list_calls == 4
    other = FakeS3()
    other.objects[("b", "k")] = b"v"
    assert BucketStore("b", client=other).get_bytes("k") == b"v"


def test_bucket_store_propagates_other_errors():
    class Broken(FakeS3):
        def head_object(self, **kw):
            raise FakeClientError("AccessDenied")

    with pytest.raises(FakeClientError):
        BucketStore("b", client=Broken()).exists("k")


def test_volume_store_commit_batching_and_reload_guard(tmp_path):
    vol = FakeVolume()
    s = VolumeStore(tmp_path / "m", vol, commit_every=3, tmp_dir=tmp_path)
    for i in range(7):
        s.put_bytes(f"f/{i}", b"x")
    assert vol.commits == 2 and s.pending_writes == 1
    s.commit()
    assert vol.commits == 3 and s.pending_writes == 0
    with pytest.raises(RuntimeError, match="no_open_files"):
        s.reload()
    s.reload(no_open_files=True)
    assert vol.reloads == 1
    # local temp files are cleaned up; nothing staged on the mount
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".flora-tmp-")]
    assert sorted(os.listdir(tmp_path / "m/f")) == [str(i) for i in range(7)]
    with pytest.raises(ValueError):
        VolumeStore(tmp_path, commit_every=0)
