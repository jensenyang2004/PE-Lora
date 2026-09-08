"""Shared HTTP tar-streaming helpers for the ImgEdit HF dataset fetch scripts.

The dataset's raw per-sample data (masks, bboxes, judge scores) only exists
inside `Singleturn/*.tar.split.NNN` archives, split into multi-GB chunks with
no index. These helpers stream them directly over HTTP — reading a `.tar.split`
list back-to-back reproduces the original tar byte stream, since that's all a
`.tar.split.NNN` is — so `fetch_*.py` scripts never need a local copy of the
full archives.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import HfApi, hf_hub_url

REPO_ID = "sysuyy/ImgEdit"
REPO_TYPE = "dataset"


class MultiUrlReader:
    """Sequential (non-seekable) reader across an ordered list of URLs.

    Only `.read(size)` is implemented since tarfile's streaming mode
    (`mode="r|"`) never seeks.
    """

    def __init__(self, urls: list[str], session: requests.Session):
        self._urls = list(urls)
        self._session = session
        self._idx = -1
        self._resp: requests.Response | None = None
        self._advance()

    def _advance(self) -> None:
        if self._resp is not None:
            self._resp.close()
        self._idx += 1
        if self._idx >= len(self._urls):
            self._resp = None
            return
        url = self._urls[self._idx]
        print(f"    -> streaming {url.rsplit('/', 1)[-1]}", file=sys.stderr)
        self._resp = self._session.get(url, stream=True, timeout=60)
        self._resp.raise_for_status()
        self._resp.raw.decode_content = True

    def read(self, size: int = -1) -> bytes:
        if self._resp is None:
            return b""
        chunks: list[bytes] = []
        remaining = size if size and size > 0 else None
        while True:
            want = remaining if remaining is not None else 1 << 20
            chunk = self._resp.raw.read(want)
            if chunk:
                chunks.append(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        return b"".join(chunks)
                else:
                    return b"".join(chunks)
            else:
                self._advance()
                if self._resp is None:
                    return b"".join(chunks)

    def close(self) -> None:
        if self._resp is not None:
            self._resp.close()


def list_split_urls(api: HfApi, tar_base: str, max_splits: int | None = None) -> list[str]:
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    splits = sorted(f for f in files if f.startswith(f"Singleturn/{tar_base}.tar.split."))
    if not splits:
        raise FileNotFoundError(f"No split files found for {tar_base!r} under Singleturn/")
    if max_splits is not None:
        splits = splits[:max_splits]
    return [hf_hub_url(REPO_ID, f, repo_type=REPO_TYPE) for f in splits]


def list_split_urls_with_sizes(api: HfApi, tar_base: str, session: requests.Session) -> list[tuple[str, int]]:
    """Like `list_split_urls`, but also HEADs each split (following the
    redirect to the actual CDN URL) for its real size via Content-Length, so
    `SeekableMultiUrlReader` can map a global byte offset to (split, local
    offset) for Range requests."""
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    splits = sorted(f for f in files if f.startswith(f"Singleturn/{tar_base}.tar.split."))
    if not splits:
        raise FileNotFoundError(f"No split files found for {tar_base!r} under Singleturn/")
    out = []
    for f in splits:
        url = hf_hub_url(REPO_ID, f, repo_type=REPO_TYPE)
        resp = session.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        out.append((url, int(resp.headers["content-length"])))
    return out


class SeekableMultiUrlReader:
    """Seekable, HTTP-Range-backed reader across an ordered list of
    (url, size) parts, presented as one concatenated byte stream (same
    concatenation trick as `MultiUrlReader`, but random-access).

    Use with `tarfile.open(mode="r")` instead of the streaming `"r|"` mode:
    a seekable tarfile skips a member's data with `seek()` (zero network
    cost — verified HF's split files return HTTP 206 for arbitrary Range
    requests) instead of reading through every byte of it. That makes it
    possible to walk an entire multi-GB tar's headers and small files (e.g.
    result.json) and decide per-sample whether the large files are even
    worth fetching, before fetching them.
    """

    def __init__(self, urls_sizes: list[tuple[str, int]], session: requests.Session, chunk_size: int = 4 << 20):
        self._parts = urls_sizes
        self._offsets: list[int] = []
        offset = 0
        for _, size in urls_sizes:
            self._offsets.append(offset)
            offset += size
        self._total = offset
        self._session = session
        self._chunk_size = chunk_size
        self._pos = 0
        self._buf = b""
        self._buf_start = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self._pos + offset
        elif whence == 2:
            new_pos = self._total + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        self._pos = max(0, min(new_pos, self._total))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._total - self._pos
        if size <= 0:
            return b""
        end = self._pos + size
        if self._buf and self._pos >= self._buf_start and end <= self._buf_start + len(self._buf):
            data = self._buf[self._pos - self._buf_start : end - self._buf_start]
            self._pos = end
            return data
        data = self._fetch_range(self._pos, max(size, self._chunk_size))
        self._buf = data
        self._buf_start = self._pos
        take = data[:size]
        self._pos += len(take)
        return take

    def read_at(self, offset: int, size: int) -> bytes:
        """One-off fetch of an exact byte range, bypassing the read-ahead
        buffer and leaving the current stream position untouched."""
        return self._fetch_range(offset, size)

    def _fetch_range(self, start: int, size: int) -> bytes:
        start = max(0, min(start, self._total))
        stop = max(start, min(start + size, self._total))
        if stop <= start:
            return b""
        out = []
        for i, (url, psize) in enumerate(self._parts):
            part_start = self._offsets[i]
            part_end = part_start + psize
            if part_end <= start or part_start >= stop:
                continue
            lo = max(start, part_start) - part_start
            hi = min(stop, part_end) - part_start - 1
            resp = self._session.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=60)
            resp.raise_for_status()
            out.append(resp.content)
        return b"".join(out)

    def close(self) -> None:
        pass


def iter_grouped_samples_lazy(tf, small_files: set[str], large_files: set[str]):
    """Like `iter_grouped_samples`, but for a seekable tarfile (mode="r").

    `small_files` (e.g. result.json) are read in full immediately — tarfile
    has to touch their bytes anyway to reach the next header, so this costs
    nothing extra. `large_files` are NOT read: only their (offset_data,
    size) is recorded, since a seekable tarfile skips straight past them at
    zero network cost. Yields (sample_dir, small_bytes, large_offsets) once
    a sample directory has produced every name in small_files|large_files;
    the caller decides whether `large_files` are worth fetching (via
    `fetch_large_files`) based on `small_bytes` alone.
    """
    expected = small_files | large_files
    pending: dict[str, dict[str, object]] = {}
    for member in tf:
        if not member.isfile():
            continue
        parts = member.name.split("/")
        if len(parts) < 3:
            continue
        sample_dir = "/".join(parts[:2])
        fname = parts[-1]
        if fname not in expected:
            continue
        bucket = pending.setdefault(sample_dir, {})
        if fname in small_files:
            f = tf.extractfile(member)
            bucket[fname] = f.read() if f is not None else b""
        else:
            bucket[fname] = (member.offset_data, member.size)
        if expected.issubset(bucket.keys()):
            del pending[sample_dir]
            small_bytes = {k: v for k, v in bucket.items() if k in small_files}
            large_offsets = {k: v for k, v in bucket.items() if k in large_files}
            yield sample_dir, small_bytes, large_offsets


def fetch_large_files(reader: SeekableMultiUrlReader, large_offsets: dict[str, tuple[int, int]]) -> dict[str, bytes]:
    """Range-fetch the actual bytes for files whose (offset, size) were
    recorded by `iter_grouped_samples_lazy` — call only once a cheap filter
    on `small_bytes` has decided the sample is worth paying for. The files
    are independent byte ranges, so they're fetched concurrently (each
    HTTP Range request on this CDN has been observed to carry ~1s of fixed
    latency independent of size — worth hiding behind concurrency here)."""
    if not large_offsets:
        return {}
    with ThreadPoolExecutor(max_workers=len(large_offsets)) as ex:
        futures = {fname: ex.submit(reader.read_at, offset, size) for fname, (offset, size) in large_offsets.items()}
        return {fname: fut.result() for fname, fut in futures.items()}


def iter_grouped_samples(tf, expected_files: set[str]):
    """Yield (sample_dir, {filename: bytes}) once every file in
    `expected_files` for that sample directory has been seen. Grouping is
    keyed by directory name (not stream order) so it's correct even if
    entries for one folder aren't perfectly contiguous. Files not in
    `expected_files` (e.g. redundant duplicate images) are left alone so
    tarfile skips their body without buffering it in memory.
    """
    pending: dict[str, dict[str, bytes]] = {}
    for member in tf:
        if not member.isfile():
            continue
        parts = member.name.split("/")
        if len(parts) < 3:
            continue
        sample_dir = "/".join(parts[:2])
        fname = parts[-1]
        if fname not in expected_files:
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        bucket = pending.setdefault(sample_dir, {})
        bucket[fname] = f.read()
        if expected_files.issubset(bucket.keys()):
            del pending[sample_dir]
            yield sample_dir, bucket
