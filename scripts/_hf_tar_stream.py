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
