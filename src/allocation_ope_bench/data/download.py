"""Download + cache helper for datasets not bundled with scikit-uplift."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def default_data_home() -> Path:
    home = os.environ.get("AOB_DATA_HOME", os.path.join("~", ".allocation_ope_bench_data"))
    path = Path(os.path.expanduser(home))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_download(url: str, filename: str, data_home: Path | None = None) -> Path:
    """Download ``url`` to ``data_home/filename`` unless already cached."""
    data_home = data_home or default_data_home()
    dest = data_home / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted benchmark URLs)
    tmp.rename(dest)
    return dest
