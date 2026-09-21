"""A size-capped LRU cache for derived raster products.

ATARRA reads imagery over the network, so every window read costs latency and the
obvious optimisation is to memoise derived arrays (band stacks, index layers,
tiles). The obvious optimisation is also how a project like this silently fills a
disk: a windowed-read pipeline can generate cached tiles far faster than anyone
notices.

This cache therefore enforces a byte ceiling. It is deliberately simple -- LRU by
access time, evicting whole entries -- because a cache that needs its own
debugging is worse than no cache.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np

from atarra.core.logging import get_logger

log = get_logger("cache")


class DiskCache:
    """Numpy array cache on disk, capped at ``max_bytes``."""

    def __init__(self, root: Path, max_bytes: int, *, name: str = "cache") -> None:
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes))
        self.name = name
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # --- key handling --------------------------------------------------------
    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # Two-level fan-out keeps directory listings small on Windows, where a
        # single directory with tens of thousands of entries gets slow.
        return self.root / digest[:2] / f"{digest}.npy"

    # --- public API ----------------------------------------------------------
    def load(self, key: str) -> np.ndarray | None:
        """Return the cached array, or ``None`` on miss."""
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            # A truncated or partial entry (e.g. interrupted write) is treated as
            # a miss and removed, rather than poisoning every later run.
            log.warning("dropping unreadable cache entry %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return None
        os.utime(path, None)  # mark as recently used
        return array

    def store(self, key: str, array: np.ndarray) -> Path:
        """Persist an array and enforce the ceiling."""
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(tmp, path)  # atomic: readers never observe a partial file
        self.evict_if_needed()
        return path

    def get_or_compute(self, key: str, compute) -> np.ndarray:
        """Return the cached array, computing and storing it on miss."""
        cached = self.load(key)
        if cached is not None:
            return cached
        array = compute()
        self.store(key, array)
        return array

    # --- eviction ------------------------------------------------------------
    def entries(self) -> list[tuple[float, int, Path]]:
        """Return ``(mtime, size, path)`` for every entry."""
        found: list[tuple[float, int, Path]] = []
        for path in self.root.rglob("*.npy"):
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append((stat.st_mtime, stat.st_size, path))
        return found

    def size_bytes(self) -> int:
        return sum(size for _, size, _ in self.entries())

    def evict_if_needed(self) -> int:
        """Evict least-recently-used entries until under the ceiling.

        Returns the number of entries removed.
        """
        with self._lock:
            entries = self.entries()
            total = sum(size for _, size, _ in entries)
            if total <= self.max_bytes:
                return 0

            # Oldest access first.
            entries.sort(key=lambda item: item[0])
            removed = 0
            for _mtime, size, path in entries:
                if total <= self.max_bytes:
                    break
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - racing process
                    continue
                total -= size
                removed += 1

            if removed:
                log.info(
                    "%s cache evicted %d entries; now %.1f MB (ceiling %.1f MB)",
                    self.name,
                    removed,
                    total / 1e6,
                    self.max_bytes / 1e6,
                )
            return removed

    def stats(self) -> dict:
        """Summary suitable for logging or a health endpoint."""
        entries = self.entries()
        total = sum(size for _, size, _ in entries)
        newest = max((mtime for mtime, _, _ in entries), default=None)
        return {
            "name": self.name,
            "root": str(self.root),
            "entries": len(entries),
            "bytes": total,
            "megabytes": round(total / 1e6, 2),
            "max_bytes": self.max_bytes,
            "usage_fraction": round(total / self.max_bytes, 4) if self.max_bytes else None,
            "last_write_age_s": round(time.time() - newest, 1) if newest else None,
        }

    def clear(self) -> None:
        """Remove every entry."""
        with self._lock:
            for _mtime, _size, path in self.entries():
                try:
                    path.unlink()
                except OSError:  # pragma: no cover
                    continue
