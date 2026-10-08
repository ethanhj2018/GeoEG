"""SQLite cache for every external call (VLM, Nominatim, Overpass ...), keyed by a hash of the request."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class Cache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, value TEXT, created REAL, "
                         "PRIMARY KEY (ns, key))")
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(request) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, ns: str, request):
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE ns=? AND key=?", (ns, self.key(request))).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, ns: str, request, value) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?,?)",
                             (ns, self.key(request), json.dumps(value, ensure_ascii=False), time.time()))
            self._db.commit()

    def cached(self, ns: str, request, fn):
        """Return the cached value for (ns, request) or compute it with fn() and store it."""
        val = self.get(ns, request)
        if val is not None:
            self.hits += 1
            return val
        self.misses += 1
        val = fn()
        self.set(ns, request, val)
        return val


class RateLimiter:
    """Minimum interval between calls (per external service)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            dt = time.time() - self._last
            if dt < self.min_interval_s:
                time.sleep(self.min_interval_s - dt)
            self._last = time.time()
