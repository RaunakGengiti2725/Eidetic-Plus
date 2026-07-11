"""Persistent embedding cache (S4): repeats and re-embeds are free across restarts.

Keyed by (model_id, embed_dim, sha256(text)) -- the model + dim are part of the key so a model
rename or an EMBED_DIM change MISSES the cache and never returns a stale-dimension vector. SQLite
(per-thread connection + WAL), the same pattern as the rest of the stack.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np


class PersistentEmbedCache:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
            with self._connections_lock:
                self._connections.add(c)
        return c

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _init(self) -> None:
        self._conn().execute(
            "CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self._conn().commit()

    @staticmethod
    def _key(model: str, dim: int, text: str) -> str:
        return hashlib.sha256(f"{model}\x1f{dim}\x1f{text}".encode()).hexdigest()

    def get(self, model: str, dim: int, text: str) -> Optional[np.ndarray]:
        row = self._conn().execute(
            "SELECT vec FROM embed_cache WHERE key=?", (self._key(model, dim, text),)).fetchone()
        if row is None:
            return None
        v = np.frombuffer(row[0], dtype=np.float32)
        return v if v.size == dim else None        # dim mismatch -> treat as a miss (never stale)

    def put(self, model: str, dim: int, text: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32)
        if v.size != dim:
            return
        self._conn().execute(
            "INSERT OR REPLACE INTO embed_cache(key, dim, vec) VALUES (?,?,?)",
            (self._key(model, dim, text), int(dim), v.tobytes()))
        self._conn().commit()

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0])
