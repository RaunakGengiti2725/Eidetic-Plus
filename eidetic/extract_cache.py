"""Persistent extraction-result cache: re-ingesting identical content is free.

Extraction runs at temperature 0 over deterministic windows, yet every re-ingestion of the same
content re-paid the model call (~120k write tokens per long-haystack row). Keyed by
sha256(model \\x1f full-system-prompt \\x1f window) -- the PROMPT TEXT is part of the key, so any
prompt change is an automatic cache miss with no manual revision tag to forget. The cached value
is the RAW model output (parsed on read by the same truncation-resilient parsers); errors and
moderation skips are never cached. SQLite per-thread connection + WAL, the same pattern as the
embedding cache.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional


class PersistentExtractCache:
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
            "CREATE TABLE IF NOT EXISTS extract_cache (key TEXT PRIMARY KEY, raw TEXT)")
        self._conn().commit()

    @staticmethod
    def _key(model: str, system_prompt: str, window: str) -> str:
        return hashlib.sha256(
            f"{model}\x1f{system_prompt}\x1f{window}".encode()).hexdigest()

    def get(self, model: str, system_prompt: str, window: str) -> Optional[str]:
        row = self._conn().execute(
            "SELECT raw FROM extract_cache WHERE key=?",
            (self._key(model, system_prompt, window),)).fetchone()
        return row[0] if row is not None else None

    def put(self, model: str, system_prompt: str, window: str, raw: str) -> None:
        if not isinstance(raw, str):
            return
        self._conn().execute(
            "INSERT OR REPLACE INTO extract_cache(key, raw) VALUES (?,?)",
            (self._key(model, system_prompt, window), raw))
        self._conn().commit()

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM extract_cache").fetchone()[0])
