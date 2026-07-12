"""ResearchAgenda: the frontier queue (own SQLite file, WAL, append-mostly).

Priority is the epistemic frontier order (contested resolution > unknown x info-gain
> live ask failures > repair proposals > surprise ingest > knob imbalance), computed
by ResearchTask.priority.

THE INTEGRITY WALL applies here exactly as it does to the feedback buffer: a task
whose namespace matches the benchmark harness pattern is REFUSED unless it declares
source="dev_lab" AND every sample it references is dev-split -- the lab runner is
the only caller that sets that. Research can therefore never be steered by a holdout
item.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..feedback import is_benchmark_namespace
from ..models import now
from .types import ResearchTask


class ResearchAgenda:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
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

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS agenda (
                dedup_key   TEXT PRIMARY KEY,
                ts          REAL NOT NULL,
                namespace   TEXT NOT NULL,
                priority    REAL NOT NULL,
                origin      TEXT NOT NULL,
                failure_class TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'queued',   -- queued|running|done|dropped
                attempts    INTEGER NOT NULL DEFAULT 0,
                json        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agenda_status_prio ON agenda(status, priority DESC);
            CREATE INDEX IF NOT EXISTS idx_agenda_ns ON agenda(namespace);
            """
        )
        c.commit()

    # ---- write ---------------------------------------------------------------
    def enqueue(self, task: ResearchTask) -> Optional[str]:
        """Insert or refresh (bump priority/ts) a task. Returns the dedup key, or
        None when the integrity wall refuses it."""
        if is_benchmark_namespace(task.namespace) and task.source != "dev_lab":
            return None                        # holdout/test items can never steer research
        c = self._conn()
        key = task.dedup_key
        row = c.execute("SELECT priority, attempts FROM agenda WHERE dedup_key=?",
                        (key,)).fetchone()
        if row is not None:
            priority = max(float(row["priority"]), task.priority) + 1.0   # recurrence bumps
            c.execute("UPDATE agenda SET ts=?, priority=?, status=CASE status WHEN 'done' "
                      "THEN 'done' ELSE 'queued' END, json=? WHERE dedup_key=?",
                      (now(), priority, task.to_json(), key))
        else:
            c.execute("INSERT INTO agenda(dedup_key, ts, namespace, priority, origin,"
                      " failure_class, status, json) VALUES (?,?,?,?,?,?,'queued',?)",
                      (key, now(), task.namespace, task.priority, task.origin,
                       task.failure_class.value, task.to_json()))
        c.commit()
        return key

    def pop_highest_priority(self) -> Optional[tuple[str, ResearchTask]]:
        """Highest-priority queued task -> running. Returns (dedup_key, task)."""
        c = self._conn()
        row = c.execute("SELECT dedup_key, json FROM agenda WHERE status='queued'"
                        " ORDER BY priority DESC, ts ASC LIMIT 1").fetchone()
        if row is None:
            return None
        c.execute("UPDATE agenda SET status='running', attempts=attempts+1"
                  " WHERE dedup_key=?", (row["dedup_key"],))
        c.commit()
        return row["dedup_key"], ResearchTask.from_json(row["json"])

    def mark(self, dedup_key: str, status: str) -> None:
        if status not in ("queued", "running", "done", "dropped"):
            raise ValueError(f"bad agenda status: {status}")
        c = self._conn()
        c.execute("UPDATE agenda SET status=? WHERE dedup_key=?", (status, dedup_key))
        c.commit()

    # ---- read ----------------------------------------------------------------
    def stats(self) -> dict:
        c = self._conn()
        by_status = {r["status"]: int(r["n"]) for r in c.execute(
            "SELECT status, COUNT(*) n FROM agenda GROUP BY status").fetchall()}
        by_origin = {r["origin"]: int(r["n"]) for r in c.execute(
            "SELECT origin, COUNT(*) n FROM agenda WHERE status='queued'"
            " GROUP BY origin").fetchall()}
        by_class = {r["failure_class"]: int(r["n"]) for r in c.execute(
            "SELECT failure_class, COUNT(*) n FROM agenda WHERE status='queued'"
            " GROUP BY failure_class").fetchall()}
        return {"queued": by_status.get("queued", 0),
                "running": by_status.get("running", 0),
                "done": by_status.get("done", 0),
                "dropped": by_status.get("dropped", 0),
                "queued_by_origin": by_origin,
                "queued_by_failure_class": by_class}

    def peek(self, limit: int = 10) -> list[dict]:
        c = self._conn()
        rows = c.execute("SELECT dedup_key, priority, origin, failure_class, ts FROM agenda"
                         " WHERE status='queued' ORDER BY priority DESC, ts ASC LIMIT ?",
                         (int(limit),)).fetchall()
        return [dict(r) for r in rows]
