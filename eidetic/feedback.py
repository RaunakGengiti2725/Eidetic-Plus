"""The dev-split feedback replay buffer (SQLite) shared by the continuous optimizers.

This is the spine of the always-on daemon: the hot path emits a (query, features, arm,
reward) tuple after each answer; the idle cadence reads the buffer to update fusion
weights (FTRL/EG), bandit posteriors, and Rocchio centroids.

THE INTEGRITY WALL lives here, enforced two ways:

  1. A benchmark namespace (the harness writes ``{system}-{dataset}-g{n}-r{n}``) is
     recorded with ``is_dev=0`` -- write-for-audit only, NEVER sampled by a learner.
  2. ``sample()`` returns only ``is_dev=1`` rows. So even if a benchmark run were wired
     to emit feedback, no online learner could ever read a benchmark test item.

It is its own SQLite file (never the WORM substrate) and is purely additive -- it stores
learning signal, never the only copy of anything.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .models import now

# Datasets that constitute the held-out benchmark test set. A namespace touching any of
# these (in the harness group/run pattern) is treated as benchmark -> audit-only.
_BENCHMARK_DATASETS = ("locomo", "longmemeval", "memoryagentbench", "beam")
# The neutral harness namespace pattern: ...-g<group>-r<run>. Strong benchmark signal;
# production/user namespaces never carry this suffix.
_HARNESS_NS_RE = re.compile(r"-g\d+-r\d+$")


def is_benchmark_namespace(namespace: str) -> bool:
    """True if a namespace belongs to the neutral benchmark harness (so its feedback is
    audit-only and never feeds a learner). Defense in depth: matches either the harness
    group/run suffix or a known benchmark dataset token."""
    ns = (namespace or "").lower()
    if _HARNESS_NS_RE.search(ns):
        return True
    # Bare EXACT dataset name only (defense in depth). The loose substring/prefix match used to
    # over-flag ordinary user namespaces that merely contain a generic token like 'beam' (e.g.
    # 'team-beam-knowledge', 'beam-search-notes'), silently forcing their feedback to audit-only.
    # Every real harness namespace carries the -g<n>-r<n> suffix above, so this loses nothing.
    return ns in _BENCHMARK_DATASETS


@dataclass
class FeedbackRow:
    namespace: str
    query: str
    features: dict
    arm: str = ""
    reward: float = 0.0
    qvec: Optional[np.ndarray] = None
    ts: float = field(default_factory=now)
    is_dev: int = 1
    rowid: Optional[int] = None


class FeedbackBuffer:
    """Append-only (query, features, arm, reward) store, dev-split only for learners."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return c

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                namespace  TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                qvec       BLOB,
                features   TEXT NOT NULL,
                arm        TEXT NOT NULL DEFAULT '',
                reward     REAL NOT NULL DEFAULT 0.0,
                is_dev     INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_fb_dev ON feedback(is_dev, id);
            CREATE INDEX IF NOT EXISTS idx_fb_ns ON feedback(namespace);
            CREATE INDEX IF NOT EXISTS idx_fb_arm ON feedback(arm, is_dev);
            """
        )
        c.commit()

    # ---- write ------------------------------------------------------------
    def append(self, namespace: str, query: str, features: dict, *, arm: str = "",
               reward: float = 0.0, qvec: Optional[np.ndarray] = None,
               ts: Optional[float] = None) -> int:
        """Record one feedback tuple. A benchmark namespace is forced to is_dev=0
        (audit-only); everything else is learnable dev data. Returns the row id."""
        is_dev = 0 if is_benchmark_namespace(namespace) else 1
        blob = None
        if qvec is not None:
            blob = np.asarray(qvec, dtype=np.float32).tobytes()
        c = self._conn()
        cur = c.execute(
            "INSERT INTO feedback(ts, namespace, query_hash, qvec, features, arm, reward, is_dev)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (float(ts if ts is not None else now()), str(namespace),
             str(abs(hash(query)) % (1 << 62)), blob, json.dumps(features),
             str(arm), float(reward), int(is_dev)),
        )
        c.commit()
        return int(cur.lastrowid)

    # ---- read (LEARNERS: dev-only by construction) ------------------------
    def sample(self, limit: int = 256, *, arm: Optional[str] = None,
               dim: Optional[int] = None) -> list[FeedbackRow]:
        """Most-recent learnable rows (is_dev=1 ONLY). Optionally filter by arm. This is
        the only read path a learner should use -- it can never return a benchmark item."""
        c = self._conn()
        if arm is None:
            rows = c.execute(
                "SELECT * FROM feedback WHERE is_dev=1 ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM feedback WHERE is_dev=1 AND arm=? ORDER BY id DESC LIMIT ?",
                (str(arm), int(limit)),
            ).fetchall()
        return [self._row(r, dim) for r in rows]

    def arm_stats(self) -> dict[str, dict[str, float]]:
        """Per-arm (pulls, mean reward) over dev rows -- the sufficient statistics a
        bandit needs without re-reading every tuple."""
        c = self._conn()
        rows = c.execute(
            "SELECT arm, COUNT(*) n, AVG(reward) mean, SUM(reward) total "
            "FROM feedback WHERE is_dev=1 GROUP BY arm"
        ).fetchall()
        return {r["arm"]: {"n": float(r["n"]), "mean": float(r["mean"] or 0.0),
                           "total": float(r["total"] or 0.0)} for r in rows}

    def count(self, *, dev_only: bool = True) -> int:
        c = self._conn()
        q = "SELECT COUNT(*) n FROM feedback" + (" WHERE is_dev=1" if dev_only else "")
        return int(c.execute(q).fetchone()["n"])

    def clear(self, namespace: Optional[str] = None) -> None:
        """Drop rows (a namespace, or all). Used by the benchmark adapter reset so a
        benchmark namespace never lingers; safe because the buffer is purely additive
        learning signal, not a record of truth."""
        c = self._conn()
        if namespace is None:
            c.execute("DELETE FROM feedback")
        else:
            c.execute("DELETE FROM feedback WHERE namespace=?", (str(namespace),))
        c.commit()

    @staticmethod
    def _row(r: sqlite3.Row, dim: Optional[int]) -> FeedbackRow:
        qvec = None
        if r["qvec"] is not None:
            qvec = np.frombuffer(r["qvec"], dtype=np.float32)
            if dim is not None and qvec.size != dim:
                qvec = qvec[:dim] if qvec.size > dim else qvec
        return FeedbackRow(
            namespace=r["namespace"], query="", features=json.loads(r["features"]),
            arm=r["arm"], reward=float(r["reward"]), qvec=qvec, ts=float(r["ts"]),
            is_dev=int(r["is_dev"]), rowid=int(r["id"]),
        )
