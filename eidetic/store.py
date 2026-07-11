"""Mutable index/state store (SQLite). Holds MemoryRecords and bi-temporal edges.

This is the forgettable, updatable side of the system. It points at immutable raw
bytes by content_hash but never holds the only copy of anything: the substrate is
ground truth. FSRS updates and bi-temporal invalidation mutate rows here; the raw
record they reference is untouched.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .events import EventRecord
from .models import ClaimRecord, DerivedRecord, Edge, MemoryRecord, Scope, now
from .preferences import preference_dedup_key, preference_polarity, preference_update_key


class RecordStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_schema()

    # SQLite connections are not shareable across threads; keep one per thread.
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

    def __enter__(self) -> "RecordStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id   TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                namespace   TEXT NOT NULL DEFAULT 'default',
                agent_id    TEXT,
                project_id  TEXT,
                created_at  REAL NOT NULL,
                valid_at    REAL NOT NULL,
                invalid_at  REAL,
                expired_at  REAL,
                json        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash);
            CREATE INDEX IF NOT EXISTS idx_mem_valid ON memories(valid_at);
            CREATE INDEX IF NOT EXISTS idx_mem_ns ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_mem_ns_valid ON memories(namespace, valid_at);
            CREATE INDEX IF NOT EXISTS idx_mem_scope_valid
                ON memories(namespace, agent_id, project_id, valid_at);

            CREATE TABLE IF NOT EXISTS edges (
                edge_id     TEXT PRIMARY KEY,
                src         TEXT NOT NULL,
                dst         TEXT NOT NULL,
                relation    TEXT NOT NULL,
                namespace   TEXT NOT NULL DEFAULT 'default',
                agent_id    TEXT,
                project_id  TEXT,
                valid_at    REAL NOT NULL,
                invalid_at  REAL,
                created_at  REAL NOT NULL,
                expired_at  REAL,
                inferred    INTEGER NOT NULL DEFAULT 0,
                json        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_edge_src_lower ON edges(LOWER(src));
            CREATE INDEX IF NOT EXISTS idx_edge_dst_lower ON edges(LOWER(dst));
            CREATE INDEX IF NOT EXISTS idx_edge_ns ON edges(namespace);
            CREATE INDEX IF NOT EXISTS idx_edge_scope_valid
                ON edges(namespace, agent_id, project_id, valid_at);

            -- Dreaming engine DERIVED layer (additive, reversible, content-addressed).
            -- Schema centroids / multi-resolution gist / inferred facts. NEVER the store.
            CREATE TABLE IF NOT EXISTS derived (
                cid        TEXT PRIMARY KEY,
                kind       TEXT NOT NULL,
                namespace  TEXT NOT NULL DEFAULT 'default',
                level      INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                json       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_derived_ns ON derived(namespace);
            CREATE INDEX IF NOT EXISTS idx_derived_kind ON derived(kind);

            CREATE TABLE IF NOT EXISTS events (
                event_id   TEXT PRIMARY KEY,
                namespace  TEXT NOT NULL DEFAULT 'default',
                subject    TEXT, verb TEXT, object TEXT,
                start      REAL, "end" REAL, valid_at REAL,
                json       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_event_ns ON events(namespace);

            CREATE TABLE IF NOT EXISTS profiles (
                namespace  TEXT NOT NULL,
                line       TEXT NOT NULL,
                dedup_key  TEXT,
                update_key TEXT,
                polarity   TEXT,
                source_memory_id TEXT,
                content_hash TEXT,
                raw_uri TEXT,
                valid_at REAL,
                salience   REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                invalid_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_profile_ns ON profiles(namespace);

            CREATE TABLE IF NOT EXISTS claims (
                claim_id          TEXT PRIMARY KEY,
                namespace         TEXT NOT NULL DEFAULT 'default',
                agent_id          TEXT,
                project_id        TEXT,
                claim_type        TEXT NOT NULL,
                subject           TEXT,
                predicate         TEXT,
                object            TEXT,
                valid_at          REAL NOT NULL,
                invalid_at        REAL,
                source_memory_id  TEXT NOT NULL,
                json              TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claim_scope_valid
                ON claims(namespace, agent_id, project_id, valid_at);
            CREATE INDEX IF NOT EXISTS idx_claim_type_pred
                ON claims(claim_type, predicate);
            CREATE INDEX IF NOT EXISTS idx_claim_source ON claims(source_memory_id);
            """
        )
        # Migrate older DBs that predate scope / inferred columns.
        for table in ("memories", "edges"):
            cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, decl in (("namespace", "TEXT NOT NULL DEFAULT 'default'"),
                              ("agent_id", "TEXT"), ("project_id", "TEXT")):
                if col not in cols:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        ecols = {r["name"] for r in c.execute("PRAGMA table_info(edges)").fetchall()}
        if "inferred" not in ecols:
            c.execute("ALTER TABLE edges ADD COLUMN inferred INTEGER NOT NULL DEFAULT 0")
        pcols = {r["name"] for r in c.execute("PRAGMA table_info(profiles)").fetchall()}
        for col, decl in (("dedup_key", "TEXT"), ("update_key", "TEXT"), ("polarity", "TEXT"),
                          ("source_memory_id", "TEXT"), ("content_hash", "TEXT"),
                          ("raw_uri", "TEXT"), ("valid_at", "REAL"),
                          ("invalid_at", "REAL")):
            if col not in pcols:
                c.execute(f"ALTER TABLE profiles ADD COLUMN {col} {decl}")
        c.execute("CREATE INDEX IF NOT EXISTS idx_profile_active ON profiles(namespace, invalid_at)")
        c.commit()

    @staticmethod
    def _scope_clause(scope: Optional[Scope], prefix: str = "") -> tuple[str, list]:
        """Build a SQL WHERE fragment + params enforcing scope visibility."""
        if scope is None:
            return "", []
        clause = f" AND {prefix}namespace=?"
        params: list = [scope.namespace]
        if scope.agent_id is not None:
            clause += f" AND {prefix}agent_id=?"
            params.append(scope.agent_id)
        if scope.project_id is not None:
            clause += f" AND {prefix}project_id=?"
            params.append(scope.project_id)
        return clause, params

    def _profile_scope(self, namespace: str, source_memory_id: str = "",
                       scope: Optional[Scope] = None) -> Scope:
        if scope is not None:
            return scope
        rec = self.get_record(source_memory_id) if source_memory_id else None
        return rec.scope if rec is not None else Scope(namespace=namespace or "default")

    # ---- memories ---------------------------------------------------------
    def upsert_record(self, rec: MemoryRecord) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO memories "
            "(memory_id, content_hash, namespace, agent_id, project_id, "
            " created_at, valid_at, invalid_at, expired_at, json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rec.memory_id, rec.content_hash, rec.scope.namespace, rec.scope.agent_id,
             rec.scope.project_id, rec.created_at, rec.valid_at, rec.invalid_at,
             rec.expired_at, rec.model_dump_json()),
        )
        c.commit()

    def get_record(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self._conn().execute(
            "SELECT json FROM memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return MemoryRecord.model_validate_json(row["json"]) if row else None

    def get_by_hash(self, content_hash: str, scope: Optional[Scope] = None) -> Optional[MemoryRecord]:
        """Dedup is per-scope: the same raw bytes in a different namespace get a
        distinct index record (raw bytes are shared globally by the substrate)."""
        clause, params = self._scope_clause(scope)
        row = self._conn().execute(
            "SELECT json FROM memories WHERE content_hash=?" + clause + " LIMIT 1",
            [content_hash, *params],
        ).fetchone()
        return MemoryRecord.model_validate_json(row["json"]) if row else None

    def records_by_hash(self, content_hash: str, scope: Optional[Scope] = None) -> list[MemoryRecord]:
        clause, params = self._scope_clause(scope)
        rows = self._conn().execute(
            "SELECT json FROM memories WHERE content_hash=?" + clause + " ORDER BY valid_at, memory_id",
            [content_hash, *params],
        ).fetchall()
        return [MemoryRecord.model_validate_json(row["json"]) for row in rows]

    def all_records(self, scope: Optional[Scope] = None) -> list[MemoryRecord]:
        clause, params = self._scope_clause(scope)
        where = (" WHERE 1=1" + clause) if scope is not None else ""
        rows = self._conn().execute("SELECT json FROM memories" + where, params).fetchall()
        return [MemoryRecord.model_validate_json(r["json"]) for r in rows]

    def active_ids_at(self, t: Optional[float] = None, scope: Optional[Scope] = None) -> set[str]:
        """memory_ids whose bi-temporal validity holds at time t, within scope."""
        t = now() if t is None else t
        clause, params = self._scope_clause(scope)
        rows = self._conn().execute(
            "SELECT memory_id FROM memories WHERE valid_at<=? "
            "AND (invalid_at IS NULL OR invalid_at>?) "
            "AND (expired_at IS NULL OR expired_at>?)" + clause,
            [t, t, t, *params],
        ).fetchall()
        return {r["memory_id"] for r in rows}

    def active_records_at(self, t: Optional[float] = None,
                          scope: Optional[Scope] = None) -> list[MemoryRecord]:
        """Active MemoryRecords at time t, with scope and bi-temporal filters in SQL."""
        t = now() if t is None else t
        clause, params = self._scope_clause(scope)
        rows = self._conn().execute(
            "SELECT json FROM memories WHERE valid_at<=? "
            "AND (invalid_at IS NULL OR invalid_at>?) "
            "AND (expired_at IS NULL OR expired_at>?)" + clause,
            [t, t, t, *params],
        ).fetchall()
        return [MemoryRecord.model_validate_json(r["json"]) for r in rows]

    def records_in_time_range(self, lo: float, hi: float,
                              scope: Optional[Scope] = None) -> list[MemoryRecord]:
        """Records whose valid_at falls in [lo, hi] within scope (S2: bounded tag-and-capture
        windowed query instead of an O(N) full scan)."""
        clause, params = self._scope_clause(scope)
        rows = self._conn().execute(
            "SELECT json FROM memories WHERE valid_at>=? AND valid_at<=?" + clause,
            [lo, hi, *params],
        ).fetchall()
        return [MemoryRecord.model_validate_json(r["json"]) for r in rows]

    def ids_in_scope(self, scope: Optional[Scope] = None) -> set[str]:
        """All memory_ids within scope, ignoring bi-temporal validity (used for
        scope-local novelty/surprise so cross-scope content can't perturb salience)."""
        clause, params = self._scope_clause(scope)
        where = (" WHERE 1=1" + clause) if scope is not None else ""
        rows = self._conn().execute("SELECT memory_id FROM memories" + where, params).fetchall()
        return {r["memory_id"] for r in rows}

    def count(self, scope: Optional[Scope] = None) -> int:
        clause, params = self._scope_clause(scope)
        where = (" WHERE 1=1" + clause) if scope is not None else ""
        return self._conn().execute("SELECT COUNT(*) AS n FROM memories" + where, params).fetchone()["n"]

    def invalidate_record(self, memory_id: str, at: Optional[float] = None) -> None:
        """Bi-temporal close (expire) a record's system-knowledge window. Never deletes."""
        at = now() if at is None else at
        rec = self.get_record(memory_id)
        if rec is None:
            return
        rec.expired_at = at
        self.upsert_record(rec)
        for claim in self.claims_by_source(memory_id):
            if claim.invalid_at is None or claim.invalid_at > at:
                claim.invalid_at = at
                self.add_claim(claim)

    def clear_namespace(self, namespace: str) -> dict[str, object]:
        """Remove all mutable index/state rows for one namespace.

        This is an administrative reset primitive for isolated benchmark/test scopes. It never
        touches the immutable substrate blobs addressed by ``content_hash``; normal product
        forgetting should continue to use FSRS decay / bi-temporal invalidation instead.
        """
        ns = namespace or "default"
        c = self._conn()
        tables = ("memories", "edges", "events", "profiles", "derived", "claims")
        counts: dict[str, int] = {}
        with c:
            rows = c.execute("SELECT memory_id FROM memories WHERE namespace=?", (ns,)).fetchall()
            memory_ids = [r["memory_id"] for r in rows]
            for table in tables:
                counts[table] = c.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE namespace=?", (ns,)
                ).fetchone()["n"]
            for table in tables:
                c.execute(f"DELETE FROM {table} WHERE namespace=?", (ns,))
        return {"namespace": ns, "memory_ids": memory_ids, **counts}

    # ---- structured memory claims ------------------------------------------
    def add_claim(self, claim: ClaimRecord) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO claims "
            "(claim_id, namespace, agent_id, project_id, claim_type, subject, predicate, object, "
            " valid_at, invalid_at, source_memory_id, json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim.claim_id,
                claim.scope.namespace,
                claim.scope.agent_id,
                claim.scope.project_id,
                claim.claim_type,
                claim.subject,
                claim.predicate,
                claim.object,
                claim.valid_at,
                claim.invalid_at,
                claim.source_memory_id,
                claim.model_dump_json(),
            ),
        )
        c.commit()

    def add_claims(self, claims: list[ClaimRecord]) -> int:
        if not claims:
            return 0
        c = self._conn()
        with c:
            c.executemany(
                "INSERT OR REPLACE INTO claims "
                "(claim_id, namespace, agent_id, project_id, claim_type, subject, predicate, object, "
                " valid_at, invalid_at, source_memory_id, json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        claim.claim_id,
                        claim.scope.namespace,
                        claim.scope.agent_id,
                        claim.scope.project_id,
                        claim.claim_type,
                        claim.subject,
                        claim.predicate,
                        claim.object,
                        claim.valid_at,
                        claim.invalid_at,
                        claim.source_memory_id,
                        claim.model_dump_json(),
                    )
                    for claim in claims
                ],
            )
        return len(claims)

    def dedupe_claims(self, scope: Optional[Scope] = None) -> dict[str, int]:
        """Collapse duplicate claim rows minted under historic random ids (wave 0.1
        migration). Rows group by the deterministic identity key; the winner per group is
        the latest invalid_at-aware state (open beats closed, later close beats earlier,
        later created_at breaks ties) and is rewritten UNDER the deterministic id, so
        future re-ingests REPLACE the same row."""
        from eidetic.smqe.claim_extraction import deterministic_claim_id

        clause, params = self._scope_clause(scope)
        c = self._conn()
        rows = c.execute("SELECT claim_id, json FROM claims WHERE 1=1" + clause,
                         params).fetchall()
        before = len(rows)
        winners: dict[str, tuple[tuple, ClaimRecord]] = {}
        old_ids: list[str] = []
        for r in rows:
            claim = ClaimRecord.model_validate_json(r["json"])
            key = deterministic_claim_id(claim)
            old_ids.append(r["claim_id"])
            rank = (claim.invalid_at is None, claim.invalid_at or 0.0,
                    claim.created_at or 0.0)
            prev = winners.get(key)
            if prev is None or rank > prev[0]:
                winners[key] = (rank, claim)
        insert_rows = []
        for key, (_rank, claim) in winners.items():
            claim.claim_id = key
            insert_rows.append((
                claim.claim_id, claim.scope.namespace, claim.scope.agent_id,
                claim.scope.project_id, claim.claim_type, claim.subject,
                claim.predicate, claim.object, claim.valid_at, claim.invalid_at,
                claim.source_memory_id, claim.model_dump_json(),
            ))
        # Delete + re-insert in ONE transaction: a crash between a committed delete
        # and a separate insert would leave the claims table empty (repair runs this
        # across every namespace, and the index rebuild does not re-extract claims).
        with c:
            c.executemany("DELETE FROM claims WHERE claim_id=?",
                          [(cid,) for cid in old_ids])
            c.executemany(
                "INSERT OR REPLACE INTO claims "
                "(claim_id, namespace, agent_id, project_id, claim_type, subject, predicate, object, "
                " valid_at, invalid_at, source_memory_id, json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                insert_rows,
            )
        return {"before": before, "after": len(insert_rows)}

    def claims_in_scope(self, scope: Optional[Scope] = None,
                        claim_type: Optional[str] = None) -> list[ClaimRecord]:
        clause, params = self._scope_clause(scope)
        q = "SELECT json FROM claims WHERE 1=1" + clause
        if claim_type:
            q += " AND claim_type=?"
            params.append(claim_type)
        rows = self._conn().execute(q, params).fetchall()
        return [ClaimRecord.model_validate_json(r["json"]) for r in rows]

    def active_claims_at(self, t: Optional[float] = None, scope: Optional[Scope] = None,
                         claim_type: Optional[str] = None) -> list[ClaimRecord]:
        t = now() if t is None else t
        clause, params = self._scope_clause(scope, prefix="m.")
        q = (
            "SELECT cl.json FROM claims cl "
            "JOIN memories m ON m.memory_id=cl.source_memory_id "
            "WHERE cl.valid_at<=? "
            "AND (cl.invalid_at IS NULL OR cl.invalid_at>?) "
            "AND m.valid_at<=? "
            "AND (m.invalid_at IS NULL OR m.invalid_at>?) "
            "AND (m.expired_at IS NULL OR m.expired_at>?)" + clause
        )
        args: list = [t, t, t, t, t, *params]
        if claim_type:
            q += " AND cl.claim_type=?"
            args.append(claim_type)
        rows = self._conn().execute(q, args).fetchall()
        return [ClaimRecord.model_validate_json(r["json"]) for r in rows]

    def claims_by_source(self, memory_id: str) -> list[ClaimRecord]:
        rows = self._conn().execute(
            "SELECT json FROM claims WHERE source_memory_id=?", (memory_id,)
        ).fetchall()
        return [ClaimRecord.model_validate_json(r["json"]) for r in rows]

    # ---- edges ------------------------------------------------------------
    def add_edge(self, edge: Edge) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO edges "
            "(edge_id, src, dst, relation, namespace, agent_id, project_id, "
            " valid_at, invalid_at, created_at, expired_at, inferred, json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (edge.edge_id, edge.src, edge.dst, edge.relation, edge.scope.namespace,
             edge.scope.agent_id, edge.scope.project_id, edge.valid_at, edge.invalid_at,
             edge.created_at, edge.expired_at, 1 if edge.inferred else 0, edge.model_dump_json()),
        )
        c.commit()

    def all_edges(self, scope: Optional[Scope] = None, include_inferred: bool = False) -> list[Edge]:
        """Observed reads EXCLUDE machine-inferred edges by default (the inferred layer is
        separate, flagged, and provenance-tagged). Pass include_inferred=True to include them."""
        clause, params = self._scope_clause(scope)
        inf = "" if include_inferred else " AND inferred=0"
        where = " WHERE 1=1" + clause + inf if (scope is not None or not include_inferred) else ""
        rows = self._conn().execute("SELECT json FROM edges" + where, params).fetchall()
        return [Edge.model_validate_json(r["json"]) for r in rows]

    def active_edges_at(self, t: Optional[float] = None, scope: Optional[Scope] = None,
                        include_inferred: bool = False) -> list[Edge]:
        """Active graph edges at time t, with scope and inferred filters in SQL."""
        t = now() if t is None else t
        clause, params = self._scope_clause(scope)
        inf = "" if include_inferred else " AND inferred=0"
        rows = self._conn().execute(
            "SELECT json FROM edges WHERE valid_at<=? "
            "AND (invalid_at IS NULL OR invalid_at>?) "
            "AND (expired_at IS NULL OR expired_at>?)" + clause + inf,
            [t, t, t, *params],
        ).fetchall()
        return [Edge.model_validate_json(r["json"]) for r in rows]

    def active_edges_touching_many(self, names: set[str], t: Optional[float] = None,
                                   scope: Optional[Scope] = None,
                                   include_inferred: bool = False) -> list[Edge]:
        """Active edges touching any normalized entity name in `names`."""
        if not names:
            return []
        wanted = {n.strip().lower() for n in names if n.strip()}
        if not wanted:
            return []
        t = now() if t is None else t
        marks = ",".join("?" for _ in wanted)
        values = sorted(wanted)
        clause, params = self._scope_clause(scope)
        inf = "" if include_inferred else " AND inferred=0"
        rows = self._conn().execute(
            "SELECT json FROM edges WHERE valid_at<=? "
            "AND (invalid_at IS NULL OR invalid_at>?) "
            "AND (expired_at IS NULL OR expired_at>?) "
            f"AND (LOWER(src) IN ({marks}) OR LOWER(dst) IN ({marks}))" + clause + inf,
            [t, t, t, *values, *values, *params],
        ).fetchall()
        return [Edge.model_validate_json(r["json"]) for r in rows]

    def edges_touching(self, name: str, scope: Optional[Scope] = None,
                       include_inferred: bool = False) -> list[Edge]:
        clause, params = self._scope_clause(scope)
        inf = "" if include_inferred else " AND inferred=0"
        # Case-insensitive to match the system-wide entity identity: the contradiction check in
        # graph.add_fact compares with _norm() (lowercased), the read path uses LOWER(src/dst),
        # and the idx_edge_src_lower/idx_edge_dst_lower indexes exist for exactly this. A
        # case-sensitive fetch here let 'Alice' vs 'alice' escape contradiction detection and
        # leave both single-valued facts active.
        rows = self._conn().execute(
            "SELECT json FROM edges WHERE (LOWER(src)=LOWER(?) OR LOWER(dst)=LOWER(?))"
            + clause + inf, [name, name, *params]
        ).fetchall()
        return [Edge.model_validate_json(r["json"]) for r in rows]

    def invalidate_edge(self, edge_id: str, at: Optional[float] = None) -> None:
        """Close a contradicted edge (set expired_at + invalid_at). Never deletes."""
        at = now() if at is None else at
        row = self._conn().execute(
            "SELECT json FROM edges WHERE edge_id=?", (edge_id,)
        ).fetchone()
        if not row:
            return
        edge = Edge.model_validate_json(row["json"])
        edge.expired_at = at
        if edge.invalid_at is None:
            edge.invalid_at = at
        self.add_edge(edge)

    # ---- structured event calendar ---------------------------------------
    def add_event(self, ev: EventRecord) -> None:
        c = self._conn()
        c.execute(
            'INSERT OR REPLACE INTO events '
            '(event_id, namespace, subject, verb, object, start, "end", valid_at, json) '
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ev.event_id, ev.namespace, ev.subject, ev.verb, ev.object,
             ev.start, ev.end, ev.valid_at, ev.model_dump_json()),
        )
        c.commit()

    def events_in_scope(self, namespace: str = "default", *,
                        scope: Optional[Scope] = None,
                        at: Optional[float] = None) -> list[EventRecord]:
        rows = self._conn().execute(
            "SELECT json FROM events WHERE namespace=?", (namespace,)
        ).fetchall()
        events = [EventRecord.model_validate_json(r["json"]) for r in rows]
        if scope is None:
            return events
        read_at = now() if at is None else at
        narrowed = scope.agent_id is not None or scope.project_id is not None
        visible = []
        for ev in events:
            rec = self.get_record(ev.source_memory_id) if ev.source_memory_id else None
            if rec is None:
                if not ev.source_memory_id and not narrowed:
                    visible.append(ev)
                continue
            if rec.scope.visible_to(scope) and rec.is_active_at(read_at):
                visible.append(ev)
        return visible

    # ---- per-user preference profile -------------------------------------
    def add_profile_line(self, namespace: str, line: str, salience: float = 0.5,
                         dedup_key: Optional[str] = None, source_memory_id: str = "",
                         content_hash: str = "", raw_uri: str = "",
                         valid_at: Optional[float] = None,
                         scope: Optional[Scope] = None) -> None:
        c = self._conn()
        created_at = now()
        source_valid_at = created_at if valid_at is None else float(valid_at)
        effective_dedup_key = dedup_key or preference_dedup_key(line)
        update_key = preference_update_key(line)
        polarity = preference_polarity(line)
        profile_scope = self._profile_scope(namespace, source_memory_id, scope)
        if dedup_key is None:
            # Default (all existing callers): exact-string dedup -- byte-identical to before.
            rows = c.execute(
                "SELECT line, source_memory_id FROM profiles "
                "WHERE namespace=? AND line=? AND invalid_at IS NULL",
                (namespace, line),
            ).fetchall()
            if any(self._profile_scope(namespace, row["source_memory_id"]).key() == profile_scope.key()
                   for row in rows):
                return
        else:
            # Near-duplicate dedup: skip if any existing line normalizes to the same key (the
            # sentence-scan path passes a canonical preference key, so casing/spacing and simple
            # paraphrase variants of the same preference don't bloat the profile). Profiles are small.
            rows = c.execute(
                "SELECT line, source_memory_id FROM profiles WHERE namespace=? AND invalid_at IS NULL",
                (namespace,),
            ).fetchall()
            if any(
                self._profile_scope(namespace, r["source_memory_id"]).key() == profile_scope.key()
                and (
                    preference_dedup_key(r["line"]) == dedup_key
                    or " ".join(r["line"].lower().split()) == dedup_key
                )
                for r in rows
            ):
                return
        new_invalid_at: Optional[float] = None
        if update_key:
            active = c.execute(
                "SELECT rowid, line, update_key, polarity, valid_at, created_at, source_memory_id "
                "FROM profiles "
                "WHERE namespace=? AND invalid_at IS NULL",
                (namespace,),
            ).fetchall()
            invalidate: list[int] = []
            for row in active:
                if self._profile_scope(namespace, row["source_memory_id"]).key() != profile_scope.key():
                    continue
                old_key = row["update_key"] or preference_update_key(row["line"])
                if old_key != update_key:
                    continue
                old_polarity = row["polarity"] or preference_polarity(row["line"])
                same_preference = preference_dedup_key(row["line"]) == effective_dedup_key
                favorite_update = update_key.startswith("favorite:")
                opposite_polarity = (
                    polarity in {"positive", "negative"}
                    and old_polarity in {"positive", "negative"}
                    and polarity != old_polarity
                )
                if not same_preference and (favorite_update or opposite_polarity):
                    old_valid_at = (
                        float(row["valid_at"])
                        if row["valid_at"] is not None else float(row["created_at"])
                    )
                    if old_valid_at <= source_valid_at:
                        invalidate.append(int(row["rowid"]))
                    else:
                        new_invalid_at = (
                            old_valid_at
                            if new_invalid_at is None else min(new_invalid_at, old_valid_at)
                        )
            if invalidate:
                placeholders = ",".join("?" for _ in invalidate)
                c.execute(
                    f"UPDATE profiles SET invalid_at=? WHERE rowid IN ({placeholders})",
                    (source_valid_at, *invalidate),
                )
        c.execute(
            "INSERT INTO profiles "
            "(namespace, line, dedup_key, update_key, polarity, source_memory_id, content_hash, "
            " raw_uri, valid_at, salience, created_at, invalid_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                namespace, line, effective_dedup_key, update_key, polarity,
                source_memory_id, content_hash, raw_uri, source_valid_at,
                salience, created_at, new_invalid_at,
            ),
        )
        c.commit()

    def get_profile_entries(self, namespace: str = "default", *,
                            include_inactive: bool = False) -> list[dict]:
        where = "namespace=?" if include_inactive else "namespace=? AND invalid_at IS NULL"
        rows = self._conn().execute(
            f"SELECT line, salience, created_at, invalid_at, source_memory_id, content_hash, "
            f"raw_uri, valid_at FROM profiles WHERE {where} "
            "ORDER BY salience DESC, created_at DESC",
            (namespace,),
        ).fetchall()
        return [
            {
                "line": r["line"],
                "salience": float(r["salience"]),
                "created_at": r["created_at"],
                "invalid_at": r["invalid_at"],
                "source_memory_id": r["source_memory_id"] or "",
                "content_hash": r["content_hash"] or "",
                "raw_uri": r["raw_uri"] or "",
                "valid_at": r["valid_at"],
            }
            for r in rows
        ]

    def get_profile(self, namespace: str = "default", *, include_inactive: bool = False) -> list[str]:
        return [entry["line"] for entry in self.get_profile_entries(
            namespace, include_inactive=include_inactive)]

    # ---- dreaming-engine DERIVED layer (additive; never the observed store) ----------
    def add_derived(self, rec: DerivedRecord) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO derived (cid, kind, namespace, level, created_at, json) "
            "VALUES (?,?,?,?,?,?)",
            (rec.cid, rec.kind, rec.namespace, rec.level, rec.created_at, rec.model_dump_json()),
        )
        c.commit()

    def derived_in_scope(self, namespace: str = "default", kind: Optional[str] = None,
                         level: Optional[int] = None) -> list[DerivedRecord]:
        q = "SELECT json FROM derived WHERE namespace=?"
        params: list = [namespace]
        if kind is not None:
            q += " AND kind=?"
            params.append(kind)
        if level is not None:
            q += " AND level=?"
            params.append(level)
        rows = self._conn().execute(q, params).fetchall()
        return [DerivedRecord.model_validate_json(r["json"]) for r in rows]

    def derived_count(self, namespace: Optional[str] = None) -> int:
        if namespace is None:
            return self._conn().execute("SELECT COUNT(*) AS n FROM derived").fetchone()["n"]
        return self._conn().execute(
            "SELECT COUNT(*) AS n FROM derived WHERE namespace=?", (namespace,)).fetchone()["n"]
