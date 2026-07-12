"""KnowledgeMap: the persisted epistemic map (derived, recomputable, own SQLite file).

Two populations of cells live in one table, reconciled by `rebuild()`:
  * enumerated cells (origin="enumerator") -- recomputed from the store each rebuild;
    stale ones are closed with state history preserved in the transitions ledger.
  * minted cells (origin in ask/probe/contradiction/law/contested) -- born from live
    outcomes; only their OWN lifecycle events (proof, resolution) move them.

State transitions are append-only in `transitions` -- the map_delta artifact is a
mechanical fold over that ledger, so "Unknown shrank overnight" is auditable per cell
with a WHY (probe proof / contested resolution / promotion / re-enumeration).

`mark_known` is the ONLY door into KNOWN and it demands a VERIFIED answer carrying
citations; the proof (answer text, citation hashes) is stored on the cell.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..models import Answer, AnswerStatus, Scope, now
from .cells import CellKind, CellState, EpistemicCell, cell_id_for

_WS_RE = re.compile(r"\s+")


def _qnorm(q: str) -> str:
    return _WS_RE.sub(" ", (q or "").strip().lower())


def query_cell_subject(query: str) -> str:
    return _qnorm(query)[:300]


class KnowledgeMap:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_schema()

    # ---- plumbing (same discipline as FeedbackBuffer) ----------------------
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
            CREATE TABLE IF NOT EXISTS cells (
                cell_id      TEXT PRIMARY KEY,
                namespace    TEXT NOT NULL,
                agent_id     TEXT,
                project_id   TEXT,
                kind         TEXT NOT NULL,
                subject      TEXT NOT NULL,
                relation     TEXT NOT NULL DEFAULT '',
                state        TEXT NOT NULL,
                reason       TEXT NOT NULL DEFAULT '',
                evidence     TEXT NOT NULL DEFAULT '[]',
                proof        TEXT,
                origin       TEXT NOT NULL DEFAULT 'enumerator',
                info_gain    REAL NOT NULL DEFAULT 0.0,
                first_seen   REAL NOT NULL,
                last_updated REAL NOT NULL,
                last_probed  REAL,
                probe_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_cells_ns_state ON cells(namespace, state);
            CREATE INDEX IF NOT EXISTS idx_cells_origin ON cells(origin);

            -- Append-only state ledger: every transition, with cause. map_delta folds this.
            CREATE TABLE IF NOT EXISTS transitions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                cell_id   TEXT NOT NULL,
                namespace TEXT NOT NULL,
                from_state TEXT,
                to_state  TEXT NOT NULL,
                cause     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trans_ns_ts ON transitions(namespace, ts);
            """
        )
        c.commit()

    # ---- core mutations -----------------------------------------------------
    def _record_transition(self, c, cell_id: str, namespace: str,
                           from_state: Optional[str], to_state: str, cause: str) -> None:
        c.execute(
            "INSERT INTO transitions(ts, cell_id, namespace, from_state, to_state, cause)"
            " VALUES (?,?,?,?,?,?)",
            (now(), cell_id, namespace, from_state, to_state, cause[:300]),
        )

    def upsert_cell(self, cell: EpistemicCell, *, cause: str = "") -> str:
        """Insert or refresh a cell. A state CHANGE is ledgered; a same-state refresh
        only bumps metadata. KNOWN is protected against UNKNOWN re-enumeration (proof
        outranks a rescan) -- but an enumerator CONTESTED does reopen it: a structural
        conflict discovered UNDER an accepted proof is new information, not noise
        (day0 night wave laundered a cross-layer conflict into KNOWN this way)."""
        c = self._conn()
        row = c.execute("SELECT state, first_seen, probe_count, last_probed, proof "
                        "FROM cells WHERE cell_id=?", (cell.cell_id,)).fetchone()
        if row is not None:
            prior_state = row["state"]
            if prior_state == CellState.KNOWN.value \
                    and cell.state == CellState.UNKNOWN.value \
                    and cell.origin == "enumerator":
                return cell.cell_id            # proof outranks a re-enumeration
            cell.first_seen = float(row["first_seen"])
            cell.probe_count = max(cell.probe_count, int(row["probe_count"] or 0))
            if row["last_probed"] is not None:
                cell.last_probed = max(cell.last_probed or 0.0, float(row["last_probed"]))
            if cell.proof is None and row["proof"] and cell.state == CellState.KNOWN.value:
                cell.proof = json.loads(row["proof"])
            if prior_state != cell.state:
                self._record_transition(c, cell.cell_id, cell.namespace,
                                        prior_state, cell.state, cause or cell.reason)
        else:
            self._record_transition(c, cell.cell_id, cell.namespace,
                                    None, cell.state, cause or cell.reason)
        cell.last_updated = now()
        c.execute(
            "INSERT OR REPLACE INTO cells(cell_id, namespace, agent_id, project_id, kind,"
            " subject, relation, state, reason, evidence, proof, origin, info_gain,"
            " first_seen, last_updated, last_probed, probe_count)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            cell.to_row(),
        )
        c.commit()
        return cell.cell_id

    def get_cell(self, cell_id: str) -> Optional[EpistemicCell]:
        row = self._conn().execute("SELECT * FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
        return EpistemicCell.from_row(row) if row else None

    def close_cell(self, cell_id: str, *, cause: str) -> None:
        """A cell whose gap no longer exists (re-enumeration stopped producing it):
        transition to KNOWN is NOT implied -- the cell is deleted with a ledger entry
        ('resolved_by_reenumeration' keeps the history auditable)."""
        c = self._conn()
        row = c.execute("SELECT state, namespace FROM cells WHERE cell_id=?", (cell_id,)).fetchone()
        if row is None:
            return
        self._record_transition(c, cell_id, row["namespace"], row["state"], "CLOSED", cause)
        c.execute("DELETE FROM cells WHERE cell_id=?", (cell_id,))
        c.commit()

    def mark_known(self, cell_id: str, answer: Answer, *, cause: str) -> bool:
        """The only door into KNOWN: requires a VERIFIED answer WITH citations.
        Stores the proof object on the cell. Returns False (and changes nothing)
        for any unverified/abstained/citation-free answer."""
        if answer is None or answer.status != AnswerStatus.VERIFIED or not answer.citations:
            return False
        cell = self.get_cell(cell_id)
        if cell is None:
            return False
        proof = {
            "answer": (answer.answer or "")[:500],
            "citations": [{"memory_id": ct.memory_id, "content_hash": ct.content_hash}
                          for ct in answer.citations][:8],
            "ts": now(),
        }
        c = self._conn()
        self._record_transition(c, cell_id, cell.namespace, cell.state,
                                CellState.KNOWN.value, cause)
        c.execute(
            "UPDATE cells SET state=?, proof=?, reason=?, last_updated=? WHERE cell_id=?",
            (CellState.KNOWN.value, json.dumps(proof), cause[:240], now(), cell_id),
        )
        c.commit()
        return True

    def record_probe(self, cell_id: str) -> None:
        c = self._conn()
        c.execute("UPDATE cells SET last_probed=?, probe_count=probe_count+1 WHERE cell_id=?",
                  (now(), cell_id))
        c.commit()

    # ---- live-outcome hooks -------------------------------------------------
    def on_answer(self, query: str, answer: Answer, scope: Scope) -> Optional[str]:
        """Abstained ask -> mint/refresh an UNKNOWN query cell. Verified ask -> the
        matching query cell (if any) becomes KNOWN with the answer as proof."""
        subject = query_cell_subject(query)
        if not subject:
            return None
        cid = cell_id_for(scope.namespace, scope.agent_id, scope.project_id,
                          CellKind.QUERY.value, subject, "")
        if answer.status == AnswerStatus.VERIFIED and answer.citations:
            if self.get_cell(cid) is not None:
                self.mark_known(cid, answer, cause="verified answer (live ask)")
            return cid
        if answer.status == AnswerStatus.ABSTAINED:
            existing = self.get_cell(cid)
            recurrence = (existing.info_gain + 0.2) if existing is not None else 0.6
            cell = EpistemicCell(
                namespace=scope.namespace, agent_id=scope.agent_id,
                project_id=scope.project_id,
                kind=CellKind.QUERY.value, subject=subject, relation="",
                state=CellState.UNKNOWN.value,
                reason=(answer.note or "abstained")[:240],
                origin="ask", info_gain=min(2.0, recurrence),
            )
            return self.upsert_cell(cell, cause="abstained ask")
        return None

    def on_contradiction(self, query: str, memory_ids: list[str], scope: Scope,
                         *, note: str = "") -> str:
        subject = query_cell_subject(query)
        cell = EpistemicCell(
            namespace=scope.namespace, agent_id=scope.agent_id, project_id=scope.project_id,
            kind=CellKind.CONFLICT.value, subject=subject, relation="nli_conflict",
            state=CellState.CONTESTED.value,
            reason=(note or "active memory contradicts a drafted answer")[:240],
            evidence_ids=list(memory_ids)[:8],
            origin="contradiction", info_gain=1.2,
        )
        return self.upsert_cell(cell, cause="contradiction on ask")

    def on_probe_outcome(self, cell_id: str, answer: Answer, *, diagnosis: str) -> Optional[str]:
        """Curiosity probe result lands on its requesting cell. PASSED -> KNOWN (with
        proof); anything else refreshes reason + probe bookkeeping."""
        self.record_probe(cell_id)
        cell = self.get_cell(cell_id)
        if cell is None:
            return None
        if diagnosis == "passed":
            self.mark_known(cell_id, answer, cause="curiosity probe verified")
            return CellState.KNOWN.value
        c = self._conn()
        c.execute("UPDATE cells SET reason=?, last_updated=? WHERE cell_id=?",
                  (f"probe diagnosis: {diagnosis}"[:240], now(), cell_id))
        c.commit()
        return cell.state

    # ---- rebuild (enumerated population) ------------------------------------
    def rebuild(self, store, scope: Scope, at: Optional[float] = None) -> dict:
        """Re-derive every enumerated cell; close enumerated cells whose gap vanished;
        never touch minted cells or proven KNOWN cells. Returns count summary."""
        from .gaps import enumerate_cells
        fresh = enumerate_cells(store, scope, at)
        fresh_ids = {c.cell_id for c in fresh}
        conn = self._conn()
        prior = conn.execute(
            "SELECT cell_id, state FROM cells WHERE namespace=? AND origin='enumerator'",
            (scope.namespace,),
        ).fetchall()
        closed = 0
        for row in prior:
            if row["cell_id"] not in fresh_ids and row["state"] != CellState.KNOWN.value:
                self.close_cell(row["cell_id"], cause="gap no longer derivable from store")
                closed += 1
        for cell in fresh:
            self.upsert_cell(cell, cause="enumerator rebuild")
        return {"enumerated": len(fresh), "closed": closed, **self.counts(scope)}

    # ---- read surface --------------------------------------------------------
    def counts(self, scope: Scope) -> dict:
        c = self._conn()
        rows = c.execute(
            "SELECT state, COUNT(*) n FROM cells WHERE namespace=? GROUP BY state",
            (scope.namespace,),
        ).fetchall()
        by = {r["state"]: int(r["n"]) for r in rows}
        return {
            "known_n": by.get(CellState.KNOWN.value, 0),
            "unknown_n": by.get(CellState.UNKNOWN.value, 0),
            "contested_n": by.get(CellState.CONTESTED.value, 0),
        }

    def cells_in_state(self, scope: Scope, state: str, *, limit: int = 50,
                       order_by_gain: bool = True) -> list[EpistemicCell]:
        c = self._conn()
        order = "info_gain DESC, last_updated DESC" if order_by_gain else "last_updated DESC"
        rows = c.execute(
            f"SELECT * FROM cells WHERE namespace=? AND state=? ORDER BY {order} LIMIT ?",
            (scope.namespace, state, int(limit)),
        ).fetchall()
        return [EpistemicCell.from_row(r) for r in rows]

    def sample_frontier(self, scope: Scope, k: int, *,
                        probe_cooldown_sec: float = 3600.0) -> list[EpistemicCell]:
        """UNKNOWN cells by info gain, skipping cells probed within the cooldown so
        curiosity never thrashes one stubborn gap. CONTESTED cells are deliberately
        NOT probe fodder: a generic verified re-ask can launder one side of a live
        conflict into KNOWN (happened on the day0 demo -- the cross-layer phone cell
        closed with the losing value). Conflicts go to the resolution program
        (epistemic/contested.py), which adjudicates sides before accepting proof."""
        t = now()
        picked: list[EpistemicCell] = []
        for state in (CellState.UNKNOWN.value,):
            for cell in self.cells_in_state(scope, state, limit=k * 4):
                if cell.last_probed is not None and (t - cell.last_probed) < probe_cooldown_sec:
                    continue
                picked.append(cell)
                if len(picked) >= k:
                    return picked
        return picked

    def knowledge_map(self, scope: Scope, *, top: int = 10) -> dict:
        counts = self.counts(scope)
        gaps = [c.brief() for c in self.cells_in_state(
            scope, CellState.UNKNOWN.value, limit=top)]
        conflicts = [c.brief() for c in self.cells_in_state(
            scope, CellState.CONTESTED.value, limit=top)]
        return {**counts, "top_gaps": gaps, "top_conflicts": conflicts}

    def explain_gap(self, gap_id: str) -> Optional[dict]:
        cell = self.get_cell(gap_id)
        if cell is None:
            return None
        from .curiosity import probe_for_cell
        probe = probe_for_cell(cell)
        return {
            **cell.brief(),
            "evidence_ids": cell.evidence_ids,
            "origin": cell.origin,
            "first_seen": cell.first_seen,
            "last_probed": cell.last_probed,
            "suggested_probe": probe,
            "proof": cell.proof,
        }

    # ---- snapshot / delta (the demo artifact) --------------------------------
    def snapshot(self, scope: Scope, path: Path, *, label: str = "") -> dict:
        snap = {
            "label": label,
            "ts": now(),
            "namespace": scope.namespace,
            **self.counts(scope),
            "unknown_cells": [c.brief() for c in self.cells_in_state(
                scope, CellState.UNKNOWN.value, limit=500)],
            "contested_cells": [c.brief() for c in self.cells_in_state(
                scope, CellState.CONTESTED.value, limit=500)],
            "known_cells": [c.brief() for c in self.cells_in_state(
                scope, CellState.KNOWN.value, limit=500)],
        }
        digest_src = json.dumps(
            [snap["known_n"], snap["unknown_n"], snap["contested_n"],
             sorted(c["cell_id"] for c in snap["unknown_cells"]),
             sorted(c["cell_id"] for c in snap["contested_cells"])],
            sort_keys=True)
        snap["digest"] = hashlib.sha256(digest_src.encode()).hexdigest()[:16]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=1))
        return snap

    @staticmethod
    def delta(before: dict, after: dict) -> dict:
        """Mechanical before/after fold: which cells left UNKNOWN/CONTESTED and why
        is answered by the transitions ledger; this compares the snapshots."""
        def ids(snap, key):
            return {c["cell_id"] for c in snap.get(key, [])}
        return {
            "before": {k: before.get(k) for k in ("label", "ts", "known_n", "unknown_n",
                                                  "contested_n", "digest")},
            "after": {k: after.get(k) for k in ("label", "ts", "known_n", "unknown_n",
                                                "contested_n", "digest")},
            "unknown_delta": after.get("unknown_n", 0) - before.get("unknown_n", 0),
            "contested_delta": after.get("contested_n", 0) - before.get("contested_n", 0),
            "known_delta": after.get("known_n", 0) - before.get("known_n", 0),
            "unknown_closed": sorted(ids(before, "unknown_cells") - ids(after, "unknown_cells")),
            "unknown_opened": sorted(ids(after, "unknown_cells") - ids(before, "unknown_cells")),
            "contested_closed": sorted(ids(before, "contested_cells")
                                       - ids(after, "contested_cells")),
            "contested_opened": sorted(ids(after, "contested_cells")
                                       - ids(before, "contested_cells")),
        }

    def transitions_since(self, scope: Scope, ts: float, *, limit: int = 500) -> list[dict]:
        rows = self._conn().execute(
            "SELECT ts, cell_id, from_state, to_state, cause FROM transitions"
            " WHERE namespace=? AND ts>=? ORDER BY ts LIMIT ?",
            (scope.namespace, float(ts), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
