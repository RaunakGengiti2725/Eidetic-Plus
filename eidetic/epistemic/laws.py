"""Law induction with a lifecycle: candidate -> hold -> verify-or-discard -> law /
falsified. The organism does not just store what it saw -- it hypothesizes REGULARITY
and tests it against its own witness.

Candidates come from the SAME AnyBURL-style miner the dreaming engine ships
(`eidetic/dreaming/rules.py`, token-free). The lifecycle here adds the epistemic
discipline on top:

  candidate   a mined rule above floor confidence/support, persisted with provenance
  verifying   its untested predictions are enumerated as UNKNOWN law_prediction cells
              (gaps.py g2) and probed by curiosity through the REAL prove path
  law         >= `promote_min_checks` predictions VERIFIED and zero contradictions;
              its inferred edges may then be written through the EXISTING guarded
              inferred-edge path (flagged, provenance-tagged, excluded from observed
              reads -- graph.py's inferred layer, nothing new)
  falsified   any probe VERIFIED the negation / hit a contradiction -> the law is
              retired AND the counterexample mints a CONTESTED cell: a broken law is
              information, not noise

Everything lands in the map's own sqlite (laws table). Zero model calls in this
module; probing spends only through curiosity's governed asks.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from ..models import Scope, now
from .cells import CellKind, CellState, EpistemicCell
from .map import KnowledgeMap

CANDIDATE = "candidate"
VERIFYING = "verifying"
LAW = "law"
FALSIFIED = "falsified"


def _law_id(namespace: str, r1: str, r2: str, r3: str) -> str:
    import hashlib
    return "law_" + hashlib.sha256(
        f"{namespace}\x1f{r1}\x1f{r2}\x1f{r3}".encode()).hexdigest()[:16]


class LawBook:
    """Persisted law lifecycle, stored inside the knowledge map's sqlite file."""

    def __init__(self, kmap: KnowledgeMap):
        self.kmap = kmap
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return self.kmap._conn()

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS laws (
                law_id     TEXT PRIMARY KEY,
                namespace  TEXT NOT NULL,
                r1         TEXT NOT NULL,
                r2         TEXT NOT NULL,
                r3         TEXT NOT NULL,
                confidence REAL NOT NULL,
                support    INTEGER NOT NULL,
                status     TEXT NOT NULL DEFAULT 'candidate',
                checks_passed INTEGER NOT NULL DEFAULT 0,
                checks_failed INTEGER NOT NULL DEFAULT 0,
                counterexample TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_laws_ns_status ON laws(namespace, status);
            """
        )
        c.commit()

    # ---- candidate intake (token-free) --------------------------------------
    def mine_candidates(self, store, scope: Scope, *, min_confidence: float = 0.6,
                        min_support: int = 2) -> dict:
        """Mine rules over the ACTIVE observed graph; persist new candidates."""
        from ..dreaming.rules import mine_rules
        active = store.active_edges_at(None, scope, include_inferred=False)
        triples = [(e.src, e.relation, e.dst) for e in active
                   if e.src and e.dst and e.relation]
        rules = mine_rules(triples, min_confidence=min_confidence,
                           min_support=min_support) if len(triples) >= 3 else []
        c = self._conn()
        added = 0
        for rule in rules:
            lid = _law_id(scope.namespace, rule.r1, rule.r2, rule.r3)
            row = c.execute("SELECT status FROM laws WHERE law_id=?", (lid,)).fetchone()
            if row is not None:
                if row["status"] in (CANDIDATE, VERIFYING):   # refresh mining stats
                    c.execute("UPDATE laws SET confidence=?, support=?, updated_at=?"
                              " WHERE law_id=?",
                              (float(rule.confidence), int(rule.support), now(), lid))
                continue
            c.execute("INSERT INTO laws(law_id, namespace, r1, r2, r3, confidence,"
                      " support, status, created_at, updated_at)"
                      " VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (lid, scope.namespace, rule.r1, rule.r2, rule.r3,
                       float(rule.confidence), int(rule.support), CANDIDATE,
                       now(), now()))
            added += 1
        c.commit()
        return {"mined": len(rules), "new_candidates": added, **self.counts(scope)}

    # ---- lifecycle -----------------------------------------------------------
    def laws(self, scope: Scope, status: Optional[str] = None) -> list[dict]:
        c = self._conn()
        if status:
            rows = c.execute("SELECT * FROM laws WHERE namespace=? AND status=?"
                             " ORDER BY confidence DESC", (scope.namespace, status)).fetchall()
        else:
            rows = c.execute("SELECT * FROM laws WHERE namespace=?"
                             " ORDER BY confidence DESC", (scope.namespace,)).fetchall()
        return [dict(r) for r in rows]

    def counts(self, scope: Scope) -> dict:
        c = self._conn()
        rows = c.execute("SELECT status, COUNT(*) n FROM laws WHERE namespace=?"
                         " GROUP BY status", (scope.namespace,)).fetchall()
        by = {r["status"]: int(r["n"]) for r in rows}
        return {"laws_candidate": by.get(CANDIDATE, 0), "laws_verifying": by.get(VERIFYING, 0),
                "laws_promoted": by.get(LAW, 0), "laws_falsified": by.get(FALSIFIED, 0)}

    def begin_verification(self, law_id: str) -> None:
        c = self._conn()
        c.execute("UPDATE laws SET status=?, updated_at=? WHERE law_id=? AND status=?",
                  (VERIFYING, now(), law_id, CANDIDATE))
        c.commit()

    def record_check(self, law_id: str, *, passed: bool, counterexample: str = "",
                     promote_min_checks: int = 3) -> str:
        """One probe outcome against a law's prediction. Falsification is immediate
        and permanent; promotion needs `promote_min_checks` clean passes."""
        c = self._conn()
        row = c.execute("SELECT * FROM laws WHERE law_id=?", (law_id,)).fetchone()
        if row is None:
            return "missing"
        if row["status"] in (LAW, FALSIFIED):
            return row["status"]
        if not passed:
            c.execute("UPDATE laws SET status=?, checks_failed=checks_failed+1,"
                      " counterexample=?, updated_at=? WHERE law_id=?",
                      (FALSIFIED, counterexample[:300], now(), law_id))
            c.commit()
            # A broken law is information: mint the CONTESTED cell.
            self.kmap.upsert_cell(EpistemicCell(
                namespace=row["namespace"], kind=CellKind.CONFLICT.value,
                subject=f"law {row['r1']} & {row['r2']} => {row['r3']}",
                relation="falsified_law", state=CellState.CONTESTED.value,
                reason=(f"law falsified by counterexample: {counterexample[:160]}"
                        if counterexample else "law falsified by probe"),
                origin="law", info_gain=1.1,
            ), cause="law falsified")
            return FALSIFIED
        passed_n = int(row["checks_passed"]) + 1
        status = LAW if passed_n >= promote_min_checks else VERIFYING
        c.execute("UPDATE laws SET status=?, checks_passed=?, updated_at=? WHERE law_id=?",
                  (status, passed_n, now(), law_id))
        c.commit()
        return status

    # ---- guarded application ---------------------------------------------------
    def apply_promoted(self, engine, scope: Scope, *, max_edges: int = 32) -> dict:
        """Write inferred edges for PROMOTED laws only, through the EXISTING guarded
        inferred-edge path (Edge.inferred=True, provenance=law text, confidence from
        the law). Never observed edges; excluded from observed reads by default."""
        from ..models import Edge
        applied = 0
        skipped = 0
        for law in self.laws(scope, LAW):
            adjacency: dict[str, list[tuple[str, str]]] = {}
            witnessed: set[tuple[str, str, str]] = set()
            active = engine.store.active_edges_at(None, scope, include_inferred=True)
            for e in active:
                adjacency.setdefault(e.src, []).append((e.relation, e.dst))
                witnessed.add((e.src.lower(), e.relation.lower(), e.dst.lower()))
            for x, edges_x in adjacency.items():
                for r1, y in edges_x:
                    if r1 != law["r1"]:
                        continue
                    for r2, z in adjacency.get(y, []):
                        if r2 != law["r2"] or x == z:
                            continue
                        if (x.lower(), law["r3"].lower(), z.lower()) in witnessed:
                            continue
                        if applied >= max_edges:
                            skipped += 1
                            continue
                        engine.store.add_edge(Edge(
                            src=x, dst=z, relation=law["r3"], scope=scope,
                            inferred=True, confidence=float(law["confidence"]),
                            provenance=f"law:{law['r1']}&{law['r2']}=>{law['r3']}",
                            fact=f"{x} {law['r3']} {z} (inferred by verified law)"))
                        applied += 1
        return {"applied_inferred_edges": applied, "skipped_over_cap": skipped}


def law_verification_wave(engine, scope: Scope, *, max_probes: int = 6,
                          promote_min_checks: int = 3) -> dict:
    """Curiosity for laws: probe untested predictions of candidate/verifying laws
    through the REAL prove path. PASSED -> record_check(passed); CONTRADICTED or a
    verified NEGATION -> falsify with the counterexample. Bounded model spend."""
    kmap = engine.knowledge_map_store
    book = LawBook(kmap)
    from ..dreaming.repair import Diagnosis
    from .curiosity import _diagnose_answer
    report = {"probed": 0, "passed": 0, "falsified": 0, "promoted": 0}
    for law in (book.laws(scope, CANDIDATE) + book.laws(scope, VERIFYING)):
        if report["probed"] >= max_probes:
            break
        book.begin_verification(law["law_id"])
        cells = [c for c in kmap.cells_in_state(scope, CellState.UNKNOWN.value, limit=200)
                 if c.kind == CellKind.LAW_PREDICTION.value
                 and law["r3"].lower() in c.relation.lower()]
        for cell in cells[:2]:                    # at most 2 predictions per law per wave
            if report["probed"] >= max_probes:
                break
            from .curiosity import probe_for_cell
            probe = probe_for_cell(cell)
            try:
                ans = engine.ask(probe, scope=scope, verify=True, use_cache=False)
            except Exception:
                continue
            report["probed"] += 1
            diag = _diagnose_answer(ans, engine.settings.abstention_threshold)
            kmap.on_probe_outcome(cell.cell_id, ans, diagnosis=diag.value)
            if diag == Diagnosis.PASSED:
                report["passed"] += 1
                status = book.record_check(law["law_id"], passed=True,
                                           promote_min_checks=promote_min_checks)
                if status == LAW:
                    report["promoted"] += 1
            elif diag == Diagnosis.CONTRADICTED:
                book.record_check(law["law_id"], passed=False,
                                  counterexample=f"probe contradicted: {probe[:120]}")
                report["falsified"] += 1
    return report
