"""ContestedResolutionProgram: a contradiction is a RESEARCH PROGRAM, not a delete.

For one CONTESTED cell the program runs a bounded, deterministic-first pipeline:

  1. gather   -- pull the conflicting witnesses (evidence ids on the cell; for
                 (s, r) conflict cells, the live conflicting edges + their sources)
  2. adjudicate (token-free first) -- bi-temporal order: if the conflict is really a
                 supersession the graph missed (distinct valid_at, single-valued
                 relation), propose closing the older edge THROUGH THE EXISTING
                 supersession path (invalidate_edge = bi-temporal close, never delete)
  3. probe    -- otherwise ask the REAL prove path for the current value; a VERIFIED
                 answer whose citation postdates every witness on one side resolves
                 the cell (resolution recorded with proof; losing edges closed
                 bi-temporally)
  4. hold     -- no verified resolution -> the cell STAYS CONTESTED with the program
                 transcript appended; abstention on the contested question remains
                 the correct public behavior

Resolutions never touch raw bytes: the losers remain queryable history via as-of
reads; only their validity windows close. Every step lands in the cell's reason +
the map's transition ledger.
"""
from __future__ import annotations

from typing import Optional

from ..models import AnswerStatus, Scope, now
from .cells import CellState, EpistemicCell
from .map import KnowledgeMap


def _conflict_edges(engine, cell: EpistemicCell, scope: Scope):
    """The live conflicting edges for a (subject, relation) conflict cell."""
    if cell.relation in ("nli_conflict", "falsified_law"):
        return []
    active = engine.store.active_edges_at(None, scope, include_inferred=False)
    subject = cell.subject.strip().lower()
    relation = cell.relation.strip().lower()
    return [e for e in active
            if (e.src or "").strip().lower() == subject
            and (e.relation or "").strip().lower() == relation]


def resolve_contested_cell(engine, cell_id: str, *, scope: Optional[Scope] = None,
                           allow_probe: bool = True) -> dict:
    """Run the program on one cell. Returns a transcript dict; updates the map."""
    kmap: KnowledgeMap = engine.knowledge_map_store
    cell = kmap.get_cell(cell_id)
    if cell is None:
        return {"cell_id": cell_id, "outcome": "missing"}
    if cell.state != CellState.CONTESTED.value:
        return {"cell_id": cell_id, "outcome": "not_contested", "state": cell.state}
    scope = scope or Scope(namespace=cell.namespace, agent_id=cell.agent_id,
                           project_id=cell.project_id)
    transcript: dict = {"cell_id": cell_id, "subject": cell.subject,
                        "relation": cell.relation, "steps": []}

    # -- step 0: cross-layer repair (token-free, found live on day0) -------------
    # The graph already adjudicated; the claim layer still serves the loser. The
    # resolution is not a probe -- it is making the surfaces agree: bi-temporally
    # close the loser-carrying claims at the graph's own adjudication time.
    if cell.relation.endswith("[cross-layer]"):
        base_relation = cell.relation.replace("[cross-layer]", "").strip()
        all_edges = engine.store.all_edges(scope)
        siblings = [e for e in all_edges
                    if (e.src or "").strip().lower() == cell.subject
                    and (e.relation or "").strip().lower() == base_relation]
        closed = [e for e in siblings if e.invalid_at is not None]
        alive = [e for e in siblings if e.is_active_at(now())]
        closed_claims = []
        if closed and alive:
            for c in engine.store.active_claims_at(None, scope):
                text = str(c.value or c.object or "").lower()
                if cell.subject not in (c.subject or "").lower():
                    continue
                for loser in closed:
                    lv = (loser.dst or "").strip().lower()
                    if len(lv) >= 4 and lv in text:
                        engine.store.invalidate_claim(
                            c.claim_id, at=loser.invalid_at)
                        closed_claims.append({"claim_id": c.claim_id,
                                              "carried": lv})
                        break
        transcript["steps"].append({"step": "cross_layer_repair",
                                    "claims_closed": closed_claims,
                                    "kept": [e.dst for e in alive]})
        if closed_claims:
            kmap.close_cell(cell_id, cause=(
                f"cross-layer repair: {len(closed_claims)} claim(s) carrying the "
                f"graph-closed value bi-temporally closed; surfaces agree on "
                f"'{alive[0].dst if alive else '?'}'"))
            transcript["outcome"] = "resolved_cross_layer_repair"
            return transcript

    # -- step 1+2: token-free bi-temporal adjudication ---------------------------
    edges = _conflict_edges(engine, cell, scope)
    if len(edges) >= 2:
        by_time = sorted(edges, key=lambda e: e.valid_at)
        newest = by_time[-1]
        older = by_time[:-1]
        distinct_times = len({round(e.valid_at, 3) for e in edges}) > 1
        if distinct_times:
            for e in older:
                engine.store.invalidate_edge(e.edge_id, at=newest.valid_at)
            transcript["steps"].append({
                "step": "bitemporal_supersession",
                "kept": {"dst": newest.dst, "valid_at": newest.valid_at},
                "closed": [{"dst": e.dst, "valid_at": e.valid_at} for e in older],
            })
            kmap.close_cell(cell_id, cause=(
                f"resolved by bi-temporal supersession: '{newest.dst}' is newest "
                f"(losers closed, never deleted; as-of reads keep the history)"))
            transcript["outcome"] = "resolved_supersession"
            return transcript
        transcript["steps"].append({"step": "bitemporal_supersession",
                                    "note": "identical valid_at on both sides; "
                                            "time alone cannot adjudicate"})

    # -- step 3: probe through the real prove path --------------------------------
    if allow_probe:
        from .curiosity import probe_for_cell
        probe = probe_for_cell(cell)
        try:
            ans = engine.ask(probe, scope=scope, verify=True, use_cache=False)
        except Exception as e:
            ans = None
            transcript["steps"].append({"step": "probe",
                                        "error": f"{type(e).__name__}: {str(e)[:120]}"})
        if ans is not None:
            transcript["steps"].append({"step": "probe",
                                        "status": ans.status.value,
                                        "citations": len(ans.citations)})
            if ans.status == AnswerStatus.VERIFIED and ans.citations:
                cited = {c.memory_id for c in ans.citations}
                losers = [e for e in edges if e.source_memory_id
                          and e.source_memory_id not in cited]
                winners = [e for e in edges if e.source_memory_id in cited]
                if edges and winners:
                    newest_win = max(w.valid_at for w in winners)
                    for e in losers:
                        engine.store.invalidate_edge(e.edge_id, at=max(
                            newest_win, e.valid_at))
                    transcript["steps"].append({
                        "step": "verified_resolution",
                        "closed": [{"dst": e.dst} for e in losers]})
                kmap.mark_known(cell_id, ans, cause="contested resolved by verified probe")
                transcript["outcome"] = "resolved_verified"
                return transcript

    # -- step 4: hold ---------------------------------------------------------------
    held_reason = (f"{cell.reason} | resolution program held at {int(now())}: "
                   "no verified adjudication; abstention remains correct")[:240]
    c = kmap._conn()
    c.execute("UPDATE cells SET reason=?, last_updated=? WHERE cell_id=?",
              (held_reason, now(), cell_id))
    c.commit()
    transcript["outcome"] = "held_contested"
    return transcript


def contested_wave(engine, scope: Scope, *, max_programs: int = 4,
                   allow_probe: bool = True) -> dict:
    """Run the program over the highest-gain contested cells. Bounded; each probe is
    one governed ask."""
    kmap: KnowledgeMap = engine.knowledge_map_store
    cells = kmap.cells_in_state(scope, CellState.CONTESTED.value, limit=max_programs)
    outcomes = []
    for cell in cells:
        outcomes.append(resolve_contested_cell(engine, cell.cell_id, scope=scope,
                                               allow_probe=allow_probe))
    from collections import Counter
    return {"programs": len(outcomes),
            "outcomes": dict(Counter(o.get("outcome", "?") for o in outcomes)),
            "transcripts": outcomes}
