"""Deterministic UNKNOWN/CONTESTED enumerators (zero LLM, pure over the store).

Every rule here is recomputable by a judge from the witness-derived index layer, so
the knowledge map's counts cannot be inflated or gamed:

  g1 superseded_no_current   -- a single-valued (subject, relation) whose edges are
                                ALL invalidated: the system once knew a value, knows
                                it stopped holding, and does not know the current one.
  g2 law_predicted_unwitnessed -- an AnyBURL rule head (X, r3, Z) whose body holds in
                                the active graph but which no active edge witnesses:
                                the structure of what is known NAMES what is missing.
  g3 event_missing_date      -- an extracted event with no resolvable start date.
  g4 temporal_hole           -- a gap inside a fact chain: a value invalidated at T1
                                with the successor only valid from T2 >> T1.
  c1 multi_active_conflict   -- >1 distinct active object for a single-valued
                                (subject, relation): contradictory witnesses that
                                supersession has not resolved.

Query cells (g5) and NLI-conflict cells (c2) are minted INCREMENTALLY by the map's
on_answer/on_contradiction hooks -- they come from live outcomes, not store scans.
Falsified-law cells (c3) are minted by laws.py.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..graph import CO_ACTIVATED
from ..models import Scope, now
from ..store import RecordStore
from .cells import CellKind, CellState, EpistemicCell

# Relations that are structurally multi-valued: many simultaneously-true objects are
# expected, so multiple active edges are knowledge, not conflict, and an all-invalid
# history is churn, not a nameable gap.
MULTI_VALUED_RELATIONS = frozenset({
    CO_ACTIVATED, "mentions", "likes", "dislikes", "visited", "attended", "owns",
    "knows", "friend_of", "recommended", "read", "watched", "tried",
})

# A temporal hole must be wide enough to be a real era of not-knowing, not a
# same-conversation supersession artifact.
TEMPORAL_HOLE_MIN_SECONDS = 30 * 24 * 3600.0


def _single_valued(relation: str) -> bool:
    return (relation or "").lower() not in MULTI_VALUED_RELATIONS


def _fact_groups(store: RecordStore, scope: Scope):
    """All OBSERVED (never inferred) edges grouped by normalized (subject, relation)."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for e in store.all_edges(scope, include_inferred=False):
        if not _single_valued(e.relation):
            continue
        groups[((e.src or "").strip().lower(), (e.relation or "").strip().lower())].append(e)
    return groups


def superseded_no_current(store: RecordStore, scope: Scope,
                          at: Optional[float] = None) -> list[EpistemicCell]:
    """g1: every edge of a single-valued (s, r) is invalid at `at` -> UNKNOWN."""
    t = now() if at is None else at
    out: list[EpistemicCell] = []
    for (subject, relation), edges in _fact_groups(store, scope).items():
        if not subject or not relation:
            continue
        if any(e.is_active_at(t) for e in edges):
            continue
        # Only edges that EVER held (valid_at <= t) count as "once knew".
        past = [e for e in edges if e.valid_at <= t]
        if not past:
            continue
        latest = max(past, key=lambda e: float(e.invalid_at or e.valid_at))
        out.append(EpistemicCell(
            namespace=scope.namespace, agent_id=scope.agent_id, project_id=scope.project_id,
            kind=CellKind.FACT.value, subject=subject, relation=relation,
            state=CellState.UNKNOWN.value,
            reason=(f"all {len(past)} witnessed value(s) invalidated; last value "
                    f"'{latest.dst}' stopped holding and no current value is witnessed"),
            evidence_ids=[e.edge_id for e in past][:8],
            origin="enumerator", info_gain=0.5 + 0.1 * min(5, len(past)),
        ))
    return out


def multi_active_conflict(store: RecordStore, scope: Scope,
                          at: Optional[float] = None) -> list[EpistemicCell]:
    """c1: >1 distinct active object for a single-valued (s, r) -> CONTESTED."""
    t = now() if at is None else at
    out: list[EpistemicCell] = []
    for (subject, relation), edges in _fact_groups(store, scope).items():
        if not subject or not relation:
            continue
        active = [e for e in edges if e.is_active_at(t)]
        values = {(e.dst or "").strip().lower() for e in active}
        values.discard("")
        if len(values) <= 1:
            continue
        out.append(EpistemicCell(
            namespace=scope.namespace, agent_id=scope.agent_id, project_id=scope.project_id,
            kind=CellKind.CONFLICT.value, subject=subject, relation=relation,
            state=CellState.CONTESTED.value,
            reason=(f"{len(values)} simultaneously-active values for a single-valued "
                    f"relation: {sorted(values)[:4]}"),
            evidence_ids=[e.edge_id for e in active][:8],
            origin="enumerator", info_gain=1.0,
        ))
    return out


def event_missing_date(store: RecordStore, scope: Scope,
                       at: Optional[float] = None, *, limit: int = 200) -> list[EpistemicCell]:
    """g3: an extracted event with no resolvable start date -> UNKNOWN(event_date)."""
    out: list[EpistemicCell] = []
    for ev in store.events_in_scope(scope.namespace, scope=scope, at=at):
        if len(out) >= limit:
            break
        if getattr(ev, "start", None) is not None:
            continue
        fact = (getattr(ev, "fact", "") or
                f"{getattr(ev, 'subject', '')} {getattr(ev, 'verb', '')} "
                f"{getattr(ev, 'object', '')}").strip()
        if not fact:
            continue
        out.append(EpistemicCell(
            namespace=scope.namespace, agent_id=scope.agent_id, project_id=scope.project_id,
            kind=CellKind.EVENT_DATE.value, subject=fact[:200], relation="event_date",
            state=CellState.UNKNOWN.value,
            reason="event witnessed without a resolvable date",
            evidence_ids=[i for i in [getattr(ev, "source_memory_id", ""),
                                      getattr(ev, "event_id", "")] if i][:4],
            origin="enumerator", info_gain=0.4,
        ))
    return out


def temporal_hole(store: RecordStore, scope: Scope, at: Optional[float] = None,
                  *, min_gap_seconds: float = TEMPORAL_HOLE_MIN_SECONDS) -> list[EpistemicCell]:
    """g4: inside a single-valued fact chain, a period with no held value -> UNKNOWN."""
    t = now() if at is None else at
    out: list[EpistemicCell] = []
    for (subject, relation), edges in _fact_groups(store, scope).items():
        if not subject or not relation or len(edges) < 2:
            continue
        chain = sorted((e for e in edges if e.valid_at <= t), key=lambda e: e.valid_at)
        for prev, nxt in zip(chain, chain[1:]):
            end_prev = prev.invalid_at
            if end_prev is None or end_prev > t:
                continue
            gap = nxt.valid_at - end_prev
            if gap >= min_gap_seconds:
                out.append(EpistemicCell(
                    namespace=scope.namespace, agent_id=scope.agent_id,
                    project_id=scope.project_id,
                    kind=CellKind.TEMPORAL_HOLE.value, subject=subject,
                    relation=f"{relation}@{int(end_prev)}",
                    state=CellState.UNKNOWN.value,
                    reason=(f"no witnessed value for '{relation}' during a "
                            f"{gap / 86400.0:.0f}-day window between supersessions"),
                    evidence_ids=[prev.edge_id, nxt.edge_id],
                    origin="enumerator", info_gain=0.3,
                ))
    return out


def law_predicted_unwitnessed(store: RecordStore, scope: Scope,
                              at: Optional[float] = None, *,
                              min_confidence: float = 0.6, min_support: int = 2,
                              max_cells: int = 64) -> list[EpistemicCell]:
    """g2: mine Horn rules over the ACTIVE observed graph, then name every (X, r3, Z)
    the rules predict that no active edge witnesses. Rule confidence = info gain."""
    from ..dreaming.rules import mine_rules
    t = now() if at is None else at
    active = [e for e in store.active_edges_at(t, scope, include_inferred=False)
              if _single_valued(e.relation) or True]  # rules may use multi-valued bodies
    triples = [(e.src, e.relation, e.dst) for e in active if e.src and e.dst and e.relation]
    if len(triples) < 3:
        return []
    rules = mine_rules(triples, min_confidence=min_confidence, min_support=min_support)
    if not rules:
        return []
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    witnessed: set[tuple[str, str, str]] = set()
    for h, r, tl in triples:
        adjacency[h].append((r, tl))
        witnessed.add((h.lower(), r.lower(), tl.lower()))
    out: list[EpistemicCell] = []
    for rule in rules:
        for x, edges_x in adjacency.items():
            for r1, y in edges_x:
                if r1 != rule.r1:
                    continue
                for r2, z in adjacency.get(y, ()):  # bounded 2-hop, same as the miner
                    if r2 != rule.r2 or x == z:
                        continue
                    if (x.lower(), rule.r3.lower(), z.lower()) in witnessed:
                        continue
                    out.append(EpistemicCell(
                        namespace=scope.namespace, agent_id=scope.agent_id,
                        project_id=scope.project_id,
                        kind=CellKind.LAW_PREDICTION.value,
                        subject=x.strip().lower(),
                        relation=f"{rule.r3.strip().lower()}?{z.strip().lower()}",
                        state=CellState.UNKNOWN.value,
                        reason=f"predicted by law [{rule.text()}] but unwitnessed",
                        evidence_ids=[],
                        origin="enumerator", info_gain=float(rule.confidence),
                    ))
                    if len(out) >= max_cells:
                        return out
    return out


ENUMERATORS = (
    superseded_no_current,       # g1
    law_predicted_unwitnessed,   # g2
    event_missing_date,          # g3
    temporal_hole,               # g4
    multi_active_conflict,       # c1
)


def enumerate_cells(store: RecordStore, scope: Scope,
                    at: Optional[float] = None) -> list[EpistemicCell]:
    """Run every deterministic enumerator; dedupe by cell_id (first writer wins,
    CONTESTED outranks UNKNOWN when both name the same (s, r))."""
    t = now() if at is None else at
    by_id: dict[str, EpistemicCell] = {}
    for fn in ENUMERATORS:
        for cell in fn(store, scope, t):
            existing = by_id.get(cell.cell_id)
            if existing is None or (existing.state == CellState.UNKNOWN.value
                                    and cell.state == CellState.CONTESTED.value):
                by_id[cell.cell_id] = cell
    return list(by_id.values())
