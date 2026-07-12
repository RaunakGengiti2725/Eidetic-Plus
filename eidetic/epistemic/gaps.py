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
  g6 end_without_newer_begin -- an END-verb fact (cancelled/ended/quit) with no newer
                                BEGIN-verb fact in the same subject+object family:
                                the current state is unwitnessed (day0 forensics --
                                live extraction speaks in begin/end verb pairs, not
                                same-relation supersessions).
  c2 cross_layer_conflict    -- the graph adjudicated a conflict but the claim layer
                                still actively carries the LOSING value: the proof
                                surfaces disagree (found live: a structured read
                                served the graph-closed phone number, verified).

Query cells (g5) and NLI-conflict cells are minted INCREMENTALLY by the map's
on_answer/on_contradiction hooks -- they come from live outcomes, not store scans.
Falsified-law cells are minted by laws.py.
"""
from __future__ import annotations

import re
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


# Live extraction speaks in BEGIN/END VERB PAIRS ('joined' / 'cancelled membership'),
# not tidy same-relation supersessions -- measured on the day0 demo corpus, where g1
# missed a cancelled-gym gap because the two edges carry different relations. The
# family table is deterministic and small on purpose: precision over recall (a wrong
# gap wastes one probe; a fabricated pairing would poison the map).
_BEGIN_VERBS = ("join", "sign", "start", "begin", "enroll", "subscribe", "adopt",
                "move in", "moved in", "lease")
_END_VERBS = ("cancel", "end", "quit", "leav", "stop", "terminat", "expir",
              "move out", "moved out", "resign", "unsubscrib")


def _verb_phase(relation: str) -> str:
    low = (relation or "").lower()
    if any(v in low for v in _END_VERBS):
        return "end"
    if any(v in low for v in _BEGIN_VERBS):
        return "begin"
    return ""


def end_without_newer_begin(store: RecordStore, scope: Scope,
                            at: Optional[float] = None) -> list[EpistemicCell]:
    """g6: an END-verb edge (s, cancelled/ended/quit, o) with no BEGIN-verb edge for
    the same subject+object family AFTER it -> the current state is UNKNOWN."""
    t = now() if at is None else at
    active = store.active_edges_at(t, scope, include_inferred=False)
    all_edges = [e for e in store.all_edges(scope) if e.valid_at <= t]
    out: list[EpistemicCell] = []
    for e in active:
        if _verb_phase(e.relation) != "end" or not e.src or not e.dst:
            continue
        obj_terms = {w for w in re.findall(r"[a-z0-9]+", e.dst.lower()) if len(w) > 3}
        newer_begin = any(
            b.src == e.src and _verb_phase(b.relation) == "begin"
            and b.valid_at > e.valid_at
            and (obj_terms & {w for w in re.findall(r"[a-z0-9]+", b.dst.lower())
                              if len(w) > 3})
            for b in all_edges)
        if newer_begin:
            continue
        out.append(EpistemicCell(
            namespace=scope.namespace, agent_id=scope.agent_id,
            project_id=scope.project_id,
            kind=CellKind.FACT.value, subject=e.src.strip().lower(),
            relation=f"current_state_of {e.dst.strip().lower()}",
            state=CellState.UNKNOWN.value,
            reason=(f"'{e.relation}' witnessed with no newer begin-verb fact for "
                    f"'{e.dst}' -- the current state is unwitnessed"),
            evidence_ids=[e.edge_id],
            origin="enumerator", info_gain=0.7,
        ))
    return out


def cross_layer_conflict(store: RecordStore, scope: Scope,
                         at: Optional[float] = None) -> list[EpistemicCell]:
    """c2 (found live, day0 demo): the GRAPH adjudicated a conflict (closed one edge,
    kept a sibling) but the CLAIM layer still actively carries the LOSING value --
    the two proof surfaces disagree, and a structured read can serve the loser with
    a citation. That is CONTESTED, not settled."""
    t = now() if at is None else at
    active_claims = store.active_claims_at(t, scope)
    if not active_claims:
        return []
    groups = _fact_groups(store, scope)
    out: list[EpistemicCell] = []
    for (subject, relation), edges in groups.items():
        if not subject or not relation:
            continue
        closed = [e for e in edges if e.invalid_at is not None and e.invalid_at <= t]
        alive = [e for e in edges if e.is_active_at(t)]
        if not closed or not alive:
            continue
        subject_low = subject.lower()
        for loser in closed:
            loser_val = (loser.dst or "").strip()
            if len(loser_val) < 4:
                continue
            carriers = [c for c in active_claims
                        if subject_low in (c.subject or "").lower()
                        and loser_val.lower() in str(c.value or c.object or "").lower()]
            if not carriers:
                continue
            out.append(EpistemicCell(
                namespace=scope.namespace, agent_id=scope.agent_id,
                project_id=scope.project_id,
                kind=CellKind.CONFLICT.value, subject=subject,
                relation=f"{relation} [cross-layer]",
                state=CellState.CONTESTED.value,
                reason=(f"graph closed '{loser_val}' (kept '{alive[0].dst}') but "
                        f"{len(carriers)} active claim(s) still carry the losing "
                        "value -- proof surfaces disagree"),
                evidence_ids=([loser.edge_id, alive[0].edge_id]
                              + [c.claim_id for c in carriers])[:8],
                origin="enumerator", info_gain=1.3,
            ))
    return out


ENUMERATORS = (
    superseded_no_current,       # g1
    law_predicted_unwitnessed,   # g2
    event_missing_date,          # g3
    temporal_hole,               # g4
    multi_active_conflict,       # c1
    end_without_newer_begin,     # g6 (day0 forensics)
    cross_layer_conflict,        # c2 (day0 forensics)
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
