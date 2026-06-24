"""Components 2 (graph half) & 7: the bi-temporal knowledge graph.

Entities and relations carry valid_at/invalid_at (world truth) and created_at/
expired_at (system knowledge). A new fact that contradicts an existing one CLOSES
(invalidates) the old edge -- it is never deleted, so the full history is queryable.

Personalized PageRank runs in-app (networkx), exactly as HippoRAG does, to give
associative multi-hop retrieval. Graph node features (PPR, degree) also feed the
Component 3 structure code.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

from .models import Edge, MemoryRecord, Scope, now
from .store import RecordStore

# Relation used for memory-linking-by-co-activation (Section 7.3). These edges connect
# memory_ids (not entities) and are excluded from entity PPR; they expand the candidate
# set at retrieval time but never enter the ranking score (recall stays age-independent).
CO_ACTIVATED = "co_activated"


def _norm(name: str) -> str:
    return name.strip().lower()


class KnowledgeGraph:
    def __init__(self, store: RecordStore, deterministic_conflicts: bool = False):
        self.store = store
        self.deterministic_conflicts = deterministic_conflicts

    # ---- write path -------------------------------------------------------
    def add_fact(
        self,
        src: str,
        relation: str,
        dst: str,
        *,
        fact: str = "",
        source_memory_id: str = "",
        valid_at: Optional[float] = None,
        scope: Optional[Scope] = None,
    ) -> tuple[Edge, list[Edge]]:
        """Add a fact edge. Returns (new_edge, invalidated_edges).

        Contradiction rule (bi-temporal): an active edge WITHIN THE SAME SCOPE with the
        same (src, relation) but a different dst is closed as of `valid_at`. History is
        retained. Contradictions never cross namespaces."""
        valid_at = now() if valid_at is None else valid_at
        scope = scope or Scope()
        invalidated: list[Edge] = []
        # co_activated is a MULTI-valued association (a memory links to many others), so
        # it is exempt from the single-valued (src, relation) contradiction rule.
        single_valued = relation != CO_ACTIVATED
        for e in self.store.edges_touching(src, scope) if single_valued else []:
            if (
                _norm(e.src) == _norm(src)
                and _norm(e.relation) == _norm(relation)
                and _norm(e.dst) != _norm(dst)
                and e.is_active_at(valid_at)
            ):
                e.invalid_at = valid_at
                e.expired_at = valid_at
                self.store.add_edge(e)  # close, never delete
                invalidated.append(e)

        edge = Edge(
            src=src, dst=dst, relation=relation,
            fact=fact or f"{src} {relation} {dst}",
            source_memory_id=source_memory_id, valid_at=valid_at, scope=scope,
            supersedes=invalidated[0].edge_id if invalidated else None,
        )
        self.store.add_edge(edge)
        if self.deterministic_conflicts and single_valued:
            self._intervalize_single_value(src, relation, scope)
        return edge, invalidated

    def _intervalize_single_value(self, src: str, relation: str, scope: Scope) -> None:
        """Recompute validity intervals for one single-valued key by valid_at order."""
        edges = [
            e for e in self.store.edges_touching(src, scope)
            if (
                not e.inferred
                and _norm(e.src) == _norm(src)
                and _norm(e.relation) == _norm(relation)
                and e.relation != CO_ACTIVATED
            )
        ]
        edges.sort(key=lambda e: (e.valid_at, e.created_at, e.edge_id))
        for i, edge in enumerate(edges):
            close_at = None
            for later in edges[i + 1:]:
                if _norm(later.dst) != _norm(edge.dst):
                    close_at = later.valid_at
                    break
            changed = False
            if close_at is None:
                changed = edge.invalid_at is not None or edge.expired_at is not None
                edge.invalid_at = None
                edge.expired_at = None
            elif edge.invalid_at != close_at or edge.expired_at != close_at:
                edge.invalid_at = close_at
                edge.expired_at = close_at
                changed = True
            if changed:
                self.store.add_edge(edge)

    # ---- memory linking by co-activation (Section 7.3) -------------------
    def link_memories(self, memory_ids: list[str], scope: Optional[Scope] = None,
                      valid_at: Optional[float] = None) -> int:
        """Strengthen a co_activated edge between every pair of co-confirmed memories.
        Self-organizes the graph around what is actually used together."""
        scope = scope or Scope()
        valid_at = now() if valid_at is None else valid_at
        ids = list(dict.fromkeys(memory_ids))
        added = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.add_fact(ids[i], CO_ACTIVATED, ids[j], fact="co-activated recall",
                              valid_at=valid_at, scope=scope)
                added += 1
        return added

    def linked_memories(self, memory_id: str, scope: Optional[Scope] = None,
                        at: Optional[float] = None) -> list[str]:
        at = now() if at is None else at
        out: list[str] = []
        for e in self.store.edges_touching(memory_id, scope):
            if e.relation == CO_ACTIVATED and e.is_active_at(at):
                out.append(e.dst if _norm(e.src) == _norm(memory_id) else e.src)
        return out

    # ---- graph construction (entity facts only; co_activated excluded) ---
    def build_nx(self, at: Optional[float] = None, scope: Optional[Scope] = None,
                 include_inferred: bool = False) -> nx.DiGraph:
        at = now() if at is None else at
        g = nx.DiGraph()
        for e in self.store.active_edges_at(at, scope, include_inferred=include_inferred):
            # Skip co-activation links and SHY-pruned edges (pruned from the INDEX, not store).
            if e.relation == CO_ACTIVATED or getattr(e, "pruned", False):
                continue
            self._add_edge_to_nx(g, e)
        return g

    @staticmethod
    def _add_edge_to_nx(g: nx.DiGraph, e: Edge) -> None:
        s, d = _norm(e.src), _norm(e.dst)
        w = getattr(e, "weight", 1.0) or 1.0
        g.add_node(s)
        g.add_node(d)
        if g.has_edge(s, d):
            g[s][d]["weight"] += w
        else:
            g.add_edge(s, d, weight=w, relation=e.relation)

    def build_seed_neighborhood_nx(
        self,
        seed_entities: list[str],
        at: Optional[float] = None,
        scope: Optional[Scope] = None,
        *,
        hops: int = 2,
        include_inferred: bool = False,
    ) -> nx.DiGraph:
        """Build a bounded graph around query seeds instead of the full scoped graph."""
        at = now() if at is None else at
        frontier = {_norm(s) for s in seed_entities if _norm(s)}
        seen = set(frontier)
        g = nx.DiGraph()
        for _ in range(max(1, hops)):
            edges = self.store.active_edges_touching_many(
                frontier, at, scope, include_inferred=include_inferred
            )
            next_frontier: set[str] = set()
            for e in edges:
                if e.relation == CO_ACTIVATED or getattr(e, "pruned", False):
                    continue
                self._add_edge_to_nx(g, e)
                for n in (_norm(e.src), _norm(e.dst)):
                    if n not in seen:
                        next_frontier.add(n)
            if not next_frontier:
                break
            seen.update(next_frontier)
            frontier = next_frontier
        return g

    # ---- Personalized PageRank -------------------------------------------
    def ppr_entities(
        self, seed_entities: list[str], at: Optional[float] = None,
        alpha: float = 0.85, scope: Optional[Scope] = None,
    ) -> dict[str, float]:
        g = self.build_seed_neighborhood_nx(seed_entities, at, scope)
        if g.number_of_nodes() == 0:
            return {}
        seeds = {_norm(s) for s in seed_entities if _norm(s) in g}
        if not seeds:
            return {}
        personalization = {n: (1.0 if n in seeds else 0.0) for n in g.nodes()}
        ug = g.to_undirected()  # associative spreading is bidirectional
        try:
            scores = nx.pagerank(ug, alpha=alpha, personalization=personalization, weight="weight")
        except nx.PowerIterationFailedConvergence:  # pragma: no cover - rare
            scores = {n: personalization[n] for n in g.nodes()}
        return scores

    def score_memories(
        self,
        seed_entities: list[str],
        records: list[MemoryRecord],
        at: Optional[float] = None,
        scope: Optional[Scope] = None,
    ) -> dict[str, float]:
        """Map entity PPR scores onto candidate memories (sum over their entities)."""
        ent_scores = self.ppr_entities(seed_entities, at, scope=scope)
        if not ent_scores:
            return {}
        out: dict[str, float] = {}
        for rec in records:
            s = sum(ent_scores.get(_norm(e), 0.0) for e in rec.entities)
            if s > 0:
                out[rec.memory_id] = s
        # normalize to [0,1]
        if out:
            mx = max(out.values())
            if mx > 0:
                out = {k: v / mx for k, v in out.items()}
        return out

    def node_features(self, at: Optional[float] = None,
                      scope: Optional[Scope] = None) -> dict[str, dict[str, float]]:
        """Per-entity {ppr, degree} for structure codes (global PPR = uniform seeds)."""
        g = self.build_nx(at, scope)
        if g.number_of_nodes() == 0:
            return {}
        ug = g.to_undirected()
        try:
            ppr = nx.pagerank(ug, alpha=0.85, weight="weight")
        except nx.PowerIterationFailedConvergence:  # pragma: no cover
            ppr = {n: 1.0 / g.number_of_nodes() for n in g.nodes()}
        deg = dict(ug.degree())
        return {n: {"ppr": float(ppr.get(n, 0.0)), "degree": float(deg.get(n, 0))} for n in g.nodes()}
