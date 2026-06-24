"""Offline inference orchestrator (token-free): KG-embedding link prediction + Horn-rule
mining + Louvain schemas, every proposed edge/fact confidence-gated into the SEPARATE
inferred layer (Edge.inferred=True) or derived schemas. NEVER mutates observed memory.

Candidate generation is bounded to 2-hop paths (near-linear); schemas use Louvain
(near-linear). Embedding support for the gate is the cosine between entity content-centroids
(local math, no LLM). Optional real-NLI enrichment via settings.dream_use_llm_nli.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np

from ..config import Settings, get_settings
from ..models import DerivedRecord, Edge, Scope
from . import gate as gatemod
from .kg_embed import TransE
from .rules import apply_rules, mine_rules


def _entity_centroids(engine, scope: Scope) -> dict[str, np.ndarray]:
    """Mean content-embedding per entity (token-free; from the vector index)."""
    recs = [r for r in engine.store.all_records(scope) if r.entities]
    ids = [r.memory_id for r in recs]
    vecs = engine.index.get_vectors(ids)
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in recs:
        v = vecs.get(r.memory_id)
        if v is None:
            continue
        for e in r.entities:
            acc[e.lower()].append(v)
    return {e: np.mean(vs, axis=0) for e, vs in acc.items() if vs}


def _support(cents: dict[str, np.ndarray], a: str, b: str) -> float:
    va, vb = cents.get(a.lower()), cents.get(b.lower())
    if va is None or vb is None:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(max(0.0, va @ vb / (na * nb)))


def derive(engine, scope: Optional[Scope] = None, settings: Optional[Settings] = None) -> dict:
    scope = scope or Scope()
    s = settings or get_settings()
    observed = [e for e in engine.store.all_edges(scope) if e.is_active_at()]
    triples = [(e.src, e.relation, e.dst) for e in observed]
    if len(triples) < 3:
        return {"observed": len(triples), "proposed": 0, "admitted": 0, "schemas": 0,
                "nli_pass_rate": 1.0, "note": "too few observed facts"}

    cents = _entity_centroids(engine, scope)
    model = TransE(dim=s.dream_kg_dim, margin=s.dream_kg_margin).fit(triples, epochs=s.dream_kg_epochs)

    # Bounded 2-hop candidates (src..mid..dst not directly linked) -> best relation by TransE.
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    existing: set[tuple[str, str]] = set()
    for h, r, t in triples:
        out[h].append((r, t))
        existing.add((h, t))
    proposals: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for x, edges in out.items():
        for _, y in edges:
            for _, z in out.get(y, ()):
                if x == z or (x, z) in existing or (x, z) in seen_pairs:
                    continue
                seen_pairs.add((x, z))
                best_r, best_score = None, 0.0
                for r in model.relations:
                    sc = model.score(x, r, z)
                    if sc > best_score:
                        best_r, best_score = r, sc
                if best_r:
                    proposals.append({"src": x, "relation": best_r, "dst": z,
                                      "score": best_score, "provenance": "transe"})

    # Rule-mined facts (confidence from rule support).
    rules = mine_rules(triples, min_confidence=s.dream_rule_min_confidence)
    for f in apply_rules(triples, rules):
        proposals.append({"src": f["src"], "relation": f["relation"], "dst": f["dst"],
                          "score": f["confidence"], "provenance": f["provenance"]})

    # Gate every proposal (token-free confidence + embedding support); cap admitted.
    admitted, passed, total = [], 0, 0
    for p in sorted(proposals, key=lambda x: -x["score"]):
        if len(admitted) >= s.dream_infer_topk:
            break
        total += 1
        support = _support(cents, p["src"], p["dst"])
        llm = None
        if s.dream_use_llm_nli:
            prem = f"{p['src']} and {p['dst']} are related."
            llm = lambda: engine.client.nli(prem, f"{p['src']} {p['relation']} {p['dst']}")[0] == "entailment"
        res = gatemod.gate(p["score"], support, s.dream_infer_confidence, llm)
        if res.passed:
            passed += 1
            edge = Edge(src=p["src"], dst=p["dst"], relation=p["relation"],
                        fact=f"{p['src']} {p['relation']} {p['dst']}", scope=scope,
                        inferred=True, confidence=res.confidence, provenance=p["provenance"])
            engine.store.add_edge(edge)
            admitted.append(edge)

    schemas = _build_schemas(engine, scope, observed, cents)
    return {"observed": len(triples), "proposed": len(proposals), "gated": total,
            "admitted": len(admitted), "schemas": schemas,
            "nli_pass_rate": (passed / total) if total else 1.0}


def _build_schemas(engine, scope: Scope, observed: list[Edge], cents: dict[str, np.ndarray]) -> int:
    """Louvain communities over the observed graph -> schema centroids (DerivedRecords)."""
    g = nx.Graph()
    for e in observed:
        g.add_edge(e.src.lower(), e.dst.lower())
    if g.number_of_nodes() < 4:
        return 0
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(g, seed=0)
    except Exception:
        return 0
    n = 0
    for comm in comms:
        members = [m for m in comm]
        if len(members) < 3:
            continue
        vs = [cents[m] for m in members if m in cents]
        if not vs:
            continue
        centroid = np.mean(vs, axis=0)
        import hashlib
        cid = "sch_" + hashlib.sha1(("|".join([scope.namespace] + sorted(members))).encode()).hexdigest()[:16]
        engine.store.add_derived(DerivedRecord(
            cid=cid, kind="schema", namespace=scope.namespace, level=1,
            text=f"schema: {', '.join(members[:6])}", member_ids=members,
            vector=centroid.tolist(), provenance="louvain"))
        n += 1
    return n
