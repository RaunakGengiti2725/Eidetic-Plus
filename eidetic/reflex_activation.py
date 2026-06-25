"""Reflex Recall, Component 3: the activation burst.

`build_memory_packet` turns a query into a ranked MemoryPacket using only local graph math --
no embedding, no NLI, no answer generation. It is deliberately a free function with no model
client parameter, so "no network call" is structural.

Pipeline:
  1. parse the query (entities, time ranges, lexical terms),
  2. seed candidate ids from the derived inverted index (a non-scan lookup),
  3. load every seed back through the STORE (authoritative: re-applies scope + bi-temporal
     validity, so staleness can only lower coverage, never leak/expire-violate),
  4. score each survivor on decomposed axes (entity, lexical, query-time overlap, hot set),
  5. expand by co-activation from the strongest content seeds (multi-hop recall),
  6. attach live fact edges + supersession chains for the matched entities,
  7. rank by aggregate activation and project into a packet.

Age-independence: the temporal axis rewards overlap with the QUERY's time constraint only; with
no time constraint it is 0 for everyone. Recency of a memory is never a score term.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from .config import Settings
from .events import parse_query
from .models import MemoryRecord, RetrievalCandidate, Scope
from .reflex import MemoryPacket, PacketCandidate, ReflexScore
from .reflex_index import ReflexIndex, _norm_entity, tokenize


def _range_epochs(ranges: list[dict]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for r in ranges:
        try:
            out.append((datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S").timestamp(),
                        datetime.strptime(r["end"], "%Y-%m-%dT%H:%M:%S").timestamp()))
        except (ValueError, KeyError):
            continue
    return out


def _temporal_match(valid_at: float, invalid_at: Optional[float],
                    ranges: list[tuple[float, float]]) -> bool:
    """A memory matches a query time window if its event point / state start (`valid_at`) falls
    inside the window, or a CLOSED validity interval [valid_at, invalid_at] overlaps it. An
    open-ended fact is matched only by its start, so a 2019 fact does not light up on a 2023
    query merely because it is still true. Never compares against `now` -> age-independent."""
    for s, e in ranges:
        if s <= valid_at <= e:
            return True
        if invalid_at is not None and valid_at <= e and invalid_at >= s:
            return True
    return False


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def build_memory_packet(query: str, scope: Optional[Scope] = None, *, store, graph,
                        index: ReflexIndex, settings: Settings, as_of: Optional[float] = None,
                        hot_ids: Optional[set] = None, activation: Optional[dict] = None,
                        reference_time: Optional[float] = None) -> MemoryPacket:
    t0 = time.perf_counter()
    scope = scope or Scope()
    hot_ids = hot_ids or set()
    act = activation or {}
    lat: dict[str, float] = {}

    def mark(stage: str, since: float) -> float:
        now = time.perf_counter()
        lat[stage] = (now - since) * 1000.0
        return now

    parsed = parse_query(query, reference_time if reference_time is not None else as_of)
    q_entities = parsed.get("entities", [])
    q_entity_norms = [_norm_entity(e) for e in q_entities if e.strip()]
    q_terms = tokenize(query)
    q_term_set = set(q_terms)
    q_ranges = _range_epochs(parsed.get("ranges", []))
    t1 = mark("parse", t0)

    seed_ids = index.seeds(scope.namespace, entities=q_entities, terms=q_terms)
    # Field-seed (Track 9): union the top-activated ids into the seed set so a hot memory the query
    # never named can be recalled. The store-load below re-applies scope + bi-temporal validity, so
    # these are gated exactly like lexical seeds (cannot leak, cannot expire-violate).
    if act and settings.flow_field_seed:
        top_active = sorted(act.items(), key=lambda kv: -kv[1])[: max(0, settings.flow_seed_topk)]
        seed_ids = set(seed_ids) | {mid for mid, _ in top_active}
    if settings.reflex_max_seeds and len(seed_ids) > settings.reflex_max_seeds:
        # keep activated seeds preferentially when truncating (instinct should not be dropped first).
        ordered = sorted(seed_ids, key=lambda m: (-act.get(m, 0.0), m))
        seed_ids = set(ordered[:settings.reflex_max_seeds])
    t2 = mark("seed", t1)

    # Load every seed through the store; the store re-applies scope + bi-temporal validity.
    records: dict[str, MemoryRecord] = {}
    for mid in seed_ids:
        rec = store.get_record(mid)
        if rec is not None and rec.scope.visible_to(scope) and rec.is_active_at(as_of):
            records[mid] = rec
    t3 = mark("load", t2)

    scores: dict[str, ReflexScore] = {}
    paths: dict[str, list[str]] = {}
    entity_matches: dict[str, list[str]] = {}
    temporal_match_ids: list[str] = []
    for mid, rec in records.items():
        rec_terms = set(tokenize(rec.text))
        rec_entity_norms = {_norm_entity(e) for e in rec.entities} | rec_terms
        if q_entity_norms:
            matched = [e for e, n in zip(q_entities, q_entity_norms) if n in rec_entity_norms]
            entity = len(matched) / len(q_entity_norms)
            for e in matched:
                entity_matches.setdefault(e, []).append(mid)
        else:
            entity = 0.0
        lexical = (len(q_term_set & rec_terms) / len(q_term_set)) if q_term_set else 0.0
        temporal = 1.0 if (q_ranges and _temporal_match(rec.valid_at, rec.invalid_at, q_ranges)) else 0.0
        if temporal > 0.0:
            temporal_match_ids.append(mid)
        hot = 1.0 if mid in hot_ids else 0.0
        # `hot` stays BINARY -> it (and only it) feeds match_strength, so the coverage gate and the
        # flag-off path are byte-identical. Continuous field `activation` is a SEPARATE axis that
        # feeds the aggregate (ranking) only -- it never touches match_strength, so a field-seeded
        # content-less memory has coverage 0 and is still NLI-gated. Instinct surfaces, never fabricates.
        act_val = float(act.get(mid, 0.0))
        # Content coverage is the STRONGER of the two content axes (a perfect lexical-only match is
        # a confident hit, not a half one), with a small bonus when both fire, then small time/hot
        # bonuses. Bounded to [0,1]; it becomes the candidate's dense_score, and reflex_min_coverage
        # stays >= abstention_threshold so a hit can never spuriously abstain on coverage.
        content = max(lexical, entity) + 0.2 * min(lexical, entity)
        match_strength = _clamp01(content + 0.1 * temporal + 0.05 * hot)
        scores[mid] = ReflexScore(entity=entity, lexical=lexical, temporal=temporal,
                                  hotset=hot, activation=act_val, match_strength=match_strength)
        axes = [name for name, val in (("entity", entity), ("lexical", lexical),
                                       ("temporal", temporal), ("hotset", hot),
                                       ("activation", act_val)) if val > 0.0]
        paths[mid] = axes
    t4 = mark("score", t3)

    # Co-activation expansion from the strongest CONTENT seeds (multi-hop recall). A pulled-in
    # memory may have no lexical match; it still rides the graph in as a candidate.
    coactivation_paths: dict[str, list[str]] = {}
    coact_raw: dict[str, int] = {}
    strong = sorted((m for m in records if scores[m].match_strength > 0.0),
                    key=lambda m: (-scores[m].match_strength, m))[:max(1, settings.reflex_coact_seeds)]
    n_strong = max(1, len(strong))
    for seed in strong:
        linked = graph.linked_memories(seed, scope, as_of)
        kept: list[str] = []
        for lid in linked:
            if lid == seed:
                continue
            rec = records.get(lid)
            if rec is None:
                rec = store.get_record(lid)
                if rec is None or not rec.scope.visible_to(scope) or not rec.is_active_at(as_of):
                    continue
                records[lid] = rec
            kept.append(lid)
            coact_raw[lid] = coact_raw.get(lid, 0) + 1
        if kept:
            coactivation_paths[seed] = kept
    for mid in records:
        s = scores.get(mid)
        if s is None:
            s = ReflexScore(activation=float(act.get(mid, 0.0)))
            scores[mid] = s
            paths.setdefault(mid, [])
            if s.activation > 0.0:
                paths[mid].append("activation")
        if coact_raw.get(mid):
            s.coactivation = _clamp01(coact_raw[mid] / n_strong)
            if "coactivation" not in paths[mid]:
                paths[mid].append("coactivation")
    t5 = mark("coactivation", t4)

    # Final aggregate (now that co-activation is known) + ranking. The continuous activation axis
    # contributes to RANKING only (reflex_w_activation); match_strength/coverage stay content-only.
    for mid, s in scores.items():
        s.aggregate = (settings.reflex_w_entity * s.entity
                       + settings.reflex_w_lexical * s.lexical
                       + settings.reflex_w_temporal * s.temporal
                       + settings.reflex_w_coactivation * s.coactivation
                       + settings.reflex_w_hotset * s.hotset
                       + settings.reflex_w_activation * s.activation)
    ranked = sorted(records.keys(), key=lambda m: (-scores[m].aggregate, m))[:settings.reflex_topk]

    # Live fact edges + supersession chains for the matched entities (bounded, read on demand).
    active_fact_edges: list[dict] = []
    supersession_chains: dict[str, list[str]] = {}
    lookup_entities = set(q_entities[:8])
    if lookup_entities:
        for e in store.active_edges_touching_many({x for x in lookup_entities}, as_of, scope):
            active_fact_edges.append({"edge_id": e.edge_id, "src": e.src, "relation": e.relation,
                                      "dst": e.dst, "fact": e.fact, "valid_at": e.valid_at,
                                      "supersedes": e.supersedes})
        for ent in list(lookup_entities)[:8]:
            for e in store.edges_touching(ent, scope):
                if e.supersedes:
                    key = e.source_memory_id or e.edge_id
                    supersession_chains.setdefault(key, []).append(
                        e.fact or f"{e.src} {e.relation} {e.dst}")
    t6 = mark("edges", t5)

    coverage = max((scores[m].match_strength for m in ranked), default=0.0)
    items: list[PacketCandidate] = []
    candidates: list[RetrievalCandidate] = []
    for mid in ranked:
        rec = records[mid]
        s = scores[mid]
        snippet = (rec.summary or rec.text or "")[:240]
        items.append(PacketCandidate(memory_id=mid, content_hash=rec.content_hash,
                                     raw_uri=rec.raw_uri, snippet=snippet, valid_at=rec.valid_at,
                                     invalid_at=rec.invalid_at, score=s,
                                     retrieval_paths=paths.get(mid, [])))
        candidates.append(RetrievalCandidate(
            record=rec, dense_score=s.match_strength, rerank_score=s.match_strength,
            fused_score=s.aggregate, graph_score=s.coactivation, bm25_score=s.lexical))

    lat["total"] = (time.perf_counter() - t0) * 1000.0
    packet = MemoryPacket(
        query=query, scope=scope, as_of=as_of, coverage=coverage, items=items,
        scores={m: scores[m] for m in ranked},
        snippets={c.memory_id: c.snippet for c in items},
        content_hashes={c.memory_id: c.content_hash for c in items},
        entity_matches=entity_matches, temporal_match_ids=temporal_match_ids,
        coactivation_paths=coactivation_paths, supersession_chains=supersession_chains,
        active_fact_edges=active_fact_edges, hot_ids=sorted(hot_ids & set(records.keys())),
        latency_ms=lat,
    )
    packet._candidates = candidates
    return packet
