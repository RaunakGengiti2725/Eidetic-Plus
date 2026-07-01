"""The continuous replay scheduler (the substrate of the dreaming engine), token-free.

Priority = surprise^w_s * need^w_n * (1 - retrievability)^w_r
  surprise        = 1 - cosine to the nearest other in-scope memory (novelty; via the ANN
                    index -> near-linear, NOT O(N^2))
  need            = mean entity PPR (graph centrality) + recency
  retrievability  = FSRS R(t, s)
Each cycle pops the top-k, reinforces their FSRS stability (capped, to stop rich-get-richer),
bumps the weight of edges in their neighborhood, then a global SHY-style pass renormalizes
edge weights and PRUNES the weakest edges from the INDEX **by weight only** (never by
retrievability -- pruning by age would slope the flat recall-vs-age curve) -- reversible
(`pruned` flag), never deleting from the lossless store.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import fsrs
from ..config import Settings, get_settings
from ..models import Scope, now


def cycle(engine, scope: Optional[Scope] = None, settings: Optional[Settings] = None) -> dict:
    scope = scope or Scope()
    s = settings or get_settings()
    recs = engine.store.all_records(scope)
    if not recs:
        return {"replayed": 0, "edges_pruned": 0, "edges_total": 0, "memories": 0}

    ids = [r.memory_id for r in recs]
    vecs = engine.index.get_vectors(ids)
    feats = engine.graph.node_features(scope=scope)   # entity PPR/degree, computed ONCE
    t = now()

    def surprise(r) -> float:
        v = vecs.get(r.memory_id)
        if v is None:
            return 1.0
        nn = engine.index.search(v, 2)               # nearest other (ANN, sub-linear)
        sims = [sim for mid, sim in nn if mid != r.memory_id]
        return float(max(0.0, 1.0 - (max(sims) if sims else 0.0)))

    def need(r) -> float:
        vals = [feats[e.lower()]["ppr"] for e in r.entities if e.lower() in feats] if (
            feats and r.entities) else []
        ppr = float(np.mean(vals)) if vals else 0.0
        recency = 1.0 / (1.0 + r.age_days(t))
        return float(0.5 * min(1.0, ppr * 20.0) + 0.5 * recency)

    scored = []
    for r in recs:
        sup = surprise(r) ** s.dream_w_surprise
        nd = max(1e-6, need(r)) ** s.dream_w_need
        rt = max(1e-6, 1.0 - fsrs.current_retrievability(r.fsrs, t)) ** s.dream_w_retr
        scored.append((sup * nd * rt, r))
    scored.sort(key=lambda x: -x[0])

    # Replay the top-k: reinforce FSRS (capped), bump neighborhood edge weights.
    replayed = 0
    for _, r in scored[: s.dream_replay_topk]:
        fsrs.reinforce(r.fsrs, importance=max(0.3, r.importance))
        r.fsrs.stability = min(r.fsrs.stability, s.dream_salience_cap)   # cap rich-get-richer
        engine.store.upsert_record(r)
        for ent in r.entities[:8]:
            for e in engine.store.edges_touching(ent, scope, include_inferred=True):
                e.weight = min(s.dream_salience_cap, e.weight * 1.05)
                engine.store.add_edge(e)
        replayed += 1

    # SHY-style global pass: renormalize edge weights, prune the weakest by WEIGHT (reversible).
    edges = engine.store.all_edges(scope, include_inferred=True)
    pruned = 0
    if edges:
        weights = np.array([e.weight for e in edges], dtype=np.float64)
        mean_w = float(weights.mean()) or 1.0
        thresh = float(np.percentile(weights, s.dream_prune_percentile)) if s.dream_prune_percentile > 0 else -1.0
        for e, w in zip(edges, weights):
            new_w = w / mean_w                       # downscale toward baseline (preserve order)
            should_prune = s.dream_prune_percentile > 0 and w <= thresh
            if e.weight != new_w or e.pruned != should_prune:
                e.weight = new_w
                e.pruned = should_prune
                engine.store.add_edge(e)
            if should_prune:
                pruned += 1

    return {"replayed": replayed, "edges_pruned": pruned, "edges_total": len(edges),
            "memories": len(recs)}
