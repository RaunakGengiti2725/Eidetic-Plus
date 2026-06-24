"""Layer 2c -- Maximal Marginal Relevance diversity post-pass.

MMR(d) = (1 - lambda) * rel(d) - lambda * max_{s in selected} sim(d, s).

Greedily pick the next item that is relevant yet least redundant with what is already
chosen. lambda in [0.3, 0.7]: higher = more diversity. This fixes the case where several
near-duplicate-but-not-identical memories crowd the top-k (exact-dup removal alone misses
those). Pure numpy; O(k * N) over the shortlist.

Relevance and similarity are put on a common [0,1] scale before mixing so lambda behaves
predictably regardless of the incoming relevance scale (RRF vs rerank vs cosine).
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _minmax01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


def mmr_order(relevances, vectors, lam: float = 0.5, k: int | None = None) -> list[int]:
    """Return indices in MMR-selected order. `relevances` are the first-stage scores;
    `vectors` are the candidate content embeddings (any scale -- normalized here). lam is
    the diversity weight; k caps the selection (default: all)."""
    rel = np.asarray(list(relevances), dtype=float)
    n = rel.size
    if n == 0:
        return []
    k = n if k is None else min(int(k), n)
    rel01 = _minmax01(rel)
    V = np.stack([_unit(v) for v in vectors]) if len(vectors) else np.zeros((n, 1), np.float32)
    selected: list[int] = []
    remaining = list(range(n))
    # max cosine of each candidate to the currently-selected set (starts at 0).
    max_sim = np.zeros(n, dtype=float)
    while remaining and len(selected) < k:
        best_j, best_score = remaining[0], -np.inf
        for j in remaining:
            score = (1.0 - lam) * rel01[j] - lam * max_sim[j]
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        remaining.remove(best_j)
        if remaining:
            sims_to_new = V[remaining] @ V[best_j]
            for idx, j in enumerate(remaining):
                if sims_to_new[idx] > max_sim[j]:
                    max_sim[j] = float(sims_to_new[idx])
    return selected
