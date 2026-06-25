"""Per-triple anomaly / confidence scoring over the OBSERVED knowledge graph (PDF 1d).

A numpy, model-free signal that flags low-confidence existing edges so the (LLM-gated) MemMA
repair sweep is AIMED at suspect triples instead of run blindly. Three deterministic signals,
blended into a per-edge confidence in [0,1]:

  * Local Outlier Factor (LOF) over the edge's endpoint-pair embedding -- a triple whose
    (src, dst) vectors sit where few other triples do is structurally anomalous (the
    TransT/CCA "noise vs semantically-similar-correct" intuition, numpy-only).
  * Endpoint coherence -- cosine(src_vec, dst_vec): an edge linking two unrelated entities is
    suspect.
  * TransE plausibility (reused from kg_embed) -- low exp(-||h+r-t||) for an observed edge that
    does not fit the learned structure.

All pure numpy / deterministic -> fully offline-unit-testable. Nothing here mutates the store;
it only produces a score the repair sweep prioritizes by.
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _lookup(entity_vectors: dict, name: str):
    """Case-insensitive entity-vector lookup: exact key first, then lowercased. Edge endpoints
    are stored in raw case while centroid maps (infer._entity_centroids) are lowercased, and the
    system treats entity identity as case-insensitive, so match both."""
    v = entity_vectors.get(name)
    if v is None:
        v = entity_vectors.get((name or "").lower())
    return v


def local_outlier_factor(X, k: int = 5) -> np.ndarray:
    """Breunig et al. LOF. Returns one score per row: ~1 is an inlier, >>1 is an outlier."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return np.ones(n)
    k = max(1, min(k, n - 1))
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, axis=1)[:, :k]                      # k nearest neighbours per point
    kdist = D[np.arange(n), knn[:, -1]]                     # distance to the k-th neighbour
    # local reachability density: 1 / mean reach-dist over the kNN
    lrd = np.zeros(n)
    for p in range(n):
        reach = np.maximum(kdist[knn[p]], D[p, knn[p]])
        lrd[p] = 1.0 / (reach.mean() + 1e-12)
    lof = np.array([(lrd[knn[p]] / (lrd[p] + 1e-12)).mean() for p in range(n)])
    return lof


def _lof01(X, k: int) -> np.ndarray:
    lof = local_outlier_factor(X, k)
    excess = np.clip(lof - 1.0, 0.0, None)                  # 0 for inliers
    m = excess.max()
    return excess / m if m > 0 else excess


def edge_coherence(edges, entity_vectors: dict) -> np.ndarray:
    """cosine(src_vec, dst_vec) mapped to [0,1] per edge; 0.5 (neutral) when a vector is absent."""
    out = []
    for e in edges:
        vs, vd = _lookup(entity_vectors, e.src), _lookup(entity_vectors, e.dst)
        if vs is None or vd is None:
            out.append(0.5)
        else:
            out.append(float((np.dot(_unit(vs), _unit(vd)) + 1.0) / 2.0))
    return np.asarray(out, dtype=np.float64)


def edge_anomaly_scores(edges, entity_vectors: dict, *, transe=None, k: int = 5,
                        w_lof: float = 0.5, w_coh: float = 0.3, w_transe: float = 0.2) -> np.ndarray:
    """Per-edge anomaly in [0,1] (higher = more anomalous / lower confidence). Blends the LOF of
    the endpoint-pair embedding, the endpoint incoherence, and (optionally) 1 - TransE plausibility."""
    if not edges:
        return np.zeros(0)
    dim = next((len(v) for v in entity_vectors.values()), 1)
    zero = np.zeros(dim)
    feats = []
    for e in edges:
        vs = _lookup(entity_vectors, e.src)
        vd = _lookup(entity_vectors, e.dst)
        feats.append(np.concatenate([_unit(vs if vs is not None else zero),
                                     _unit(vd if vd is not None else zero)]))
    lof = _lof01(np.asarray(feats), k)
    incoherence = 1.0 - edge_coherence(edges, entity_vectors)
    if transe is not None:
        transe_anom = np.array([1.0 - float(transe.score(e.src, e.relation, e.dst)) for e in edges])
        wt = w_transe
    else:
        transe_anom = np.zeros(len(edges))
        wt = 0.0
    total = w_lof + w_coh + wt
    blended = (w_lof * lof + w_coh * incoherence + wt * transe_anom) / (total if total else 1.0)
    return np.clip(blended, 0.0, 1.0)


def flag_anomalous(edges, anomaly_scores, threshold: float = 0.35) -> list:
    """Return the edges whose CONFIDENCE (1 - anomaly) is below `threshold` -- the repair-sweep
    targets, highest-anomaly first."""
    scored = sorted(zip(edges, np.asarray(anomaly_scores)), key=lambda x: -x[1])
    return [e for e, a in scored if (1.0 - a) < threshold]
