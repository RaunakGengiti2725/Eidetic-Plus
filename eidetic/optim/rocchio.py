"""Layer 3b -- Rocchio pseudo-relevance feedback (query expansion).

q_new = alpha*q + beta*(1/|R|)*sum_{d in R} d  -  gamma*(1/|N|)*sum_{d in N} d

Push the query embedding toward the centroid of the top assumed-relevant memories. Defaults
alpha=1, beta=0.6, gamma=0 (positive-only is the safest: negative feedback risks topic
drift). The result is L2-normalized so it stays a valid query direction.

CAVEAT (enforced by the caller via should_expand): PRF causes topic drift when the initial
retrieval is bad. So expansion is confidence-gated -- only applied when the first-pass
evidence is strong enough that its centroid is trustworthy.
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rocchio_expand(qvec, relevant_vecs, nonrelevant_vecs=None,
                   alpha: float = 1.0, beta: float = 0.6, gamma: float = 0.0) -> np.ndarray:
    """Return the expanded, L2-normalized query vector. With no relevant vectors it returns
    the (normalized) original query unchanged."""
    q = np.asarray(qvec, dtype=np.float32)
    new = alpha * q
    if relevant_vecs is not None and len(relevant_vecs) > 0:
        new = new + beta * np.mean(np.asarray(relevant_vecs, dtype=np.float32), axis=0)
    if gamma > 0 and nonrelevant_vecs is not None and len(nonrelevant_vecs) > 0:
        new = new - gamma * np.mean(np.asarray(nonrelevant_vecs, dtype=np.float32), axis=0)
    return _unit(new)


def should_expand(confidence: float, threshold: float) -> bool:
    """Gate PRF on first-pass confidence so a bad initial retrieval cannot drift the query.
    threshold <= 0 disables the gate's lower bound (always expand); >1 disables expansion."""
    return float(confidence) >= float(threshold)
