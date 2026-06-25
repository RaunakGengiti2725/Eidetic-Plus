"""RAPTOR-style multi-resolution gist tree, token-free.

Recursively clusters memory embeddings (bounded-k numpy k-means -> near-linear, k capped by a
constant per level, NEVER O(N^2)) and stores a CENTROID per cluster as a derived gist node at
each level. Retrieval can hit any level (raw memory or any gist). The centroid IS the gist; an
LLM-written summary sentence is optional enrichment, not required for retrieval to work. Gist
nodes are additive DerivedRecords -- the lossless store is never touched.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..models import DerivedRecord


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _kmeans(X: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    # k-means++ seeding
    centers = [int(rng.integers(n))]
    d2 = np.full(n, np.inf)
    for _ in range(1, k):
        diff = X - X[centers[-1]]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
        s = float(d2.sum())
        if s <= 1e-12:
            # No remaining distinct directions: every point coincides with an existing center
            # (duplicate / float32-identical embeddings). Stop seeding; k collapses to the number
            # of real centers. Otherwise probs would be all-zero and rng.choice raises
            # "probabilities do not sum to 1".
            break
        centers.append(int(rng.choice(n, p=d2 / s)))
    k = len(centers)                                # effective cluster count actually seeded
    C = X[centers].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        sims = X @ C.T                              # cosine (X, C normalized) -> assign
        labels = sims.argmax(axis=1)
        newC = np.zeros_like(C)
        for c in range(k):
            m = labels == c
            if m.any():
                v = X[m].mean(axis=0)
                newC[c] = v / (np.linalg.norm(v) + 1e-9)
            else:
                newC[c] = C[c]
        if np.allclose(newC, C):
            C = newC
            break
        C = newC
    return labels, C


def _cid(kind: str, namespace: str, level: int, members: list[str]) -> str:
    h = hashlib.sha1(("|".join([kind, namespace, str(level)] + sorted(members))).encode()).hexdigest()
    return f"der_{h[:16]}"


def build_tree(items: list[tuple[str, np.ndarray]], *, namespace: str = "default",
               levels: int = 3, min_cluster: int = 4, max_k: int = 64) -> list[DerivedRecord]:
    """items = [(member_id, content_vector)]. Returns derived gist nodes across levels.
    NEVER mutates inputs; produces only additive DerivedRecords."""
    if len(items) < min_cluster:
        return []
    ids = [i for i, _ in items]
    vecs = _normalize(np.array([v for _, v in items], dtype=np.float32))
    derived: list[DerivedRecord] = []
    level = 1
    while len(ids) > min_cluster and level <= levels:
        k = min(max_k, max(2, len(ids) // min_cluster))   # bounded k -> near-linear
        labels, centers = _kmeans(vecs, k, seed=level)
        next_ids, next_vecs = [], []
        for c in range(centers.shape[0]):
            members = [ids[i] for i in range(len(ids)) if labels[i] == c]
            if not members:
                continue
            cid = _cid("gist", namespace, level, members)
            derived.append(DerivedRecord(
                cid=cid, kind="gist", namespace=namespace, level=level,
                text=f"gist[L{level}] of {len(members)} items",
                member_ids=members, vector=centers[c].tolist(),
                confidence=1.0, provenance="kmeans-centroid",
            ))
            next_ids.append(cid)
            next_vecs.append(centers[c])
        if len(next_ids) <= 1:
            break
        ids, vecs = next_ids, _normalize(np.array(next_vecs, dtype=np.float32))
        level += 1
    return derived


def search(query_vec: np.ndarray, gists: list[DerivedRecord], k: int = 5) -> list[tuple[DerivedRecord, float]]:
    """Rank gist nodes by cosine to the query (multi-resolution: any level can match)."""
    if not gists:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scored = []
    for g in gists:
        if not g.vector:
            continue
        v = np.array(g.vector, dtype=np.float32)
        scored.append((g, float(v @ q / (np.linalg.norm(v) + 1e-9))))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
