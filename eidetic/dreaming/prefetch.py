"""Predictive pre-fetch: cluster the query log, pre-assemble each cluster's answer CONTEXT
during idle time, and at query time match the incoming query embedding to the nearest
pre-assembled context (cosine >= threshold) -- near-zero assembly latency, ZERO tokens.

Pre-assembly uses retrieval + context assembly, both token-free (no reader/LLM call here);
the reader still generates the answer, but the expensive retrieve+assemble is skipped on a
hit. Deploy only where the cluster hit-rate clears the threshold (measured).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .multires import _kmeans, _normalize


class PrefetchCache:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.centroids: list[np.ndarray] = []
        self.contexts: list[list[str]] = []
        self.hits = 0
        self.misses = 0

    def add(self, centroid_vec: np.ndarray, context_blocks: list[str]) -> None:
        v = np.asarray(centroid_vec, dtype=np.float32)
        self.centroids.append(v / (np.linalg.norm(v) + 1e-9))
        self.contexts.append(context_blocks)

    def get(self, query_vec: np.ndarray) -> Optional[list[str]]:
        if not self.centroids:
            self.misses += 1
            return None
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = [float(c @ q) for c in self.centroids]
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            self.hits += 1
            return self.contexts[i]
        self.misses += 1
        return None

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @staticmethod
    def cluster_queries(query_vecs: np.ndarray, max_clusters: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Cluster the query log (bounded-k k-means, near-linear). Returns (labels, centroids)."""
        if query_vecs.shape[0] == 0:
            return np.zeros(0, dtype=int), np.zeros((0, query_vecs.shape[1] if query_vecs.ndim == 2 else 0))
        X = _normalize(query_vecs.astype(np.float32))
        k = min(max_clusters, X.shape[0])
        return _kmeans(X, k)
