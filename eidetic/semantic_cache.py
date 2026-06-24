"""Semantic answer cache in front of the read path: exact-hash + cosine >= threshold.

Repeated or near-duplicate queries skip the whole retrieve+generate+verify pipeline,
which is a large share of the cost win on real workloads. Scoped per namespace so a
cache hit never crosses a scope boundary. In-memory and bounded.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


class SemanticCache:
    def __init__(self, cosine_threshold: float = 0.9, max_entries: int = 2048,
                 adaptive: bool = True):
        self.threshold = cosine_threshold
        self.max_entries = max_entries
        self.adaptive = adaptive
        self._exact: dict[str, Any] = {}                     # (scope_key, query) -> value
        self._vecs: list[tuple[str, np.ndarray, Any]] = []   # (scope_key, qvec, value)

    @staticmethod
    def _hash(scope_key: str, query: str) -> str:
        return hashlib.sha256(f"{scope_key}\x1f{query.strip().lower()}".encode()).hexdigest()

    def threshold_for(self, query: str) -> float:
        if not self.adaptive:
            return self.threshold
        terms = re.findall(r"[a-z0-9]+", query.lower())
        t = self.threshold
        if len(terms) <= 4:
            t += 0.05
        elif len(terms) >= 12:
            t -= 0.04
        if re.search(r"\b(before|after|previous|previously|current|latest|now|today|when)\b", query.lower()):
            t += 0.03
        if re.search(r"\b[A-Z]{2,}[-_A-Z0-9]*\b|\d", query):
            t += 0.02
        return max(0.80, min(0.98, t))

    def get(self, scope_key: str, query: str, qvec: Optional[np.ndarray]) -> Optional[Any]:
        h = self._hash(scope_key, query)
        if h in self._exact:
            return self._exact[h]
        if qvec is None or not self._vecs:
            return None
        q = _norm(qvec)
        best, best_sim = None, 0.0
        for sk, v, value in self._vecs:
            if sk != scope_key:
                continue
            sim = float(v @ q)
            if sim > best_sim:
                best, best_sim = value, sim
        return best if best_sim >= self.threshold_for(query) else None

    def put(self, scope_key: str, query: str, qvec: Optional[np.ndarray], value: Any) -> None:
        self._exact[self._hash(scope_key, query)] = value
        if qvec is not None:
            self._vecs.append((scope_key, _norm(qvec), value))
            if len(self._vecs) > self.max_entries:
                self._vecs.pop(0)

    def clear(self) -> None:
        self._exact.clear()
        self._vecs.clear()
