"""Semantic answer cache in front of the read path: exact-hash + cosine >= threshold.

Repeated or near-duplicate queries skip the whole retrieve+generate+verify pipeline,
which is a large share of the cost win on real workloads. Scoped per namespace so a
cache hit never crosses a scope boundary. In-memory and bounded.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
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
        # Both bounded to max_entries. _exact is an OrderedDict so it evicts oldest-first like
        # _vecs; an unbounded _exact pinned every historical Answer object and leaked memory in a
        # long-lived server (e.g. the MCP server) serving many distinct queries.
        self._exact: "OrderedDict[str, Any]" = OrderedDict()  # (scope_key, query) -> value
        self._vecs: list[tuple[str, np.ndarray, Any]] = []    # (scope_key, qvec, value)
        # Shared across concurrent asks on one Engine -> guard every mutation/read of the two
        # containers (OrderedDict mutation + list append/pop are not atomic under the GIL across
        # the move_to_end/popitem sequence).
        self._lock = threading.Lock()

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
        with self._lock:
            if h in self._exact:
                return self._exact[h]
            if qvec is None or not self._vecs:
                return None
            vecs = list(self._vecs)              # snapshot under lock; scoring is read-only
        q = _norm(qvec)
        best, best_sim = None, 0.0
        for sk, v, value in vecs:
            if sk != scope_key:
                continue
            sim = float(v @ q)
            if sim > best_sim:
                best, best_sim = value, sim
        return best if best_sim >= self.threshold_for(query) else None

    def put(self, scope_key: str, query: str, qvec: Optional[np.ndarray], value: Any) -> None:
        h = self._hash(scope_key, query)
        with self._lock:
            self._exact[h] = value
            self._exact.move_to_end(h)
            if len(self._exact) > self.max_entries:
                self._exact.popitem(last=False)          # evict oldest, mirroring _vecs
            if qvec is not None:
                self._vecs.append((scope_key, _norm(qvec), value))
                if len(self._vecs) > self.max_entries:
                    self._vecs.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._exact.clear()
            self._vecs.clear()
