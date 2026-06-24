"""Layer 3f -- Markov predictive prefetcher: P(next | current).

Learn a first-order transition model over query-cluster sequences from the access log, then
prefetch the most-likely next cluster's context during idle time. Complements the existing
similarity-based prefetch (which matches the CURRENT query); this predicts the NEXT one from
observed order. Pure stdlib (a SQLite-able transition table); token-free.
"""
from __future__ import annotations

from collections import defaultdict


class MarkovPrefetcher:
    def __init__(self):
        self._counts: dict = defaultdict(lambda: defaultdict(int))
        self._last = None

    def observe_sequence(self, current, nxt) -> None:
        self._counts[current][nxt] += 1

    def observe(self, cluster) -> None:
        """Stream one access; the transition from the previous access is recorded."""
        if self._last is not None:
            self.observe_sequence(self._last, cluster)
        self._last = cluster

    def transition_prob(self, current, nxt) -> float:
        row = self._counts.get(current)
        if not row:
            return 0.0
        total = sum(row.values())
        return row.get(nxt, 0) / total if total else 0.0

    def predict(self, current, top_k: int = 3) -> list[tuple]:
        """Most-likely next clusters given the current one, as [(cluster, prob)] desc."""
        row = self._counts.get(current)
        if not row:
            return []
        total = sum(row.values())
        ranked = sorted(((c, n / total) for c, n in row.items()), key=lambda x: -x[1])
        return ranked[:top_k]
