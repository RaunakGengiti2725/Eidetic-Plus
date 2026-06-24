"""TransE knowledge-graph embedding in pure numpy (no torch). h + r ~= t.

Trains by margin-ranking SGD over the observed triples (near-linear: O(epochs * |triples|)),
then scores candidate triples for link prediction. plausibility(h,r,t) = exp(-||h+r-t||) in
(0,1]. Token-free. The CALLER (infer.py) generates a BOUNDED candidate set (2-hop paths),
so we never score all entity pairs (no O(N^2)).
"""
from __future__ import annotations

import numpy as np


class TransE:
    def __init__(self, dim: int = 32, margin: float = 1.0, lr: float = 0.05, seed: int = 0):
        self.dim = dim
        self.margin = margin
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.ent_ix: dict[str, int] = {}
        self.rel_ix: dict[str, int] = {}
        self.E: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self.R: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._triples: set[tuple[int, int, int]] = set()

    def _idx(self, table: dict, key: str) -> int:
        if key not in table:
            table[key] = len(table)
        return table[key]

    def fit(self, triples: list[tuple[str, str, str]], epochs: int = 30) -> "TransE":
        if not triples:
            return self
        ids = []
        for h, r, t in triples:
            ids.append((self._idx(self.ent_ix, h), self._idx(self.rel_ix, r), self._idx(self.ent_ix, t)))
        self._triples = set(ids)
        ne, nr = len(self.ent_ix), len(self.rel_ix)
        bound = 6.0 / np.sqrt(self.dim)
        self.E = self.rng.uniform(-bound, bound, (ne, self.dim)).astype(np.float32)
        self.R = self.rng.uniform(-bound, bound, (nr, self.dim)).astype(np.float32)
        self.R /= (np.linalg.norm(self.R, axis=1, keepdims=True) + 1e-9)
        arr = np.array(ids)
        for _ in range(epochs):
            self.E /= (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-9)  # TransE constraint
            self.rng.shuffle(arr)
            for h, r, t in arr:
                # corrupt the tail (cheap negative sampling)
                t_neg = int(self.rng.integers(ne))
                pos = self.E[h] + self.R[r] - self.E[t]
                neg = self.E[h] + self.R[r] - self.E[t_neg]
                d_pos, d_neg = float(np.linalg.norm(pos)), float(np.linalg.norm(neg))
                if self.margin + d_pos - d_neg <= 0:
                    continue
                gp = pos / (d_pos + 1e-9)
                gn = neg / (d_neg + 1e-9)
                self.E[h] -= self.lr * (gp - gn)
                self.R[r] -= self.lr * (gp - gn)
                self.E[t] -= self.lr * (-gp)
                self.E[t_neg] -= self.lr * (gn)
        return self

    def known(self, h: str, r: str, t: str) -> bool:
        try:
            return (self.ent_ix[h], self.rel_ix[r], self.ent_ix[t]) in self._triples
        except KeyError:
            return False

    def score(self, h: str, r: str, t: str) -> float:
        """plausibility in (0,1]; 0.0 if any symbol is unknown."""
        if h not in self.ent_ix or t not in self.ent_ix or r not in self.rel_ix:
            return 0.0
        d = float(np.linalg.norm(self.E[self.ent_ix[h]] + self.R[self.rel_ix[r]] - self.E[self.ent_ix[t]]))
        return float(np.exp(-d))

    @property
    def relations(self) -> list[str]:
        return list(self.rel_ix)
