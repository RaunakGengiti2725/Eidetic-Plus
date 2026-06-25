"""Track 9 Flow / Instinct Recall: the ActivationField.

A per-namespace, in-memory, ephemeral working-memory substrate. Confirmed recalls inject
activation; every turn decays it; reads snapshot it. It is pure local math -- no model call, no
store/graph write -- and rebuilds empty on restart. Activation is ACCESS-recency only: it is never
a function of a memory's age (valid_at), so it cannot reintroduce age bias into recall.

The field knows nothing about the graph; one-hop spreading lives in the Engine Flow hub (which has
the graph) and lands back here as plain inject() calls. Namespace is the hard isolation boundary --
one namespace's activation is invisible to another, matching the rest of the engine.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class ActivationField:
    def __init__(self, decay: float = 0.85, floor: float = 0.05, cap: int = 512) -> None:
        self.decay_factor = float(decay)
        self.floor = float(floor)
        self.cap = max(1, int(cap))
        self._lock = threading.Lock()
        self._field: dict[str, dict[str, float]] = {}

    def inject(self, namespace: str, ids, amount: float = 1.0) -> None:
        """Additively raise activation for `ids` (clamped to 1.0), then enforce the cap."""
        if not ids:
            return
        with self._lock:
            m = self._field.setdefault(namespace, {})
            for i in ids:
                if not i:
                    continue
                m[i] = min(1.0, m.get(i, 0.0) + amount)
            self._evict_locked(m)

    def decay(self, namespace: str, factor: Optional[float] = None,
              salience: Optional[Callable[[str], float]] = None) -> None:
        """Multiply every id's activation by the decay factor and prune anything below the floor.
        With a `salience` callable, each id decays by `factor ** (1 - clamp01(salience(id)))`, so a
        more salient memory fades slower (salience=1 -> no decay, salience=0 -> full decay).
        `salience` MUST be access-time only (static importance / usage counts), never an age term."""
        base = self.decay_factor if factor is None else float(factor)
        with self._lock:
            m = self._field.get(namespace)
            if not m:
                return
            out: dict[str, float] = {}
            for k, v in m.items():
                f = base if salience is None else base ** (1.0 - _clamp01(salience(k)))
                nv = v * f
                if nv >= self.floor:
                    out[k] = nv
            self._field[namespace] = out

    def get(self, namespace: str, memory_id: str) -> float:
        with self._lock:
            return self._field.get(namespace, {}).get(memory_id, 0.0)

    def snapshot(self, namespace: str) -> dict[str, float]:
        with self._lock:
            return dict(self._field.get(namespace, {}))

    def _evict_locked(self, m: dict[str, float]) -> None:
        if len(m) <= self.cap:
            return
        for k, _ in sorted(m.items(), key=lambda kv: kv[1])[: len(m) - self.cap]:
            del m[k]
