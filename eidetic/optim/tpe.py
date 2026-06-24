"""Layer 1a -- Tree-structured Parzen Estimator (TPE), numpy-only (no Optuna).

TPE models p(x | y) as two densities over past trials: l(x) from the trials whose loss is
below the gamma-quantile (the "good" set) and g(x) from the rest. It proposes the config
maximizing l(x)/g(x), which is monotone in Expected Improvement. Lightweight, handles
continuous / integer / categorical knobs, and recommended for < 1000 trials -- exactly a
memory-benchmark eval budget.

Search-space spec per knob:
    ("uniform", lo, hi)     continuous
    ("int", lo, hi)         integer (rounded)
    ("categorical", [vals]) discrete choice
"""
from __future__ import annotations

import numpy as np

_SQRT2PI = np.sqrt(2.0 * np.pi)


class TPESampler:
    def __init__(self, space: dict, gamma: float = 0.25, n_startup: int = 8,
                 n_candidates: int = 24, seed: int = 0):
        self.space = space
        self.gamma = gamma
        self.n_startup = n_startup
        self.n_candidates = n_candidates
        self.rng = np.random.default_rng(seed)
        self.trials: list[tuple[dict, float]] = []

    # ---- random draw for the startup phase --------------------------------
    def _sample_one(self, spec):
        kind = spec[0]
        if kind == "categorical":
            return self.rng.choice(spec[1]).item() if hasattr(self.rng.choice(spec[1]), "item") \
                else self.rng.choice(spec[1])
        lo, hi = spec[1], spec[2]
        v = self.rng.uniform(lo, hi)
        return int(round(v)) if kind == "int" else float(v)

    def suggest(self) -> dict:
        if len(self.trials) < self.n_startup:
            return {n: self._sample_one(s) for n, s in self.space.items()}
        losses = np.array([t[1] for t in self.trials])
        order = np.argsort(losses)
        n_good = max(1, int(np.ceil(self.gamma * len(self.trials))))
        good_idx = set(order[:n_good].tolist())
        good = [self.trials[i][0] for i in range(len(self.trials)) if i in good_idx]
        bad = [self.trials[i][0] for i in range(len(self.trials)) if i not in good_idx]
        return {n: self._suggest_knob(n, s, good, bad) for n, s in self.space.items()}

    def _suggest_knob(self, name, spec, good, bad):
        kind = spec[0]
        if kind == "categorical":
            choices = list(spec[1])

            def prob(vals, c):
                return (sum(1 for v in vals if v == c) + 1.0) / (len(vals) + len(choices))

            gvals = [g[name] for g in good]
            bvals = [b[name] for b in bad] if bad else []
            ratios = {c: prob(gvals, c) / prob(bvals, c) for c in choices}
            return max(ratios, key=ratios.get)

        lo, hi = spec[1], spec[2]
        gv = np.array([g[name] for g in good], dtype=float)
        bv = np.array([b[name] for b in bad], dtype=float) if bad else np.array([])
        # Per-set bandwidths (textbook TPE): l(x) and g(x) each adapt to their own size, so the
        # larger bad set is not over-smoothed by the good set's bandwidth.
        h_good = max((hi - lo) / np.sqrt(max(len(gv), 1)), (hi - lo) * 0.1, 1e-6)
        h_bad = max((hi - lo) / np.sqrt(max(len(bv), 1)), (hi - lo) * 0.1, 1e-6)

        cands = np.clip(self.rng.choice(gv, size=self.n_candidates) +
                        self.rng.normal(0.0, h_good, size=self.n_candidates), lo, hi)
        cands = np.concatenate([cands, gv])           # include the good points themselves

        def kde(x, pts, h):
            if pts.size == 0:
                return np.full(x.shape, 1e-9)
            d = (x[:, None] - pts[None, :]) / h
            return np.mean(np.exp(-0.5 * d * d), axis=1) / (h * _SQRT2PI) + 1e-12

        gx = kde(cands, gv, h_good)
        bx = kde(cands, bv, h_bad) if bv.size else np.full(cands.shape, 1.0)  # empty bad -> flat
        best = float(cands[int(np.argmax(gx / bx))])
        return int(round(best)) if kind == "int" else best

    def observe(self, config: dict, loss: float) -> None:
        self.trials.append((dict(config), float(loss)))

    def best(self) -> tuple[dict, float] | None:
        if not self.trials:
            return None
        return min(self.trials, key=lambda t: t[1])
