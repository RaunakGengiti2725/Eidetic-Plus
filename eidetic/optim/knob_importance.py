"""Cross-cutting (OtterTune-style) -- Lasso knob-importance, numpy-only.

Fit a Lasso (L1) regression of trial outcome on the one-hot knob assignments and rank knobs
by |coefficient|. The L1 penalty zeros out knobs with no measurable effect, so the expensive
TPE/NSGA search can be restricted to the knobs that actually matter (pruning the search space,
the OtterTune transfer). No sklearn -- coordinate-descent soft-thresholding in numpy.
"""
from __future__ import annotations

import numpy as np


def _soft_threshold(z: float, gamma: float) -> float:
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def lasso_coordinate_descent(X, y, lam: float = 0.1, iters: int = 200,
                             tol: float = 1e-6) -> np.ndarray:
    """Minimize (1/2n)||y - Xw||^2 + lam*||w||_1 over standardized columns. Returns w in the
    standardized space (|w_j| is the importance of column j)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    # standardize columns (unit variance) so lam is comparable across knobs.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    yc = y - y.mean()
    w = np.zeros(p)
    col_ss = (Xs ** 2).sum(axis=0) / n
    col_ss[col_ss == 0] = 1.0
    for _ in range(iters):
        w_old = w.copy()
        for j in range(p):
            r = yc - Xs @ w + Xs[:, j] * w[j]
            rho = (Xs[:, j] @ r) / n
            w[j] = _soft_threshold(rho, lam) / col_ss[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def knob_importance(trials: list[dict], knobs: list[str], score_key: str = "score",
                    lam: float = 0.05) -> list[tuple[str, float]]:
    """Rank knobs by Lasso importance. trials = [{knob: value, ..., score: float}]. Categorical
    values are one-hot encoded; the per-knob importance is the max |coef| over its levels.
    Returns [(knob, importance)] sorted descending."""
    if not trials:
        return [(k, 0.0) for k in knobs]
    columns: list[tuple[str, list[float]]] = []
    for k in knobs:
        levels = sorted({t.get(k) for t in trials})
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in levels)
        if numeric and len(levels) > 2:
            columns.append((k, [float(t.get(k, 0.0)) for t in trials]))
        else:
            for lv in levels:                    # one-hot per level
                columns.append((k, [1.0 if t.get(k) == lv else 0.0 for t in trials]))
    X = np.array([col for _, col in columns], dtype=float).T
    y = np.array([float(t.get(score_key, 0.0)) for t in trials])
    w = np.abs(lasso_coordinate_descent(X, y, lam=lam))
    agg: dict[str, float] = {}
    for (k, _), wi in zip(columns, w):
        agg[k] = max(agg.get(k, 0.0), float(wi))
    return sorted(agg.items(), key=lambda kv: -kv[1])
