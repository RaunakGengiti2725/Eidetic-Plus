"""Layer 1b -- multi-objective Pareto tools (NSGA-II), numpy-only.

For tuning over (accuracy, latency, tokens) the right object is the Pareto front: the configs
where improving one objective worsens another. NSGA-II ranks trials by fast non-dominated
sorting and breaks ties by crowding distance (diversity). All objectives here are MINIMIZED;
maximize accuracy by passing -accuracy.
"""
from __future__ import annotations

import numpy as np


def dominates(a, b) -> bool:
    """True if a dominates b (minimization): a <= b in every objective and < in at least one."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_nondominated_sort(objectives) -> list[list[int]]:
    """Return the NSGA-II fronts as lists of indices (front 0 = the Pareto-optimal set)."""
    P = [np.asarray(o, dtype=float) for o in objectives]
    n = len(P)
    S = [[] for _ in range(n)]          # solutions each point dominates
    ndom = [0] * n                       # how many dominate each point
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(P[p], P[q]):
                S[p].append(q)
            elif dominates(P[q], P[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]                    # drop the trailing empty front


def pareto_front(objectives) -> list[int]:
    """Indices of the non-dominated (Pareto-optimal) trials."""
    fronts = fast_nondominated_sort(objectives)
    return fronts[0] if fronts else []


def crowding_distance(objectives, indices: list[int]) -> dict[int, float]:
    """NSGA-II crowding distance within one front. Boundary points get +inf (always kept)."""
    if not indices:
        return {}
    M = np.asarray([objectives[i] for i in indices], dtype=float)
    n, m = M.shape
    dist = {i: 0.0 for i in indices}
    for obj in range(m):
        order = np.argsort(M[:, obj])
        lo, hi = M[order[0], obj], M[order[-1], obj]
        dist[indices[order[0]]] = float("inf")
        dist[indices[order[-1]]] = float("inf")
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, n - 1):
            i = indices[order[k]]
            if dist[i] == float("inf"):
                continue
            dist[i] += (M[order[k + 1], obj] - M[order[k - 1], obj]) / span
    return dist


def hypervolume_2d(points, ref) -> float:
    """2D hypervolume dominated by the (minimization) point set up to reference `ref`.
    Larger = a better-spread, closer-to-origin front. Points worse than ref are ignored."""
    ref = np.asarray(ref, dtype=float)
    pts = [np.asarray(p, dtype=float) for p in points if np.all(np.asarray(p) <= ref)]
    front = [pts[i] for i in pareto_front(pts)] if pts else []
    front.sort(key=lambda p: p[0])       # ascending objective 0
    hv, prev_x = 0.0, ref[0]
    for p in reversed(front):            # sweep from the largest x downward
        hv += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return float(hv)
