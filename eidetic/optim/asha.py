"""Layer 1c -- Successive Halving / ASHA early stopping, numpy-only.

Evaluate many configs cheaply, keep only the top 1/eta at each rung, and re-evaluate the
survivors at a larger budget. Weak configs die early, so the eval budget concentrates on the
promising ones (the AutoRAG-HP ~5x tuning-cost saving). Scores are HIGHER-IS-BETTER.
"""
from __future__ import annotations

import math


def top_fraction(results: list[tuple], eta: int = 3) -> list:
    """Keep the best ceil(n/eta) items. results = [(item, score), ...], higher score better.
    Returns the surviving items (not the scores)."""
    if not results:
        return []
    k = max(1, math.ceil(len(results) / eta))
    ranked = sorted(results, key=lambda r: r[1], reverse=True)
    return [item for item, _ in ranked[:k]]


def rung_budgets(min_budget: int, max_budget: int, eta: int = 3) -> list[int]:
    """The geometric budget ladder min, min*eta, ... up to max_budget."""
    budgets, b = [], int(min_budget)
    while b < max_budget:
        budgets.append(b)
        b *= eta
    budgets.append(int(max_budget))
    return budgets


def successive_halving(configs: list, eval_fn, min_budget: int = 1, max_budget: int = 9,
                       eta: int = 3) -> dict:
    """Run synchronous Successive Halving. eval_fn(config, budget) -> score (higher better).
    Returns {survivor, score, rungs:[{budget, n_in, n_out}]}. The total work is far below a
    full grid because each rung culls to the top 1/eta."""
    budgets = rung_budgets(min_budget, max_budget, eta)
    alive = list(configs)
    rungs = []
    best_item, best_score = None, -math.inf
    for b in budgets:
        scored = [(c, float(eval_fn(c, b))) for c in alive]
        for c, s in scored:
            if s > best_score:
                best_item, best_score = c, s
        survivors = top_fraction(scored, eta)
        rungs.append({"budget": b, "n_in": len(alive), "n_out": len(survivors)})
        alive = survivors
        if len(alive) <= 1:
            break
    return {"survivor": alive[0] if alive else best_item, "score": best_score, "rungs": rungs}
