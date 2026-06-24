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
    """The geometric budget ladder min, min*eta, ... up to max_budget. Guards against the
    non-terminating cases (min_budget < 1 would never grow; eta < 2 would never advance)."""
    if min_budget < 1:
        raise ValueError("min_budget must be >= 1")
    if eta < 2:
        raise ValueError("eta must be >= 2")
    budgets, b = [], int(min_budget)
    while b < max_budget:
        budgets.append(b)
        b *= eta
    if not budgets or budgets[-1] != int(max_budget):
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
    survivor, survivor_score = (alive[0] if alive else None), -math.inf
    for b in budgets:
        scored = [(c, float(eval_fn(c, b))) for c in alive]
        ranked = sorted(scored, key=lambda r: r[1], reverse=True)
        k = max(1, math.ceil(len(ranked) / eta))
        survivors_scored = ranked[:k]
        # The reported (survivor, score) is always the best survivor AT THIS rung's budget, so
        # a config that spiked at a smaller budget but was then culled never mislabels the
        # result (correctness under a non-monotone eval_fn).
        survivor, survivor_score = survivors_scored[0]
        rungs.append({"budget": b, "n_in": len(alive), "n_out": len(survivors_scored)})
        alive = [c for c, _ in survivors_scored]
        if len(alive) <= 1:
            break
    return {"survivor": survivor, "score": survivor_score, "rungs": rungs}
