"""Layer 3e -- FadeMem utility decay (stretched-exponential), numpy-only.

Clean closed-form forgetting math that complements the existing FSRS power-law layer:

  decay:        v_i(t) = v_i(0) * exp(-lambda_i * (t - tau_i)^beta_i)
  rate:         lambda_i = lambda_base * exp(-mu * I_i)        (importance slows decay)
  reinforce:    v <- v + dv*(1 - v)*exp(-n/N)                  (saturating; spacing effect)
  half-life:    t_half = (ln 2 / lambda_i)^(1/beta_i)

AGE-INDEPENDENCE INVARIANT (critical): FadeMem strength feeds INDEX priority / pruning ONLY,
exactly like FSRS. It must NEVER enter the retrieval ranking score, or the flat recall-vs-age
curve slopes. The repo deliberately keeps retrievability out of the ranker; this honors that.
"""
from __future__ import annotations

import math


def lambda_from_importance(importance: float, lambda_base: float = 0.1,
                           mu: float = 1.0) -> float:
    """Importance-modulated decay rate: more important -> smaller lambda -> slower forgetting."""
    return float(lambda_base * math.exp(-mu * max(0.0, importance)))


def retention(v0: float, lam: float, beta: float, t: float, tau: float = 0.0) -> float:
    """Stretched-exponential retained strength at time t (t,tau in the same units)."""
    dt = max(0.0, t - tau)
    return float(v0 * math.exp(-lam * (dt ** beta)))


def half_life(lam: float, beta: float = 1.0) -> float:
    """Time for strength to fall to half v0."""
    if lam <= 0:
        return math.inf
    return float((math.log(2.0) / lam) ** (1.0 / beta))


def reinforce(v: float, dv: float, n: int, N: float) -> float:
    """Access reinforcement: bump strength toward 1, saturating (1-v) and with diminishing
    returns as the access count n grows relative to N (the spacing effect)."""
    return float(min(1.0, v + dv * (1.0 - v) * math.exp(-n / max(1e-9, N))))
