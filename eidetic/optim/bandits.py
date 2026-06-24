"""Layer 1c -- bandits for always-on config selection (closed-form numpy, O(log T) regret).

The backbone of "always-on" knob selection: discretize config knobs (fusion-method presets,
rerank depth, efSearch, top-k) into arms and let a bandit pick per query class from DEV-split
reward. Bandits dominate RL on ROI here (most of the benefit, provable regret, no GPU).

  * UCB1            -- deterministic optimism: mu_i + sqrt(2 ln t / n_i).
  * ThompsonBeta    -- Beta(a,b) posteriors; sample, argmax, update (a,b)+=(r,1-r). Lower
                       empirical regret early (the prior calibrates sparse-data pulls).
  * LinUCB          -- contextual: a_t = argmax theta_a.x + alpha*sqrt(x^T A_a^{-1} x), so the
                       arm choice depends on query features (length, type, cluster).
  * DiscountedUCB   -- sliding-window/discounted counts for NON-STATIONARITY (a named pitfall):
                       recent reward weighs more, so the bandit tracks drift.

All persist trivially as (counts, values) / (alpha, beta) -- feed them from FeedbackBuffer.
"""
from __future__ import annotations

import numpy as np


class UCB1:
    def __init__(self, n_arms: int):
        self.n = n_arms
        self.counts = np.zeros(n_arms, dtype=np.float64)
        self.values = np.zeros(n_arms, dtype=np.float64)
        self.t = 0

    def select(self) -> int:
        for a in range(self.n):
            if self.counts[a] == 0:          # play every arm once first
                return a
        t = max(1, self.t)
        ucb = self.values + np.sqrt(2.0 * np.log(t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class ThompsonBeta:
    def __init__(self, n_arms: int, seed: int = 0):
        self.alpha = np.ones(n_arms, dtype=np.float64)
        self.beta = np.ones(n_arms, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r

    def best_arm(self) -> int:
        return int(np.argmax(self.alpha / (self.alpha + self.beta)))


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010): a ridge model per arm over the query context."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, lam: float = 1.0):
        self.n, self.d, self.alpha = n_arms, dim, alpha
        self.A = [lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, x) -> int:
        x = np.asarray(x, dtype=np.float64)
        best_p, best_a = -np.inf, 0
        for a in range(self.n):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            p = float(theta @ x + self.alpha * np.sqrt(max(0.0, x @ Ainv @ x)))
            if p > best_p:
                best_p, best_a = p, a
        return best_a

    def update(self, arm: int, x, reward: float) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x


class DiscountedUCB:
    """Discounted UCB (Garivier & Moulines) for non-stationary rewards: every pull decays the
    accumulated counts/sums by gamma so recent outcomes dominate."""

    def __init__(self, n_arms: int, gamma: float = 0.95):
        self.n, self.gamma = n_arms, gamma
        self.counts = np.zeros(n_arms, dtype=np.float64)
        self.sums = np.zeros(n_arms, dtype=np.float64)

    def select(self) -> int:
        for a in range(self.n):
            if self.counts[a] < 1e-9:
                return a
        total = self.counts.sum()
        means = self.sums / np.maximum(self.counts, 1e-9)
        ucb = means + np.sqrt(2.0 * np.log(max(total, 1.0)) / np.maximum(self.counts, 1e-9))
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts *= self.gamma
        self.sums *= self.gamma
        self.counts[arm] += 1.0
        self.sums[arm] += float(reward)

    def means(self) -> np.ndarray:
        return self.sums / np.maximum(self.counts, 1e-9)
