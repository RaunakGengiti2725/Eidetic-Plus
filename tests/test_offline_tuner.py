"""Offline tests for the Layer-1 tuner: each algorithm checked against a KNOWN answer."""
from __future__ import annotations

import numpy as np

from eidetic.optim.asha import rung_budgets, successive_halving, top_fraction
from eidetic.optim.knob_importance import knob_importance, lasso_coordinate_descent
from eidetic.optim.pareto import (crowding_distance, dominates, fast_nondominated_sort,
                                  hypervolume_2d, pareto_front)
from eidetic.optim.tpe import TPESampler


# ---- TPE: concentrates near the known optimum -------------------------------
def test_tpe_finds_and_concentrates_near_optimum():
    # minimize (x - 0.7)^2 on [0,1]; optimum at x=0.7.
    s = TPESampler({"x": ("uniform", 0.0, 1.0)}, gamma=0.25, n_startup=8, seed=3)
    last = []
    for i in range(60):
        cfg = s.suggest()
        s.observe(cfg, (cfg["x"] - 0.7) ** 2)
        if i >= 50:
            last.append(cfg["x"])
    best_cfg, best_loss = s.best()
    assert abs(best_cfg["x"] - 0.7) < 0.06        # found the optimum
    assert abs(np.mean(last) - 0.7) < 0.2          # late samples cluster around it (not random)


def test_tpe_categorical_prefers_the_good_choice():
    s = TPESampler({"c": ("categorical", ["a", "b", "c"])}, n_startup=6, seed=1)
    for _ in range(40):
        cfg = s.suggest()
        s.observe(cfg, 0.0 if cfg["c"] == "b" else 1.0)   # 'b' is best (loss 0)
    # after learning, suggestions should favor 'b'
    picks = [s.suggest()["c"] for _ in range(10)]
    assert picks.count("b") >= 8


# ---- NSGA-II: hand-checked non-dominated set --------------------------------
def test_nondominated_sort_known_points():
    # minimization. (1,1),(2,0.5),(0.5,2) are mutually non-dominated; (3,3),(1.5,1.5) dominated.
    pts = [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (3.0, 3.0), (1.5, 1.5)]
    assert set(pareto_front(pts)) == {0, 1, 2}
    assert dominates((1.0, 1.0), (1.5, 1.5))
    assert not dominates((2.0, 0.5), (0.5, 2.0))
    fronts = fast_nondominated_sort(pts)
    assert set(fronts[0]) == {0, 1, 2}
    assert 3 in fronts[-1]                          # the strictly-worst point is in the last front


def test_crowding_distance_boundaries_are_infinite():
    pts = [(0.0, 3.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.0)]
    cd = crowding_distance(pts, pareto_front(pts))
    infs = [i for i, d in cd.items() if d == float("inf")]
    assert len(infs) == 2                            # the two extremes are always kept


def test_hypervolume_increases_with_a_better_front():
    ref = (4.0, 4.0)
    worse = hypervolume_2d([(3.0, 3.0)], ref)
    better = hypervolume_2d([(1.0, 3.0), (3.0, 1.0), (2.0, 2.0)], ref)
    assert better > worse > 0


# ---- ASHA / Successive Halving ----------------------------------------------
def test_top_fraction_keeps_ceiling():
    survivors = top_fraction([("a", 1), ("b", 5), ("c", 3), ("d", 2), ("e", 4)], eta=2)
    assert set(survivors) == {"b", "e", "c"}        # ceil(5/2)=3 best


def test_rung_budgets_ladder():
    assert rung_budgets(1, 9, eta=3) == [1, 3, 9]


def test_successive_halving_keeps_the_best():
    # config index == true quality; SHA over 9 configs with eta=3 should survive #8.
    res = successive_halving(list(range(9)), eval_fn=lambda c, b: c,
                             min_budget=1, max_budget=9, eta=3)
    assert res["survivor"] == 8
    assert [r["n_out"] for r in res["rungs"]][:2] == [3, 1]   # 9 -> 3 -> 1


def test_successive_halving_score_matches_survivor_under_nonmonotone():
    # config 8 spikes at budget 1 then collapses; config 7 is steadily good. The reported
    # score must belong to the actual final survivor (7@7), not 8's stale early spike (100).
    def eval_fn(c, b):
        if c == 8:
            return 100.0 if b == 1 else 0.0
        return float(c)
    res = successive_halving(list(range(9)), eval_fn, min_budget=1, max_budget=9, eta=3)
    assert res["survivor"] == 7
    assert res["score"] == 7.0


def test_rung_budgets_rejects_nonterminating_inputs():
    import pytest
    with pytest.raises(ValueError):
        rung_budgets(0, 9, eta=3)        # min_budget < 1 would never grow
    with pytest.raises(ValueError):
        rung_budgets(1, 9, eta=1)        # eta < 2 would never advance


# ---- Lasso knob importance --------------------------------------------------
def test_lasso_selects_the_driving_feature():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = 3.0 * X[:, 0] + 0.01 * rng.standard_normal(200)   # only feature 0 matters
    w = np.abs(lasso_coordinate_descent(X, y, lam=0.05))
    assert w[0] > 0.5
    assert w[1] < 0.1 and w[2] < 0.1                 # irrelevant knobs zeroed


def test_knob_importance_ranks_the_impactful_knob_first():
    # 'rerank' drives the score; 'noise' does nothing.
    trials = []
    rng = np.random.default_rng(1)
    for _ in range(60):
        rk = rng.choice([0, 1])
        noise = rng.choice(["x", "y"])
        trials.append({"rerank": rk, "noise": noise,
                       "score": 0.9 * rk + 0.005 * rng.standard_normal()})
    ranked = knob_importance(trials, ["rerank", "noise"], lam=0.02)
    assert ranked[0][0] == "rerank"
    assert ranked[0][1] > ranked[1][1]


def test_tpe_sweep_space_respects_the_wall():
    # The new TPE sampler path must inherit the rebuild-knob blacklist (no online index rebuilds).
    from bench.sweep import STAGES, build_tpe_space
    from eidetic.optim import REBUILD_KNOBS_ENV

    space = build_tpe_space()
    assert set(space) == {env for env, _ in STAGES}
    assert not (set(space) & REBUILD_KNOBS_ENV)

