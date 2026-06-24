"""Always-on optimization: a formula-level, lightweight (numpy/SQLite) menu of
continuous optimizers layered onto the Eidetic-Plus memory system.

This package implements the three-tier optimizer menu:

  * Layer 1 (offline auto-tuning): numpy TPE, NSGA-II/MOTPE Pareto search, ASHA
    early-stopping, Lasso knob-importance.                       -> optim/tpe.py,
                                                pareto.py, asha.py, knob_importance.py
  * Layer 2 (per-query hot path): adaptive-k largest-gap cut, split-conformal
    abstention/depth, score-fusion variants (z-score/min-max/Borda/DBSF/RSF),
    MMR diversity, TARG-style retrieve/skip gating.   -> optim/adaptive_k.py,
                                              conformal.py, fusion.py, mmr.py, gating.py
  * Layer 3 (background/continuous): FTRL + Exponentiated-Gradient online fusion
    weights, Rocchio PRF, hard-negative mining, scalar/binary (RaBitQ)
    quantization, UCB1/Thompson/LinUCB bandits.  -> optim/online_weights.py,
                                              rocchio.py, quantize.py, bandits.py

Three load-bearing invariants every optimizer here MUST honor:

  1. THE INTEGRITY WALL. No optimizer may read, fit to, or cache a benchmark TEST
     item. Learners read only the private dev split (see eidetic.feedback and
     bench.datasets.split_of). Reported numbers come from the disjoint test split.
  2. AGE-INDEPENDENCE. Learned weights, FadeMem strengths, and salience feed INDEX
     priority / pruning ONLY -- never the retrieval RANKING score. This preserves
     the signature flat recall-vs-age curve. A reward signal must never smuggle a
     recency/age term into the ranker.
  3. EVERYTHING DEFAULTS OFF. Each optimizer is behind a config flag defaulting to
     the current behavior, so the shipped baseline is unchanged and every optimizer
     is independently A/B-testable (the playbook's bandit-vs-static-RRF discipline).
"""
from __future__ import annotations

# OtterTune-style knob blacklist: changing any of these forces an HNSW index rebuild,
# so an always-on (hot/idle) tuner must NEVER flip them live. Only the explicit offline
# rebuild cadence may touch them. Mirrored as env-var names in bench/sweep.py.
REBUILD_KNOBS_SETTINGS = frozenset({"hnsw_m", "hnsw_ef_construction"})
REBUILD_KNOBS_ENV = frozenset({"HNSW_M", "HNSW_EF_CONSTRUCTION"})
