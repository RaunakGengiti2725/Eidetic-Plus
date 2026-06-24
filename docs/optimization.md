# Always-On Optimization — a three-tier menu of continuous optimizers

This implements the formula-level optimizer menu from *"Always-On Optimization for Eidetic-Plus."*
Everything here is **lightweight (numpy / SQLite / DashScope)** and lands behind config flags
that **default to the current behavior** — so the shipped baseline is unchanged and every
optimizer is an independent A/B flag.

## The integrity wall (non-negotiable)

No continuous optimizer may read, fit to, or cache a benchmark **test** item.

- `bench/datasets.split_of(sample_id)` deterministically partitions every dataset into a
  private **dev** split (~20%) and a held-out **test** split. It is a stable hash — no extra file.
- `bench.run --split test` (and `reproduce.sh`) produce the **reported** numbers.
- `bench.sweep` is **locked to `--split dev`** (argparse rejects `test`); `bench.calibrate`
  firewalls to dev log rows.
- `eidetic/feedback.py` `FeedbackBuffer` records benchmark namespaces as `is_dev=0`
  (audit-only); `sample()` returns `is_dev=1` rows only, so no online learner can ever reach a
  benchmark item. The neutral benchmark adapter calls `retrieve()`/`assemble_context()`
  directly (never `ask()`), so feedback is double-walled from the benchmark.

Two more invariants every optimizer honors:

- **Age-independence.** Learned fusion weights and FadeMem strength feed **index priority /
  pruning only — never the retrieval ranking score**, preserving the flat recall-vs-age curve.
  (The recency channel weight is never learned.)
- **Never delete a raw record.** Index tombstones / pruning are index-layer only; the WORM
  substrate is untouched.

## The menu (what maps where)

| Layer | Optimizer | Module | Flag (default) |
|---|---|---|---|
| 2a | Adaptive-k (largest-gap cut) | `optim/adaptive_k.py` | `ADAPTIVE_K=0` |
| 2a | Adaptive efSearch (hard queries) | `vector_index.py` | `ADAPTIVE_EF=0` |
| 2a | TARG-style margin/entropy gating | `optim/gating.py` | `RERANK_SKIP_MARGIN=0` |
| 2b | Split-conformal depth/abstention | `optim/conformal.py` | `CONFORMAL_DEPTH=0` |
| 2c | MMR diversity | `optim/mmr.py` | `MMR_ENABLED=0` |
| 2d | Fusion: z-score / min-max / DBSF / Borda | `optim/fusion.py` | `FUSION_METHOD=rrf` |
| 2e | Parallel channel fan-out | `retrieval.py` | `PARALLEL_CHANNELS=0` |
| 2e | Semantic cache (LRU/LFU/ARC policies) | `optim/cache_policies.py` | (cache on) |
| 3a | FTRL + Exponentiated-Gradient fusion weights | `optim/online_weights.py` | `FUSION_LEARNER=0` |
| 3b | Rocchio PRF | `optim/rocchio.py` | `ROCCHIO=0` |
| 3b | Hard-negative mining | `feedback.py` | `FEEDBACK=0` |
| 3c | SQ8 + RaBitQ quantization + refine | `optim/quantize.py` | `VECTOR_QUANT=none` |
| 3e | FadeMem decay / reinforcement | `dreaming/fademem.py` | (index-priority) |
| 3f | Markov prefetcher | `optim/markov.py` | (idle cadence) |
| 1a | TPE sampler | `optim/tpe.py` | `bench.sweep --sampler tpe` |
| 1b | NSGA-II / MOTPE Pareto | `optim/pareto.py` | (sweep multi-objective) |
| 1c | UCB1 / Thompson / LinUCB / discounted | `optim/bandits.py` | (config selection) |
| 1c | ASHA / Successive Halving | `optim/asha.py` | (early stopping) |
| x | Lasso knob-importance (OtterTune) | `optim/knob_importance.py` | (search-space pruning) |

The daemon (`optim/daemon.py`) ties the three cadences together as **deterministic,
individually-callable tick methods** (no live thread): `idle_tick()` replays dev feedback into
fusion weights; `offline_sweep_command()` emits the dev-locked sweep command; `swap_config()`
applies a sweep artifact between requests and **refuses rebuild knobs live** (the OtterTune
blacklist).

**Reserved (heavyweight, only if bandits + TPE plateau, per the PDF's own verdict):** GP-BO with
qNEHVI (BoTorch/torch), RL retrieval policies (R3-style), GPU TransE/IncDE training. Not built.

## Honest status

Every optimizer above is **validated for correctness** by offline unit tests (synthetic data,
no API key) — including known-answer tests for TPE/NSGA-II/ASHA/Lasso and recall tests for
quantization. **They are NOT validated for benchmark lift**: a live run is currently blocked by
DashScope quota (HTTP 403, free tier exhausted), and the methodology forbids tuning on test
items. So as shipped, the model's behavior is **byte-for-byte the prior baseline plus dormant,
flag-gated machinery**. Realizing measurable gains requires the runbook below.

## Runbook — turn the machinery into measured gains

```bash
# 0. Restore paid quota: disable "use free tier only" in the DashScope console (or add billing).

# 1. Baseline on the held-out TEST split (the reported number).
python -m bench.run --systems eidetic,mem0,graphiti --dataset both --runs 10 --split test

# 2. Tune on the private DEV split only (numpy TPE, multi-objective Pareto). Never touches test.
python -m bench.sweep --sampler tpe --dataset locomo --subset 50 --trials 24 --split dev
#    -> writes artifacts/bench/sweep/best_config.json (best_env + Pareto front)

# 3. Calibrate the conformal / abstention thresholds on DEV logs.
python -m bench.calibrate --method conformal --split dev   # -> CONFORMAL_QHAT
python -m bench.calibrate --method precision --split dev   # -> ABSTENTION_THRESHOLD

# 4. Apply the winning flags (from best_config.json) to .env, then RE-MEASURE on test.
python -m bench.run --systems eidetic --dataset both --runs 10 --split test

# Promote a config only if DEV accuracy holds and the chosen objective improves. Report every
# number with its token cost + latency + variance, and treat published targets as DIRECTION.
```

Thresholds that change the plan (from the PDF): semantic-cache false-positive rate > 0.5% ->
raise the cache threshold; conformal coverage drifts below 1-alpha on dev -> recalibrate; a
bandit arm dominates for > N rounds -> freeze it; quantization recall drop > 1% -> force the
refine pass on.
