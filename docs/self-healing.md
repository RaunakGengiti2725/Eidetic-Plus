# Self-healing memory + society-of-memory (Revolutionary Architectures)

Implements the highest-evidence, API-buildable mechanisms from *"Revolutionary Architectures for
LLM Agent Long-Term Memory."* Everything is behind config flags defaulting **off**, so the
shipped baseline is unchanged.

## The honest frame (read first)

This menu's value lives mostly in **LLM orchestration that cannot run or be measured here**
(MemMA probes, CoVe, debate rounds, recognition filtering, HaluMem QA grading) -- all real
DashScope calls, currently **403-blocked (free tier exhausted)**, and **no-mock** by discipline.
So a green offline test suite proves the **deterministic cores** are correct; it does **not**
prove a benchmark lift. As shipped, behavior is the prior baseline plus dormant machinery.

The PDF itself is explicit: **LoCoMo is near-saturated** (85–92%+), so a decisive win is more
credible on **LongMemEval** (knowledge-update, multi-session), **MemoryAgentBench** (conflict
resolution -- where all systems are weak), **HaluMem** (operation-level hallucination: update
accuracy <26%, QA <56%), and **BEAM** (1M/10M tokens). And: *no mechanism guarantees a benchmark
win before it is measured on the held-out test set.*

## What's built -- live vs scaffolding vs skipped

**Live (deterministic, offline-tested, flag-gated):**

| Mechanism | PDF | Module | Flag |
|---|---|---|---|
| EvolveMem auto-revert **Guard** (dev-proxy regression gate, paired McNemar) | 3e | `bench/guard.py` | `GUARD_ENABLED` |
| Per-triple **anomaly scoring** over observed edges (LOF + coherence + TransE) | 1d | `eidetic/dreaming/anomaly.py` | `ANOMALY_THRESHOLD` |
| Heuristic **memory manager** router (ADD/UPDATE/DELETE-tombstone/NOOP) | 3a-RL | `eidetic/dreaming/manager.py` | `MEMORY_MANAGER` |
| MemMA **repair router** (diagnose → SKIP/MERGE/INSERT) + anomaly targeting | 1a/1c | `eidetic/dreaming/repair.py` | `DREAM_REPAIR` |
| MIRIX **role typing** (6 types) + coordinator | 2c | `eidetic/memory_types.py` | `MEMORY_TYPING` |
| **Markov** prospective prefetch (P(next-signature\|current), wired into ask) | 3c | `eidetic/optim/markov.py` | `MARKOV_PREFETCH` |
| Bounded-**debate aggregation** guard (≥2 agree, else abstain) | 2a/2b | `eidetic/debate.py` | `DEBATE` |
| HaluMem **operation-level grading** (extraction recall / update acc / QA rates) | -- | `bench/halumem.py` | (bench) |

**Gated scaffolding (real LLM, fail-loud, OFF, NOT run under quota -- available to A/B, not yet a
proven feature):** MemMA probe generation + probe answering (the LLM half of `DREAM_REPAIR`,
proposal-only -- never auto-mutates), the memory-manager execution path (`MEMORY_MANAGER`),
`run_conflict_debate` rounds (`DEBATE`), Chain-of-Verification (`client.plan_verification_questions`,
`COVE`), MIRIX Active Retrieval (`client.generate_topic`, `ACTIVE_RETRIEVAL`), LLM role typing.
One unverified prompt is not a feature; these are honest integration points.

**Deliberately skipped (per the PDF's own verdict):** GPU **GRPO/PPO** training of a memory
manager (Memory-R1 proper) -- torch/GPU, out of an API-only stack; the heuristic manager above is
the recommended approximation. **Titans / HOPE / Nested Learning** -- paradigm-shifting but
unbenchmarked on these suites and torch/GPU-only; watch-list. Literal **artificial immune
systems / stigmergy / clonal selection** -- decades-old but with zero measured LLM-memory results;
used only as design vocabulary (danger signal = anomaly score; affinity maturation = FSRS).

## The integrity wall holds at every new entry point

- The **Guard** reads dev-split log dirs only and refuses an unpaired comparison (champion and
  challenger must score the SAME dev `sample_id`s, or McNemar is invalid).
- HaluMem op-level eval routes every memory point through `split_of` (dev/test disjoint).
- The MemMA sweep is **proposal-only** (never mutates the store) and writes to the derived/inferred
  layer; the memory manager never deletes a raw record (DELETE = a reversible tombstone).
- The dev-only `FeedbackBuffer` continues to exclude benchmark namespaces from any learner.

## Guard scope -- exactly what it guarantees (and doesn't)

The Guard **prevents a per-swap dev-proxy regression**: a tuned config is promoted only if it
beats the champion on the dev split by `GUARD_MIN_DELTA_PP` AND a paired McNemar test is
significant (`p < GUARD_ALPHA`). It does **not** guarantee a test-set improvement, and repeated
guarded swaps overfit dev (multiple comparisons). "Revert" = keep the champion artifact; there is
no hot-swap (frozen Settings -- applying a config means writing `.env` + restart). Always report
the final number on the held-out **test** split.

## Runbook -- turn the machinery into measured gains

```bash
# 0. Restore paid quota (disable "use free tier only" / add billing).
# 1. Baseline on the held-out TEST split.
python -m bench.run --systems eidetic --dataset both --runs 10 --split test
# 2. A/B a mechanism on the DEV split only (e.g. the MemMA repair sweep):
DREAM_REPAIR=1 python -m bench.run --systems eidetic --split dev --out artifacts/dev/repair_on
python -m bench.run --systems eidetic --split dev --out artifacts/dev/repair_off
# 3. Guard the change: promote only if it beats the champion significantly on dev.
python -m bench.guard --champion artifacts/dev/repair_off --challenger artifacts/dev/repair_on
# 4. Measure self-repair on HaluMem first (largest headroom) -- place the real export in
#    data/bench/halumem/, then grade extraction recall / update accuracy / QA rates.
# 5. Re-measure the winning config on the TEST split. Report with variance.
```
Stage-1 threshold (PDF): a measurable drop in HaluMem QA-hallucination and a LongMemEval
knowledge-update gain over the current baseline. Otherwise the change does not ship.
