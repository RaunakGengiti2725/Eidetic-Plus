# Benchmark-Dominance Plan -- Implementation Progress

## UPDATE (forgetting-machine model): key live, proof program RUNNING

The DashScope key is now LIVE with quota, so the measurement program is no longer blocked. The
forgetting-machine slice (`scripts/proof_slice.sh`) runs the head-to-head + attribution ablation on
LongMemEval-S through one shared reader (`qwen-plus`) + judge (`qwen3-max`). First DIRECTIONAL result
(multi-session, n=6, 1 run -- NOT significant, every McNemar p=1.0; see
`bench/RESULTS_metabolism_multisession_n6.md`):
- eidetic **33.3%** vs rag-full **16.7%** at **7,996 vs 122,752 tokens/query** (~15x cheaper) and
  lower latency; ties rag-vector 33.3%.
- Attribution: metabolism memory OFF (reader+proof fixed) drops 33.3 -> 16.7pp -- the long-horizon
  gain is attributable to the memory layer (1 question flip, predicted direction).
- Snap-back fidelity 285/285 = 100% over the run's corpus.
Three benchmark-blocking robustness bugs were found+fixed to make the run complete at all (transient
5xx retry, extraction-JSON truncation salvage, content-moderation-400 skip); the mem0 adapter now
skips content-specific 4xx sessions for a fair row. Off-suite **567 passed**. A larger n=20
multi-session run is in progress. Honesty bind unchanged (`docs/claims.md`): directional only; the
significance gate is `bench/reproduce.sh` (multi-run); NO SOTA claim.

---

**Original status (pre-key): code shipped, proof pending.** Every code deliverable below is landed,
default-OFF, and offline-unit-tested (full suite **520 passed** at the time of writing). The
*measurement* program (live runs, sweeps, calibration, significance) was **DashScope-quota-blocked**
and unrun. **No accuracy numbers are claimed in this section** -- a number that does not reproduce
does not exist. The plan's target figures (≥75% LME, +10pp, "best memory agent") are *gated on the
measurement program*, not assertable from that session.

Branch: `connected-brain-loop`. All changes preserve the flag-off invariant: with every new flag
at its default, the neutral bench write/read path is byte-identical to the prior runs.

---

## What shipped (code, default-off, tested)

| Plan item | Where | Flag (default) | Path it fires on | Test |
|---|---|---|---|---|
| Chunked extraction (capture beyond char 6000) | `dashscope_client.extract_edges` | `EXTRACT_CHUNKING=0` | write/consolidate | `test_capture_fidelity` |
| Memory typing on the async write path | `engine.consolidate_pending` | `MEMORY_TYPING=0` | write/consolidate | (suite) |
| Preference sentence scan (all, not first) | `preferences.extract_all_preferences` + engine | `PREF_SENTENCE_SCAN=0` | write/consolidate | `test_capture_fidelity` |
| Reader per-block char cap | `bench/reader.py` | `READER_BLOCK_CHARS=3000` | shared reader (all systems) | `test_bench_plumbing` |
| Ingest granularity session/turn/hybrid | `bench/adapters/eidetic_adapter` | `INGEST_GRANULARITY=session` | write | `test_bench_plumbing` |
| Full lifecycle sleep (dream+gist available) | adapter `consolidate` | `FULL_SLEEP=0` | consolidate | `test_bench_plumbing` |
| `eidetic-product` bench row (engine.ask path) | adapter + `bench/run.py` | n/a (new row) | product | `test_bench_plumbing` |
| Wire **ACTIVE_RETRIEVAL** (was dead) | `retrieval.retrieve` | `ACTIVE_RETRIEVAL=0` | **all eidetic rows** (embed query) | `test_retrieval_wiring` |
| Wire **COVE** (was dead) | `retrieval.answer` | `COVE=0` | **product row only** (engine.ask) | `test_retrieval_wiring` |
| **SPAN_NLI** per-claim verification (new) | `retrieval.answer` | `SPAN_NLI=0` | **product row only** | `test_retrieval_wiring` |
| Photographic / extractive reader prompt | `bench/judge.py` + `reader.py` | `READER_MODE=default` | shared reader (all systems) | `test_bench_plumbing` |
| Scoreboard integrity row | `bench/scoreboard.py` | always (render) | reporting | `test_integrity_metrics` |
| Photographic prove/get_raw demo | `scripts/photographic_demo.py` | n/a (script) | demo | compile-checked |

**Precision on which path each verification flag touches** (the neutral rows answer via
`answer_with_fixed_reader` + `_verify_candidates`, NOT `retriever.answer()`):
- `ACTIVE_RETRIEVAL` is on the shared `retrieve()`, so it scaffolds the embed query for **every**
  eidetic row when enabled.
- `COVE` and `SPAN_NLI` live in `retrieval.answer()`, which only the **`eidetic-product`** row
  (engine.ask) calls. They are **inert in the two neutral rows**. Report them as such.

## Measurement foundation (landed, no quota needed)

- **LongMemEval-S cached + verified**: `data/bench/longmemeval/longmemeval_s.json` (the
  HF `..._cleaned.json`, saved to the loader's expected name). Loader returns **500 samples**,
  category counts exact (single-session-user 70, -assistant 56, -preference 30, multi-session 133,
  knowledge-update 78, temporal-reasoning 133). The prior "download failed" was a wrong filename
  (`longmemeval_s.json` vs `longmemeval_s_cleaned.json`), not quota.
- **Baselines installed**: `mem0ai==2.0.7`, `graphiti-core==0.29.2` import cleanly. Graphiti still
  needs a running Neo4j to *run*.

---

## What is NOT done (blocked on a funded DashScope key)

Everything in this list requires live model calls and is unrun. Do **not** report any of these as
results until executed on `--split test` with the significance gate:

- LongMemEval dev50 architectural proof; full LoCoMo; `--split test`.
- `bench.sweep` coordinate descent; `bench.calibrate` (abstention τ, conformal qhat).
- Temporal bundle / dream+gist ablations; INGEST_GRANULARITY ablation.
- Mem0 / Graphiti head-to-head runs.
- `eidetic-product` ceiling run; 10-run variance + McNemar (`bench/reproduce.sh`).
- Promoting any proven flag to a code default (the plan's `promote-graph-defaults`) -- only after a
  dev-gate win. Defaults are intentionally unchanged this session.

### What the existing n40 logs already show (re-rendered, not a new run)

Re-rendering the integrity row over the **existing** `artifacts/bench_n40b` logs (no new calls):
the verifying row carries entailment proofs while the RAG baselines have **no verify step at all**
(100% of their answers are unproven). It also honestly shows the verifying row's **own**
unverified-emit gap -- exactly the integrity hole COVE/SPAN_NLI/abstention-v2 target. The point of
the row is to make that gap visible and drive it down; it is not yet driven down (needs runs).

---

## Runbook (execute when a funded key is available)

```bash
export DASHSCOPE_MAX_CONCURRENCY=2 DASHSCOPE_RPM=30
export BATCH_NLI=1 FAST_VERIFY=1 COACTIVATION_CHANNEL=1 GRAPH_VOCAB_SEEDING=1

# 0. Smoke (cheap) -- confirm the pipeline end-to-end on 5 questions.
python -m bench.run --systems eidetic-full --dataset locomo --subset 5 --split dev \
  --out artifacts/smoke --overwrite

# 1. LongMemEval dev50 architectural proof (gate: eidetic-full >= rag-full + 10pp).
python -m bench.run --systems eidetic-full,rag-full,rag-vector \
  --dataset longmemeval --subset 50 --runs 1 --split dev \
  --out artifacts/bench_lme_dev50 --overwrite

# 2. Capture-fidelity stack on dev (the flags shipped this session).
EXTRACT_CHUNKING=1 MEMORY_TYPING=1 PREF_SENTENCE_SCAN=1 FULL_SLEEP=1 GIST_CHANNEL=1 \
READER_BLOCK_CHARS=8000 \
  python -m bench.run --systems eidetic-full --dataset locomo --subset 50 --split dev \
  --out artifacts/bench_dev50_recipe1 --overwrite

# 3. Wire-dead-paths ablation (now live): active retrieval + CoVe + photographic + span.
READER_MODE=photographic ACTIVE_RETRIEVAL=1 \
  python -m bench.run --systems eidetic-product --dataset locomo --subset 50 --split dev \
  --out artifacts/bench_dev50_photo --overwrite
#   (COVE=1 SPAN_NLI=1 only affect eidetic-product / engine.ask.)

# 4. Baselines (Mem0 needs OpenAI-compatible base_url -> DashScope; Graphiti needs Neo4j).
python -m bench.run --systems eidetic-full,rag-full,rag-vector,mem0 \
  --dataset locomo --subset 40 --split all --out artifacts/bench_n40_mem0 --overwrite

# 5. Significance gate (public-claim wall): test split, 10 runs, McNemar.
bash bench/reproduce.sh
```

**Promotion rule:** a flag becomes a default ONLY after `GUARD_ENABLED=1 GUARD_MIN_DELTA_PP=1.0
GUARD_ALPHA=0.05` shows a significant dev win. Public claims ONLY from `--split test` with
McNemar p<0.05. Update this file with CIs when runs land; do not edit target numbers in as if real.
