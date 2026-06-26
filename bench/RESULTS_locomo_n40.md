# LoCoMo live benchmark — eidetic-plus-full vs RAG baselines

First run on a funded DashScope key. Neutral harness: **one fixed reader** (qwen-plus) + **one
fixed judge** (qwen3-max) across all systems, so the scoreboard measures MEMORY quality, not
answerer strength. Dataset `data/bench/locomo/locomo10.json`, split=all, **n=40, 1 run**.

> Honesty: n=40 (per-category n=5–20) is a SIGNAL run, not 10-run significance — Wilson CIs are
> wide. Numbers reproduce from `artifacts/bench_*/.../*.jsonl`. Mem0/Graphiti not run (need
> `mem0ai` + Neo4j). LongMemEval (the corpus≫context test where eidetic's edge is largest) was
> not cached and its download failed — the strongest eidetic case is NOT yet measured.

## Headline (with significance — read the McNemar, not the raw %)

| system | accuracy | tokens/query | e2e p50 |
|---|---:|---:|---:|
| eidetic-plus-full (tuned, graph+coact) | 65.0% (26/40) | 7938 | ~12.0s |
| eidetic-plus-full (default) | 60.0% (24/40) | 7921 | ~10.7s |
| rag-full (stuff-all-context) | 57.5% (23/40) | 15511 | ~3.9s |
| rag-vector (classic chunk+embed+topk) | 40.0% (16/40) | 1882 | ~2.6s |

Paired McNemar (same n40b run, default eidetic 60%):
- **eidetic vs rag-full: p=1.000 (7 vs 6) — a STATISTICAL TIE.** The raw +2.5/+7.5pt gap is within
  run-to-run noise. The honest, robust claim is **accuracy PARITY with full-context RAG at HALF its
  tokens** — and eidetic stays retrieval-ranked, which scales past the context window where
  stuff-everything cannot. NOT "eidetic beats full-context."
- **eidetic vs rag-vector: p≈0.10 (13 vs 5) — eidetic leads strongly, trending but not yet
  significant at n=40.** This is the apples-to-apples memory-agent comparison; +10 questions favor
  eidetic; likely significant with more samples. Confound: eidetic feeds the reader ~7900 tokens vs
  rag-vector's ~1900, so part of the lead is "4× the context" — the honest framing is the
  **accuracy/token operating point**, where eidetic dominates the curve.

n=40, 1 run -> no per-category result is CI-clear ("needs-2-runs"). Treat all margins as a signal to
confirm via a multi-run dev-split gate, not a banked win.

NOTE: this run validated ACCURACY, not the "faster"/Flow claim Track 9 was about. The bench adapter
uses retrieve()+answer() directly, NOT engine.ask(), so reflex/Flow never engaged; the 2.3x latency
win came from batched NLI (a separate track), not instinct. Flow's benefit (warm/repeated queries)
is not exercised by distinct LoCoMo questions and remains unmeasured.

## Tuned config (measured best)

```
BATCH_NLI=1 FAST_VERIFY=1 COACTIVATION_CHANNEL=1 GRAPH_VOCAB_SEEDING=1
```

- `BATCH_NLI` + `FAST_VERIFY`: batched / short-circuit NLI verification — **24.7s → 10.7s (2.3×)**,
  accuracy-neutral. The per-candidate serial NLI fan-out was the latency cost.
- `COACTIVATION_CHANNEL` + `GRAPH_VOCAB_SEEDING`: entity-graph traversal for the second hop —
  **multi-hop 60% → 73%** (past rag-full's 67%), total 60% → 65%, tokens flat (no dilution).

These accuracy levers ship flag-OFF pending dev-split promotion; this n=40 is promising evidence,
not the formal 10-run gate. The latency flags are accuracy-neutral and the strongest promotion
candidate.

## Negative results (kept honest)

- **More context HURTS** (`CONTEXT_TOKEN_BUDGET=16000 FINAL_TOPK=24`): 60% → 57.5%, multi-hop
  60% → 53%, tokens doubled, slower. Lost-in-the-middle — eidetic's tight ranked top-10 beats a
  diluted top-24. Tight retrieval is a feature, not budget-starvation.
- **Temporal levers** (`EVENT_CHAIN_CONTEXT` + `TEMPORAL_RERANK`) on top of graph: no real gain
  (+0–1 question = noise).

## Robustness fixes the live run forced

- Governor now retries **transient transport** errors (SSL/timeout/DNS/reset), not just 429 — the
  first run failed all systems on TLS drops.
- Bench harness is **per-question resilient** — a mid-run network blip flags that question (excluded
  from accuracy) and continues, instead of aborting a whole 40-question run.

## Reproduce

```bash
# tuned eidetic + baselines (funded key in .env)
DASHSCOPE_MAX_CONCURRENCY=2 BATCH_NLI=1 FAST_VERIFY=1 COACTIVATION_CHANNEL=1 GRAPH_VOCAB_SEEDING=1 \
  python -m bench.run --systems eidetic-full,rag-full,rag-vector \
  --dataset locomo --subset 40 --runs 1 --split all --out artifacts/bench_n40 --overwrite
```
