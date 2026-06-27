# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | rag-full | rag-vector |
|---|---|---|
| single-session-user (n=1) | 0.0±0.0 | 100.0±0.0 |

## longmemeval - Wilson 95% CI by category

| category | rag-full | rag-vector |
|---|---|---|
| single-session-user | 0/1, 0.0-79.3 | 1/1, 20.7-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| rag-full vs rag-vector | longmemeval/single-session-user | 1 | 0 | 1 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| rag-full | 122650 | 122650 |
| rag-vector | 122650 | 1350 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| rag-full | 0.0 | 0.0 | 4713.4 | 4713.4 |
| rag-vector | 1063.6 | 1063.6 | 2478.1 | 2478.1 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| rag-full | 1 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 1 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
