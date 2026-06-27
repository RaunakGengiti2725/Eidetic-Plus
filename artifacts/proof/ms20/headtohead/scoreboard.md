# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 | rag-full | rag-vector |
|---|---|---|---|---|
| multi-session (n=20) | 30.0±0.0 | 20.0±0.0 | 10.5±0.0 | 15.0±0.0 |

## longmemeval - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 | rag-full | rag-vector |
|---|---|---|---|---|
| multi-session | 6/20, 14.5-51.9 | 4/20, 8.1-41.6 | 2/19, 2.9-31.4 | 3/20, 5.2-36.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | longmemeval/multi-session | 20 | 4 | 2 | 0.6875 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | longmemeval/multi-session | 19 | 5 | 1 | 0.2188 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/multi-session | 20 | 4 | 1 | 0.3750 | no | needs-2-runs |
| mem0 vs rag-full | longmemeval/multi-session | 19 | 4 | 2 | 0.6875 | no | needs-2-runs |
| mem0 vs rag-vector | longmemeval/multi-session | 20 | 2 | 1 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | longmemeval/multi-session | 19 | 2 | 3 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 123201 | 7995 |
| mem0 | 118106 | 472 |
| rag-full | 123085 | 123085 |
| rag-vector | 123201 | 1937 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 2740.0 | 3343.4 | 11428.2 | 13369.0 |
| mem0 | 366.7 | 1010.8 | 6073.1 | 9721.0 |
| rag-full | 0.0 | 0.0 | 11326.7 | 21496.5 |
| rag-vector | 1.7 | 3.1 | 4180.5 | 8907.0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 20 | 20.0% | 5.0% | 35.0% |
| mem0 | 20 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-full | 19 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 20 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
