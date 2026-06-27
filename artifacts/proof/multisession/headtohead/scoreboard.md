# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | rag-full | rag-vector |
|---|---|---|---|
| multi-session (n=6) | 33.3±0.0 | 16.7±0.0 | 33.3±0.0 |

## longmemeval - Wilson 95% CI by category

| category | eidetic-plus-full | rag-full | rag-vector |
|---|---|---|---|
| multi-session | 2/6, 9.7-70.0 | 1/6, 3.0-56.4 | 2/6, 9.7-70.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs rag-full | longmemeval/multi-session | 6 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/multi-session | 6 | 0 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | longmemeval/multi-session | 6 | 0 | 1 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 122752 | 7996 |
| rag-full | 122752 | 122752 |
| rag-vector | 122752 | 1944 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 2955.6 | 3289.9 | 11963.9 | 13872.4 |
| rag-full | 0.0 | 0.0 | 18824.2 | 23036.3 |
| rag-vector | 2.4 | 3.0 | 3340.8 | 4037.8 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 6 | 16.7% | 0.0% | 33.3% |
| rag-full | 6 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 6 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
