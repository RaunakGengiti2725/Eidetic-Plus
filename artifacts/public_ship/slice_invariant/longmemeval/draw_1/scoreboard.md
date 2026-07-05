# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full |
|---|---|
| single-session-user (n=4) | 100.0±0.0 |
| single-session-assistant (n=4) | 25.0±0.0 |
| single-session-preference (n=4) | 0.0±0.0 |
| multi-session (n=4) | 25.0±0.0 |
| knowledge-update (n=4) | 75.0±0.0 |
| temporal-reasoning (n=4) | 50.0±0.0 |

## longmemeval - Wilson 95% CI by category

| category | eidetic-plus-full |
|---|---|
| single-session-user | 4/4, 51.0-100.0 |
| single-session-assistant | 1/4, 4.6-69.9 |
| single-session-preference | 0/4, 0.0-49.0 |
| multi-session | 1/4, 4.6-69.9 |
| knowledge-update | 3/4, 30.1-95.4 |
| temporal-reasoning | 2/4, 15.0-85.0 |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 123690 | 3113 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 4356.8 | 6307.7 | 15655.7 | 21108.0 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 24 | 915 | 0 | 0 | 0 | 0 | 3089 | 0 | 915 | 0 | 0 | 24 | 24 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 24 | 45.8% | 0.0% | 29.2% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
