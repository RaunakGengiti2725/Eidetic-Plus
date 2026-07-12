# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full |
|---|---|
| single-session-user (n=2) | 100.0±0.0 |
| single-session-assistant (n=2) | 100.0±0.0 |
| single-session-preference (n=2) | 50.0±0.0 |
| multi-session (n=2) | 100.0±0.0 |
| knowledge-update (n=2) | 50.0±0.0 |
| temporal-reasoning (n=2) | 0.0±0.0 |

## longmemeval - Wilson 95% CI by category

| category | eidetic-plus-full |
|---|---|
| single-session-user | 2/2, 34.2-100.0 |
| single-session-assistant | 2/2, 34.2-100.0 |
| single-session-preference | 1/2, 9.5-90.5 |
| multi-session | 2/2, 34.2-100.0 |
| knowledge-update | 1/2, 9.5-90.5 |
| temporal-reasoning | 0/2, 0.0-65.8 |

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full |
|---|---|
| single-hop (n=3) | 66.7±0.0 |
| multi-hop (n=3) | 100.0±0.0 |
| temporal (n=3) | 66.7±0.0 |
| open-domain (n=3) | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full |
|---|---|
| single-hop | 2/3, 20.8-93.9 |
| multi-hop | 3/3, 43.9-100.0 |
| temporal | 2/3, 20.8-93.9 |
| open-domain | 3/3, 43.9-100.0 |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 92129 | 4948 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 4138.9 | 5991.8 | 15472.8 | 24812.6 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 17 | 642 | 345 | 345 | 0 | 0 | 1773 | 117 | 525 | 0 | 0 | 12 | 12 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 24 | 75.0% | 0.0% | 12.5% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
