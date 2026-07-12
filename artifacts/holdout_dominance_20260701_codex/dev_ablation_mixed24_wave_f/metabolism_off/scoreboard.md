# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full |
|---|---|
| single-session-user (n=2) | 50.0±0.0 |
| single-session-assistant (n=2) | 50.0±0.0 |
| single-session-preference (n=2) | 50.0±0.0 |
| multi-session (n=2) | 50.0±0.0 |
| knowledge-update (n=2) | 50.0±0.0 |
| temporal-reasoning (n=2) | 50.0±0.0 |

## longmemeval - Wilson 95% CI by category

| category | eidetic-plus-full |
|---|---|
| single-session-user | 1/2, 9.5-90.5 |
| single-session-assistant | 1/2, 9.5-90.5 |
| single-session-preference | 1/2, 9.5-90.5 |
| multi-session | 1/2, 9.5-90.5 |
| knowledge-update | 1/2, 9.5-90.5 |
| temporal-reasoning | 1/2, 9.5-90.5 |

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full |
|---|---|
| single-hop (n=3) | 66.7±0.0 |
| multi-hop (n=3) | 66.7±0.0 |
| temporal (n=3) | 33.3±0.0 |
| open-domain (n=3) | 66.7±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full |
|---|---|
| single-hop | 2/3, 20.8-93.9 |
| multi-hop | 2/3, 20.8-93.9 |
| temporal | 1/3, 6.1-79.2 |
| open-domain | 2/3, 20.8-93.9 |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 92129 | 4648 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3326.4 | 4486.0 | 13547.0 | 20711.1 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 17 | 576 | 1691 | 1691 | 1 | 0 | 576 | 576 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 24 | 45.8% | 8.3% | 12.5% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
