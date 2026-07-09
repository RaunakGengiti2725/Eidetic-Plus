# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## longmemeval - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | rag-vector |
|---|---|---|
| single-session-user (n=4) | 75.0±0.0 | 75.0±0.0 |
| single-session-assistant (n=4) | 75.0±0.0 | 100.0±0.0 |
| single-session-preference (n=2) | 50.0±0.0 | 50.0±0.0 |
| multi-session (n=7) | 28.6±0.0 | 57.1±0.0 |
| knowledge-update (n=5) | 60.0±0.0 | 100.0±0.0 |
| temporal-reasoning (n=8) | 50.0±0.0 | 25.0±0.0 |

## longmemeval - Wilson 95% CI by category

| category | eidetic-plus-full | rag-vector |
|---|---|---|
| single-session-user | 3/4, 30.1-95.4 | 3/4, 30.1-95.4 |
| single-session-assistant | 3/4, 30.1-95.4 | 4/4, 51.0-100.0 |
| single-session-preference | 1/2, 9.5-90.5 | 1/2, 9.5-90.5 |
| multi-session | 2/7, 8.2-64.1 | 4/7, 25.0-84.2 |
| knowledge-update | 3/5, 23.1-88.2 | 5/5, 56.6-100.0 |
| temporal-reasoning | 4/8, 21.5-78.5 | 2/8, 7.1-59.1 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs rag-vector | longmemeval/knowledge-update | 5 | 0 | 2 | 0.5000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/multi-session | 7 | 0 | 2 | 0.5000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/single-session-assistant | 4 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/single-session-preference | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/single-session-user | 4 | 1 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | longmemeval/temporal-reasoning | 8 | 2 | 0 | 0.5000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 123604 | 4057 |
| rag-vector | 123604 | 1915 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3297.7 | 6651.3 | 10194.3 | 24333.1 |
| rag-vector | 878.4 | 1088.9 | 5334.4 | 7545.7 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 30 | 1245 | 0 | 0 | 0 | 0 | 4036 | 0 | 1245 | 0 | 0 | 30 | 30 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 30 | 43.3% | 10.0% | 13.3% |
| rag-vector | 30 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
