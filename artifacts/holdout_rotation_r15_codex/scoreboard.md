# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | rag-vector |
|---|---|---|
| single-hop (n=21) | 76.2±0.0 | 85.7±0.0 |
| multi-hop (n=8) | 12.5±0.0 | 25.0±0.0 |
| temporal (n=9) | 44.4±0.0 | 22.2±0.0 |
| open-domain (n=2) | 50.0±0.0 | 50.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | rag-vector |
|---|---|---|
| single-hop | 16/21, 54.9-89.4 | 18/21, 65.4-95.0 |
| multi-hop | 1/8, 2.2-47.1 | 2/8, 7.1-59.1 |
| temporal | 4/9, 18.9-73.3 | 2/9, 6.3-54.7 |
| open-domain | 1/2, 9.5-90.5 | 1/2, 9.5-90.5 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 8 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/single-hop | 21 | 1 | 3 | 0.6250 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 9 | 4 | 2 | 0.6875 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 3856 |
| rag-vector | 21604 | 1839 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3345.3 | 6379.8 | 10025.7 | 22740.1 |
| rag-vector | 870.5 | 972.9 | 4300.1 | 8028.9 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 252 | 878 | 878 | 0 | 0 | 260 | 260 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 52.5% | 7.5% | 2.5% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
