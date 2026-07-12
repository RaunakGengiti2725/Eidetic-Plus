# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 | rag-vector |
|---|---|---|---|
| single-hop (n=21) | 57.1±0.0 | 76.2±0.0 | 90.5±0.0 |
| multi-hop (n=8) | 25.0±0.0 | 37.5±0.0 | 50.0±0.0 |
| temporal (n=8) | 50.0±0.0 | 12.5±0.0 | 12.5±0.0 |
| open-domain (n=3) | 0.0±0.0 | 33.3±0.0 | 66.7±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 | rag-vector |
|---|---|---|---|
| single-hop | 12/21, 36.5-75.5 | 16/21, 54.9-89.4 | 19/21, 71.1-97.3 |
| multi-hop | 2/8, 7.1-59.1 | 3/8, 13.7-69.4 | 4/8, 21.5-78.5 |
| temporal | 4/8, 21.5-78.5 | 1/8, 2.2-47.1 | 1/8, 2.2-47.1 |
| open-domain | 0/3, 0.0-56.1 | 1/3, 6.1-79.2 | 2/3, 20.8-93.9 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 8 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 3 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 21 | 2 | 6 | 0.2891 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 8 | 3 | 0 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 8 | 0 | 2 | 0.5000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 3 | 0 | 2 | 0.5000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/single-hop | 21 | 0 | 7 | 0.0156 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 8 | 3 | 0 | 0.2500 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/multi-hop | 8 | 0 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/open-domain | 3 | 0 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/single-hop | 21 | 0 | 3 | 0.2500 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/temporal | 8 | 0 | 0 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21284 | 4516 |
| mem0 | 21284 | 392 |
| rag-vector | 21284 | 1858 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3289.0 | 3863.6 | 11665.4 | 22064.4 |
| mem0 | 1023.1 | 1502.0 | 4265.0 | 7820.1 |
| rag-vector | 855.3 | 1054.7 | 4553.2 | 8387.2 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 9 | 217 | 749 | 749 | 0 | 0 | 225 | 225 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 45.0% | 0.0% | 30.0% |
| mem0 | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
