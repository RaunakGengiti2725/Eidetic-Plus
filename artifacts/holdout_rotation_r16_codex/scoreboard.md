# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 | rag-vector |
|---|---|---|---|
| single-hop (n=38) | 85.7±0.0 | 92.1±0.0 | 90.5±0.0 |
| multi-hop (n=13) | 37.5±0.0 | 46.2±0.0 | 37.5±0.0 |
| temporal (n=10) | 25.0±0.0 | 20.0±0.0 | 0.0±0.0 |
| open-domain (n=6) | 66.7±0.0 | 50.0±0.0 | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 | rag-vector |
|---|---|---|---|
| single-hop | 18/21, 65.4-95.0 | 35/38, 79.2-97.3 | 19/21, 71.1-97.3 |
| multi-hop | 3/8, 13.7-69.4 | 6/13, 23.2-70.9 | 3/8, 13.7-69.4 |
| temporal | 2/8, 7.1-59.1 | 2/10, 5.7-51.0 | 0/8, 0.0-32.4 |
| open-domain | 2/3, 20.8-93.9 | 3/6, 18.8-81.2 | 3/3, 43.9-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 7 | 2 | 2 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 3 | 1 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 21 | 1 | 2 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 7 | 2 | 2 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 8 | 2 | 2 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 3 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/single-hop | 21 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 8 | 2 | 0 | 0.5000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/multi-hop | 7 | 1 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/open-domain | 3 | 0 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/single-hop | 21 | 1 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/temporal | 7 | 2 | 0 | 0.5000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 5914 |
| mem0 | 22038 | 390 |
| rag-vector | 21604 | 1842 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 2319.9 | 4171.9 | 20806.8 | 24594.4 |
| mem0 | 680.4 | 1151.5 | 2719.3 | 4479.5 |
| rag-vector | 0.5 | 893.2 | 2561.4 | 4260.3 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 272 | 771 | 771 | 0 | 0 | 272 | 272 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 37.5% | 35.0% | 2.5% |
| mem0 | 67 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
