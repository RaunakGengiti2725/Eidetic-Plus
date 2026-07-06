# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 | rag-full | rag-vector |
|---|---|---|---|---|
| single-hop (n=22) | 63.6±0.0 | 55.0±0.0 | 86.4±0.0 | 86.4±0.0 |
| multi-hop (n=8) | 25.0±0.0 | 42.9±0.0 | 87.5±0.0 | 62.5±0.0 |
| temporal (n=8) | 25.0±0.0 | 0.0±0.0 | 12.5±0.0 | 12.5±0.0 |
| open-domain (n=2) | 50.0±0.0 | 0.0±0.0 | 50.0±0.0 | 0.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 | rag-full | rag-vector |
|---|---|---|---|---|
| single-hop | 14/22, 43.0-80.3 | 11/20, 34.2-74.2 | 19/22, 66.7-95.3 | 19/22, 66.7-95.3 |
| multi-hop | 2/8, 7.1-59.1 | 3/7, 15.8-75.0 | 7/8, 52.9-97.8 | 5/8, 30.6-86.3 |
| temporal | 2/8, 7.1-59.1 | 0/8, 0.0-32.4 | 1/8, 2.2-47.1 | 1/8, 2.2-47.1 |
| open-domain | 1/2, 9.5-90.5 | 0/2, 0.0-65.8 | 1/2, 9.5-90.5 | 0/2, 0.0-65.8 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 7 | 1 | 2 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 20 | 5 | 4 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 8 | 2 | 0 | 0.5000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/multi-hop | 8 | 0 | 5 | 0.0625 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/open-domain | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/single-hop | 22 | 2 | 7 | 0.1797 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/temporal | 8 | 2 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 8 | 0 | 3 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/single-hop | 22 | 2 | 7 | 0.1797 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 8 | 2 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-full | locomo/multi-hop | 7 | 0 | 3 | 0.2500 | no | needs-2-runs |
| mem0 vs rag-full | locomo/open-domain | 2 | 0 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-full | locomo/single-hop | 20 | 1 | 7 | 0.0703 | no | needs-2-runs |
| mem0 vs rag-full | locomo/temporal | 8 | 0 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/multi-hop | 7 | 0 | 1 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/open-domain | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/single-hop | 20 | 1 | 7 | 0.0703 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/temporal | 8 | 0 | 1 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/multi-hop | 8 | 2 | 0 | 0.5000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/open-domain | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/single-hop | 22 | 1 | 1 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/temporal | 8 | 0 | 0 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 3441 |
| mem0 | 22464 | 402 |
| rag-full | 21604 | 21435 |
| rag-vector | 21604 | 1837 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3184.4 | 9549.0 | 12558.0 | 20774.2 |
| mem0 | 740.8 | 1293.1 | 3424.4 | 6842.2 |
| rag-full | 0.0 | 0.0 | 4928.0 | 10369.7 |
| rag-vector | 733.3 | 771.9 | 3937.3 | 9261.5 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 257 | 1183 | 1183 | 0 | 0 | 265 | 265 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 42.5% | 10.0% | 17.5% |
| mem0 | 37 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-full | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
