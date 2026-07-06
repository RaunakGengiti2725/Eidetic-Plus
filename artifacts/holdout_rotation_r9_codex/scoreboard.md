# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 | rag-full | rag-vector |
|---|---|---|---|---|
| single-hop (n=21) | 66.7±0.0 | 43.8±0.0 | 85.7±0.0 | 71.4±0.0 |
| multi-hop (n=8) | 12.5±0.0 | 50.0±0.0 | 75.0±0.0 | 62.5±0.0 |
| temporal (n=8) | 37.5±0.0 | 0.0±0.0 | 0.0±0.0 | 12.5±0.0 |
| open-domain (n=3) | 66.7±0.0 | 66.7±0.0 | 66.7±0.0 | 33.3±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 | rag-full | rag-vector |
|---|---|---|---|---|
| single-hop | 14/21, 45.4-82.8 | 7/16, 23.1-66.8 | 18/21, 65.4-95.0 | 15/21, 50.0-86.2 |
| multi-hop | 1/8, 2.2-47.1 | 4/8, 21.5-78.5 | 6/8, 40.9-92.9 | 5/8, 30.6-86.3 |
| temporal | 3/8, 13.7-69.4 | 0/7, 0.0-35.4 | 0/8, 0.0-32.4 | 1/8, 2.2-47.1 |
| open-domain | 2/3, 20.8-93.9 | 2/3, 20.8-93.9 | 2/3, 20.8-93.9 | 1/3, 6.1-79.2 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 8 | 0 | 3 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 3 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 16 | 4 | 0 | 0.1250 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 7 | 3 | 0 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/multi-hop | 8 | 0 | 5 | 0.0625 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/open-domain | 3 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/single-hop | 21 | 0 | 4 | 0.1250 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/temporal | 8 | 3 | 0 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 8 | 0 | 4 | 0.1250 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 3 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/single-hop | 21 | 2 | 3 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 8 | 3 | 1 | 0.6250 | no | needs-2-runs |
| mem0 vs rag-full | locomo/multi-hop | 8 | 0 | 2 | 0.5000 | no | needs-2-runs |
| mem0 vs rag-full | locomo/open-domain | 3 | 0 | 0 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-full | locomo/single-hop | 16 | 0 | 8 | 0.0078 | no | needs-2-runs |
| mem0 vs rag-full | locomo/temporal | 7 | 0 | 0 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/multi-hop | 8 | 1 | 2 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/open-domain | 3 | 1 | 0 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/single-hop | 16 | 0 | 7 | 0.0156 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/temporal | 7 | 0 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/multi-hop | 8 | 2 | 1 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/open-domain | 3 | 1 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/single-hop | 21 | 3 | 0 | 0.2500 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/temporal | 8 | 0 | 1 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 22578 | 3177 |
| mem0 | 23134 | 381 |
| rag-full | 22578 | 22503 |
| rag-vector | 22578 | 1837 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3165.9 | 4723.8 | 11055.4 | 16865.0 |
| mem0 | 747.4 | 1131.8 | 3391.3 | 6677.4 |
| rag-full | 0.0 | 0.1 | 5153.5 | 8771.7 |
| rag-vector | 727.8 | 812.3 | 4589.7 | 8283.8 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 9 | 242 | 1042 | 1042 | 0 | 0 | 250 | 250 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 47.5% | 2.5% | 10.0% |
| mem0 | 34 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-full | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
