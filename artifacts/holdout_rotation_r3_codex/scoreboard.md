# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 |
|---|---|---|
| single-hop (n=22) | 86.4±0.0 | 76.2±0.0 |
| multi-hop (n=8) | 50.0±0.0 | 87.5±0.0 |
| temporal (n=8) | 37.5±0.0 | 0.0±0.0 |
| open-domain (n=2) | 50.0±0.0 | 0.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 |
|---|---|---|
| single-hop | 19/22, 66.7-95.3 | 16/21, 54.9-89.4 |
| multi-hop | 4/8, 21.5-78.5 | 7/8, 52.9-97.8 |
| temporal | 3/8, 13.7-69.4 | 0/5, 0.0-43.4 |
| open-domain | 1/2, 9.5-90.5 | 0/1, 0.0-79.3 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 8 | 0 | 3 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 1 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 21 | 4 | 2 | 0.6875 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 5 | 2 | 0 | 0.5000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 3838 |
| mem0 | 22038 | 391 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3297.0 | 4956.3 | 10037.8 | 17757.2 |
| mem0 | 760.4 | 1333.3 | 3520.2 | 7552.5 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 250 | 877 | 877 | 0 | 0 | 258 | 258 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 65.0% | 2.5% | 2.5% |
| mem0 | 35 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
