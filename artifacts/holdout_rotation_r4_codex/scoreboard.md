# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 |
|---|---|---|
| single-hop (n=20) | 60.0±0.0 | 75.0±0.0 |
| multi-hop (n=8) | 62.5±0.0 | 50.0±0.0 |
| temporal (n=9) | 55.6±0.0 | 11.1±0.0 |
| open-domain (n=3) | 33.3±0.0 | 33.3±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 |
|---|---|---|
| single-hop | 12/20, 38.7-78.1 | 15/20, 53.1-88.8 |
| multi-hop | 5/8, 30.6-86.3 | 4/8, 21.5-78.5 |
| temporal | 5/9, 26.7-81.1 | 1/9, 2.0-43.5 |
| open-domain | 1/3, 6.1-79.2 | 1/3, 6.1-79.2 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 8 | 2 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 3 | 1 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 20 | 1 | 4 | 0.3750 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 9 | 4 | 0 | 0.1250 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 3535 |
| mem0 | 21604 | 390 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3079.0 | 5378.1 | 11066.4 | 16643.6 |
| mem0 | 798.9 | 1370.5 | 3991.1 | 6622.2 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 250 | 878 | 878 | 0 | 0 | 258 | 258 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 55.0% | 2.5% | 15.0% |
| mem0 | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
