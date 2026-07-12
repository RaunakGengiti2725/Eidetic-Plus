# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 |
|---|---|---|
| single-hop (n=22) | 72.7±0.0 | 59.1±0.0 |
| multi-hop (n=8) | 75.0±0.0 | 25.0±0.0 |
| temporal (n=8) | 25.0±0.0 | 25.0±0.0 |
| open-domain (n=2) | 0.0±0.0 | 0.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 |
|---|---|---|
| single-hop | 16/22, 51.8-86.8 | 13/22, 38.7-76.7 |
| multi-hop | 6/8, 40.9-92.9 | 2/8, 7.1-59.1 |
| temporal | 2/8, 7.1-59.1 | 2/8, 7.1-59.1 |
| open-domain | 0/2, 0.0-65.8 | 0/2, 0.0-65.8 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 8 | 4 | 0 | 0.1250 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/open-domain | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 22 | 7 | 4 | 0.5488 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 8 | 1 | 1 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 4428 |
| mem0 | 21604 | 386 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3152.6 | 6829.6 | 11701.1 | 21179.9 |
| mem0 | 838.8 | 1436.0 | 4075.3 | 7614.8 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 248 | 842 | 842 | 0 | 0 | 256 | 256 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 60.0% | 0.0% | 12.5% |
| mem0 | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
