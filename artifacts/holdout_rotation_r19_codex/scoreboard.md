# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | mem0 | rag-vector |
|---|---|---|---|
| single-hop (n=21) | 38.1±0.0 | 81.2±0.0 | 90.5±0.0 |
| multi-hop (n=8) | 12.5±0.0 | 33.3±0.0 | 50.0±0.0 |
| temporal (n=8) | 50.0±0.0 | 0.0±0.0 | 12.5±0.0 |
| open-domain (n=3) | 0.0±0.0 | - | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | mem0 | rag-vector |
|---|---|---|---|
| single-hop | 8/21, 20.8-59.1 | 13/16, 57.0-93.4 | 19/21, 71.1-97.3 |
| multi-hop | 1/8, 2.2-47.1 | 1/3, 6.1-79.2 | 4/8, 21.5-78.5 |
| temporal | 4/8, 21.5-78.5 | 0/7, 0.0-35.4 | 1/8, 2.2-47.1 |
| open-domain | 0/3, 0.0-56.1 | - | 3/3, 43.9-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs mem0 | locomo/multi-hop | 3 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/single-hop | 16 | 1 | 8 | 0.0391 | no | needs-2-runs |
| eidetic-plus-full vs mem0 | locomo/temporal | 7 | 3 | 0 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 8 | 0 | 3 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 3 | 0 | 3 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/single-hop | 21 | 0 | 11 | 0.0010 | rag-vector | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 8 | 4 | 1 | 0.3750 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/multi-hop | 3 | 0 | 0 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/single-hop | 16 | 1 | 2 | 1.0000 | no | needs-2-runs |
| mem0 vs rag-vector | locomo/temporal | 7 | 0 | 1 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 21604 | 4966 |
| mem0 | 21300 | 402 |
| rag-vector | 21604 | 1816 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 2158.5 | 2531.4 | 21388.6 | 61480.7 |
| mem0 | 709.9 | 839.7 | 2387.0 | 3000.4 |
| rag-vector | 0.9 | 971.1 | 2308.5 | 3413.0 |

## Consolidation Health

_Counts are logged once per ingested conversation/run. Timeouts mean the record stayed searchable as raw memory, but did not finish fact/event extraction within the configured sleep deadline._

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 272 | 828 | 828 | 0 | 0 | 272 | 272 | 0 | 0 | 0 | 0 | 0 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 32.5% | 0.0% | 37.5% |
| mem0 | 26 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
