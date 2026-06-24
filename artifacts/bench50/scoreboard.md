# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus | mem0 |
|---|---|---|
| multi-hop (n=19) | 63.2±0.0 | 47.4±0.0 |
| temporal (n=24) | 16.7±0.0 | 8.3±0.0 |
| open-domain (n=7) | 57.1±0.0 | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus | mem0 |
|---|---|---|
| multi-hop | 12/19, 41.0-80.9 | 9/19, 27.3-68.3 |
| temporal | 4/24, 6.7-35.9 | 2/24, 2.3-25.8 |
| open-domain | 4/7, 25.0-84.2 | 7/7, 64.6-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus vs mem0 | locomo/multi-hop | 19 | 6 | 3 | 0.5078 | no | needs-2-runs |
| eidetic-plus vs mem0 | locomo/open-domain | 7 | 0 | 3 | 0.2500 | no | needs-2-runs |
| eidetic-plus vs mem0 | locomo/temporal | 24 | 4 | 2 | 0.6875 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus | 15511 | 7961 |
| mem0 | 15511 | 498 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus | 2036.5 | 2386.8 | 3740.1 | 4728.6 |
| mem0 | 326.5 | 506.7 | 1564.8 | 2142.0 |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
