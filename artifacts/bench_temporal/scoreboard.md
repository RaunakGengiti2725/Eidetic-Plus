# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full |
|---|---|
| multi-hop (n=15) | 73.3±0.0 |
| temporal (n=19) | 57.9±0.0 |
| open-domain (n=5) | 80.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full |
|---|---|
| multi-hop | 11/15, 48.0-89.1 |
| temporal | 11/19, 36.3-76.9 |
| open-domain | 4/5, 37.6-96.4 |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 15511 | 7934 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 2044.5 | 2420.0 | 11606.3 | 13873.3 |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
