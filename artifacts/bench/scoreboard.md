# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs._

## locomo — accuracy by category (%)

| category | eidetic-plus | graphiti | mem0 |
|---|---|---|---|
| multi-hop | 66.7±0.0 | 33.3±0.0 | 100.0±0.0 |
| temporal | 66.7±0.0 | 16.7±0.0 | 0.0±0.0 |
| open-domain | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus | 15511 | 7903 |
| graphiti | 15511 | 144 |
| mem0 | 15511 | 488 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus | 2266.8 | 2546.0 | 5330.1 | 7849.5 |
| graphiti | 756.0 | 1198.5 | 2622.3 | 3232.7 |
| mem0 | 338.6 | 747.5 | 2169.7 | 3580.0 |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
