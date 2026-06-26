# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | rag-full |
|---|---|
| multi-hop (n=3) | 100.0±0.0 |
| temporal (n=6) | 50.0±0.0 |
| open-domain (n=1) | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | rag-full |
|---|---|
| multi-hop | 3/3, 43.9-100.0 |
| temporal | 3/6, 18.8-81.2 |
| open-domain | 1/1, 20.7-100.0 |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| rag-full | 15511 | 15511 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| rag-full | 0.0 | 0.0 | 16513.2 | 78775.4 |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
