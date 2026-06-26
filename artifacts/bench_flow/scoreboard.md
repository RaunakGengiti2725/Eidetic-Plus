# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | rag-full | rag-vector |
|---|---|---|---|
| multi-hop (n=4) | 50.0±0.0 | 100.0±0.0 | 50.0±0.0 |
| temporal (n=7) | 57.1±0.0 | 42.9±0.0 | 42.9±0.0 |
| open-domain (n=1) | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | rag-full | rag-vector |
|---|---|---|---|
| multi-hop | 2/4, 15.0-85.0 | 4/4, 51.0-100.0 | 2/4, 15.0-85.0 |
| temporal | 4/7, 25.0-84.2 | 3/7, 15.8-75.0 | 3/7, 15.8-75.0 |
| open-domain | 1/1, 20.7-100.0 | 1/1, 20.7-100.0 | 1/1, 20.7-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs rag-full | locomo/multi-hop | 4 | 0 | 2 | 0.5000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/temporal | 7 | 2 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 4 | 1 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 7 | 3 | 2 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/multi-hop | 4 | 2 | 0 | 0.5000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/temporal | 7 | 1 | 1 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 15511 | 7932 |
| rag-full | 15511 | 15511 |
| rag-vector | 15511 | 1857 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 2999.2 | 3335.4 | 24712.1 | 26600.4 |
| rag-full | 0.0 | 0.0 | 3540.8 | 6213.0 |
| rag-vector | 0.4 | 0.6 | 2243.3 | 3361.2 |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
