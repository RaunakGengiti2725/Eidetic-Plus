# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | eidetic-product | rag-full | rag-vector |
|---|---|---|---|---|
| multi-hop (n=15) | 73.3±0.0 | 38.5±0.0 | 73.3±0.0 | 60.0±0.0 |
| temporal (n=20) | 75.0±0.0 | 60.0±0.0 | 40.0±0.0 | 25.0±0.0 |
| open-domain (n=5) | 50.0±0.0 | 80.0±0.0 | 80.0±0.0 | 60.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | eidetic-product | rag-full | rag-vector |
|---|---|---|---|---|
| multi-hop | 11/15, 48.0-89.1 | 5/13, 17.7-64.5 | 11/15, 48.0-89.1 | 9/15, 35.7-80.2 |
| temporal | 15/20, 53.1-88.8 | 12/20, 38.7-78.1 | 8/20, 21.9-61.3 | 5/20, 11.2-46.9 |
| open-domain | 2/4, 15.0-85.0 | 4/5, 37.6-96.4 | 4/5, 37.6-96.4 | 3/5, 23.1-88.2 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs eidetic-product | locomo/multi-hop | 13 | 5 | 1 | 0.2188 | no | needs-2-runs |
| eidetic-plus-full vs eidetic-product | locomo/open-domain | 4 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs eidetic-product | locomo/temporal | 20 | 3 | 0 | 0.2500 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/multi-hop | 15 | 1 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/open-domain | 4 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/temporal | 20 | 8 | 1 | 0.0391 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 15 | 4 | 2 | 0.6875 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 4 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 20 | 11 | 1 | 0.0063 | eidetic-plus-full | needs-2-runs |
| eidetic-product vs rag-full | locomo/multi-hop | 13 | 0 | 4 | 0.1250 | no | needs-2-runs |
| eidetic-product vs rag-full | locomo/open-domain | 5 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-full | locomo/temporal | 20 | 6 | 2 | 0.2891 | no | needs-2-runs |
| eidetic-product vs rag-vector | locomo/multi-hop | 13 | 2 | 4 | 0.6875 | no | needs-2-runs |
| eidetic-product vs rag-vector | locomo/open-domain | 5 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-vector | locomo/temporal | 20 | 9 | 2 | 0.0654 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/multi-hop | 15 | 3 | 1 | 0.6250 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/open-domain | 5 | 1 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/temporal | 20 | 3 | 0 | 0.2500 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 15511 | 3476 |
| eidetic-product | 15511 | 3331 |
| rag-full | 15511 | 15511 |
| rag-vector | 15511 | 1882 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3264.7 | 3799.9 | 10739.5 | 16835.7 |
| eidetic-product | 2431.6 | 3438.9 | 46387.4 | 79665.1 |
| rag-full | 0.0 | 0.0 | 5273.7 | 14529.9 |
| rag-vector | 0.6 | 1.5 | 4140.9 | 8907.1 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 39 | 53.8% | 30.8% | 2.6% |
| eidetic-product | 38 | 21.1% | 55.3% | 7.9% |
| rag-full | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
