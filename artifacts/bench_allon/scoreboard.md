# Eidetic-Plus benchmark scoreboard

_Judge: **qwen3-max** (dashscope), one fixed judge + one fixed reader across all systems. Per-category accuracy = mean±std over runs; CI = Wilson 95% interval over logged questions._

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | eidetic-product | rag-full | rag-vector |
|---|---|---|---|---|
| multi-hop (n=2) | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 50.0±0.0 |
| temporal (n=2) | 100.0±0.0 | 50.0±0.0 | 50.0±0.0 | 50.0±0.0 |
| open-domain (n=1) | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | eidetic-product | rag-full | rag-vector |
|---|---|---|---|---|
| multi-hop | 2/2, 34.2-100.0 | 1/1, 20.7-100.0 | 2/2, 34.2-100.0 | 1/2, 9.5-90.5 |
| temporal | 2/2, 34.2-100.0 | 1/2, 9.5-90.5 | 1/2, 9.5-90.5 | 1/2, 9.5-90.5 |
| open-domain | 1/1, 20.7-100.0 | 1/1, 20.7-100.0 | 1/1, 20.7-100.0 | 1/1, 20.7-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs eidetic-product | locomo/multi-hop | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs eidetic-product | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs eidetic-product | locomo/temporal | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/multi-hop | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/temporal | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-full | locomo/multi-hop | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-full | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-full | locomo/temporal | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-vector | locomo/multi-hop | 1 | 1 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-vector | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-product vs rag-vector | locomo/temporal | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/multi-hop | 2 | 1 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/open-domain | 1 | 0 | 0 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/temporal | 2 | 0 | 0 | 1.0000 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 15511 | 3547 |
| eidetic-product | 15511 | 3206 |
| rag-full | 15511 | 15511 |
| rag-vector | 15511 | 1842 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 3295.3 | 3481.5 | 12833.1 | 24839.2 |
| eidetic-product | 2418.0 | 2862.9 | 62844.6 | 84172.5 |
| rag-full | 0.0 | 0.0 | 7270.1 | 7963.1 |
| rag-vector | 0.4 | 0.6 | 4632.5 | 5841.1 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 5 | 80.0% | 20.0% | 0.0% |
| eidetic-product | 4 | 75.0% | 0.0% | 0.0% |
| rag-full | 5 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 5 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
