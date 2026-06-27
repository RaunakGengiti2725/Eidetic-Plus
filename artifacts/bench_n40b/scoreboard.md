# Eidetic-Plus benchmark scoreboard

## locomo - accuracy by category (%), mean±std; n = questions/run

| category (n) | eidetic-plus-full | rag-full | rag-vector |
|---|---|---|---|
| multi-hop (n=15) | 60.0±0.0 | 66.7±0.0 | 46.7±0.0 |
| temporal (n=20) | 55.0±0.0 | 45.0±0.0 | 20.0±0.0 |
| open-domain (n=5) | 80.0±0.0 | 80.0±0.0 | 100.0±0.0 |

## locomo - Wilson 95% CI by category

| category | eidetic-plus-full | rag-full | rag-vector |
|---|---|---|---|
| multi-hop | 9/15, 35.7-80.2 | 10/15, 41.7-84.8 | 7/15, 24.8-69.9 |
| temporal | 11/20, 34.2-74.2 | 9/20, 25.8-65.8 | 4/20, 8.1-41.6 |
| open-domain | 4/5, 37.6-96.4 | 4/5, 37.6-96.4 | 5/5, 56.6-100.0 |

## Head-to-head paired tests

| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |
|---|---|---:|---:|---:|---:|---|---|
| eidetic-plus-full vs rag-full | locomo/multi-hop | 15 | 2 | 3 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/open-domain | 5 | 0 | 0 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-full | locomo/temporal | 20 | 5 | 3 | 0.7266 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/multi-hop | 15 | 4 | 2 | 0.6875 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/open-domain | 5 | 0 | 1 | 1.0000 | no | needs-2-runs |
| eidetic-plus-full vs rag-vector | locomo/temporal | 20 | 9 | 2 | 0.0654 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/multi-hop | 15 | 4 | 1 | 0.3750 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/open-domain | 5 | 0 | 1 | 1.0000 | no | needs-2-runs |
| rag-full vs rag-vector | locomo/temporal | 20 | 6 | 1 | 0.1250 | no | needs-2-runs |

## Cost (approx tokens, uniform ~4 chars/token across all systems)

| system | tokens / write (per conversation) | tokens / query |
|---|---|---|
| eidetic-plus-full | 15511 | 7922 |
| rag-full | 15511 | 15511 |
| rag-vector | 15511 | 1882 |

## Latency (ms)

| system | search p50 | search p95 | e2e p50 | e2e p95 |
|---|---|---|---|---|
| eidetic-plus-full | 1890.7 | 3000.8 | 10714.1 | 14192.6 |
| rag-full | 0.0 | 0.0 | 3892.6 | 8828.5 |
| rag-vector | 0.7 | 2.1 | 2640.1 | 4921.4 |

## Integrity (verified recall) - from logged verify/abstain flags

_verified accuracy = correct AND entailment-proven, over ALL questions (so abstentions and unverifiable categories depress it -- it is a recall metric, not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an entailment proof (NOT a fabrication count: an unproven answer can still be correct). abstention rate = declined for lack of evidence. Systems without a verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by construction, which is not the same as fabricating._

| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |
|---|---:|---:|---:|---:|
| eidetic-plus-full | 40 | 47.5% | 45.0% | 0.0% |
| rag-full | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |
| rag-vector | 40 | N/A (no verify step) | N/A (no verify step) | 0.0% |

> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale (1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction ~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two categorical wins no competitor has (flat recall-vs-age, verified recall with a citable immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box.
