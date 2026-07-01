# Forgetting-machine proof: LongMemEval-S multi-session (n=20, 1 run)

**Status: DIRECTIONAL, trend consistent, NOT yet significant.** One run, n=20, one category. eidetic
is the top scorer and every discordant comparison favors it, but all McNemar p in [0.22, 0.69] -- the
direction is clear, the significance is not. This is NOT the multi-run gate (`bench/reproduce.sh`) and
supports NO SOTA claim. Honesty bind: [docs/claims.md](../docs/claims.md). Manifest:
`artifacts/proof/ms20/headtohead/run_manifest.json`.

## Setup
LongMemEval-S `multi-session`, offset 70, n=20, `--split all`, 1 run. ONE shared reader (`qwen-plus`,
`answer_with_fixed_reader`) + ONE judge (`qwen3-max`) for every system. METABOLISM_MODE=1 (Tier-A
reader on for all). Each sample is a ~123k-token, ~50-session haystack.

## Claim A -- competence under a fair reader (eidetic leads all baselines)

| system | accuracy | Wilson 95% CI | tokens / query | e2e p50 |
|---|---:|---|---:|---:|
| **eidetic-plus-full** | **30.0% (6/20)** | 14.5-51.9 | **7,995** | 11.4s |
| mem0 | 20.0% (4/20) | 8.1-41.6 | 472 | 6.1s |
| rag-vector | 15.0% (3/20) | 5.2-36.0 | 1,937 | 4.2s |
| rag-full (stuff all context) | 10.5% (2/19) | 2.9-31.4 | 123,085 | 11.3s |

Paired McNemar (discordant pairs, all favoring eidetic):
- eidetic vs rag-full: 5 vs 1, p=0.2188
- eidetic vs rag-vector: 4 vs 1, p=0.3750
- eidetic vs mem0: 4 vs 2, p=0.6875

Reading it honestly: eidetic is the **top scorer**, ahead of full-context RAG by ~19.5pp at **~15x
fewer query tokens** (7,995 vs 123,085), ahead of vector RAG by 15pp, and ahead of Mem0 by 10pp.
Every head-to-head's discordant pairs point the same way (eidetic wins more than it loses), so the
*direction* is consistent across all three baselines -- but n=20/1-run lacks the power to make any of
them significant (p>=0.22). The cost gap is deterministic and real regardless of n.

## Claim B -- attribution by ablation (metabolism memory layer earns +10pp)

Ablate the metabolism MEMORY layer (consolidation/dreaming channels, capture fidelity,
graph-temporal) with the shared reader AND proof gate HELD FIXED, so the delta is memory quality.

| eidetic config | accuracy | delta |
|---|---:|---:|
| metabolism ON | 30.0% (6/20) | -- |
| metabolism memory OFF (reader+proof fixed) | 20.0% (4/20) | **-10pp** |

Per-question flips (`artifacts/proof/ms20/attribution_compare.md`): **3 gained**, **1 regressed**.
Sample identifiers are intentionally omitted from source docs so holdout leakage audits can guard the
real test split. Net +2 questions came from the memory layer, in the predicted direction. McNemar
p=0.6250 (directional, not significant).

## Deterministic wins (real at any n)
- **Cost:** eidetic answers from 7,995 tokens vs rag-full's 123,085 (~15.4x cheaper) while scoring higher.
- **Snap-back fidelity:** `DATA_DIR=data/proof_ms20_h2h scripts/snap_back_audit.py` -> **938/938 =
  100.0000% lossless** -- every content-addressed memory snaps back byte-identical from the immutable
  substrate after forgetting lowered only its FSRS index priority.
- **Verified recall + abstention:** eidetic verified-accuracy 20.0%, unproven-answer 5.0%, abstained
  35.0% (declined rather than guess). No baseline emits a proof or abstains -- that is the integrity
  differentiator, and the abstention rate is the honesty tax on the latency (11.4s vs rag-vector 4.2s).

## What this run is NOT
- Not significant (n=20, 1 run; every McNemar p in [0.22, 0.69]). The significance gate is a larger
  n AND the multi-run protocol (`bench/reproduce.sh`), which is a multi-hour compute job.
- Not multi-category (multi-session only; temporal-reasoning / knowledge-update pending).
- mem0 competed (4/20) after the adapter was made to skip content-specific 4xx sessions, the same
  graceful degradation eidetic uses -- so this is an apples-to-apples row, not a crippled baseline.
