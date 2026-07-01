# Forgetting-machine proof slice: LongMemEval-S multi-session (n=6, 1 run)

**Status: DIRECTIONAL, not significant.** One run, n=6 questions, one category. Every McNemar
p=1.0000. This validates the full proof pipeline (head-to-head + attribution ablation through one
shared reader/judge, with a recorded manifest) and shows the predicted signals; it is NOT the
multi-run gate (`bench/reproduce.sh`) and supports NO SOTA claim. See [docs/claims.md](../docs/claims.md).

## Setup (the fairness pins)
- Dataset: LongMemEval-S, `multi-session` category, samples at offset 70, n=6, `--split all`, 1 run.
- Shared reader: `qwen-plus` via `answer_with_fixed_reader` (identical for every system).
- Shared judge: `qwen3-max`. METABOLISM_MODE=1 for the head-to-head (Tier-A reader on for all).
- Each sample is a ~122k-token, ~50-session haystack. Manifest: `artifacts/proof/multisession/headtohead/run_manifest.json`.

## Claim A -- competence under a fair reader (directional)

| system | accuracy | tokens / query | e2e p50 |
|---|---:|---:|---:|
| eidetic-plus-full | **33.3% (2/6)** | **7,996** | 11.9s |
| rag-full (stuff all context) | 16.7% (1/6) | 122,752 | 18.8s |
| rag-vector (chunk+embed+topk) | 33.3% (2/6) | 1,944 | 3.3s |
| mem0 | FAILED (DashScope 400 on add) | -- | -- |

- eidetic **beats full-context RAG (+16.7pp) at ~1/15th its query tokens AND lower latency**: rag-full
  pays 122,752 tokens/query to stuff the whole haystack and is both less accurate and slower here.
  This is the cost/latency operating-point win, and cost/latency are deterministic (not noisy at n=6).
- eidetic **ties vector RAG** on accuracy; eidetic feeds 7,996 vs 1,944 tokens (frame as the
  accuracy/token operating point, not a clean accuracy win).
- Accuracy is low across the board (16-33%) because multi-session LongMemEval is genuinely hard --
  this is the test where published long-context systems degrade.
- McNemar: eidetic vs rag-full 1 vs 0 discordant, p=1.0000 (not significant at n=6).

## Claim B -- attribution by ablation (directional; the centerpiece)

Ablate the metabolism MEMORY layer (FULL_SLEEP, gist/co-activation/struct channels, capture
fidelity, graph-temporal reasoning) while holding the shared reader AND the proof gate fixed, so the
delta isolates memory quality, not reader strength.

| eidetic config | accuracy | delta |
|---|---:|---:|
| metabolism ON (full memory) | 33.3% (2/6) | -- |
| metabolism memory OFF (reader+proof fixed) | 16.7% (1/6) | **-16.7pp** |

Per-question flip (`artifacts/proof/multisession/attribution_compare.md`): **1 question gained**
by turning the memory layer on, 0 regressed. Sample identifiers are intentionally omitted from source
docs so holdout leakage audits can guard the real test split. The long-horizon gain points to the
memory layer, in the predicted direction. McNemar p=1.0000 (1 discordant, n=6) -- directional, not
significant.

## Integrity row (honest tax)
eidetic abstained on 33.3% (2/6) rather than guess; verified accuracy 16.7%. The RAG baselines emit
no proofs by construction (N/A), which is not the same as fabricating.

## Snap-back fidelity over THIS corpus (deterministic guarantee)
`DATA_DIR=data/proof_multisession_h2h scripts/snap_back_audit.py` -> **285/285 = 100.0000% lossless**:
every content-addressed memory ingested in this run snaps back byte-identical from the immutable
substrate (`sha256(get_raw(h)) == h`). Forgetting lowered only FSRS index priority; no raw record was
mutated. This is a number, not a demo, and it holds at any corpus size.

## What this run is NOT
- Not significant (n=6, 1 run, every p=1.0000). The significance gate is a larger n + multi-run.
- Not a Mem0 comparison (mem0 add() failed with a DashScope 400 on this content; see forensics).
- Not a temporal/knowledge-update result (multi-session only).

## Robustness fixes this run forced (real bugs; the run could not complete without them)
1. Transient HTTP 5xx (embedding pipeline 500) now retried -- one hiccup was aborting whole runs.
2. Truncated extraction JSON now salvaged (max_tokens overflow) instead of aborting consolidation.
3. Content-moderation 400 on a real passage now skips that extraction window (raw stays in the
   substrate) instead of aborting the sample.
Each fix re-issues identical requests or drops only un-processable input; none fabricates.
