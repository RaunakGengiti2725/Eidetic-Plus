# Cost A/B ledger — measured arms and promotion verdicts

All numbers from dev-split runs with artifact paths. Holdout claims live elsewhere.
Rule: no default flip without a dev A/B AND an offline proof; n>=40 confirmation before
profile promotion.

## ADAPTIVE_CONTEXT — GO for bench profile, n>=40 confirmed

| arm | verified-correct | median qtok | artifact |
|---|---|---|---|
| baseline dev-20 | 15/20 | 5,586 | wave_i_adaptive_ab_codex |
| ADAPTIVE_CONTEXT=1 dev-20 | 15/20 | 4,033 (−27.8%) | wave_i_adaptive_ab2_codex |
| ADAPTIVE_CONTEXT=1 dev-40 fresh ingest | 25/40 (h2h context) | −28% held | wave_j_h2h_locomo40_codex |

Verdict: **GO** for the bench/product profile as a flag-on default candidate; the −28%
reproduced twice at n=20 and held at n=40 with accuracy inside the ±1-row noise band.
Holdout slices deliberately ran it OFF (unpromoted-flag discipline); the flip decision
belongs to the holdout session after r5 (HANDOFF item), because flipping mid-rotation
would confound the window trend.

## EXTRACT_COMBINED — NO-GO tonight (quality shift unassessed)

| arm | accuracy | store composition | artifact |
|---|---|---|---|
| stack ON dev-20 (COMBINED+RESULT_CACHE+ADAPTIVE) | 15/20 holds | claims −8%, edges +20% | wave_k_writecost_ab_codex |

Call-halving is proven by counting tests (1 call vs 2 per window) but INVISIBLE to the
harness's write_tokens proxy (counts content volume — measurement defect recorded). The
claims −8% composition shift needs the claim-coverage sidecar assessment before any
promotion: the enumerator's ceiling IS claim coverage, so trading claims for edges could
silently starve the structured path we are trying to grow. Verdict: **NO-GO** until the
sidecar comparison runs (queued; needs one dev ingest pair, ~15 min API, deferred while
slice 5 owns the budget).

## FAST_ABSTAIN — measurement designed, pending API budget

Offline facts: the gate short-circuits BEFORE context assembly and the reader call; the
reader-path abstention measured 3.8–7.2s e2e in the MCP UX exercise vs ~30ms structured.
The flag forfeits only NLI rescues of drafts whose best dense coverage is under the 0.25
floor (strictly below the 0.4 abstention threshold).

Probe (run when slice 5 releases the budget): dev-20, arms FAST_ABSTAIN=0/1, compare
abstention count, verified-correct, and e2e_ms on abstained rows. GO criteria: zero
verified-correct lost, abstained-row e2e_ms down >10x.

## record_ops deletion tranche 1 — attempted, result below

Static scan: 0 zero-reference helpers among 177 (everything is wired). Deletion must be
earned by claim-path supersession, not dead-code sweeps. Tranche 1 candidate list, in
supersession order (each requires: suite green, wave-F replay byte-identical, rotating
SMQE sidecars green, on-store probes of the shapes the collector served):

1. `_open_or_preference_answer` movie-specific branch — VERDICT: NO-GO. Inspection shows
   it owns the copular title extraction ('X is one of my favorite movies' -> title) that
   the category-family gate does NOT replace (the gate filters atoms; this extracts the
   value), with positive coverage in test_extractive_verification. Not superseded;
   deletion here would be accuracy loss dressed as hygiene.
2. `_generic_itemized_count_answer` vs `_generic_list_count_answer` overlap — merge
   candidate once claim-tier counting exists (P1); NOT deletable while count_aggregate
   answers depend on them (claims cannot count yet).
3. Junk gates superseded by verify-layer form floors — each needs a proof that the floor
   catches every shape the gate caught (the floors run at verify; the gates run at
   execute, so replay-visible junk differs).

Honest bottom line: the big shrink (~400–600 lines) is gated on P1 write-time claim
coverage, not on tonight. Anything deleted before claims can answer the load is accuracy
loss dressed as hygiene.

## Wave-2 status (feature/acceleration)

FAST_ABSTAIN and EXTRACT_COMBINED dev arms DEFERRED again tonight: slice-5 eidetic phase
finished (40/40) but the mem0 tail is mid-run and owns the DashScope RPM. Both arms are
one command each (FAST_LOOP.md); run them the moment the tail lands.

Claim-tier counting shipped (050513f0d): the head->verbs COUNT path now exists, which is
the gating condition for collector-deletion tranche items #2 (the _generic_*_count
family). Next measurement: claim-coverage sidecar comparison on a fresh dev ingest, then
the deletion tranche with the full matrix net.

## Deletion tranche 2 (wave 3): attempted, NO-GO with evidence

`_generic_list_count_answer` (89 lines) deleted behind the full gate battery: two suite
failures -- the collector carries EXISTENCE counts over itemized lists ('how many
release blockers are there?' from 'release blockers: A, B, C' atoms), a shape with no
action verb, which claim_count's head->verbs SELECT cannot serve by design. REVERTED.
Supersession precondition recorded: claim extraction would need itemized-list claims
(claim per listed item with a shared list_id) before this collector can die. The
verb-backed count load (books read / cities visited) IS now claim-served; the
existence-count load is not.

## Wave-4 closure status

FAST_ABSTAIN and EXTRACT_COMBINED arms deferred a THIRD time (mem0 tail active at every
gate check across waves 2-4). They are the first two commands to run when the tail
lands -- staged in FAST_LOOP.md, GO criteria above. Nothing else in this ledger is
blocked on API.

## FAST_ABSTAIN dev-20 A/B — NO-GO (measured 2026-07-04)

| arm | verified-correct | abstained | abstained e2e_ms | qtok med |
|---|---|---|---|---|
| OFF | 17/20 | 3 | 16.1s / 16.6s / 25.8s | 5,664 |
| ON  | 16/20 | 3 | 18.6s / 19.2s / 18.5s | 5,356 |

The pre-gate NEVER FIRED: every real abstention carries dense coverage above the 0.25
floor, so the flag changed nothing on the target path (latency identical within noise;
the single row flip is the known 3-vs-4-weeks gold-ambiguity shape, judge noise). GO
criterion was a 10x abstained-latency drop; measured drop is 0x. Verdict: NO-GO as
designed. The 16-26s abstention cost is cascade+retry time at coverage 0.25-1.0 --
cutting it needs the reflex coverage plane (P4) or an ABSTENTION_V2-aware early signal,
NOT a higher floor (that trades measured NLI rescues away). Flag stays default-off;
WEAKNESS_QUEUE #6 updated to point at P4.

Side observation, free of charge: dev-20 baseline now reads 17/20 verified-correct --
above the 15/15/14 historical band -- on the post-wave build.

## EXTRACT_COMBINED + RESULT_CACHE dev-20 arm (measured 2026-07-04)

| arm | verified-correct | qtok med | claims | edges |
|---|---|---|---|---|
| baseline (same-day, same protocol) | 17/20 | 5,664 | 18,059 | 1,141 |
| COMBINED+CACHE | 15/20 | 5,070 | 19,295 (+6.8%) | 1,322 (+15.9%) |

THE BLOCKER IS RESOLVED: the wave-K claims -8% composition shift did NOT reproduce on
the current build -- the combined prompt now yields MORE claims (+6.8%), likely because
the wave-2/3 extraction patterns changed what the schema asks for. Accuracy: -2 rows,
of which one (the 3-vs-4-weeks shape) flipped in BOTH of today's arms independently --
pure judge noise -- and one is unexamined. Verdict: NO default flip on a -2-at-n=20
reading, but candidacy UPGRADED: the write-call halving is proven, the composition fear
is gone, and the remaining question is one n=40 confirmation run. Recommend: holdout
session runs the n=40 pair after slice 6 lands.

Instrumentation note RESOLVED: the right fields are extra.structured_recall and
extra.smqe_operator (extra.note is reader-path only). Structured rate on the current
build: 8/20 baseline, 9/20 COMBINED (40-45%, vs ~9/20 measured on the older live probe
at twice the question count) -- and COMBINED's +1 structured row is consistent with its
+6.8% claims. miss_taxonomy and slice-6 forensics should read structured_recall.

## EXTRACT_COMBINED n=40 confirmation (measured 2026-07-04) -- GO as promotion candidate

| arm | verified-correct | structured rows | claims | total qtok | qtok mean |
|---|---|---|---|---|---|
| OFF (same-day baseline) | 25/40 vc (26 correct) | 18/40 | 27,919 | 139,600 | 3,490 |
| ON (COMBINED+CACHE) | 24/40 vc (25 correct) | 20/40 | 27,268 (-2.3%) | 123,432 (-11.6%) | 3,086 |

The n=20 -2 did NOT hold at n=40: accuracy is inside the noise band (7 churn flips, 3
gained / 4 lost, no direction). Claims composition flat (-2.3%, vs the -8% blocker that
started this) with +2 structured rows. Write-call halving already proven by counting
tests. One measurement trap documented: the ON arm's qtok MEDIAN read -49%, an artifact
of structured coverage crossing 50% (median falls off the reader plateau); totals are
the honest metric. The parity story in one number: a structured row costs 18 tokens
median vs ~6,000 for a reader row -- every question that crosses saves ~330x.

VERDICT: GO as a bench-profile promotion candidate alongside ADAPTIVE_CONTEXT. Default
flip remains the holdout session's call (mid-rotation flips confound the window trend).

Free n=40 finding: the current build's dev-40 baseline reads 26/40 with TEMPORAL 9/10
and 45% structured coverage -- the first fresh-ingest n=40 carrying the wave-2/3 write
path, and the targeted class is nearly clean off-holdout.

## Promotion profiles shipped (2026-07-04)

GO flags now live as stackable profile files -- code defaults remain 0 everywhere;
holdout stays OFF by design until the user says "promote on holdout" (slice 7+).

- `bench/profiles/product_cost.json` -- ADAPTIVE_CONTEXT=1, EXTRACT_COMBINED=1,
  EXTRACT_RESULT_CACHE=1 (the measured-GO stack, nothing speculative)
- `bench/profiles/dev_cost_ab.json` -- same three flags pinned to 0 (baseline arm)
- `bench/run_cost_profile.sh` -- inherited wave-F env + profile overlay + fresh
  DATA_DIR per arm, output to `artifacts/cost_<arm>_<timestamp>/`

Copy-paste (one API bench at a time, dev split only):

```bash
bench/run_cost_profile.sh bench/profiles/dev_cost_ab.json base 40
bench/run_cost_profile.sh bench/profiles/product_cost.json product 40
COST_PROFILE_DRY=1 bench/run_cost_profile.sh bench/profiles/product_cost.json check 20
.venv/bin/python -m bench.cost_report artifacts/cost_base_<ts> artifacts/cost_product_<ts>
```

## Real write-side cost measured -- the proxy defect is closed (2026-07-04)

`bench/cost_report.py` reads `extra.consolidate.model_usage` (real DashScope
input+output tokens, deduplicated per conversation) instead of the `write_tokens`
content-volume proxy that was arm-invariant by construction. Re-scored the existing
dev-40 pair with zero new API spend:

| arm | vc | structured | qtok total | write tok (real) | write calls | total tok | tok/vc |
|---|---|---|---|---|---|---|---|
| dev40_combined_off | 25/40 | 18/40 | 139,600 | 856,459 | 882 | 996,059 | 39,842 |
| dev40_combined_on | 24/40 | 20/40 | 123,432 (-11.6%) | 545,374 (**-36.3%**) | 552 (-37.4%) | 668,806 (-32.9%) | **27,867 (-30.1%)** |

The write side dominates total cost (~86% of tokens on the OFF arm), so the COMBINED
call-halving that was invisible to the proxy is the single largest cost lever measured
to date: **-30.1% total tokens per verified-correct answer** with accuracy inside the
noise band. This upgrades the GO case for `product_cost.json` -- the -11.6% qtok
reading understated the win by ~3x.

## Verified-cost vs Mem0 -- honest framing (holdout r6 as reference)

Read-side qtok is the wrong parity axis: Mem0 is cheap AND unverified. The metric this
program optimizes is total DashScope tokens per verified-correct answer.

| system | median qtok | judge-correct | verified-correct | total tok/verified answer |
|---|---|---|---|---|
| eidetic-full (holdout r6) | 4,898 | 25/40 | 25/40 | 41,322 |
| eidetic + product_cost stack (dev-40 ON arm) | 2,515 | 25/40 | 24/40 | 27,867 |
| mem0 (holdout r6) | 382 | 18/40 | **0/40** | N/A (nothing verified) |

Mem0's write-side cost is not instrumented in its adapter (add() calls unlogged), so
even its raw total is understated. At 0 verified answers its cost per verified answer
is infinite; the honest gap to close is our own 27,867 -> down, via structured
coverage growth (COST_ROADMAP.md), not via chasing an unverified system's qtok floor.
