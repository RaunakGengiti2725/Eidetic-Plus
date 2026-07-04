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
