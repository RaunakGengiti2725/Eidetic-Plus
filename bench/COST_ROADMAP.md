# Cost roadmap — structured coverage growth (dev-only, no holdout)

Source: dev-40 ON-arm jsonl (`artifacts/dev40_combined_on_codex`), 20/40 rows still hit
the reader at ~5-7k qtok each vs ~28 tok structured. Every row that crosses saves ~330x.
North star: total DashScope tokens per verified-correct answer (currently 27,867 with
the product_cost stack; write side is 82% of that — see COST_AB.md).

## Next 3 claim families (priority order)

### 1. Itemized-list claims (P1b) — the deletion unblock
One claim per listed item with a shared `list_id`, plus existence-count support in
claim_count ("release blockers: A, B, C" → count=3 with no action verb).
- Serves: multi-hop attribute enumerations — 6/20 reader rows ("what snacks does Sam
  enjoy", "what tricks do James's pets know", "interests Joanna and Nate share").
- Expected lift: +4-6 structured rows (50% → ~62%), ~−25-35k qtok on dev-40.
- Unblocks: deletion tranche 3 — `_generic_list_count_answer` (89 lines, NO-GO'd twice
  on exactly this missing shape) and the `_generic_itemized_count_answer` merge.
- Status: NOT started this session (stop-rule scope); design + gates recorded in
  COST_AB tranche-2 entry. Requires wave-F replay + form-floor matrix (write path).

### 2. Alias / naming claims
Claim shape for naming relations: nicknames ("Nate calls Joanna 'Jo'"), named
artifacts ("board game Mafia", "dance piece 'Finding Freedom'"), named techniques
("Pomodoro").
- Serves: 4-5/20 reader rows across open-domain + single-hop.
- Expected lift: +3-4 structured rows (~62% → ~72%), ~−15-25k qtok.
- Deletion gated on it: none directly; reduces reader-path abstentions (c6_q17
  "board game where you find the imposter" abstained at 5.9k qtok).

### 3. Event-date claims
Event → explicit date attribute, queryable via `value_as_of`/temporal operators
("when was Calvin's concert in Tokyo", "after how many weeks did Tim reconnect").
- Serves: 2/20 reader rows now, but hardens temporal_delta/relative_temporal (already
  the biggest structured operators) against regression.
- Expected lift: +1-2 structured rows, ~−8-14k qtok; main value is temporal-class
  robustness on the class we already win.

Combined ceiling if all three land: structured ~28-30/40 (70-75%), dev-40 read-side
~−45-70k qtok (−36-57% of the ON arm's 123k), before any write-side change.

## Flags: promoted vs experimental

| flag | state |
|---|---|
| ADAPTIVE_CONTEXT | GO — in product_cost.json |
| EXTRACT_COMBINED + EXTRACT_RESULT_CACHE | GO candidate — in product_cost.json |
| FAST_ABSTAIN | closed NO-GO (gate never fires) |
| REFLEX_COVERAGE_GATE (P4) | design-only, below |

## P4 — reflex coverage plane (design only, not built)

FAST_ABSTAIN failed because real abstentions carry dense coverage 0.25-1.0; the 16-26s
cost is cascade+retry AFTER retrieval looks plausible. A pre-dense floor can never fire.
Design: short-circuit before context assembly + reader when the verify path is hopeless
on cheap signals available pre-reader —
- structured path declined AND no claim-tier candidates for the planned operator, AND
- entity/lexical seed overlap between query terms and candidate set below a floor
  measured from abstained-row forensics (not guessed like 0.25 was).
Flag `REFLEX_COVERAGE_GATE=1`, default OFF. Measurement: dev-20 A/B, GO = abstained-row
e2e_ms down >5x with verified-correct unchanged. Prereq: pull the two cheap signals for
the 5 abstained dev-40 rows offline first — if they don't separate abstentions from
rescued rows, the design is dead before spending an API arm (FAST_ABSTAIN lesson).

## Deletion tranches gated on the families

| tranche | lines | gate |
|---|---|---|
| 3: `_generic_list_count_answer` + itemized-count merge | ~150 | family 1 shipped + suite + wave-F replay + SMQE sidecars + on-store probes |
| 4: junk gates superseded by verify form floors | ~100-200 | per-gate floor-coverage proof (execute-vs-verify replay differs) |
| 5: remaining collector shrink toward ~400-600 total | rest | families 1-3 all claim-serving their loads |
