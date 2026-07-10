# Design: dual-timestamp claims (event_time vs observed_at)

Status: **DESIGN FOR REVIEW — no code**. Written 2026-07-10 against measured evidence;
the build itself is a fresh-session task per the operating rules (bi-temporal semantics
change to the trust substrate).

## Problem, measured

`bench/dual_timestamp_probe.py` (committed, with per-window results):

- Claims whose proof_atom states an **explicit calendar date** are stamped with the
  record's SESSION time instead: **460/460 (100%) across r13 / r15 / LME-S** — the claim
  channel never backdates.
- The event channel (`EventRecord.start`) carries the stated day only **14.3%** of the
  time on LME-S (0% in the small LoCoMo samples).
- The existing event-date claim family's `filters.event_date` tag covers **15/455 (3.3%)**
  of the affected LME-S claims — its precision-first patterns are correct but narrow.
- Downstream consumers of `filters.event_date` ALREADY exist (date-anchored verification
  `:date_anchored`, event-identity tagging), so widened coverage is consumed for free.

Impact class: temporal-reasoning questions (fixed-reader band 2/8–6/9 per window;
LME-S temporal 29–71% depending on path). 455 mis-stamped dated claims on ONE LME-S
window is the single largest measured, unfixed accuracy surface.

## Two separable stages

### Stage 1 — widen `filters.event_date` tagging (additive; no valid_at change)

Tag a claim with `event_date` + `date_precision` when ALL of:

1. The proof_atom contains an **explicit** date phrase (the probe's grammar: month-day,
   month-day-year, or ISO; **relative phrases excluded** — they are stage-1b at best).
2. The date phrase and the claim's event verb are in the **same sentence** (reuse
   `_segment_event_date` / the event-identity segmenter — do NOT write a new resolver).
3. The claim does not already carry `event_date` (never overwrite the precision-first
   family's output).
4. The resolved date is **not the session date itself** (tagging "today" restatements
   adds noise, no signal).

Known trap (why this is not a late-night patch): a date in an atom is not always the
claim's event date — "we talked about the July 11 concert" vs "I went on July 11".
The precision gate is the verb-adjacency rule (2) plus the dated-verb families the
event-date claim work already validated. **Promotion gate:** on the r13/r15/LME-S
stores, sample ≥50 newly-tagged claims, hand-verify ≥95% precision; ANY mis-tag class
found gets a written exclusion rule before ship. Regression: the probe's
claims-with-tag count (15 → target ≥300 on LME-S) AND zero changes to already-tagged
claims (byte-diff).

### Stage 2 — the bi-temporal decision (`valid_at` semantics)

Two options, decide at build time with a dev-split A/B:

- **A (conservative, recommended):** keep `claim.valid_at` = observed/session time
  (unchanged everywhere), and teach the READ paths to prefer `filters.event_date` when
  the query is temporal (the `:date_anchored` machinery already half-does this).
  Zero risk to supersession/`as_of`; wins bounded by reader/executor consumption.
- **B (full):** `claim.valid_at` = resolved event time; add `observed_at` field.
  Touches ~42 `valid_at`-conditioned code paths in `store.py`/`executor.py` alone
  (`active_claims_at`, supersession ordering, `as_of` reads, `_source_cutoff`).
  A claim about a PAST event becoming "valid" before its source record exists inverts
  write-once assumptions. Requires: bi-temporal invariant tests (event_time ≤
  observed_at OR explicitly future-dated), supersession decided by observed_at (never
  event_time — later statements win regardless of when their events happened), and
  `as_of` reads defined against BOTH axes. Do not attempt B until A's ceiling is
  measured.

## Regression gates (all committed already)

- `bench/dual_timestamp_probe.py` — mismatch rate + tag coverage, per window.
- `tests/test_dataset_time_roundtrip.py` — calendar-day round-trip, TZ-independent.
- Live: `bench/measure_sum_live_probe.py` (temporal_delta rows must stay correct) and
  the fixed-reader temporal category on the NEXT fresh window (current band 2/8).

## Explicitly out of scope

Relative-date resolution ("last June", "two weeks ago") — harder class, needs the
anchor-event machinery, separate design. Blanket tagging of any date-bearing atom —
the measured whack-a-mole trap; precision gates or nothing.
