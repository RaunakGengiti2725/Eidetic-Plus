# Changelog

## 1.0.0 (release candidate, 2026-07-04)

Measured across six rotating, disjoint, never-touched holdout windows (n=240):
139/240 verified-or-correct answers vs mem0's 119/240, four consecutive window wins,
209 verified answers vs 0, temporal 19/43 vs 3/43. Full suite 1344 green; leakage
audit and zero-flip form-floor matrix enforced on every change.

### Memory core
- Verify-or-abstain answers with per-citation NLI labels, content hashes, and validity
  windows; abstentions ship zero citations; proofs RESOLVE their references (raw bytes
  re-hashed, snippets located) instead of asserting them.
- Bitemporal store: `valid_at` backdating, `as_of` reads answering each era correctly,
  first-class retractions with latest-wins supersession, truth ledger with full
  supersession chains.
- Write-time event identity: once-ish events carry action-lemma + object-head + dated
  precision tags at extraction; when-questions resolve the right event instance
  (release vs launch party) with month-honest granularity.
- Claim-tier SELECTs: typed counting (metric-unit aware -- assists never answer a
  points question), enumerations with per-item proof atoms, stated-age and month-of-
  superlative composition.
- Deterministic form floors on every answer path: junk, echoes, type-mismatched and
  fragment answers can never ship verified (zero correct answers flipped across all
  six holdout windows, enforced by `bench/form_floor_matrix.sh`).

### War room (new)
- Shared problem memory: immutable revision chains for goals, blockers, hypotheses
  (with scope-validated evidence refs), decisions, and handoffs; `as_of` replays any
  moment of the investigation.
- Hash-verified witness files in the content-addressed substrate.
- `ask_problem`: natural-language questions against the history through the same
  verify-or-abstain path, citations marked revision-backed.
- Optional flags: `PROBLEM_EXTRACT=1` folds explicit markers from plain conversation;
  `PROBLEM_CLAIMS=1` answers war-room questions structurally in milliseconds.

### MCP surface
- 36 tools; cold `pip install` / `uvx eidetic-plus` boots the full surface.
- `remember_many` bulk import (batched embedding, in-batch + store dedup), `repair`
  index rebuild, UTF-8-safe raw paging, per-tool worker threads (one slow call never
  freezes the server), concurrency-safe under parallel writers (tested).
- Abbreviation-aware segmentation ("Dr. Okafor" is never split; "Dr" is never an
  answer) -- caught by live multi-turn exercises, fixed with regression tests.

### Honest limits (documented, not hidden)
- Reads cost more than summary-store rivals (~13x median query tokens); structured
  coverage (43-50% and rising) is the lever -- a structured row costs ~18 tokens vs
  ~6,000 for a reader row.
- Abstentions take reader-pipeline time (measured: no pre-reader signal separates
  them; the latency is the price of verify-or-abstain).
- Open weakness classes live in bench/WEAKNESS_QUEUE.md with fix classes or
  data-backed closure verdicts.
