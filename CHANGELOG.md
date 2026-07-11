# Changelog

## 1.1.2 (temporal derivation boundary + age-neutral ranker, 2026-07-11)

### relative_temporal derivation boundary (the measured 57%-VW class, closed selection-side)
- The legacy candidate loop tags every shipped date `:atom_derived` or `:mention_selected` (dominance-scoped contest: only rivals matching the event at least as well on target and entity hits contest a selection; score is excluded because score embeds recency). Deterministic resolutions — ordinal-first, unique bare-year statement, identical-statement recurrence, question-lemma exclusivity — keep `atom_derived`; week/month phrases deliberately never reassign or shield (replay-refuted shape).
- New note-keyed `temporal_selection` floor fails `mention_selected` dates closed.
- New `bench.selection_replay`: re-runs current selection code against burned windows' frozen store snapshots offline (zero provider calls). Measured: 12/35 verified-wrong convert to abstention, 26/28 verified-correct keep exact answers, 0 diff-attributable atom-derived regressions (pre-diff drift row + deliberate fail-closed tie itemized; stashed-tree baseline report shipped for attribution).

### Age-neutral ranking enforced on the shipped path (WP5 subset)
- `RRF_W_RECENCY` defaults to 0.0; the recency fusion channel and its underfill fallback exist only on explicit opt-in.
- `Engine.prove_age_independence` probes both the raw index and the full `retrieve()` fusion path; `flat` requires both.
- Offline proof: full-path ranking invariant under age permutation at the default, divergent only under opt-in.

### Hygiene
- Ephemeral live-run artifact dirs and the unrelated `Karaoke-Minimalistic/` tree are gitignored; evidence artifacts (replay, guard projection, selection replay, forensics) are tracked.

## 1.1.1 (accuracy-guard wave, 2026-07-10)

### Deterministic accuracy guards (mined from burned-window replay forensics)
- Clean-fact form floor gains three shapes: dangling separator tails, degenerate conjunction repetition, and junk-stripped question echoes; all pure structured-answer form floors are consolidated in `structured_answer_form_floor` (one source of truth for runtime and replay projection).
- Preference answers to `genre/kind/type/style of X` questions must anchor to the X object class in the answer or a cited atom; cue-matched atoms about a different class abstain.
- `relative_temporal` candidates carry a stated-vs-deictic derivation flag: on evidence ties, a unique explicitly stated period outranks deictic session resolutions, and conflicting stated periods abstain. All-deictic ties keep the sanctioned latest-instance convention (time-invariant suite unchanged).

### Measured effect (mechanical, zero provider calls)
- New `bench.replay --project-guards` artifact (`artifacts/guard_projection_r1_r10`): of 341 frozen verified rows, 21 of 129 verified-wrong rows convert to abstention under today's pure floors; 3 verified-correct rows are lost (all pre-existing first-person shape-1 rejections, enumerated in full in the artifact).
- Honest gates: the stated-evidence temporal guard operates on live candidate sets and is NOT included in projected numbers; the reader family (47 wrong) and purely deictic wrong-instance selection remain documented residuals requiring READ-stage work and a live dev A/B.

### Hygiene
- Leakage audit green: five pre-existing forensic sample-id references in `bench/HANDOFF.md` registered as documented exemptions with evidence pointers.
- Live-capability probe (`eidetic.doctor`): embed/chat/rerank/image healthy; optional `read_document` model id 404s (typed, fail-loud).

## 1.1.0 (proof-boundary release candidate, 2026-07-10)

### Canonical factual-answer boundary
- Added explicit `AnswerStatus`: public factual results are only `VERIFIED` or `ABSTAINED`; legacy `verify=false` transport inputs cannot bypass proof.
- Centralized Python, HTTP, MCP, structured recall, fixed-reader evaluation, cache hits, and external drafts on `Engine` governance.
- Verification now requires exact scope, query-time record activity, immutable raw-byte rehashing, resolvable source spans, no active contradiction, and independent support for every sentence/list item. Abstentions are citation-free and never expose the discarded draft as an answer.
- Images verify against raw pixels; PDF/audio/video claims are re-read from immutable media; unsupported binary descriptions fail closed.

### NotebookLM
- Direct Gemini output is typed `UNTRUSTED_DRAFT`; MCP exposes it only as `draft`.
- Added `NotebookLMBridge.governed_recall`: the draft and exact exported evidence IDs pass through `Engine.prove_external_draft`, returning only `VERIFIED` or `ABSTAINED` with immutable proof.
- Separated zero caller-generation tokens from proof-model usage; no zero-total-cost claim is made for governed recall.

### Evaluation and replay
- Fixed-reader full/product adapters consume the exact Engine evidence telemetry; removed the second representative retrieval and all coverage/dense unverified-answer escape hatches, including active profile flags.
- Added deterministic `bench.replay` with SHA-256 binding to source logs, outputs, and implementation bytes. The r1–r10 artifact covers 400 frozen rows: 12 historical unverified deliveries become abstentions, 0 remain, and 212/212 verified-correct rows are preserved with zero provider/NLI/generation calls.
- Replay also reports the unresolved accuracy frontier honestly: 129/341 historical verified rows were judge-wrong. The replay proves output-policy closure, not current-runtime accuracy.
- Added adversarial proof, scope, temporal validity, contradiction, multimodal, transport-parity, benchmark-parity, NotebookLM, and replay tests.

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
