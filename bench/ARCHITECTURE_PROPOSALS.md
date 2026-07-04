# Architecture proposals — build faster, stop whack-a-mole

Written on `feature/acceleration` while rotation slice 5 runs. Each proposal: problem,
design, what gets DELETED, measurement, risk. Ordered by leverage.

## P1. Write-time claim schema replaces read-time rescue

**Problem.** record_ops.py is 5,431 lines / 177 helpers of read-time text re-parsing.
Every fresh holdout window surfaces a new phrasing the regexes never saw; the fix loop
is per-shape and permanent. The claim tier (subject + predicate + object + proof atom,
extracted ONCE at write time) already answers enumerations with per-item proofs — the
enumerator needed claims, not smarter regexes, and travel/reading rows proved it.

**Design.** Consolidation extracts typed claims for every event/state/preference atom
(LLM extraction already runs; the marginal cost is schema breadth, not new calls). The
SMQE planner answers from claims FIRST; record-atom scanning becomes the fallback tier;
the reader is the last resort. Collector deletion proceeds tranche by tranche: a legacy
collector dies when the claim path answers its test load.

**Delete.** The `_generic_*_count_answer` family (~400 lines of overlapping count
collectors), `_process_list_answer` legacy branches, per-shape junk gates the form
floors superseded. Tranche 1 in this branch; wave-F replay + rotating sidecars are the
regression net.

**Measure.** Structured-row rate on dev-40 (today ~9/20 live), median qtok (structured
row ≈ 28 tokens vs ~5k reader), verified-correct unchanged or better.

**Risk.** Claim starvation on old stores (claims only exist for fresh ingests) — the
fallback tier stays until claim coverage is measured at scale.

## P2. Event identity at write time (Wave S completion)

**Problem.** Three read-time designs for the temporal wrong-instance class; two reverted
(date proximity cannot distinguish retellings from distinct events sharing a noun), the
third (question-verb lemma families) ships but only helps when the ATOM carries the verb.
A dateless first report ("finally opened!") and an ambiguous second mention still lose.

**Design.** Extraction tags every event claim with `filters["lemma"]` (canonical action
base via the existing verb-family tables) and `filters["obj_head"]` (object head noun).
An event INSTANCE = same lemma + same obj_head; retellings collapse into the instance at
write time, carrying the most precise date any retelling stated. `relative_temporal`
answers when-questions by instance lookup: lemma match to the question verb, then the
instance's best date. Read-time date heuristics shrink to a fallback.

**Delete.** The future-polarity floor, first-report ceiling attempts, and half the
relative_temporal candidate gates become redundant once instances carry authoritative
dates.

**Measure.** The recorded conv-row shapes (album/shop/Gina/course) on a fresh dev
ingest; time-invariant sidecar stays green; relative_temporal line count.

**Risk.** Lemma table coverage; mitigated because the fallback path stays.

## P3. Two-phase holdout pipeline

**Problem.** ~2.5h per slice; the mem0 tail is ~70% of wall clock and blocks forensics
on OUR rows, which are done in ~40 min.

**Design.** Phase A runs `--systems eidetic-full` alone; forensics + fixes start
immediately. Phase B runs mem0 overnight with the same `--out` and samples file (the
scoreboard merges per-system jsonl whenever both exist — bench.run --render-only
rebuilds it). Pin the code SHA in launch.log for both phases.

**Measure.** Iteration latency: forensics start at +40 min instead of +2.5h.

**Risk.** None to integrity — same samples, same judge; phases documented in the ledger.

## P4. Reflex coverage plane (cheap abstention)

**Problem.** The most expensive path is saying "I don't know": full retrieval + context
assembly + a 5–7s reader call whose draft the coverage gate discards.

**Design.** FAST_ABSTAIN already ships flag-off (pre-reader gate under a coverage
floor strictly below the abstention threshold). Promotion path: dev A/B for accuracy
delta + e2e_ms, then REFLEX_RECALL's index as the earlier, cheaper signal (entity/lexical
seed overlap before dense retrieval runs at all).

**Measure.** Abstention-path e2e_ms (7.2s baseline from the MCP UX exercise), zero lost
verified-correct on dev-20.

**Risk.** Forfeits rare NLI rescues of low-coverage drafts — the floor sits below the
abstention threshold precisely so only hopeless queries short-circuit.

## P5. Dual-session worktrees (process, for the user)

Session A (holdout): owns rotation launches + DOMINANCE_PROGRESS; runs on
`connected-brain-loop`. Session B (acceleration): `feature/acceleration` worktree; owns
COST_AB / WINGS / HANDOFF / WEAKNESS_QUEUE docs and feature code. Each slice pins the
SHA it ran; the user merges B→A between slices so every window measures a known build.
`git worktree add ../eidetic-accel feature/acceleration` gives both sessions disjoint
checkouts with one object store — no stash dances, no cross-session dirty state.

## P6. Wings as first-class claim types (integration sketch)

Problem memory (Wings 7) and witness files (Wings 8) reuse the existing spine instead of
growing a side store: a problem is a record whose claims have `claim_type="problem"`
(goal/blocker/hypothesis/decision as predicates, status in filters), so bitemporal
supersession, verify-or-abstain, and the truth ledger work unchanged. Witness blobs are
ordinary substrate objects; a witness claim carries the content hash, so `prove` resolves
them like any citation. No new consistency machinery — the MCP tools are thin doors.

## What NOT to build

- Another read-time clustering/ceiling pass on temporal instances (two reverts, one
  measured no-op — the class is write-time work, P2).
- NLI as a form arbiter (observed entailing fragment soup; deterministic floors only).
- Default flag flips without a dev A/B and an offline proof (standing rule).
- Benchmark-shaped source scans or sample-ID matchers anywhere (leakage audit enforces).
