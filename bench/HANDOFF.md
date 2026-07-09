# Cross-session handoff queue

Holdout session (connected-brain-loop) reads this after each slice lands. Acceleration
session (feature/acceleration) appends tasks with pin SHAs. Shapes as convN-rowM only.

## For the holdout session, after r5 scoreboard

- [ ] Paste r5 scoreboard + eidetic jsonl path here; acceleration session fills the
      forensics section below (WEAKNESS_QUEUE gets the class updates, DOMINANCE_PROGRESS
      stays holdout-owned).
- [ ] Slice 6 should run TWO-PHASE (FAST_LOOP.md): Phase A eidetic-only, forensics at
      +40 min, Phase B mem0 overnight, same --out, pinned SHA both phases.
- [ ] Slice 6 pin: merge feature/acceleration first if COST_AB verdicts below say GO;
      otherwise pin the r5 SHA and note the divergence.
- [ ] City-visit write path + proper-noun place enumeration measure on r5 (first fresh
      ingest carrying them). Check the which-cities shape (conv4-row30 class) before
      claiming anything.
- [ ] Lemma instance selection (c796a0463) measures on r5: check conv9 album shape.

## r5 FINAL (both phases landed)

- h2h: eidetic 24/40 (60%) vs mem0 17/40 (42%) -- THIRD consecutive window win, largest
  margin yet (+7). Multi-hop 6/8 vs mem0 2/8 this window.
- Rolling n=200 never-touched: 114/200 (57%) vs 101/200 (50%); verified answers 173 vs 0.
  Window margins: +1, -1, +4, +2, +7.
- Cost gap unchanged: qtok median 5,438 vs 380 (~14x) -- the arms below are the lever.

## r5 forensics (eidetic phase, read-only; mem0 tail still running at fill time)

- scoreline (eidetic only): 24/40 (60%), verified 35, abstained 5, verified-wrong 11;
  multi-hop 6/8 (best window yet), temporal 2/8, single-hop 16/22, open-domain 0/2.
- verified-wrong taxonomy: temporal 6/11 -- tagged {lemma-miss} and {week-window} by
  miss_taxonomy, i.e. EXACTLY the classes the wave-2/3 write path (P2 event identity,
  lemma families, month-granularity windows) targets; r5 ingested on the pre-P2 build,
  so slice 6 is the measurement. Non-temporal VW: two junk shapes (greeting echo,
  filler-item list) -- floor gaps closed same-day (2a8e7a10f, matrix 2/5/6/2/2 kills,
  zero flips on five windows); one two-item credible-list content miss (form cannot
  judge); two start/collaborate shapes that the shipped open/team lemma families cover.
- abstained (5): three {abstained-late} single-hops (FAST_ABSTAIN arm relevance) + two
  judgment-call open-domain rows where abstention is arguably correct.
- new classes -> WEAKNESS_QUEUE: gerund-object heads ('started playing Civilization VI'
  -- instance head lands on the gerund's object, verify tagging on slice 6), month-of
  achievement questions ('in which MONTH's game...' answered with a bare later date).
- form-floor matrix on r5: kills 2, flips none.

## From acceleration → holdout (this branch's deliverables)

| Item | SHA | State |
|------|-----|-------|
| Planning docs + miss_taxonomy | 566f25ac9 | shipped |
| Wings 7 problem memory + tests | e076014a2 | shipped |
| Wings 8 witness scaffold + test | e9452e7b5 | shipped |
| COST_AB verdicts (ADAPTIVE GO, COMBINED NO-GO, tranche#1 NO-GO) | 990ab2e5f | measured |
| P2 write-time event identity (lemma+head+date tags, instance answers) | 120446db5 | shipped |
| P1 claim-tier counting (head->verbs, unblocks deletion tranche #2) | 050513f0d | shipped |
| phase_holdout.sh + taxonomy subshapes | 3eecd12a9 | shipped |
| ask_problem NL war-room recall | 22ce903e8 | shipped |
| FAST_ABSTAIN dev A/B | (COST_AB) | NO-GO -- gate never fires; fix class is P4 reflex plane |
| EXTRACT_COMBINED dev-20 + n=40 | fb683149d | GO as promotion candidate (claims flat, qtok total -11.6%, accuracy in noise); flip = holdout call |
| Likelihood-exemption fragment kill + head-stop hygiene | e5830efc4, b2cd95789 | shipped |
| P6 wings-as-claims (PROBLEM_CLAIMS, typed :problem SELECT) | a27f4b96a | shipped |
| P2 breadth (lemma-family compat, week->month honesty, adverb/work-on shapes) | 6566561a6 | shipped |
| Deletion tranche 2 | e2ddc90f6 | NO-GO with evidence (existence counts need itemized-list claims first) |
| PROBLEM_EXTRACT ingest hook (default off) | 5fbd6bba7 | shipped |
| SLICE6_PLAN + form_floor_matrix.sh + dev_fast profile | 3637b8487 | shipped |
| demo_war_room.sh (offline, zero API) | (this commit) | shipped |

## Slice 6 pin recommendation

Merge feature/acceleration into connected-brain-loop BEFORE slice 6 Phase A: the write
-path changes (event-identity tags, count claims, extraction patterns for adverb/
particle/possessive-subject events) only measure on fresh ingests. Expected movement:
temporal wrong-instance shapes (event_instance path answers release/open/team-up
questions with exact or month-precision dates) and count questions (claim_count path).
Run slice 6 with bench/phase_holdout.sh A for forensics at +40 min.

## UX2 exercise (2026-07-04, current build, live key, multi-turn + cross-session)

Battery: contradiction chain with backdating (dentist switch), as_of era answers, bulk
import with in-batch dedup, honest zero-citation abstention, war-room built from PLAIN
conversation via PROBLEM_EXTRACT (decision + rationale extracted, ask_problem verified
with a revision-backed citation in 4ms), cross-namespace scope isolation.

CAUGHT AND FIXED SAME HOUR (c14b3d0d8): abbreviation-blind segmentation truncated
answers at honorifics in TWO places -- the recalled dentist was literally 'Dr'. Shared
abbreviation-aware splitter + honorific-tolerant value capture; live re-run fully green.

Latency profile: remember ~1.1-2.1s (embed + auto-sleep), structured recall 3-34ms,
recall_problem/ask_problem 0-4ms, unknown-question abstention 10.2s (the P4 reflex-plane
class, unchanged).

## SLICE 6 PHASE A (window 5, digest 06a923.., branch SHA pinned in launch_A.log)

25/40 (62%) | verified 36, abstained 3, VW 11 | structured 17/40 | temporal 6/9.

THE WRITE PATH GENERALIZED: temporal 6/9 on never-touched holdout vs 2/8 on r5's
pre-P2 ingest, with a live :event_instance row (near-miss: April vs May -- the mechanism
identified the instance at month granularity, the earlier statement won the date; not
junk). Remaining temporal misses are the known no-verb-family week-window class.
Second-best window headline (r3 = 27). Phase B (mem0) launched for the h2h; two-phase
runner delivered forensics at +40 min as designed.

## SLICE 6 FINAL -- fourth consecutive window win

| | correct | verified | temporal | qtok med |
|---|---|---|---|---|
| eidetic-plus-full | 25/40 (62%) | 36 | 6/9 | 4,898 |
| mem0 | 18/40 (45%) | 0 | 0/9 | 382 |

Margins across six disjoint never-touched windows: +1, -1, +4, +2, +7, +7. Rolling
n=240: 139/240 (58%) vs 119/240 (50%); verified answers 209 vs 0. Temporal across all
windows: ours 19/43, mem0's 3/43. The write-path waves measured on holdout: temporal
2/8 (pre-P2 ingest) -> 6/9 (wave-2/3 build), :event_instance firing live. The two-phase
runner cut forensics latency to +40 min in production use.

## SHIPPED TO USER #1 (2026-07-04)

`claude mcp add eidetic` at user scope -- every Claude Code session on this machine now
runs the RC 1.0.0 build (branch venv, persistent store at ~/.eidetic-plus/data,
namespace raunak-main). Connection verified by the client; first real memory written and
recalled VERIFIED with citation in 1.9s through the deployed configuration. The
real-user feedback loop is live: every future session exercises remember/recall/war-room
against the honest build. Reversible with `claude mcp remove eidetic`.

Wheel eidetic_plus-1.0.0-py3-none-any.whl built and cold-verified; PyPI publication +
public announcement remain authorization-gated (irreversible, outward-facing) -- the
one-liner is `twine upload <scratchpad>/dist/eidetic_plus-1.0.0-py3-none-any.whl`.

## SLICE 7 PHASE A (window 6, digest af54f5.., SHA 6f56d7743) -- promotion measurement

20/40 (50%) | vc 18 | verified 32 | abstained 6 | VW 14 | structured 13/40 |
qtok med 4,029 | write tok real 516,036 at 527 calls (COMBINED halving held on holdout).

Hard window; both bars missed (vc 18 vs r6's 25; structured 13 vs >=21). Claim plateau
held (structured rows 6-55 qtok) but coverage did not transfer from dev mix. VW: 5
partial-list, 2 temporal wrong-instance, 1 week-window (honest), 1 preference_synth
junk fragment (form-floor gap), 1 lemma-miss. Full forensics in DOMINANCE_PROGRESS.
Phase B mem0 running same --out. Phase-6 conditional triggered: event-date family,
dev-40 probe first, zero holdout tuning; Phase A re-run ONLY on +2 structured at zero
vc regression (fresh ingest, same draw, same SHA discipline).

## SLICE 7 FINAL (h2h landed)

h2h: eidetic 20/40 (50%) vs mem0 12/40 (30%) -- FIFTH consecutive window win, +8 the
largest margin yet, ON THE HARD WINDOW (our worst absolute score of seven). Rolling
n=280 never-touched: 159/280 (57%) vs 131/280 (47%); verified 241 vs 0; temporal
22/51 vs 3/51. Window margins: +1, -1, +4, +2, +7, +7, +8. Event-date claim family
shipped post-r7 (9437ce421, 18 workflow-confirmed defects fixed pre-commit); dev-40
probe product3 launched for the +2-structured/zero-vc-regression re-run decision.

### Rolling never-touched holdout table, r1-r7 (n=280; regenerate: bench/rolling_holdout_table.py)

| window | eidetic correct | eidetic verified | eidetic structured | eidetic qtok med | mem0 correct | margin |
|---|---|---|---|---|---|---|
| holdout_rotation_r1_codex | 23/40 | 34 | 11/40 | 5344.5 | 22/40 | +1 |
| holdout_rotation_r2_codex | 17/40 | 33 | 17/40 | 4900.5 | 18/40 | -1 |
| holdout_rotation_r3_codex | 27/40 | 38 | 15/40 | 5016.5 | 23/40 | +4 |
| holdout_rotation_r4_codex | 23/40 | 33 | 16/40 | 5138.5 | 21/40 | +2 |
| holdout_rotation_r5_codex | 24/40 | 35 | 12/40 | 5438.0 | 17/40 | +7 |
| holdout_rotation_r6_codex | 25/40 | 36 | 17/40 | 4898.5 | 18/40 | +7 |
| holdout_rotation_r7_codex | 20/40 | 32 | 13/40 | 4029.0 | 12/40 | +8 |
| **rolling** | **159/280** | **241** | **101/280** | | **131/280** | **+28** |

## Hackathon checklist (public ship 2026-07-04)

- [x] main = submission default, fast-forwarded from feature/acceleration; all ship
      work committed on main.
- [x] Slice 7 committed: artifacts/holdout_rotation_r7_codex (scoreboard, manifest,
      curves, jsonls, pinned launch logs A/B, product_cost profile copies).
- [x] Proof bundle: artifacts/public_ship (claim_scope limited + 7 limitations,
      snap_back 100% 272/272, rolling n=280 json, holdout_audit PASS, 22 offline
      sidecars green, release_gate report committed at honest FAIL + GATE_STATUS.md).
- [x] docs/PUBLIC_CLAIMS.md + docs/JUDGES.md -- every bullet artifact-cited, losses
      published, dev-vs-holdout cost caveat inline.
- [x] docs/HACKATHON.md shot list (6 shots, all re-runnable, honest-limits shot
      included). scripts/judge_quickstart.sh tested green offline. README judges +
      war-room sections. .env.example current.
- [x] Leakage audit PASS end-state (digit-boundary matcher fix + ledger id scrub +
      benchmark-entity literal removed from _identity_entailment).
- [x] Event-date claim family shipped (9437ce421) behind the full offline gate
      battery; dev-40 probe product3 = the promotion evidence, in flight at handoff.
- [x] Tag v1.0.0-public at 407445393 (probe verdict recorded: no window-6 re-run).
- [ ] PyPI publish + public announcement remain authorization-gated (wheel built,
      twine one-liner in SHIPPED TO USER #1 above).

## Slice-invariant single draws (directional, 2026-07-04 late)

One stratified test-split draw per dataset with the product_cost stack (pinned in each
draw's run_manifest): LME 11/24 vc (single-session-user 4/4, knowledge-update 3/4,
preference 0/4), LoCoMo 10/20 vc, structured 8/20, temporal 3/5. Combined 21/44
verified-correct. artifacts/public_ship/slice_invariant.json + per-dataset sidecars.
NOT the 5-draw standard; gate fails it correctly; directional evidence that the
rotation-window band (~45-60%) holds under stratified random draws too.

## SLICE 8 FINAL (r8, h2h landed) -- new-build validation, largest margin

h2h: eidetic 23/40 (58%) vs mem0 9/40 (23%), margin +14 (LARGEST of eight windows).
First holdout carrying VW-killer + event-date family + date-overflow fix on
product_cost. Temporal 6/9 vs 0/9, verified 36 vs 0, ZERO errors (crash class closed).
Partial-list VW 5->2, no junk fragments. tok/vc 31,761 (best of series). Rolling n=320:
182/320 (57%) vs 140/320 (44%), +42, verified 277 vs 0, six consecutive wins.

| window | eidetic correct | eidetic verified | eidetic structured | eidetic qtok med | mem0 correct | margin |
|---|---|---|---|---|---|---|
| holdout_rotation_r1_codex | 23/40 | 34 | 11/40 | 5344.5 | 22/40 | +1 |
| holdout_rotation_r2_codex | 17/40 | 33 | 17/40 | 4900.5 | 18/40 | -1 |
| holdout_rotation_r3_codex | 27/40 | 38 | 15/40 | 5016.5 | 23/40 | +4 |
| holdout_rotation_r4_codex | 23/40 | 33 | 16/40 | 5138.5 | 21/40 | +2 |
| holdout_rotation_r5_codex | 24/40 | 35 | 12/40 | 5438.0 | 17/40 | +7 |
| holdout_rotation_r6_codex | 25/40 | 36 | 17/40 | 4898.5 | 18/40 | +7 |
| holdout_rotation_r7_codex | 20/40 | 32 | 13/40 | 4029.0 | 12/40 | +8 |
| holdout_rotation_r8_codex | 23/40 | 36 | 14/40 | 4029.0 | 9/40 | +14 |
| **rolling** | **182/320** | **277** | **115/320** | | **140/320** | **+42** |

## P0 NUMERIC TRUST LEAK CLOSED (2026-07-09) — aggregate citation floor

**Blocker resolved:** the live LME-S numeric panel shipped **5/13 verified-WRONG** (one/23,
1226.3/70, 4/3 weddings, 6/5, 3/4). Now **0 verified-WRONG** (4 correct preserved: negroni "10",
2 temporal deltas, $25k; 9 abstain). Gate: `DATA_DIR=artifacts/lme_s_r1_codex/data python
bench/measure_sum_live_probe.py`.

- **Change:** `eidetic/smqe/verify.py` — aggregate CITATION floor. For `count_aggregate` /
  `multi_session_sum` only, `verified=True` requires `len(supports)==1` AND the answer's cardinal
  verbatim in that sole atom; else `return None` (fail-closed). Comparative-difference queries
  ("how many MORE ... than", "difference between") are exempt (fixed 2-anchor arithmetic,
  recompute-exact). `temporal_delta` untouched (anchors are `valid_at`, not atom text; 0-wrong).
- **Why not recompute-from-atoms** (goal's literal ask): the error is WRONG ATOM SET, not
  arithmetic — recompute re-derives the same wrong number from the same wrong set. Source
  cardinality is the honest discriminator (all 5 live-wrong are n_supports≥2; the 1 correct count
  is n_supports==1, stated: "tried a Negroni 10 times").
- **Blast radius (measured, net-positive):** derived-count path was **5/6 verified-WRONG on
  holdout** (r6/r8/r9/r12/r14, only r8 c2_q62 correct). Gate removes a liability, not a feature.
  Cross-session sums are inherently multi-support → they now abstain (a sum no single source
  states cannot be citation-verified = correct-or-silent). Accuracy cost: ≤1 verified-correct
  holdout row.
- **Suite: 1608 green.** ~27 synthetic tests + 9 SMQE harnesses updated to assert fail-closed via
  `expect_abstain` (derivation-value + atom-exclusion coverage preserved). fullpath reports
  `reader_consults` (aggregates fall to reader tier + abstain — surfaced, not hidden). Product
  tests (`test_bench_eidetic_full`, `test_retrieval_wiring`) assert abstention.
- **Next / blockers:** (1) `python -m bench.release_gate` re-run (P4). (2) aggregate accuracy
  recovery is a SEPARATE dev-split lever (stated-total detection / retrieval-guided read), NOT to
  be pursued by weakening the gate — any re-open needs a live-probe-proven zero regression.
  (3) NotebookLM Tier-1 unified recall + provenance (P1/P2) still open.

## P4 + P1 SAME DAY (2026-07-09, commits ce3b5a041..17850edcd)

- **Release gate re-run honest FAIL 156/471** (`d265f4e7a`): gate checks re-scoped to
  answerable cases (expected_abstain_cases published in reports — auditable denominator);
  9 SMQE sidecars regenerated green under the fail-closed contract; the one flipped check
  (ablation:valid_json) reports a transient never-committed file honestly.
- **P1 provenance bar MET LIVE (`6923015ec` + `17850edcd`):** quote-content citation_map —
  [n] → memory_id → content_hash by quote bytes (verbatim → unique-best overlap ≥0.8;
  ambiguous/junk unattributed with reason). Live re-query of 26 surviving LME-S rgi
  notebooks: **25/26 rows (96%), 198/204 references (97%)** — was 0/28 token-based. Gate:
  `bench/provenance_live_probe.py`. Single miss = zero-citation Gemini answer.
- **Unified recall wired:** MCP `notebooklm_recall` (retrieval-guided one-call:
  qwen retrieval → focused export → free read) + `recall_routed` (T0 reflex → T1 structured
  → T2 free read / T3 metered gate). `retrieval_guided_answer()` replaces ad-hoc collectors;
  `routed_answer` abstains honestly without notebook_id (no silent metered escalation).
  Suite 1613 green.
- **Goal checklist state:** numeric fail-closed ✅ (0 VW live) | provenance >80% ✅ (96%) |
  unified routed recall ✅ | release gate documented ✅ | REMAINING: product-beats-RAG on ≥2
  windows w/ judged significance, ≥10-run NBL gate, polar/inhibitory-edge eval — all
  live-quota-gated (≥10-run = ~400 free-tier queries; ~30 burned today, ~130 on 07-07).

## HARDENING WAVE (2026-07-09 late, commit 410b53b2f) — the fix's own bypasses, closed

28-agent adversarial review (4 dimensions × verify) of the day's P0/P1 commits confirmed
22 findings; all criticals/majors fixed same-day, each regression-proven on the review's
own end-to-end repros:

- **Citation floor layer 2** (`eidetic/smqe/verify.py`): op-mistagged aggregations
  (planner routes "average"/money-worded counts to latest_value, whose executor still
  derives multi-atom arithmetic) now fail closed via query-shaped floors; difference
  exemption requires comparative ADJACENCY ("...longer than five miles" = filter, no
  exemption); comma-grouped cardinals compare numerically ($1,220 vs '1' — both the
  bypass AND the stated-total false-abstain fixed).
- **Reader path** (`eidetic/retrieval.py`): aggregation-shaped question whose number is
  not stated in any entailed source → unverified (`reader_numeric_floor_enabled`).
- **Gate honesty**: release_gate + merge_artifacts FAIL CLOSED on sidecars missing
  `expected_abstain_cases` (a stale all-verified pass contains the leak); fullpath
  expect_abstain also requires the reader tier was consulted.
- **Provenance resolver**: symmetric normalization, 0.1 overlap margin, `superseded`
  flag, active-record verbatim tiebreak, zero-record guard, APPEND-semantics note.
- **Tests**: fail-closed aggregate tests pin the DERIVATION VALUE via structured_recall
  trace — abstention alone can't distinguish fail-closed from SMQE never running.

Live panel unchanged: 4 verified-correct / 0 verified-wrong / 9 abstain. Suite 1613.
NBL 10-run gate: 3 full runs (85.0/82.5/82.5), quota-blocked until reset (run3 healed to
35/40 then RESOURCE_EXHAUSTED; resume via bench/nbl_run_cycle.sh <window> 3 then 4..9).
Hindsight r15: relaunched on fresh profile eidetic-bench-r15b after pg0-corruption
5-attempt start failure; ingesting (5/40 at handoff). Provenance live re-validation of
the hardened resolver ALSO waits on quota (unit tests cover it meanwhile).

## READ-STAGE PIVOT (2026-07-09 night, commits 474c57b56 + a2736fd39)

Asked "which new algorithm buys 8->10": measured the premise first.
`bench/retrieval_recall_probe.py` over six burned LoCoMo windows + LME-S:
dense recall@10 = 116/119 (97.5%), 1-hop graph ceiling 119/119 -- RETRIEVAL IS
NOT THE BOTTLENECK; spreading-activation/polar/quantum re-ranking ideas target
a stage at ceiling. The gap is the READ stage (same top-k, better reader:
53.3 -> 78.6 LME-S; the 95.6% comparator reads agentically).

Shipped the corresponding lever: `iterative_recall` (bridge + MCP
notebooklm_recall iterative=True) -- reader-declared insufficiency triggers
free decomposition (reader proposes sub-questions -> qwen re-retrieves) plus
one claim-graph hop (spreading activation at read-SET construction, where the
probe showed it recovers), re-export (deduped vs round 1), re-read. 0 metered
tokens; +1-2 free queries per widened round. Offline-tested end-to-end with
fakes; LIVE A/B is quota-gated (queued with run4+).

## MISS-FORENSICS FLEET + FIXES (2026-07-09 night, commits 474c57b56..373fa60e9)

69-agent fleet root-caused ALL 58 judged-wrong product-path rows against real
stores; top clusters adversarially audited. Evidence:
artifacts/forensics/miss_forensics_fleet_20260709.json.

FIXED SAME NIGHT (373fa60e9):
1. SILENT EXPORT TRUNCATION (16/58 rows, 3/3 skeptics): pack_record_sources
   dropped everything past ~23x12KB via packed[:max_sources] -- "no
   information" answers about facts IN the store. Now lossless-or-loud
   (data-driven budget with fragmentation bound + two raising invariants).
   Predicted +10-14 rows on LME-S whole-export. RE-COLLECT (fresh notebooks!)
   at quota reset = the measurement.
2. TEMPORAL ANCHOR CORRUPTION (8/58, 2/3 skeptics): naive-local parse + UTC
   render = ±1 day on evening sessions. Both loaders now tz-aware; invariant
   test sweeps machine TZs. Fresh ingests only (committed stores immutable).

NOT model work: 19/58 = judge/gold miscalibration (incl. 2-3 LoCoMo label
defects). Remaining small clusters: enumeration completeness-mode (4), reader
verification loop (3), rg blind spots (2, iterative_recall covers).
Suite 1621 green. Hindsight r15 at 26/40 (19 correct, 0 errors) at handoff.
