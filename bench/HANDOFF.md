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
