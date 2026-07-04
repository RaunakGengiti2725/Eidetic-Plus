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

## r5 forensics (fill when scoreboard lands)

- scoreline: _pending_
- verified-wrong taxonomy: _pending_
- new classes → WEAKNESS_QUEUE: _pending_
- form-floor matrix on r5 jsonl (kills / flips): _pending_

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
| FAST_ABSTAIN + EXTRACT_COMBINED dev arms | — | DEFERRED twice (mem0 tail active both waves); one command each in FAST_LOOP.md |
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
