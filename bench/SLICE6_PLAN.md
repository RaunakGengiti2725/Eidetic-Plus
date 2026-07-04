# Slice 6 plan (holdout session executes; acceleration branch prepared it)

## Merge checklist
1. `git checkout connected-brain-loop && git merge feature/acceleration` (fast-forward
   expected; resolve nothing silently -- any conflict aborts and gets a human look).
2. Full gates on the merged SHA: `.venv/bin/pytest -q` (1338 at branch tip), wave-F
   replay (18), `bench/form_floor_matrix.sh` (zero flips), leakage audit.
3. Pin: record the merged SHA in HANDOFF.md before Phase A.

## Phase A (eidetic-only, forensics at +40 min)
```bash
.venv/bin/python -m bench.rotating_holdout --dataset locomo --n 40 \
  --state bench/holdout_rotation_state.json \
  --out artifacts/holdout_rotation_r6_codex/holdout40.samples.json
bench/phase_holdout.sh A artifacts/holdout_rotation_r6_codex/holdout40.samples.json \
  artifacts/holdout_rotation_r6_codex
```
Phase B (mem0) overnight: `bench/phase_holdout.sh B <same args>`; then `render`.

Profile: default (unpromoted flags OFF). `bench/profiles/dev_fast.json` is the
ADAPTIVE_CONTEXT candidate overlay -- only on the user's explicit call.

## What to watch (first fresh ingest carrying waves R/2/3 write paths)
- temporal: event_instance notes (`:event_instance`) on release/open/team-up/marry/
  graduate/move when-questions; week-only evidence answering at month granularity.
- multi-hop/count: `:claim_count` notes on how-many rows; which-cities enumerations from
  was-in/trip-to/flew-to claims (proper nouns only).
- misses: `python -m bench.miss_taxonomy <jsonl>` -- week-window/partial-list/lemma-miss
  subshapes feed WEAKNESS_QUEUE directly.

## Honest expectations
Windows vary +-5pp on 40 rows; judge the BUILD by shape-level evidence (which notes fire,
which verified-wrong classes recur) rather than the headline number of one window.
