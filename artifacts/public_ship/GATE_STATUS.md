# Release gate status — honest summary (2026-07-04)

`python -m bench.release_gate --out artifacts/public_ship` → **FAIL** (146 PASS / 481 FAIL).
This is the expected and honest result: the gate encodes the full public-SOTA standard
(≥10-run reproduce sweep, all six systems, both datasets, ablation evidence, calibrated
abstention, slice-invariant draws). This artifact dir contains ONE rotation window
(slice 7, runs=1, LoCoMo, eidetic-full + mem0). We ship the FAIL report on purpose;
the claim scope is `limited` accordingly (`claim_scope.json`).

## What PASSES (the part a skeptical reader should check first)

- All six required artifacts present (manifest, scoreboards, both curves, snap_back).
- Log fingerprint stable; scoreboard fingerprint matches the raw jsonl logs.
- claim_scope: limited scope declared, 7 limitations enumerated, no SOTA wording,
  harness names have logs.
- Manifest: test split, holdout profile, no dataset source scans, session-granularity
  ingest, samples file + data dir recorded, zero system failures / error rows.
- Holdout leakage audit: PASS embedded (`holdout_audit.json`, 1,670 needles).
- Snap-back fidelity: 272/272 = 100%.
- All 22 offline invariant sidecars green (SMQE synthetic/planner/fullpath/paraphrase/
  conflict/composition/relative-phrase/temporal-window/attribution/abstention/scope/
  subscope/time/invalidation/dialogue/lacuna, claim coverage at 46/46, affect salience,
  scratchpad, region routing, reflex recall, crystal demotion).

## What FAILS, and why we did not "fix" it

| class | count (approx) | reason it fails | honest path to green |
|---|---|---|---|
| dominance / paired stats | ~232 | runs=1; CI checks need ≥10 runs | run `bench/reproduce.sh` |
| missing systems | ~160 | only eidetic-full + mem0 in this window | full sweep with rag-full/rag-vector/graphiti/eidetic-product |
| LongMemEval categories | ~19 | LoCoMo-only window | full sweep |
| abstention calibration | 11 | calibration RUN on 264 dev rows (`abstention_v2_tau.json`): no tau reaches the 0.95 precision target — the verifier's current verified-wrong rate caps precision below 95%, so tau=1.0 abstains everything. The report is committed; the gap is a capability gap, not a missing file | reduce verified-wrong classes (partial-list, junk fragments — in progress), then recalibrate |
| ablation evidence | 2 | no ablation dirs in this artifact | `python -m bench.run_dev_ablation` |
| slice_invariant | 3 | single directional draw per dataset committed (LME 11/24 vc, LoCoMo 10/20 vc); gate standard is 5 fresh draws | rerun `bench.run_slice_invariant_eval` with --draws 5 |
| integrity: verified_accuracy 45% < 50% | 1 | r7's real vc is 18/40 — the hard window's true number | better build, more windows; not editable |
| integrity: unverified_answered 2 | 1 | two correct-but-unverified rows in r7 logs | real behavior, reported as-is |
| smqe log policy structured_rate 32.5% < 80% | 1 | r7 structured coverage 13/40 — the real gap this program is closing | write-side claim families (see COST_ROADMAP) |

No number in this directory was edited to make a check pass. The three integrity-class
failures are real properties of the r7 run and are quoted in the public docs as limits.
