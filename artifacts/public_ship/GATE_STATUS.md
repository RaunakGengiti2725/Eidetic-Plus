# Release gate status — honest summary (2026-07-05, slice 8)

`python -m bench.release_gate --out artifacts/public_ship` → **FAIL** (157 PASS / 470 FAIL).
This is the expected and honest result: the gate encodes the full public-SOTA standard
(≥10-run reproduce sweep, all six systems, both datasets, ablation evidence, calibrated
abstention, slice-invariant draws). This artifact dir contains ONE rotation window
(slice 8 / r8, runs=1, LoCoMo, eidetic-full + mem0). We ship the FAIL report on purpose;
the claim scope is `limited` accordingly (`claim_scope.json`).

On the r8 refresh, three integrity checks that were red on the r7 hard window flipped
to PASS on their own merits (no number edited): verified_accuracy 57.5% ≥ 50%,
proof_support 36/36 verified rows carry proof, verify_step present. The remaining
failures are the genuine multi-run / multi-system / coverage requirements below.

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
| ablation evidence | 2 | ENVIRONMENT-BLOCKED. The 5-arm dev ablation (~6h of DashScope ingest, re-ingesting the corpus per arm) has been attempted twice; both times the subprocess was killed on session teardown before completing (full arm clean at 40/40, 23 correct; metabolism-off reached only 8/40). A first, pre-date-fix attempt produced an error-polluted report (22/26 full-arm rows crashed) that was removed rather than shipped. Even a clean complete run would report `pass:False` on two HONEST negatives (region routing ~neutral, forgetting cost-neutral at dev scale) per docs/claims.md. Claim B's headline — metabolism-memory-off drops accuracy — showed a directional ~+19pp in the polluted run but is NOT cited as clean evidence. | run `bench/reproduce.sh` in an environment that sustains an uninterrupted multi-hour job |
| slice_invariant | 3 | single directional draw per dataset committed (LME 11/24 vc, LoCoMo 10/20 vc); gate standard is 5 fresh draws | rerun `bench.run_slice_invariant_eval` with --draws 5 |
| smqe log policy structured_rate 35% < 80% | 1 | r8 structured coverage 14/40 — the real gap this program is closing | write-side claim families (see COST_ROADMAP) |

No number in this directory was edited to make a check pass. The r8 refresh flipped the
r7 integrity failures to PASS on their merits (57.5% verified accuracy, full proof
support); the structured-rate gap is the real remaining capability lever.

## Update 2026-07-09 — P0 aggregate citation floor (gate re-run: FAIL 156 PASS / 471 FAIL)

The live LME-S numeric panel had shipped **5/13 verified-WRONG** through the deterministic
count/sum path (`bench/measure_sum_live_probe.py`; the 1226.3-vs-70 case among them). Fix
shipped in `eidetic/smqe/verify.py`: a DERIVED count or cross-session sum no longer verifies —
`verified=True` now requires a SINGLE cited source that STATES the value; everything else
fails closed. Live panel now **0 verified-wrong** (4 verified-correct preserved, 9 abstain).

Effect on this directory, all honest:

- The 9 affected SMQE sidecars were regenerated under the new contract (fresh random seeds)
  and are green. `smqe_fullpath` now reads **"37/46 verified full-path"** — the 9 missing are
  the derived-aggregate cases that ABSTAIN BY DESIGN (`expected_abstain_cases: 9` in the
  report). The gate's claim-backed/all-verified checks were re-scoped to answerable cases
  (`cases - expected_abstain`), with the abstain count published in each report so the
  denominator change is auditable. Requiring 100% claim-backing over ALL cases would require
  re-opening the verified-wrong leak.
- Headline moved 157/470 → **156/471**. The single flip is `ablation:valid_json` PASS→FAIL:
  the earlier `ablation_report.json` was a transient on-disk file from the killed ablation
  run (never committed — see the ablation row above); this re-run reports its absence
  honestly instead of inheriting a stale PASS.
- The structured-rate FAIL (r8 logs) is unchanged — frozen committed logs are not re-scored.
  Note for future windows: fail-closed aggregates will slightly LOWER structured coverage on
  fresh runs (counts/sums route to the reader now); that is the trust trade chosen, measured
  before it lands in a scoreboard.
