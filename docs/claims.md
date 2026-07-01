# Claims we make, and claims we refuse to make

This file is the honesty guardrail. The understated version of a result survives a skeptical reader
and is closer to true, so the caveats here are louder than the wins. Every number in any writeup is
either tied to an actual run with a recorded manifest (`artifacts/.../run_manifest.json`), or it is
explicitly labeled "not yet measured." A passing demo is not a measurement.

## The two claims worth making (when the runs support them)

**Claim A -- competence under a fair reader.** With the *same* shared reader and the *same* judge
given to every system (`bench/reader.py:answer_with_fixed_reader`, pinned `qwen-plus`; `bench/judge.py`,
pinned `qwen3-max`), eidetic matches or beats full-context RAG, vector RAG, and Mem0 on LongMemEval
at a fraction of the token cost. The shared reader is the whole point: it makes this a memory
comparison, not a reader comparison. The Tier-A reader fixes are applied to every system equally.

**Claim B -- attribution.** The long-horizon gain is mechanistically earned, shown by ablation:
turn the metabolism *memory* components off (consolidation/dreaming, the gist/co-activation/struct
channels, capture fidelity, graph-temporal reasoning) while holding the reader and proof gate fixed,
and long-horizon accuracy drops. If it does not drop, the memory layer is not earning its keep, and
we report that. Forgetting (FSRS/fademem) is a separate ablation with a separate claim: it should be
accuracy-neutral while buying cost/scale, because it is rank-neutral by invariant
(`tests/test_age_independence.py`). If cost stays flat with forgetting off, forgetting is not buying
anything, and we report that too. Both negative results are worth finding.
For release artifacts this is generated, not hand-written: `python -m bench.run_dev_ablation`
runs comparable dev-split ablations and builds `ablation_report.json` from real logs and source log
fingerprints. The runner fails before spending if inherited env or CLI overrides make the full and
forgetting-off pruning profiles identical, or make forgetting-off prune more than full. If the three
artifact directories already exist, `python -m bench.build_ablation_report`
can build only the sidecar. `bench.release_gate` fails closed if that evidence is missing or weak.

## Claims we refuse to make

- "SOTA" or "best memory agent in the world" without the multi-run gate behind it. A small slice or
  one seed is a directional signal, never proof. The run is `bench/reproduce.sh` (>=10 runs on the
  held-out test split); public eligibility is checked by `python -m bench.release_gate --out artifacts/bench`.
- "Best in the world" without structured evidence for the named top comparators: Chronos, Mastra,
  ByteRover, and Hindsight. A `claim_scope.json` name is not enough: the release gate requires raw
  harness logs for harness systems or per-system external evidence records covering every required
  benchmark dataset before SOTA wording is allowed.
- Any accuracy claim resting on a single seed, a single conversation, or one category, or any phrasing
  that implies a 10-run full-set sweep we did not run.
- Any presentation of a dream/inferred edge as observed fact. Inferred edges are labeled
  (`Edge.inferred=True`) and gated (`eidetic/dreaming/gate.py`); they are never verified as ground truth.
- Any suggestion that forgetting drives the accuracy number. The ablation says forgetting drives cost;
  we lead with that.
- Any number whose score-affecting flags are not recorded in the run manifest
  (`bench/run.py:_MANIFEST_ENV`, which records `METABOLISM_MODE` + the full profile + every other
  score-affecting flag).

## Standing facts the architecture guarantees (measurements, not slogans)

- **Snap-back fidelity.** `engine.snap_back_audit()` confirms `sha256(get_raw(h)) == h` for every
  content-addressed memory; forgetting lowers only FSRS index priority, never the substrate. Reported
  as a rate over the corpus (`scripts/snap_back_audit.py`); the guarantee is 100% or it is a bug.
- **Forget != delete.** The substrate refuses `delete` by design (`eidetic/substrate.py`); guarded by
  `tests/test_no_delete_on_forget.py`, `tests/test_write_once.py`.
- **Age-independence.** Recall ranking carries no absolute-age term; re-prove with
  `engine.prove_age_independence` after enabling any channel.
- **Fairness.** Every system answers through the one shared reader; the comparison isolates memory
  quality, not answerer strength (verified in `bench/adapters/*` -- all call `answer_with_fixed_reader`).

## On the test gate

The neutral baseline (`METABOLISM_MODE` unset) runs the full `tests/` suite byte-identically. The
metabolism profile itself is validated by `tests/test_metabolism_mode.py` and by the live end-to-end
benchmark run; the full unit suite is *not* run under a globally-forced `METABOLISM_MODE=1`, because
the unit tests legitimately control flags individually and their fakes implement only the methods the
default-off paths use. That is a property of the test harness, not a product defect.

For public claims, `bench.release_gate` must also pass over the produced artifact directory. It fails
closed on wrong split, too few runs, missing systems/datasets/categories, tiny or overly-wide
sample-clustered confidence intervals, benchmark error rows, weak paired dominance, non-CI-clear
baseline wins, missed token/latency/age-flat operating budgets, missing verified-recall integrity,
snap-back fidelity below 100%, benchmark logs where structured recall or claim-backed tier-1 recall
does not clear the 80% default gate, slice-invariant sidecars that are plan-only, fixed-seed, non-test,
carry wrong-split sample IDs, duplicate IDs inside a draw, declare too small a sample pool, reuse too few unique samples across draws, not marked `holdout`, or backed only by unverified correctness, an enabled `ABSTENTION_V2` threshold without a matching dev-split
calibration report, missing/failed Mem0 baseline reproduction, stale rendered reports whose raw-log
fingerprints do not match the current JSONL files, or consolidation timeout/deferred debt hidden in
the logs. Debug bypass switches on `bench.release_gate` are for local diagnosis only; a public claim
uses the fail-closed defaults.
