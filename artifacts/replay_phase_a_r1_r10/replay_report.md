# Burned-Window Mechanical Replay

Status: **PASS**

This artifact performs no provider, generation, NLI, or judge calls. It replays the Phase A output policy over frozen rows.

| metric | original | replay |
|---|---:|---:|
| rows | 400 | 400 |
| correct | 221 | 212 |
| verified answered | 341 | 341 |
| verified correct | 212 | 212 |
| abstained | 47 | 59 |
| unverified answered | 12 | 0 |
| verified wrong | 129 | 129 |

| source | rows | original correct | replay correct | converted unverified | invalid verified proof |
|---|---:|---:|---:|---:|---:|
| holdout_rotation_r10_codex/eidetic-plus-full__run0.jsonl | 40 | 19 | 17 | 4 | 0 |
| holdout_rotation_r1_codex/eidetic-plus-full__run0.jsonl | 40 | 23 | 23 | 0 | 0 |
| holdout_rotation_r2_codex/eidetic-plus-full__run0.jsonl | 40 | 17 | 15 | 2 | 0 |
| holdout_rotation_r3_codex/eidetic-plus-full__run0.jsonl | 40 | 27 | 26 | 1 | 0 |
| holdout_rotation_r4_codex/eidetic-plus-full__run0.jsonl | 40 | 23 | 22 | 1 | 0 |
| holdout_rotation_r5_codex/eidetic-plus-full__run0.jsonl | 40 | 24 | 24 | 0 | 0 |
| holdout_rotation_r6_codex/eidetic-plus-full__run0.jsonl | 40 | 25 | 25 | 1 | 0 |
| holdout_rotation_r7_codex/eidetic-plus-full__run0.jsonl | 40 | 20 | 18 | 2 | 0 |
| holdout_rotation_r8_codex/eidetic-plus-full__run0.jsonl | 40 | 23 | 23 | 0 | 0 |
| holdout_rotation_r9_codex/eidetic-plus-full__run0.jsonl | 40 | 20 | 19 | 1 | 0 |

## Checks

- **all_sources_nonempty**: PASS
- **row_count_preserved**: PASS
- **no_unverified_answered**: PASS
- **no_invalid_verified_proof_metadata**: PASS
- **zero_verified_correct_regression**: PASS
- **zero_provider_calls**: PASS

## Limitations

- **Boundary**: Reuses frozen historical correctness and verification labels; it does not rerun retrieval, generation, NLI, or judges.
- **Boundary**: Validates immutable proof metadata linkage in logs but cannot re-fetch historical CAS bytes without the original data stores.
- **Boundary**: Measures the exact policy effect of forbidding unverified delivery, not the full quality of the current runtime.
