# Public Release Gate

Status: **FAIL**
Artifact directory: `artifacts/public_ship`
Log fingerprint: `ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f` (2 files)

## Checks

| check | status | detail |
|---|---|---|
| artifact:run_manifest.json | PASS | present |
| artifact:scoreboard.md | PASS | present |
| artifact:scoreboard.json | PASS | present |
| artifact:recall_vs_age.png | PASS | present |
| artifact:latency_vs_age.png | PASS | present |
| artifact:snap_back_audit.json | PASS | present |
| logs:fingerprint_stable | PASS | before ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f (2 files); after ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f (2 files) |
| scoreboard:valid_json | PASS | valid |
| scoreboard:log_fingerprint_present | PASS | ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f (2 files) |
| scoreboard:log_fingerprint_matches | PASS | ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f (2 files) (current ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f (2 files)) |
| claim_scope:valid_json | PASS | valid |
| claim_scope:scope_declared | PASS | limited |
| claim_scope:harness_names_have_logs | PASS | all harness names have logs |
| claim_scope:external_evidence_valid | PASS | not a SOTA claim |
| claim_scope:external_names_have_evidence | PASS | all external names have evidence |
| claim_scope:no_unsupported_sota | PASS | not a SOTA claim |
| claim_scope:top_system_dataset_coverage | PASS | not a SOTA claim |
| claim_scope:top_system_score_floor | PASS | not a SOTA claim |
| claim_scope:limitations_for_limited_claim | PASS | 8 limitations |
| manifest:split | PASS | test (expected test) |
| manifest:runs | FAIL | 1 (required >= 10) |
| manifest:not_render_only | FAIL | render_only=True |
| manifest:systems_cover_required | FAIL | missing: eidetic-plus, eidetic-product, rag-full, rag-vector, graphiti |
| manifest:sample_rows_present | PASS | 40 sample rows |
| manifest:longmemeval:categories_cover_required | FAIL | missing: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning |
| manifest:longmemeval:single-session-user:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:single-session-assistant:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:single-session-preference:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:multi-session:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:knowledge-update:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:temporal-reasoning:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:locomo:categories_cover_required | PASS | all required categories present |
| manifest:locomo:single-hop:sample_rows | PASS | 22 unique sample rows (required >= 20) |
| manifest:locomo:multi-hop:sample_rows | FAIL | 7 unique sample rows (required >= 20) |
| manifest:locomo:temporal:sample_rows | FAIL | 9 unique sample rows (required >= 20) |
| manifest:locomo:open-domain:sample_rows | FAIL | 2 unique sample rows (required >= 20) |
| manifest:no_system_failures | PASS | none |
| manifest:data_dir_recorded | PASS | artifacts/holdout_rotation_r8_codex/data |
| manifest:no_dataset_source_scans | PASS | disabled |
| manifest:session_ingest_granularity | PASS | session |
| manifest:holdout_profile | PASS | holdout |
| manifest:samples_file_recorded | PASS | artifacts/holdout_rotation_r8_codex/holdout40.samples.json |
| holdout_audit:valid_json | PASS | valid |
| holdout_audit:evidence | PASS | 1639 holdout needles, 0 findings |
| ablation:valid_json | PASS | valid |
| ablation:evidence | FAIL | pass:FAIL; region_delta_pp:-4.18<required:2.00; forgetting_accuracy_regression_pp:3.98>allowed:1.00; forgetting_cost_ratio:0.982<required:1.050 |
| affect_salience:valid_json | PASS | valid |
| affect_salience:evidence | PASS | 168/168 checks, boost ratio 0.489803 |
| scratchpad:valid_json | PASS | valid |
| scratchpad:evidence | PASS | 264/264 checks, proof links 96 |
| region_routing:valid_json | PASS | valid |
| region_routing:evidence | PASS | 288/288 checks, proof links 48 |
| reflex_recall:valid_json | PASS | valid |
| reflex_recall:evidence | PASS | 288/288 checks, p95 1.130575 ms |
| slice_invariant:valid_json | PASS | valid |
| slice_invariant:pass | FAIL | <missing> |
| slice_invariant:evidence | FAIL | dataset:<missing>; missing:locomo,longmemeval |
| smqe_planner:valid_json | PASS | valid |
| smqe_planner:evidence | PASS | 162/162 planner checks, p95 0.667917 ms |
| smqe_synthetic:valid_json | PASS | valid |
| smqe_synthetic:evidence | PASS | 46/46 cases, avg proof 30.72 |
| smqe_claim_coverage:valid_json | PASS | valid |
| smqe_claim_coverage:evidence | PASS | 46/46 claim-backed, rate 1.0 |
| smqe_fullpath:valid_json | PASS | valid |
| smqe_fullpath:evidence | PASS | 46/46 verified full-path, reader_calls 0, proof links 46, claim 46, avg context 36.54, p95 12.351729 ms |
| smqe_paraphrase:valid_json | PASS | valid |
| smqe_paraphrase:evidence | PASS | 24/24 cases, record 24, claim 24 |
| smqe_conflict:valid_json | PASS | valid |
| smqe_conflict:evidence | PASS | 24/24 cases, record 24, claim 24 |
| smqe_composition:valid_json | PASS | valid |
| smqe_composition:evidence | PASS | 24/24 composition cases, record 24, claim 24 |
| smqe_relative_phrase:valid_json | PASS | valid |
| smqe_relative_phrase:evidence | PASS | 24/24 relative phrase cases, record 24, claim 24 |
| smqe_temporal_window:valid_json | PASS | valid |
| smqe_temporal_window:evidence | PASS | 24/24 temporal window cases, record 24, claim 24 |
| smqe_attribution:valid_json | PASS | valid |
| smqe_attribution:evidence | PASS | 24/24 attribution cases, record 24, claim 24 |
| smqe_abstention:valid_json | PASS | valid |
| smqe_abstention:evidence | PASS | 24/24 cases abstained, record 24, claim 24 |
| smqe_scope:valid_json | PASS | valid |
| smqe_scope:evidence | PASS | 96/96 scoped checks, record 48, claim 48 |
| smqe_subscope:valid_json | PASS | valid |
| smqe_subscope:evidence | PASS | 96/96 sub-scope checks, record 48, claim 48 |
| smqe_time:valid_json | PASS | valid |
| smqe_time:evidence | PASS | 96/96 as-of checks, record 48, claim 48 |
| smqe_invalidation:valid_json | PASS | valid |
| smqe_invalidation:evidence | PASS | 96/96 invalidation checks, record 48, claim 48 |
| smqe_dialogue:valid_json | PASS | valid |
| smqe_dialogue:evidence | PASS | 24/24 dialogue Q->A crystal checks, seed_mode:random |
| smqe_lacuna:valid_json | PASS | valid |
| smqe_lacuna:evidence | PASS | 24/24 lacuna polarity/retraction/absence checks, seed_mode:random |
| crystal_demotion:valid_json | PASS | valid |
| crystal_demotion:evidence | PASS | 20/20 demotion checks, avg_ratio=0.1545, seed_mode:random |
| abstention_calibration:valid_json | PASS | valid |
| abstention_calibration:ok | PASS | True |
| abstention_calibration:method | PASS | abstention_v2_tau |
| abstention_calibration:split | PASS | dev (expected dev) |
| abstention_calibration:system | PASS | eidetic-plus-full (expected eidetic-plus-full) |
| abstention_calibration:samples | PASS | 264 (required >= 50) |
| abstention_calibration:target_precision | PASS | 0.950 (required >= 0.950) |
| abstention_calibration:precision_at_tau | FAIL | 0.000 (target 0.950) |
| abstention_calibration:nonzero_coverage | FAIL | 0.000 |
| abstention_calibration:tau_applied | FAIL | report=1.000000001, manifest=<unset> |
| abstention_calibration:log_fingerprint_present | PASS | 3f0c24806df6973eff9bff291ade279edaef8d7392cb47affc95bbf9f2b9eb62 (7 files) |
| logs:nonempty | PASS | 80 rows |
| logs:no_error_rows | FAIL | 27 error rows |
| smqe:notes_clean | PASS | no legacy structured-recall policies |
| smqe:log_policy_shape | FAIL | structured_rate:0.350<required:0.800 |
| region:telemetry | PASS | 25/40 rows with hints; 75 hints |
| logs:match_manifest_sample_rows | FAIL | expected=280, actual=80, missing=200, extra=0 |
| logs:held_out_split | PASS | 0 rows outside test split |
| systems:required_present | FAIL | missing: eidetic-plus, eidetic-product, rag-full, rag-vector, graphiti |
| eidetic-plus:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| eidetic-plus:runs | FAIL | 0 runs: [] |
| eidetic-plus:datasets | FAIL | missing: longmemeval, locomo |
| eidetic-plus:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| eidetic-plus:longmemeval:runs | FAIL | 0 runs: [] |
| eidetic-plus:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| eidetic-plus:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| eidetic-plus:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| eidetic-plus:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| eidetic-plus:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| eidetic-plus:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| eidetic-plus:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| eidetic-plus:locomo:runs | FAIL | 0 runs: [] |
| eidetic-plus:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:locomo:single-hop:runs | FAIL | 0 runs: [] |
| eidetic-plus:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| eidetic-plus:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:locomo:temporal:runs | FAIL | 0 runs: [] |
| eidetic-plus:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus:locomo:open-domain:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:questions | FAIL | 40 unique samples, 40 rows (required >= 1000 unique samples) |
| eidetic-plus-full:runs | FAIL | 1 runs: [0] |
| eidetic-plus-full:datasets | FAIL | missing: longmemeval |
| eidetic-plus-full:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| eidetic-plus-full:longmemeval:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:locomo:questions | FAIL | 40 unique samples, 40 rows (required >= 300 unique samples) |
| eidetic-plus-full:locomo:runs | FAIL | 1 runs: [0] |
| eidetic-plus-full:locomo:single-hop:questions | PASS | 22 unique samples, 22 rows (required >= 20) |
| eidetic-plus-full:locomo:single-hop:runs | FAIL | 1 runs: [0] |
| eidetic-plus-full:locomo:multi-hop:questions | FAIL | 7 unique samples, 7 rows (required >= 20) |
| eidetic-plus-full:locomo:multi-hop:runs | FAIL | 1 runs: [0] |
| eidetic-plus-full:locomo:temporal:questions | FAIL | 9 unique samples, 9 rows (required >= 20) |
| eidetic-plus-full:locomo:temporal:runs | FAIL | 1 runs: [0] |
| eidetic-plus-full:locomo:open-domain:questions | FAIL | 2 unique samples, 2 rows (required >= 20) |
| eidetic-plus-full:locomo:open-domain:runs | FAIL | 1 runs: [0] |
| eidetic-product:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| eidetic-product:runs | FAIL | 0 runs: [] |
| eidetic-product:datasets | FAIL | missing: longmemeval, locomo |
| eidetic-product:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| eidetic-product:longmemeval:runs | FAIL | 0 runs: [] |
| eidetic-product:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| eidetic-product:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| eidetic-product:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| eidetic-product:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| eidetic-product:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| eidetic-product:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| eidetic-product:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| eidetic-product:locomo:runs | FAIL | 0 runs: [] |
| eidetic-product:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:locomo:single-hop:runs | FAIL | 0 runs: [] |
| eidetic-product:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| eidetic-product:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:locomo:temporal:runs | FAIL | 0 runs: [] |
| eidetic-product:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-product:locomo:open-domain:runs | FAIL | 0 runs: [] |
| rag-full:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| rag-full:runs | FAIL | 0 runs: [] |
| rag-full:datasets | FAIL | missing: longmemeval, locomo |
| rag-full:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| rag-full:longmemeval:runs | FAIL | 0 runs: [] |
| rag-full:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| rag-full:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| rag-full:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| rag-full:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| rag-full:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| rag-full:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| rag-full:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| rag-full:locomo:runs | FAIL | 0 runs: [] |
| rag-full:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:locomo:single-hop:runs | FAIL | 0 runs: [] |
| rag-full:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| rag-full:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:locomo:temporal:runs | FAIL | 0 runs: [] |
| rag-full:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-full:locomo:open-domain:runs | FAIL | 0 runs: [] |
| rag-vector:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| rag-vector:runs | FAIL | 0 runs: [] |
| rag-vector:datasets | FAIL | missing: longmemeval, locomo |
| rag-vector:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| rag-vector:longmemeval:runs | FAIL | 0 runs: [] |
| rag-vector:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| rag-vector:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| rag-vector:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| rag-vector:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| rag-vector:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| rag-vector:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| rag-vector:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| rag-vector:locomo:runs | FAIL | 0 runs: [] |
| rag-vector:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:locomo:single-hop:runs | FAIL | 0 runs: [] |
| rag-vector:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| rag-vector:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:locomo:temporal:runs | FAIL | 0 runs: [] |
| rag-vector:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| rag-vector:locomo:open-domain:runs | FAIL | 0 runs: [] |
| mem0:questions | FAIL | 40 unique samples, 40 rows (required >= 1000 unique samples) |
| mem0:runs | FAIL | 1 runs: [0] |
| mem0:datasets | FAIL | missing: longmemeval |
| mem0:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| mem0:longmemeval:runs | FAIL | 0 runs: [] |
| mem0:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| mem0:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| mem0:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| mem0:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| mem0:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| mem0:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| mem0:locomo:questions | FAIL | 40 unique samples, 40 rows (required >= 300 unique samples) |
| mem0:locomo:runs | FAIL | 1 runs: [0] |
| mem0:locomo:single-hop:questions | PASS | 22 unique samples, 22 rows (required >= 20) |
| mem0:locomo:single-hop:runs | FAIL | 1 runs: [0] |
| mem0:locomo:multi-hop:questions | FAIL | 7 unique samples, 7 rows (required >= 20) |
| mem0:locomo:multi-hop:runs | FAIL | 1 runs: [0] |
| mem0:locomo:temporal:questions | FAIL | 9 unique samples, 9 rows (required >= 20) |
| mem0:locomo:temporal:runs | FAIL | 1 runs: [0] |
| mem0:locomo:open-domain:questions | FAIL | 2 unique samples, 2 rows (required >= 20) |
| mem0:locomo:open-domain:runs | FAIL | 1 runs: [0] |
| graphiti:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| graphiti:runs | FAIL | 0 runs: [] |
| graphiti:datasets | FAIL | missing: longmemeval, locomo |
| graphiti:longmemeval:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| graphiti:longmemeval:runs | FAIL | 0 runs: [] |
| graphiti:longmemeval:single-session-user:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:longmemeval:single-session-user:runs | FAIL | 0 runs: [] |
| graphiti:longmemeval:single-session-assistant:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:longmemeval:single-session-assistant:runs | FAIL | 0 runs: [] |
| graphiti:longmemeval:single-session-preference:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:longmemeval:single-session-preference:runs | FAIL | 0 runs: [] |
| graphiti:longmemeval:multi-session:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:longmemeval:multi-session:runs | FAIL | 0 runs: [] |
| graphiti:longmemeval:knowledge-update:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:longmemeval:knowledge-update:runs | FAIL | 0 runs: [] |
| graphiti:longmemeval:temporal-reasoning:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:longmemeval:temporal-reasoning:runs | FAIL | 0 runs: [] |
| graphiti:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| graphiti:locomo:runs | FAIL | 0 runs: [] |
| graphiti:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:locomo:single-hop:runs | FAIL | 0 runs: [] |
| graphiti:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| graphiti:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:locomo:temporal:runs | FAIL | 0 runs: [] |
| graphiti:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| graphiti:locomo:open-domain:runs | FAIL | 0 runs: [] |
| competitor_health:mem0 | PASS | rows=13, missing=0, bad=0 |
| competitor_health:graphiti | FAIL | rows=0, missing=0, bad=0 |
| eidetic-plus:longmemeval:accuracy | FAIL | 0.0% (required >= 85.0%) |
| evidence:eidetic-plus:longmemeval:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 10.0pp) |
| evidence:eidetic-plus:longmemeval/single-session-user:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:longmemeval/single-session-assistant:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:longmemeval/single-session-preference:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:longmemeval/multi-session:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:longmemeval/knowledge-update:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:longmemeval/temporal-reasoning:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| eidetic-plus:locomo:accuracy | FAIL | 0.0% (required >= 85.0%) |
| evidence:eidetic-plus:locomo:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 10.0pp) |
| evidence:eidetic-plus:locomo/single-hop:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:locomo/multi-hop:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:locomo/temporal:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| evidence:eidetic-plus:locomo/open-domain:sample_clustered_accuracy_ci_width | FAIL | sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 30.0pp) |
| operating:eidetic-plus:query_tokens_median | FAIL | n/a (allowed <= 7000) |
| operating:eidetic-plus:search_p95_ms | FAIL | n/a (allowed <= 500.0) |
| operating:eidetic-plus:e2e_p50_ms | FAIL | n/a (allowed <= 5000.0) |
| operating:eidetic-plus:token_efficiency_vs:rag-full | FAIL | n/ax (required >= 10.0x) |
| operating:eidetic-plus:age_slope_samples | FAIL | n=0, distinct_ages=0 (required n >= 20, distinct >= 2) |
| operating:eidetic-plus:age_flatness | FAIL | n/a per year (allowed abs <= 0.100) |
| dominance:eidetic-plus:vs:rag-full:paired | FAIL | paired_n=0, unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:rag-full:significance | FAIL | p=n/a (required < 0.05) |
| dominance:eidetic-plus:vs:rag-full:sample_clustered_paired | FAIL | sample_n=0 (required >= 1000), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:sample_clustered_delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:rag-full:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-user | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-user:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-user:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-user:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-user:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-user:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-assistant | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-assistant:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-assistant:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-assistant:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-assistant:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-assistant:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-preference | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-preference:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-preference:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-preference:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-preference:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/single-session-preference:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/multi-session | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/multi-session:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:longmemeval/multi-session:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/multi-session:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:longmemeval/multi-session:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/multi-session:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/knowledge-update | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/knowledge-update:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:longmemeval/knowledge-update:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/knowledge-update:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:longmemeval/knowledge-update:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/knowledge-update:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/temporal-reasoning | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/temporal-reasoning:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:longmemeval/temporal-reasoning:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/temporal-reasoning:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:longmemeval/temporal-reasoning:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:longmemeval/temporal-reasoning:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:locomo/single-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:locomo/multi-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/multi-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:locomo/multi-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/multi-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:locomo/multi-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:locomo/multi-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:locomo/temporal | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/temporal:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:locomo/temporal:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/temporal:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:locomo/temporal:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:locomo/temporal:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-full:locomo/open-domain | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/open-domain:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-full:locomo/open-domain:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-full:locomo/open-domain:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-full:locomo/open-domain:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-full:locomo/open-domain:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:paired | FAIL | paired_n=0, unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:rag-vector:significance | FAIL | p=n/a (required < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:sample_clustered_paired | FAIL | sample_n=0 (required >= 1000), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:sample_clustered_delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:rag-vector:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-user | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-user:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-user:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-user:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-user:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-user:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-assistant | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-assistant:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-assistant:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-assistant:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-assistant:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-assistant:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-preference | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-preference:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-preference:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-preference:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-preference:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/single-session-preference:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/multi-session | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/multi-session:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/multi-session:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/multi-session:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/multi-session:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/multi-session:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/knowledge-update | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/knowledge-update:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/knowledge-update:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/knowledge-update:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/knowledge-update:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/knowledge-update:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/temporal-reasoning | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/temporal-reasoning:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/temporal-reasoning:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/temporal-reasoning:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/temporal-reasoning:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:longmemeval/temporal-reasoning:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:locomo/single-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/single-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:locomo/single-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/single-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:locomo/single-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:locomo/single-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:locomo/multi-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/multi-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:locomo/multi-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/multi-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:locomo/multi-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:locomo/multi-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:locomo/temporal | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/temporal:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:locomo/temporal:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/temporal:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:locomo/temporal:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:locomo/temporal:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:rag-vector:locomo/open-domain | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/open-domain:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:rag-vector:locomo/open-domain:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:rag-vector:locomo/open-domain:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:rag-vector:locomo/open-domain:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:rag-vector:locomo/open-domain:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:paired | FAIL | paired_n=0, unpaired=13 |
| dominance:eidetic-plus:vs:mem0:delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:mem0:significance | FAIL | p=n/a (required < 0.05) |
| dominance:eidetic-plus:vs:mem0:sample_clustered_paired | FAIL | sample_n=0 (required >= 1000), unpaired=13 |
| dominance:eidetic-plus:vs:mem0:sample_clustered_delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:mem0:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=13, acc=69.2%, Wilson 42.4-87.3 (width 44.9pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-user | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-user:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-user:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-user:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-user:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-user:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-assistant | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-assistant:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-assistant:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-assistant:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-assistant:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-assistant:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-preference | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-preference:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-preference:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-preference:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-preference:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:longmemeval/single-session-preference:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:longmemeval/multi-session | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/multi-session:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:longmemeval/multi-session:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/multi-session:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:longmemeval/multi-session:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:longmemeval/multi-session:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:longmemeval/knowledge-update | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/knowledge-update:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:longmemeval/knowledge-update:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/knowledge-update:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:longmemeval/knowledge-update:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:longmemeval/knowledge-update:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:longmemeval/temporal-reasoning | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/temporal-reasoning:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:longmemeval/temporal-reasoning:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:longmemeval/temporal-reasoning:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:longmemeval/temporal-reasoning:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:longmemeval/temporal-reasoning:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=8 |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=8, acc=87.5%, Wilson 52.9-97.8 (width 44.8pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=2 |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=2, acc=50.0%, Wilson 9.5-90.5 (width 81.1pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=2 |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=2, acc=0.0%, Wilson 0.0-65.8 (width 65.8pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=1 |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=1, acc=100.0%, Wilson 20.7-100.0 (width 79.3pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:paired | FAIL | paired_n=0, unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:graphiti:significance | FAIL | p=n/a (required < 0.05) |
| dominance:eidetic-plus:vs:graphiti:sample_clustered_paired | FAIL | sample_n=0 (required >= 1000), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:sample_clustered_delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:graphiti:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-user | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-user:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-user:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-user:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-user:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-user:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-assistant | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-assistant:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-assistant:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-assistant:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-assistant:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-assistant:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-preference | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-preference:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-preference:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-preference:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-preference:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/single-session-preference:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/multi-session | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/multi-session:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:longmemeval/multi-session:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/multi-session:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:longmemeval/multi-session:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/multi-session:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/knowledge-update | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/knowledge-update:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:longmemeval/knowledge-update:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/knowledge-update:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:longmemeval/knowledge-update:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/knowledge-update:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/temporal-reasoning | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/temporal-reasoning:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:longmemeval/temporal-reasoning:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/temporal-reasoning:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:longmemeval/temporal-reasoning:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:longmemeval/temporal-reasoning:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:locomo/single-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/single-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:locomo/single-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/single-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:locomo/single-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:locomo/single-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:locomo/multi-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/multi-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:locomo/multi-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/multi-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:locomo/multi-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:locomo/multi-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:locomo/temporal | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/temporal:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:locomo/temporal:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/temporal:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:locomo/temporal:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:locomo/temporal:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:graphiti:locomo/open-domain | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/open-domain:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:graphiti:locomo/open-domain:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:graphiti:locomo/open-domain:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:graphiti:locomo/open-domain:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:graphiti:locomo/open-domain:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| integrity:eidetic-plus-full:verify_step | PASS | has_verify=True, n=40 |
| integrity:eidetic-plus-full:verified_accuracy | PASS | 57.5% (required >= 50.0%) |
| integrity:eidetic-plus-full:proof_support | PASS | 36/36 verified rows carry proof support |
| eidetic-plus:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| eidetic-plus:consolidation_deferred | PASS | 0 (allowed <= 0) |
| eidetic-plus-full:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| eidetic-plus-full:consolidation_deferred | PASS | 0 (allowed <= 0) |
| eidetic-product:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| eidetic-product:consolidation_deferred | PASS | 0 (allowed <= 0) |
| rag-full:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| rag-full:consolidation_deferred | PASS | 0 (allowed <= 0) |
| rag-vector:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| rag-vector:consolidation_deferred | PASS | 0 (allowed <= 0) |
| mem0:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| mem0:consolidation_deferred | PASS | 0 (allowed <= 0) |
| graphiti:consolidation_timeouts | PASS | 0 (allowed <= 0) |
| graphiti:consolidation_deferred | PASS | 0 (allowed <= 0) |
| snap_back:valid_json | PASS | valid |
| snap_back:records | PASS | 272 (required >= 1) |
| snap_back:lossless | PASS | 272/272, rate=1.000000 |
| snap_back:no_failures | PASS | 0 failures |
| snap_back:audited_hashes_present | PASS | 272 audited hash(es) |
| snap_back:covers_verified_proof_hashes | PASS | 51 proof hash(es) covered |
| snap_back:data_dir_matches_manifest | PASS | /Users/raunakgengiti/Eidetic-Plus/artifacts/holdout_rotation_r8_codex/data (expected /Users/raunakgengiti/Eidetic-Plus/artifacts/holdout_rotation_r8_codex/data) |
| baseline_reproduction:valid_json | FAIL | missing |
| baseline_reproduction:status | FAIL | <missing> |
| baseline_reproduction:system | FAIL | <missing> (expected mem0) |
| baseline_reproduction:dataset | FAIL | <missing> (expected locomo) |
| baseline_reproduction:rows | FAIL | total_n=0 |
| baseline_reproduction:comparisons | PASS | all PASS |
| baseline_reproduction:log_fingerprint_present | FAIL | <missing> |
| baseline_reproduction:log_fingerprint_matches | FAIL | <missing> (current ae231e635c2e6926fe0a0c69fb7648031137982f0986f85a9187eefa4145cb5f (2 files)) |

## Evidence Strength

| slice | sample n | accuracy | Wilson low | Wilson high | width pp |
|---|---:|---:|---:|---:|---:|
| locomo|* | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| locomo|multi-hop | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| locomo|open-domain | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| locomo|single-hop | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| locomo|temporal | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|* | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|knowledge-update | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|multi-session | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|single-session-assistant | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|single-session-preference | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|single-session-user | 0 | 0.0% | 0.0% | 0.0% | 0.0 |
| longmemeval|temporal-reasoning | 0 | 0.0% | 0.0% | 0.0% | 0.0 |

## Paired Dominance

| baseline | paired n | delta pp | McNemar p | sample n | sample discordants | sample McNemar p | CI-clear |
|---|---:|---:|---:|---:|---:|---:|---|
| graphiti | 0 | 0.0 | - | 0 | 0 | - | no |
| mem0 | 0 | 0.0 | - | 0 | 0 | - | no |
| rag-full | 0 | 0.0 | - | 0 | 0 | - | no |
| rag-vector | 0 | 0.0 | - | 0 | 0 | - | no |

## Category Clustered Dominance

| baseline | category | row n | row delta pp | sample n | sample delta pp | sample discordants | sample McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| graphiti | locomo|multi-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | locomo|open-domain | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | locomo|single-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | locomo|temporal | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | longmemeval|knowledge-update | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | longmemeval|multi-session | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | longmemeval|single-session-assistant | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | longmemeval|single-session-preference | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | longmemeval|single-session-user | 0 | 0.0 | 0 | 0.0 | 0 | - |
| graphiti | longmemeval|temporal-reasoning | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | locomo|multi-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | locomo|open-domain | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | locomo|single-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | locomo|temporal | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | longmemeval|knowledge-update | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | longmemeval|multi-session | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | longmemeval|single-session-assistant | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | longmemeval|single-session-preference | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | longmemeval|single-session-user | 0 | 0.0 | 0 | 0.0 | 0 | - |
| mem0 | longmemeval|temporal-reasoning | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | locomo|multi-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | locomo|open-domain | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | locomo|single-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | locomo|temporal | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | longmemeval|knowledge-update | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | longmemeval|multi-session | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | longmemeval|single-session-assistant | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | longmemeval|single-session-preference | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | longmemeval|single-session-user | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-full | longmemeval|temporal-reasoning | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | locomo|multi-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | locomo|open-domain | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | locomo|single-hop | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | locomo|temporal | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | longmemeval|knowledge-update | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | longmemeval|multi-session | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | longmemeval|single-session-assistant | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | longmemeval|single-session-preference | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | longmemeval|single-session-user | 0 | 0.0 | 0 | 0.0 | 0 | - |
| rag-vector | longmemeval|temporal-reasoning | 0 | 0.0 | 0 | 0.0 | 0 | - |

## Operating Point

| system | n | median query tokens | search p95 ms | e2e p50 ms |
|---|---:|---:|---:|---:|
| eidetic-plus | 0 | - | - | - |
| rag-full | 0 | - | - | - |

Recall-vs-age slope for headline row: `n/a` per year.

## Claim Scope

Public claim scope: `limited`
Measured external systems: `-`

## Ablation Evidence

System: `eidetic-plus-full`  Split: `dev`  n: `26`  Evidence refs: `5`
Metabolism delta: `18.9853` pp  Regions delta: `-4.1758` pp  Affect delta: `9.5385` pp  Forgetting cost ratio: `0.9824`  Accuracy regression: `3.9787` pp

## Abstention Calibration

Method: `abstention_v2_tau`  Split: `dev`  System: `eidetic-plus-full`  n: `264`  tau: `1.000000001`
Calibration log fingerprint: `3f0c24806df6973eff9bff291ade279edaef8d7392cb47affc95bbf9f2b9eb62`

## Baseline Reproduction

_No baseline reproduction report loaded._

## Snap-Back Fidelity

| records with raw blob | lossless | rate | failures | data dir |
|---:|---:|---:|---:|---|
| 272 | 272 | 100.0000% | 0 | /Users/raunakgengiti/Eidetic-Plus/artifacts/holdout_rotation_r8_codex/data |

## Consolidation Health

| system | groups | timed out | deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 10 | 0 | 0 | 253 | 253 | 0 | 0 | 0 | 0 | 0 |
