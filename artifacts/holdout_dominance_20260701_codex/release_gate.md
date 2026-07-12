# Public Release Gate

Status: **FAIL**
Artifact directory: `artifacts/holdout_dominance_20260701_codex`
Log fingerprint: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 files)

## Checks

| check | status | detail |
|---|---|---|
| artifact:run_manifest.json | PASS | present |
| artifact:scoreboard.md | PASS | present |
| artifact:scoreboard.json | PASS | present |
| artifact:recall_vs_age.png | FAIL | missing |
| artifact:latency_vs_age.png | FAIL | missing |
| artifact:snap_back_audit.json | FAIL | missing |
| logs:fingerprint_stable | PASS | before e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 files); after e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 files) |
| scoreboard:valid_json | PASS | valid |
| scoreboard:log_fingerprint_present | PASS | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 files) |
| scoreboard:log_fingerprint_matches | PASS | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 files) (current e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 files)) |
| claim_scope:valid_json | PASS | valid |
| claim_scope:scope_declared | PASS | limited |
| claim_scope:harness_names_have_logs | PASS | all harness names have logs |
| claim_scope:external_evidence_valid | PASS | not a SOTA claim |
| claim_scope:external_names_have_evidence | PASS | all external names have evidence |
| claim_scope:no_unsupported_sota | PASS | not a SOTA claim |
| claim_scope:top_system_dataset_coverage | PASS | not a SOTA claim |
| claim_scope:top_system_score_floor | PASS | not a SOTA claim |
| claim_scope:limitations_for_limited_claim | PASS | 5 limitations |
| manifest:split | FAIL | dev (expected test) |
| manifest:runs | FAIL | 0 (required >= 10) |
| manifest:not_render_only | FAIL | render_only=True |
| manifest:systems_cover_required | FAIL | missing: eidetic-plus, eidetic-product, rag-full, rag-vector, mem0, graphiti |
| manifest:sample_rows_present | FAIL | 0 sample rows |
| manifest:longmemeval:categories_cover_required | FAIL | missing: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning |
| manifest:longmemeval:single-session-user:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:single-session-assistant:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:single-session-preference:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:multi-session:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:knowledge-update:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:longmemeval:temporal-reasoning:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:locomo:categories_cover_required | FAIL | missing: single-hop, multi-hop, temporal, open-domain |
| manifest:locomo:single-hop:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:locomo:multi-hop:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:locomo:temporal:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:locomo:open-domain:sample_rows | FAIL | 0 unique sample rows (required >= 20) |
| manifest:no_system_failures | PASS | none |
| manifest:data_dir_recorded | FAIL | <unset> |
| manifest:no_dataset_source_scans | PASS | disabled |
| manifest:session_ingest_granularity | PASS | session |
| manifest:holdout_profile | FAIL | not_run |
| manifest:samples_file_recorded | FAIL | <missing> |
| holdout_audit:valid_json | PASS | valid |
| holdout_audit:evidence | PASS | 1639 holdout needles, 0 findings |
| ablation:valid_json | PASS | valid |
| ablation:evidence | FAIL | pass:FAIL; metabolism_delta_pp:0.00<required:5.00; region_delta_pp:-10.00<required:2.00; forgetting_accuracy_regression_pp:10.00>allowed:1.00; forgetting_cost_ratio:1.000<required:1.050 |
| affect_salience:valid_json | FAIL | missing |
| affect_salience:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; flip_checks:0<expected:0; age_free_checks:0<expected:0; bounded_checks:0<expected:0 |
| scratchpad:valid_json | FAIL | missing |
| scratchpad:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; ordering_checks:0<expected:0; active_scope_filter_checks:0<expected:0; proof_link_checks:0<expected:0 |
| region_routing:valid_json | FAIL | missing |
| region_routing:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; dense_miss_recovery_checks:0<expected:0; active_scope_filter_checks:0<expected:0; nested_cocoon_checks:0<expected:0 |
| reflex_recall:valid_json | FAIL | missing |
| reflex_recall:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; direct_hit_checks:0<expected:0; coactivation_checks:0<expected:0; active_scope_filter_checks:0<expected:0 |
| slice_invariant:valid_json | PASS | valid |
| slice_invariant:pass | FAIL | False |
| slice_invariant:evidence | FAIL | missing:locomo,longmemeval |
| smqe_planner:valid_json | PASS | valid |
| smqe_planner:evidence | PASS | 162/162 planner checks, p95 0.344459 ms |
| smqe_synthetic:valid_json | PASS | valid |
| smqe_synthetic:evidence | FAIL | ops_below_2:relative_temporal,speaker_fact,table_lookup |
| smqe_claim_coverage:valid_json | PASS | valid |
| smqe_claim_coverage:evidence | FAIL | ops_below_2:relative_temporal,speaker_fact,table_lookup |
| smqe_fullpath:valid_json | FAIL | missing |
| smqe_fullpath:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; correct:0/0; ops_below_2:count_aggregate,latest_value,multi_session_sum,open_inference,preference_synth,relative_temporal,speaker_fact,table_lookup,temporal_delta; backend:claim; avg_proof_tokens:inf>80.0; verified:0/0 |
| smqe_paraphrase:valid_json | FAIL | missing |
| smqe_paraphrase:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; correct:0/0; ops_below_2:count_aggregate,latest_value,multi_session_sum,open_inference,preference_synth,relative_temporal,speaker_fact,table_lookup,temporal_delta; backend:claim; backend:record; avg_proof_tokens:inf>80.0 |
| smqe_conflict:valid_json | FAIL | missing |
| smqe_conflict:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; types_below_2:amount,location,status; avg_proof_tokens:inf>80.0 |
| smqe_composition:valid_json | FAIL | missing |
| smqe_composition:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; types_below_2:event_order,relative_event_time,shared_value |
| smqe_relative_phrase:valid_json | FAIL | missing |
| smqe_relative_phrase:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; types_below_2:ago_days,ago_weeks,fortnight_ago,in_days,next_month,next_week |
| smqe_temporal_window:valid_json | FAIL | missing |
| smqe_temporal_window:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; types_below_2:fortnight_count,most_recent_latest,past_days_count,past_few_months_count,past_week_count,past_week_list,recent_count,recent_hours_sum,recent_list,source_action_variant_window,source_location_window |
| smqe_attribution:valid_json | FAIL | missing |
| smqe_attribution:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; types_below_2:gave_actor,recommend_actor,shared_actor,told_actor |
| smqe_abstention:valid_json | FAIL | missing |
| smqe_abstention:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; abstained:0/0; record_only_abstained:0/0; claims_present_abstained:0/0; types_below_2:count_neutral_quantity,count_target_mismatch,latest_future_only,latest_missing_subject,preference_no_positive,speaker_crossed_support,table_missing_row,temporal_missing_anchor |
| smqe_scope:valid_json | FAIL | missing |
| smqe_scope:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; ops_below_2:count_aggregate,latest_value,multi_session_sum,preference_synth,relative_temporal,speaker_fact,table_lookup,temporal_delta |
| smqe_subscope:valid_json | FAIL | missing |
| smqe_subscope:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; ops_below_2:count_aggregate,latest_value,multi_session_sum,preference_synth,relative_temporal,speaker_fact,table_lookup,temporal_delta |
| smqe_time:valid_json | FAIL | missing |
| smqe_time:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; ops_below_2:count_aggregate,latest_value,multi_session_sum,preference_synth,relative_temporal,speaker_fact,table_lookup,temporal_delta |
| smqe_invalidation:valid_json | FAIL | missing |
| smqe_invalidation:evidence | FAIL | seed_mode:<missing>; pass:false; cases:0<required:24; checks:0/expected:0; correct:0/0; record_backend_correct:0/0; claim_backend_correct:0/0; ops_below_2:count_aggregate,latest_value,multi_session_sum,preference_synth,relative_temporal,speaker_fact,table_lookup,temporal_delta |
| logs:nonempty | FAIL | 0 rows |
| logs:no_error_rows | PASS | 0 error rows |
| smqe:notes_clean | PASS | no legacy structured-recall policies |
| smqe:log_policy_shape | FAIL | system:eidetic-plus-full:no_rows; structured_rate:0.000<required:0.800; claim_backend_rate:0.000<required:0.800 |
| region:telemetry | FAIL | system:eidetic-plus-full:no_rows |
| logs:match_manifest_sample_rows | FAIL | expected=0, actual=0, missing=0, extra=0 |
| logs:held_out_split | PASS | 0 rows outside test split |
| systems:required_present | FAIL | missing: eidetic-plus, eidetic-plus-full, eidetic-product, rag-full, rag-vector, mem0, graphiti |
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
| eidetic-plus-full:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| eidetic-plus-full:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:datasets | FAIL | missing: longmemeval, locomo |
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
| eidetic-plus-full:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| eidetic-plus-full:locomo:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:locomo:single-hop:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:locomo:temporal:runs | FAIL | 0 runs: [] |
| eidetic-plus-full:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| eidetic-plus-full:locomo:open-domain:runs | FAIL | 0 runs: [] |
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
| mem0:questions | FAIL | 0 unique samples, 0 rows (required >= 1000 unique samples) |
| mem0:runs | FAIL | 0 runs: [] |
| mem0:datasets | FAIL | missing: longmemeval, locomo |
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
| mem0:locomo:questions | FAIL | 0 unique samples, 0 rows (required >= 300 unique samples) |
| mem0:locomo:runs | FAIL | 0 runs: [] |
| mem0:locomo:single-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:locomo:single-hop:runs | FAIL | 0 runs: [] |
| mem0:locomo:multi-hop:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:locomo:multi-hop:runs | FAIL | 0 runs: [] |
| mem0:locomo:temporal:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:locomo:temporal:runs | FAIL | 0 runs: [] |
| mem0:locomo:open-domain:questions | FAIL | 0 unique samples, 0 rows (required >= 20) |
| mem0:locomo:open-domain:runs | FAIL | 0 runs: [] |
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
| competitor_health:mem0 | FAIL | rows=0, missing=0, bad=0 |
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
| dominance:eidetic-plus:vs:mem0:paired | FAIL | paired_n=0, unpaired=0 |
| dominance:eidetic-plus:vs:mem0:delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:mem0:significance | FAIL | p=n/a (required < 0.05) |
| dominance:eidetic-plus:vs:mem0:sample_clustered_paired | FAIL | sample_n=0 (required >= 1000), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:sample_clustered_delta | FAIL | 0.0pp (required >= 10.0pp) |
| dominance:eidetic-plus:vs:mem0:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
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
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/single-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/multi-hop:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_discordants | FAIL | 0 discordant samples (required >= 6) |
| dominance:eidetic-plus:vs:mem0:locomo/temporal:sample_clustered_significance | FAIL | p=n/a, discordant=0 (required p < 0.05) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain | FAIL | n=0, delta=0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_paired | FAIL | sample_n=0 (required >= 20), unpaired=0 |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_delta | PASS | 0.0pp (required >= 0.0pp) |
| dominance:eidetic-plus:vs:mem0:locomo/open-domain:sample_clustered_ci_clear | FAIL | headline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); baseline sample_n=0, acc=0.0%, Wilson 0.0-0.0 (width 0.0pp; allowed <= 100.0pp); need headline lower CI > baseline upper CI |
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
| integrity:eidetic-plus-full:verify_step | FAIL | has_verify=None, n=0 |
| integrity:eidetic-plus-full:verified_accuracy | FAIL | 0.0% (required >= 50.0%) |
| integrity:eidetic-plus-full:proof_support | FAIL | system:eidetic-plus-full:no_rows; verified_rows:0 |
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
| snap_back:valid_json | FAIL | missing |
| snap_back:records | FAIL | 0 (required >= 1) |
| snap_back:lossless | FAIL | 0/0, rate=0.000000 |
| snap_back:no_failures | PASS | 0 failures |
| snap_back:audited_hashes_present | FAIL | 0 audited hash(es) |
| snap_back:covers_verified_proof_hashes | FAIL | missing=0, proof_hashes=0 |
| baseline_reproduction:valid_json | FAIL | missing |
| baseline_reproduction:status | FAIL | <missing> |
| baseline_reproduction:system | FAIL | <missing> (expected mem0) |
| baseline_reproduction:dataset | FAIL | <missing> (expected locomo) |
| baseline_reproduction:rows | FAIL | total_n=0 |
| baseline_reproduction:comparisons | PASS | all PASS |
| baseline_reproduction:log_fingerprint_present | FAIL | <missing> |
| baseline_reproduction:log_fingerprint_matches | FAIL | <missing> (current e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 files)) |

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

System: `eidetic-plus-full`  Split: `dev`  n: `20`  Evidence refs: `5`
Metabolism delta: `0.0` pp  Regions delta: `-10.0` pp  Affect delta: `15.0` pp  Forgetting cost ratio: `1.0001`  Accuracy regression: `10.0` pp

## Abstention Calibration

_No abstention calibration report loaded._

## Baseline Reproduction

_No baseline reproduction report loaded._

## Snap-Back Fidelity

| records with raw blob | lossless | rate | failures | data dir |
|---:|---:|---:|---:|---|
| 0 | 0 | 0.0000% | 0 |  |

## Consolidation Health

| system | groups | timed out | deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
