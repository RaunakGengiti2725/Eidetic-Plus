"""Offline tests for the public-release benchmark gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bench.datasets import split_of
from bench.fingerprints import log_fingerprint
from bench.release_gate import (
    _paired_stats,
    _sample_clustered_accuracy,
    _sample_clustered_paired_stats,
    _unique_sample_count,
    render_markdown,
    run_release_gate,
)


def _ids(n: int, *, split: str = "test") -> list[str]:
    out: list[str] = []
    i = 0
    while len(out) < n:
        sid = f"release_{split}_{i}_q0"
        if split_of(sid) == split:
            out.append(sid)
        i += 1
    return out


def _affect_salience_report(*, cases: int = 24) -> dict:
    return {
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": cases,
        "checks": cases * 7,
        "correct": cases * 7,
        "flip_checks": cases * 2,
        "age_free_checks": cases,
        "bounded_checks": cases,
        "lambda_salience": 0.5,
        "max_boost_ratio": 0.49,
        "min_age_gap_seconds": 90_000_000.0,
        "case_type_counts": {"affect_salience_retrieval": cases},
        "failures": [],
    }


def _scratchpad_report(*, cases: int = 24) -> dict:
    return {
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": cases,
        "checks": cases * 11,
        "correct": cases * 11,
        "ordering_checks": cases,
        "active_scope_filter_checks": cases,
        "proof_link_checks": cases * 4,
        "top_k_checks": cases,
        "retrieval_channel_checks": cases * 4,
        "case_type_counts": {"scratchpad_active_proof_surface": cases},
        "failures": [],
    }


def _region_routing_report(*, cases: int = 24) -> dict:
    return {
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": cases,
        "checks": cases * 12,
        "correct": cases * 12,
        "dense_miss_recovery_checks": cases * 3,
        "active_scope_filter_checks": cases * 2,
        "nested_cocoon_checks": cases * 2,
        "proof_link_checks": cases * 2,
        "telemetry_trace_checks": cases * 2,
        "route_only_context_checks": cases,
        "case_type_counts": {"region_routing_cocoon_proof": cases},
        "failures": [],
    }


def _reflex_recall_report(*, cases: int = 24) -> dict:
    return {
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": cases,
        "checks": cases * 12,
        "correct": cases * 12,
        "direct_hit_checks": cases * 2,
        "coactivation_checks": cases * 2,
        "active_scope_filter_checks": cases * 4,
        "proof_link_checks": cases,
        "score_contract_checks": cases * 2,
        "latency_budget_checks": cases,
        "max_latency_ms": 2.5,
        "p95_latency_ms": 2.0,
        "latency_budget_ms": 100,
        "case_type_counts": {"reflex_recall_proof_surface": cases},
        "failures": [],
    }


def _smqe_planner_report(*, cases: int = 24) -> dict:
    return {
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": cases,
        "checks": cases * 6,
        "correct": cases * 6,
        "generic_term_checks": cases,
        "operator_counts": {
            "count_aggregate": 3,
            "event_order": 2,
            "latest_value": 3,
            "multi_session_sum": 2,
            "open_inference": 2,
            "preference_synth": 2,
            "relative_temporal": 2,
            "speaker_fact": 2,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "case_type_counts": {"smqe_planner_generic_shape": cases},
        "p95_latency_ms": 1.0,
        "max_latency_ms": 2.0,
        "failures": [],
    }


def _write_artifacts(path: Path, *, split: str = "test", runs: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data_dir = (path / "data").resolve()
    (path / "scoreboard.md").write_text("# scoreboard\n")
    (path / "scoreboard.json").write_text("{}")
    (path / "recall_vs_age.png").write_bytes(b"png")
    (path / "latency_vs_age.png").write_bytes(b"png")
    (path / "snap_back_audit.json").write_text(json.dumps({
        "status": "PASS",
        "data_dir": str(data_dir),
        "records_with_raw_blob": 8,
        "lossless_byte_identical": 8,
        "rate": 1.0,
        "rate_pct": 100.0,
        "min_records": 1,
        "failures": [],
    }))
    (path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "measured-harness-only",
        "measured_harness_systems": ["eidetic-plus", "eidetic-plus-full", "rag-full"],
        "measured_external_systems": [],
        "limitations": ["Synthetic fixture for release-gate tests."],
    }))
    (path / "holdout_audit.json").write_text(json.dumps({
        "pass": True,
        "findings": [],
        "needles_checked": 12,
        "holdout_needles_checked": 8,
        "legacy_policy_scan_enabled": True,
        "forbidden_policy_strings_checked": 5,
        "forbidden_fixed_answer_strings_checked": 10,
        "forbidden_runtime_symbols_checked": 24,
        "scan_roots": ["eidetic", "bench", "tests", "docs"],
        "registry_error": "",
    }))
    (path / "ablation_report.json").write_text(json.dumps({
        "pass": True,
        "system": "eidetic-plus-full",
        "split": "dev",
        "full": {
            "n": 24,
            "verified_accuracy": 0.90,
            "query_tokens_median": 100,
        },
        "ablations": {
            "metabolism_off": {
                "n": 24,
                "verified_accuracy": 0.80,
                "query_tokens_median": 100,
            },
            "regions_off": {
                "n": 24,
                "verified_accuracy": 0.84,
                "query_tokens_median": 100,
            },
            "forgetting_off": {
                "n": 24,
                "verified_accuracy": 0.90,
                "query_tokens_median": 130,
            },
            "affect_off": {
                "n": 24,
                "verified_accuracy": 0.86,
                "query_tokens_median": 100,
            },
        },
        "artifact_fingerprints": [{"combined_sha256": "unit-ablation"}],
    }))
    (path / "affect_salience_invariant.json").write_text(json.dumps(_affect_salience_report()))
    (path / "scratchpad_invariant.json").write_text(json.dumps(_scratchpad_report()))
    (path / "region_routing_invariant.json").write_text(json.dumps(_region_routing_report()))
    (path / "reflex_recall_invariant.json").write_text(json.dumps(_reflex_recall_report()))
    (path / "smqe_planner_invariant.json").write_text(json.dumps(_smqe_planner_report()))
    (path / "mem0_gate.json").write_text(json.dumps({
        "status": "PASS",
        "reason": "within tolerance",
        "system": "mem0",
        "dataset": "locomo",
        "tolerance": 0.02,
        "min_n": 8,
        "total_n": 8,
        "expected": {"single-hop": 0.5},
        "observed": {"single-hop": {"n": 8, "accuracy": 0.5, "runs": [0, 1]}},
        "comparisons": {
            "single-hop": {
                "status": "PASS",
                "n": 8,
                "runs": [0, 1],
                "observed": 0.5,
                "expected": 0.5,
                "delta": 0.0,
            },
        },
    }))
    sample_rows = [
        {"dataset": "locomo", "sample_id": sid, "category": "single-hop"}
        for sid in _ids(4, split=split)
    ]
    samples_file = path / "release.samples.json"
    samples_file.write_text(json.dumps([
        {"dataset": row["dataset"], "sample_id": row["sample_id"]}
        for row in sample_rows
    ]))
    slice_ids = _ids(20, split="test")
    (path / "slice_invariant.json").write_text(json.dumps({
        "pass": True,
        "dataset": "locomo",
        "split": "test",
        "holdout_profile": "holdout",
        "draws": 5,
        "subset": 4,
        "seed": 424242,
        "seed_mode": "random",
        "draw_seeds": [424242 + draw for draw in range(5)],
        "pool_unique_sample_ids": len(slice_ids),
        "required_unique_sample_ids": 20,
        "unique_sample_ids": len(slice_ids),
        "system_under_test": "eidetic-plus-full",
        "runs": [
            {
                "draw": draw + 1,
                "seed": 424242 + draw,
                "sample_ids": slice_ids[draw * 4:(draw + 1) * 4],
                "executed": True,
                "returncode": 0,
                "score": {
                    "pass": True,
                    "verified": True,
                    "verified_correct": 4,
                    "correct": 4,
                    "total": 4,
                },
            }
            for draw in range(5)
        ],
    }))
    (path / "smqe_synthetic_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 27,
        "correct": 27,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 3, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_claim_coverage.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 27,
        "correct": 27,
        "claim_backend_correct": 27,
        "claims_extracted": 86,
        "avg_claims_per_case": 3.0,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "claim_backend_operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "claim_type_counts": {"event": 26, "preference": 9, "quantity": 9, "state": 33, "table": 9},
        "backend_counts": {"claim": 27},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_fullpath_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 27,
        "correct": 27,
        "verified": 27,
        "structured_recall": 27,
        "reader_calls": 0,
        "proof_link_checks": 27,
        "claim_backend_correct": 27,
        "claims_extracted": 86,
        "avg_claims_per_case": 3.0,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "case_operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 27},
        "avg_proof_tokens": 18.0,
        "avg_context_tokens": 18.0,
        "latency_budget_checks": 27,
        "latency_budget_ms": 100.0,
        "p95_latency_ms": 2.0,
        "max_latency_ms": 2.5,
    }))
    (path / "smqe_paraphrase_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 24,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 2,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 2,
            "table_lookup": 3,
            "temporal_delta": 2,
        },
        "backend_counts": {"claim": 24, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_conflict_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 24,
        "failures": [],
        "value_type_counts": {"amount": 8, "location": 8, "status": 8},
        "backend_counts": {"claim": 24, "record": 24},
        "avg_proof_tokens": 11.0,
    }))
    (path / "smqe_composition_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 24,
        "failures": [],
        "case_type_counts": {
            "event_order": 4,
            "relative_event_time": 4,
            "shared_value": 16,
        },
        "backend_counts": {"claim": 24, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_relative_phrase_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 24,
        "failures": [],
        "case_type_counts": {
            "ago_days": 4,
            "ago_weeks": 4,
            "fortnight_ago": 4,
            "in_days": 4,
            "next_month": 4,
            "next_week": 4,
        },
        "backend_counts": {"claim": 24, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_temporal_window_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 24,
        "failures": [],
        "case_type_counts": {
            "fortnight_count": 3,
            "most_recent_latest": 2,
            "past_days_count": 2,
            "past_few_months_count": 2,
            "past_week_count": 2,
            "past_week_list": 2,
            "recent_count": 3,
            "recent_hours_sum": 2,
            "recent_list": 2,
            "source_action_variant_window": 2,
            "source_location_window": 2,
        },
        "backend_counts": {"claim": 24, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_attribution_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 24,
        "failures": [],
        "case_type_counts": {
            "gave_actor": 6,
            "recommend_actor": 6,
            "shared_actor": 6,
            "told_actor": 6,
        },
        "backend_counts": {"claim": 24, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_abstention_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 48,
        "abstained": 24,
        "record_only_abstained": 24,
        "claims_present_abstained": 24,
        "failures": [],
        "case_type_counts": {
            "count_neutral_quantity": 3,
            "count_target_mismatch": 3,
            "latest_future_only": 3,
            "latest_missing_subject": 3,
            "preference_no_positive": 3,
            "speaker_crossed_support": 3,
            "table_missing_row": 3,
            "temporal_missing_anchor": 3,
        },
    }))
    (path / "smqe_scope_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 96,
        "correct": 96,
        "record_backend_correct": 48,
        "claim_backend_correct": 48,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 48, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_subscope_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 96,
        "correct": 96,
        "record_backend_correct": 48,
        "claim_backend_correct": 48,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 48, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_time_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 96,
        "correct": 96,
        "record_backend_correct": 48,
        "claim_backend_correct": 48,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 48, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_invalidation_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 424242,
        "seed_mode": "random",
        "cases": 24,
        "checks": 96,
        "correct": 96,
        "record_backend_correct": 48,
        "claim_backend_correct": 48,
        "preference_supersession_pass": True,
        "preference_supersession_cases": 3,
        "preference_supersession_checks": 12,
        "preference_supersession_correct": 12,
        "preference_supersession_record_correct": 6,
        "preference_supersession_claim_correct": 6,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 48, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    (path / "smqe_dialogue_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 434343,
        "seed_mode": "random",
        "cases": 24,
        "correct": 24,
        "failures": [],
        "case_type_counts": {"paraphrase_slot": 6, "entity_guard": 6,
                             "advice_deferral": 6, "unrelated_guard": 6},
    }))
    (path / "smqe_lacuna_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 454545,
        "seed_mode": "random",
        "cases": 24,
        "correct": 24,
        "failures": [],
        "case_type_counts": {"positive_confirmation": 6, "negative_assertion": 6,
                             "retraction_order": 6, "absent_proposition": 6},
    }))
    (path / "crystal_demotion_invariant.json").write_text(json.dumps({
        "pass": True,
        "seed": 444444,
        "seed_mode": "random",
        "cases": 20,
        "correct": 20,
        "avg_demotion_ratio": 0.15,
        "failures": [],
    }))
    (path / "run_manifest.json").write_text(json.dumps({
        "systems": "eidetic,eidetic-full,rag-full",
        "dataset": "locomo",
        "split": split,
        "subset": 0,
        "samples_file": str(samples_file),
        "holdout_profile": "holdout",
        "runs": runs,
        "run_offset": 0,
        "render_only": False,
        "sample_rows": sample_rows,
        "category_counts": {"single-hop": 4},
        "env": {"READER_MODEL": "qwen-plus", "DATA_DIR": str(data_dir)},
    }))


def _proof_hashes_from_logs(path: Path) -> list[str]:
    hashes: set[str] = set()
    for log_path in path.glob("*__run*.jsonl"):
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra") or {}
            if not isinstance(extra, dict) or not extra.get("verified"):
                continue
            for h in extra.get("entailed_content_hashes", []) or []:
                h = str(h).strip().lower()
                if len(h) == 64:
                    hashes.add(h)
    return sorted(hashes)


def _write_fingerprinted_reports(path: Path) -> None:
    fingerprint = log_fingerprint(path)
    proof_hashes = _proof_hashes_from_logs(path)
    snap_path = path / "snap_back_audit.json"
    if snap_path.exists():
        snap = json.loads(snap_path.read_text())
        snap["audited_content_hashes"] = proof_hashes
        if str(snap.get("status", "")).upper() == "PASS":
            n = max(int(snap.get("records_with_raw_blob", 0) or 0), len(proof_hashes))
            snap["records_with_raw_blob"] = n
            snap["lossless_byte_identical"] = n
            snap["rate"] = 1.0
            snap["rate_pct"] = 100.0
        snap_path.write_text(json.dumps(snap))
    (path / "scoreboard.json").write_text(json.dumps({
        "log_fingerprint": fingerprint,
    }))
    (path / "mem0_gate.json").write_text(json.dumps({
        "status": "PASS",
        "reason": "within tolerance",
        "system": "mem0",
        "dataset": "locomo",
        "tolerance": 0.02,
        "min_n": 8,
        "total_n": 8,
        "expected": {"single-hop": 0.5},
        "observed": {"single-hop": {"n": 8, "accuracy": 0.5, "runs": [0, 1]}},
        "comparisons": {
            "single-hop": {
                "status": "PASS",
                "n": 8,
                "runs": [0, 1],
                "observed": 0.5,
                "expected": 0.5,
                "delta": 0.0,
            },
        },
        "log_fingerprint": fingerprint,
    }))


def _row(system: str, sid: str, run_idx: int, correct: bool, *, verified: bool | None = None,
         timeout: int = 0, age_days: float = 0.0, query_tokens: int | None = None,
         search_ms: float = 1.0, e2e_ms: float = 2.0,
         dataset: str = "locomo", category: str = "single-hop",
         baseline_health: dict | None = None) -> dict:
    extra = {}
    if verified is not None:
        extra["verified"] = verified
        if verified:
            proof_id = f"mem_{system}_{sid}_{run_idx}"
            proof_hash = hashlib.sha256(proof_id.encode("utf-8")).hexdigest()
            extra.update({
                "citations": 1,
                "candidate_memory_ids": [proof_id],
                "entailed_memory_ids": [proof_id],
                "entailed_content_hashes": [proof_hash],
                "entailed_raw_uris": [f"cas://{proof_hash}"],
                "proof_surface_tokens": 12,
            })
    if baseline_health is not None:
        extra["baseline_health"] = baseline_health
    if timeout:
        extra["consolidate"] = {"consolidate_pending": {
            "pending_processed": 1,
            "facts_extracted": 0,
            "events_indexed": 0,
            "extraction_timed_out": timeout,
            "extraction_deferred": 0,
        }}
    return {
        "system": system,
        "dataset": dataset,
        "category": category,
        "sample_id": sid,
        "question": "q",
        "gold": "g",
        "predicted": "g" if correct else "x",
        "correct": correct,
        "write_tokens": 10,
        "query_tokens": query_tokens if query_tokens is not None else (100 if system == "rag-full" else 5),
        "search_ms": search_ms,
        "e2e_ms": e2e_ms,
        "abstained": False,
        "run_idx": run_idx,
        "age_days": age_days,
        "extra": extra,
    }


def test_release_gate_pairing_keys_include_dataset_and_category():
    rows = [
        _row("eidetic-plus", "shared_q0", 0, True, dataset="locomo", category="single-hop"),
        _row("rag-full", "shared_q0", 0, False, dataset="locomo", category="single-hop"),
        _row("eidetic-plus", "shared_q0", 0, False, dataset="longmemeval", category="multi-session"),
        _row("rag-full", "shared_q0", 0, True, dataset="longmemeval", category="multi-session"),
        _row("eidetic-plus", "shared_q0", 0, True, dataset="locomo", category="temporal"),
        _row("rag-full", "shared_q0", 0, True, dataset="locomo", category="temporal"),
    ]

    paired = _paired_stats(rows, "eidetic-plus", "rag-full")
    clustered = _sample_clustered_paired_stats(rows, "eidetic-plus", "rag-full")
    accuracy = _sample_clustered_accuracy([r for r in rows if r["system"] == "eidetic-plus"])

    assert _unique_sample_count(rows) == 3
    assert paired["n"] == 3
    assert paired["headline_only"] == 1
    assert paired["baseline_only"] == 1
    assert paired["both"] == 1
    assert clustered["n"] == 3
    assert clustered["headline_only"] == 1
    assert clustered["baseline_only"] == 1
    assert clustered["ties"] == 1
    assert accuracy["n"] == 3


def _write_logs(path: Path, *, timeout: int = 0, sample_split: str = "test",
                headline_query_tokens: int = 5, headline_search_ms: float = 1.0,
                headline_e2e_ms: float = 2.0, age_fragile: bool = False) -> None:
    ids = _ids(4, split=sample_split)
    by_system = {"eidetic-plus": [], "eidetic-plus-full": [], "rag-full": []}
    for run_idx in (0, 1):
        for i, sid in enumerate(ids):
            age_days = float(i * 365)
            headline_ok = False if age_fragile and i == len(ids) - 1 else True
            by_system["eidetic-plus"].append(_row(
                "eidetic-plus", sid, run_idx, headline_ok,
                timeout=timeout if i == 0 and run_idx == 0 else 0,
                age_days=age_days,
                query_tokens=headline_query_tokens,
                search_ms=headline_search_ms,
                e2e_ms=headline_e2e_ms,
            ))
            full = _row("eidetic-plus-full", sid, run_idx, True,
                        verified=True, age_days=age_days)
            full["extra"].update({
                "structured_recall": True,
                "smqe_operator": "latest_value",
                "smqe_backend": "claim",
                "smqe_policy": "smqe:latest_value:claim",
                "policy": "smqe:latest_value:claim",
                "region_hint_count": 0,
                "region_ids": [],
                "region_member_ids": [],
            })
            by_system["eidetic-plus-full"].append(full)
            by_system["rag-full"].append(_row("rag-full", sid, run_idx, i % 2 == 0,
                                              age_days=age_days))
    for system, rows in by_system.items():
        (path / f"{system}__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _write_fingerprinted_reports(path)


def _run_gate(path: Path, **kwargs):
    params = {
        "required_systems": ["eidetic-plus", "eidetic-plus-full", "rag-full"],
        "baseline_systems": ["rag-full"],
        "required_datasets": ["locomo"],
        "required_categories_by_dataset": {"locomo": ["single-hop"]},
        "min_runs": 2,
        "min_questions_per_system": 4,
        "min_dataset_accuracy": 0.90,
        "min_overall_delta_pp": 40.0,
        "min_category_delta_pp": 40.0,
        "alpha": 1.0,
        "max_dataset_accuracy_ci_width_pp": 100.0,
        "max_category_accuracy_ci_width_pp": 100.0,
        "require_ci_clear_dominance": False,
        "min_clustered_discordant_samples": 2,
        "min_verified_accuracy": 0.90,
        "min_age_slope_points": 4,
        "min_slice_invariant_draws": 5,
        "min_slice_invariant_subset": 4,
        "min_smqe_synthetic_cases": 24,
        "min_smqe_claim_coverage_cases": 24,
    }
    params.update(kwargs)
    return run_release_gate(path, **params)


def _lower_headline_accuracy(path: Path) -> None:
    log_path = path / "eidetic-plus__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows[0]["correct"] = False
    rows[0]["predicted"] = "x"
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(path)


def _write_best_world_scope(path: Path, *, score) -> None:
    (path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["chronos"],
        "external_system_evidence": [{
            "system": "chronos",
            "dataset": "locomo",
            "split": "test",
            "n": 100,
            "runs": 1,
            "score": score,
            "metric": "verified_accuracy",
            "evaluation_protocol": "same fixed reader and judge where applicable",
            "date": "2026-06-29",
            "source": "paper",
            "artifact_fingerprint": "f" * 64,
        }],
        "limitations": [],
    }))


def test_release_gate_passes_complete_test_artifact(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    report = _run_gate(tmp_path)
    assert report["status"] == "PASS"
    assert report["paired"]["rag-full"]["overall"]["delta_pp"] == 50.0
    assert report["paired"]["rag-full"]["sample_clustered"]["delta_pp"] == 50.0
    category = report["paired"]["rag-full"]["categories"]["locomo|single-hop"]
    assert category["row"]["delta_pp"] == 50.0
    assert category["sample_clustered"]["delta_pp"] == 50.0
    assert category["sample_clustered"]["discordant"] == 2
    assert report["operating"]["eidetic-plus"]["query_tokens_median"] == 5
    assert report["operating"]["rag-full"]["query_tokens_median"] == 100
    assert report["age_slope"]["slope_per_year"] == 0.0
    assert report["log_fingerprint"]["file_count"] == 3
    assert report["evidence_strength"]["locomo|*"]["n"] == 4
    assert report["thresholds"]["require_ci_clear_dominance"] is False
    assert report["ablation_evidence"]["pass"] is True
    assert report["ablation_evidence"]["forgetting_cost_ratio"] == 1.3
    assert report["ablation_evidence"]["region_delta_pp"] == 6.0
    assert report["ablation_evidence"]["affect_delta_pp"] == 4.0
    out = render_markdown(report, tmp_path / "release_gate.md")
    rendered = out.read_text()
    assert "Status: **PASS**" in rendered
    assert "Evidence Strength" in rendered
    assert "Category Clustered Dominance" in rendered
    assert (tmp_path / "release_gate.json").exists()


def test_release_gate_fails_when_legacy_scan_env_var_enabled(tmp_path):
    _write_artifacts(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["EIDETIC_ENABLE_DATASET_SOURCE_SCANS"] = "1"
    manifest_path.write_text(json.dumps(manifest))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "manifest:no_dataset_source_scans" in failed


def test_release_gate_fails_when_eidetic_only_ingest_granularity_enabled(tmp_path):
    _write_artifacts(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["INGEST_GRANULARITY"] = "hybrid"
    manifest_path.write_text(json.dumps(manifest))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "manifest:session_ingest_granularity" in failed


def test_release_gate_fails_when_logs_contain_legacy_structured_policy(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows[0].setdefault("extra", {})["policy"] = "long" + "memeval-direct"
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe:notes_clean" in failed


def test_release_gate_rejects_malformed_smqe_log_policy(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows[0]["extra"]["smqe_policy"] = "smqe:latest_value"
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe:log_policy_shape" in failed


def test_release_gate_rejects_record_dominated_smqe_log_policy(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["extra"]["smqe_backend"] = "record"
        row["extra"]["smqe_policy"] = "smqe:latest_value:record"
        row["extra"]["policy"] = "smqe:latest_value:record"
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe:log_policy_shape" in failed


def test_release_gate_requires_claim_backend_to_dominate_structured_rows(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    for row in rows[:2]:
        row["extra"]["smqe_backend"] = "record"
        row["extra"]["smqe_policy"] = "smqe:latest_value:record"
        row["extra"]["policy"] = "smqe:latest_value:record"
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe:log_policy_shape" in failed
    smqe_check = next(check for check in report["checks"] if check["name"] == "smqe:log_policy_shape")
    assert "claim_backend_rate:0.750<required:0.800" in json.dumps(smqe_check)


def test_release_gate_requires_structured_recall_to_dominate_integrity_rows(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    for row in rows[:2]:
        row["extra"] = {"verified": True, "policy": "fixed-reader + verify+abstain+proof"}
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe:log_policy_shape" in failed
    smqe_check = next(check for check in report["checks"] if check["name"] == "smqe:log_policy_shape")
    assert "structured_rate:0.750<required:0.800" in json.dumps(smqe_check)


def test_release_gate_requires_region_telemetry_in_integrity_rows(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["extra"].pop("region_hint_count", None)
        row["extra"].pop("region_ids", None)
        row["extra"].pop("region_member_ids", None)
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "region:telemetry" in failed
    assert report["region_telemetry"]["failures"] == ["missing:8"]


def test_release_gate_requires_verified_rows_to_carry_proof_support(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows[0]["extra"].pop("citations", None)
    rows[0]["extra"].pop("entailed_memory_ids", None)
    rows[0]["extra"].pop("entailed_content_hashes", None)
    rows[0]["extra"].pop("entailed_raw_uris", None)
    rows[0]["extra"].pop("proof_surface_tokens", None)
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "integrity:eidetic-plus-full:proof_support" in failed
    assert report["proof_support"]["failures"] == ["missing_support:1"]
    assert report["proof_support"]["missing_support"] == [
        f"{rows[0]['sample_id']}:citations,entailed_memory_ids,"
        "entailed_content_hashes,entailed_raw_uris,proof_surface_tokens"
    ]


def test_release_gate_requires_snap_back_to_cover_verified_proof_hashes(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    snap_path = tmp_path / "snap_back_audit.json"
    snap = json.loads(snap_path.read_text())
    removed = snap["audited_content_hashes"][0]
    snap["audited_content_hashes"] = [
        h for h in snap["audited_content_hashes"] if h != removed
    ]
    snap_path.write_text(json.dumps(snap))

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "snap_back:covers_verified_proof_hashes" in failed
    snap_check = next(
        check for check in report["checks"]
        if check["name"] == "snap_back:covers_verified_proof_hashes"
    )
    assert removed in snap_check["missing_proof_hashes"]


def test_release_gate_rejects_unverified_answered_integrity_rows(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows[0]["extra"]["verified"] = False
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path, min_verified_accuracy=0.80)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "integrity:eidetic-plus-full:proof_support" in failed
    assert report["proof_support"]["failures"] == ["unverified_answered:1"]
    assert report["proof_support"]["unverified_answered"] == [rows[0]["sample_id"]]


def test_release_gate_rejects_reader_dominated_integrity_rows(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    log_path = tmp_path / "eidetic-plus-full__run0.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    for row in rows[:5]:
        row["extra"] = {"verified": True, "policy": "fixed-reader + verify+abstain+proof"}
    log_path.write_text("\n".join(json.dumps(row) for row in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe:log_policy_shape" in failed


def test_release_gate_requires_holdout_audit_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "holdout_audit.json").unlink()
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "holdout_audit:valid_json" in failed
    assert "holdout_audit:evidence" in failed


def test_release_gate_rejects_empty_holdout_audit_registry(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "holdout_audit.json").write_text(json.dumps({
        "pass": True,
        "findings": [],
        "needles_checked": 0,
        "holdout_needles_checked": 0,
        "legacy_policy_scan_enabled": True,
        "forbidden_policy_strings_checked": 5,
        "forbidden_fixed_answer_strings_checked": 10,
        "forbidden_runtime_symbols_checked": 24,
        "registry_error": "",
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "holdout_audit:evidence" in failed


def test_release_gate_rejects_holdout_audit_findings(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "holdout_audit.json").write_text(json.dumps({
        "pass": False,
        "findings": [{"path": "eidetic/example.py", "needle": "sample123", "line": 7}],
        "needles_checked": 12,
        "holdout_needles_checked": 8,
        "legacy_policy_scan_enabled": True,
        "forbidden_policy_strings_checked": 5,
        "forbidden_fixed_answer_strings_checked": 10,
        "forbidden_runtime_symbols_checked": 24,
        "registry_error": "",
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "holdout_audit:evidence" in failed


def test_release_gate_requires_holdout_audit_to_scan_legacy_shortcuts(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "holdout_audit.json").write_text(json.dumps({
        "pass": True,
        "findings": [],
        "needles_checked": 8,
        "holdout_needles_checked": 8,
        "legacy_policy_scan_enabled": False,
        "forbidden_policy_strings_checked": 0,
        "forbidden_fixed_answer_strings_checked": 0,
        "forbidden_runtime_symbols_checked": 0,
        "registry_error": "",
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "holdout_audit:evidence" in failed
    holdout_check = next(check for check in report["checks"] if check["name"] == "holdout_audit:evidence")
    assert "legacy_policy_scan_enabled:false" in holdout_check["failures"]
    assert "forbidden_runtime_symbols_checked:0" in holdout_check["failures"]


def test_release_gate_requires_ablation_report(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "ablation_report.json").unlink()
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:valid_json" in failed
    assert "ablation:evidence" in failed


def test_release_gate_rejects_weak_metabolism_ablation(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"]["metabolism_off"]["verified_accuracy"] = 0.88
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed


def test_release_gate_rejects_forgetting_ablation_without_cost_savings(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"]["forgetting_off"]["query_tokens_median"] = 101
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed


def test_release_gate_rejects_forgetting_ablation_accuracy_regression(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"]["forgetting_off"]["verified_accuracy"] = 0.94
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed


def test_release_gate_rejects_missing_affect_ablation(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"].pop("affect_off")
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed
    failures = report["ablation_evidence"]["failures"]
    assert "ablation:affect_off_missing" in failures


def test_release_gate_rejects_weak_affect_salience_invariant(tmp_path):
    _write_artifacts(tmp_path)
    report_data = _affect_salience_report()
    report_data["correct"] -= 1
    report_data["max_boost_ratio"] = 0.75
    (tmp_path / "affect_salience_invariant.json").write_text(json.dumps(report_data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    affect_check = next(
        check for check in report["failed_checks"]
        if check["name"] == "affect_salience:evidence"
    )
    assert "correct:" in affect_check["detail"]
    assert "max_boost_ratio" in affect_check["detail"]


def test_release_gate_rejects_weak_scratchpad_invariant(tmp_path):
    _write_artifacts(tmp_path)
    report_data = _scratchpad_report()
    report_data["proof_link_checks"] -= 1
    report_data["retrieval_channel_checks"] -= 1
    report_data["correct"] -= 1
    (tmp_path / "scratchpad_invariant.json").write_text(json.dumps(report_data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    scratchpad_check = next(
        check for check in report["failed_checks"]
        if check["name"] == "scratchpad:evidence"
    )
    assert "correct:" in scratchpad_check["detail"]
    assert "proof_link_checks" in scratchpad_check["detail"]
    assert "retrieval_channel_checks" in scratchpad_check["detail"]


def test_release_gate_rejects_weak_region_routing_invariant(tmp_path):
    _write_artifacts(tmp_path)
    report_data = _region_routing_report()
    report_data["proof_link_checks"] -= 1
    report_data["telemetry_trace_checks"] -= 1
    report_data["correct"] -= 1
    (tmp_path / "region_routing_invariant.json").write_text(json.dumps(report_data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    region_check = next(
        check for check in report["failed_checks"]
        if check["name"] == "region_routing:evidence"
    )
    assert "correct:" in region_check["detail"]
    assert "proof_link_checks" in region_check["detail"]
    assert "telemetry_trace_checks" in region_check["detail"]


def test_release_gate_rejects_weak_reflex_recall_invariant(tmp_path):
    _write_artifacts(tmp_path)
    report_data = _reflex_recall_report()
    report_data["coactivation_checks"] -= 1
    report_data["correct"] -= 1
    report_data["p95_latency_ms"] = 150.0
    (tmp_path / "reflex_recall_invariant.json").write_text(json.dumps(report_data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    reflex_check = next(
        check for check in report["failed_checks"]
        if check["name"] == "reflex_recall:evidence"
    )
    assert "correct:" in reflex_check["detail"]
    assert "coactivation_checks" in reflex_check["detail"]
    assert "p95_latency_ms" in reflex_check["detail"]


def test_release_gate_rejects_weak_smqe_planner_invariant(tmp_path):
    _write_artifacts(tmp_path)
    report_data = _smqe_planner_report()
    report_data["operator_counts"]["open_inference"] = 1
    report_data["correct"] -= 1
    report_data["p95_latency_ms"] = 20.0
    (tmp_path / "smqe_planner_invariant.json").write_text(json.dumps(report_data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    planner_check = next(
        check for check in report["failed_checks"]
        if check["name"] == "smqe_planner:evidence"
    )
    assert "correct:" in planner_check["detail"]
    assert "ops_below_2" in planner_check["detail"]
    assert "p95_latency_ms" in planner_check["detail"]


def test_release_gate_rejects_fixed_seed_for_every_rotating_sidecar(tmp_path):
    sidecars = [
        ("affect_salience_invariant.json", "affect_salience:evidence"),
        ("scratchpad_invariant.json", "scratchpad:evidence"),
        ("region_routing_invariant.json", "region_routing:evidence"),
        ("reflex_recall_invariant.json", "reflex_recall:evidence"),
        ("smqe_planner_invariant.json", "smqe_planner:evidence"),
        ("smqe_synthetic_invariant.json", "smqe_synthetic:evidence"),
        ("smqe_claim_coverage.json", "smqe_claim_coverage:evidence"),
        ("smqe_fullpath_invariant.json", "smqe_fullpath:evidence"),
        ("smqe_paraphrase_invariant.json", "smqe_paraphrase:evidence"),
        ("smqe_conflict_invariant.json", "smqe_conflict:evidence"),
        ("smqe_composition_invariant.json", "smqe_composition:evidence"),
        ("smqe_relative_phrase_invariant.json", "smqe_relative_phrase:evidence"),
        ("smqe_temporal_window_invariant.json", "smqe_temporal_window:evidence"),
        ("smqe_attribution_invariant.json", "smqe_attribution:evidence"),
        ("smqe_abstention_invariant.json", "smqe_abstention:evidence"),
        ("smqe_scope_invariant.json", "smqe_scope:evidence"),
        ("smqe_subscope_invariant.json", "smqe_subscope:evidence"),
        ("smqe_time_invariant.json", "smqe_time:evidence"),
        ("smqe_invalidation_invariant.json", "smqe_invalidation:evidence"),
        ("smqe_dialogue_invariant.json", "smqe_dialogue:evidence"),
        ("smqe_lacuna_invariant.json", "smqe_lacuna:evidence"),
        ("crystal_demotion_invariant.json", "crystal_demotion:evidence"),
    ]
    for idx, (filename, check_name) in enumerate(sidecars):
        artifact = tmp_path / f"fixed-seed-{idx}"
        _write_artifacts(artifact)
        sidecar = json.loads((artifact / filename).read_text())
        sidecar["seed_mode"] = "fixed"
        (artifact / filename).write_text(json.dumps(sidecar))
        _write_logs(artifact)

        report = _run_gate(artifact)

        check = next(
            item for item in report["failed_checks"]
            if item["name"] == check_name
        )
        assert "seed_mode:fixed" in check["detail"]


def test_release_gate_rejects_missing_region_ablation(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"].pop("regions_off")
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed
    failures = report["ablation_evidence"]["failures"]
    assert "ablation:regions_off_missing" in failures


def test_release_gate_rejects_weak_region_ablation(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"]["regions_off"]["verified_accuracy"] = 0.895
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed
    assert any("region_delta_pp" in item for item in report["ablation_evidence"]["failures"])


def test_release_gate_rejects_weak_affect_ablation(tmp_path):
    _write_artifacts(tmp_path)
    data = json.loads((tmp_path / "ablation_report.json").read_text())
    data["ablations"]["affect_off"]["verified_accuracy"] = 0.895
    (tmp_path / "ablation_report.json").write_text(json.dumps(data))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "ablation:evidence" in failed
    assert any("affect_delta_pp" in item for item in report["ablation_evidence"]["failures"])


def test_release_gate_rejects_empty_slice_invariant_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "slice_invariant.json").write_text(json.dumps({
        "pass": True,
        "dataset": "locomo",
        "draws": 5,
        "subset": 4,
        "system_under_test": "eidetic-plus-full",
        "runs": [],
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed


def test_release_gate_rejects_replayed_slice_invariant_draws(tmp_path):
    _write_artifacts(tmp_path)
    repeated = _ids(4, split="test")
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    for run in sidecar["runs"]:
        run["sample_ids"] = repeated
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed


def test_release_gate_rejects_low_unique_slice_sample_coverage(tmp_path):
    _write_artifacts(tmp_path)
    ids = _ids(8, split="test")
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    for idx, run in enumerate(sidecar["runs"]):
        run["sample_ids"] = [ids[(idx + offset) % len(ids)] for offset in range(4)]
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "unique_sample_ids:8<required:20" in json.dumps(slice_check)


def test_release_gate_rejects_insufficient_declared_slice_pool(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar["pool_unique_sample_ids"] = 19
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "pool_unique_sample_ids:19<required:20" in json.dumps(slice_check)


def test_release_gate_rejects_unmet_declared_slice_coverage(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar["pool_unique_sample_ids"] = 21
    sidecar["required_unique_sample_ids"] = 21
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "unique_sample_ids:20<required:21" in json.dumps(slice_check)


def test_release_gate_rejects_duplicate_ids_within_slice_draw(tmp_path):
    _write_artifacts(tmp_path)
    ids = _ids(21, split="test")
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar["runs"][0]["sample_ids"] = [ids[0], ids[0], ids[1], ids[2]]
    sidecar["runs"][1]["sample_ids"].append(ids[20])
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "draw1:unique_sample_ids:3<required:4" in json.dumps(slice_check)


def test_release_gate_rejects_correct_only_slice_invariant_scores(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    for run in sidecar["runs"]:
        run["score"] = {"pass": True, "correct": 4, "total": 4}
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "verified=False" in json.dumps(slice_check)


def test_release_gate_rejects_wrong_split_slice_sample_ids(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar["runs"][0]["sample_ids"][0] = _ids(1, split="dev")[0]
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "sample_split:test" in json.dumps(slice_check)


def test_release_gate_rejects_fixed_seed_slice_invariant_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar["seed_mode"] = "fixed"
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed


def test_release_gate_rejects_dev_split_slice_invariant_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar["split"] = "dev"
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "split:dev" in json.dumps(slice_check)


def test_release_gate_rejects_missing_holdout_profile_slice_invariant_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "slice_invariant.json").read_text())
    sidecar.pop("holdout_profile")
    (tmp_path / "slice_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "slice_invariant:evidence" in failed
    slice_check = next(check for check in report["checks"] if check["name"] == "slice_invariant:evidence")
    assert "holdout_profile:<missing>" in json.dumps(slice_check)


def test_release_gate_rejects_fixed_seed_smqe_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_synthetic_invariant.json").read_text())
    sidecar["seed_mode"] = "fixed"
    (tmp_path / "smqe_synthetic_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_synthetic:evidence" in failed


def test_release_gate_rejects_fixed_seed_smqe_fullpath_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_fullpath_invariant.json").read_text())
    sidecar["seed_mode"] = "fixed"
    (tmp_path / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_fullpath:evidence" in failed


def test_release_gate_rejects_reader_calls_in_smqe_fullpath_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_fullpath_invariant.json").read_text())
    sidecar["reader_calls"] = 1
    (tmp_path / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_fullpath:evidence" in failed


def test_release_gate_rejects_smqe_fullpath_case_operator_coverage_gap(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_fullpath_invariant.json").read_text())
    sidecar["case_operator_counts"]["latest_value"] = 1
    sidecar["case_operator_counts"]["count_aggregate"] = 5
    (tmp_path / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert "smqe_fullpath:evidence" in failed
    assert "case_ops_below_2:latest_value" in details


def test_release_gate_rejects_bloated_smqe_fullpath_context(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_fullpath_invariant.json").read_text())
    sidecar["avg_context_tokens"] = 120.0
    (tmp_path / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert "smqe_fullpath:evidence" in failed
    assert "avg_context_tokens:120.0>80.0" in details


def test_release_gate_rejects_slow_smqe_fullpath_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_fullpath_invariant.json").read_text())
    sidecar["p95_latency_ms"] = 150.0
    (tmp_path / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert "smqe_fullpath:evidence" in failed
    assert "p95_latency_ms:150.0>100.0" in details


def test_release_gate_rejects_missing_smqe_fullpath_proof_links(tmp_path):
    _write_artifacts(tmp_path)
    sidecar = json.loads((tmp_path / "smqe_fullpath_invariant.json").read_text())
    sidecar["proof_link_checks"] = 26
    (tmp_path / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert "smqe_fullpath:evidence" in failed
    assert "proof_link_checks:26<expected:27" in details


def test_release_gate_rejects_weak_smqe_synthetic_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_synthetic_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "correct": 24,
        "failures": [],
        "operator_counts": {"latest_value": 24},
        "backend_counts": {"claim": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_synthetic:evidence" in failed


def test_release_gate_rejects_weak_smqe_claim_coverage_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_claim_coverage.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "correct": 24,
        "claim_backend_correct": 12,
        "claims_extracted": 24,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "claim_type_counts": {"state": 24},
        "backend_counts": {"claim": 12, "record": 12},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_claim_coverage:evidence" in failed


def test_release_gate_rejects_operator_gap_in_smqe_claim_coverage_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    op_counts = {
        "count_aggregate": 3,
        "latest_value": 3,
        "multi_session_sum": 3,
        "open_inference": 3,
        "preference_synth": 3,
        "relative_temporal": 3,
        "speaker_fact": 3,
        "table_lookup": 3,
        "temporal_delta": 3,
    }
    claim_op_counts = dict(op_counts)
    claim_op_counts["latest_value"] = 2
    (tmp_path / "smqe_claim_coverage.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "correct": 24,
        "claim_backend_correct": 24,
        "claims_extracted": 24,
        "failures": [],
        "operator_counts": op_counts,
        "claim_backend_operator_counts": claim_op_counts,
        "claim_type_counts": {"state": 24},
        "backend_counts": {"claim": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert "smqe_claim_coverage:evidence" in failed
    assert "operator_claim_backend:latest_value:2/3" in details


def test_release_gate_rejects_weak_smqe_paraphrase_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_paraphrase_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 12,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 2,
            "open_inference": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 2,
            "table_lookup": 3,
            "temporal_delta": 2,
        },
        "backend_counts": {"claim": 12, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_paraphrase:evidence" in failed


def test_release_gate_rejects_weak_smqe_conflict_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_conflict_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "correct": 24,
        "record_backend_correct": 24,
        "claim_backend_correct": 12,
        "failures": [],
        "value_type_counts": {"amount": 8, "location": 8, "status": 8},
        "backend_counts": {"claim": 12, "record": 24},
        "avg_proof_tokens": 11.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_conflict:evidence" in failed


def test_release_gate_rejects_weak_smqe_composition_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_composition_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "correct": 23,
        "record_backend_correct": 24,
        "claim_backend_correct": 23,
        "failures": [],
        "case_type_counts": {"shared_value": 24},
        "backend_counts": {"claim": 23, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_composition:evidence" in failed


def test_release_gate_rejects_weak_smqe_relative_phrase_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_relative_phrase_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "correct": 23,
        "record_backend_correct": 24,
        "claim_backend_correct": 23,
        "failures": [],
        "case_type_counts": {"ago_days": 24},
        "backend_counts": {"claim": 23, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_relative_phrase:evidence" in failed


def test_release_gate_rejects_weak_smqe_temporal_window_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_temporal_window_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "correct": 23,
        "record_backend_correct": 24,
        "claim_backend_correct": 23,
        "failures": [],
        "case_type_counts": {"recent_count": 24},
        "backend_counts": {"claim": 23, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_temporal_window:evidence" in failed


def test_release_gate_rejects_weak_smqe_attribution_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_attribution_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "correct": 23,
        "record_backend_correct": 24,
        "claim_backend_correct": 23,
        "failures": [],
        "case_type_counts": {"gave_actor": 24},
        "backend_counts": {"claim": 23, "record": 24},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_attribution:evidence" in failed


def test_release_gate_rejects_weak_smqe_abstention_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_abstention_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 48,
        "abstained": 23,
        "record_only_abstained": 24,
        "claims_present_abstained": 23,
        "failures": [],
        "case_type_counts": {"table_missing_row": 24},
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_abstention:evidence" in failed


def test_release_gate_rejects_weak_smqe_scope_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_scope_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 96,
        "correct": 95,
        "record_backend_correct": 48,
        "claim_backend_correct": 47,
        "failures": [],
        "operator_counts": {"latest_value": 24},
        "backend_counts": {"claim": 47, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_scope:evidence" in failed


def test_release_gate_rejects_weak_smqe_subscope_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_subscope_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 96,
        "correct": 95,
        "record_backend_correct": 48,
        "claim_backend_correct": 47,
        "failures": [],
        "operator_counts": {"latest_value": 24},
        "backend_counts": {"claim": 47, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_subscope:evidence" in failed


def test_release_gate_rejects_weak_smqe_time_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_time_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 96,
        "correct": 95,
        "record_backend_correct": 48,
        "claim_backend_correct": 47,
        "failures": [],
        "operator_counts": {"latest_value": 24},
        "backend_counts": {"claim": 47, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_time:evidence" in failed


def test_release_gate_rejects_weak_smqe_invalidation_sidecar(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_invalidation_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 96,
        "correct": 95,
        "record_backend_correct": 48,
        "claim_backend_correct": 47,
        "failures": [],
        "operator_counts": {"latest_value": 24},
        "backend_counts": {"claim": 47, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    failed = {check["name"] for check in report["failed_checks"]}
    assert "smqe_invalidation:evidence" in failed


def test_release_gate_rejects_invalidation_sidecar_without_preference_supersession(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "smqe_invalidation_invariant.json").write_text(json.dumps({
        "pass": True,
        "cases": 24,
        "checks": 96,
        "correct": 96,
        "record_backend_correct": 48,
        "claim_backend_correct": 48,
        "failures": [],
        "operator_counts": {
            "count_aggregate": 3,
            "latest_value": 3,
            "multi_session_sum": 3,
            "preference_synth": 3,
            "relative_temporal": 3,
            "speaker_fact": 3,
            "table_lookup": 3,
            "temporal_delta": 3,
        },
        "backend_counts": {"claim": 48, "record": 48},
        "avg_proof_tokens": 18.0,
    }))
    _write_logs(tmp_path)

    report = _run_gate(tmp_path)

    invalidation_check = next(
        check for check in report["failed_checks"]
        if check["name"] == "smqe_invalidation:evidence"
    )
    assert "preference_supersession_cases" in invalidation_check["detail"]


def test_release_gate_fails_tiny_slice_with_wide_confidence_interval(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)

    report = _run_gate(
        tmp_path,
        max_dataset_accuracy_ci_width_pp=10.0,
        max_category_accuracy_ci_width_pp=10.0,
        require_ci_clear_dominance=False,
    )

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "evidence:eidetic-plus:locomo:sample_clustered_accuracy_ci_width" in failed
    assert "evidence:eidetic-plus:locomo/single-hop:sample_clustered_accuracy_ci_width" in failed
    assert report["evidence_strength"]["locomo|*"]["n"] == 4
    assert report["evidence_strength"]["locomo|*"]["wilson_width_pp"] > 10.0


def test_release_gate_requires_ci_clear_dominance_when_enabled(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)

    report = _run_gate(tmp_path, require_ci_clear_dominance=True)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "dominance:eidetic-plus:vs:rag-full:sample_clustered_ci_clear" in failed
    assert (
        "dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_ci_clear"
        in failed
    )
    ci = report["paired"]["rag-full"]["sample_clustered_ci"]
    assert ci["headline"]["wilson_low"] <= ci["baseline"]["wilson_high"]


def test_release_gate_fails_artifact_with_recorded_system_startup_failure(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["system_failures"] = [{
        "system": "graphiti",
        "error_type": "RuntimeError",
        "error": "Neo4j DNS failure",
    }]
    manifest_path.write_text(json.dumps(manifest))

    report = _run_gate(tmp_path)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "manifest:no_system_failures" in failed


def test_release_gate_fails_logs_that_do_not_match_manifest_sample_rows(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sample_rows"][0]["sample_id"] = _ids(1)[0] + "_manifest_only"
    manifest_path.write_text(json.dumps(manifest))

    report = _run_gate(tmp_path)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "logs:match_manifest_sample_rows" in failed
    assert report["manifest_contract"]["missing"]
    assert report["manifest_contract"]["extra"]


def test_release_gate_fails_missing_required_categories_by_default(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)

    report = _run_gate(tmp_path, required_categories_by_dataset=None)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "manifest:locomo:categories_cover_required" in failed
    assert "manifest:locomo:multi-hop:sample_rows" in failed
    assert "manifest:locomo:temporal:sample_rows" in failed
    assert "manifest:locomo:open-domain:sample_rows" in failed
    assert "eidetic-plus:locomo:multi-hop:questions" in failed
    assert "rag-full:locomo:temporal:questions" in failed
    assert report["thresholds"]["required_categories_by_dataset"]["locomo"] == [
        "single-hop",
        "multi-hop",
        "temporal",
        "open-domain",
    ]


def test_release_gate_requires_v2_calibration_when_enabled(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["ABSTENTION_V2"] = "1"
    manifest["env"]["ABSTENTION_V2_TAU"] = "0.7"
    manifest_path.write_text(json.dumps(manifest))

    report = _run_gate(tmp_path, min_abstention_calibration_samples=4)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "abstention_calibration:valid_json" in failed
    assert "abstention_calibration:tau_applied" in failed


def test_release_gate_accepts_matching_v2_calibration_report(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["ABSTENTION_V2"] = "1"
    manifest["env"]["ABSTENTION_V2_TAU"] = "0.7"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "abstention_v2_tau.json").write_text(json.dumps({
        "ok": True,
        "method": "abstention_v2_tau",
        "split": "dev",
        "system": "eidetic-plus-full",
        "tau": 0.7,
        "n": 4,
        "target": 0.95,
        "precision_at_tau": 1.0,
        "coverage_at_tau": 0.75,
        "log_fingerprint": {"combined_sha256": "a" * 64, "file_count": 1, "files": []},
    }))

    report = _run_gate(tmp_path, min_abstention_calibration_samples=4)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "PASS"
    assert not any(name.startswith("abstention_calibration:") for name in failed)
    assert report["abstention_calibration"]["tau"] == 0.7


def test_release_gate_fails_v2_calibration_tau_mismatch(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["ABSTENTION_V2"] = "1"
    manifest["env"]["ABSTENTION_V2_TAU"] = "0.6"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "abstention_v2_tau.json").write_text(json.dumps({
        "ok": True,
        "method": "abstention_v2_tau",
        "split": "dev",
        "system": "eidetic-plus-full",
        "tau": 0.7,
        "n": 4,
        "target": 0.95,
        "precision_at_tau": 1.0,
        "coverage_at_tau": 0.75,
        "log_fingerprint": {"combined_sha256": "a" * 64, "file_count": 1},
    }))

    report = _run_gate(tmp_path, min_abstention_calibration_samples=4)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "abstention_calibration:tau_applied" in failed


def test_release_gate_fails_v2_calibration_that_abstains_everything(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["ABSTENTION_V2"] = "1"
    manifest["env"]["ABSTENTION_V2_TAU"] = "0.7"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "abstention_v2_tau.json").write_text(json.dumps({
        "ok": True,
        "method": "abstention_v2_tau",
        "split": "dev",
        "system": "eidetic-plus-full",
        "tau": 0.7,
        "n": 4,
        "target": 0.95,
        "precision_at_tau": 0.0,
        "coverage_at_tau": 0.0,
        "log_fingerprint": {"combined_sha256": "a" * 64, "file_count": 1},
    }))

    report = _run_gate(tmp_path, min_abstention_calibration_samples=4)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "abstention_calibration:precision_at_tau" in failed
    assert "abstention_calibration:nonzero_coverage" in failed


def test_release_gate_fails_replayed_slice_despite_row_level_significance(tmp_path):
    _write_artifacts(tmp_path, runs=10)
    ids = _ids(2)
    by_system = {"eidetic-plus": [], "eidetic-plus-full": [], "rag-full": []}
    for run_idx in range(10):
        by_system["eidetic-plus"].append(_row("eidetic-plus", ids[0], run_idx, True,
                                              age_days=0.0))
        by_system["eidetic-plus"].append(_row("eidetic-plus", ids[1], run_idx, True,
                                              age_days=365.0))
        by_system["eidetic-plus-full"].append(_row("eidetic-plus-full", ids[0], run_idx, True,
                                                   verified=True, age_days=0.0))
        by_system["eidetic-plus-full"].append(_row("eidetic-plus-full", ids[1], run_idx, True,
                                                   verified=True, age_days=365.0))
        by_system["rag-full"].append(_row("rag-full", ids[0], run_idx, False,
                                          age_days=0.0))
        by_system["rag-full"].append(_row("rag-full", ids[1], run_idx, True,
                                          age_days=365.0))
    for system, rows in by_system.items():
        (tmp_path / f"{system}__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(
        tmp_path,
        min_runs=10,
        min_questions_per_system=2,
        min_questions_per_dataset_per_system=2,
        alpha=0.05,
        min_age_slope_points=2,
        min_overall_delta_pp=40.0,
        min_category_delta_pp=0.0,
    )

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "dominance:eidetic-plus:vs:rag-full:significance" not in failed
    assert "dominance:eidetic-plus:vs:rag-full:sample_clustered_discordants" in failed
    assert "dominance:eidetic-plus:vs:rag-full:sample_clustered_significance" in failed
    assert "dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_discordants" in failed
    assert "dominance:eidetic-plus:vs:rag-full:locomo/single-hop:sample_clustered_significance" in failed
    assert report["paired"]["rag-full"]["overall"]["headline_only"] == 10
    assert report["paired"]["rag-full"]["sample_clustered"]["headline_only"] == 1
    category = report["paired"]["rag-full"]["categories"]["locomo|single-hop"]
    assert category["row"]["headline_only"] == 10
    assert category["sample_clustered"]["headline_only"] == 1


def test_release_gate_counts_unique_samples_for_question_coverage(tmp_path):
    _write_artifacts(tmp_path, runs=10)
    ids = _ids(2)
    by_system = {"eidetic-plus": [], "eidetic-plus-full": [], "rag-full": []}
    for run_idx in range(10):
        for sid in ids:
            by_system["eidetic-plus"].append(_row("eidetic-plus", sid, run_idx, True,
                                                  age_days=float(run_idx)))
            by_system["eidetic-plus-full"].append(_row("eidetic-plus-full", sid, run_idx, True,
                                                       verified=True, age_days=float(run_idx)))
            by_system["rag-full"].append(_row("rag-full", sid, run_idx, False,
                                              age_days=float(run_idx)))
    for system, rows in by_system.items():
        (tmp_path / f"{system}__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(
        tmp_path,
        min_runs=10,
        min_questions_per_system=20,
        min_questions_per_dataset_per_system=20,
        min_dataset_accuracy=0.50,
        min_overall_delta_pp=0.0,
        min_category_delta_pp=0.0,
        min_age_slope_points=2,
    )

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "eidetic-plus:questions" in failed
    assert "eidetic-plus:locomo:questions" in failed
    assert "rag-full:questions" in failed


def test_release_gate_allows_explicit_empty_baseline_list(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    report = _run_gate(
        tmp_path,
        required_systems=["eidetic-plus-full"],
        baseline_systems=[],
        headline_system="eidetic-plus-full",
        integrity_system="eidetic-plus-full",
        token_efficiency_baseline="",
        min_overall_delta_pp=0.0,
        require_category_wins=False,
    )

    assert report["status"] == "PASS"
    assert report["paired"] == {}
    assert not any(c["name"].startswith("dominance:") for c in report["checks"])


def test_release_gate_fails_dev_manifest_or_dev_rows(tmp_path):
    _write_artifacts(tmp_path, split="dev")
    _write_logs(tmp_path)
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "manifest:split" in failed

    test_rows_dev_manifest = tmp_path / "dev_rows"
    _write_artifacts(test_rows_dev_manifest)
    _write_logs(test_rows_dev_manifest, sample_split="dev")
    report = _run_gate(test_rows_dev_manifest)
    failed = {c["name"] for c in report["failed_checks"]}
    assert "logs:held_out_split" in failed


def test_release_gate_fails_consolidation_timeouts(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path, timeout=1)
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "eidetic-plus:consolidation_timeouts" in failed


def test_release_gate_fails_snap_back_corruption(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "snap_back_audit.json").write_text(json.dumps({
        "status": "FAIL",
        "data_dir": str((tmp_path / "data").resolve()),
        "records_with_raw_blob": 8,
        "lossless_byte_identical": 7,
        "rate": 0.875,
        "rate_pct": 87.5,
        "min_records": 1,
        "failures": [{"memory_id": "m1", "content_hash": "bad", "error": "hash_mismatch"}],
    }))
    _write_logs(tmp_path)
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "snap_back:lossless" in failed
    assert "snap_back:no_failures" in failed


def test_release_gate_fails_missing_or_bad_baseline_reproduction(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "mem0_gate.json").unlink()
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "baseline_reproduction:valid_json" in failed
    assert "baseline_reproduction:status" in failed

    bad = tmp_path / "bad"
    _write_artifacts(bad)
    _write_logs(bad)
    (bad / "mem0_gate.json").write_text(json.dumps({
        "status": "FAIL",
        "system": "mem0",
        "dataset": "locomo",
        "total_n": 8,
        "comparisons": {"single-hop": {"status": "FAIL"}},
        "log_fingerprint": log_fingerprint(bad),
    }))
    report = _run_gate(bad)
    failed = {c["name"] for c in report["failed_checks"]}
    assert "baseline_reproduction:status" in failed
    assert "baseline_reproduction:comparisons" in failed


def test_release_gate_fails_stale_rendered_report_fingerprints(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    scoreboard = json.loads((tmp_path / "scoreboard.json").read_text())
    scoreboard["log_fingerprint"]["combined_sha256"] = "0" * 64
    (tmp_path / "scoreboard.json").write_text(json.dumps(scoreboard))
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "scoreboard:log_fingerprint_matches" in failed

    stale_mem0 = tmp_path / "stale_mem0"
    _write_artifacts(stale_mem0)
    _write_logs(stale_mem0)
    mem0_report = json.loads((stale_mem0 / "mem0_gate.json").read_text())
    mem0_report["log_fingerprint"]["combined_sha256"] = "1" * 64
    (stale_mem0 / "mem0_gate.json").write_text(json.dumps(mem0_report))
    report = _run_gate(stale_mem0)
    failed = {c["name"] for c in report["failed_checks"]}
    assert "baseline_reproduction:log_fingerprint_matches" in failed


def test_release_gate_fails_operating_budget(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path, headline_query_tokens=8000, headline_search_ms=750.0,
                headline_e2e_ms=6000.0)
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "operating:eidetic-plus:query_tokens_median" in failed
    assert "operating:eidetic-plus:search_p95_ms" in failed
    assert "operating:eidetic-plus:e2e_p50_ms" in failed
    assert "operating:eidetic-plus:token_efficiency_vs:rag-full" in failed


def test_release_gate_fails_age_fragility(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path, age_fragile=True)
    report = _run_gate(
        tmp_path,
        min_dataset_accuracy=0.50,
        min_overall_delta_pp=0.0,
        min_category_delta_pp=0.0,
        max_abs_recall_slope_per_year=0.01,
    )
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "operating:eidetic-plus:age_flatness" in failed


def test_release_gate_fails_per_dataset_undercoverage(tmp_path):
    _write_artifacts(tmp_path)
    locomo_ids = _ids(4)
    lme_ids = _ids(1)
    by_system = {"eidetic-plus": [], "eidetic-plus-full": [], "rag-full": []}
    for run_idx in (0, 1):
        for i, sid in enumerate(locomo_ids):
            age_days = float(i * 365)
            by_system["eidetic-plus"].append(_row("eidetic-plus", sid, run_idx, True,
                                                  age_days=age_days))
            by_system["eidetic-plus-full"].append(_row("eidetic-plus-full", sid, run_idx, True,
                                                       verified=True, age_days=age_days))
            by_system["rag-full"].append(_row("rag-full", sid, run_idx, i % 2 == 0,
                                              age_days=age_days))
        for sid in lme_ids:
            by_system["eidetic-plus"].append(_row("eidetic-plus", sid, run_idx, True,
                                                  dataset="longmemeval"))
            by_system["eidetic-plus-full"].append(_row("eidetic-plus-full", sid, run_idx, True,
                                                       verified=True, dataset="longmemeval"))
            by_system["rag-full"].append(_row("rag-full", sid, run_idx, False,
                                              dataset="longmemeval"))
    for system, rows in by_system.items():
        (tmp_path / f"{system}__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(
        tmp_path,
        required_datasets=["locomo", "longmemeval"],
        min_questions_per_system=10,
        min_questions_per_dataset_per_system=4,
        min_dataset_accuracy=0.50,
        min_overall_delta_pp=0.0,
        min_category_delta_pp=0.0,
    )
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "eidetic-plus:longmemeval:questions" in failed
    assert "rag-full:longmemeval:questions" in failed


def test_release_gate_fails_degraded_required_baseline_health(tmp_path):
    _write_artifacts(tmp_path)
    ids = _ids(4)
    degraded = {
        "status": "degraded",
        "system": "mem0",
        "missing_optional": ["spacy", "fastembed"],
    }
    by_system = {"eidetic-plus": [], "eidetic-plus-full": [], "rag-full": [], "mem0": []}
    for run_idx in (0, 1):
        for i, sid in enumerate(ids):
            age_days = float(i * 365)
            by_system["eidetic-plus"].append(_row("eidetic-plus", sid, run_idx, True,
                                                  age_days=age_days))
            by_system["eidetic-plus-full"].append(_row("eidetic-plus-full", sid, run_idx, True,
                                                       verified=True, age_days=age_days))
            by_system["rag-full"].append(_row("rag-full", sid, run_idx, i % 2 == 0,
                                              age_days=age_days))
            by_system["mem0"].append(_row("mem0", sid, run_idx, i % 2 == 0,
                                          age_days=age_days, baseline_health=degraded))
    for system, rows in by_system.items():
        (tmp_path / f"{system}__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _write_fingerprinted_reports(tmp_path)

    report = _run_gate(
        tmp_path,
        required_systems=["eidetic-plus", "eidetic-plus-full", "rag-full", "mem0"],
        baseline_systems=["rag-full"],
        min_dataset_accuracy=0.50,
        min_overall_delta_pp=0.0,
        min_category_delta_pp=0.0,
        health_required_systems=["mem0"],
    )
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "competitor_health:mem0" in failed


def test_release_gate_fails_missing_or_unsupported_claim_scope(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "claim_scope.json").unlink()
    report = _run_gate(tmp_path)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "claim_scope:valid_json" in failed

    bad = tmp_path / "bad_scope"
    _write_artifacts(bad)
    _write_logs(bad)
    (bad / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["mem0"],
        "limitations": [],
    }))
    report = _run_gate(bad)
    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "claim_scope:no_unsupported_sota" in failed
    assert "claim_scope:external_names_have_evidence" in failed

    names_only = tmp_path / "names_only_sota"
    _write_artifacts(names_only)
    _write_logs(names_only)
    (names_only / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["chronos", "mastra"],
        "limitations": [],
    }))
    report = _run_gate(names_only)
    failed = {c["name"] for c in report["failed_checks"]}
    assert "claim_scope:no_unsupported_sota" in failed
    assert "claim_scope:external_names_have_evidence" in failed


def test_release_gate_rejects_top_systems_claimed_as_harness_without_logs(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": [
            "eidetic-plus", "rag-full", "chronos", "mastra", "byterover", "hindsight",
        ],
        "measured_external_systems": [],
        "limitations": [],
    }))

    report = _run_gate(tmp_path)

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "claim_scope:harness_names_have_logs" in failed
    assert "claim_scope:no_unsupported_sota" in failed


def test_release_gate_requires_top_external_evidence_for_each_required_dataset(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["chronos"],
        "external_system_evidence": [{
            "system": "chronos",
            "dataset": "locomo",
            "split": "test",
            "n": 100,
            "runs": 1,
            "score": 0.9,
            "metric": "verified_accuracy",
            "evaluation_protocol": "same fixed reader and judge where applicable",
            "date": "2026-06-29",
            "source": "paper",
            "artifact_fingerprint": "f" * 64,
        }],
        "limitations": [],
    }))

    report = _run_gate(
        tmp_path,
        top_systems_for_sota=["chronos"],
        required_datasets=["locomo", "longmemeval"],
        required_categories_by_dataset={"locomo": ["single-hop"]},
    )

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "FAIL"
    assert "claim_scope:top_system_dataset_coverage" in failed


def test_release_gate_rejects_tiny_top_external_evidence(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["chronos"],
        "external_system_evidence": [{
            "system": "chronos",
            "dataset": "locomo",
            "split": "test",
            "n": 12,
            "runs": 1,
            "score": 0.9,
            "metric": "verified_accuracy",
            "evaluation_protocol": "same fixed reader and judge where applicable",
            "date": "2026-06-29",
            "source": "paper",
            "artifact_fingerprint": "f" * 64,
        }],
        "limitations": [],
    }))

    report = _run_gate(tmp_path, top_systems_for_sota=["chronos"])

    failed = {c["name"] for c in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert report["status"] == "FAIL"
    assert "claim_scope:external_evidence_valid" in failed
    assert "claim_scope:external_names_have_evidence" in failed
    assert "claim_scope:no_unsupported_sota" in failed
    assert "n 12 < required 100" in details


def test_release_gate_rejects_top_external_evidence_without_sha_fingerprint(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["chronos"],
        "external_system_evidence": [{
            "system": "chronos",
            "dataset": "locomo",
            "split": "test",
            "n": 100,
            "runs": 1,
            "score": 0.9,
            "metric": "verified_accuracy",
            "evaluation_protocol": "same fixed reader and judge where applicable",
            "date": "2026-06-29",
            "source": "paper",
            "artifact_fingerprint": "not-a-digest",
        }],
        "limitations": [],
    }))

    report = _run_gate(tmp_path, top_systems_for_sota=["chronos"])

    failed = {c["name"] for c in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert report["status"] == "FAIL"
    assert "claim_scope:external_evidence_valid" in failed
    assert "claim_scope:external_names_have_evidence" in failed
    assert "claim_scope:no_unsupported_sota" in failed
    assert "artifact_fingerprint must be a sha256 hex digest" in details


def test_release_gate_rejects_top_external_evidence_without_metric_protocol(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    (tmp_path / "claim_scope.json").write_text(json.dumps({
        "public_claim_scope": "best-in-world",
        "measured_harness_systems": ["eidetic-plus", "rag-full"],
        "measured_external_systems": ["chronos"],
        "external_system_evidence": [{
            "system": "chronos",
            "dataset": "locomo",
            "split": "test",
            "n": 100,
            "runs": 1,
            "score": 0.9,
            "date": "2026-06-29",
            "source": "paper",
            "artifact_fingerprint": "f" * 64,
        }],
        "limitations": [],
    }))

    report = _run_gate(tmp_path, top_systems_for_sota=["chronos"])

    failed = {c["name"] for c in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert report["status"] == "FAIL"
    assert "claim_scope:external_evidence_valid" in failed
    assert "claim_scope:external_names_have_evidence" in failed
    assert "claim_scope:no_unsupported_sota" in failed
    assert "missing evaluation_protocol, metric" in details


def test_release_gate_requires_headline_to_meet_external_top_score(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    _lower_headline_accuracy(tmp_path)
    _write_best_world_scope(tmp_path, score=0.95)

    report = _run_gate(
        tmp_path,
        top_systems_for_sota=["chronos"],
        min_dataset_accuracy=0.50,
        min_overall_delta_pp=20.0,
        min_category_delta_pp=20.0,
        alpha=1.01,
        max_abs_recall_slope_per_year=1.0,
    )

    failed = {c["name"] for c in report["failed_checks"]}
    details = json.dumps(report["failed_checks"])
    assert report["status"] == "FAIL"
    assert "claim_scope:top_system_score_floor" in failed
    assert "headline 0.875<external 0.950" in details


def test_release_gate_accepts_external_percent_score_below_headline(tmp_path):
    _write_artifacts(tmp_path)
    _write_logs(tmp_path)
    _lower_headline_accuracy(tmp_path)
    _write_best_world_scope(tmp_path, score=87.0)

    report = _run_gate(
        tmp_path,
        top_systems_for_sota=["chronos"],
        min_dataset_accuracy=0.50,
        min_overall_delta_pp=20.0,
        min_category_delta_pp=20.0,
        alpha=1.01,
        max_abs_recall_slope_per_year=1.0,
    )

    failed = {c["name"] for c in report["failed_checks"]}
    assert report["status"] == "PASS"
    assert "claim_scope:top_system_score_floor" not in failed
