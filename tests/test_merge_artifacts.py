from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bench.datasets import split_of
from bench.fingerprints import log_fingerprint
from bench.merge_artifacts import merge_artifacts
from bench.release_gate import run_release_gate


def _ids(n: int, *, split: str = "test", prefix: str = "merge_slice") -> list[str]:
    out: list[str] = []
    i = 0
    while len(out) < n:
        sid = f"{prefix}_{i}_q0"
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


def _row(system: str, dataset: str, category: str, sample_id: str, run_idx: int,
         correct: bool, *, verified: bool | None = None) -> dict:
    extra = {}
    if verified is not None:
        extra["verified"] = verified
        if verified:
            proof_id = f"mem_{system}_{dataset}_{sample_id}_{run_idx}"
            proof_hash = hashlib.sha256(proof_id.encode("utf-8")).hexdigest()
            extra.update({
                "citations": 1,
                "candidate_memory_ids": [proof_id],
                "entailed_memory_ids": [proof_id],
                "entailed_content_hashes": [proof_hash],
                "entailed_raw_uris": [f"cas://{proof_hash}"],
                "proof_surface_tokens": 12,
            })
    if system.startswith("eidetic"):
        extra.update({
            "structured_recall": True,
            "smqe_operator": "latest_value",
            "smqe_backend": "claim",
            "smqe_policy": "smqe:latest_value:claim",
            "policy": "smqe:latest_value:claim",
            "region_hint_count": 0,
            "region_ids": [],
            "region_member_ids": [],
        })
    return {
        "system": system,
        "dataset": dataset,
        "category": category,
        "sample_id": sample_id,
        "question": "q",
        "gold": "g",
        "predicted": "g" if correct else "x",
        "correct": correct,
        "write_tokens": 10,
        "query_tokens": 5 if system == "eidetic-plus" else 100,
        "search_ms": 1.0,
        "e2e_ms": 2.0,
        "abstained": False,
        "run_idx": run_idx,
        "age_days": float(run_idx + (0 if dataset == "locomo" else 365)),
        "n_sessions": 1,
        "extra": extra,
        "error": "",
    }


def _proof_hashes_from_rows(rows_by_system: dict[str, list[dict]]) -> list[str]:
    hashes: set[str] = set()
    for rows in rows_by_system.values():
        for row in rows:
            extra = row.get("extra") or {}
            if not isinstance(extra, dict) or not extra.get("verified"):
                continue
            for h in extra.get("entailed_content_hashes", []) or []:
                h = str(h).strip().lower()
                if len(h) == 64:
                    hashes.add(h)
    return sorted(hashes)


def _write_source(path: Path, *, dataset: str, category: str, sample_id: str,
                  render_only: bool = False) -> Path:
    path.mkdir(parents=True)
    rows_by_system = {
        "eidetic-plus": [
            _row("eidetic-plus", dataset, category, sample_id, 0, True, verified=True),
            _row("eidetic-plus", dataset, category, sample_id, 1, True, verified=True),
        ],
        "rag-full": [
            _row("rag-full", dataset, category, sample_id, 0, False),
            _row("rag-full", dataset, category, sample_id, 1, False),
        ],
    }
    for system, rows in rows_by_system.items():
        (path / f"{system}__run0.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    data_dir = (path / "data").resolve()
    samples_file = path / "release.samples.json"
    samples_file.write_text(json.dumps([{"dataset": dataset, "sample_id": sample_id}]))
    slice_ids = _ids(5, prefix=f"{sample_id}_slice")
    (path / "slice_invariant.json").write_text(json.dumps({
        "pass": True,
        "dataset": dataset,
        "split": "test",
        "holdout_profile": "holdout",
        "draws": 5,
        "subset": 1,
        "seed": 424242,
        "seed_mode": "random",
        "draw_seeds": [424242 + draw for draw in range(5)],
        "pool_unique_sample_ids": len(slice_ids),
        "required_unique_sample_ids": 5,
        "unique_sample_ids": len(slice_ids),
        "system_under_test": "eidetic-plus",
        "runs": [
            {
                "draw": draw + 1,
                "seed": 424242 + draw,
                "sample_ids": [slice_ids[draw]],
                "executed": True,
                "returncode": 0,
                "score": {
                    "pass": True,
                    "verified": True,
                    "verified_correct": 1,
                    "correct": 1,
                    "total": 1,
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
    (path / "snap_back_audit.json").write_text(json.dumps({
        "status": "PASS",
        "data_dir": str(data_dir),
        "records_with_raw_blob": 2,
        "lossless_byte_identical": 2,
        "rate": 1.0,
        "rate_pct": 100.0,
        "min_records": 1,
        "audited_content_hashes": _proof_hashes_from_rows(rows_by_system),
        "failures": [],
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
        "system": "eidetic-plus",
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
        "artifact_fingerprints": [{"combined_sha256": f"unit-ablation-{dataset}-{sample_id}"}],
    }))
    (path / "affect_salience_invariant.json").write_text(json.dumps(_affect_salience_report()))
    (path / "scratchpad_invariant.json").write_text(json.dumps(_scratchpad_report()))
    (path / "region_routing_invariant.json").write_text(json.dumps(_region_routing_report()))
    (path / "reflex_recall_invariant.json").write_text(json.dumps(_reflex_recall_report()))
    (path / "smqe_planner_invariant.json").write_text(json.dumps(_smqe_planner_report()))
    (path / "run_manifest.json").write_text(json.dumps({
        "systems": "eidetic-plus,rag-full",
        "dataset": dataset,
        "split": "test",
        "subset": 0,
        "runs": 2,
        "run_offset": 0,
        "samples_file": str(samples_file),
        "holdout_profile": "holdout",
        "render_only": render_only,
        "judge": {"judge_model": "unit-judge", "judge_backend": "fake"},
        "sample_count": 1,
        "category_counts": {category: 1},
        "sample_rows": [{"dataset": dataset, "category": category, "sample_id": sample_id}],
        "env": {
            "READER_MODEL": "unit-reader",
            "READER_MODE": "default",
            "JUDGE_MODEL": "unit-judge",
            "JUDGE_BASE_URL": "",
            "DATA_DIR": str(data_dir),
        },
    }))
    return path


def test_merge_artifacts_builds_release_gate_visible_composite(tmp_path):
    locomo = _write_source(
        tmp_path / "locomo",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    lme = _write_source(
        tmp_path / "lme",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    out = tmp_path / "composite"

    manifest_path = merge_artifacts([locomo, lme], out)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["artifact_kind"] == "composite"
    assert manifest["render_only"] is False
    assert manifest["dataset"] == "both"
    assert manifest["runs"] == 2
    assert manifest["sample_count"] == 2
    assert len(manifest["composite_sources"]) == 2
    assert (out / "scoreboard.md").exists()
    assert (out / "recall_vs_age.png").exists()
    snap = json.loads((out / "snap_back_audit.json").read_text())
    assert snap["status"] == "PASS"
    assert snap["records_with_raw_blob"] == 4
    holdout = json.loads((out / "holdout_audit.json").read_text())
    assert holdout["pass"] is True
    assert holdout["holdout_needles_checked"] == 16
    assert holdout["findings"] == []
    ablation = json.loads((out / "ablation_report.json").read_text())
    assert ablation["pass"] is True
    assert ablation["system"] == "eidetic-plus"
    assert ablation["split"] == "dev"
    assert ablation["full"]["n"] == 48
    assert ablation["ablations"]["regions_off"]["verified_accuracy"] == 0.84
    assert ablation["ablations"]["forgetting_off"]["query_tokens_median"] == 130
    assert ablation["ablations"]["affect_off"]["verified_accuracy"] == 0.86
    affect_salience = json.loads((out / "affect_salience_invariant.json").read_text())
    assert affect_salience["pass"] is True
    assert affect_salience["cases"] == 48
    assert affect_salience["checks"] == 336
    assert affect_salience["correct"] == 336
    assert affect_salience["flip_checks"] == 96
    assert affect_salience["age_free_checks"] == 48
    assert affect_salience["bounded_checks"] == 48
    scratchpad = json.loads((out / "scratchpad_invariant.json").read_text())
    assert scratchpad["pass"] is True
    assert scratchpad["cases"] == 48
    assert scratchpad["checks"] == 528
    assert scratchpad["correct"] == 528
    assert scratchpad["proof_link_checks"] == 192
    assert scratchpad["retrieval_channel_checks"] == 192
    region_routing = json.loads((out / "region_routing_invariant.json").read_text())
    assert region_routing["pass"] is True
    assert region_routing["cases"] == 48
    assert region_routing["checks"] == 576
    assert region_routing["correct"] == 576
    assert region_routing["dense_miss_recovery_checks"] == 144
    assert region_routing["proof_link_checks"] == 96
    assert region_routing["telemetry_trace_checks"] == 96
    reflex = json.loads((out / "reflex_recall_invariant.json").read_text())
    assert reflex["pass"] is True
    assert reflex["cases"] == 48
    assert reflex["checks"] == 576
    assert reflex["correct"] == 576
    assert reflex["direct_hit_checks"] == 96
    assert reflex["coactivation_checks"] == 96
    assert reflex["p95_latency_ms"] == 2.0
    planner = json.loads((out / "smqe_planner_invariant.json").read_text())
    assert planner["pass"] is True
    assert planner["cases"] == 48
    assert planner["checks"] == 288
    assert planner["correct"] == 288
    assert planner["generic_term_checks"] == 48
    assert planner["operator_counts"]["open_inference"] == 4
    assert planner["p95_latency_ms"] == 1.0
    fullpath = json.loads((out / "smqe_fullpath_invariant.json").read_text())
    assert fullpath["pass"] is True
    assert fullpath["cases"] == 54
    assert fullpath["verified"] == 54
    assert fullpath["structured_recall"] == 54
    assert fullpath["reader_calls"] == 0
    assert fullpath["proof_link_checks"] == 54
    assert fullpath["claim_backend_correct"] == 54
    assert sum(fullpath["case_operator_counts"].values()) == 54
    assert min(fullpath["case_operator_counts"].values()) >= 6
    assert fullpath["latency_budget_checks"] == 54
    assert fullpath["p95_latency_ms"] == 2.0
    claim_coverage = json.loads((out / "smqe_claim_coverage.json").read_text())
    assert claim_coverage["pass"] is True
    assert claim_coverage["cases"] == 54
    assert claim_coverage["claim_backend_correct"] == 54
    assert claim_coverage["claim_backend_operator_counts"] == claim_coverage["operator_counts"]
    abstention = json.loads((out / "smqe_abstention_invariant.json").read_text())
    assert abstention["pass"] is True
    assert abstention["cases"] == 48
    assert abstention["record_only_abstained"] == 48
    assert abstention["claims_present_abstained"] == 48
    scope = json.loads((out / "smqe_scope_invariant.json").read_text())
    assert scope["pass"] is True
    assert scope["cases"] == 48
    assert scope["checks"] == 192
    assert scope["record_backend_correct"] == 96
    assert scope["claim_backend_correct"] == 96
    composition = json.loads((out / "smqe_composition_invariant.json").read_text())
    assert composition["pass"] is True
    assert composition["cases"] == 48
    assert composition["checks"] == 96
    assert composition["record_backend_correct"] == 48
    assert composition["claim_backend_correct"] == 48
    relative_phrase = json.loads((out / "smqe_relative_phrase_invariant.json").read_text())
    assert relative_phrase["pass"] is True
    assert relative_phrase["cases"] == 48
    assert relative_phrase["checks"] == 96
    assert relative_phrase["record_backend_correct"] == 48
    assert relative_phrase["claim_backend_correct"] == 48
    temporal_window = json.loads((out / "smqe_temporal_window_invariant.json").read_text())
    assert temporal_window["pass"] is True
    assert temporal_window["cases"] == 48
    assert temporal_window["checks"] == 96
    assert temporal_window["record_backend_correct"] == 48
    assert temporal_window["claim_backend_correct"] == 48
    attribution = json.loads((out / "smqe_attribution_invariant.json").read_text())
    assert attribution["pass"] is True
    assert attribution["cases"] == 48
    assert attribution["checks"] == 96
    assert attribution["record_backend_correct"] == 48
    assert attribution["claim_backend_correct"] == 48
    subscope = json.loads((out / "smqe_subscope_invariant.json").read_text())
    assert subscope["pass"] is True
    assert subscope["cases"] == 48
    assert subscope["checks"] == 192
    assert subscope["record_backend_correct"] == 96
    assert subscope["claim_backend_correct"] == 96
    time = json.loads((out / "smqe_time_invariant.json").read_text())
    assert time["pass"] is True
    assert time["cases"] == 48
    assert time["checks"] == 192
    assert time["record_backend_correct"] == 96
    assert time["claim_backend_correct"] == 96
    invalidation = json.loads((out / "smqe_invalidation_invariant.json").read_text())
    assert invalidation["pass"] is True
    assert invalidation["cases"] == 48
    assert invalidation["checks"] == 192
    assert invalidation["record_backend_correct"] == 96
    assert invalidation["claim_backend_correct"] == 96
    assert invalidation["preference_supersession_pass"] is True
    assert invalidation["preference_supersession_cases"] == 6
    assert invalidation["preference_supersession_checks"] == 24
    assert invalidation["preference_supersession_correct"] == 24
    assert invalidation["preference_supersession_record_correct"] == 12
    assert invalidation["preference_supersession_claim_correct"] == 12

    report = run_release_gate(
        out,
        required_systems=["eidetic-plus", "rag-full"],
        baseline_systems=["rag-full"],
        required_datasets=["locomo", "longmemeval"],
        required_categories_by_dataset={
            "locomo": ["single-hop"],
            "longmemeval": ["multi-session"],
        },
        headline_system="eidetic-plus",
        integrity_system="eidetic-plus",
        min_runs=2,
        min_questions_per_system=2,
        min_questions_per_dataset_per_system=1,
        min_category_questions_per_system=1,
        min_dataset_accuracy=0.90,
        min_overall_delta_pp=40.0,
        min_category_delta_pp=0.0,
        alpha=1.0,
        max_dataset_accuracy_ci_width_pp=100.0,
        max_category_accuracy_ci_width_pp=100.0,
        require_ci_clear_dominance=False,
        min_clustered_discordant_samples=1,
        min_verified_accuracy=0.90,
        max_search_p95_ms=1000.0,
        max_e2e_p50_ms=1000.0,
        min_age_slope_points=2,
        require_baseline_reproduction=False,
        require_competitor_health=False,
        require_claim_scope=False,
        require_abstention_calibration=False,
        require_category_wins=False,
        min_slice_invariant_subset=1,
    )
    failed = {check["name"] for check in report["failed_checks"]}
    assert report["status"] == "PASS"
    assert "manifest:composite_source_fingerprints_match" not in failed
    assert report["log_fingerprint"] == log_fingerprint(out)


def test_merge_artifacts_marks_nonrandom_composite_sidecar_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "affect_salience_invariant.json").read_text())
    sidecar["seed_mode"] = "fixed"
    (first / "affect_salience_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "affect_salience_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["seed_mode"] == "mixed"
    assert "seed_mode:mixed" in composite["failures"]


def test_merge_artifacts_marks_smqe_fullpath_case_operator_gap_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "smqe_fullpath_invariant.json").read_text())
    sidecar["case_operator_counts"]["latest_value"] = 1
    sidecar["case_operator_counts"]["count_aggregate"] = 5
    (first / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "smqe_fullpath_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["case_operator_counts"]["latest_value"] == 4
    assert any(source["pass"] is False for source in composite["sources"])


def test_merge_artifacts_marks_smqe_fullpath_context_bloat_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "smqe_fullpath_invariant.json").read_text())
    sidecar["avg_context_tokens"] = 180.0
    (first / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "smqe_fullpath_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["avg_context_tokens"] > 80.0
    assert any(source["avg_context_tokens"] == 180.0 and source["pass"] is False
               for source in composite["sources"])


def test_merge_artifacts_marks_smqe_fullpath_slow_source_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "smqe_fullpath_invariant.json").read_text())
    sidecar["p95_latency_ms"] = 150.0
    (first / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "smqe_fullpath_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["p95_latency_ms"] == 150.0
    assert any(source["p95_latency_ms"] == 150.0 and source["pass"] is False
               for source in composite["sources"])


def test_merge_artifacts_marks_smqe_fullpath_missing_proof_links_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "smqe_fullpath_invariant.json").read_text())
    sidecar["proof_link_checks"] = 26
    (first / "smqe_fullpath_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "smqe_fullpath_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["proof_link_checks"] == 53
    assert any(source["proof_link_checks"] == 26 and source["pass"] is False
               for source in composite["sources"])


def test_merge_artifacts_can_filter_systems_per_source(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    third = _write_source(
        tmp_path / "third",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    fourth = _write_source(
        tmp_path / "fourth",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    out = tmp_path / "composite"

    manifest_path = merge_artifacts(
        [first, second, third, fourth],
        out,
        source_system_filters={
            0: {"eidetic-plus"},
            1: {"rag-full"},
            2: {"eidetic-plus"},
            3: {"rag-full"},
        },
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["systems"] == "eidetic-plus,rag-full"
    assert manifest["composite_sources"][0]["system_filter"] == ["eidetic-plus"]
    assert manifest["composite_sources"][1]["system_filter"] == ["rag-full"]
    assert sorted(manifest["copied_logs"]) == [
        "src0_first__eidetic-plus__run0.jsonl",
        "src1_second__rag-full__run0.jsonl",
        "src2_third__eidetic-plus__run0.jsonl",
        "src3_fourth__rag-full__run0.jsonl",
    ]
    snap = json.loads((out / "snap_back_audit.json").read_text())
    assert snap["status"] == "PASS"
    assert snap["records_with_raw_blob"] == 4
    holdout = json.loads((out / "holdout_audit.json").read_text())
    assert holdout["pass"] is True
    assert holdout["holdout_needles_checked"] == 32
    assert [source["status"] for source in snap["sources"]] == [
        "PASS",
        "SKIP",
        "PASS",
        "SKIP",
    ]

    report = run_release_gate(
        out,
        required_systems=["eidetic-plus", "rag-full"],
        baseline_systems=["rag-full"],
        required_datasets=["locomo"],
        required_categories_by_dataset={"locomo": ["single-hop"]},
        headline_system="eidetic-plus",
        integrity_system="eidetic-plus",
        min_runs=2,
        min_questions_per_system=2,
        min_questions_per_dataset_per_system=1,
        min_category_questions_per_system=1,
        min_dataset_accuracy=0.90,
        min_overall_delta_pp=40.0,
        min_category_delta_pp=0.0,
        alpha=1.0,
        max_dataset_accuracy_ci_width_pp=100.0,
        max_category_accuracy_ci_width_pp=100.0,
        require_ci_clear_dominance=False,
        min_clustered_discordant_samples=1,
        min_verified_accuracy=0.90,
        max_search_p95_ms=1000.0,
        max_e2e_p50_ms=1000.0,
        min_age_slope_points=2,
        require_baseline_reproduction=False,
        require_competitor_health=False,
        require_claim_scope=False,
        require_abstention_calibration=False,
        require_category_wins=False,
        min_slice_invariant_subset=1,
    )
    assert report["status"] == "PASS"


def test_merge_artifacts_marks_old_slice_sidecar_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "slice_invariant.json").read_text())
    sidecar.pop("holdout_profile")
    (first / "slice_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "slice_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["sources"][0]["pass"] is False
    assert "report1:holdout_profile:<missing>" in composite["sources"][0]["failures"]


def test_merge_artifacts_marks_wrong_split_slice_sample_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "slice_invariant.json").read_text())
    sidecar["runs"][0]["sample_ids"][0] = _ids(1, split="dev", prefix="wrong_split")[0]
    (first / "slice_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "slice_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["sources"][0]["pass"] is False
    assert any("sample_split:test" in item for item in composite["sources"][0]["failures"])


def test_merge_artifacts_marks_low_unique_slice_coverage_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "slice_invariant.json").read_text())
    repeated = _ids(1, split="test", prefix="low_unique")
    for run in sidecar["runs"]:
        run["sample_ids"] = repeated
    (first / "slice_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "slice_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["sources"][0]["pass"] is False
    assert "report1:unique_sample_ids:1<required:5" in composite["sources"][0]["failures"]


def test_merge_artifacts_marks_insufficient_declared_slice_pool_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "slice_invariant.json").read_text())
    sidecar["pool_unique_sample_ids"] = 4
    (first / "slice_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "slice_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["sources"][0]["pass"] is False
    assert "report1:pool_unique_sample_ids:4<required:5" in composite["sources"][0]["failures"]


def test_merge_artifacts_marks_duplicate_ids_within_slice_draw_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    ids = _ids(11, split="test", prefix="draw_duplicate")
    sidecar = json.loads((first / "slice_invariant.json").read_text())
    sidecar["subset"] = 2
    sidecar["pool_unique_sample_ids"] = 11
    sidecar["required_unique_sample_ids"] = 10
    sidecar["unique_sample_ids"] = 10
    for idx, run in enumerate(sidecar["runs"]):
        run["sample_ids"] = ids[idx * 2:idx * 2 + 2]
        run["score"] = {
            "pass": True,
            "verified": True,
            "verified_correct": 2,
            "correct": 2,
            "total": 2,
        }
    sidecar["runs"][0]["sample_ids"] = [ids[0], ids[0]]
    sidecar["runs"][1]["sample_ids"].append(ids[10])
    (first / "slice_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "slice_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["sources"][0]["pass"] is False
    assert "report1:draw1:unique_sample_ids:1<required:2" in composite["sources"][0]["failures"]


def test_merge_artifacts_marks_correct_only_slice_score_nonpassing(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_1_q0",
    )
    sidecar = json.loads((first / "slice_invariant.json").read_text())
    sidecar["runs"][0]["score"] = {"pass": True, "correct": 1, "total": 1}
    (first / "slice_invariant.json").write_text(json.dumps(sidecar))

    merge_artifacts([first, second], tmp_path / "composite")
    composite = json.loads((tmp_path / "composite" / "slice_invariant.json").read_text())

    assert composite["pass"] is False
    assert composite["sources"][0]["pass"] is False
    assert "report1:draw1:score:not_verified" in composite["sources"][0]["failures"]


def test_merge_artifacts_tolerates_blank_env_values(tmp_path):
    locomo = _write_source(
        tmp_path / "locomo",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    lme = _write_source(
        tmp_path / "lme",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )
    manifest_path = lme / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["READER_MODE"] = ""
    manifest_path.write_text(json.dumps(manifest))

    merged = json.loads(merge_artifacts([locomo, lme], tmp_path / "composite").read_text())

    assert merged["env"]["READER_MODE"] == "default"


def test_merge_artifacts_rejects_render_only_source(tmp_path):
    locomo = _write_source(
        tmp_path / "locomo",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
        render_only=True,
    )
    lme = _write_source(
        tmp_path / "lme",
        dataset="longmemeval",
        category="multi-session",
        sample_id="release_test_1_q0",
    )

    with pytest.raises(ValueError, match="render_only"):
        merge_artifacts([locomo, lme], tmp_path / "composite")


def test_merge_artifacts_rejects_duplicate_row_identity(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )

    with pytest.raises(ValueError, match="duplicate row identity"):
        merge_artifacts([first, second], tmp_path / "composite")


def test_merge_artifacts_failure_leaves_existing_output_untouched(tmp_path):
    first = _write_source(
        tmp_path / "first",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    second = _write_source(
        tmp_path / "second",
        dataset="locomo",
        category="single-hop",
        sample_id="release_test_0_q0",
    )
    out = tmp_path / "composite"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("keep me\n")

    with pytest.raises(ValueError, match="duplicate row identity"):
        merge_artifacts([first, second], out, overwrite=True)

    assert sentinel.read_text() == "keep me\n"
    assert sorted(path.name for path in out.iterdir()) == ["sentinel.txt"]
