"""Public-release gate for benchmark artifacts.

This is the "do we get to say it publicly?" check. It reads real harness JSONL logs plus the
manifest and fails closed unless the artifact directory proves a held-out, multi-run benchmark with
the required systems, datasets, dominance margins, integrity row, and healthy consolidation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .compare import _load_logs_strict
from .datasets import split_of
from .fingerprints import log_fingerprint
from .scoreboard import _mcnemar_pvalue, _wilson_ci, aggregate, consolidation_rollup

_DEFAULT_REQUIRED_CATEGORIES = {
    "longmemeval": [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "knowledge-update",
        "temporal-reasoning",
    ],
    "locomo": ["single-hop", "multi-hop", "temporal", "open-domain"],
}


def _csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _parse_required_categories(value: str) -> dict[str, list[str]]:
    """Parse `dataset:cat|cat,dataset:cat|cat` CLI syntax."""
    out: dict[str, list[str]] = {}
    for item in _csv(value):
        if ":" not in item:
            raise ValueError("--required-categories entries must look like dataset:cat|cat")
        dataset, raw_cats = item.split(":", 1)
        cats = [cat.strip() for cat in raw_cats.split("|") if cat.strip()]
        if dataset.strip() and cats:
            out[dataset.strip()] = cats
    return out


def _required_categories_for(
    required_datasets: list[str],
    required_categories_by_dataset: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    if required_categories_by_dataset is not None:
        source = required_categories_by_dataset
    else:
        source = _DEFAULT_REQUIRED_CATEGORIES
    out: dict[str, list[str]] = {}
    for dataset in required_datasets:
        cats = [
            str(cat).strip()
            for cat in (source.get(dataset, []) or [])
            if str(cat).strip()
        ]
        if cats:
            out[dataset] = list(dict.fromkeys(cats))
    return out


def _load_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {out_dir}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("run_manifest.json must be a JSON object.")
    return data


def _artifact_checks(out_dir: Path) -> list[dict]:
    required = ["run_manifest.json", "scoreboard.md", "scoreboard.json",
                "recall_vs_age.png", "latency_vs_age.png", "snap_back_audit.json"]
    checks = []
    for name in required:
        path = Path(out_dir) / name
        checks.append({"name": f"artifact:{name}", "pass": path.exists(),
                       "detail": "present" if path.exists() else "missing"})
    return checks


_LEGACY_POLICY_BITS = (
    "product-" + "source-scan",
    "long" + "memeval-direct",
    "locomo-" + "direct-fact",
    "open-domain-bridge-" + "source-scan",
    "direct-fact-" + "source-scan",
)
_SMQE_ALLOWED_OPERATORS = {
    "count_aggregate",
    "event_order",
    "latest_value",
    "multi_session_sum",
    "open_inference",
    "preference_synth",
    "relative_temporal",
    "speaker_fact",
    "table_lookup",
    "temporal_delta",
}
_SMQE_REQUIRED_SYNTHETIC_OPS = {
    "count_aggregate",
    "latest_value",
    "multi_session_sum",
    "open_inference",
    "preference_synth",
    "relative_temporal",
    "speaker_fact",
    "table_lookup",
    "temporal_delta",
}
_SMQE_REQUIRED_PLANNER_OPS = _SMQE_REQUIRED_SYNTHETIC_OPS | {"event_order"}
_SMQE_ALLOWED_BACKENDS = {"claim", "record"}
_METABOLISM_ABLATION_KEYS = (
    "metabolism_off",
    "consolidation_off",
    "memory_off",
    "metabolism_memory_off",
)
_FORGETTING_ABLATION_KEYS = (
    "forgetting_off",
    "fsrs_off",
    "priority_forgetting_off",
)
_AFFECT_ABLATION_KEYS = (
    "affect_off",
    "salience_off",
    "affective_salience_off",
    "affect_salience_off",
)
_REGION_ABLATION_KEYS = (
    "regions_off",
    "region_off",
    "memory_regions_off",
    "cocoon_off",
    "gist_channel_off",
)
_ACCURACY_KEYS = (
    "verified_accuracy",
    "accuracy",
    "score",
    "correct_rate",
    "exact_match",
)
_COST_KEYS = (
    "query_tokens_median",
    "query_tokens_mean",
    "query_tokens_p50",
    "total_tokens_median",
    "tokens_median",
    "cost_median",
    "dollars_per_1k_queries",
    "search_p95_ms",
    "e2e_p50_ms",
    "index_entries",
    "index_tokens",
    "storage_bytes",
)


def _legacy_policy_rows(rows: list[dict]) -> list[str]:
    findings: list[str] = []
    for row in rows:
        extra = row.get("extra") or {}
        hay = " ".join(
            str(extra.get(key, "") or "")
            for key in ("policy", "note", "smqe_policy")
        ).lower()
        hit = next((bit for bit in _LEGACY_POLICY_BITS if bit in hay), "")
        if hit:
            findings.append(
                f"{row.get('system', '<system>')}:{row.get('sample_id', '<sample>')}:{hit}"
            )
    return findings


def _smqe_policy_note(extra: dict) -> str:
    raw = str(extra.get("smqe_policy") or "")
    if raw:
        return raw
    for key in ("policy", "note"):
        value = str(extra.get(key, "") or "")
        if value.startswith("smqe:"):
            return value
    return ""


def _smqe_log_policy_summary(rows: list[dict], *, system: str,
                             min_structured_rate: float,
                             min_claim_backend_rate: float) -> dict:
    relevant = [
        row for row in rows
        if row.get("system") == system and not row.get("error")
    ]
    dirty: list[str] = []
    structured = 0
    claim = 0
    record = 0
    fallback = 0
    operators: dict[str, int] = defaultdict(int)
    for row in relevant:
        extra = row.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        note = _smqe_policy_note(extra)
        structured_row = bool(extra.get("structured_recall")) or note.startswith("smqe:")
        if not structured_row:
            fallback += 1
            continue
        structured += 1
        parts = note.split(":")
        op = parts[1] if len(parts) >= 3 and parts[0] == "smqe" else ""
        backend = parts[2] if len(parts) >= 3 and parts[0] == "smqe" else ""
        declared_op = str(extra.get("smqe_operator") or op)
        declared_backend = str(extra.get("smqe_backend") or backend)
        ok = (
            len(parts) >= 3
            and parts[0] == "smqe"
            and op in _SMQE_ALLOWED_OPERATORS
            and backend in _SMQE_ALLOWED_BACKENDS
            and declared_op == op
            and declared_backend == backend
        )
        if not ok:
            dirty.append(
                f"{row.get('sample_id', '<sample>')}:"
                f"note={note or '<missing>'}:op={declared_op or '<missing>'}:"
                f"backend={declared_backend or '<missing>'}"
            )
            continue
        operators[op] += 1
        if backend == "claim":
            claim += 1
        elif backend == "record":
            record += 1
    total = len(relevant)
    structured_rate = (structured / total) if total else 0.0
    claim_rate = (claim / structured) if structured else 0.0
    failures: list[str] = []
    if not relevant:
        failures.append(f"system:{system}:no_rows")
    if dirty:
        failures.append(f"dirty:{len(dirty)}")
    if structured_rate < min_structured_rate:
        failures.append(
            f"structured_rate:{structured_rate:.3f}<required:{min_structured_rate:.3f}"
        )
    if claim_rate < min_claim_backend_rate:
        failures.append(
            f"claim_backend_rate:{claim_rate:.3f}<required:{min_claim_backend_rate:.3f}"
        )
    return {
        "pass": not failures,
        "system": system,
        "rows": total,
        "structured": structured,
        "claim": claim,
        "record": record,
        "fallback": fallback,
        "structured_rate": round(structured_rate, 4),
        "claim_backend_rate": round(claim_rate, 4),
        "dirty": dirty,
        "operators": dict(sorted(operators.items())),
        "failures": failures,
    }


def _region_telemetry_summary(rows: list[dict], *, system: str) -> dict:
    relevant = [
        row for row in rows
        if row.get("system") == system and not row.get("error")
    ]
    missing: list[str] = []
    malformed: list[str] = []
    hint_rows = 0
    total_hints = 0
    for row in relevant:
        extra = row.get("extra") or {}
        if not isinstance(extra, dict) or "region_hint_count" not in extra:
            missing.append(str(row.get("sample_id", "<sample>")))
            continue
        try:
            count = int(extra.get("region_hint_count", 0) or 0)
        except (TypeError, ValueError):
            malformed.append(f"{row.get('sample_id', '<sample>')}:region_hint_count")
            continue
        if count < 0:
            malformed.append(f"{row.get('sample_id', '<sample>')}:region_hint_count<0")
        if not isinstance(extra.get("region_ids", []), list):
            malformed.append(f"{row.get('sample_id', '<sample>')}:region_ids")
        if not isinstance(extra.get("region_member_ids", []), list):
            malformed.append(f"{row.get('sample_id', '<sample>')}:region_member_ids")
        if count > 0:
            hint_rows += 1
            total_hints += count
    failures: list[str] = []
    if not relevant:
        failures.append(f"system:{system}:no_rows")
    if missing:
        failures.append(f"missing:{len(missing)}")
    if malformed:
        failures.append(f"malformed:{len(malformed)}")
    return {
        "pass": not failures,
        "system": system,
        "rows": len(relevant),
        "hint_rows": hint_rows,
        "total_hints": total_hints,
        "missing": missing,
        "malformed": malformed,
        "failures": failures,
    }


def _positive_count(value) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _positive_number(value) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _nonempty_string_list(value) -> bool:
    return isinstance(value, list) and any(str(item).strip() for item in value)


def _sha256_strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip().lower()
        for item in value
        if re.fullmatch(r"[0-9a-fA-F]{64}", str(item).strip())
    ]


def _sha256_string_list(value) -> bool:
    return bool(_sha256_strings(value))


def _immutable_raw_uri_list(value) -> bool:
    return isinstance(value, list) and any(
        str(item).strip().startswith(("cas://", "oss://"))
        for item in value
    )


def _verified_proof_support_summary(rows: list[dict], *, system: str) -> dict:
    relevant = [
        row for row in rows
        if row.get("system") == system and not row.get("error")
    ]
    verified_rows = 0
    supported_verified_rows = 0
    unverified_answered: list[str] = []
    missing_support: list[str] = []
    proof_content_hashes: set[str] = set()
    for row in relevant:
        sample_id = str(row.get("sample_id", "<sample>"))
        if row.get("abstained"):
            continue
        extra = row.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        if not bool(extra.get("verified")):
            unverified_answered.append(sample_id)
            continue

        verified_rows += 1
        missing_fields: list[str] = []
        if not _positive_count(extra.get("citations")):
            missing_fields.append("citations")
        if not _nonempty_string_list(extra.get("entailed_memory_ids")):
            missing_fields.append("entailed_memory_ids")
        if not _sha256_string_list(extra.get("entailed_content_hashes")):
            missing_fields.append("entailed_content_hashes")
        else:
            proof_content_hashes.update(_sha256_strings(extra.get("entailed_content_hashes")))
        if not _immutable_raw_uri_list(extra.get("entailed_raw_uris")):
            missing_fields.append("entailed_raw_uris")
        if not _positive_number(extra.get("proof_surface_tokens")):
            missing_fields.append("proof_surface_tokens")
        if missing_fields:
            missing_support.append(f"{sample_id}:{','.join(missing_fields)}")
        else:
            supported_verified_rows += 1

    failures: list[str] = []
    if not relevant:
        failures.append(f"system:{system}:no_rows")
    if unverified_answered:
        failures.append(f"unverified_answered:{len(unverified_answered)}")
    if verified_rows <= 0:
        failures.append("verified_rows:0")
    if missing_support:
        failures.append(f"missing_support:{len(missing_support)}")
    support_rate = (supported_verified_rows / verified_rows) if verified_rows else 0.0
    return {
        "pass": not failures,
        "system": system,
        "rows": len(relevant),
        "verified_rows": verified_rows,
        "supported_verified_rows": supported_verified_rows,
        "support_rate": round(support_rate, 4),
        "proof_content_hashes": sorted(proof_content_hashes),
        "unverified_answered": unverified_answered[:20],
        "missing_support": missing_support[:20],
        "failures": failures,
    }


def _slice_reports(report: dict) -> list[dict]:
    reports = report.get("reports")
    if isinstance(reports, list):
        return [item for item in reports if isinstance(item, dict)]
    return [report] if isinstance(report, dict) else []


def _slice_invariant_summary(report: dict, *, required_datasets: list[str],
                             expected_system: str, min_draws: int,
                             min_subset: int) -> dict:
    datasets: dict[str, dict] = {}
    failures: list[str] = []
    for item in _slice_reports(report):
        dataset = str(item.get("dataset", "") or "").strip().lower()
        if not dataset:
            failures.append("dataset:<missing>")
            continue
        runs = item.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        seed_mode = str(item.get("seed_mode", "") or "").strip().lower()
        split = str(item.get("split", "") or "").strip().lower()
        holdout_profile = str(item.get("holdout_profile", "") or "").strip().lower()
        declared_system = str(item.get("system_under_test", "") or "").strip().lower()
        metadata_failures = []
        if split != "test":
            metadata_failures.append(f"split:{split or '<missing>'}")
        if holdout_profile != "holdout":
            metadata_failures.append(f"holdout_profile:{holdout_profile or '<missing>'}")
        draw_sets = []
        run_failures = []
        all_sample_ids: list[str] = []
        declared_draws = _as_int(item.get("draws", 0), 0)
        declared_subset = _as_int(item.get("subset", 0), 0)
        subset = max(min_subset, declared_subset)
        for idx, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                run_failures.append(f"draw{idx}:malformed")
                continue
            sample_ids = [str(s) for s in (run.get("sample_ids") or []) if str(s).strip()]
            draw_unique_sample_ids = len(set(sample_ids))
            all_sample_ids.extend(sample_ids)
            draw_sets.append(tuple(sorted(sample_ids)))
            if draw_unique_sample_ids < subset:
                run_failures.append(
                    f"draw{idx}:unique_sample_ids:{draw_unique_sample_ids}<required:{subset}"
                )
            if split in ("dev", "test"):
                split_bad = [sid for sid in sample_ids if split_of(sid) != split]
                if split_bad:
                    run_failures.append(
                        f"draw{idx}:sample_split:{split}:bad={len(split_bad)}:"
                        + ",".join(split_bad[:3])
                    )
            score = run.get("score") or {}
            if not isinstance(score, dict):
                score = {}
            verified_correct = _as_int(score.get("verified_correct", 0), 0)
            correct = _as_int(score.get("correct", verified_correct), 0)
            total = _as_int(score.get("total", 0), 0)
            verified = bool(score.get("verified")) and "verified_correct" in score
            ok = (
                bool(run.get("executed"))
                and _as_int(run.get("returncode", 1), 1) == 0
                and bool(score.get("pass"))
                and verified
                and verified_correct >= subset
                and verified_correct >= total
                and total >= subset
                and len(sample_ids) >= subset
                and draw_unique_sample_ids >= subset
            )
            if not ok:
                run_failures.append(
                    f"draw{idx}:executed={bool(run.get('executed'))},"
                    f"rc={run.get('returncode')},score={score.get('pass')},"
                    f"verified={verified},verified_correct={verified_correct},"
                    f"correct={correct},total={total},samples={len(sample_ids)}"
                )
        distinct_draw_sets = len(set(draw_sets))
        unique_sample_ids = len(set(all_sample_ids))
        requested_unique_sample_ids = max(0, declared_draws * declared_subset)
        expected_min_unique_sample_ids = max(min_draws * min_subset, requested_unique_sample_ids)
        min_unique_sample_ids = expected_min_unique_sample_ids
        coverage_failures = []
        declared_unique_sample_ids = _as_int(item.get("unique_sample_ids"), -1)
        declared_pool_unique_sample_ids = _as_int(item.get("pool_unique_sample_ids"), -1)
        declared_required_unique_sample_ids = _as_int(item.get("required_unique_sample_ids"), -1)
        if declared_required_unique_sample_ids >= 0:
            min_unique_sample_ids = max(min_unique_sample_ids, declared_required_unique_sample_ids)
        if declared_unique_sample_ids < 0:
            coverage_failures.append("unique_sample_ids:<missing>")
        elif declared_unique_sample_ids != unique_sample_ids:
            coverage_failures.append(
                f"unique_sample_ids:{declared_unique_sample_ids}!=observed:{unique_sample_ids}"
            )
        if declared_required_unique_sample_ids < 0:
            coverage_failures.append("required_unique_sample_ids:<missing>")
        elif declared_required_unique_sample_ids < expected_min_unique_sample_ids:
            coverage_failures.append(
                f"required_unique_sample_ids:{declared_required_unique_sample_ids}"
                f"<required:{expected_min_unique_sample_ids}"
            )
        if declared_pool_unique_sample_ids < 0:
            coverage_failures.append("pool_unique_sample_ids:<missing>")
        else:
            required_for_pool = max(min_unique_sample_ids, declared_required_unique_sample_ids)
            if declared_pool_unique_sample_ids < required_for_pool:
                coverage_failures.append(
                    f"pool_unique_sample_ids:{declared_pool_unique_sample_ids}"
                    f"<required:{required_for_pool}"
                )
            if declared_pool_unique_sample_ids < unique_sample_ids:
                coverage_failures.append(
                    f"pool_unique_sample_ids:{declared_pool_unique_sample_ids}"
                    f"<observed:{unique_sample_ids}"
                )
        if unique_sample_ids < min_unique_sample_ids:
            coverage_failures.append(
                f"unique_sample_ids:{unique_sample_ids}<required:{min_unique_sample_ids}"
            )
        draw_seeds = [str(seed) for seed in (item.get("draw_seeds") or []) if str(seed).strip()]
        distinct_draw_seeds = len(set(draw_seeds))
        dataset_ok = (
            bool(item.get("pass"))
            and declared_draws >= min_draws
            and declared_subset >= min_subset
            and len(runs) >= min_draws
            and seed_mode == "random"
            and split == "test"
            and holdout_profile == "holdout"
            and distinct_draw_sets >= min_draws
            and distinct_draw_seeds >= min_draws
            and unique_sample_ids >= min_unique_sample_ids
            and not coverage_failures
            and not run_failures
            and (not expected_system or declared_system == expected_system.lower())
        )
        datasets[dataset] = {
            "pass": dataset_ok,
            "draws": declared_draws,
            "subset": declared_subset,
            "runs": len(runs),
            "distinct_draws": distinct_draw_sets,
            "unique_sample_ids": unique_sample_ids,
            "declared_unique_sample_ids": declared_unique_sample_ids,
            "pool_unique_sample_ids": declared_pool_unique_sample_ids,
            "required_unique_sample_ids": declared_required_unique_sample_ids,
            "min_unique_sample_ids": min_unique_sample_ids,
            "seed_mode": seed_mode,
            "split": split,
            "holdout_profile": holdout_profile,
            "distinct_draw_seeds": distinct_draw_seeds,
            "system_under_test": declared_system,
            "failures": [*metadata_failures, *coverage_failures, *run_failures],
        }
        if not dataset_ok:
            item_failures = [*metadata_failures, *coverage_failures, *run_failures]
            failures.append(f"{dataset}:{';'.join(item_failures[:3]) or 'metadata'}")
    required = {str(d).strip().lower() for d in required_datasets if str(d).strip()}
    covered = {dataset for dataset, item in datasets.items() if item.get("pass")}
    missing = sorted(required - covered)
    if missing:
        failures.append("missing:" + ",".join(missing))
    return {
        "pass": not failures,
        "datasets": datasets,
        "covered": sorted(covered),
        "missing": missing,
        "failures": failures,
    }


def _seed_mode_failures(report: dict) -> tuple[str, list[str]]:
    seed_mode = str(report.get("seed_mode", "") or "").strip().lower()
    failures = [] if seed_mode == "random" else [f"seed_mode:{seed_mode or '<missing>'}"]
    return seed_mode, failures


def _affect_salience_summary(report: dict, *, min_cases: int,
                             max_lambda_salience: float,
                             max_boost_ratio: float,
                             min_age_gap_seconds: float) -> dict:
    case_type_counts = report.get("case_type_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    flip_checks = _as_int(report.get("flip_checks", 0), 0)
    age_free_checks = _as_int(report.get("age_free_checks", 0), 0)
    bounded_checks = _as_int(report.get("bounded_checks", 0), 0)
    lambda_salience = _as_float(report.get("lambda_salience", float("inf")))
    observed_boost_ratio = _as_float(report.get("max_boost_ratio", float("inf")))
    observed_min_age_gap = _as_float(report.get("min_age_gap_seconds", 0.0))
    expected_checks = cases * 7
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != expected_checks or cases <= 0:
        failures.append(f"checks:{checks}/expected:{expected_checks}")
    if correct != checks or cases <= 0:
        failures.append(f"correct:{correct}/{checks}")
    if flip_checks < cases * 2 or cases <= 0:
        failures.append(f"flip_checks:{flip_checks}<expected:{cases * 2}")
    if age_free_checks < cases or cases <= 0:
        failures.append(f"age_free_checks:{age_free_checks}<expected:{cases}")
    if bounded_checks < cases or cases <= 0:
        failures.append(f"bounded_checks:{bounded_checks}<expected:{cases}")
    if _as_int(case_type_counts.get("affect_salience_retrieval", 0), 0) < cases:
        failures.append("case_type:affect_salience_retrieval")
    if lambda_salience <= 0.0 or lambda_salience > max_lambda_salience:
        failures.append(f"lambda_salience:{lambda_salience}>{max_lambda_salience}")
    if observed_boost_ratio > max_boost_ratio:
        failures.append(f"max_boost_ratio:{observed_boost_ratio}>{max_boost_ratio}")
    if observed_min_age_gap < min_age_gap_seconds:
        failures.append(f"min_age_gap_seconds:{observed_min_age_gap}<required:{min_age_gap_seconds}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "flip_checks": flip_checks,
        "age_free_checks": age_free_checks,
        "bounded_checks": bounded_checks,
        "lambda_salience": lambda_salience,
        "max_boost_ratio": observed_boost_ratio,
        "min_age_gap_seconds": observed_min_age_gap,
        "case_type_counts": case_type_counts,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _scratchpad_summary(report: dict, *, min_cases: int) -> dict:
    case_type_counts = report.get("case_type_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    ordering_checks = _as_int(report.get("ordering_checks", 0), 0)
    active_scope_filter_checks = _as_int(report.get("active_scope_filter_checks", 0), 0)
    proof_link_checks = _as_int(report.get("proof_link_checks", 0), 0)
    top_k_checks = _as_int(report.get("top_k_checks", 0), 0)
    retrieval_channel_checks = _as_int(report.get("retrieval_channel_checks", 0), 0)
    expected_checks = cases * 11
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != expected_checks or cases <= 0:
        failures.append(f"checks:{checks}/expected:{expected_checks}")
    if correct != checks or cases <= 0:
        failures.append(f"correct:{correct}/{checks}")
    if ordering_checks < cases or cases <= 0:
        failures.append(f"ordering_checks:{ordering_checks}<expected:{cases}")
    if active_scope_filter_checks < cases or cases <= 0:
        failures.append(f"active_scope_filter_checks:{active_scope_filter_checks}<expected:{cases}")
    if proof_link_checks < cases * 4 or cases <= 0:
        failures.append(f"proof_link_checks:{proof_link_checks}<expected:{cases * 4}")
    if top_k_checks < cases or cases <= 0:
        failures.append(f"top_k_checks:{top_k_checks}<expected:{cases}")
    if retrieval_channel_checks < cases * 4 or cases <= 0:
        failures.append(f"retrieval_channel_checks:{retrieval_channel_checks}<expected:{cases * 4}")
    if _as_int(case_type_counts.get("scratchpad_active_proof_surface", 0), 0) < cases:
        failures.append("case_type:scratchpad_active_proof_surface")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "ordering_checks": ordering_checks,
        "active_scope_filter_checks": active_scope_filter_checks,
        "proof_link_checks": proof_link_checks,
        "top_k_checks": top_k_checks,
        "retrieval_channel_checks": retrieval_channel_checks,
        "case_type_counts": case_type_counts,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _region_routing_summary(report: dict, *, min_cases: int) -> dict:
    case_type_counts = report.get("case_type_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    dense_miss_recovery_checks = _as_int(report.get("dense_miss_recovery_checks", 0), 0)
    active_scope_filter_checks = _as_int(report.get("active_scope_filter_checks", 0), 0)
    nested_cocoon_checks = _as_int(report.get("nested_cocoon_checks", 0), 0)
    proof_link_checks = _as_int(report.get("proof_link_checks", 0), 0)
    telemetry_trace_checks = _as_int(report.get("telemetry_trace_checks", 0), 0)
    route_only_context_checks = _as_int(report.get("route_only_context_checks", 0), 0)
    expected_checks = cases * 12
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != expected_checks or cases <= 0:
        failures.append(f"checks:{checks}/expected:{expected_checks}")
    if correct != checks or cases <= 0:
        failures.append(f"correct:{correct}/{checks}")
    if dense_miss_recovery_checks < cases * 3 or cases <= 0:
        failures.append(f"dense_miss_recovery_checks:{dense_miss_recovery_checks}<expected:{cases * 3}")
    if active_scope_filter_checks < cases * 2 or cases <= 0:
        failures.append(f"active_scope_filter_checks:{active_scope_filter_checks}<expected:{cases * 2}")
    if nested_cocoon_checks < cases * 2 or cases <= 0:
        failures.append(f"nested_cocoon_checks:{nested_cocoon_checks}<expected:{cases * 2}")
    if proof_link_checks < cases * 2 or cases <= 0:
        failures.append(f"proof_link_checks:{proof_link_checks}<expected:{cases * 2}")
    if telemetry_trace_checks < cases * 2 or cases <= 0:
        failures.append(f"telemetry_trace_checks:{telemetry_trace_checks}<expected:{cases * 2}")
    if route_only_context_checks < cases or cases <= 0:
        failures.append(f"route_only_context_checks:{route_only_context_checks}<expected:{cases}")
    if _as_int(case_type_counts.get("region_routing_cocoon_proof", 0), 0) < cases:
        failures.append("case_type:region_routing_cocoon_proof")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "dense_miss_recovery_checks": dense_miss_recovery_checks,
        "active_scope_filter_checks": active_scope_filter_checks,
        "nested_cocoon_checks": nested_cocoon_checks,
        "proof_link_checks": proof_link_checks,
        "telemetry_trace_checks": telemetry_trace_checks,
        "route_only_context_checks": route_only_context_checks,
        "case_type_counts": case_type_counts,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _reflex_recall_summary(report: dict, *, min_cases: int,
                           max_p95_latency_ms: float) -> dict:
    case_type_counts = report.get("case_type_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    direct_hit_checks = _as_int(report.get("direct_hit_checks", 0), 0)
    coactivation_checks = _as_int(report.get("coactivation_checks", 0), 0)
    active_scope_filter_checks = _as_int(report.get("active_scope_filter_checks", 0), 0)
    proof_link_checks = _as_int(report.get("proof_link_checks", 0), 0)
    score_contract_checks = _as_int(report.get("score_contract_checks", 0), 0)
    latency_budget_checks = _as_int(report.get("latency_budget_checks", 0), 0)
    p95_latency_ms = _as_float(report.get("p95_latency_ms", float("inf")))
    max_latency_ms = _as_float(report.get("max_latency_ms", float("inf")))
    expected_checks = cases * 12
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != expected_checks or cases <= 0:
        failures.append(f"checks:{checks}/expected:{expected_checks}")
    if correct != checks or cases <= 0:
        failures.append(f"correct:{correct}/{checks}")
    if direct_hit_checks < cases * 2 or cases <= 0:
        failures.append(f"direct_hit_checks:{direct_hit_checks}<expected:{cases * 2}")
    if coactivation_checks < cases * 2 or cases <= 0:
        failures.append(f"coactivation_checks:{coactivation_checks}<expected:{cases * 2}")
    if active_scope_filter_checks < cases * 4 or cases <= 0:
        failures.append(f"active_scope_filter_checks:{active_scope_filter_checks}<expected:{cases * 4}")
    if proof_link_checks < cases or cases <= 0:
        failures.append(f"proof_link_checks:{proof_link_checks}<expected:{cases}")
    if score_contract_checks < cases * 2 or cases <= 0:
        failures.append(f"score_contract_checks:{score_contract_checks}<expected:{cases * 2}")
    if latency_budget_checks < cases or cases <= 0:
        failures.append(f"latency_budget_checks:{latency_budget_checks}<expected:{cases}")
    if p95_latency_ms > max_p95_latency_ms:
        failures.append(f"p95_latency_ms:{p95_latency_ms}>{max_p95_latency_ms}")
    if _as_int(case_type_counts.get("reflex_recall_proof_surface", 0), 0) < cases:
        failures.append("case_type:reflex_recall_proof_surface")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "direct_hit_checks": direct_hit_checks,
        "coactivation_checks": coactivation_checks,
        "active_scope_filter_checks": active_scope_filter_checks,
        "proof_link_checks": proof_link_checks,
        "score_contract_checks": score_contract_checks,
        "latency_budget_checks": latency_budget_checks,
        "p95_latency_ms": p95_latency_ms,
        "max_latency_ms": max_latency_ms,
        "case_type_counts": case_type_counts,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_synthetic_summary(report: dict, *, min_cases: int,
                            max_avg_proof_tokens: float,
                            require_record_backend: bool = True) -> dict:
    min_op_count = 2
    op_counts = report.get("operator_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(op_counts, dict):
        op_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if correct != cases or cases <= 0:
        failures.append(f"correct:{correct}/{cases}")
    missing_ops = sorted(
        _SMQE_REQUIRED_SYNTHETIC_OPS - {str(k) for k, v in op_counts.items() if _as_int(v, 0) >= min_op_count}
    )
    if missing_ops:
        failures.append(f"ops_below_{min_op_count}:" + ",".join(missing_ops))
    if _as_int(backend_counts.get("claim", 0), 0) <= 0:
        failures.append("backend:claim")
    if require_record_backend and _as_int(backend_counts.get("record", 0), 0) <= 0:
        failures.append("backend:record")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "correct": correct,
        "operator_counts": op_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_planner_summary(report: dict, *, min_cases: int,
                          max_p95_latency_ms: float) -> dict:
    min_op_count = 2
    op_counts = report.get("operator_counts") or {}
    if not isinstance(op_counts, dict):
        op_counts = {}
    case_type_counts = report.get("case_type_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    generic_terms = _as_int(report.get("generic_term_checks", 0), 0)
    p95_latency_ms = _as_float(report.get("p95_latency_ms", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks <= 0 or correct != checks:
        failures.append(f"correct:{correct}/{checks}")
    if generic_terms < cases:
        failures.append(f"generic_term_checks:{generic_terms}<cases:{cases}")
    if _as_int(case_type_counts.get("smqe_planner_generic_shape", 0), 0) < cases:
        failures.append("case_type:smqe_planner_generic_shape")
    missing_ops = sorted(
        _SMQE_REQUIRED_PLANNER_OPS - {str(k) for k, v in op_counts.items() if _as_int(v, 0) >= min_op_count}
    )
    if missing_ops:
        failures.append(f"ops_below_{min_op_count}:" + ",".join(missing_ops))
    if p95_latency_ms > max_p95_latency_ms:
        failures.append(f"p95_latency_ms:{p95_latency_ms}>{max_p95_latency_ms}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "correct": correct,
        "total_checks": checks,
        "generic_term_checks": generic_terms,
        "operator_counts": op_counts,
        "case_type_counts": case_type_counts,
        "p95_latency_ms": p95_latency_ms,
        "max_latency_ms": _as_float(report.get("max_latency_ms", 0.0)),
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_claim_coverage_summary(report: dict, *, min_cases: int,
                                 min_claim_backend_rate: float,
                                 max_avg_proof_tokens: float) -> dict:
    base = _smqe_synthetic_summary(
        report,
        min_cases=min_cases,
        max_avg_proof_tokens=max_avg_proof_tokens,
        require_record_backend=False,
    )
    cases = int(base["cases"] or 0)
    claim_backend = _as_int((base["backend_counts"] or {}).get("claim", 0), 0)
    rate = (claim_backend / cases) if cases else 0.0
    failures = list(base["failures"])
    if rate < min_claim_backend_rate:
        failures.append(f"claim_backend_rate:{rate:.3f}<required:{min_claim_backend_rate:.3f}")
    claims_extracted = _as_int(report.get("claims_extracted", 0), 0)
    if claims_extracted < cases:
        failures.append(f"claims_extracted:{claims_extracted}<cases:{cases}")
    if _as_int((base["backend_counts"] or {}).get("record", 0), 0) > 0:
        failures.append("record_backend_used")
    claim_backend_operator_counts = report.get("claim_backend_operator_counts") or {}
    if not isinstance(claim_backend_operator_counts, dict):
        claim_backend_operator_counts = {}
    operator_counts = base["operator_counts"] or {}
    for op, expected in sorted(operator_counts.items()):
        expected_n = _as_int(expected, 0)
        claim_n = _as_int(claim_backend_operator_counts.get(op, 0), 0)
        if expected_n > 0 and claim_n < expected_n:
            failures.append(f"operator_claim_backend:{op}:{claim_n}/{expected_n}")
    return {
        **base,
        "pass": not failures,
        "claim_backend_rate": round(rate, 4),
        "claim_backend_correct": _as_int(report.get("claim_backend_correct", claim_backend), 0),
        "claims_extracted": claims_extracted,
        "avg_claims_per_case": _as_float(report.get("avg_claims_per_case", 0.0)),
        "claim_backend_operator_counts": claim_backend_operator_counts,
        "claim_type_counts": report.get("claim_type_counts") or {},
        "failures": failures,
    }


def _smqe_fullpath_summary(report: dict, *, min_cases: int,
                           max_avg_proof_tokens: float,
                           max_avg_context_tokens: float,
                           max_p95_latency_ms: float) -> dict:
    base = _smqe_synthetic_summary(
        report,
        min_cases=min_cases,
        max_avg_proof_tokens=max_avg_proof_tokens,
        require_record_backend=False,
    )
    cases = int(base["cases"] or 0)
    failures = list(base["failures"])
    verified = _as_int(report.get("verified", 0), 0)
    structured = _as_int(report.get("structured_recall", 0), 0)
    reader_calls = _as_int(report.get("reader_calls", 0), 0)
    proof_link_checks = _as_int(report.get("proof_link_checks", 0), 0)
    claim_backend = _as_int(report.get("claim_backend_correct", 0), 0)
    claims_extracted = _as_int(report.get("claims_extracted", 0), 0)
    avg_context = _as_float(report.get("avg_context_tokens", float("inf")))
    latency_budget_checks = _as_int(report.get("latency_budget_checks", 0), 0)
    p95_latency_ms = _as_float(report.get("p95_latency_ms", float("inf")))
    max_latency_ms = _as_float(report.get("max_latency_ms", float("inf")))
    case_operator_counts = report.get("case_operator_counts") or {}
    if not isinstance(case_operator_counts, dict):
        case_operator_counts = {}
    if verified != cases or cases <= 0:
        failures.append(f"verified:{verified}/{cases}")
    if structured != cases or cases <= 0:
        failures.append(f"structured_recall:{structured}/{cases}")
    if reader_calls != 0:
        failures.append(f"reader_calls:{reader_calls}")
    if proof_link_checks < cases or cases <= 0:
        failures.append(f"proof_link_checks:{proof_link_checks}<expected:{cases}")
    if claim_backend != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_backend}/{cases}")
    if claims_extracted < cases:
        failures.append(f"claims_extracted:{claims_extracted}<cases:{cases}")
    if avg_context > max_avg_context_tokens:
        failures.append(f"avg_context_tokens:{avg_context}>{max_avg_context_tokens}")
    if latency_budget_checks < cases or cases <= 0:
        failures.append(f"latency_budget_checks:{latency_budget_checks}<expected:{cases}")
    if p95_latency_ms > max_p95_latency_ms:
        failures.append(f"p95_latency_ms:{p95_latency_ms}>{max_p95_latency_ms}")
    expected_total = sum(_as_int(v, 0) for v in case_operator_counts.values())
    if not case_operator_counts:
        failures.append("case_operator_counts:missing")
    elif expected_total != cases:
        failures.append(f"case_operator_counts_total:{expected_total}/{cases}")
    min_case_op_count = 2
    missing_case_ops = sorted(
        _SMQE_REQUIRED_SYNTHETIC_OPS
        - {str(k) for k, v in case_operator_counts.items() if _as_int(v, 0) >= min_case_op_count}
    )
    if missing_case_ops:
        failures.append(f"case_ops_below_{min_case_op_count}:" + ",".join(missing_case_ops))
    return {
        **base,
        "pass": not failures,
        "verified": verified,
        "structured_recall": structured,
        "reader_calls": reader_calls,
        "proof_link_checks": proof_link_checks,
        "claim_backend_correct": claim_backend,
        "claims_extracted": claims_extracted,
        "avg_claims_per_case": _as_float(report.get("avg_claims_per_case", 0.0)),
        "avg_context_tokens": avg_context,
        "latency_budget_checks": latency_budget_checks,
        "p95_latency_ms": p95_latency_ms,
        "max_latency_ms": max_latency_ms,
        "case_operator_counts": case_operator_counts,
        "failures": failures,
    }


def _smqe_paraphrase_summary(report: dict, *, min_cases: int,
                             max_avg_proof_tokens: float) -> dict:
    base = _smqe_synthetic_summary(
        report,
        min_cases=min_cases,
        max_avg_proof_tokens=max_avg_proof_tokens,
        require_record_backend=True,
    )
    cases = int(base["cases"] or 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    failures = list(base["failures"])
    if record_correct != cases or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{cases}")
    if claim_correct != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{cases}")
    backend_counts = base["backend_counts"] or {}
    if _as_int(backend_counts.get("record", 0), 0) < cases:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{cases}")
    if _as_int(backend_counts.get("claim", 0), 0) < cases:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{cases}")
    return {
        **base,
        "pass": not failures,
        "total_checks": _as_int(report.get("checks", cases * 2), cases * 2),
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "failures": failures,
    }


def _smqe_conflict_summary(report: dict, *, min_cases: int,
                           max_avg_proof_tokens: float) -> dict:
    required_types = {"amount", "location", "status"}
    min_type_count = 2
    value_type_counts = report.get("value_type_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(value_type_counts, dict):
        value_type_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    correct = _as_int(report.get("correct", 0), 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if correct != cases or cases <= 0:
        failures.append(f"correct:{correct}/{cases}")
    if record_correct != cases or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{cases}")
    if claim_correct != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{cases}")
    missing_types = sorted(
        required_types - {str(k) for k, v in value_type_counts.items() if _as_int(v, 0) >= min_type_count}
    )
    if missing_types:
        failures.append(f"types_below_{min_type_count}:" + ",".join(missing_types))
    if _as_int(backend_counts.get("record", 0), 0) < cases:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{cases}")
    if _as_int(backend_counts.get("claim", 0), 0) < cases:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{cases}")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "correct": correct,
        "total_checks": _as_int(report.get("checks", cases * 2), cases * 2),
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "value_type_counts": value_type_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_composition_summary(report: dict, *, min_cases: int,
                              max_avg_proof_tokens: float) -> dict:
    required_types = {"event_order", "relative_event_time", "shared_value"}
    min_type_count = 2
    case_type_counts = report.get("case_type_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", cases * 2), cases * 2)
    correct = _as_int(report.get("correct", 0), 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != cases * 2 or cases <= 0:
        failures.append(f"checks:{checks}/expected:{cases * 2}")
    if correct != cases or cases <= 0:
        failures.append(f"correct:{correct}/{cases}")
    if record_correct != cases or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{cases}")
    if claim_correct != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{cases}")
    missing_types = sorted(
        required_types - {str(k) for k, v in case_type_counts.items() if _as_int(v, 0) >= min_type_count}
    )
    if missing_types:
        failures.append(f"types_below_{min_type_count}:" + ",".join(missing_types))
    if _as_int(backend_counts.get("record", 0), 0) < cases:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{cases}")
    if _as_int(backend_counts.get("claim", 0), 0) < cases:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{cases}")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "case_type_counts": case_type_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_relative_phrase_summary(report: dict, *, min_cases: int,
                                  max_avg_proof_tokens: float) -> dict:
    required_types = {"ago_days", "ago_weeks", "fortnight_ago", "in_days", "next_month", "next_week"}
    min_type_count = 2
    case_type_counts = report.get("case_type_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", cases * 2), cases * 2)
    correct = _as_int(report.get("correct", 0), 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != cases * 2 or cases <= 0:
        failures.append(f"checks:{checks}/expected:{cases * 2}")
    if correct != cases or cases <= 0:
        failures.append(f"correct:{correct}/{cases}")
    if record_correct != cases or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{cases}")
    if claim_correct != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{cases}")
    missing_types = sorted(
        required_types - {str(k) for k, v in case_type_counts.items() if _as_int(v, 0) >= min_type_count}
    )
    if missing_types:
        failures.append(f"types_below_{min_type_count}:" + ",".join(missing_types))
    if _as_int(backend_counts.get("record", 0), 0) < cases:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{cases}")
    if _as_int(backend_counts.get("claim", 0), 0) < cases:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{cases}")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "case_type_counts": case_type_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_temporal_window_summary(report: dict, *, min_cases: int,
                                  max_avg_proof_tokens: float) -> dict:
    required_types = {
        "fortnight_count", "most_recent_latest", "past_days_count",
        "past_few_months_count", "past_week_count", "past_week_list",
        "recent_count", "recent_hours_sum", "recent_list", "source_action_variant_window",
        "source_location_window",
    }
    min_type_count = 2
    case_type_counts = report.get("case_type_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", cases * 2), cases * 2)
    correct = _as_int(report.get("correct", 0), 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != cases * 2 or cases <= 0:
        failures.append(f"checks:{checks}/expected:{cases * 2}")
    if correct != cases or cases <= 0:
        failures.append(f"correct:{correct}/{cases}")
    if record_correct != cases or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{cases}")
    if claim_correct != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{cases}")
    missing_types = sorted(
        required_types - {str(k) for k, v in case_type_counts.items() if _as_int(v, 0) >= min_type_count}
    )
    if missing_types:
        failures.append(f"types_below_{min_type_count}:" + ",".join(missing_types))
    if _as_int(backend_counts.get("record", 0), 0) < cases:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{cases}")
    if _as_int(backend_counts.get("claim", 0), 0) < cases:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{cases}")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "case_type_counts": case_type_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_attribution_summary(report: dict, *, min_cases: int,
                              max_avg_proof_tokens: float) -> dict:
    required_types = {"gave_actor", "recommend_actor", "shared_actor", "told_actor"}
    min_type_count = 2
    case_type_counts = report.get("case_type_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", cases * 2), cases * 2)
    correct = _as_int(report.get("correct", 0), 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != cases * 2 or cases <= 0:
        failures.append(f"checks:{checks}/expected:{cases * 2}")
    if correct != cases or cases <= 0:
        failures.append(f"correct:{correct}/{cases}")
    if record_correct != cases or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{cases}")
    if claim_correct != cases or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{cases}")
    missing_types = sorted(
        required_types - {str(k) for k, v in case_type_counts.items() if _as_int(v, 0) >= min_type_count}
    )
    if missing_types:
        failures.append(f"types_below_{min_type_count}:" + ",".join(missing_types))
    if _as_int(backend_counts.get("record", 0), 0) < cases:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{cases}")
    if _as_int(backend_counts.get("claim", 0), 0) < cases:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{cases}")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "case_type_counts": case_type_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_abstention_summary(report: dict, *, min_cases: int) -> dict:
    required_types = {
        "count_neutral_quantity",
        "count_target_mismatch",
        "latest_future_only",
        "latest_missing_subject",
        "preference_no_positive",
        "speaker_crossed_support",
        "table_missing_row",
        "temporal_missing_anchor",
    }
    min_type_count = 2
    case_type_counts = report.get("case_type_counts") or {}
    if not isinstance(case_type_counts, dict):
        case_type_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", cases * 2), cases * 2)
    abstained = _as_int(report.get("abstained", 0), 0)
    record_only_abstained = _as_int(report.get("record_only_abstained", 0), 0)
    claims_present_abstained = _as_int(report.get("claims_present_abstained", 0), 0)
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != cases * 2 or cases <= 0:
        failures.append(f"checks:{checks}/expected:{cases * 2}")
    if abstained != cases or cases <= 0:
        failures.append(f"abstained:{abstained}/{cases}")
    if record_only_abstained != cases or cases <= 0:
        failures.append(f"record_only_abstained:{record_only_abstained}/{cases}")
    if claims_present_abstained != cases or cases <= 0:
        failures.append(f"claims_present_abstained:{claims_present_abstained}/{cases}")
    missing_types = sorted(
        required_types - {str(k) for k, v in case_type_counts.items() if _as_int(v, 0) >= min_type_count}
    )
    if missing_types:
        failures.append(f"types_below_{min_type_count}:" + ",".join(missing_types))
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "abstained": abstained,
        "record_only_abstained": record_only_abstained,
        "claims_present_abstained": claims_present_abstained,
        "case_type_counts": case_type_counts,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_scope_summary(report: dict, *, min_cases: int,
                        max_avg_proof_tokens: float) -> dict:
    required_ops = {
        "count_aggregate",
        "latest_value",
        "multi_session_sum",
        "preference_synth",
        "relative_temporal",
        "speaker_fact",
        "table_lookup",
        "temporal_delta",
    }
    min_op_count = 2
    operator_counts = report.get("operator_counts") or {}
    backend_counts = report.get("backend_counts") or {}
    if not isinstance(operator_counts, dict):
        operator_counts = {}
    if not isinstance(backend_counts, dict):
        backend_counts = {}
    seed_mode, failures = _seed_mode_failures(report)
    cases = _as_int(report.get("cases", 0), 0)
    checks = _as_int(report.get("checks", cases * 4), cases * 4)
    correct = _as_int(report.get("correct", 0), 0)
    record_correct = _as_int(report.get("record_backend_correct", 0), 0)
    claim_correct = _as_int(report.get("claim_backend_correct", 0), 0)
    expected_backend_checks = cases * 2
    avg_proof = _as_float(report.get("avg_proof_tokens", float("inf")))
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if cases < min_cases:
        failures.append(f"cases:{cases}<required:{min_cases}")
    if checks != cases * 4 or cases <= 0:
        failures.append(f"checks:{checks}/expected:{cases * 4}")
    if correct != checks or cases <= 0:
        failures.append(f"correct:{correct}/{checks}")
    if record_correct != expected_backend_checks or cases <= 0:
        failures.append(f"record_backend_correct:{record_correct}/{expected_backend_checks}")
    if claim_correct != expected_backend_checks or cases <= 0:
        failures.append(f"claim_backend_correct:{claim_correct}/{expected_backend_checks}")
    missing_ops = sorted(
        required_ops - {str(k) for k, v in operator_counts.items() if _as_int(v, 0) >= min_op_count}
    )
    if missing_ops:
        failures.append(f"ops_below_{min_op_count}:" + ",".join(missing_ops))
    if _as_int(backend_counts.get("record", 0), 0) < expected_backend_checks:
        failures.append(f"record_backend_count:{_as_int(backend_counts.get('record', 0), 0)}<{expected_backend_checks}")
    if _as_int(backend_counts.get("claim", 0), 0) < expected_backend_checks:
        failures.append(f"claim_backend_count:{_as_int(backend_counts.get('claim', 0), 0)}<{expected_backend_checks}")
    if avg_proof > max_avg_proof_tokens:
        failures.append(f"avg_proof_tokens:{avg_proof}>{max_avg_proof_tokens}")
    if report.get("failures"):
        failures.append(f"failures:{len(report.get('failures') or [])}")
    return {
        "pass": not failures,
        "cases": cases,
        "total_checks": checks,
        "correct": correct,
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "operator_counts": operator_counts,
        "backend_counts": backend_counts,
        "avg_proof_tokens": avg_proof,
        "seed_mode": seed_mode,
        "failures": failures,
    }


def _smqe_time_summary(report: dict, *, min_cases: int,
                       max_avg_proof_tokens: float) -> dict:
    base = _smqe_scope_summary(
        report,
        min_cases=min_cases,
        max_avg_proof_tokens=max_avg_proof_tokens,
    )
    return base


def _smqe_subscope_summary(report: dict, *, min_cases: int,
                           max_avg_proof_tokens: float) -> dict:
    base = _smqe_scope_summary(
        report,
        min_cases=min_cases,
        max_avg_proof_tokens=max_avg_proof_tokens,
    )
    return base


def _smqe_invalidation_summary(report: dict, *, min_cases: int,
                               max_avg_proof_tokens: float) -> dict:
    base = _smqe_scope_summary(
        report,
        min_cases=min_cases,
        max_avg_proof_tokens=max_avg_proof_tokens,
    )
    failures = list(base["failures"])
    preference_cases = _as_int(report.get("preference_supersession_cases", 0), 0)
    preference_checks = _as_int(report.get("preference_supersession_checks", 0), 0)
    preference_correct = _as_int(report.get("preference_supersession_correct", 0), 0)
    preference_record_correct = _as_int(
        report.get("preference_supersession_record_correct", 0), 0
    )
    preference_claim_correct = _as_int(
        report.get("preference_supersession_claim_correct", 0), 0
    )
    expected_preference_checks = preference_cases * 4
    expected_preference_backend_checks = preference_cases * 2
    min_preference_cases = 2
    if preference_cases < min_preference_cases:
        failures.append(
            f"preference_supersession_cases:{preference_cases}<required:{min_preference_cases}"
        )
    if preference_checks != expected_preference_checks or preference_cases <= 0:
        failures.append(
            f"preference_supersession_checks:{preference_checks}/expected:{expected_preference_checks}"
        )
    if preference_correct != preference_checks or preference_cases <= 0:
        failures.append(f"preference_supersession_correct:{preference_correct}/{preference_checks}")
    if preference_record_correct != expected_preference_backend_checks or preference_cases <= 0:
        failures.append(
            "preference_supersession_record_correct:"
            f"{preference_record_correct}/{expected_preference_backend_checks}"
        )
    if preference_claim_correct != expected_preference_backend_checks or preference_cases <= 0:
        failures.append(
            "preference_supersession_claim_correct:"
            f"{preference_claim_correct}/{expected_preference_backend_checks}"
        )
    if not bool(report.get("preference_supersession_pass")):
        failures.append("preference_supersession_pass:false")
    return {
        **base,
        "pass": not failures,
        "preference_supersession_pass": bool(report.get("preference_supersession_pass")),
        "preference_supersession_cases": preference_cases,
        "preference_supersession_checks": preference_checks,
        "preference_supersession_correct": preference_correct,
        "preference_supersession_record_correct": preference_record_correct,
        "preference_supersession_claim_correct": preference_claim_correct,
        "failures": failures,
    }


def _rows_by_system(rows: list[dict]) -> dict[str, list[dict]]:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sys[str(row.get("system", ""))].append(row)
    return by_sys


def _unique_sample_count(rows: list[dict]) -> int:
    return len({
        (
            str(r.get("dataset", "")),
            str(r.get("category", "")),
            str(r.get("sample_id", "")),
        )
        for r in rows
        if str(r.get("sample_id", "")).strip()
    })


def _sample_unit_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("category", "")),
        str(row.get("sample_id", "")),
    )


def _sample_run_key(row: dict) -> tuple[str, str, str, int]:
    return (*_sample_unit_key(row), _as_int(row.get("run_idx", 0), 0))


def _canonical_system_name(name: str) -> str:
    value = (name or "").strip().lower()
    aliases = {
        "eidetic": "eidetic-plus",
        "eidetic-plus": "eidetic-plus",
        "eidetic-full": "eidetic-plus-full",
        "eidetic-plus-full": "eidetic-plus-full",
        "eidetic-product": "eidetic-product",
        "eidetic-plus-product": "eidetic-product",
        "rag": "rag-vector",
        "ragvector": "rag-vector",
        "rag-vector": "rag-vector",
        "ragfull": "rag-full",
        "rag-full": "rag-full",
    }
    return aliases.get(value, value)


def _manifest_systems(manifest: dict) -> set[str]:
    raw = manifest.get("systems", "")
    if isinstance(raw, str):
        values = _csv(raw)
    elif isinstance(raw, list):
        values = [str(item) for item in raw]
    else:
        values = []
    return {_canonical_system_name(item) for item in values if str(item).strip()}


def _manifest_sample_rows(manifest: dict) -> list[dict]:
    rows = manifest.get("sample_rows", [])
    if not isinstance(rows, list):
        return []
    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        dataset = str(item.get("dataset", "")).strip()
        category = str(item.get("category", "")).strip()
        sample_id = str(item.get("sample_id", "")).strip()
        if dataset and category and sample_id:
            out.append({"dataset": dataset, "category": category, "sample_id": sample_id})
    return out


def _manifest_log_contract(rows: list[dict], manifest: dict,
                           required_systems: list[str]) -> dict:
    sample_rows = _manifest_sample_rows(manifest)
    run_offset = _as_int(manifest.get("run_offset", 0), 0)
    runs = max(0, _as_int(manifest.get("runs", 0), 0))
    run_indices = list(range(run_offset, run_offset + runs))
    required = {_canonical_system_name(system) for system in required_systems}
    expected = {
        (system, sample["dataset"], sample["category"], sample["sample_id"], run_idx)
        for system in required
        for sample in sample_rows
        for run_idx in run_indices
    }
    actual = {
        (
            _canonical_system_name(str(row.get("system", ""))),
            str(row.get("dataset", "")),
            str(row.get("category", "")),
            str(row.get("sample_id", "")),
            _as_int(row.get("run_idx", 0), 0),
        )
        for row in rows
        if _canonical_system_name(str(row.get("system", ""))) in required
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "sample_rows": sample_rows,
        "run_indices": run_indices,
        "expected": len(expected),
        "actual": len(actual),
        "missing": missing,
        "extra": extra,
    }


def _composite_source_checks(manifest: dict) -> list[dict]:
    checks: list[dict] = []
    if str(manifest.get("artifact_kind", "")).strip().lower() != "composite":
        return checks
    sources = manifest.get("composite_sources", [])
    if not isinstance(sources, list):
        sources = []
    _append(checks, "manifest:composite_sources_present", bool(sources),
            f"{len(sources)} source artifact(s)")
    missing_paths = []
    render_only = []
    fingerprint_mismatches = []
    invalid_fingerprints = []
    for source in sources:
        if not isinstance(source, dict):
            invalid_fingerprints.append("<malformed-source>")
            continue
        raw_path = str(source.get("path", "")).strip()
        if not raw_path:
            missing_paths.append("<missing>")
            continue
        path = Path(raw_path)
        if not path.exists():
            missing_paths.append(raw_path)
            continue
        try:
            src_manifest = _load_manifest(path)
        except Exception as exc:  # noqa: BLE001 - release gate reports, not raises
            invalid_fingerprints.append(f"{raw_path}: manifest error {exc}")
            continue
        if bool(src_manifest.get("render_only")) or bool(source.get("render_only")):
            render_only.append(raw_path)
        stored = source.get("log_fingerprint", {})
        if not isinstance(stored, dict):
            invalid_fingerprints.append(f"{raw_path}: missing stored fingerprint")
            continue
        current = log_fingerprint(path)
        if stored != current:
            fingerprint_mismatches.append(
                f"{raw_path}: stored {_fingerprint_detail(stored)}; "
                f"current {_fingerprint_detail(current)}"
            )
    _append(checks, "manifest:composite_source_paths_exist", not missing_paths,
            "all source paths exist" if not missing_paths else "; ".join(missing_paths[:5]))
    _append(checks, "manifest:composite_sources_not_render_only", not render_only,
            "all source manifests are real runs" if not render_only
            else "render_only sources: " + ", ".join(render_only[:5]))
    _append(checks, "manifest:composite_source_fingerprints_valid",
            not invalid_fingerprints,
            "valid" if not invalid_fingerprints else "; ".join(invalid_fingerprints[:5]))
    _append(checks, "manifest:composite_source_fingerprints_match",
            not fingerprint_mismatches,
            "all source fingerprints match" if not fingerprint_mismatches
            else "; ".join(fingerprint_mismatches[:5]))
    return checks


def _row_extra(row: dict) -> dict:
    extra = row.get("extra", {}) or {}
    return extra if isinstance(extra, dict) else {}


def _baseline_health(row: dict) -> dict:
    health = _row_extra(row).get("baseline_health", {}) or {}
    return health if isinstance(health, dict) else {}


def _accuracy(rows: list[dict]) -> float:
    return mean(1.0 if r.get("correct") else 0.0 for r in rows) if rows else 0.0


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _operating_metrics(rows: list[dict]) -> dict:
    query_tokens = [_as_float(r.get("query_tokens")) for r in rows
                    if r.get("query_tokens") is not None]
    search_ms = [_as_float(r.get("search_ms")) for r in rows if r.get("search_ms") is not None]
    e2e_ms = [_as_float(r.get("e2e_ms")) for r in rows if r.get("e2e_ms") is not None]
    return {
        "n": len(rows),
        "query_tokens_median": median(query_tokens) if query_tokens else None,
        "search_p95_ms": _percentile(search_ms, 95),
        "e2e_p50_ms": _percentile(e2e_ms, 50),
    }


def _recall_age_slope(rows: list[dict]) -> dict:
    pairs = [
        (_as_float(r.get("age_days")) / 365.25, 1.0 if r.get("correct") else 0.0)
        for r in rows
        if r.get("age_days") is not None
    ]
    distinct_ages = len({round(x, 9) for x, _ in pairs})
    out = {"n": len(pairs), "distinct_ages": distinct_ages, "slope_per_year": None}
    if len(pairs) < 2 or distinct_ages < 2:
        return out
    x_bar = mean(x for x, _ in pairs)
    y_bar = mean(y for _, y in pairs)
    denom = sum((x - x_bar) ** 2 for x, _ in pairs)
    if denom <= 0:
        return out
    out["slope_per_year"] = sum((x - x_bar) * (y - y_bar) for x, y in pairs) / denom
    return out


def _paired_stats(rows: list[dict], headline: str, baseline: str,
                  *, dataset: str | None = None, category: str | None = None) -> dict:
    left: dict[tuple[str, str, str, int], bool] = {}
    right: dict[tuple[str, str, str, int], bool] = {}
    for row in rows:
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if category is not None and row.get("category") != category:
            continue
        if row.get("error"):
            continue
        key = _sample_run_key(row)
        if row.get("system") == headline:
            left[key] = bool(row.get("correct"))
        elif row.get("system") == baseline:
            right[key] = bool(row.get("correct"))
    common = sorted(set(left) & set(right))
    headline_only = baseline_only = both = neither = 0
    for key in common:
        hv, bv = left[key], right[key]
        if hv and bv:
            both += 1
        elif hv and not bv:
            headline_only += 1
        elif bv and not hv:
            baseline_only += 1
        else:
            neither += 1
    n = len(common)
    headline_acc = (headline_only + both) / n if n else 0.0
    baseline_acc = (baseline_only + both) / n if n else 0.0
    return {
        "n": n,
        "headline_accuracy": headline_acc,
        "baseline_accuracy": baseline_acc,
        "delta_pp": (headline_acc - baseline_acc) * 100.0,
        "headline_only": headline_only,
        "baseline_only": baseline_only,
        "both": both,
        "neither": neither,
        "p_mcnemar": _mcnemar_pvalue(headline_only, baseline_only) if n else None,
        "unpaired_headline": len(set(left) - set(right)),
        "unpaired_baseline": len(set(right) - set(left)),
    }


def _sample_clustered_paired_stats(rows: list[dict], headline: str, baseline: str,
                                   *, dataset: str | None = None,
                                   category: str | None = None) -> dict:
    """Paired stats with dataset/category/sample_id as the independent unit.

    A public run may repeat the same samples across many run_idx values. Row-level McNemar is useful
    for reproducibility, but it can overstate evidence if repeated runs over the same tiny slice are
    treated as independent questions. This collapses each dataset/category/sample_id to mean
    correctness per system before computing delta and discordant sample wins.
    """
    left: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    right: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if category is not None and row.get("category") != category:
            continue
        if row.get("error"):
            continue
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            continue
        key = _sample_unit_key(row)
        if row.get("system") == headline:
            left[key].append(bool(row.get("correct")))
        elif row.get("system") == baseline:
            right[key].append(bool(row.get("correct")))

    common = sorted(set(left) & set(right))
    headline_only = baseline_only = ties = 0
    deltas: list[float] = []
    headline_rates: list[float] = []
    baseline_rates: list[float] = []
    for sid in common:
        h_rate = mean(1.0 if x else 0.0 for x in left[sid])
        b_rate = mean(1.0 if x else 0.0 for x in right[sid])
        headline_rates.append(h_rate)
        baseline_rates.append(b_rate)
        deltas.append(h_rate - b_rate)
        if h_rate > b_rate:
            headline_only += 1
        elif b_rate > h_rate:
            baseline_only += 1
        else:
            ties += 1

    n = len(common)
    discordant = headline_only + baseline_only
    return {
        "n": n,
        "headline_accuracy": mean(headline_rates) if headline_rates else 0.0,
        "baseline_accuracy": mean(baseline_rates) if baseline_rates else 0.0,
        "delta_pp": mean(deltas) * 100.0 if deltas else 0.0,
        "headline_only": headline_only,
        "baseline_only": baseline_only,
        "ties": ties,
        "discordant": discordant,
        "p_mcnemar": _mcnemar_pvalue(headline_only, baseline_only) if n else None,
        "unpaired_headline": len(set(left) - set(right)),
        "unpaired_baseline": len(set(right) - set(left)),
    }


def _sample_clustered_accuracy(rows: list[dict]) -> dict:
    by_sample: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            by_sample[_sample_unit_key(row)].append(bool(row.get("correct")))
    rates = [
        mean(1.0 if correct else 0.0 for correct in values)
        for values in by_sample.values()
        if values
    ]
    n = len(rates)
    successes = sum(rates)
    lo, hi = _wilson_ci(successes, n)
    return {
        "n": n,
        "accuracy": (successes / n if n else 0.0),
        "wilson_low": lo,
        "wilson_high": hi,
        "wilson_width_pp": (hi - lo) * 100.0,
    }


def _ci_detail(stats: dict, max_width_pp: float) -> str:
    return (
        f"sample_n={stats.get('n', 0)}, acc={stats.get('accuracy', 0.0) * 100:.1f}%, "
        f"Wilson {stats.get('wilson_low', 0.0) * 100:.1f}-"
        f"{stats.get('wilson_high', 0.0) * 100:.1f} "
        f"(width {stats.get('wilson_width_pp', 0.0):.1f}pp; allowed <= {max_width_pp:.1f}pp)"
    )


def _append(checks: list[dict], name: str, passed: bool, detail: str, **extra) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail, **extra})


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_json_report(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, "missing"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return {}, f"{type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return {}, "not a JSON object"
    return data, ""


def _holdout_audit_summary(report: dict, *, min_needles: int) -> dict:
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        findings = ["findings is not a list"]
    registry_error = str(report.get("registry_error", "") or "")
    holdout_needles = _as_int(report.get("holdout_needles_checked", 0), 0)
    needles = _as_int(report.get("needles_checked", 0), 0)
    legacy_scan_enabled = bool(report.get("legacy_policy_scan_enabled"))
    forbidden_policy_count = _as_int(report.get("forbidden_policy_strings_checked", 0), 0)
    forbidden_fixed_answer_count = _as_int(report.get("forbidden_fixed_answer_strings_checked", 0), 0)
    forbidden_runtime_count = _as_int(report.get("forbidden_runtime_symbols_checked", 0), 0)
    failures: list[str] = []
    if not bool(report.get("pass")):
        failures.append("pass:false")
    if findings:
        failures.append(f"findings:{len(findings)}")
    if registry_error:
        failures.append(f"registry_error:{registry_error}")
    if holdout_needles < min_needles:
        failures.append(f"holdout_needles_checked:{holdout_needles}<required:{min_needles}")
    if not legacy_scan_enabled:
        failures.append("legacy_policy_scan_enabled:false")
    if forbidden_policy_count <= 0:
        failures.append("forbidden_policy_strings_checked:0")
    if forbidden_fixed_answer_count <= 0:
        failures.append("forbidden_fixed_answer_strings_checked:0")
    if forbidden_runtime_count <= 0:
        failures.append("forbidden_runtime_symbols_checked:0")
    return {
        "pass": not failures,
        "needles_checked": needles,
        "holdout_needles_checked": holdout_needles,
        "legacy_policy_scan_enabled": legacy_scan_enabled,
        "forbidden_policy_strings_checked": forbidden_policy_count,
        "forbidden_fixed_answer_strings_checked": forbidden_fixed_answer_count,
        "forbidden_runtime_symbols_checked": forbidden_runtime_count,
        "findings_count": len(findings),
        "registry_error": registry_error,
        "failures": failures,
    }


def _metric_fraction(row: dict, keys: tuple[str, ...] = _ACCURACY_KEYS) -> tuple[str, float | None]:
    if not isinstance(row, dict):
        return "", None
    for key in keys:
        if key not in row:
            continue
        value = _float_or_none(row.get(key))
        if value is None:
            continue
        if 1.0 < value <= 100.0:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return key, value
    return "", None


def _metric_n(row: dict) -> int:
    if not isinstance(row, dict):
        return 0
    for key in ("n", "samples", "total_n", "cases", "questions"):
        value = _as_int(row.get(key, 0), 0)
        if value > 0:
            return value
    return 0


def _paired_metric_float(left: dict, right: dict,
                         keys: tuple[str, ...] = _COST_KEYS) -> tuple[str, float | None, float | None]:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return "", None, None
    for key in keys:
        if key not in left or key not in right:
            continue
        left_value = _float_or_none(left.get(key))
        right_value = _float_or_none(right.get(key))
        if left_value is None or right_value is None:
            continue
        if left_value > 0.0 and right_value > 0.0:
            return key, left_value, right_value
    return "", None, None


def _first_ablation(ablations: dict, labels: tuple[str, ...]) -> tuple[str, dict]:
    if not isinstance(ablations, dict):
        return "", {}
    for label in labels:
        row = ablations.get(label)
        if isinstance(row, dict):
            return label, row
    return "", {}


def _evidence_ref_count(report: dict) -> int:
    count = 0
    for key in ("log_fingerprints", "artifact_fingerprints", "artifacts"):
        value = report.get(key)
        if isinstance(value, list):
            count += sum(1 for item in value if item)
        elif isinstance(value, dict) and value:
            count += 1
    if isinstance(report.get("log_fingerprint"), dict) and report["log_fingerprint"].get("combined_sha256"):
        count += 1
    return count


def _ablation_evidence_summary(report: dict, *, expected_system: str, expected_split: str,
                               min_samples: int, min_metabolism_accuracy_delta_pp: float,
                               min_region_accuracy_delta_pp: float,
                               min_affect_accuracy_delta_pp: float,
                               min_forgetting_cost_ratio: float,
                               max_forgetting_accuracy_regression_pp: float) -> dict:
    failures: list[str] = []
    status = str(report.get("status", "") or "").strip().upper()
    declared_pass = bool(report.get("pass")) or status == "PASS"
    if not declared_pass:
        failures.append(f"pass:{status or report.get('pass', '<missing>')}")

    system = str(report.get("system", "") or "").strip()
    if expected_system and system != expected_system:
        failures.append(f"system:{system or '<missing>'}:expected:{expected_system}")

    split = str(report.get("split", "") or "").strip().lower()
    if expected_split and split != expected_split:
        failures.append(f"split:{split or '<missing>'}:expected:{expected_split}")

    evidence_refs = _evidence_ref_count(report)
    if evidence_refs <= 0:
        failures.append("evidence_refs:0")

    full = report.get("full") if isinstance(report.get("full"), dict) else {}
    ablations = report.get("ablations") if isinstance(report.get("ablations"), dict) else {}
    metabolism_label, metabolism = _first_ablation(ablations, _METABOLISM_ABLATION_KEYS)
    region_label, region = _first_ablation(ablations, _REGION_ABLATION_KEYS)
    forgetting_label, forgetting = _first_ablation(ablations, _FORGETTING_ABLATION_KEYS)
    affect_label, affect = _first_ablation(ablations, _AFFECT_ABLATION_KEYS)
    if not metabolism_label:
        failures.append("ablation:metabolism_off_missing")
    if not region_label:
        failures.append("ablation:regions_off_missing")
    if not forgetting_label:
        failures.append("ablation:forgetting_off_missing")
    if not affect_label:
        failures.append("ablation:affect_off_missing")

    full_n = _metric_n(full)
    metabolism_n = _metric_n(metabolism)
    region_n = _metric_n(region)
    forgetting_n = _metric_n(forgetting)
    affect_n = _metric_n(affect)
    for label, n in (
        ("full", full_n),
        (metabolism_label or "metabolism_off", metabolism_n),
        (region_label or "regions_off", region_n),
        (forgetting_label or "forgetting_off", forgetting_n),
        (affect_label or "affect_off", affect_n),
    ):
        if n < min_samples:
            failures.append(f"{label}:n:{n}<required:{min_samples}")

    full_acc_key, full_acc = _metric_fraction(full)
    metabolism_acc_key, metabolism_acc = _metric_fraction(metabolism)
    region_acc_key, region_acc = _metric_fraction(region)
    forgetting_acc_key, forgetting_acc = _metric_fraction(forgetting)
    affect_acc_key, affect_acc = _metric_fraction(affect)
    if full_acc is None:
        failures.append("full:accuracy_missing")
    if metabolism_acc is None:
        failures.append(f"{metabolism_label or 'metabolism_off'}:accuracy_missing")
    if region_acc is None:
        failures.append(f"{region_label or 'regions_off'}:accuracy_missing")
    if forgetting_acc is None:
        failures.append(f"{forgetting_label or 'forgetting_off'}:accuracy_missing")
    if affect_acc is None:
        failures.append(f"{affect_label or 'affect_off'}:accuracy_missing")

    metabolism_delta_pp = None
    region_delta_pp = None
    affect_delta_pp = None
    forgetting_accuracy_regression_pp = None
    if full_acc is not None and metabolism_acc is not None:
        metabolism_delta_pp = (full_acc - metabolism_acc) * 100.0
        if metabolism_delta_pp < min_metabolism_accuracy_delta_pp:
            failures.append(
                f"metabolism_delta_pp:{metabolism_delta_pp:.2f}<required:"
                f"{min_metabolism_accuracy_delta_pp:.2f}"
            )
    if full_acc is not None and region_acc is not None:
        region_delta_pp = (full_acc - region_acc) * 100.0
        if region_delta_pp < min_region_accuracy_delta_pp:
            failures.append(
                f"region_delta_pp:{region_delta_pp:.2f}<required:"
                f"{min_region_accuracy_delta_pp:.2f}"
            )
    if full_acc is not None and affect_acc is not None:
        affect_delta_pp = (full_acc - affect_acc) * 100.0
        if affect_delta_pp < min_affect_accuracy_delta_pp:
            failures.append(
                f"affect_delta_pp:{affect_delta_pp:.2f}<required:"
                f"{min_affect_accuracy_delta_pp:.2f}"
            )
    if full_acc is not None and forgetting_acc is not None:
        forgetting_accuracy_regression_pp = max(0.0, (forgetting_acc - full_acc) * 100.0)
        if forgetting_accuracy_regression_pp > max_forgetting_accuracy_regression_pp:
            failures.append(
                f"forgetting_accuracy_regression_pp:{forgetting_accuracy_regression_pp:.2f}>allowed:"
                f"{max_forgetting_accuracy_regression_pp:.2f}"
            )

    cost_metric, full_cost, forgetting_cost = _paired_metric_float(full, forgetting)
    forgetting_cost_ratio = None
    if full_cost is None or forgetting_cost is None:
        failures.append("forgetting_cost_metric_missing")
    else:
        forgetting_cost_ratio = forgetting_cost / full_cost
        if forgetting_cost_ratio < min_forgetting_cost_ratio:
            failures.append(
                f"forgetting_cost_ratio:{forgetting_cost_ratio:.3f}<required:"
                f"{min_forgetting_cost_ratio:.3f}"
            )

    return {
        "pass": not failures,
        "system": system,
        "split": split,
        "evidence_refs": evidence_refs,
        "full_n": full_n,
        "metabolism_label": metabolism_label,
        "metabolism_n": metabolism_n,
        "region_label": region_label,
        "region_n": region_n,
        "forgetting_label": forgetting_label,
        "forgetting_n": forgetting_n,
        "affect_label": affect_label,
        "affect_n": affect_n,
        "accuracy_metric": full_acc_key or metabolism_acc_key or region_acc_key or forgetting_acc_key or affect_acc_key,
        "full_accuracy": full_acc,
        "metabolism_off_accuracy": metabolism_acc,
        "regions_off_accuracy": region_acc,
        "forgetting_off_accuracy": forgetting_acc,
        "affect_off_accuracy": affect_acc,
        "metabolism_delta_pp": None if metabolism_delta_pp is None else round(metabolism_delta_pp, 4),
        "region_delta_pp": None if region_delta_pp is None else round(region_delta_pp, 4),
        "affect_delta_pp": None if affect_delta_pp is None else round(affect_delta_pp, 4),
        "forgetting_accuracy_regression_pp": (
            None if forgetting_accuracy_regression_pp is None
            else round(forgetting_accuracy_regression_pp, 4)
        ),
        "cost_metric": cost_metric,
        "full_cost": full_cost,
        "forgetting_off_cost": forgetting_cost,
        "forgetting_cost_ratio": None if forgetting_cost_ratio is None else round(forgetting_cost_ratio, 4),
        "failures": failures,
    }


_EXTERNAL_EVIDENCE_REQUIRED_FIELDS = {
    "system", "dataset", "split", "n", "runs", "score",
    "metric", "evaluation_protocol", "date", "source", "artifact_fingerprint",
}


def _external_evidence_systems(
    claim_scope: dict,
    *,
    min_n: int = 100,
    min_runs: int = 1,
) -> tuple[set[str], list[str]]:
    """Return external systems backed by structured evidence records.

    A public SOTA/best-in-world claim cannot be supported by names alone. Each external comparison
    needs enough metadata for a reviewer to see what was measured and where the evidence lives.
    """
    raw = claim_scope.get("external_system_evidence", []) or []
    if not isinstance(raw, list):
        return set(), ["external_system_evidence is not a list"]
    systems: set[str] = set()
    errors: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"record {i} is not an object")
            continue
        missing = sorted(
            field for field in _EXTERNAL_EVIDENCE_REQUIRED_FIELDS
            if item.get(field) in (None, "")
        )
        if missing:
            errors.append(f"{item.get('system', f'record {i}')}: missing {', '.join(missing)}")
            continue
        try:
            n = int(item.get("n"))
            runs = int(item.get("runs"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            errors.append(f"{item.get('system', f'record {i}')}: n/runs/score must be numeric")
            continue
        label = f"{item.get('system', f'record {i}')}:{item.get('dataset', '<dataset>')}"
        if n < min_n:
            errors.append(f"{label}: n {n} < required {min_n}")
            continue
        if runs < min_runs:
            errors.append(f"{label}: runs {runs} < required {min_runs}")
            continue
        if not (0.0 <= score <= 1.0 or 0.0 <= score <= 100.0):
            errors.append(f"{label}: score out of range")
            continue
        fingerprint = str(item.get("artifact_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint.lower()):
            errors.append(f"{label}: artifact_fingerprint must be a sha256 hex digest")
            continue
        systems.add(str(item["system"]).strip().lower())
    return systems, errors


def _external_evidence_datasets(
    claim_scope: dict,
    *,
    min_n: int = 100,
    min_runs: int = 1,
) -> dict[str, set[str]]:
    """Valid external evidence records grouped by system -> covered datasets."""
    raw = claim_scope.get("external_system_evidence", []) or []
    if not isinstance(raw, list):
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    for item in raw:
        if not isinstance(item, dict):
            continue
        if any(item.get(field) in (None, "") for field in _EXTERNAL_EVIDENCE_REQUIRED_FIELDS):
            continue
        try:
            n = int(item.get("n"))
            runs = int(item.get("runs"))
            float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if n < min_n or runs < min_runs:
            continue
        if str(item.get("split", "")).strip().lower() != "test":
            continue
        fingerprint = str(item.get("artifact_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint.lower()):
            continue
        system = str(item.get("system", "")).strip().lower()
        dataset = str(item.get("dataset", "")).strip().lower()
        if system and dataset:
            out[system].add(dataset)
    return out


def _normalized_external_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    if 1.0 < score <= 100.0:
        return score / 100.0
    return None


def _external_evidence_scores(
    claim_scope: dict,
    *,
    min_n: int = 100,
    min_runs: int = 1,
) -> dict[str, dict[str, float]]:
    """Best valid external score per system/dataset, normalized to 0..1."""
    raw = claim_scope.get("external_system_evidence", []) or []
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for item in raw:
        if not isinstance(item, dict):
            continue
        if any(item.get(field) in (None, "") for field in _EXTERNAL_EVIDENCE_REQUIRED_FIELDS):
            continue
        try:
            n = int(item.get("n"))
            runs = int(item.get("runs"))
        except (TypeError, ValueError):
            continue
        if n < min_n or runs < min_runs:
            continue
        if str(item.get("split", "")).strip().lower() != "test":
            continue
        fingerprint = str(item.get("artifact_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint.lower()):
            continue
        score = _normalized_external_score(item.get("score"))
        if score is None:
            continue
        system = str(item.get("system", "")).strip().lower()
        dataset = str(item.get("dataset", "")).strip().lower()
        if not system or not dataset:
            continue
        out[system][dataset] = max(score, out[system].get(dataset, 0.0))
    return out


def _fingerprint_detail(fingerprint: dict) -> str:
    return (
        f"{fingerprint.get('combined_sha256', '<missing>')} "
        f"({fingerprint.get('file_count', 0)} files)"
    )


def _matching_fingerprint(report: dict, current: dict) -> tuple[bool, bool, str]:
    fingerprint = report.get("log_fingerprint")
    if not isinstance(fingerprint, dict):
        return False, False, "<missing>"
    return True, fingerprint == current, _fingerprint_detail(fingerprint)


def run_release_gate(
    out_dir: Path,
    *,
    required_systems: list[str] | None = None,
    required_datasets: list[str] | None = None,
    headline_system: str = "eidetic-plus",
    baseline_systems: list[str] | None = None,
    required_categories_by_dataset: dict[str, list[str]] | None = None,
    integrity_system: str = "eidetic-plus-full",
    split: str = "test",
    min_runs: int = 10,
    min_questions_per_system: int = 1000,
    min_questions_per_dataset_per_system: int | None = None,
    min_category_questions_per_system: int | None = None,
    min_dataset_accuracy: float = 0.85,
    min_overall_delta_pp: float = 10.0,
    min_category_delta_pp: float = 0.0,
    alpha: float = 0.05,
    max_dataset_accuracy_ci_width_pp: float = 10.0,
    max_category_accuracy_ci_width_pp: float = 30.0,
    min_sample_clustered_paired_samples: int | None = None,
    min_category_sample_clustered_paired_samples: int | None = None,
    require_ci_clear_dominance: bool = True,
    min_clustered_discordant_samples: int = 6,
    min_verified_accuracy: float = 0.50,
    max_consolidation_timeouts: int = 0,
    max_consolidation_deferred: int = 0,
    min_snap_back_records: int = 1,
    max_query_tokens_median: float = 7000.0,
    max_search_p95_ms: float = 500.0,
    max_e2e_p50_ms: float = 5000.0,
    token_efficiency_baseline: str = "rag-full",
    min_token_efficiency_ratio: float = 10.0,
    max_abs_recall_slope_per_year: float = 0.10,
    min_age_slope_points: int = 20,
    require_baseline_reproduction: bool = True,
    baseline_reproduction_report: str = "mem0_gate.json",
    baseline_reproduction_system: str = "mem0",
    baseline_reproduction_dataset: str = "locomo",
    require_category_wins: bool = True,
    require_competitor_health: bool = True,
    health_required_systems: list[str] | None = None,
    require_claim_scope: bool = True,
    claim_scope_report: str = "claim_scope.json",
    top_systems_for_sota: list[str] | None = None,
    min_external_evidence_n: int = 100,
    min_external_evidence_runs: int = 1,
    require_abstention_calibration: bool = True,
    abstention_calibration_report: str = "abstention_v2_tau.json",
    abstention_calibration_system: str | None = None,
    min_abstention_calibration_samples: int = 50,
    min_abstention_precision_target: float = 0.95,
    require_holdout_profile: bool = True,
    require_holdout_audit: bool = True,
    holdout_audit_report: str = "holdout_audit.json",
    min_holdout_audit_needles: int = 1,
    require_smqe_log_policy: bool = True,
    min_smqe_log_structured_rate: float = 0.80,
    min_smqe_log_claim_backend_rate: float = 0.80,
    require_ablation_evidence: bool = True,
    ablation_report: str = "ablation_report.json",
    ablation_split: str = "dev",
    min_ablation_samples: int = 20,
    min_metabolism_accuracy_delta_pp: float = 5.0,
    min_region_accuracy_delta_pp: float = 2.0,
    min_affect_accuracy_delta_pp: float = 2.0,
    min_forgetting_cost_ratio: float = 1.05,
    max_forgetting_accuracy_regression_pp: float = 1.0,
    require_affect_salience_invariant: bool = True,
    affect_salience_report: str = "affect_salience_invariant.json",
    min_affect_salience_cases: int = 24,
    max_affect_salience_lambda: float = 0.5,
    max_affect_salience_boost_ratio: float = 0.5,
    min_affect_salience_age_gap_seconds: float = 2_592_000.0,
    require_scratchpad_invariant: bool = True,
    scratchpad_report: str = "scratchpad_invariant.json",
    min_scratchpad_cases: int = 24,
    require_region_routing_invariant: bool = True,
    region_routing_report: str = "region_routing_invariant.json",
    min_region_routing_cases: int = 24,
    require_reflex_recall_invariant: bool = True,
    reflex_recall_report: str = "reflex_recall_invariant.json",
    min_reflex_recall_cases: int = 24,
    max_reflex_recall_p95_ms: float = 100.0,
    require_slice_invariant: bool = True,
    slice_invariant_report: str = "slice_invariant.json",
    min_slice_invariant_draws: int = 5,
    min_slice_invariant_subset: int = 20,
    require_smqe_planner_invariant: bool = True,
    smqe_planner_report: str = "smqe_planner_invariant.json",
    min_smqe_planner_cases: int = 24,
    max_smqe_planner_p95_ms: float = 10.0,
    require_smqe_synthetic_invariant: bool = True,
    smqe_synthetic_report: str = "smqe_synthetic_invariant.json",
    min_smqe_synthetic_cases: int = 24,
    max_smqe_synthetic_avg_proof_tokens: float = 80.0,
    require_smqe_claim_coverage: bool = True,
    smqe_claim_coverage_report: str = "smqe_claim_coverage.json",
    min_smqe_claim_coverage_cases: int = 24,
    min_smqe_claim_backend_rate: float = 1.0,
    max_smqe_claim_avg_proof_tokens: float = 80.0,
    require_smqe_fullpath_invariant: bool = True,
    smqe_fullpath_report: str = "smqe_fullpath_invariant.json",
    min_smqe_fullpath_cases: int = 24,
    max_smqe_fullpath_avg_proof_tokens: float = 80.0,
    max_smqe_fullpath_avg_context_tokens: float = 80.0,
    max_smqe_fullpath_p95_ms: float = 100.0,
    require_smqe_paraphrase_invariant: bool = True,
    smqe_paraphrase_report: str = "smqe_paraphrase_invariant.json",
    min_smqe_paraphrase_cases: int = 24,
    max_smqe_paraphrase_avg_proof_tokens: float = 80.0,
    require_smqe_conflict_invariant: bool = True,
    smqe_conflict_report: str = "smqe_conflict_invariant.json",
    min_smqe_conflict_cases: int = 24,
    max_smqe_conflict_avg_proof_tokens: float = 80.0,
    require_smqe_composition_invariant: bool = True,
    smqe_composition_report: str = "smqe_composition_invariant.json",
    min_smqe_composition_cases: int = 24,
    max_smqe_composition_avg_proof_tokens: float = 80.0,
    require_smqe_relative_phrase_invariant: bool = True,
    smqe_relative_phrase_report: str = "smqe_relative_phrase_invariant.json",
    min_smqe_relative_phrase_cases: int = 24,
    max_smqe_relative_phrase_avg_proof_tokens: float = 80.0,
    require_smqe_temporal_window_invariant: bool = True,
    smqe_temporal_window_report: str = "smqe_temporal_window_invariant.json",
    min_smqe_temporal_window_cases: int = 24,
    max_smqe_temporal_window_avg_proof_tokens: float = 80.0,
    require_smqe_attribution_invariant: bool = True,
    smqe_attribution_report: str = "smqe_attribution_invariant.json",
    min_smqe_attribution_cases: int = 24,
    max_smqe_attribution_avg_proof_tokens: float = 80.0,
    require_smqe_abstention_invariant: bool = True,
    smqe_abstention_report: str = "smqe_abstention_invariant.json",
    min_smqe_abstention_cases: int = 24,
    require_smqe_scope_invariant: bool = True,
    smqe_scope_report: str = "smqe_scope_invariant.json",
    min_smqe_scope_cases: int = 24,
    max_smqe_scope_avg_proof_tokens: float = 80.0,
    require_smqe_subscope_invariant: bool = True,
    smqe_subscope_report: str = "smqe_subscope_invariant.json",
    min_smqe_subscope_cases: int = 24,
    max_smqe_subscope_avg_proof_tokens: float = 80.0,
    require_smqe_time_invariant: bool = True,
    smqe_time_report: str = "smqe_time_invariant.json",
    min_smqe_time_cases: int = 24,
    max_smqe_time_avg_proof_tokens: float = 80.0,
    require_smqe_invalidation_invariant: bool = True,
    smqe_invalidation_report: str = "smqe_invalidation_invariant.json",
    min_smqe_invalidation_cases: int = 24,
    max_smqe_invalidation_avg_proof_tokens: float = 80.0,
) -> dict:
    out_dir = Path(out_dir)
    required_systems = required_systems if required_systems is not None else [
        "eidetic-plus", "eidetic-plus-full", "eidetic-product",
        "rag-full", "rag-vector", "mem0", "graphiti",
    ]
    required_datasets = required_datasets if required_datasets is not None else ["longmemeval", "locomo"]
    baseline_systems = (
        baseline_systems if baseline_systems is not None
        else ["rag-full", "rag-vector", "mem0", "graphiti"]
    )
    min_questions_per_dataset = (
        int(min_questions_per_dataset_per_system)
        if min_questions_per_dataset_per_system is not None
        else min(300, max(1, int(min_questions_per_system) // max(1, len(required_datasets))))
    )
    required_categories = _required_categories_for(
        required_datasets,
        required_categories_by_dataset,
    )
    max_required_category_count = max((len(cats) for cats in required_categories.values()), default=1)
    min_category_questions = (
        int(min_category_questions_per_system)
        if min_category_questions_per_system is not None
        else min(20, max(1, min_questions_per_dataset // max(1, max_required_category_count)))
    )
    min_clustered_paired = (
        int(min_sample_clustered_paired_samples)
        if min_sample_clustered_paired_samples is not None
        else int(min_questions_per_system)
    )
    min_category_clustered_paired = (
        int(min_category_sample_clustered_paired_samples)
        if min_category_sample_clustered_paired_samples is not None
        else int(min_category_questions)
    )
    health_required_systems = (
        health_required_systems if health_required_systems is not None else ["mem0", "graphiti"]
    )
    top_systems_for_sota = (
        top_systems_for_sota
        if top_systems_for_sota is not None
        else ["chronos", "mastra", "byterover", "hindsight"]
    )
    abstention_calibration_system = abstention_calibration_system or integrity_system

    checks: list[dict] = []
    checks.extend(_artifact_checks(out_dir))
    manifest = _load_manifest(out_dir)
    initial_fingerprint = log_fingerprint(out_dir)
    rows = _load_logs_strict(out_dir).rows
    current_fingerprint = log_fingerprint(out_dir)
    _append(checks, "logs:fingerprint_stable", initial_fingerprint == current_fingerprint,
            f"before {_fingerprint_detail(initial_fingerprint)}; "
            f"after {_fingerprint_detail(current_fingerprint)}")
    agg = aggregate(rows)
    by_sys = _rows_by_system(rows)
    systems_present = set(by_sys)
    checks.extend(_composite_source_checks(manifest))

    scoreboard_report, scoreboard_error = _load_json_report(out_dir / "scoreboard.json")
    _append(checks, "scoreboard:valid_json", not scoreboard_error,
            "valid" if not scoreboard_error else scoreboard_error)
    scoreboard_has_fp, scoreboard_fp_matches, scoreboard_fp_detail = _matching_fingerprint(
        scoreboard_report, current_fingerprint
    )
    _append(checks, "scoreboard:log_fingerprint_present", scoreboard_has_fp,
            scoreboard_fp_detail)
    _append(checks, "scoreboard:log_fingerprint_matches", scoreboard_fp_matches,
            f"{scoreboard_fp_detail} (current {_fingerprint_detail(current_fingerprint)})")

    claim_scope = {}
    if require_claim_scope:
        claim_scope, claim_scope_error = _load_json_report(out_dir / claim_scope_report)
        _append(checks, "claim_scope:valid_json", not claim_scope_error,
                "valid" if not claim_scope_error else claim_scope_error)
        raw_scope = str(
            claim_scope.get("public_claim_scope", claim_scope.get("scope", ""))
            or ""
        ).strip().lower()
        allowed_scopes = {
            "measured-harness-only", "measured_baselines_only", "limited",
            "sota", "best-in-world", "best_in_world", "field-leading", "field_leading",
        }
        sota_scopes = {"sota", "best-in-world", "best_in_world", "field-leading", "field_leading"}
        _append(checks, "claim_scope:scope_declared", raw_scope in allowed_scopes,
                raw_scope or "<missing>")
        measured_external = {
            str(x).strip().lower()
            for x in (claim_scope.get("measured_external_systems", []) or [])
            if str(x).strip()
        }
        measured_harness = {
            str(x).strip().lower()
            for x in (claim_scope.get("measured_harness_systems", []) or [])
            if str(x).strip()
        }
        evidence_external, evidence_errors = _external_evidence_systems(
            claim_scope,
            min_n=min_external_evidence_n,
            min_runs=min_external_evidence_runs,
        )
        evidence_datasets = _external_evidence_datasets(
            claim_scope,
            min_n=min_external_evidence_n,
            min_runs=min_external_evidence_runs,
        )
        evidence_scores = _external_evidence_scores(
            claim_scope,
            min_n=min_external_evidence_n,
            min_runs=min_external_evidence_runs,
        )
        logged_systems = {s.lower() for s in systems_present}
        measured_all = logged_systems | evidence_external
        harness_without_logs = sorted(measured_harness - logged_systems)
        external_without_evidence = sorted(measured_external - evidence_external)
        _append(checks, "claim_scope:harness_names_have_logs",
                not harness_without_logs,
                "all harness names have logs" if not harness_without_logs
                else "missing logs for: " + ", ".join(harness_without_logs))
        _append(checks, "claim_scope:external_evidence_valid",
                raw_scope not in sota_scopes or not evidence_errors,
                "not a SOTA claim" if raw_scope not in sota_scopes
                else ("valid" if not evidence_errors else "; ".join(evidence_errors[:5])))
        _append(checks, "claim_scope:external_names_have_evidence",
                not external_without_evidence,
                "all external names have evidence" if not external_without_evidence
                else "missing evidence for: " + ", ".join(external_without_evidence))
        required_top = {s.strip().lower() for s in top_systems_for_sota if s.strip()}
        missing_top = sorted(required_top - measured_all)
        _append(checks, "claim_scope:no_unsupported_sota",
                raw_scope not in sota_scopes or not missing_top,
                "not a SOTA claim" if raw_scope not in sota_scopes
                else ("all required top systems measured" if not missing_top
                      else "missing: " + ", ".join(missing_top)))
        missing_top_datasets = []
        required_dataset_set = {d.strip().lower() for d in required_datasets if d.strip()}
        for system in sorted(required_top):
            if system in logged_systems:
                continue
            missing_ds = sorted(required_dataset_set - evidence_datasets.get(system, set()))
            if missing_ds:
                missing_top_datasets.append(f"{system}: {','.join(missing_ds)}")
        _append(checks, "claim_scope:top_system_dataset_coverage",
                raw_scope not in sota_scopes or not missing_top_datasets,
                "not a SOTA claim" if raw_scope not in sota_scopes
                else ("all required top-system datasets covered" if not missing_top_datasets
                      else "; ".join(missing_top_datasets[:8])))
        external_score_failures = []
        if raw_scope in sota_scopes:
            for system in sorted(required_top):
                if system in logged_systems:
                    continue
                for dataset, external_score in sorted(evidence_scores.get(system, {}).items()):
                    if dataset not in required_dataset_set:
                        continue
                    headline_acc = _accuracy([
                        row for row in rows
                        if row.get("system") == headline_system
                        and str(row.get("dataset", "")).strip().lower() == dataset
                        and not row.get("error")
                    ])
                    if headline_acc + 1e-12 < external_score:
                        external_score_failures.append(
                            f"{system}:{dataset}:headline {headline_acc:.3f}<external {external_score:.3f}"
                        )
        _append(checks, "claim_scope:top_system_score_floor",
                raw_scope not in sota_scopes or not external_score_failures,
                "not a SOTA claim" if raw_scope not in sota_scopes
                else ("headline meets/exceeds external top-system scores"
                      if not external_score_failures
                      else "; ".join(external_score_failures[:8])))
        limitations = claim_scope.get("limitations", []) or []
        _append(checks, "claim_scope:limitations_for_limited_claim",
                raw_scope in sota_scopes or bool(limitations),
                "SOTA claim" if raw_scope in sota_scopes
                else f"{len(limitations) if isinstance(limitations, list) else 1} limitations")

    _append(checks, "manifest:split", manifest.get("split") == split,
            f"{manifest.get('split')} (expected {split})")
    _append(checks, "manifest:runs", int(manifest.get("runs", 0) or 0) >= min_runs,
            f"{manifest.get('runs')} (required >= {min_runs})")
    _append(checks, "manifest:not_render_only", not bool(manifest.get("render_only")),
            f"render_only={manifest.get('render_only')}")
    manifest_systems = _manifest_systems(manifest)
    missing_manifest_systems = [
        system for system in required_systems
        if _canonical_system_name(system) not in manifest_systems
    ]
    _append(checks, "manifest:systems_cover_required",
            not missing_manifest_systems,
            "all required systems recorded" if not missing_manifest_systems
            else "missing: " + ", ".join(missing_manifest_systems))
    manifest_sample_rows = _manifest_sample_rows(manifest)
    _append(checks, "manifest:sample_rows_present",
            bool(manifest_sample_rows),
            f"{len(manifest_sample_rows)} sample rows")
    manifest_samples_by_ds_cat: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in manifest_sample_rows:
        manifest_samples_by_ds_cat[(sample["dataset"], sample["category"])].add(sample["sample_id"])
    for dataset, required_cats in required_categories.items():
        missing_manifest_cats = [
            category
            for category in required_cats
            if not manifest_samples_by_ds_cat.get((dataset, category))
        ]
        _append(checks, f"manifest:{dataset}:categories_cover_required",
                not missing_manifest_cats,
                "all required categories present" if not missing_manifest_cats
                else "missing: " + ", ".join(missing_manifest_cats))
        for category in required_cats:
            n_manifest_cat = len(manifest_samples_by_ds_cat.get((dataset, category), set()))
            _append(checks, f"manifest:{dataset}:{category}:sample_rows",
                    n_manifest_cat >= min_category_questions,
                    f"{n_manifest_cat} unique sample rows "
                    f"(required >= {min_category_questions})")
    raw_system_failures = manifest.get("system_failures", []) or []
    system_failures = raw_system_failures if isinstance(raw_system_failures, list) else [
        {"system": "<malformed>", "error_type": "MalformedManifest", "error": str(raw_system_failures)}
    ]
    failure_names = [
        str(item.get("system", "<unknown>")) if isinstance(item, dict) else "<malformed>"
        for item in system_failures
    ]
    _append(checks, "manifest:no_system_failures", not system_failures,
            "none" if not system_failures else "failed: " + ", ".join(failure_names[:8]))
    env = manifest.get("env", {}) if isinstance(manifest.get("env"), dict) else {}
    manifest_data_dir = str(env.get("DATA_DIR", "") or "")
    is_composite = str(manifest.get("artifact_kind", "")).strip().lower() == "composite"
    source_data_dirs = []
    for source in (manifest.get("composite_sources", []) or []):
        if not isinstance(source, dict):
            continue
        snap = source.get("snap_back_audit", {}) or {}
        if str(snap.get("status", "")).strip().upper() == "SKIP":
            continue
        if not bool(source.get("snap_back_required", True)):
            continue
        source_data_dirs.append(str(snap.get("data_dir", "") or ""))
    composite_data_dirs_ok = bool(source_data_dirs) and all(source_data_dirs)
    _append(checks, "manifest:data_dir_recorded",
            bool(manifest_data_dir) or (is_composite and composite_data_dirs_ok),
            manifest_data_dir or (
                f"composite source data dirs: {len(source_data_dirs)}"
                if is_composite and composite_data_dirs_ok else "<unset>"
            ))
    dataset_scan_env_enabled = _truthy(env.get("EIDETIC_ENABLE_DATASET_SOURCE_SCANS"))
    _append(checks, "manifest:no_dataset_source_scans",
            not dataset_scan_env_enabled,
            "disabled" if not dataset_scan_env_enabled
            else "EIDETIC_ENABLE_DATASET_SOURCE_SCANS is enabled")
    ingest_granularity = str(env.get("INGEST_GRANULARITY", "") or "").strip().lower()
    session_ingest = ingest_granularity in ("", "session")
    _append(checks, "manifest:session_ingest_granularity",
            session_ingest,
            "session" if session_ingest else f"INGEST_GRANULARITY={ingest_granularity}")
    holdout_profile = str(manifest.get("holdout_profile", "") or "").strip().lower()
    samples_file = str(manifest.get("samples_file", "") or "").strip()
    if require_holdout_profile:
        _append(checks, "manifest:holdout_profile",
                holdout_profile == "holdout",
                holdout_profile or "<missing>")
        _append(checks, "manifest:samples_file_recorded",
                bool(samples_file),
                samples_file or "<missing>")
    holdout_audit = {}
    if require_holdout_audit:
        holdout_audit, holdout_audit_error = _load_json_report(out_dir / holdout_audit_report)
        _append(checks, "holdout_audit:valid_json",
                not holdout_audit_error,
                "valid" if not holdout_audit_error else holdout_audit_error)
        holdout_audit_summary = _holdout_audit_summary(
            holdout_audit,
            min_needles=min_holdout_audit_needles,
        )
        _append(checks, "holdout_audit:evidence",
                holdout_audit_summary["pass"],
                (
                    f"{holdout_audit_summary['holdout_needles_checked']} holdout needles, "
                    f"{holdout_audit_summary['findings_count']} findings"
                )
                if holdout_audit_summary["pass"]
                else "; ".join(holdout_audit_summary["failures"][:8]),
                **holdout_audit_summary)
    ablation_evidence = {}
    if require_ablation_evidence:
        raw_ablation, ablation_error = _load_json_report(out_dir / ablation_report)
        _append(checks, "ablation:valid_json",
                not ablation_error,
                "valid" if not ablation_error else ablation_error)
        ablation_evidence = _ablation_evidence_summary(
            raw_ablation,
            expected_system=integrity_system,
            expected_split=ablation_split,
            min_samples=min_ablation_samples,
            min_metabolism_accuracy_delta_pp=min_metabolism_accuracy_delta_pp,
            min_region_accuracy_delta_pp=min_region_accuracy_delta_pp,
            min_affect_accuracy_delta_pp=min_affect_accuracy_delta_pp,
            min_forgetting_cost_ratio=min_forgetting_cost_ratio,
            max_forgetting_accuracy_regression_pp=max_forgetting_accuracy_regression_pp,
        )
        _append(checks, "ablation:evidence",
                ablation_evidence["pass"],
                (
                    f"metabolism +{ablation_evidence['metabolism_delta_pp']:.1f}pp, "
                    f"regions +{ablation_evidence['region_delta_pp']:.1f}pp, "
                    f"affect +{ablation_evidence['affect_delta_pp']:.1f}pp, "
                    f"forgetting cost ratio {ablation_evidence['forgetting_cost_ratio']:.3f}"
                )
                if ablation_evidence["pass"]
                else "; ".join(ablation_evidence["failures"][:8]),
                **ablation_evidence)
    affect_salience_invariant = {}
    if require_affect_salience_invariant:
        affect_salience_invariant, affect_salience_error = _load_json_report(out_dir / affect_salience_report)
        _append(checks, "affect_salience:valid_json",
                not affect_salience_error,
                "valid" if not affect_salience_error else affect_salience_error)
        affect_salience_summary = _affect_salience_summary(
            affect_salience_invariant,
            min_cases=min_affect_salience_cases,
            max_lambda_salience=max_affect_salience_lambda,
            max_boost_ratio=max_affect_salience_boost_ratio,
            min_age_gap_seconds=min_affect_salience_age_gap_seconds,
        )
        _append(checks, "affect_salience:evidence",
                affect_salience_summary["pass"],
                (
                    f"{affect_salience_summary['correct']}/{affect_salience_summary['total_checks']} checks, "
                    f"boost ratio {affect_salience_summary['max_boost_ratio']}"
                )
                if affect_salience_summary["pass"]
                else "; ".join(affect_salience_summary["failures"][:8]),
                **affect_salience_summary)
    scratchpad_invariant = {}
    if require_scratchpad_invariant:
        scratchpad_invariant, scratchpad_error = _load_json_report(out_dir / scratchpad_report)
        _append(checks, "scratchpad:valid_json",
                not scratchpad_error,
                "valid" if not scratchpad_error else scratchpad_error)
        scratchpad_summary = _scratchpad_summary(
            scratchpad_invariant,
            min_cases=min_scratchpad_cases,
        )
        _append(checks, "scratchpad:evidence",
                scratchpad_summary["pass"],
                (
                    f"{scratchpad_summary['correct']}/{scratchpad_summary['total_checks']} checks, "
                    f"proof links {scratchpad_summary['proof_link_checks']}"
                )
                if scratchpad_summary["pass"]
                else "; ".join(scratchpad_summary["failures"][:8]),
                **scratchpad_summary)
    region_routing_invariant = {}
    if require_region_routing_invariant:
        region_routing_invariant, region_routing_error = _load_json_report(
            out_dir / region_routing_report
        )
        _append(checks, "region_routing:valid_json",
                not region_routing_error,
                "valid" if not region_routing_error else region_routing_error)
        region_routing_summary = _region_routing_summary(
            region_routing_invariant,
            min_cases=min_region_routing_cases,
        )
        _append(checks, "region_routing:evidence",
                region_routing_summary["pass"],
                (
                    f"{region_routing_summary['correct']}/{region_routing_summary['total_checks']} checks, "
                    f"proof links {region_routing_summary['proof_link_checks']}"
                )
                if region_routing_summary["pass"]
                else "; ".join(region_routing_summary["failures"][:8]),
                **region_routing_summary)
    reflex_recall_invariant = {}
    if require_reflex_recall_invariant:
        reflex_recall_invariant, reflex_recall_error = _load_json_report(
            out_dir / reflex_recall_report
        )
        _append(checks, "reflex_recall:valid_json",
                not reflex_recall_error,
                "valid" if not reflex_recall_error else reflex_recall_error)
        reflex_recall_summary = _reflex_recall_summary(
            reflex_recall_invariant,
            min_cases=min_reflex_recall_cases,
            max_p95_latency_ms=max_reflex_recall_p95_ms,
        )
        _append(checks, "reflex_recall:evidence",
                reflex_recall_summary["pass"],
                (
                    f"{reflex_recall_summary['correct']}/{reflex_recall_summary['total_checks']} checks, "
                    f"p95 {reflex_recall_summary['p95_latency_ms']} ms"
                )
                if reflex_recall_summary["pass"]
                else "; ".join(reflex_recall_summary["failures"][:8]),
                **reflex_recall_summary)
    slice_invariant = {}
    if require_slice_invariant:
        slice_invariant, slice_error = _load_json_report(out_dir / slice_invariant_report)
        _append(checks, "slice_invariant:valid_json",
                not slice_error,
                "valid" if not slice_error else slice_error)
        _append(checks, "slice_invariant:pass",
                bool(slice_invariant.get("pass")),
                str(slice_invariant.get("pass", "<missing>")))
        slice_summary = _slice_invariant_summary(
            slice_invariant,
            required_datasets=required_datasets,
            expected_system=integrity_system,
            min_draws=min_slice_invariant_draws,
            min_subset=min_slice_invariant_subset,
        )
        _append(checks, "slice_invariant:evidence",
                slice_summary["pass"],
                "covered: " + ", ".join(slice_summary["covered"])
                if slice_summary["pass"]
                else "; ".join(slice_summary["failures"][:8]),
                **slice_summary)
    smqe_planner = {}
    if require_smqe_planner_invariant:
        smqe_planner, planner_error = _load_json_report(out_dir / smqe_planner_report)
        _append(checks, "smqe_planner:valid_json",
                not planner_error,
                "valid" if not planner_error else planner_error)
        planner_summary = _smqe_planner_summary(
            smqe_planner,
            min_cases=min_smqe_planner_cases,
            max_p95_latency_ms=max_smqe_planner_p95_ms,
        )
        _append(checks, "smqe_planner:evidence",
                planner_summary["pass"],
                f"{planner_summary['correct']}/{planner_summary['total_checks']} planner checks, "
                f"p95 {planner_summary['p95_latency_ms']} ms"
                if planner_summary["pass"]
                else "; ".join(planner_summary["failures"][:8]),
                **planner_summary)
    smqe_synthetic = {}
    if require_smqe_synthetic_invariant:
        smqe_synthetic, smqe_error = _load_json_report(out_dir / smqe_synthetic_report)
        _append(checks, "smqe_synthetic:valid_json",
                not smqe_error,
                "valid" if not smqe_error else smqe_error)
        smqe_summary = _smqe_synthetic_summary(
            smqe_synthetic,
            min_cases=min_smqe_synthetic_cases,
            max_avg_proof_tokens=max_smqe_synthetic_avg_proof_tokens,
        )
        _append(checks, "smqe_synthetic:evidence",
                smqe_summary["pass"],
                f"{smqe_summary['correct']}/{smqe_summary['cases']} cases, "
                f"avg proof {smqe_summary['avg_proof_tokens']}"
                if smqe_summary["pass"]
                else "; ".join(smqe_summary["failures"][:8]),
                **smqe_summary)
    smqe_claim_coverage = {}
    if require_smqe_claim_coverage:
        smqe_claim_coverage, claimcov_error = _load_json_report(out_dir / smqe_claim_coverage_report)
        _append(checks, "smqe_claim_coverage:valid_json",
                not claimcov_error,
                "valid" if not claimcov_error else claimcov_error)
        claimcov_summary = _smqe_claim_coverage_summary(
            smqe_claim_coverage,
            min_cases=min_smqe_claim_coverage_cases,
            min_claim_backend_rate=min_smqe_claim_backend_rate,
            max_avg_proof_tokens=max_smqe_claim_avg_proof_tokens,
        )
        _append(checks, "smqe_claim_coverage:evidence",
                claimcov_summary["pass"],
                f"{claimcov_summary['claim_backend_correct']}/{claimcov_summary['cases']} claim-backed, "
                f"rate {claimcov_summary['claim_backend_rate']}"
                if claimcov_summary["pass"]
                else "; ".join(claimcov_summary["failures"][:8]),
                **claimcov_summary)
    smqe_fullpath = {}
    if require_smqe_fullpath_invariant:
        smqe_fullpath, fullpath_error = _load_json_report(out_dir / smqe_fullpath_report)
        _append(checks, "smqe_fullpath:valid_json",
                not fullpath_error,
                "valid" if not fullpath_error else fullpath_error)
        fullpath_summary = _smqe_fullpath_summary(
            smqe_fullpath,
            min_cases=min_smqe_fullpath_cases,
            max_avg_proof_tokens=max_smqe_fullpath_avg_proof_tokens,
            max_avg_context_tokens=max_smqe_fullpath_avg_context_tokens,
            max_p95_latency_ms=max_smqe_fullpath_p95_ms,
        )
        _append(checks, "smqe_fullpath:evidence",
                fullpath_summary["pass"],
                f"{fullpath_summary['verified']}/{fullpath_summary['cases']} verified full-path, "
                f"reader_calls {fullpath_summary['reader_calls']}, "
                f"proof links {fullpath_summary['proof_link_checks']}, "
                f"claim {fullpath_summary['claim_backend_correct']}, "
                f"avg context {fullpath_summary['avg_context_tokens']}, "
                f"p95 {fullpath_summary['p95_latency_ms']} ms"
                if fullpath_summary["pass"]
                else "; ".join(fullpath_summary["failures"][:8]),
                **fullpath_summary)
    smqe_paraphrase = {}
    if require_smqe_paraphrase_invariant:
        smqe_paraphrase, paraphrase_error = _load_json_report(out_dir / smqe_paraphrase_report)
        _append(checks, "smqe_paraphrase:valid_json",
                not paraphrase_error,
                "valid" if not paraphrase_error else paraphrase_error)
        paraphrase_summary = _smqe_paraphrase_summary(
            smqe_paraphrase,
            min_cases=min_smqe_paraphrase_cases,
            max_avg_proof_tokens=max_smqe_paraphrase_avg_proof_tokens,
        )
        _append(checks, "smqe_paraphrase:evidence",
                paraphrase_summary["pass"],
                f"{paraphrase_summary['correct']}/{paraphrase_summary['cases']} cases, "
                f"record {paraphrase_summary['record_backend_correct']}, "
                f"claim {paraphrase_summary['claim_backend_correct']}"
                if paraphrase_summary["pass"]
                else "; ".join(paraphrase_summary["failures"][:8]),
                **paraphrase_summary)
    smqe_conflict = {}
    if require_smqe_conflict_invariant:
        smqe_conflict, conflict_error = _load_json_report(out_dir / smqe_conflict_report)
        _append(checks, "smqe_conflict:valid_json",
                not conflict_error,
                "valid" if not conflict_error else conflict_error)
        conflict_summary = _smqe_conflict_summary(
            smqe_conflict,
            min_cases=min_smqe_conflict_cases,
            max_avg_proof_tokens=max_smqe_conflict_avg_proof_tokens,
        )
        _append(checks, "smqe_conflict:evidence",
                conflict_summary["pass"],
                f"{conflict_summary['correct']}/{conflict_summary['cases']} cases, "
                f"record {conflict_summary['record_backend_correct']}, "
                f"claim {conflict_summary['claim_backend_correct']}"
                if conflict_summary["pass"]
                else "; ".join(conflict_summary["failures"][:8]),
                **conflict_summary)
    smqe_composition = {}
    if require_smqe_composition_invariant:
        smqe_composition, composition_error = _load_json_report(out_dir / smqe_composition_report)
        _append(checks, "smqe_composition:valid_json",
                not composition_error,
                "valid" if not composition_error else composition_error)
        composition_summary = _smqe_composition_summary(
            smqe_composition,
            min_cases=min_smqe_composition_cases,
            max_avg_proof_tokens=max_smqe_composition_avg_proof_tokens,
        )
        _append(checks, "smqe_composition:evidence",
                composition_summary["pass"],
                f"{composition_summary['correct']}/{composition_summary['cases']} composition cases, "
                f"record {composition_summary['record_backend_correct']}, "
                f"claim {composition_summary['claim_backend_correct']}"
                if composition_summary["pass"]
                else "; ".join(composition_summary["failures"][:8]),
                **composition_summary)
    smqe_relative_phrase = {}
    if require_smqe_relative_phrase_invariant:
        smqe_relative_phrase, relative_phrase_error = _load_json_report(out_dir / smqe_relative_phrase_report)
        _append(checks, "smqe_relative_phrase:valid_json",
                not relative_phrase_error,
                "valid" if not relative_phrase_error else relative_phrase_error)
        relative_phrase_summary = _smqe_relative_phrase_summary(
            smqe_relative_phrase,
            min_cases=min_smqe_relative_phrase_cases,
            max_avg_proof_tokens=max_smqe_relative_phrase_avg_proof_tokens,
        )
        _append(checks, "smqe_relative_phrase:evidence",
                relative_phrase_summary["pass"],
                f"{relative_phrase_summary['correct']}/{relative_phrase_summary['cases']} relative phrase cases, "
                f"record {relative_phrase_summary['record_backend_correct']}, "
                f"claim {relative_phrase_summary['claim_backend_correct']}"
                if relative_phrase_summary["pass"]
                else "; ".join(relative_phrase_summary["failures"][:8]),
                **relative_phrase_summary)
    smqe_temporal_window = {}
    if require_smqe_temporal_window_invariant:
        smqe_temporal_window, temporal_window_error = _load_json_report(out_dir / smqe_temporal_window_report)
        _append(checks, "smqe_temporal_window:valid_json",
                not temporal_window_error,
                "valid" if not temporal_window_error else temporal_window_error)
        temporal_window_summary = _smqe_temporal_window_summary(
            smqe_temporal_window,
            min_cases=min_smqe_temporal_window_cases,
            max_avg_proof_tokens=max_smqe_temporal_window_avg_proof_tokens,
        )
        _append(checks, "smqe_temporal_window:evidence",
                temporal_window_summary["pass"],
                f"{temporal_window_summary['correct']}/{temporal_window_summary['cases']} temporal window cases, "
                f"record {temporal_window_summary['record_backend_correct']}, "
                f"claim {temporal_window_summary['claim_backend_correct']}"
                if temporal_window_summary["pass"]
                else "; ".join(temporal_window_summary["failures"][:8]),
                **temporal_window_summary)
    smqe_attribution = {}
    if require_smqe_attribution_invariant:
        smqe_attribution, attribution_error = _load_json_report(out_dir / smqe_attribution_report)
        _append(checks, "smqe_attribution:valid_json",
                not attribution_error,
                "valid" if not attribution_error else attribution_error)
        attribution_summary = _smqe_attribution_summary(
            smqe_attribution,
            min_cases=min_smqe_attribution_cases,
            max_avg_proof_tokens=max_smqe_attribution_avg_proof_tokens,
        )
        _append(checks, "smqe_attribution:evidence",
                attribution_summary["pass"],
                f"{attribution_summary['correct']}/{attribution_summary['cases']} attribution cases, "
                f"record {attribution_summary['record_backend_correct']}, "
                f"claim {attribution_summary['claim_backend_correct']}"
                if attribution_summary["pass"]
                else "; ".join(attribution_summary["failures"][:8]),
                **attribution_summary)
    smqe_abstention = {}
    if require_smqe_abstention_invariant:
        smqe_abstention, abstention_error = _load_json_report(out_dir / smqe_abstention_report)
        _append(checks, "smqe_abstention:valid_json",
                not abstention_error,
                "valid" if not abstention_error else abstention_error)
        abstention_summary = _smqe_abstention_summary(
            smqe_abstention,
            min_cases=min_smqe_abstention_cases,
        )
        _append(checks, "smqe_abstention:evidence",
                abstention_summary["pass"],
                f"{abstention_summary['abstained']}/{abstention_summary['cases']} cases abstained, "
                f"record {abstention_summary['record_only_abstained']}, "
                f"claim {abstention_summary['claims_present_abstained']}"
                if abstention_summary["pass"]
                else "; ".join(abstention_summary["failures"][:8]),
                **abstention_summary)
    smqe_scope = {}
    if require_smqe_scope_invariant:
        smqe_scope, scope_error = _load_json_report(out_dir / smqe_scope_report)
        _append(checks, "smqe_scope:valid_json",
                not scope_error,
                "valid" if not scope_error else scope_error)
        scope_summary = _smqe_scope_summary(
            smqe_scope,
            min_cases=min_smqe_scope_cases,
            max_avg_proof_tokens=max_smqe_scope_avg_proof_tokens,
        )
        _append(checks, "smqe_scope:evidence",
                scope_summary["pass"],
                f"{scope_summary['correct']}/{scope_summary['total_checks']} scoped checks, "
                f"record {scope_summary['record_backend_correct']}, "
                f"claim {scope_summary['claim_backend_correct']}"
                if scope_summary["pass"]
                else "; ".join(scope_summary["failures"][:8]),
                **scope_summary)
    smqe_subscope = {}
    if require_smqe_subscope_invariant:
        smqe_subscope, subscope_error = _load_json_report(out_dir / smqe_subscope_report)
        _append(checks, "smqe_subscope:valid_json",
                not subscope_error,
                "valid" if not subscope_error else subscope_error)
        subscope_summary = _smqe_subscope_summary(
            smqe_subscope,
            min_cases=min_smqe_subscope_cases,
            max_avg_proof_tokens=max_smqe_subscope_avg_proof_tokens,
        )
        _append(checks, "smqe_subscope:evidence",
                subscope_summary["pass"],
                f"{subscope_summary['correct']}/{subscope_summary['total_checks']} sub-scope checks, "
                f"record {subscope_summary['record_backend_correct']}, "
                f"claim {subscope_summary['claim_backend_correct']}"
                if subscope_summary["pass"]
                else "; ".join(subscope_summary["failures"][:8]),
                **subscope_summary)
    smqe_time = {}
    if require_smqe_time_invariant:
        smqe_time, time_error = _load_json_report(out_dir / smqe_time_report)
        _append(checks, "smqe_time:valid_json",
                not time_error,
                "valid" if not time_error else time_error)
        time_summary = _smqe_time_summary(
            smqe_time,
            min_cases=min_smqe_time_cases,
            max_avg_proof_tokens=max_smqe_time_avg_proof_tokens,
        )
        _append(checks, "smqe_time:evidence",
                time_summary["pass"],
                f"{time_summary['correct']}/{time_summary['total_checks']} as-of checks, "
                f"record {time_summary['record_backend_correct']}, "
                f"claim {time_summary['claim_backend_correct']}"
                if time_summary["pass"]
                else "; ".join(time_summary["failures"][:8]),
                **time_summary)
    smqe_invalidation = {}
    if require_smqe_invalidation_invariant:
        smqe_invalidation, invalidation_error = _load_json_report(out_dir / smqe_invalidation_report)
        _append(checks, "smqe_invalidation:valid_json",
                not invalidation_error,
                "valid" if not invalidation_error else invalidation_error)
        invalidation_summary = _smqe_invalidation_summary(
            smqe_invalidation,
            min_cases=min_smqe_invalidation_cases,
            max_avg_proof_tokens=max_smqe_invalidation_avg_proof_tokens,
        )
        _append(checks, "smqe_invalidation:evidence",
                invalidation_summary["pass"],
                f"{invalidation_summary['correct']}/{invalidation_summary['total_checks']} invalidation checks, "
                f"record {invalidation_summary['record_backend_correct']}, "
                f"claim {invalidation_summary['claim_backend_correct']}"
                if invalidation_summary["pass"]
                else "; ".join(invalidation_summary["failures"][:8]),
                **invalidation_summary)
    abstention_calibration = {}
    abstention_v2_enabled = _truthy(env.get("ABSTENTION_V2"))
    if require_abstention_calibration and abstention_v2_enabled:
        abstention_calibration, abstention_calibration_error = _load_json_report(
            out_dir / abstention_calibration_report
        )
        _append(checks, "abstention_calibration:valid_json",
                not abstention_calibration_error,
                "valid" if not abstention_calibration_error else abstention_calibration_error)
        _append(checks, "abstention_calibration:ok",
                bool(abstention_calibration.get("ok")),
                str(abstention_calibration.get("ok", "<missing>")))
        _append(checks, "abstention_calibration:method",
                abstention_calibration.get("method") == "abstention_v2_tau",
                str(abstention_calibration.get("method", "<missing>")))
        _append(checks, "abstention_calibration:split",
                abstention_calibration.get("split") == "dev",
                f"{abstention_calibration.get('split', '<missing>')} (expected dev)")
        _append(checks, "abstention_calibration:system",
                abstention_calibration.get("system") == abstention_calibration_system,
                f"{abstention_calibration.get('system', '<missing>')} "
                f"(expected {abstention_calibration_system})")
        n_cal = _as_int(abstention_calibration.get("n", 0))
        _append(checks, "abstention_calibration:samples",
                n_cal >= min_abstention_calibration_samples,
                f"{n_cal} (required >= {min_abstention_calibration_samples})")
        target = _as_float(abstention_calibration.get("target", 0.0))
        _append(checks, "abstention_calibration:target_precision",
                target >= min_abstention_precision_target,
                f"{target:.3f} (required >= {min_abstention_precision_target:.3f})")
        precision_at_tau = _as_float(abstention_calibration.get("precision_at_tau", 0.0))
        coverage_at_tau = _as_float(abstention_calibration.get("coverage_at_tau", 0.0))
        _append(checks, "abstention_calibration:precision_at_tau",
                precision_at_tau >= target and target > 0.0,
                f"{precision_at_tau:.3f} (target {target:.3f})")
        _append(checks, "abstention_calibration:nonzero_coverage",
                coverage_at_tau > 0.0,
                f"{coverage_at_tau:.3f}")
        tau = abstention_calibration.get("tau")
        manifest_tau_raw = str(env.get("ABSTENTION_V2_TAU", "") or "").strip()
        try:
            tau_value = float(tau)
            manifest_tau_value = float(manifest_tau_raw)
            tau_matches = abs(tau_value - manifest_tau_value) <= 1e-12
        except (TypeError, ValueError):
            tau_matches = False
        _append(checks, "abstention_calibration:tau_applied",
                tau_matches,
                f"report={tau!r}, manifest={manifest_tau_raw or '<unset>'}")
        fp = abstention_calibration.get("log_fingerprint")
        _append(checks, "abstention_calibration:log_fingerprint_present",
                isinstance(fp, dict) and bool(fp.get("combined_sha256")),
                _fingerprint_detail(fp) if isinstance(fp, dict) else "<missing>")
    _append(checks, "logs:nonempty", bool(rows), f"{len(rows)} rows")

    errors = [r for r in rows if r.get("error")]
    _append(checks, "logs:no_error_rows", not errors, f"{len(errors)} error rows")

    legacy_policy_rows = _legacy_policy_rows(rows)
    _append(
        checks,
        "smqe:notes_clean",
        not legacy_policy_rows,
        "no legacy structured-recall policies"
        if not legacy_policy_rows
        else "; ".join(legacy_policy_rows[:8]),
        count=len(legacy_policy_rows),
    )
    smqe_log_policy = {}
    if require_smqe_log_policy:
        smqe_log_policy = _smqe_log_policy_summary(
            rows,
            system=integrity_system,
            min_structured_rate=min_smqe_log_structured_rate,
            min_claim_backend_rate=min_smqe_log_claim_backend_rate,
        )
        _append(
            checks,
            "smqe:log_policy_shape",
            smqe_log_policy["pass"],
            (
                f"{smqe_log_policy['structured']}/{smqe_log_policy['rows']} structured, "
                f"claim rate {smqe_log_policy['claim_backend_rate']:.3f}"
            )
            if smqe_log_policy["pass"]
            else "; ".join(smqe_log_policy["failures"][:8]),
            **smqe_log_policy,
        )
    region_telemetry = _region_telemetry_summary(rows, system=integrity_system)
    _append(
        checks,
        "region:telemetry",
        region_telemetry["pass"],
        (
            f"{region_telemetry['hint_rows']}/{region_telemetry['rows']} rows with hints; "
            f"{region_telemetry['total_hints']} hints"
            if region_telemetry["pass"]
            else "; ".join(region_telemetry["failures"][:8])
        ),
        **region_telemetry,
    )

    manifest_contract = _manifest_log_contract(rows, manifest, required_systems)
    missing_contract = manifest_contract["missing"]
    extra_contract = manifest_contract["extra"]
    _append(checks, "logs:match_manifest_sample_rows",
            bool(manifest_sample_rows) and not missing_contract and not extra_contract,
            f"expected={manifest_contract['expected']}, actual={manifest_contract['actual']}, "
            f"missing={len(missing_contract)}, extra={len(extra_contract)}")

    split_bad = [r for r in rows if split in ("dev", "test") and split_of(str(r.get("sample_id", ""))) != split]
    _append(checks, "logs:held_out_split", not split_bad,
            f"{len(split_bad)} rows outside {split} split")

    missing_systems = [s for s in required_systems if s not in systems_present]
    _append(checks, "systems:required_present", not missing_systems,
            "missing: " + ", ".join(missing_systems) if missing_systems else "all present")

    datasets_by_system = {
        sysname: sorted({str(r.get("dataset", "")) for r in sys_rows})
        for sysname, sys_rows in by_sys.items()
    }
    for sysname in required_systems:
        sys_rows = by_sys.get(sysname, [])
        datasets = set(datasets_by_system.get(sysname, []))
        missing_ds = [d for d in required_datasets if d not in datasets]
        runs = sorted({int(r.get("run_idx", 0)) for r in sys_rows})
        unique_samples = _unique_sample_count(sys_rows)
        _append(checks, f"{sysname}:questions", unique_samples >= min_questions_per_system,
                f"{unique_samples} unique samples, {len(sys_rows)} rows "
                f"(required >= {min_questions_per_system} unique samples)")
        _append(checks, f"{sysname}:runs", len(runs) >= min_runs,
                f"{len(runs)} runs: {runs}")
        _append(checks, f"{sysname}:datasets", not missing_ds,
                "missing: " + ", ".join(missing_ds) if missing_ds else "all present")
        for dataset in required_datasets:
            ds_rows = [r for r in sys_rows if r.get("dataset") == dataset]
            ds_runs = sorted({int(r.get("run_idx", 0)) for r in ds_rows})
            ds_unique_samples = _unique_sample_count(ds_rows)
            _append(checks, f"{sysname}:{dataset}:questions",
                    ds_unique_samples >= min_questions_per_dataset,
                    f"{ds_unique_samples} unique samples, {len(ds_rows)} rows "
                    f"(required >= {min_questions_per_dataset} unique samples)")
            _append(checks, f"{sysname}:{dataset}:runs",
                    len(ds_runs) >= min_runs,
                    f"{len(ds_runs)} runs: {ds_runs}")
            for category in required_categories.get(dataset, []):
                cat_rows = [r for r in ds_rows if r.get("category") == category]
                cat_runs = sorted({int(r.get("run_idx", 0)) for r in cat_rows})
                cat_unique_samples = _unique_sample_count(cat_rows)
                _append(checks, f"{sysname}:{dataset}:{category}:questions",
                        cat_unique_samples >= min_category_questions,
                        f"{cat_unique_samples} unique samples, {len(cat_rows)} rows "
                        f"(required >= {min_category_questions})")
                _append(checks, f"{sysname}:{dataset}:{category}:runs",
                        len(cat_runs) >= min_runs,
                        f"{len(cat_runs)} runs: {cat_runs}")

    if require_competitor_health:
        systems_requiring_health = [
            s for s in health_required_systems
            if s in required_systems or s in baseline_systems
        ]
        for sysname in systems_requiring_health:
            sys_rows = [r for r in by_sys.get(sysname, []) if not r.get("error")]
            missing_health = [r for r in sys_rows if not _baseline_health(r)]
            bad_health = [
                r for r in sys_rows
                if _baseline_health(r)
                and str(_baseline_health(r).get("status", "")).lower() != "ok"
            ]
            bad_statuses = sorted({
                str(_baseline_health(r).get("status", "<missing>"))
                for r in bad_health
            })
            _append(checks, f"competitor_health:{sysname}",
                    bool(sys_rows) and not missing_health and not bad_health,
                    f"rows={len(sys_rows)}, missing={len(missing_health)}, "
                    f"bad={len(bad_health)}"
                    + (f", statuses={bad_statuses}" if bad_statuses else ""))

    headline_rows = [r for r in by_sys.get(headline_system, []) if not r.get("error")]
    evidence_strength: dict[str, dict] = {}
    for dataset in required_datasets:
        ds_rows = [r for r in headline_rows if r.get("dataset") == dataset]
        acc = _accuracy(ds_rows)
        _append(checks, f"{headline_system}:{dataset}:accuracy",
                bool(ds_rows) and acc >= min_dataset_accuracy,
                f"{acc * 100:.1f}% (required >= {min_dataset_accuracy * 100:.1f}%)",
                n=len(ds_rows), accuracy=acc)
        ds_ci = _sample_clustered_accuracy(ds_rows)
        evidence_strength[f"{dataset}|*"] = ds_ci
        _append(checks, f"evidence:{headline_system}:{dataset}:sample_clustered_accuracy_ci_width",
                ds_ci["n"] > 0 and ds_ci["wilson_width_pp"] <= max_dataset_accuracy_ci_width_pp,
                _ci_detail(ds_ci, max_dataset_accuracy_ci_width_pp),
                **ds_ci)
        for category in required_categories.get(dataset, []):
            cat_rows = [r for r in ds_rows if r.get("category") == category]
            cat_ci = _sample_clustered_accuracy(cat_rows)
            evidence_strength[f"{dataset}|{category}"] = cat_ci
            _append(checks, f"evidence:{headline_system}:{dataset}/{category}:sample_clustered_accuracy_ci_width",
                    cat_ci["n"] > 0
                    and cat_ci["wilson_width_pp"] <= max_category_accuracy_ci_width_pp,
                    _ci_detail(cat_ci, max_category_accuracy_ci_width_pp),
                    **cat_ci)

    operating = {
        headline_system: _operating_metrics(headline_rows),
    }
    if token_efficiency_baseline:
        operating[token_efficiency_baseline] = _operating_metrics(
            [r for r in by_sys.get(token_efficiency_baseline, []) if not r.get("error")]
        )
    head_ops = operating[headline_system]
    qmed = head_ops.get("query_tokens_median")
    search_p95 = head_ops.get("search_p95_ms")
    e2e_p50 = head_ops.get("e2e_p50_ms")
    _append(checks, f"operating:{headline_system}:query_tokens_median",
            qmed is not None and qmed <= max_query_tokens_median,
            f"{qmed if qmed is not None else 'n/a'} (allowed <= {max_query_tokens_median:.0f})")
    _append(checks, f"operating:{headline_system}:search_p95_ms",
            search_p95 is not None and search_p95 <= max_search_p95_ms,
            f"{search_p95 if search_p95 is not None else 'n/a'} (allowed <= {max_search_p95_ms:.1f})")
    _append(checks, f"operating:{headline_system}:e2e_p50_ms",
            e2e_p50 is not None and e2e_p50 <= max_e2e_p50_ms,
            f"{e2e_p50 if e2e_p50 is not None else 'n/a'} (allowed <= {max_e2e_p50_ms:.1f})")
    if token_efficiency_baseline:
        base_qmed = operating[token_efficiency_baseline].get("query_tokens_median")
        if qmed == 0 and base_qmed and base_qmed > 0:
            token_ratio = float("inf")
        elif qmed and base_qmed is not None:
            token_ratio = base_qmed / qmed
        else:
            token_ratio = None
        _append(checks, f"operating:{headline_system}:token_efficiency_vs:{token_efficiency_baseline}",
                token_ratio is not None and token_ratio >= min_token_efficiency_ratio,
                f"{token_ratio if token_ratio is not None else 'n/a'}x "
                f"(required >= {min_token_efficiency_ratio:.1f}x)")

    age_slope = _recall_age_slope(headline_rows)
    slope = age_slope.get("slope_per_year")
    _append(checks, f"operating:{headline_system}:age_slope_samples",
            age_slope["n"] >= min_age_slope_points and age_slope["distinct_ages"] >= 2,
            f"n={age_slope['n']}, distinct_ages={age_slope['distinct_ages']} "
            f"(required n >= {min_age_slope_points}, distinct >= 2)")
    _append(checks, f"operating:{headline_system}:age_flatness",
            slope is not None and abs(float(slope)) <= max_abs_recall_slope_per_year,
            f"{slope if slope is not None else 'n/a'} per year "
            f"(allowed abs <= {max_abs_recall_slope_per_year:.3f})")

    paired: dict[str, dict] = {}
    required_category_pairs = [
        (dataset, category)
        for dataset in required_datasets
        for category in required_categories.get(dataset, [])
    ]
    observed_category_pairs = sorted({
        (str(r.get("dataset")), str(r.get("category")))
        for r in rows
        if r.get("dataset") in required_datasets
    })
    categories = required_category_pairs or observed_category_pairs
    for baseline in baseline_systems:
        stats = _paired_stats(rows, headline_system, baseline)
        clustered = _sample_clustered_paired_stats(rows, headline_system, baseline)
        headline_ci = _sample_clustered_accuracy([
            r for r in rows
            if r.get("system") == headline_system and r.get("dataset") in required_datasets
        ])
        baseline_ci = _sample_clustered_accuracy([
            r for r in rows
            if r.get("system") == baseline and r.get("dataset") in required_datasets
        ])
        paired[baseline] = {
            "overall": stats,
            "sample_clustered": clustered,
            "sample_clustered_ci": {
                "headline": headline_ci,
                "baseline": baseline_ci,
            },
            "categories": {},
        }
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:paired",
                stats["n"] > 0 and not stats["unpaired_headline"] and not stats["unpaired_baseline"],
                f"paired_n={stats['n']}, unpaired={stats['unpaired_headline'] + stats['unpaired_baseline']}")
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:delta",
                stats["delta_pp"] >= min_overall_delta_pp,
                f"{stats['delta_pp']:.1f}pp (required >= {min_overall_delta_pp:.1f}pp)")
        p = stats["p_mcnemar"]
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:significance",
                p is not None and p < alpha,
                f"p={p if p is not None else 'n/a'} (required < {alpha})")
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:sample_clustered_paired",
                clustered["n"] >= min_clustered_paired
                and not clustered["unpaired_headline"]
                and not clustered["unpaired_baseline"],
                f"sample_n={clustered['n']} (required >= {min_clustered_paired}), "
                f"unpaired={clustered['unpaired_headline'] + clustered['unpaired_baseline']}")
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:sample_clustered_delta",
                clustered["delta_pp"] >= min_overall_delta_pp,
                f"{clustered['delta_pp']:.1f}pp (required >= {min_overall_delta_pp:.1f}pp)")
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:sample_clustered_ci_clear",
                not require_ci_clear_dominance
                or (
                    headline_ci["n"] >= min_clustered_paired
                    and baseline_ci["n"] >= min_clustered_paired
                    and headline_ci["wilson_low"] > baseline_ci["wilson_high"]
                ),
                "not required" if not require_ci_clear_dominance else (
                    f"headline {_ci_detail(headline_ci, 100.0)}; "
                    f"baseline {_ci_detail(baseline_ci, 100.0)}; "
                    "need headline lower CI > baseline upper CI"
                ))
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:sample_clustered_discordants",
                clustered["discordant"] >= min_clustered_discordant_samples,
                f"{clustered['discordant']} discordant samples "
                f"(required >= {min_clustered_discordant_samples})")
        cp = clustered["p_mcnemar"]
        _append(checks, f"dominance:{headline_system}:vs:{baseline}:sample_clustered_significance",
                clustered["discordant"] >= min_clustered_discordant_samples
                and cp is not None and cp < alpha,
                f"p={cp if cp is not None else 'n/a'}, "
                f"discordant={clustered['discordant']} (required p < {alpha})")

        if require_category_wins:
            for dataset, category in categories:
                cat_stats = _paired_stats(rows, headline_system, baseline,
                                          dataset=str(dataset), category=str(category))
                cat_clustered = _sample_clustered_paired_stats(
                    rows,
                    headline_system,
                    baseline,
                    dataset=str(dataset),
                    category=str(category),
                )
                cat_headline_ci = _sample_clustered_accuracy([
                    r for r in rows
                    if r.get("system") == headline_system
                    and r.get("dataset") == dataset
                    and r.get("category") == category
                ])
                cat_baseline_ci = _sample_clustered_accuracy([
                    r for r in rows
                    if r.get("system") == baseline
                    and r.get("dataset") == dataset
                    and r.get("category") == category
                ])
                paired[baseline]["categories"][f"{dataset}|{category}"] = {
                    "row": cat_stats,
                    "sample_clustered": cat_clustered,
                    "sample_clustered_ci": {
                        "headline": cat_headline_ci,
                        "baseline": cat_baseline_ci,
                    },
                }
                _append(checks, f"dominance:{headline_system}:vs:{baseline}:{dataset}/{category}",
                        cat_stats["n"] > 0 and cat_stats["delta_pp"] >= min_category_delta_pp,
                        f"n={cat_stats['n']}, delta={cat_stats['delta_pp']:.1f}pp "
                        f"(required >= {min_category_delta_pp:.1f}pp)")
                _append(checks, f"dominance:{headline_system}:vs:{baseline}:{dataset}/{category}:sample_clustered_paired",
                        cat_clustered["n"] >= min_category_clustered_paired
                        and not cat_clustered["unpaired_headline"]
                        and not cat_clustered["unpaired_baseline"],
                        f"sample_n={cat_clustered['n']} "
                        f"(required >= {min_category_clustered_paired}), "
                        f"unpaired={cat_clustered['unpaired_headline'] + cat_clustered['unpaired_baseline']}")
                _append(checks, f"dominance:{headline_system}:vs:{baseline}:{dataset}/{category}:sample_clustered_delta",
                        cat_clustered["delta_pp"] >= min_category_delta_pp,
                        f"{cat_clustered['delta_pp']:.1f}pp "
                        f"(required >= {min_category_delta_pp:.1f}pp)")
                _append(checks, f"dominance:{headline_system}:vs:{baseline}:{dataset}/{category}:sample_clustered_ci_clear",
                        not require_ci_clear_dominance
                        or (
                            cat_headline_ci["n"] >= min_category_clustered_paired
                            and cat_baseline_ci["n"] >= min_category_clustered_paired
                            and cat_headline_ci["wilson_low"] > cat_baseline_ci["wilson_high"]
                        ),
                        "not required" if not require_ci_clear_dominance else (
                            f"headline {_ci_detail(cat_headline_ci, 100.0)}; "
                            f"baseline {_ci_detail(cat_baseline_ci, 100.0)}; "
                            "need headline lower CI > baseline upper CI"
                        ))
                _append(checks, f"dominance:{headline_system}:vs:{baseline}:{dataset}/{category}:sample_clustered_discordants",
                        cat_clustered["discordant"] >= min_clustered_discordant_samples,
                        f"{cat_clustered['discordant']} discordant samples "
                        f"(required >= {min_clustered_discordant_samples})")
                cat_cp = cat_clustered["p_mcnemar"]
                _append(checks, f"dominance:{headline_system}:vs:{baseline}:{dataset}/{category}:sample_clustered_significance",
                        cat_clustered["discordant"] >= min_clustered_discordant_samples
                        and cat_cp is not None and cat_cp < alpha,
                        f"p={cat_cp if cat_cp is not None else 'n/a'}, "
                        f"discordant={cat_clustered['discordant']} (required p < {alpha})")

    integrity = agg.get("integrity", {}).get(integrity_system, {})
    integrity_n = int(integrity.get("n", 0) or 0)
    verified_accuracy = (
        float(integrity.get("verified_correct", 0) or 0) / integrity_n if integrity_n else 0.0
    )
    _append(checks, f"integrity:{integrity_system}:verify_step",
            bool(integrity.get("has_verify")) and integrity_n > 0,
            f"has_verify={integrity.get('has_verify')}, n={integrity_n}")
    _append(checks, f"integrity:{integrity_system}:verified_accuracy",
            verified_accuracy >= min_verified_accuracy,
            f"{verified_accuracy * 100:.1f}% (required >= {min_verified_accuracy * 100:.1f}%)")
    proof_support = _verified_proof_support_summary(rows, system=integrity_system)
    _append(
        checks,
        f"integrity:{integrity_system}:proof_support",
        proof_support["pass"],
        (
            f"{proof_support['supported_verified_rows']}/{proof_support['verified_rows']} "
            f"verified rows carry proof support"
            if proof_support["pass"]
            else "; ".join(proof_support["failures"][:8])
        ),
        **proof_support,
    )

    consolidation = consolidation_rollup(rows)
    for sysname in required_systems:
        con = consolidation.get(sysname, {})
        timeouts = int(con.get("extraction_timed_out", 0) or 0)
        deferred = int(con.get("extraction_deferred", 0) or 0)
        _append(checks, f"{sysname}:consolidation_timeouts",
                timeouts <= max_consolidation_timeouts,
                f"{timeouts} (allowed <= {max_consolidation_timeouts})")
        _append(checks, f"{sysname}:consolidation_deferred",
                deferred <= max_consolidation_deferred,
                f"{deferred} (allowed <= {max_consolidation_deferred})")

    snap_back, snap_error = _load_json_report(out_dir / "snap_back_audit.json")
    _append(checks, "snap_back:valid_json", not snap_error,
            "valid" if not snap_error else snap_error)
    snap_records = _as_int(snap_back.get("records_with_raw_blob", 0))
    snap_lossless = _as_int(snap_back.get("lossless_byte_identical", 0))
    snap_rate = _as_float(snap_back.get("rate", 0.0))
    raw_failures = snap_back.get("failures", []) or []
    snap_failures = raw_failures if isinstance(raw_failures, list) else [{"error": "malformed_failures"}]
    _append(checks, "snap_back:records",
            snap_records >= min_snap_back_records,
            f"{snap_records} (required >= {min_snap_back_records})")
    _append(checks, "snap_back:lossless",
            snap_records > 0 and snap_lossless == snap_records and snap_rate >= 1.0,
            f"{snap_lossless}/{snap_records}, rate={snap_rate:.6f}")
    _append(checks, "snap_back:no_failures", not snap_failures,
            f"{len(snap_failures)} failures")
    audited_content_hashes = set(_sha256_strings(snap_back.get("audited_content_hashes", [])))
    proof_content_hashes = set(proof_support.get("proof_content_hashes", []) or [])
    missing_proof_hashes = sorted(proof_content_hashes - audited_content_hashes)
    _append(
        checks,
        "snap_back:audited_hashes_present",
        bool(audited_content_hashes),
        f"{len(audited_content_hashes)} audited hash(es)",
    )
    _append(
        checks,
        "snap_back:covers_verified_proof_hashes",
        bool(proof_content_hashes) and not missing_proof_hashes,
        (
            f"{len(proof_content_hashes)} proof hash(es) covered"
            if proof_content_hashes and not missing_proof_hashes
            else f"missing={len(missing_proof_hashes)}, proof_hashes={len(proof_content_hashes)}"
        ),
        missing_proof_hashes=missing_proof_hashes[:20],
        proof_hash_count=len(proof_content_hashes),
        audited_hash_count=len(audited_content_hashes),
    )
    if manifest_data_dir:
        try:
            expected_data_dir = str(Path(manifest_data_dir).expanduser().resolve())
        except OSError:
            expected_data_dir = manifest_data_dir
        got_data_dir = str(snap_back.get("data_dir", "") or "")
        _append(checks, "snap_back:data_dir_matches_manifest",
                got_data_dir == expected_data_dir,
                f"{got_data_dir or '<unset>'} (expected {expected_data_dir})")

    baseline_reproduction = {}
    if require_baseline_reproduction:
        baseline_reproduction, baseline_reproduction_error = _load_json_report(
            out_dir / baseline_reproduction_report
        )
        _append(checks, "baseline_reproduction:valid_json", not baseline_reproduction_error,
                "valid" if not baseline_reproduction_error else baseline_reproduction_error)
        _append(checks, "baseline_reproduction:status",
                baseline_reproduction.get("status") == "PASS",
                str(baseline_reproduction.get("status", "<missing>")))
        _append(checks, "baseline_reproduction:system",
                baseline_reproduction.get("system") == baseline_reproduction_system,
                f"{baseline_reproduction.get('system', '<missing>')} "
                f"(expected {baseline_reproduction_system})")
        _append(checks, "baseline_reproduction:dataset",
                baseline_reproduction.get("dataset") == baseline_reproduction_dataset,
                f"{baseline_reproduction.get('dataset', '<missing>')} "
                f"(expected {baseline_reproduction_dataset})")
        _append(checks, "baseline_reproduction:rows",
                _as_int(baseline_reproduction.get("total_n", 0)) > 0,
                f"total_n={baseline_reproduction.get('total_n', 0)}")
        comparison_failures = [
            cat for cat, item in (baseline_reproduction.get("comparisons", {}) or {}).items()
            if not isinstance(item, dict) or item.get("status") != "PASS"
        ]
        _append(checks, "baseline_reproduction:comparisons",
                not comparison_failures,
                "all PASS" if not comparison_failures else "failed: " + ", ".join(comparison_failures))
        baseline_has_fp, baseline_fp_matches, baseline_fp_detail = _matching_fingerprint(
            baseline_reproduction, current_fingerprint
        )
        _append(checks, "baseline_reproduction:log_fingerprint_present", baseline_has_fp,
                baseline_fp_detail)
        _append(checks, "baseline_reproduction:log_fingerprint_matches", baseline_fp_matches,
                f"{baseline_fp_detail} (current {_fingerprint_detail(current_fingerprint)})")

    status = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    return {
        "status": status,
        "out_dir": str(out_dir),
        "log_fingerprint": current_fingerprint,
        "checks": checks,
        "failed_checks": [c for c in checks if not c["pass"]],
        "manifest": manifest,
        "scoreboard": scoreboard_report,
        "claim_scope": claim_scope,
        "holdout_audit": holdout_audit,
        "ablation_evidence": ablation_evidence,
        "affect_salience_invariant": affect_salience_invariant,
        "scratchpad_invariant": scratchpad_invariant,
        "region_routing_invariant": region_routing_invariant,
        "reflex_recall_invariant": reflex_recall_invariant,
        "smqe_planner_invariant": smqe_planner,
        "smqe_log_policy": smqe_log_policy,
        "region_telemetry": region_telemetry,
        "manifest_contract": {
            "sample_rows": len(manifest_contract["sample_rows"]),
            "run_indices": manifest_contract["run_indices"],
            "expected": manifest_contract["expected"],
            "actual": manifest_contract["actual"],
            "missing": manifest_contract["missing"][:20],
            "extra": manifest_contract["extra"][:20],
        },
        "systems": sorted(systems_present),
        "datasets_by_system": datasets_by_system,
        "integrity": integrity,
        "proof_support": proof_support,
        "verified_accuracy": verified_accuracy,
        "consolidation": consolidation,
        "snap_back": snap_back,
        "operating": operating,
        "age_slope": age_slope,
        "evidence_strength": evidence_strength,
        "abstention_calibration": abstention_calibration,
        "baseline_reproduction": baseline_reproduction,
        "paired": paired,
        "thresholds": {
            "split": split,
            "min_runs": min_runs,
            "min_questions_per_system": min_questions_per_system,
            "min_questions_per_dataset_per_system": min_questions_per_dataset,
            "min_category_questions_per_system": min_category_questions,
            "required_categories_by_dataset": required_categories,
            "min_dataset_accuracy": min_dataset_accuracy,
            "min_overall_delta_pp": min_overall_delta_pp,
            "min_category_delta_pp": min_category_delta_pp,
            "alpha": alpha,
            "max_dataset_accuracy_ci_width_pp": max_dataset_accuracy_ci_width_pp,
            "max_category_accuracy_ci_width_pp": max_category_accuracy_ci_width_pp,
            "min_sample_clustered_paired_samples": min_clustered_paired,
            "min_category_sample_clustered_paired_samples": min_category_clustered_paired,
            "require_ci_clear_dominance": require_ci_clear_dominance,
            "min_clustered_discordant_samples": min_clustered_discordant_samples,
            "min_verified_accuracy": min_verified_accuracy,
            "max_consolidation_timeouts": max_consolidation_timeouts,
            "max_consolidation_deferred": max_consolidation_deferred,
            "min_snap_back_records": min_snap_back_records,
            "max_query_tokens_median": max_query_tokens_median,
            "max_search_p95_ms": max_search_p95_ms,
            "max_e2e_p50_ms": max_e2e_p50_ms,
            "token_efficiency_baseline": token_efficiency_baseline,
            "min_token_efficiency_ratio": min_token_efficiency_ratio,
            "max_abs_recall_slope_per_year": max_abs_recall_slope_per_year,
            "min_age_slope_points": min_age_slope_points,
            "require_baseline_reproduction": require_baseline_reproduction,
            "baseline_reproduction_report": baseline_reproduction_report,
            "baseline_reproduction_system": baseline_reproduction_system,
            "baseline_reproduction_dataset": baseline_reproduction_dataset,
            "require_category_wins": require_category_wins,
            "require_competitor_health": require_competitor_health,
            "health_required_systems": health_required_systems,
            "require_claim_scope": require_claim_scope,
            "claim_scope_report": claim_scope_report,
            "top_systems_for_sota": top_systems_for_sota,
            "min_external_evidence_n": min_external_evidence_n,
            "min_external_evidence_runs": min_external_evidence_runs,
            "require_abstention_calibration": require_abstention_calibration,
            "abstention_calibration_report": abstention_calibration_report,
            "abstention_calibration_system": abstention_calibration_system,
            "min_abstention_calibration_samples": min_abstention_calibration_samples,
            "min_abstention_precision_target": min_abstention_precision_target,
            "require_holdout_profile": require_holdout_profile,
            "require_holdout_audit": require_holdout_audit,
            "holdout_audit_report": holdout_audit_report,
            "min_holdout_audit_needles": min_holdout_audit_needles,
            "require_smqe_log_policy": require_smqe_log_policy,
            "min_smqe_log_structured_rate": min_smqe_log_structured_rate,
            "min_smqe_log_claim_backend_rate": min_smqe_log_claim_backend_rate,
            "require_ablation_evidence": require_ablation_evidence,
            "ablation_report": ablation_report,
            "ablation_split": ablation_split,
            "min_ablation_samples": min_ablation_samples,
            "min_metabolism_accuracy_delta_pp": min_metabolism_accuracy_delta_pp,
            "min_region_accuracy_delta_pp": min_region_accuracy_delta_pp,
            "min_affect_accuracy_delta_pp": min_affect_accuracy_delta_pp,
            "min_forgetting_cost_ratio": min_forgetting_cost_ratio,
            "max_forgetting_accuracy_regression_pp": max_forgetting_accuracy_regression_pp,
            "require_affect_salience_invariant": require_affect_salience_invariant,
            "affect_salience_report": affect_salience_report,
            "min_affect_salience_cases": min_affect_salience_cases,
            "max_affect_salience_lambda": max_affect_salience_lambda,
            "max_affect_salience_boost_ratio": max_affect_salience_boost_ratio,
            "min_affect_salience_age_gap_seconds": min_affect_salience_age_gap_seconds,
            "require_scratchpad_invariant": require_scratchpad_invariant,
            "scratchpad_report": scratchpad_report,
            "min_scratchpad_cases": min_scratchpad_cases,
            "require_region_routing_invariant": require_region_routing_invariant,
            "region_routing_report": region_routing_report,
            "min_region_routing_cases": min_region_routing_cases,
            "require_reflex_recall_invariant": require_reflex_recall_invariant,
            "reflex_recall_report": reflex_recall_report,
            "min_reflex_recall_cases": min_reflex_recall_cases,
            "max_reflex_recall_p95_ms": max_reflex_recall_p95_ms,
            "require_slice_invariant": require_slice_invariant,
            "slice_invariant_report": slice_invariant_report,
            "min_slice_invariant_draws": min_slice_invariant_draws,
            "min_slice_invariant_subset": min_slice_invariant_subset,
            "require_smqe_planner_invariant": require_smqe_planner_invariant,
            "smqe_planner_report": smqe_planner_report,
            "min_smqe_planner_cases": min_smqe_planner_cases,
            "max_smqe_planner_p95_ms": max_smqe_planner_p95_ms,
            "require_smqe_synthetic_invariant": require_smqe_synthetic_invariant,
            "smqe_synthetic_report": smqe_synthetic_report,
            "min_smqe_synthetic_cases": min_smqe_synthetic_cases,
            "max_smqe_synthetic_avg_proof_tokens": max_smqe_synthetic_avg_proof_tokens,
            "require_smqe_claim_coverage": require_smqe_claim_coverage,
            "smqe_claim_coverage_report": smqe_claim_coverage_report,
            "min_smqe_claim_coverage_cases": min_smqe_claim_coverage_cases,
            "min_smqe_claim_backend_rate": min_smqe_claim_backend_rate,
            "max_smqe_claim_avg_proof_tokens": max_smqe_claim_avg_proof_tokens,
            "require_smqe_fullpath_invariant": require_smqe_fullpath_invariant,
            "smqe_fullpath_report": smqe_fullpath_report,
            "min_smqe_fullpath_cases": min_smqe_fullpath_cases,
            "max_smqe_fullpath_avg_proof_tokens": max_smqe_fullpath_avg_proof_tokens,
            "max_smqe_fullpath_avg_context_tokens": max_smqe_fullpath_avg_context_tokens,
            "max_smqe_fullpath_p95_ms": max_smqe_fullpath_p95_ms,
            "require_smqe_paraphrase_invariant": require_smqe_paraphrase_invariant,
            "smqe_paraphrase_report": smqe_paraphrase_report,
            "min_smqe_paraphrase_cases": min_smqe_paraphrase_cases,
            "max_smqe_paraphrase_avg_proof_tokens": max_smqe_paraphrase_avg_proof_tokens,
            "require_smqe_conflict_invariant": require_smqe_conflict_invariant,
            "smqe_conflict_report": smqe_conflict_report,
            "min_smqe_conflict_cases": min_smqe_conflict_cases,
            "max_smqe_conflict_avg_proof_tokens": max_smqe_conflict_avg_proof_tokens,
            "require_smqe_composition_invariant": require_smqe_composition_invariant,
            "smqe_composition_report": smqe_composition_report,
            "min_smqe_composition_cases": min_smqe_composition_cases,
            "max_smqe_composition_avg_proof_tokens": max_smqe_composition_avg_proof_tokens,
            "require_smqe_relative_phrase_invariant": require_smqe_relative_phrase_invariant,
            "smqe_relative_phrase_report": smqe_relative_phrase_report,
            "min_smqe_relative_phrase_cases": min_smqe_relative_phrase_cases,
            "max_smqe_relative_phrase_avg_proof_tokens": max_smqe_relative_phrase_avg_proof_tokens,
            "require_smqe_temporal_window_invariant": require_smqe_temporal_window_invariant,
            "smqe_temporal_window_report": smqe_temporal_window_report,
            "min_smqe_temporal_window_cases": min_smqe_temporal_window_cases,
            "max_smqe_temporal_window_avg_proof_tokens": max_smqe_temporal_window_avg_proof_tokens,
            "require_smqe_attribution_invariant": require_smqe_attribution_invariant,
            "smqe_attribution_report": smqe_attribution_report,
            "min_smqe_attribution_cases": min_smqe_attribution_cases,
            "max_smqe_attribution_avg_proof_tokens": max_smqe_attribution_avg_proof_tokens,
            "require_smqe_abstention_invariant": require_smqe_abstention_invariant,
            "smqe_abstention_report": smqe_abstention_report,
            "min_smqe_abstention_cases": min_smqe_abstention_cases,
            "require_smqe_scope_invariant": require_smqe_scope_invariant,
            "smqe_scope_report": smqe_scope_report,
            "min_smqe_scope_cases": min_smqe_scope_cases,
            "max_smqe_scope_avg_proof_tokens": max_smqe_scope_avg_proof_tokens,
            "require_smqe_subscope_invariant": require_smqe_subscope_invariant,
            "smqe_subscope_report": smqe_subscope_report,
            "min_smqe_subscope_cases": min_smqe_subscope_cases,
            "max_smqe_subscope_avg_proof_tokens": max_smqe_subscope_avg_proof_tokens,
            "require_smqe_time_invariant": require_smqe_time_invariant,
            "smqe_time_report": smqe_time_report,
            "min_smqe_time_cases": min_smqe_time_cases,
            "max_smqe_time_avg_proof_tokens": max_smqe_time_avg_proof_tokens,
            "require_smqe_invalidation_invariant": require_smqe_invalidation_invariant,
            "smqe_invalidation_report": smqe_invalidation_report,
            "min_smqe_invalidation_cases": min_smqe_invalidation_cases,
            "max_smqe_invalidation_avg_proof_tokens": max_smqe_invalidation_avg_proof_tokens,
        },
    }


def render_markdown(report: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Public Release Gate", ""]
    lines.append(f"Status: **{report['status']}**")
    lines.append(f"Artifact directory: `{report['out_dir']}`")
    fp = report.get("log_fingerprint", {})
    if fp:
        lines.append(f"Log fingerprint: `{fp.get('combined_sha256', '')}` ({fp.get('file_count', 0)} files)")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | status | detail |")
    lines.append("|---|---|---|")
    for check in report["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        detail = str(check.get("detail", "")).replace("|", "\\|")
        lines.append(f"| {check['name']} | {status} | {detail} |")
    lines.append("")
    lines.append("## Evidence Strength")
    lines.append("")
    lines.append("| slice | sample n | accuracy | Wilson low | Wilson high | width pp |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, stats in sorted((report.get("evidence_strength", {}) or {}).items()):
        lines.append(
            f"| {key} | {stats.get('n', 0)} | "
            f"{stats.get('accuracy', 0.0) * 100:.1f}% | "
            f"{stats.get('wilson_low', 0.0) * 100:.1f}% | "
            f"{stats.get('wilson_high', 0.0) * 100:.1f}% | "
            f"{stats.get('wilson_width_pp', 0.0):.1f} |"
        )
    lines.append("")
    lines.append("## Paired Dominance")
    lines.append("")
    lines.append("| baseline | paired n | delta pp | McNemar p | sample n | sample discordants | sample McNemar p | CI-clear |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for baseline, item in sorted(report.get("paired", {}).items()):
        stats = item.get("overall", {})
        clustered = item.get("sample_clustered", {})
        ci = item.get("sample_clustered_ci", {})
        h_ci = ci.get("headline", {}) if isinstance(ci, dict) else {}
        b_ci = ci.get("baseline", {}) if isinstance(ci, dict) else {}
        ci_clear = (
            h_ci.get("wilson_low") is not None
            and b_ci.get("wilson_high") is not None
            and h_ci.get("wilson_low", 0.0) > b_ci.get("wilson_high", 1.0)
        )
        p = stats.get("p_mcnemar")
        p_text = "-" if p is None else f"{p:.4f}"
        cp = clustered.get("p_mcnemar")
        cp_text = "-" if cp is None else f"{cp:.4f}"
        lines.append(f"| {baseline} | {stats.get('n', 0)} | {stats.get('delta_pp', 0.0):.1f} | "
                     f"{p_text} | {clustered.get('n', 0)} | "
                     f"{clustered.get('discordant', 0)} | {cp_text} | "
                     f"{'yes' if ci_clear else 'no'} |")
    lines.append("")
    lines.append("## Category Clustered Dominance")
    lines.append("")
    lines.append("| baseline | category | row n | row delta pp | sample n | sample delta pp | sample discordants | sample McNemar p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for baseline, item in sorted(report.get("paired", {}).items()):
        for category, stats_bundle in sorted((item.get("categories", {}) or {}).items()):
            if "row" in stats_bundle or "sample_clustered" in stats_bundle:
                row_stats = stats_bundle.get("row", {}) or {}
                sample_stats = stats_bundle.get("sample_clustered", {}) or {}
            else:
                row_stats = stats_bundle or {}
                sample_stats = {}
            cp = sample_stats.get("p_mcnemar")
            cp_text = "-" if cp is None else f"{cp:.4f}"
            lines.append(
                f"| {baseline} | {category} | {row_stats.get('n', 0)} | "
                f"{row_stats.get('delta_pp', 0.0):.1f} | {sample_stats.get('n', 0)} | "
                f"{sample_stats.get('delta_pp', 0.0):.1f} | "
                f"{sample_stats.get('discordant', 0)} | {cp_text} |"
            )
    lines.append("")
    lines.append("## Operating Point")
    lines.append("")
    lines.append("| system | n | median query tokens | search p95 ms | e2e p50 ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for sysname, ops in sorted(report.get("operating", {}).items()):
        q = ops.get("query_tokens_median")
        sp = ops.get("search_p95_ms")
        e2e = ops.get("e2e_p50_ms")
        q_s = "-" if q is None else f"{q:.0f}"
        sp_s = "-" if sp is None else f"{sp:.1f}"
        e2e_s = "-" if e2e is None else f"{e2e:.1f}"
        lines.append(f"| {sysname} | {ops.get('n', 0)} | {q_s} | {sp_s} | {e2e_s} |")
    slope = report.get("age_slope", {}).get("slope_per_year")
    slope_s = "n/a" if slope is None else f"{float(slope):+.4f}"
    lines.append("")
    lines.append(f"Recall-vs-age slope for headline row: `{slope_s}` per year.")
    lines.append("")
    lines.append("## Claim Scope")
    lines.append("")
    claim = report.get("claim_scope", {}) or {}
    if claim:
        scope = claim.get("public_claim_scope", claim.get("scope", ""))
        external = ", ".join(str(x) for x in (claim.get("measured_external_systems", []) or []))
        lines.append(f"Public claim scope: `{scope}`")
        lines.append(f"Measured external systems: `{external or '-'}`")
    else:
        lines.append("_No claim-scope report loaded._")
    lines.append("")
    lines.append("## Ablation Evidence")
    lines.append("")
    abl = report.get("ablation_evidence", {}) or {}
    if abl:
        lines.append(
            f"System: `{abl.get('system', '')}`  Split: `{abl.get('split', '')}`  "
            f"n: `{abl.get('full_n', 0)}`  Evidence refs: `{abl.get('evidence_refs', 0)}`"
        )
        lines.append(
            f"Metabolism delta: `{abl.get('metabolism_delta_pp')}` pp  "
            f"Regions delta: `{abl.get('region_delta_pp')}` pp  "
            f"Affect delta: `{abl.get('affect_delta_pp')}` pp  "
            f"Forgetting cost ratio: `{abl.get('forgetting_cost_ratio')}`  "
            f"Accuracy regression: `{abl.get('forgetting_accuracy_regression_pp')}` pp"
        )
    else:
        lines.append("_No ablation evidence loaded._")
    lines.append("")
    lines.append("## Abstention Calibration")
    lines.append("")
    cal = report.get("abstention_calibration", {}) or {}
    if cal:
        lines.append(
            f"Method: `{cal.get('method', '')}`  Split: `{cal.get('split', '')}`  "
            f"System: `{cal.get('system', '')}`  n: `{cal.get('n', 0)}`  "
            f"tau: `{cal.get('tau', '')}`"
        )
        fp = cal.get("log_fingerprint", {}) if isinstance(cal.get("log_fingerprint"), dict) else {}
        if fp:
            lines.append(f"Calibration log fingerprint: `{fp.get('combined_sha256', '')}`")
    else:
        lines.append("_No abstention calibration report loaded._")
    lines.append("")
    lines.append("## Baseline Reproduction")
    lines.append("")
    base = report.get("baseline_reproduction", {})
    if base:
        lines.append(f"Status: **{base.get('status', 'UNKNOWN')}**  ")
        lines.append(f"System: `{base.get('system', '')}`  Dataset: `{base.get('dataset', '')}`  "
                     f"Rows: `{base.get('total_n', 0)}`")
    else:
        lines.append("_No baseline reproduction report loaded._")
    lines.append("")
    lines.append("## Snap-Back Fidelity")
    lines.append("")
    snap = report.get("snap_back", {})
    raw_snap_failures = snap.get("failures", []) or []
    snap_failure_count = len(raw_snap_failures) if isinstance(raw_snap_failures, list) else 1
    lines.append("| records with raw blob | lossless | rate | failures | data dir |")
    lines.append("|---:|---:|---:|---:|---|")
    snap_data_dir = str(snap.get("data_dir", "")).replace("|", "\\|")
    lines.append(f"| {snap.get('records_with_raw_blob', 0)} | "
                 f"{snap.get('lossless_byte_identical', 0)} | "
                 f"{_as_float(snap.get('rate', 0.0)) * 100:.4f}% | "
                 f"{snap_failure_count} | "
                 f"{snap_data_dir} |")
    lines.append("")
    lines.append("## Consolidation Health")
    lines.append("")
    lines.append("| system | groups | timed out | deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sysname, con in sorted(report.get("consolidation", {}).items()):
        lines.append(f"| {sysname} | {con.get('groups', 0)} | "
                     f"{con.get('extraction_timed_out', 0)} | {con.get('extraction_deferred', 0)} | "
                     f"{con.get('extraction_windows_planned', 0)} | "
                     f"{con.get('extraction_windows_submitted', 0)} | "
                     f"{con.get('extraction_raw_only_bounded', 0)} | "
                     f"{con.get('record_raw_only_bounded', 0)} | "
                     f"{con.get('extraction_partial_bounded', 0)} | "
                     f"{con.get('long_haystack_bounded', 0)} | "
                     f"{con.get('long_haystack_raw_only', 0)} |")
    out_path.write_text("\n".join(lines) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed public-release benchmark gate")
    ap.add_argument("--out", default="artifacts/bench")
    ap.add_argument("--report-out", default="", help="write Markdown + JSON report")
    ap.add_argument("--required-systems",
                    default="eidetic-plus,eidetic-plus-full,eidetic-product,rag-full,rag-vector,mem0,graphiti")
    ap.add_argument("--required-datasets", default="longmemeval,locomo")
    ap.add_argument("--required-categories", default="",
                    help=("optional dataset:cat|cat,dataset:cat override; default requires all "
                          "canonical LongMemEval/LoCoMo categories"))
    ap.add_argument("--headline-system", default="eidetic-plus")
    ap.add_argument("--baseline-systems", default="rag-full,rag-vector,mem0,graphiti")
    ap.add_argument("--integrity-system", default="eidetic-plus-full")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--min-runs", type=int, default=10)
    ap.add_argument("--min-questions-per-system", type=int, default=1000)
    ap.add_argument("--min-questions-per-dataset-per-system", type=int, default=-1,
                    help="minimum rows each required system must have on each required dataset; "
                         "default is min_questions_per_system / number of datasets")
    ap.add_argument("--min-category-questions-per-system", type=int, default=-1,
                    help="minimum unique samples each required system must have in each required category")
    ap.add_argument("--min-dataset-accuracy", type=float, default=0.85)
    ap.add_argument("--min-overall-delta-pp", type=float, default=10.0)
    ap.add_argument("--min-category-delta-pp", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-dataset-accuracy-ci-width-pp", type=float, default=10.0,
                    help="maximum Wilson CI width, in percentage points, for headline dataset slices")
    ap.add_argument("--max-category-accuracy-ci-width-pp", type=float, default=30.0,
                    help="maximum Wilson CI width, in percentage points, for headline category slices")
    ap.add_argument("--min-sample-clustered-paired-samples", type=int, default=-1,
                    help="minimum independent paired sample_id units for overall dominance; "
                         "default is min_questions_per_system")
    ap.add_argument("--min-category-sample-clustered-paired-samples", type=int, default=-1,
                    help="minimum independent paired sample_id units for category dominance; "
                         "default is min_category_questions_per_system")
    ap.add_argument("--min-clustered-discordant-samples", type=int, default=6,
                    help="minimum sample_id-level discordant pairs before clustered significance can pass")
    ap.add_argument("--min-verified-accuracy", type=float, default=0.50)
    ap.add_argument("--max-consolidation-timeouts", type=int, default=0)
    ap.add_argument("--max-consolidation-deferred", type=int, default=0)
    ap.add_argument("--min-snap-back-records", type=int, default=1)
    ap.add_argument("--max-query-tokens-median", type=float, default=7000.0)
    ap.add_argument("--max-search-p95-ms", type=float, default=500.0)
    ap.add_argument("--max-e2e-p50-ms", type=float, default=5000.0)
    ap.add_argument("--token-efficiency-baseline", default="rag-full")
    ap.add_argument("--min-token-efficiency-ratio", type=float, default=10.0)
    ap.add_argument("--max-abs-recall-slope-per-year", type=float, default=0.10)
    ap.add_argument("--min-age-slope-points", type=int, default=20)
    ap.add_argument("--baseline-reproduction-report", default="mem0_gate.json")
    ap.add_argument("--baseline-reproduction-system", default="mem0")
    ap.add_argument("--baseline-reproduction-dataset", default="locomo")
    ap.add_argument("--health-required-systems", default="mem0,graphiti")
    ap.add_argument("--no-competitor-health-gate", action="store_true",
                    help="skip requiring healthy baseline metadata for Mem0/Graphiti rows")
    ap.add_argument("--claim-scope-report", default="claim_scope.json")
    ap.add_argument("--top-systems-for-sota", default="chronos,mastra,byterover,hindsight")
    ap.add_argument("--min-external-evidence-n", type=int, default=100,
                    help="minimum sample count for each external top-comparator evidence record")
    ap.add_argument("--min-external-evidence-runs", type=int, default=1,
                    help="minimum run count for each external top-comparator evidence record")
    ap.add_argument("--abstention-calibration-report", default="abstention_v2_tau.json")
    ap.add_argument("--abstention-calibration-system", default="",
                    help="system used for ABSTENTION_V2_TAU calibration; default integrity system")
    ap.add_argument("--min-abstention-calibration-samples", type=int, default=50)
    ap.add_argument("--min-abstention-precision-target", type=float, default=0.95)
    ap.add_argument("--no-claim-scope-gate", action="store_true",
                    help="skip requiring an explicit public claim-scope report")
    ap.add_argument("--no-baseline-reproduction-gate", action="store_true",
                    help="skip requiring a passing external-baseline reproduction report")
    ap.add_argument("--no-category-wins", action="store_true",
                    help="skip per-dataset/category dominance checks")
    ap.add_argument("--no-ci-clear-dominance", action="store_true",
                    help="skip requiring headline lower Wilson CI to exceed baseline upper Wilson CI")
    ap.add_argument("--no-abstention-calibration-gate", action="store_true",
                    help="skip requiring ABSTENTION_V2_TAU to be tied to a dev calibration report")
    ap.add_argument("--slice-invariant-report", default="slice_invariant.json")
    ap.add_argument("--min-slice-invariant-draws", type=int, default=5)
    ap.add_argument("--min-slice-invariant-subset", type=int, default=20)
    ap.add_argument("--smqe-planner-report", default="smqe_planner_invariant.json")
    ap.add_argument("--min-smqe-planner-cases", type=int, default=24)
    ap.add_argument("--max-smqe-planner-p95-ms", type=float, default=10.0)
    ap.add_argument("--smqe-synthetic-report", default="smqe_synthetic_invariant.json")
    ap.add_argument("--min-smqe-synthetic-cases", type=int, default=24)
    ap.add_argument("--max-smqe-synthetic-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-claim-coverage-report", default="smqe_claim_coverage.json")
    ap.add_argument("--min-smqe-claim-coverage-cases", type=int, default=24)
    ap.add_argument("--min-smqe-claim-backend-rate", type=float, default=1.0)
    ap.add_argument("--max-smqe-claim-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-fullpath-report", default="smqe_fullpath_invariant.json")
    ap.add_argument("--min-smqe-fullpath-cases", type=int, default=24)
    ap.add_argument("--max-smqe-fullpath-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--max-smqe-fullpath-avg-context-tokens", type=float, default=80.0)
    ap.add_argument("--max-smqe-fullpath-p95-ms", type=float, default=100.0)
    ap.add_argument("--smqe-paraphrase-report", default="smqe_paraphrase_invariant.json")
    ap.add_argument("--min-smqe-paraphrase-cases", type=int, default=24)
    ap.add_argument("--max-smqe-paraphrase-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-conflict-report", default="smqe_conflict_invariant.json")
    ap.add_argument("--min-smqe-conflict-cases", type=int, default=24)
    ap.add_argument("--max-smqe-conflict-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-composition-report", default="smqe_composition_invariant.json")
    ap.add_argument("--min-smqe-composition-cases", type=int, default=24)
    ap.add_argument("--max-smqe-composition-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-relative-phrase-report", default="smqe_relative_phrase_invariant.json")
    ap.add_argument("--min-smqe-relative-phrase-cases", type=int, default=24)
    ap.add_argument("--max-smqe-relative-phrase-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-temporal-window-report", default="smqe_temporal_window_invariant.json")
    ap.add_argument("--min-smqe-temporal-window-cases", type=int, default=24)
    ap.add_argument("--max-smqe-temporal-window-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-attribution-report", default="smqe_attribution_invariant.json")
    ap.add_argument("--min-smqe-attribution-cases", type=int, default=24)
    ap.add_argument("--max-smqe-attribution-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-abstention-report", default="smqe_abstention_invariant.json")
    ap.add_argument("--min-smqe-abstention-cases", type=int, default=24)
    ap.add_argument("--smqe-scope-report", default="smqe_scope_invariant.json")
    ap.add_argument("--min-smqe-scope-cases", type=int, default=24)
    ap.add_argument("--max-smqe-scope-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-subscope-report", default="smqe_subscope_invariant.json")
    ap.add_argument("--min-smqe-subscope-cases", type=int, default=24)
    ap.add_argument("--max-smqe-subscope-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-time-report", default="smqe_time_invariant.json")
    ap.add_argument("--min-smqe-time-cases", type=int, default=24)
    ap.add_argument("--max-smqe-time-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--smqe-invalidation-report", default="smqe_invalidation_invariant.json")
    ap.add_argument("--min-smqe-invalidation-cases", type=int, default=24)
    ap.add_argument("--max-smqe-invalidation-avg-proof-tokens", type=float, default=80.0)
    ap.add_argument("--holdout-audit-report", default="holdout_audit.json")
    ap.add_argument("--min-holdout-audit-needles", type=int, default=1)
    ap.add_argument("--min-smqe-log-structured-rate", type=float, default=0.80)
    ap.add_argument("--min-smqe-log-claim-backend-rate", type=float, default=0.80)
    ap.add_argument("--ablation-report", default="ablation_report.json")
    ap.add_argument("--ablation-split", default="dev",
                    help="split used for ablation evidence; default keeps tuning evidence off test")
    ap.add_argument("--min-ablation-samples", type=int, default=20)
    ap.add_argument("--min-metabolism-accuracy-delta-pp", type=float, default=5.0)
    ap.add_argument("--min-region-accuracy-delta-pp", type=float, default=2.0)
    ap.add_argument("--min-affect-accuracy-delta-pp", type=float, default=2.0)
    ap.add_argument("--min-forgetting-cost-ratio", type=float, default=1.05)
    ap.add_argument("--max-forgetting-accuracy-regression-pp", type=float, default=1.0)
    ap.add_argument("--affect-salience-report", default="affect_salience_invariant.json")
    ap.add_argument("--min-affect-salience-cases", type=int, default=24)
    ap.add_argument("--max-affect-salience-lambda", type=float, default=0.5)
    ap.add_argument("--max-affect-salience-boost-ratio", type=float, default=0.5)
    ap.add_argument("--min-affect-salience-age-gap-seconds", type=float, default=2_592_000.0)
    ap.add_argument("--scratchpad-report", default="scratchpad_invariant.json")
    ap.add_argument("--min-scratchpad-cases", type=int, default=24)
    ap.add_argument("--region-routing-report", default="region_routing_invariant.json")
    ap.add_argument("--min-region-routing-cases", type=int, default=24)
    ap.add_argument("--reflex-recall-report", default="reflex_recall_invariant.json")
    ap.add_argument("--min-reflex-recall-cases", type=int, default=24)
    ap.add_argument("--max-reflex-recall-p95-ms", type=float, default=100.0)
    ap.add_argument("--no-holdout-profile-gate", action="store_true",
                    help="skip requiring holdout profile metadata")
    ap.add_argument("--no-holdout-audit-gate", action="store_true",
                    help="skip requiring a passing holdout leakage-audit sidecar")
    ap.add_argument("--no-smqe-log-policy-gate", action="store_true",
                    help="skip requiring parseable SMQE tier-mix evidence in benchmark logs")
    ap.add_argument("--no-ablation-gate", action="store_true",
                    help="skip requiring dev ablations that prove memory/consolidation, regions, affect, and forgetting value")
    ap.add_argument("--no-affect-salience-gate", action="store_true",
                    help="skip requiring the rotating affect-salience retrieval sidecar")
    ap.add_argument("--no-scratchpad-gate", action="store_true",
                    help="skip requiring the rotating scratchpad proof-surface sidecar")
    ap.add_argument("--no-region-routing-gate", action="store_true",
                    help="skip requiring the rotating region/cocoon routing sidecar")
    ap.add_argument("--no-reflex-recall-gate", action="store_true",
                    help="skip requiring the rotating local reflex-recall sidecar")
    ap.add_argument("--no-slice-invariant-gate", action="store_true",
                    help="skip requiring a passing slice-invariant sidecar")
    ap.add_argument("--no-smqe-planner-gate", action="store_true",
                    help="skip requiring the rotating SMQE planner sidecar")
    ap.add_argument("--no-smqe-synthetic-gate", action="store_true",
                    help="skip requiring the rotating synthetic SMQE invariant sidecar")
    ap.add_argument("--no-smqe-claim-coverage-gate", action="store_true",
                    help="skip requiring the rotating synthetic SMQE claim-coverage sidecar")
    ap.add_argument("--no-smqe-fullpath-gate", action="store_true",
                    help="skip requiring the rotating full-path SMQE adapter sidecar")
    ap.add_argument("--no-smqe-paraphrase-gate", action="store_true",
                    help="skip requiring the rotating SMQE paraphrase robustness sidecar")
    ap.add_argument("--no-smqe-conflict-gate", action="store_true",
                    help="skip requiring the rotating SMQE temporal-conflict sidecar")
    ap.add_argument("--no-smqe-composition-gate", action="store_true",
                    help="skip requiring the rotating SMQE multi-record composition sidecar")
    ap.add_argument("--no-smqe-relative-phrase-gate", action="store_true",
                    help="skip requiring the rotating SMQE source-relative phrase sidecar")
    ap.add_argument("--no-smqe-temporal-window-gate", action="store_true",
                    help="skip requiring the rotating SMQE temporal-window aggregate sidecar")
    ap.add_argument("--no-smqe-attribution-gate", action="store_true",
                    help="skip requiring the rotating SMQE actor-attribution sidecar")
    ap.add_argument("--no-smqe-abstention-gate", action="store_true",
                    help="skip requiring the rotating SMQE unsupported-question abstention sidecar")
    ap.add_argument("--no-smqe-scope-gate", action="store_true",
                    help="skip requiring the rotating SMQE scope-isolation sidecar")
    ap.add_argument("--no-smqe-subscope-gate", action="store_true",
                    help="skip requiring the rotating SMQE agent/project sub-scope sidecar")
    ap.add_argument("--no-smqe-time-gate", action="store_true",
                    help="skip requiring the rotating SMQE as-of time-isolation sidecar")
    ap.add_argument("--no-smqe-invalidation-gate", action="store_true",
                    help="skip requiring the rotating SMQE invalidated-memory sidecar")
    args = ap.parse_args()

    report = run_release_gate(
        Path(args.out),
        required_systems=_csv(args.required_systems),
        required_datasets=_csv(args.required_datasets),
        required_categories_by_dataset=(
            _parse_required_categories(args.required_categories)
            if args.required_categories.strip() else None
        ),
        headline_system=args.headline_system,
        baseline_systems=_csv(args.baseline_systems),
        integrity_system=args.integrity_system,
        split=args.split,
        min_runs=args.min_runs,
        min_questions_per_system=args.min_questions_per_system,
        min_questions_per_dataset_per_system=(
            None if args.min_questions_per_dataset_per_system < 0
            else args.min_questions_per_dataset_per_system
        ),
        min_category_questions_per_system=(
            None if args.min_category_questions_per_system < 0
            else args.min_category_questions_per_system
        ),
        min_dataset_accuracy=args.min_dataset_accuracy,
        min_overall_delta_pp=args.min_overall_delta_pp,
        min_category_delta_pp=args.min_category_delta_pp,
        alpha=args.alpha,
        max_dataset_accuracy_ci_width_pp=args.max_dataset_accuracy_ci_width_pp,
        max_category_accuracy_ci_width_pp=args.max_category_accuracy_ci_width_pp,
        min_sample_clustered_paired_samples=(
            None if args.min_sample_clustered_paired_samples < 0
            else args.min_sample_clustered_paired_samples
        ),
        min_category_sample_clustered_paired_samples=(
            None if args.min_category_sample_clustered_paired_samples < 0
            else args.min_category_sample_clustered_paired_samples
        ),
        require_ci_clear_dominance=not args.no_ci_clear_dominance,
        min_clustered_discordant_samples=args.min_clustered_discordant_samples,
        min_verified_accuracy=args.min_verified_accuracy,
        max_consolidation_timeouts=args.max_consolidation_timeouts,
        max_consolidation_deferred=args.max_consolidation_deferred,
        min_snap_back_records=args.min_snap_back_records,
        max_query_tokens_median=args.max_query_tokens_median,
        max_search_p95_ms=args.max_search_p95_ms,
        max_e2e_p50_ms=args.max_e2e_p50_ms,
        token_efficiency_baseline=args.token_efficiency_baseline,
        min_token_efficiency_ratio=args.min_token_efficiency_ratio,
        max_abs_recall_slope_per_year=args.max_abs_recall_slope_per_year,
        min_age_slope_points=args.min_age_slope_points,
        require_baseline_reproduction=not args.no_baseline_reproduction_gate,
        baseline_reproduction_report=args.baseline_reproduction_report,
        baseline_reproduction_system=args.baseline_reproduction_system,
        baseline_reproduction_dataset=args.baseline_reproduction_dataset,
        require_category_wins=not args.no_category_wins,
        require_competitor_health=not args.no_competitor_health_gate,
        health_required_systems=_csv(args.health_required_systems),
        require_claim_scope=not args.no_claim_scope_gate,
        claim_scope_report=args.claim_scope_report,
        top_systems_for_sota=_csv(args.top_systems_for_sota),
        min_external_evidence_n=args.min_external_evidence_n,
        min_external_evidence_runs=args.min_external_evidence_runs,
        require_abstention_calibration=not args.no_abstention_calibration_gate,
        abstention_calibration_report=args.abstention_calibration_report,
        abstention_calibration_system=args.abstention_calibration_system or None,
        min_abstention_calibration_samples=args.min_abstention_calibration_samples,
        min_abstention_precision_target=args.min_abstention_precision_target,
        require_holdout_profile=not args.no_holdout_profile_gate,
        require_holdout_audit=not args.no_holdout_audit_gate,
        holdout_audit_report=args.holdout_audit_report,
        min_holdout_audit_needles=args.min_holdout_audit_needles,
        require_smqe_log_policy=not args.no_smqe_log_policy_gate,
        min_smqe_log_structured_rate=args.min_smqe_log_structured_rate,
        min_smqe_log_claim_backend_rate=args.min_smqe_log_claim_backend_rate,
        require_ablation_evidence=not args.no_ablation_gate,
        ablation_report=args.ablation_report,
        ablation_split=args.ablation_split,
        min_ablation_samples=args.min_ablation_samples,
        min_metabolism_accuracy_delta_pp=args.min_metabolism_accuracy_delta_pp,
        min_region_accuracy_delta_pp=args.min_region_accuracy_delta_pp,
        min_affect_accuracy_delta_pp=args.min_affect_accuracy_delta_pp,
        min_forgetting_cost_ratio=args.min_forgetting_cost_ratio,
        max_forgetting_accuracy_regression_pp=args.max_forgetting_accuracy_regression_pp,
        require_affect_salience_invariant=not args.no_affect_salience_gate,
        affect_salience_report=args.affect_salience_report,
        min_affect_salience_cases=args.min_affect_salience_cases,
        max_affect_salience_lambda=args.max_affect_salience_lambda,
        max_affect_salience_boost_ratio=args.max_affect_salience_boost_ratio,
        min_affect_salience_age_gap_seconds=args.min_affect_salience_age_gap_seconds,
        require_scratchpad_invariant=not args.no_scratchpad_gate,
        scratchpad_report=args.scratchpad_report,
        min_scratchpad_cases=args.min_scratchpad_cases,
        require_region_routing_invariant=not args.no_region_routing_gate,
        region_routing_report=args.region_routing_report,
        min_region_routing_cases=args.min_region_routing_cases,
        require_reflex_recall_invariant=not args.no_reflex_recall_gate,
        reflex_recall_report=args.reflex_recall_report,
        min_reflex_recall_cases=args.min_reflex_recall_cases,
        max_reflex_recall_p95_ms=args.max_reflex_recall_p95_ms,
        require_slice_invariant=not args.no_slice_invariant_gate,
        slice_invariant_report=args.slice_invariant_report,
        min_slice_invariant_draws=args.min_slice_invariant_draws,
        min_slice_invariant_subset=args.min_slice_invariant_subset,
        require_smqe_planner_invariant=not args.no_smqe_planner_gate,
        smqe_planner_report=args.smqe_planner_report,
        min_smqe_planner_cases=args.min_smqe_planner_cases,
        max_smqe_planner_p95_ms=args.max_smqe_planner_p95_ms,
        require_smqe_synthetic_invariant=not args.no_smqe_synthetic_gate,
        smqe_synthetic_report=args.smqe_synthetic_report,
        min_smqe_synthetic_cases=args.min_smqe_synthetic_cases,
        max_smqe_synthetic_avg_proof_tokens=args.max_smqe_synthetic_avg_proof_tokens,
        require_smqe_claim_coverage=not args.no_smqe_claim_coverage_gate,
        smqe_claim_coverage_report=args.smqe_claim_coverage_report,
        min_smqe_claim_coverage_cases=args.min_smqe_claim_coverage_cases,
        min_smqe_claim_backend_rate=args.min_smqe_claim_backend_rate,
        max_smqe_claim_avg_proof_tokens=args.max_smqe_claim_avg_proof_tokens,
        require_smqe_fullpath_invariant=not args.no_smqe_fullpath_gate,
        smqe_fullpath_report=args.smqe_fullpath_report,
        min_smqe_fullpath_cases=args.min_smqe_fullpath_cases,
        max_smqe_fullpath_avg_proof_tokens=args.max_smqe_fullpath_avg_proof_tokens,
        max_smqe_fullpath_avg_context_tokens=args.max_smqe_fullpath_avg_context_tokens,
        max_smqe_fullpath_p95_ms=args.max_smqe_fullpath_p95_ms,
        require_smqe_paraphrase_invariant=not args.no_smqe_paraphrase_gate,
        smqe_paraphrase_report=args.smqe_paraphrase_report,
        min_smqe_paraphrase_cases=args.min_smqe_paraphrase_cases,
        max_smqe_paraphrase_avg_proof_tokens=args.max_smqe_paraphrase_avg_proof_tokens,
        require_smqe_conflict_invariant=not args.no_smqe_conflict_gate,
        smqe_conflict_report=args.smqe_conflict_report,
        min_smqe_conflict_cases=args.min_smqe_conflict_cases,
        max_smqe_conflict_avg_proof_tokens=args.max_smqe_conflict_avg_proof_tokens,
        require_smqe_composition_invariant=not args.no_smqe_composition_gate,
        smqe_composition_report=args.smqe_composition_report,
        min_smqe_composition_cases=args.min_smqe_composition_cases,
        max_smqe_composition_avg_proof_tokens=args.max_smqe_composition_avg_proof_tokens,
        require_smqe_relative_phrase_invariant=not args.no_smqe_relative_phrase_gate,
        smqe_relative_phrase_report=args.smqe_relative_phrase_report,
        min_smqe_relative_phrase_cases=args.min_smqe_relative_phrase_cases,
        max_smqe_relative_phrase_avg_proof_tokens=args.max_smqe_relative_phrase_avg_proof_tokens,
        require_smqe_temporal_window_invariant=not args.no_smqe_temporal_window_gate,
        smqe_temporal_window_report=args.smqe_temporal_window_report,
        min_smqe_temporal_window_cases=args.min_smqe_temporal_window_cases,
        max_smqe_temporal_window_avg_proof_tokens=args.max_smqe_temporal_window_avg_proof_tokens,
        require_smqe_attribution_invariant=not args.no_smqe_attribution_gate,
        smqe_attribution_report=args.smqe_attribution_report,
        min_smqe_attribution_cases=args.min_smqe_attribution_cases,
        max_smqe_attribution_avg_proof_tokens=args.max_smqe_attribution_avg_proof_tokens,
        require_smqe_abstention_invariant=not args.no_smqe_abstention_gate,
        smqe_abstention_report=args.smqe_abstention_report,
        min_smqe_abstention_cases=args.min_smqe_abstention_cases,
        require_smqe_scope_invariant=not args.no_smqe_scope_gate,
        smqe_scope_report=args.smqe_scope_report,
        min_smqe_scope_cases=args.min_smqe_scope_cases,
        max_smqe_scope_avg_proof_tokens=args.max_smqe_scope_avg_proof_tokens,
        require_smqe_subscope_invariant=not args.no_smqe_subscope_gate,
        smqe_subscope_report=args.smqe_subscope_report,
        min_smqe_subscope_cases=args.min_smqe_subscope_cases,
        max_smqe_subscope_avg_proof_tokens=args.max_smqe_subscope_avg_proof_tokens,
        require_smqe_time_invariant=not args.no_smqe_time_gate,
        smqe_time_report=args.smqe_time_report,
        min_smqe_time_cases=args.min_smqe_time_cases,
        max_smqe_time_avg_proof_tokens=args.max_smqe_time_avg_proof_tokens,
        require_smqe_invalidation_invariant=not args.no_smqe_invalidation_gate,
        smqe_invalidation_report=args.smqe_invalidation_report,
        min_smqe_invalidation_cases=args.min_smqe_invalidation_cases,
        max_smqe_invalidation_avg_proof_tokens=args.max_smqe_invalidation_avg_proof_tokens,
    )
    report_out = Path(args.report_out) if args.report_out else Path(args.out) / "release_gate.md"
    render_markdown(report, report_out)
    print(f"Release gate: {report['status']} ({len(report['failed_checks'])} failed checks)")
    print(f"Report -> {report_out}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
