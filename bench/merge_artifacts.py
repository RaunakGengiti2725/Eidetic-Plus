"""Build a validated composite benchmark artifact from multiple real run artifacts.

This is for public-claim hygiene. Dataset/system slices often run in separate directories, but the
release gate evaluates one `--out`. A composite artifact must therefore copy raw logs, preserve
source fingerprints, and make provenance explicit instead of masquerading as a direct harness run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from . import curves, scoreboard
from .datasets import split_of
from .fingerprints import log_fingerprint
from .harness import load_logs

_ENV_CONTRACT_KEYS = (
    "READER_MODEL",
    "READER_MODE",
    "JUDGE_MODEL",
    "JUDGE_BASE_URL",
    "JUDGE_BACKEND",
)
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
_SMQE_FULLPATH_MAX_AVG_CONTEXT_TOKENS = 80.0
_SMQE_FULLPATH_MAX_P95_LATENCY_MS = 100.0


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_manifest(src: Path) -> dict:
    path = src / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return _load_json(path)


def _slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return out.strip("_") or "artifact"


def _row_key(row: dict) -> tuple[str, str, str, str, int]:
    try:
        run_idx = int(row.get("run_idx", 0))
    except (TypeError, ValueError):
        run_idx = 0
    return (
        str(row.get("system", "")),
        str(row.get("dataset", "")),
        str(row.get("category", "")),
        str(row.get("sample_id", "")),
        run_idx,
    )


def _sample_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("category", "")),
        str(row.get("sample_id", "")),
    )


def _dataset_label(datasets: set[str]) -> str:
    if datasets == {"longmemeval", "locomo"}:
        return "both"
    if len(datasets) == 1:
        return next(iter(datasets))
    return "all"


def _sample_rows(rows: list[dict]) -> list[dict]:
    samples = {
        _sample_key(row)
        for row in rows
        if row.get("dataset") and row.get("category") and row.get("sample_id")
    }
    return [
        {"dataset": dataset, "category": category, "sample_id": sample_id}
        for dataset, category, sample_id in sorted(samples)
    ]


def _category_counts(sample_rows: list[dict]) -> dict[str, int]:
    counts = Counter(str(row.get("category", "")) for row in sample_rows)
    return dict(sorted(counts.items()))


def _common_env(manifests: list[dict]) -> dict:
    envs = [
        m.get("env", {})
        for m in manifests
        if isinstance(m.get("env", {}), dict)
    ]
    out: dict[str, str] = {}
    for key in _ENV_CONTRACT_KEYS:
        raw_values = [str(env.get(key, "")) for env in envs]
        values = {value for value in raw_values if value}
        if len(values) > 1:
            raise ValueError(f"source artifacts disagree on env {key}: {sorted(values)}")
        out[key] = next(iter(values)) if values else ""
    # A composite can span multiple data stores; source entries carry the concrete store/audit.
    out["DATA_DIR"] = ""
    return out


def _read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _copy_logs(src: Path, out: Path, index: int,
               include_systems: set[str] | None = None) -> list[Path]:
    slug = _slug(src.name)
    copied: list[Path] = []
    for path in sorted(src.glob("*__run*.jsonl")):
        if include_systems is not None:
            rows = _read_jsonl_rows(path)
            systems = {str(row.get("system", "")) for row in rows if row.get("system")}
            if not systems:
                continue
            if systems.isdisjoint(include_systems):
                continue
            if not systems.issubset(include_systems):
                raise ValueError(
                    f"{path} mixes included and excluded systems: {sorted(systems)}"
                )
        dest = out / f"src{index}_{slug}__{path.name}"
        shutil.copy2(path, dest)
        copied.append(dest)
    if not copied:
        raise ValueError(f"{src} contains no *__run*.jsonl logs")
    return copied


def _snap_back_source(src: Path, *, required: bool) -> dict:
    if not required:
        return {
            "status": "SKIP",
            "reason": "source has no included Eidetic raw-memory system",
            "records_with_raw_blob": 0,
            "lossless_byte_identical": 0,
            "failures": [],
        }
    path = src / "snap_back_audit.json"
    if not path.exists():
        return {
            "status": "FAIL",
            "records_with_raw_blob": 0,
            "lossless_byte_identical": 0,
            "failures": [{"source": str(src), "error": "missing snap_back_audit.json"}],
        }
    try:
        return _load_json(path)
    except (OSError, ValueError) as exc:
        return {
            "status": "FAIL",
            "records_with_raw_blob": 0,
            "lossless_byte_identical": 0,
            "failures": [{"source": str(src), "error": f"invalid snap_back_audit.json: {exc}"}],
        }


def _write_composite_snap_back(out: Path, sources: list[dict]) -> Path:
    total = 0
    lossless = 0
    failures: list[dict] = []
    audited_hashes: set[str] = set()
    all_pass = True
    for source in sources:
        snap = source.get("snap_back_audit", {}) or {}
        status = str(snap.get("status", "")).upper()
        if status == "SKIP":
            continue
        all_pass = all_pass and status == "PASS"
        total += int(snap.get("records_with_raw_blob", 0) or 0)
        lossless += int(snap.get("lossless_byte_identical", 0) or 0)
        for failure in snap.get("failures", []) or []:
            item = failure if isinstance(failure, dict) else {"error": str(failure)}
            failures.append({"source": source.get("path", ""), **item})
        for h in snap.get("audited_content_hashes", []) or []:
            h = str(h).strip()
            if h:
                audited_hashes.add(h)
    rate = (lossless / total) if total else 0.0
    ok = all_pass and total > 0 and lossless == total and rate >= 1.0 and not failures
    report = {
        "status": "PASS" if ok else "FAIL",
        "data_dir": "",
        "composite": True,
        "records_with_raw_blob": total,
        "lossless_byte_identical": lossless,
        "rate": rate,
        "rate_pct": round(rate * 100.0, 4),
        "min_records": 1,
        "audited_content_hashes": sorted(audited_hashes),
        "failures": failures[:50],
        "sources": [
            {
                "path": source.get("path", ""),
                "status": (source.get("snap_back_audit", {}) or {}).get("status"),
                "reason": (source.get("snap_back_audit", {}) or {}).get("reason"),
                "records_with_raw_blob": (
                    source.get("snap_back_audit", {}) or {}
                ).get("records_with_raw_blob", 0),
            }
            for source in sources
        ],
    }
    path = out / "snap_back_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def _load_sidecar(src: Path, name: str) -> tuple[dict, str]:
    path = src / name
    if not path.exists():
        return {}, "missing"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    if not isinstance(data, dict):
        return {}, "not a JSON object"
    return data, ""


def _sidecar_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sidecar_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_ABLATION_ACCURACY_KEYS = (
    "verified_accuracy",
    "accuracy",
    "score",
    "correct_rate",
    "exact_match",
)
_ABLATION_COST_KEYS = (
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


def _sidecar_metric_n(row: dict) -> int:
    if not isinstance(row, dict):
        return 0
    for key in ("n", "samples", "total_n", "cases", "questions"):
        value = _sidecar_int(row.get(key, 0), 0)
        if value > 0:
            return value
    return 0


def _sidecar_fraction(row: dict) -> float | None:
    if not isinstance(row, dict):
        return None
    for key in _ABLATION_ACCURACY_KEYS:
        if key not in row:
            continue
        value = _sidecar_float(row.get(key), -1.0)
        if 1.0 < value <= 100.0:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return value
    return None


def _sidecar_cost(row: dict) -> tuple[str, float | None]:
    if not isinstance(row, dict):
        return "", None
    for key in _ABLATION_COST_KEYS:
        if key not in row:
            continue
        value = _sidecar_float(row.get(key), -1.0)
        if value > 0.0:
            return key, value
    return "", None


def _sidecar_first_ablation(ablations: dict, labels: tuple[str, ...]) -> tuple[str, dict]:
    if not isinstance(ablations, dict):
        return "", {}
    for label in labels:
        row = ablations.get(label)
        if isinstance(row, dict):
            return label, row
    return "", {}


def _sidecar_evidence_refs(data: dict) -> list:
    refs = []
    for key in ("log_fingerprints", "artifact_fingerprints", "artifacts"):
        value = data.get(key)
        if isinstance(value, list):
            refs.extend(item for item in value if item)
        elif isinstance(value, dict) and value:
            refs.append(value)
    fp = data.get("log_fingerprint")
    if isinstance(fp, dict) and fp.get("combined_sha256"):
        refs.append(fp)
    return refs


def _weighted_ablation_row(rows: list[dict], *, cost_key: str) -> dict:
    total_n = sum(_sidecar_metric_n(row) for row in rows)
    if total_n <= 0:
        return {"n": 0, "verified_accuracy": 0.0, cost_key: 0.0}
    acc_weighted = 0.0
    cost_weighted = 0.0
    for row in rows:
        n = _sidecar_metric_n(row)
        acc = _sidecar_fraction(row) or 0.0
        cost = _sidecar_float(row.get(cost_key), 0.0)
        acc_weighted += acc * n
        cost_weighted += cost * n
    return {
        "n": total_n,
        "verified_accuracy": round(acc_weighted / total_n, 6),
        cost_key: round(cost_weighted / total_n, 6),
    }


def _write_composite_ablation_report(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    full_rows: list[dict] = []
    metabolism_rows: list[dict] = []
    region_rows: list[dict] = []
    forgetting_rows: list[dict] = []
    affect_rows: list[dict] = []
    evidence_refs: list = []
    systems: set[str] = set()
    splits: set[str] = set()
    metabolism_labels: list[str] = []
    region_labels: list[str] = []
    forgetting_labels: list[str] = []
    affect_labels: list[str] = []
    cost_keys: list[str] = []
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "ablation_report.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        full = data.get("full") if isinstance(data.get("full"), dict) else {}
        ablations = data.get("ablations") if isinstance(data.get("ablations"), dict) else {}
        metabolism_label, metabolism = _sidecar_first_ablation(ablations, _METABOLISM_ABLATION_KEYS)
        region_label, region = _sidecar_first_ablation(ablations, _REGION_ABLATION_KEYS)
        forgetting_label, forgetting = _sidecar_first_ablation(ablations, _FORGETTING_ABLATION_KEYS)
        affect_label, affect = _sidecar_first_ablation(ablations, _AFFECT_ABLATION_KEYS)
        cost_key, full_cost = _sidecar_cost(full)
        forgetting_cost_key, forgetting_cost = _sidecar_cost(forgetting)
        refs = _sidecar_evidence_refs(data)
        child_failures = data.get("failures") or []
        passed = (
            bool(data.get("pass"))
            and bool(full)
            and bool(metabolism)
            and bool(region)
            and bool(forgetting)
            and bool(affect)
            and _sidecar_metric_n(full) > 0
            and _sidecar_fraction(full) is not None
            and _sidecar_fraction(metabolism) is not None
            and _sidecar_fraction(region) is not None
            and _sidecar_fraction(forgetting) is not None
            and _sidecar_fraction(affect) is not None
            and bool(cost_key)
            and cost_key == forgetting_cost_key
            and full_cost is not None
            and forgetting_cost is not None
            and bool(refs)
            and not child_failures
        )
        ok = ok and passed
        if passed:
            full_rows.append(full)
            metabolism_rows.append(metabolism)
            region_rows.append(region)
            forgetting_rows.append(forgetting)
            affect_rows.append(affect)
            metabolism_labels.append(metabolism_label)
            region_labels.append(region_label)
            forgetting_labels.append(forgetting_label)
            affect_labels.append(affect_label)
            cost_keys.append(cost_key)
            evidence_refs.extend(refs)
        systems.add(str(data.get("system", "") or "").strip())
        splits.add(str(data.get("split", "") or "").strip().lower())
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "system": str(data.get("system", "") or ""),
            "split": str(data.get("split", "") or ""),
            "full_n": _sidecar_metric_n(full),
            "metabolism_label": metabolism_label,
            "region_label": region_label,
            "forgetting_label": forgetting_label,
            "affect_label": affect_label,
            "cost_metric": cost_key,
            "evidence_refs": len(refs),
        })

    systems.discard("")
    splits.discard("")
    cost_key = cost_keys[0] if cost_keys and all(key == cost_keys[0] for key in cost_keys) else "query_tokens_median"
    metabolism_label = (
        metabolism_labels[0]
        if metabolism_labels and all(label == metabolism_labels[0] for label in metabolism_labels)
        else "metabolism_off"
    )
    region_label = (
        region_labels[0]
        if region_labels and all(label == region_labels[0] for label in region_labels)
        else "regions_off"
    )
    forgetting_label = (
        forgetting_labels[0]
        if forgetting_labels and all(label == forgetting_labels[0] for label in forgetting_labels)
        else "forgetting_off"
    )
    affect_label = (
        affect_labels[0]
        if affect_labels and all(label == affect_labels[0] for label in affect_labels)
        else "affect_off"
    )
    path = out / "ablation_report.json"
    path.write_text(json.dumps({
        "pass": ok and bool(full_rows) and len(systems) == 1 and len(splits) == 1 and bool(evidence_refs),
        "artifact_kind": "composite",
        "system": next(iter(systems)) if len(systems) == 1 else "mixed",
        "split": next(iter(splits)) if len(splits) == 1 else "mixed",
        "full": _weighted_ablation_row(full_rows, cost_key=cost_key),
        "ablations": {
            metabolism_label: _weighted_ablation_row(metabolism_rows, cost_key=cost_key),
            region_label: _weighted_ablation_row(region_rows, cost_key=cost_key),
            forgetting_label: _weighted_ablation_row(forgetting_rows, cost_key=cost_key),
            affect_label: _weighted_ablation_row(affect_rows, cost_key=cost_key),
        },
        "log_fingerprints": evidence_refs[:50],
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_holdout_audit(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    findings: list[dict] = []
    registry_errors: list[str] = []
    needles_checked = 0
    holdout_needles_checked = 0
    forbidden_policy_checked = 0
    forbidden_fixed_answer_checked = 0
    forbidden_runtime_checked = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "holdout_audit.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        child_findings = data.get("findings", []) or []
        if not isinstance(child_findings, list):
            child_findings = [{"error": "findings is not a list"}]
        for finding in child_findings:
            if isinstance(finding, dict):
                findings.append({"source": str(src), **finding})
            else:
                findings.append({"source": str(src), "finding": str(finding)})
        registry_error = str(data.get("registry_error", "") or "")
        if registry_error:
            registry_errors.append(f"{src}: {registry_error}")
        source_needles = _sidecar_int(data.get("needles_checked", 0), 0)
        source_holdout_needles = _sidecar_int(data.get("holdout_needles_checked", 0), 0)
        source_legacy_scan = bool(data.get("legacy_policy_scan_enabled"))
        source_forbidden_policy = _sidecar_int(data.get("forbidden_policy_strings_checked", 0), 0)
        source_forbidden_fixed = _sidecar_int(data.get("forbidden_fixed_answer_strings_checked", 0), 0)
        source_forbidden_runtime = _sidecar_int(data.get("forbidden_runtime_symbols_checked", 0), 0)
        needles_checked += source_needles
        holdout_needles_checked += source_holdout_needles
        forbidden_policy_checked += source_forbidden_policy
        forbidden_fixed_answer_checked += source_forbidden_fixed
        forbidden_runtime_checked += source_forbidden_runtime
        passed = (
            bool(data.get("pass"))
            and not child_findings
            and not registry_error
            and source_holdout_needles > 0
            and source_legacy_scan
            and source_forbidden_policy > 0
            and source_forbidden_fixed > 0
            and source_forbidden_runtime > 0
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "needles_checked": source_needles,
            "holdout_needles_checked": source_holdout_needles,
            "legacy_policy_scan_enabled": source_legacy_scan,
            "forbidden_policy_strings_checked": source_forbidden_policy,
            "forbidden_fixed_answer_strings_checked": source_forbidden_fixed,
            "forbidden_runtime_symbols_checked": source_forbidden_runtime,
            "findings": len(child_findings),
            "registry_error": registry_error,
        })
    path = out / "holdout_audit.json"
    path.write_text(json.dumps({
        "pass": ok and holdout_needles_checked > 0 and not findings and not registry_errors,
        "artifact_kind": "composite",
        "findings": findings[:50],
        "needles_checked": needles_checked,
        "holdout_needles_checked": holdout_needles_checked,
        "legacy_policy_scan_enabled": all(
            bool(row.get("legacy_policy_scan_enabled")) for row in source_rows
        ),
        "forbidden_policy_strings_checked": forbidden_policy_checked,
        "forbidden_fixed_answer_strings_checked": forbidden_fixed_answer_checked,
        "forbidden_runtime_symbols_checked": forbidden_runtime_checked,
        "registry_error": "; ".join(registry_errors),
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _composite_seed_mode_for_sources(sources: list[Path], name: str) -> str:
    modes: list[str] = []
    for src in sources:
        data, error = _load_sidecar(src, name)
        if error:
            return "missing"
        modes.append(str(data.get("seed_mode", "") or "").strip().lower())
    return "random" if modes and all(mode == "random" for mode in modes) else "mixed"


_ROTATING_SIDECARS = (
    "affect_salience_invariant.json",
    "scratchpad_invariant.json",
    "region_routing_invariant.json",
    "reflex_recall_invariant.json",
    "smqe_planner_invariant.json",
    "smqe_synthetic_invariant.json",
    "smqe_claim_coverage.json",
    "smqe_fullpath_invariant.json",
    "smqe_paraphrase_invariant.json",
    "smqe_conflict_invariant.json",
    "smqe_composition_invariant.json",
    "smqe_relative_phrase_invariant.json",
    "smqe_temporal_window_invariant.json",
    "smqe_attribution_invariant.json",
    "smqe_abstention_invariant.json",
    "smqe_scope_invariant.json",
    "smqe_subscope_invariant.json",
    "smqe_time_invariant.json",
    "smqe_invalidation_invariant.json",
    "smqe_dialogue_invariant.json",
    "smqe_lacuna_invariant.json",
    "crystal_demotion_invariant.json",
)


def _fail_nonrandom_composite_sidecars(out: Path) -> None:
    for name in _ROTATING_SIDECARS:
        path = out / name
        if not path.exists():
            continue
        data = _load_json(path)
        seed_mode = str(data.get("seed_mode", "") or "").strip().lower()
        if seed_mode == "random":
            continue
        failures = data.get("failures")
        if not isinstance(failures, list):
            failures = []
        failure = f"seed_mode:{seed_mode or '<missing>'}"
        if failure not in failures:
            failures.append(failure)
        data["pass"] = False
        data["failures"] = failures
        path.write_text(json.dumps(data, indent=2) + "\n")


def _write_composite_slice_invariant(out: Path, sources: list[Path]) -> Path:
    reports: list[dict] = []
    source_rows: list[dict] = []
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "slice_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        child_reports = data.get("reports") if isinstance(data.get("reports"), list) else [data]
        child_reports = [item for item in child_reports if isinstance(item, dict)]
        reports.extend(child_reports)
        child_failures: list[str] = []
        for idx, item in enumerate(child_reports, start=1):
            split = str(item.get("split", "") or "").strip().lower()
            holdout_profile = str(item.get("holdout_profile", "") or "").strip().lower()
            if not bool(item.get("pass")):
                child_failures.append(f"report{idx}:pass:false")
            if split != "test":
                child_failures.append(f"report{idx}:split:{split or '<missing>'}")
            if holdout_profile != "holdout":
                child_failures.append(f"report{idx}:holdout_profile:{holdout_profile or '<missing>'}")
            all_sample_ids: list[str] = []
            draws = _sidecar_int(item.get("draws"), 0)
            subset = _sidecar_int(item.get("subset"), 0)
            requested_unique = max(0, draws * subset)
            declared_unique = _sidecar_int(item.get("unique_sample_ids"), -1)
            declared_pool_unique = _sidecar_int(item.get("pool_unique_sample_ids"), -1)
            declared_required_unique = _sidecar_int(item.get("required_unique_sample_ids"), -1)
            for draw_idx, run in enumerate(item.get("runs") or [], start=1):
                if not isinstance(run, dict):
                    continue
                sample_ids = [str(s) for s in (run.get("sample_ids") or []) if str(s).strip()]
                all_sample_ids.extend(sample_ids)
                draw_unique_sample_ids = len(set(sample_ids))
                if subset > 0 and draw_unique_sample_ids < subset:
                    child_failures.append(
                        f"report{idx}:draw{draw_idx}:unique_sample_ids:"
                        f"{draw_unique_sample_ids}<required:{subset}"
                    )
                if split in ("dev", "test"):
                    split_bad = [sid for sid in sample_ids if split_of(sid) != split]
                    if split_bad:
                        child_failures.append(
                            f"report{idx}:draw{draw_idx}:sample_split:{split}:bad={len(split_bad)}"
                        )
                score = run.get("score") if isinstance(run.get("score"), dict) else {}
                if not (bool(score.get("verified")) and "verified_correct" in score):
                    child_failures.append(f"report{idx}:draw{draw_idx}:score:not_verified")
            min_unique = max(requested_unique, declared_required_unique)
            unique_sample_ids = len(set(all_sample_ids))
            if declared_unique < 0:
                child_failures.append(f"report{idx}:unique_sample_ids:<missing>")
            elif declared_unique != unique_sample_ids:
                child_failures.append(
                    f"report{idx}:unique_sample_ids:{declared_unique}!=observed:{unique_sample_ids}"
                )
            if declared_required_unique < 0:
                child_failures.append(f"report{idx}:required_unique_sample_ids:<missing>")
            elif declared_required_unique < requested_unique:
                child_failures.append(
                    f"report{idx}:required_unique_sample_ids:"
                    f"{declared_required_unique}<required:{requested_unique}"
                )
            if declared_pool_unique < 0:
                child_failures.append(f"report{idx}:pool_unique_sample_ids:<missing>")
            else:
                required_for_pool = max(requested_unique, declared_required_unique)
                if declared_pool_unique < required_for_pool:
                    child_failures.append(
                        f"report{idx}:pool_unique_sample_ids:"
                        f"{declared_pool_unique}<required:{required_for_pool}"
                    )
                if declared_pool_unique < unique_sample_ids:
                    child_failures.append(
                        f"report{idx}:pool_unique_sample_ids:"
                        f"{declared_pool_unique}<observed:{unique_sample_ids}"
                    )
            if min_unique > 0 and unique_sample_ids < min_unique:
                child_failures.append(
                    f"report{idx}:unique_sample_ids:{unique_sample_ids}<required:{min_unique}"
                )
        passed = bool(data.get("pass")) and bool(child_reports) and not child_failures
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "reports": len(child_reports),
            "failures": child_failures[:20],
        })
    path = out / "slice_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and bool(reports),
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "reports": reports,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_planner(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    op_counts: Counter[str] = Counter()
    case_type_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_generic_terms = 0
    max_p95_latency = 0.0
    max_latency = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_planner_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        generic_terms = int(data.get("generic_term_checks", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_generic_terms += generic_terms
        max_p95_latency = max(max_p95_latency, float(data.get("p95_latency_ms", 0.0) or 0.0))
        max_latency = max(max_latency, float(data.get("max_latency_ms", 0.0) or 0.0))
        op_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        case_type_counts.update({
            str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()
        })
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks > 0
            and correct == checks
            and generic_terms >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
        })
    path = out / "smqe_planner_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks > 0 and total_correct == total_checks
        and total_generic_terms >= total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "generic_term_checks": total_generic_terms,
        "failures": failures[:50],
        "operator_counts": dict(sorted(op_counts.items())),
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "p95_latency_ms": round(max_p95_latency, 6),
        "max_latency_ms": round(max_latency, 6),
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_synthetic(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    op_counts: Counter[str] = Counter()
    claim_backend_op_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_synthetic_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        total_cases += cases
        total_correct += correct
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * cases
        op_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = bool(data.get("pass")) and cases > 0 and correct == cases and not child_failures
        ok = ok and passed
        source_rows.append({"path": str(src), "pass": passed, "cases": cases, "correct": correct})
    avg_proof = round(proof_weighted / total_cases, 2) if total_cases else 0.0
    path = out / "smqe_synthetic_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "correct": total_correct,
        "failures": failures[:50],
        "operator_counts": dict(sorted(op_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_affect_salience(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    failures: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_flip_checks = 0
    total_age_free_checks = 0
    total_bounded_checks = 0
    max_boost_ratio = 0.0
    max_lambda_salience = 0.0
    min_age_gap: float | None = None
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "affect_salience_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 7) or 0)
        correct = int(data.get("correct", 0) or 0)
        flip_checks = int(data.get("flip_checks", 0) or 0)
        age_free_checks = int(data.get("age_free_checks", 0) or 0)
        bounded_checks = int(data.get("bounded_checks", 0) or 0)
        lambda_salience = float(data.get("lambda_salience", 0.0) or 0.0)
        boost_ratio = float(data.get("max_boost_ratio", 0.0) or 0.0)
        age_gap = float(data.get("min_age_gap_seconds", 0.0) or 0.0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_flip_checks += flip_checks
        total_age_free_checks += age_free_checks
        total_bounded_checks += bounded_checks
        max_boost_ratio = max(max_boost_ratio, boost_ratio)
        max_lambda_salience = max(max_lambda_salience, lambda_salience)
        min_age_gap = age_gap if min_age_gap is None else min(min_age_gap, age_gap)
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 7
            and correct == checks
            and flip_checks >= cases * 2
            and age_free_checks >= cases
            and bounded_checks >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "max_boost_ratio": boost_ratio,
            "min_age_gap_seconds": age_gap,
        })
    path = out / "affect_salience_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 7
        and total_correct == total_checks and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "flip_checks": total_flip_checks,
        "age_free_checks": total_age_free_checks,
        "bounded_checks": total_bounded_checks,
        "lambda_salience": max_lambda_salience,
        "max_boost_ratio": round(max_boost_ratio, 6),
        "min_age_gap_seconds": min_age_gap,
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "failures": failures[:50],
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_scratchpad(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    failures: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_ordering = 0
    total_active_scope_filter = 0
    total_proof_link = 0
    total_top_k = 0
    total_retrieval_channel = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "scratchpad_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 11) or 0)
        correct = int(data.get("correct", 0) or 0)
        ordering = int(data.get("ordering_checks", 0) or 0)
        active_scope_filter = int(data.get("active_scope_filter_checks", 0) or 0)
        proof_link = int(data.get("proof_link_checks", 0) or 0)
        top_k = int(data.get("top_k_checks", 0) or 0)
        retrieval_channel = int(data.get("retrieval_channel_checks", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_ordering += ordering
        total_active_scope_filter += active_scope_filter
        total_proof_link += proof_link
        total_top_k += top_k
        total_retrieval_channel += retrieval_channel
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 11
            and correct == checks
            and ordering >= cases
            and active_scope_filter >= cases
            and proof_link >= cases * 4
            and top_k >= cases
            and retrieval_channel >= cases * 4
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "proof_link_checks": proof_link,
            "retrieval_channel_checks": retrieval_channel,
        })
    path = out / "scratchpad_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 11
        and total_correct == total_checks and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "ordering_checks": total_ordering,
        "active_scope_filter_checks": total_active_scope_filter,
        "proof_link_checks": total_proof_link,
        "top_k_checks": total_top_k,
        "retrieval_channel_checks": total_retrieval_channel,
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "failures": failures[:50],
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_region_routing(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    failures: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_dense_miss = 0
    total_active_scope_filter = 0
    total_nested_cocoon = 0
    total_proof_link = 0
    total_telemetry_trace = 0
    total_route_only_context = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "region_routing_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 12) or 0)
        correct = int(data.get("correct", 0) or 0)
        dense_miss = int(data.get("dense_miss_recovery_checks", 0) or 0)
        active_scope_filter = int(data.get("active_scope_filter_checks", 0) or 0)
        nested_cocoon = int(data.get("nested_cocoon_checks", 0) or 0)
        proof_link = int(data.get("proof_link_checks", 0) or 0)
        telemetry_trace = int(data.get("telemetry_trace_checks", 0) or 0)
        route_only_context = int(data.get("route_only_context_checks", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_dense_miss += dense_miss
        total_active_scope_filter += active_scope_filter
        total_nested_cocoon += nested_cocoon
        total_proof_link += proof_link
        total_telemetry_trace += telemetry_trace
        total_route_only_context += route_only_context
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 12
            and correct == checks
            and dense_miss >= cases * 3
            and active_scope_filter >= cases * 2
            and nested_cocoon >= cases * 2
            and proof_link >= cases * 2
            and telemetry_trace >= cases * 2
            and route_only_context >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "proof_link_checks": proof_link,
            "telemetry_trace_checks": telemetry_trace,
        })
    path = out / "region_routing_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 12
        and total_correct == total_checks and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "dense_miss_recovery_checks": total_dense_miss,
        "active_scope_filter_checks": total_active_scope_filter,
        "nested_cocoon_checks": total_nested_cocoon,
        "proof_link_checks": total_proof_link,
        "telemetry_trace_checks": total_telemetry_trace,
        "route_only_context_checks": total_route_only_context,
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "failures": failures[:50],
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_reflex_recall(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    failures: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_direct_hit = 0
    total_coactivation = 0
    total_active_scope_filter = 0
    total_proof_link = 0
    total_score_contract = 0
    total_latency_budget = 0
    max_latency = 0.0
    max_p95_latency = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "reflex_recall_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 12) or 0)
        correct = int(data.get("correct", 0) or 0)
        direct_hit = int(data.get("direct_hit_checks", 0) or 0)
        coactivation = int(data.get("coactivation_checks", 0) or 0)
        active_scope_filter = int(data.get("active_scope_filter_checks", 0) or 0)
        proof_link = int(data.get("proof_link_checks", 0) or 0)
        score_contract = int(data.get("score_contract_checks", 0) or 0)
        latency_budget = int(data.get("latency_budget_checks", 0) or 0)
        p95_latency = float(data.get("p95_latency_ms", 0.0) or 0.0)
        item_max_latency = float(data.get("max_latency_ms", 0.0) or 0.0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_direct_hit += direct_hit
        total_coactivation += coactivation
        total_active_scope_filter += active_scope_filter
        total_proof_link += proof_link
        total_score_contract += score_contract
        total_latency_budget += latency_budget
        max_p95_latency = max(max_p95_latency, p95_latency)
        max_latency = max(max_latency, item_max_latency)
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 12
            and correct == checks
            and direct_hit >= cases * 2
            and coactivation >= cases * 2
            and active_scope_filter >= cases * 4
            and proof_link >= cases
            and score_contract >= cases * 2
            and latency_budget >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "p95_latency_ms": p95_latency,
            "max_latency_ms": item_max_latency,
        })
    path = out / "reflex_recall_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 12
        and total_correct == total_checks and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "direct_hit_checks": total_direct_hit,
        "coactivation_checks": total_coactivation,
        "active_scope_filter_checks": total_active_scope_filter,
        "proof_link_checks": total_proof_link,
        "score_contract_checks": total_score_contract,
        "latency_budget_checks": total_latency_budget,
        "p95_latency_ms": round(max_p95_latency, 6),
        "max_latency_ms": round(max_latency, 6),
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "failures": failures[:50],
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_claim_coverage(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    op_counts: Counter[str] = Counter()
    claim_backend_op_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    claim_type_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    total_claim_backend = 0
    total_claims = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_claim_coverage.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        claims = int(data.get("claims_extracted", 0) or 0)
        total_cases += cases
        total_correct += correct
        total_claim_backend += claim_backend
        total_claims += claims
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * cases
        child_op_counts = {
            str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()
        }
        child_claim_op_counts = {
            str(k): int(v or 0)
            for k, v in (data.get("claim_backend_operator_counts") or {}).items()
        }
        op_counts.update(child_op_counts)
        claim_backend_op_counts.update(child_claim_op_counts)
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        claim_type_counts.update({str(k): int(v or 0) for k, v in (data.get("claim_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and correct == cases
            and claim_backend == cases
            and claims >= cases
            and all(child_claim_op_counts.get(op, 0) >= n for op, n in child_op_counts.items())
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) == 0
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "correct": correct,
            "claim_backend_correct": claim_backend,
            "claims_extracted": claims,
            "operator_counts": child_op_counts,
            "claim_backend_operator_counts": child_claim_op_counts,
        })
    avg_proof = round(proof_weighted / total_cases, 2) if total_cases else 0.0
    avg_claims = round(total_claims / total_cases, 2) if total_cases else 0.0
    path = out / "smqe_claim_coverage.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases
        and total_claim_backend == total_cases and total_claims >= total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "correct": total_correct,
        "claim_backend_correct": total_claim_backend,
        "claims_extracted": total_claims,
        "avg_claims_per_case": avg_claims,
        "failures": failures[:50],
        "operator_counts": dict(sorted(op_counts.items())),
        "claim_backend_operator_counts": dict(sorted(claim_backend_op_counts.items())),
        "claim_type_counts": dict(sorted(claim_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_fullpath(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    op_counts: Counter[str] = Counter()
    case_op_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    total_verified = 0
    total_structured = 0
    total_reader_calls = 0
    total_proof_link_checks = 0
    total_claim_backend = 0
    total_claims = 0
    total_latency_budget_checks = 0
    max_p95_latency = 0.0
    max_latency = 0.0
    proof_weighted = 0.0
    context_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_fullpath_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        verified = int(data.get("verified", 0) or 0)
        structured = int(data.get("structured_recall", 0) or 0)
        reader_calls = int(data.get("reader_calls", 0) or 0)
        proof_link_checks = int(data.get("proof_link_checks", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        claims = int(data.get("claims_extracted", 0) or 0)
        latency_budget = int(data.get("latency_budget_checks", 0) or 0)
        p95_latency_raw = data.get("p95_latency_ms")
        max_latency_raw = data.get("max_latency_ms")
        try:
            p95_latency = float(p95_latency_raw)
            max_latency_item = float(max_latency_raw)
            child_latency_valid = True
        except (TypeError, ValueError):
            p95_latency = 0.0
            max_latency_item = 0.0
            child_latency_valid = False
        total_cases += cases
        total_correct += correct
        total_verified += verified
        total_structured += structured
        total_reader_calls += reader_calls
        total_proof_link_checks += proof_link_checks
        total_claim_backend += claim_backend
        total_claims += claims
        total_latency_budget_checks += latency_budget
        max_p95_latency = max(max_p95_latency, p95_latency)
        max_latency = max(max_latency, max_latency_item)
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * cases
        child_avg_context_raw = data.get("avg_context_tokens")
        try:
            child_avg_context = float(child_avg_context_raw)
            child_context_valid = True
        except (TypeError, ValueError):
            child_avg_context = 0.0
            child_context_valid = False
        context_weighted += child_avg_context * cases
        child_op_counts = {
            str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()
        }
        child_case_op_counts = {
            str(k): int(v or 0) for k, v in (data.get("case_operator_counts") or {}).items()
        }
        op_counts.update(child_op_counts)
        case_op_counts.update(child_case_op_counts)
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and correct == cases
            and verified == cases
            and structured == cases
            and reader_calls == 0
            and proof_link_checks >= cases
            and claim_backend == cases
            and claims >= cases
            and child_context_valid
            and child_latency_valid
            and latency_budget >= cases
            and p95_latency <= _SMQE_FULLPATH_MAX_P95_LATENCY_MS
            and sum(child_case_op_counts.values()) == cases
            and all(child_case_op_counts.get(op, 0) >= 2 for op in _SMQE_REQUIRED_SYNTHETIC_OPS)
            and child_avg_context <= _SMQE_FULLPATH_MAX_AVG_CONTEXT_TOKENS
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "correct": correct,
            "verified": verified,
            "reader_calls": reader_calls,
            "proof_link_checks": proof_link_checks,
            "claim_backend_correct": claim_backend,
            "claims_extracted": claims,
            "avg_context_tokens": child_avg_context,
            "latency_budget_checks": latency_budget,
            "p95_latency_ms": p95_latency,
            "max_latency_ms": max_latency_item,
            "operator_counts": child_op_counts,
            "case_operator_counts": child_case_op_counts,
        })
    avg_proof = round(proof_weighted / total_cases, 2) if total_cases else 0.0
    avg_context = round(context_weighted / total_cases, 2) if total_cases else 0.0
    avg_claims = round(total_claims / total_cases, 2) if total_cases else 0.0
    path = out / "smqe_fullpath_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases
        and total_verified == total_cases and total_structured == total_cases
        and total_reader_calls == 0 and total_claim_backend == total_cases
        and total_proof_link_checks >= total_cases
        and total_claims >= total_cases
        and total_latency_budget_checks >= total_cases
        and max_p95_latency <= _SMQE_FULLPATH_MAX_P95_LATENCY_MS
        and avg_context <= _SMQE_FULLPATH_MAX_AVG_CONTEXT_TOKENS
        and all(case_op_counts.get(op, 0) >= 2 for op in _SMQE_REQUIRED_SYNTHETIC_OPS)
        and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "correct": total_correct,
        "verified": total_verified,
        "structured_recall": total_structured,
        "reader_calls": total_reader_calls,
        "proof_link_checks": total_proof_link_checks,
        "claim_backend_correct": total_claim_backend,
        "claims_extracted": total_claims,
        "avg_claims_per_case": avg_claims,
        "latency_budget_checks": total_latency_budget_checks,
        "latency_budget_ms": _SMQE_FULLPATH_MAX_P95_LATENCY_MS,
        "p95_latency_ms": round(max_p95_latency, 6),
        "max_latency_ms": round(max_latency, 6),
        "failures": failures[:50],
        "operator_counts": dict(sorted(op_counts.items())),
        "case_operator_counts": dict(sorted(case_op_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "avg_context_tokens": avg_context,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_paraphrase(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    op_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    total_record_backend = 0
    total_claim_backend = 0
    total_checks = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_paraphrase_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_correct += correct
        total_checks += checks
        total_record_backend += record_backend
        total_claim_backend += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        op_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and correct == cases
            and record_backend == cases
            and claim_backend == cases
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) >= cases
            and int((data.get("backend_counts") or {}).get("claim", 0) or 0) >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_paraphrase_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases
        and total_record_backend == total_cases and total_claim_backend == total_cases
        and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record_backend,
        "claim_backend_correct": total_claim_backend,
        "failures": failures[:50],
        "operator_counts": dict(sorted(op_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_conflict(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    value_type_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    total_record_backend = 0
    total_claim_backend = 0
    total_checks = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_conflict_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_correct += correct
        total_checks += checks
        total_record_backend += record_backend
        total_claim_backend += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        value_type_counts.update({str(k): int(v or 0) for k, v in (data.get("value_type_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and correct == cases
            and record_backend == cases
            and claim_backend == cases
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) >= cases
            and int((data.get("backend_counts") or {}).get("claim", 0) or 0) >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_conflict_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases
        and total_record_backend == total_cases and total_claim_backend == total_cases
        and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record_backend,
        "claim_backend_correct": total_claim_backend,
        "failures": failures[:50],
        "value_type_counts": dict(sorted(value_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_composition(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record_backend = 0
    total_claim_backend = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_composition_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record_backend += record_backend
        total_claim_backend += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 2
            and correct == cases
            and record_backend == cases
            and claim_backend == cases
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) >= cases
            and int((data.get("backend_counts") or {}).get("claim", 0) or 0) >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_composition_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 2
        and total_correct == total_cases and total_record_backend == total_cases
        and total_claim_backend == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record_backend,
        "claim_backend_correct": total_claim_backend,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_relative_phrase(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record_backend = 0
    total_claim_backend = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_relative_phrase_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record_backend += record_backend
        total_claim_backend += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 2
            and correct == cases
            and record_backend == cases
            and claim_backend == cases
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) >= cases
            and int((data.get("backend_counts") or {}).get("claim", 0) or 0) >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_relative_phrase_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 2
        and total_correct == total_cases and total_record_backend == total_cases
        and total_claim_backend == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record_backend,
        "claim_backend_correct": total_claim_backend,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_temporal_window(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record_backend = 0
    total_claim_backend = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_temporal_window_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record_backend += record_backend
        total_claim_backend += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 2
            and correct == cases
            and record_backend == cases
            and claim_backend == cases
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) >= cases
            and int((data.get("backend_counts") or {}).get("claim", 0) or 0) >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_temporal_window_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 2
        and total_correct == total_cases and total_record_backend == total_cases
        and total_claim_backend == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record_backend,
        "claim_backend_correct": total_claim_backend,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_attribution(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record_backend = 0
    total_claim_backend = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_attribution_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record_backend += record_backend
        total_claim_backend += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 2
            and correct == cases
            and record_backend == cases
            and claim_backend == cases
            and int((data.get("backend_counts") or {}).get("record", 0) or 0) >= cases
            and int((data.get("backend_counts") or {}).get("claim", 0) or 0) >= cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_attribution_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 2
        and total_correct == total_cases and total_record_backend == total_cases
        and total_claim_backend == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record_backend,
        "claim_backend_correct": total_claim_backend,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_abstention(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_abstained = 0
    total_record_only = 0
    total_claims_present = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_abstention_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 2) or 0)
        abstained = int(data.get("abstained", 0) or 0)
        record_only = int(data.get("record_only_abstained", 0) or 0)
        claims_present = int(data.get("claims_present_abstained", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_abstained += abstained
        total_record_only += record_only
        total_claims_present += claims_present
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 2
            and abstained == cases
            and record_only == cases
            and claims_present == cases
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "abstained": abstained,
            "record_only_abstained": record_only,
            "claims_present_abstained": claims_present,
        })
    path = out / "smqe_abstention_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 2
        and total_abstained == total_cases and total_record_only == total_cases
        and total_claims_present == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "abstained": total_abstained,
        "record_only_abstained": total_record_only,
        "claims_present_abstained": total_claims_present,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_scope(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    operator_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record = 0
    total_claim = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_scope_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 4) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record += record_backend
        total_claim += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        operator_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 4
            and correct == checks
            and record_backend == cases * 2
            and claim_backend == cases * 2
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_scope_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 4
        and total_correct == total_checks and total_record == total_cases * 2
        and total_claim == total_cases * 2 and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record,
        "claim_backend_correct": total_claim,
        "failures": failures[:50],
        "operator_counts": dict(sorted(operator_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_subscope(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    operator_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record = 0
    total_claim = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_subscope_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 4) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record += record_backend
        total_claim += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        operator_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 4
            and correct == checks
            and record_backend == cases * 2
            and claim_backend == cases * 2
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_subscope_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 4
        and total_correct == total_checks and total_record == total_cases * 2
        and total_claim == total_cases * 2 and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record,
        "claim_backend_correct": total_claim,
        "failures": failures[:50],
        "operator_counts": dict(sorted(operator_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_time(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    operator_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record = 0
    total_claim = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_time_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 4) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record += record_backend
        total_claim += claim_backend
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        operator_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 4
            and correct == checks
            and record_backend == cases * 2
            and claim_backend == cases * 2
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    path = out / "smqe_time_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 4
        and total_correct == total_checks and total_record == total_cases * 2
        and total_claim == total_cases * 2 and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record,
        "claim_backend_correct": total_claim,
        "failures": failures[:50],
        "operator_counts": dict(sorted(operator_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_invalidation(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    operator_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_checks = 0
    total_correct = 0
    total_record = 0
    total_claim = 0
    total_preference_cases = 0
    total_preference_checks = 0
    total_preference_correct = 0
    total_preference_record = 0
    total_preference_claim = 0
    proof_weighted = 0.0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_invalidation_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        checks = int(data.get("checks", cases * 4) or 0)
        correct = int(data.get("correct", 0) or 0)
        record_backend = int(data.get("record_backend_correct", 0) or 0)
        claim_backend = int(data.get("claim_backend_correct", 0) or 0)
        preference_cases = int(data.get("preference_supersession_cases", 0) or 0)
        preference_checks = int(data.get("preference_supersession_checks", 0) or 0)
        preference_correct = int(data.get("preference_supersession_correct", 0) or 0)
        preference_record = int(data.get("preference_supersession_record_correct", 0) or 0)
        preference_claim = int(data.get("preference_supersession_claim_correct", 0) or 0)
        total_cases += cases
        total_checks += checks
        total_correct += correct
        total_record += record_backend
        total_claim += claim_backend
        total_preference_cases += preference_cases
        total_preference_checks += preference_checks
        total_preference_correct += preference_correct
        total_preference_record += preference_record
        total_preference_claim += preference_claim
        proof_weighted += float(data.get("avg_proof_tokens", 0.0) or 0.0) * max(1, checks)
        operator_counts.update({str(k): int(v or 0) for k, v in (data.get("operator_counts") or {}).items()})
        backend_counts.update({str(k): int(v or 0) for k, v in (data.get("backend_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = (
            bool(data.get("pass"))
            and cases > 0
            and checks == cases * 4
            and correct == checks
            and record_backend == cases * 2
            and claim_backend == cases * 2
            and bool(data.get("preference_supersession_pass"))
            and preference_cases > 0
            and preference_checks == preference_cases * 4
            and preference_correct == preference_checks
            and preference_record == preference_cases * 2
            and preference_claim == preference_cases * 2
            and not child_failures
        )
        ok = ok and passed
        source_rows.append({
            "path": str(src),
            "pass": passed,
            "cases": cases,
            "checks": checks,
            "correct": correct,
            "record_backend_correct": record_backend,
            "claim_backend_correct": claim_backend,
            "preference_supersession_cases": preference_cases,
            "preference_supersession_checks": preference_checks,
            "preference_supersession_correct": preference_correct,
        })
    avg_proof = round(proof_weighted / total_checks, 2) if total_checks else 0.0
    preference_pass = (
        total_preference_cases > 0
        and total_preference_checks == total_preference_cases * 4
        and total_preference_correct == total_preference_checks
        and total_preference_record == total_preference_cases * 2
        and total_preference_claim == total_preference_cases * 2
    )
    path = out / "smqe_invalidation_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_checks == total_cases * 4
        and total_correct == total_checks and total_record == total_cases * 2
        and total_claim == total_cases * 2 and preference_pass and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "checks": total_checks,
        "correct": total_correct,
        "record_backend_correct": total_record,
        "claim_backend_correct": total_claim,
        "preference_supersession_pass": preference_pass,
        "preference_supersession_cases": total_preference_cases,
        "preference_supersession_checks": total_preference_checks,
        "preference_supersession_correct": total_preference_correct,
        "preference_supersession_record_correct": total_preference_record,
        "preference_supersession_claim_correct": total_preference_claim,
        "failures": failures[:50],
        "operator_counts": dict(sorted(operator_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": avg_proof,
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_dialogue(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_dialogue_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        total_cases += cases
        total_correct += correct
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = bool(data.get("pass")) and cases > 0 and correct == cases and not child_failures
        ok = ok and passed
        source_rows.append({"path": str(src), "pass": passed, "cases": cases, "correct": correct})
    path = out / "smqe_dialogue_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "correct": total_correct,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_smqe_lacuna(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    case_type_counts: Counter[str] = Counter()
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "smqe_lacuna_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        total_cases += cases
        total_correct += correct
        case_type_counts.update({str(k): int(v or 0) for k, v in (data.get("case_type_counts") or {}).items()})
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = bool(data.get("pass")) and cases > 0 and correct == cases and not child_failures
        ok = ok and passed
        source_rows.append({"path": str(src), "pass": passed, "cases": cases, "correct": correct})
    path = out / "smqe_lacuna_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "correct": total_correct,
        "failures": failures[:50],
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _write_composite_crystal_demotion(out: Path, sources: list[Path]) -> Path:
    source_rows: list[dict] = []
    failures: list[dict] = []
    total_cases = 0
    total_correct = 0
    ratio_weighted = 0.0
    ratio_cases = 0
    ok = True
    for src in sources:
        data, error = _load_sidecar(src, "crystal_demotion_invariant.json")
        if error:
            ok = False
            source_rows.append({"path": str(src), "pass": False, "reason": error})
            continue
        cases = int(data.get("cases", 0) or 0)
        correct = int(data.get("correct", 0) or 0)
        ratio = data.get("avg_demotion_ratio")
        total_cases += cases
        total_correct += correct
        if ratio is not None:
            ratio_weighted += float(ratio) * max(1, cases)
            ratio_cases += max(1, cases)
        child_failures = data.get("failures") or []
        if isinstance(child_failures, list):
            failures.extend(item if isinstance(item, dict) else {"failure": str(item)} for item in child_failures)
        passed = bool(data.get("pass")) and cases > 0 and correct == cases and not child_failures
        ok = ok and passed
        source_rows.append({"path": str(src), "pass": passed, "cases": cases, "correct": correct,
                            "avg_demotion_ratio": ratio})
    path = out / "crystal_demotion_invariant.json"
    path.write_text(json.dumps({
        "pass": ok and total_cases > 0 and total_correct == total_cases and not failures,
        "artifact_kind": "composite",
        "seed_mode": _composite_seed_mode_for_sources(sources, path.name),
        "cases": total_cases,
        "correct": total_correct,
        "avg_demotion_ratio": round(ratio_weighted / ratio_cases, 4) if ratio_cases else None,
        "failures": failures[:50],
        "sources": source_rows,
    }, indent=2) + "\n")
    return path


def _build_composite(
    sources: list[Path],
    out: Path,
    source_system_filters: dict[int, set[str]] | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)

    manifests: list[dict] = []
    source_entries: list[dict] = []
    seen_rows: dict[tuple[str, str, str, str, int], str] = {}
    all_rows: list[dict] = []
    copied_logs: list[str] = []
    split = ""
    judge = None

    for index, src in enumerate(sources):
        manifest = _load_manifest(src)
        if bool(manifest.get("render_only")):
            raise ValueError(f"{src} is render_only; merge only real source run artifacts")
        src_split = str(manifest.get("split", "")).strip()
        if split and src_split != split:
            raise ValueError(f"source artifacts disagree on split: {split} vs {src_split}")
        split = src_split
        src_judge = manifest.get("judge", {})
        if judge is not None and src_judge != judge:
            raise ValueError("source artifacts disagree on judge metadata")
        judge = src_judge
        manifests.append(manifest)

        include_systems = (source_system_filters or {}).get(index)
        rows = load_logs(src)
        if include_systems is not None:
            rows = [row for row in rows if str(row.get("system", "")) in include_systems]
            if not rows:
                raise ValueError(
                    f"{src} has no rows matching source system filter "
                    f"{sorted(include_systems)}"
                )
        for row in rows:
            key = _row_key(row)
            if key in seen_rows:
                raise ValueError(
                    "duplicate row identity across sources: "
                    f"{key} in {src} and {seen_rows[key]}"
                )
            seen_rows[key] = str(src)
            all_rows.append(row)
        copied = _copy_logs(src, out, index, include_systems)
        copied_logs.extend(path.name for path in copied)
        included_systems = sorted({
            str(row.get("system", "")) for row in rows if row.get("system")
        })
        requires_snap_back = any(
            system.startswith("eidetic") for system in included_systems
        )
        source_entries.append({
            "path": str(src),
            "name": src.name,
            "split": src_split,
            "render_only": bool(manifest.get("render_only")),
            "systems": ",".join(included_systems),
            "source_systems": manifest.get("systems", ""),
            "system_filter": sorted(include_systems) if include_systems is not None else [],
            "dataset": manifest.get("dataset", ""),
            "runs": manifest.get("runs"),
            "log_fingerprint": log_fingerprint(src),
            "snap_back_required": requires_snap_back,
            "snap_back_audit": _snap_back_source(src, required=requires_snap_back),
        })

    run_indices = sorted({int(row.get("run_idx", 0) or 0) for row in all_rows})
    if run_indices:
        expected = list(range(min(run_indices), max(run_indices) + 1))
        if run_indices != expected:
            raise ValueError(f"composite run_idx values are not contiguous: {run_indices}")
    samples = _sample_rows(all_rows)
    samples_file = out / "composite.samples.json"
    samples_file.write_text(json.dumps([
        {"dataset": row["dataset"], "sample_id": row["sample_id"]}
        for row in samples
    ], indent=2) + "\n")
    source_holdouts = [
        str(manifest.get("holdout_profile", "") or "").strip().lower()
        for manifest in manifests
    ]
    holdout_profile = "holdout" if source_holdouts and all(v == "holdout" for v in source_holdouts) else ""
    _write_composite_holdout_audit(out, sources)
    _write_composite_ablation_report(out, sources)
    _write_composite_slice_invariant(out, sources)
    _write_composite_affect_salience(out, sources)
    _write_composite_scratchpad(out, sources)
    _write_composite_region_routing(out, sources)
    _write_composite_reflex_recall(out, sources)
    _write_composite_smqe_planner(out, sources)
    _write_composite_smqe_synthetic(out, sources)
    _write_composite_smqe_claim_coverage(out, sources)
    _write_composite_smqe_fullpath(out, sources)
    _write_composite_smqe_paraphrase(out, sources)
    _write_composite_smqe_conflict(out, sources)
    _write_composite_smqe_composition(out, sources)
    _write_composite_smqe_relative_phrase(out, sources)
    _write_composite_smqe_temporal_window(out, sources)
    _write_composite_smqe_attribution(out, sources)
    _write_composite_smqe_abstention(out, sources)
    _write_composite_smqe_scope(out, sources)
    _write_composite_smqe_subscope(out, sources)
    _write_composite_smqe_time(out, sources)
    _write_composite_smqe_invalidation(out, sources)
    _write_composite_smqe_dialogue(out, sources)
    _write_composite_smqe_lacuna(out, sources)
    _write_composite_crystal_demotion(out, sources)
    _fail_nonrandom_composite_sidecars(out)
    systems = sorted({str(row.get("system", "")) for row in all_rows if row.get("system")})
    datasets = {str(row.get("dataset", "")) for row in all_rows if row.get("dataset")}
    manifest = {
        "artifact_kind": "composite",
        "render_only": False,
        "systems": ",".join(systems),
        "dataset": _dataset_label(datasets),
        "split": split,
        "subset": 0,
        "sample_offset": 0,
        "sample_strategy": "composite",
        "runs": len(run_indices),
        "run_offset": min(run_indices) if run_indices else 0,
        "samples_file": str(samples_file),
        "holdout_profile": holdout_profile,
        "variant": "composite",
        "judge": judge or {},
        "sample_count": len(samples),
        "category_counts": _category_counts(samples),
        "sample_rows": samples,
        "system_failures": [
            failure
            for manifest in manifests
            for failure in (manifest.get("system_failures", []) or [])
        ],
        "metabolism_mode": any(bool(m.get("metabolism_mode")) for m in manifests),
        "env": _common_env(manifests),
        "composite_sources": source_entries,
        "copied_logs": copied_logs,
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    _write_composite_snap_back(out, source_entries)
    scoreboard.render(out, judge_desc=judge if isinstance(judge, dict) else None)
    curves.render(out)
    return manifest_path


def merge_artifacts(
    source_dirs: list[Path],
    out_dir: Path,
    *,
    overwrite: bool = False,
    source_system_filters: dict[int, set[str]] | None = None,
) -> Path:
    if len(source_dirs) < 2:
        raise ValueError("merge_artifacts requires at least two source artifact directories")
    sources = [Path(src).expanduser().resolve() for src in source_dirs]
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f"{out} exists and is not a directory")
        if any(out.iterdir()) and not overwrite:
            raise FileExistsError(f"{out} is not empty; pass overwrite=True to replace it")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{_slug(out.name)}.tmp-", dir=out.parent))
    try:
        manifest_path = _build_composite(sources, tmp, source_system_filters)
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(tmp), str(out))
        tmp = None
        return out / manifest_path.name
    finally:
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp)


def _parse_source_system_filters(values: list[str] | None,
                                 source_count: int) -> dict[int, set[str]]:
    filters: dict[int, set[str]] = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError("--source-systems entries must look like index:system,system")
        raw_index, raw_systems = value.split(":", 1)
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"invalid --source-systems index {raw_index!r}") from exc
        if index < 0 or index >= source_count:
            raise ValueError(
                f"--source-systems index {index} out of range for {source_count} sources"
            )
        systems = {item.strip() for item in raw_systems.split(",") if item.strip()}
        if not systems:
            raise ValueError(f"--source-systems {value!r} did not name any systems")
        filters.setdefault(index, set()).update(systems)
    return filters


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge real benchmark artifacts into a composite out dir.")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="two or more artifact directories containing raw *__run*.jsonl logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--source-systems", action="append", default=[],
                    help="optional source-index filter, e.g. 3:rag-full,rag-vector")
    args = ap.parse_args()
    filters = _parse_source_system_filters(args.source_systems, len(args.sources))
    path = merge_artifacts([Path(src) for src in args.sources], Path(args.out),
                           overwrite=args.overwrite,
                           source_system_filters=filters or None)
    print(f"Composite manifest -> {path}")
    print(f"Scoreboard          -> {Path(args.out) / 'scoreboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
