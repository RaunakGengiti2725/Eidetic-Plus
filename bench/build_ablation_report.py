"""Build release-gate ablation evidence from real benchmark artifacts.

The release gate requires ``ablation_report.json`` so attribution is measured, not narrated. This
module produces that report from five run directories: full memory, metabolism/consolidation off,
memory-region routing off, forgetting off, and affective salience off. It never inspects questions
or gold labels beyond the raw harness logs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .compare import _load_logs_strict
from .fingerprints import log_fingerprint


def _load_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _manifest_env(manifest: dict) -> dict:
    env = manifest.get("env", {}) if isinstance(manifest, dict) else {}
    return env if isinstance(env, dict) else {}


def _affect_profile_failures(manifests: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    for role in ("full", "metabolism_off", "regions_off", "forgetting_off"):
        env = _manifest_env(manifests.get(role, {}))
        if not _truthy(env.get("AFFECT_SALIENCE")):
            failures.append(f"{role}:AFFECT_SALIENCE:{env.get('AFFECT_SALIENCE', '<missing>')}:expected:on")
    affect_env = _manifest_env(manifests.get("affect_off", {}))
    if _truthy(affect_env.get("AFFECT_SALIENCE")):
        failures.append(
            f"affect_off:AFFECT_SALIENCE:{affect_env.get('AFFECT_SALIENCE', '<missing>')}:expected:off"
        )

    full_env = _manifest_env(manifests.get("full", {}))
    if not full_env or not affect_env:
        failures.append("affect_off:env_manifest_missing")
        return failures
    allowed_drift = {"AFFECT_SALIENCE", "DATA_DIR"}
    drift = [
        key
        for key in sorted(set(full_env) | set(affect_env))
        if key not in allowed_drift and str(full_env.get(key, "")) != str(affect_env.get(key, ""))
    ]
    if drift:
        suffix = "" if len(drift) <= 8 else f",+{len(drift) - 8} more"
        failures.append(f"affect_off:non_affect_env_drift:{','.join(drift[:8])}{suffix}")
    return failures


def _region_profile_failures(manifests: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    for role in ("full", "forgetting_off", "affect_off"):
        env = _manifest_env(manifests.get(role, {}))
        if not _truthy(env.get("GIST_CHANNEL")):
            failures.append(f"{role}:GIST_CHANNEL:{env.get('GIST_CHANNEL', '<missing>')}:expected:on")
    region_env = _manifest_env(manifests.get("regions_off", {}))
    if _truthy(region_env.get("GIST_CHANNEL")):
        failures.append(
            f"regions_off:GIST_CHANNEL:{region_env.get('GIST_CHANNEL', '<missing>')}:expected:off"
        )

    full_env = _manifest_env(manifests.get("full", {}))
    if not full_env or not region_env:
        failures.append("regions_off:env_manifest_missing")
        return failures
    allowed_drift = {"GIST_CHANNEL", "DATA_DIR"}
    drift = [
        key
        for key in sorted(set(full_env) | set(region_env))
        if key not in allowed_drift and str(full_env.get(key, "")) != str(region_env.get(key, ""))
    ]
    if drift:
        suffix = "" if len(drift) <= 8 else f",+{len(drift) - 8} more"
        failures.append(f"regions_off:non_region_env_drift:{','.join(drift[:8])}{suffix}")
    return failures


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _row_key(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row.get("dataset", "")),
        str(row.get("category", "")),
        str(row.get("sample_id", "")),
        int(row.get("run_idx", 0) or 0),
    )


def _sample_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("category", "")),
        str(row.get("sample_id", "")),
    )


def _verified(row: dict) -> bool:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    return bool(extra.get("verified"))


def _rows_for_system(out_dir: Path, system: str) -> tuple[list[dict], list[str]]:
    bundle = _load_logs_strict(out_dir)
    failures: list[str] = []
    rows = [row for row in bundle.rows if row.get("system") == system]
    if not rows:
        failures.append(f"{out_dir}:system:{system}:no_rows")
        return [], failures
    errored = [row for row in rows if row.get("error")]
    if errored:
        failures.append(f"{out_dir}:error_rows:{len(errored)}")
    missing_verified = [
        _row_key(row)
        for row in rows
        if not isinstance(row.get("extra"), dict) or "verified" not in row.get("extra", {})
    ]
    if missing_verified:
        failures.append(f"{out_dir}:verified_metadata_missing:{len(missing_verified)}")
    return [row for row in rows if not row.get("error")], failures


def _sample_clustered_rate(rows: list[dict], *, predicate) -> float:
    buckets: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        buckets[_sample_key(row)].append(bool(predicate(row)))
    if not buckets:
        return 0.0
    return mean(mean(1.0 if value else 0.0 for value in values) for values in buckets.values())


def _metric_row(rows: list[dict]) -> dict:
    query_tokens = [_as_float(row.get("query_tokens", 0), 0.0) for row in rows]
    search_ms = [_as_float(row.get("search_ms", 0), 0.0) for row in rows]
    e2e_ms = [_as_float(row.get("e2e_ms", 0), 0.0) for row in rows]
    sample_keys = {_sample_key(row) for row in rows}
    return {
        "n": len(sample_keys),
        "row_n": len(rows),
        "accuracy": round(_sample_clustered_rate(rows, predicate=lambda row: row.get("correct")), 6),
        "verified_accuracy": round(
            _sample_clustered_rate(rows, predicate=lambda row: row.get("correct") and _verified(row)),
            6,
        ),
        "query_tokens_median": round(median(query_tokens), 6) if query_tokens else 0.0,
        "query_tokens_mean": round(mean(query_tokens), 6) if query_tokens else 0.0,
        "search_p95_ms": round(_percentile(search_ms, 95), 6),
        "e2e_p50_ms": round(_percentile(e2e_ms, 50), 6),
        "datasets": sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))}),
        "categories": sorted({str(row.get("category", "")) for row in rows if str(row.get("category", ""))}),
        "runs": sorted({int(row.get("run_idx", 0) or 0) for row in rows}),
    }


def _artifact_ref(role: str, out_dir: Path) -> dict:
    manifest = _load_manifest(out_dir)
    return {
        "role": role,
        "path": str(Path(out_dir)),
        "split": str(manifest.get("split", "") or ""),
        "systems": str(manifest.get("systems", "") or ""),
        "sample_count": int(manifest.get("sample_count", 0) or 0),
        "log_fingerprint": log_fingerprint(out_dir),
    }


def build_ablation_report(
    full_dir: Path,
    metabolism_off_dir: Path,
    forgetting_off_dir: Path,
    affect_off_dir: Path | None = None,
    regions_off_dir: Path | None = None,
    *,
    system: str = "eidetic-plus-full",
    split: str = "dev",
    min_samples: int = 20,
    min_metabolism_accuracy_delta_pp: float = 5.0,
    min_region_accuracy_delta_pp: float = 2.0,
    min_affect_accuracy_delta_pp: float = 2.0,
    min_forgetting_cost_ratio: float = 1.05,
    max_forgetting_accuracy_regression_pp: float = 1.0,
) -> dict:
    full_dir = Path(full_dir)
    metabolism_off_dir = Path(metabolism_off_dir)
    forgetting_off_dir = Path(forgetting_off_dir)
    affect_off_dir = Path(affect_off_dir) if affect_off_dir is not None else None
    regions_off_dir = Path(regions_off_dir) if regions_off_dir is not None else None
    artifacts = {
        "full": full_dir,
        "metabolism_off": metabolism_off_dir,
        "regions_off": regions_off_dir,
        "forgetting_off": forgetting_off_dir,
        "affect_off": affect_off_dir,
    }
    failures: list[str] = []

    rows_by_role: dict[str, list[dict]] = {}
    manifests_by_role: dict[str, dict] = {}
    for role, out_dir in artifacts.items():
        if out_dir is None:
            failures.append(f"{role}:artifact_missing")
            rows_by_role[role] = []
            manifests_by_role[role] = {}
            continue
        manifest = _load_manifest(out_dir)
        manifests_by_role[role] = manifest
        got_split = str(manifest.get("split", "") or "").strip().lower()
        if got_split != split:
            failures.append(f"{role}:split:{got_split or '<missing>'}:expected:{split}")
        rows, row_failures = _rows_for_system(out_dir, system)
        failures.extend(f"{role}:{failure}" for failure in row_failures)
        rows_by_role[role] = rows
    failures.extend(_affect_profile_failures(manifests_by_role))
    failures.extend(_region_profile_failures(manifests_by_role))

    key_sets = {
        role: {_row_key(row) for row in rows}
        for role, rows in rows_by_role.items()
    }
    full_keys = key_sets.get("full", set())
    for role in ("metabolism_off", "regions_off", "forgetting_off", "affect_off"):
        missing = full_keys - key_sets.get(role, set())
        extra = key_sets.get(role, set()) - full_keys
        if missing or extra:
            failures.append(f"{role}:unpaired_rows:missing={len(missing)},extra={len(extra)}")

    full = _metric_row(rows_by_role.get("full", []))
    metabolism = _metric_row(rows_by_role.get("metabolism_off", []))
    regions = _metric_row(rows_by_role.get("regions_off", []))
    forgetting = _metric_row(rows_by_role.get("forgetting_off", []))
    affect = _metric_row(rows_by_role.get("affect_off", []))

    for role, metrics in (
        ("full", full),
        ("metabolism_off", metabolism),
        ("regions_off", regions),
        ("forgetting_off", forgetting),
        ("affect_off", affect),
    ):
        if int(metrics.get("n", 0) or 0) < min_samples:
            failures.append(f"{role}:n:{metrics.get('n', 0)}<required:{min_samples}")

    metabolism_delta_pp = (full["verified_accuracy"] - metabolism["verified_accuracy"]) * 100.0
    region_delta_pp = (full["verified_accuracy"] - regions["verified_accuracy"]) * 100.0
    affect_delta_pp = (full["verified_accuracy"] - affect["verified_accuracy"]) * 100.0
    forgetting_accuracy_regression_pp = max(
        0.0,
        (forgetting["verified_accuracy"] - full["verified_accuracy"]) * 100.0,
    )
    forgetting_cost_ratio = (
        forgetting["query_tokens_median"] / full["query_tokens_median"]
        if full["query_tokens_median"] > 0 else 0.0
    )
    if metabolism_delta_pp < min_metabolism_accuracy_delta_pp:
        failures.append(
            f"metabolism_delta_pp:{metabolism_delta_pp:.2f}<required:"
            f"{min_metabolism_accuracy_delta_pp:.2f}"
        )
    if region_delta_pp < min_region_accuracy_delta_pp:
        failures.append(
            f"region_delta_pp:{region_delta_pp:.2f}<required:"
            f"{min_region_accuracy_delta_pp:.2f}"
        )
    if affect_delta_pp < min_affect_accuracy_delta_pp:
        failures.append(
            f"affect_delta_pp:{affect_delta_pp:.2f}<required:"
            f"{min_affect_accuracy_delta_pp:.2f}"
        )
    if forgetting_cost_ratio < min_forgetting_cost_ratio:
        failures.append(
            f"forgetting_cost_ratio:{forgetting_cost_ratio:.3f}<required:"
            f"{min_forgetting_cost_ratio:.3f}"
        )
    if forgetting_accuracy_regression_pp > max_forgetting_accuracy_regression_pp:
        failures.append(
            f"forgetting_accuracy_regression_pp:{forgetting_accuracy_regression_pp:.2f}>allowed:"
            f"{max_forgetting_accuracy_regression_pp:.2f}"
        )

    refs = [
        _artifact_ref(role, out_dir)
        for role, out_dir in artifacts.items()
        if out_dir is not None
    ]
    return {
        "pass": not failures,
        "status": "PASS" if not failures else "FAIL",
        "generated_by": "bench.build_ablation_report",
        "system": system,
        "split": split,
        "full": full,
        "ablations": {
            "metabolism_off": metabolism,
            "regions_off": regions,
            "forgetting_off": forgetting,
            "affect_off": affect,
        },
        "deltas": {
            "metabolism_delta_pp": round(metabolism_delta_pp, 6),
            "region_delta_pp": round(region_delta_pp, 6),
            "affect_delta_pp": round(affect_delta_pp, 6),
            "forgetting_cost_ratio": round(forgetting_cost_ratio, 6),
            "forgetting_accuracy_regression_pp": round(forgetting_accuracy_regression_pp, 6),
        },
        "paired_coverage": {
            "row_keys": len(full_keys),
            "sample_keys": len({_sample_key(row) for row in rows_by_role.get("full", [])}),
            "exact_row_keys": not any("unpaired_rows" in failure for failure in failures),
        },
        "artifact_fingerprints": refs,
        "failures": failures,
        "thresholds": {
            "min_samples": min_samples,
            "min_metabolism_accuracy_delta_pp": min_metabolism_accuracy_delta_pp,
            "min_region_accuracy_delta_pp": min_region_accuracy_delta_pp,
            "min_affect_accuracy_delta_pp": min_affect_accuracy_delta_pp,
            "min_forgetting_cost_ratio": min_forgetting_cost_ratio,
            "max_forgetting_accuracy_regression_pp": max_forgetting_accuracy_regression_pp,
        },
    }


def write_report(report: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ablation_report.json from real benchmark logs")
    ap.add_argument("--full", required=True, help="artifact dir for the full memory system")
    ap.add_argument("--metabolism-off", required=True,
                    help="artifact dir with memory/consolidation/metabolism disabled")
    ap.add_argument("--regions-off", required=True,
                    help="artifact dir with memory-region/gist routing disabled")
    ap.add_argument("--forgetting-off", required=True,
                    help="artifact dir with priority forgetting disabled")
    ap.add_argument("--affect-off", required=True,
                    help="artifact dir with affective salience disabled")
    ap.add_argument("--system", default="eidetic-plus-full")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="",
                    help="report path; default writes ablation_report.json into --full")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--min-metabolism-accuracy-delta-pp", type=float, default=5.0)
    ap.add_argument("--min-region-accuracy-delta-pp", type=float, default=2.0)
    ap.add_argument("--min-affect-accuracy-delta-pp", type=float, default=2.0)
    ap.add_argument("--min-forgetting-cost-ratio", type=float, default=1.05)
    ap.add_argument("--max-forgetting-accuracy-regression-pp", type=float, default=1.0)
    args = ap.parse_args()

    report = build_ablation_report(
        Path(args.full),
        Path(args.metabolism_off),
        Path(args.forgetting_off),
        Path(args.affect_off),
        Path(args.regions_off),
        system=args.system,
        split=args.split,
        min_samples=args.min_samples,
        min_metabolism_accuracy_delta_pp=args.min_metabolism_accuracy_delta_pp,
        min_region_accuracy_delta_pp=args.min_region_accuracy_delta_pp,
        min_affect_accuracy_delta_pp=args.min_affect_accuracy_delta_pp,
        min_forgetting_cost_ratio=args.min_forgetting_cost_ratio,
        max_forgetting_accuracy_regression_pp=args.max_forgetting_accuracy_regression_pp,
    )
    out = Path(args.out) if args.out else Path(args.full) / "ablation_report.json"
    write_report(report, out)
    print(f"Ablation report: {report['status']} -> {out}")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
