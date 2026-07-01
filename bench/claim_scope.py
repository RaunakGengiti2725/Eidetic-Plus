"""Generate an honest public-claim scope report for benchmark artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import load_logs


def _load_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _unique_log_sample_count(rows: list[dict]) -> int:
    return len({
        (
            str(r.get("dataset", "")),
            str(r.get("sample_id", "")),
            str(r.get("category", "")),
        )
        for r in rows
        if r.get("dataset") and r.get("sample_id")
    })


def build_claim_scope(
    out_dir: Path,
    *,
    public_claim_scope: str = "measured-harness-only",
    measured_external_systems: list[str] | None = None,
    external_system_evidence: list[dict] | None = None,
    top_systems_for_sota: list[str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    rows = load_logs(out_dir)
    manifest = _load_manifest(out_dir)
    systems = sorted({str(r.get("system", "")) for r in rows if r.get("system")})
    datasets = sorted({str(r.get("dataset", "")) for r in rows if r.get("dataset")})
    runs = sorted({int(r.get("run_idx", 0)) for r in rows if r.get("run_idx") is not None})
    errors = [r for r in rows if r.get("error")]
    log_sample_count = _unique_log_sample_count(rows)
    top_systems_for_sota = top_systems_for_sota or ["chronos", "mastra", "byterover", "hindsight"]
    measured_external_systems = measured_external_systems or []
    external_system_evidence = external_system_evidence or []
    evidence_systems = [
        str(item.get("system", "")).strip()
        for item in external_system_evidence
        if isinstance(item, dict) and str(item.get("system", "")).strip()
    ]

    limitations: list[str] = []
    if public_claim_scope != "measured-harness-only":
        limitations.append(
            "Scope is not measured-harness-only; public language must match measured systems."
        )
    if len(runs) < 10:
        limitations.append(f"Only {len(runs)} run(s) are present; release claims require multi-run evidence.")
    manifest_sample_count = int(manifest.get("sample_count", 0) or 0)
    effective_sample_count = manifest_sample_count or log_sample_count
    if effective_sample_count < 100:
        source = "manifest" if manifest_sample_count else "logs"
        limitations.append(
            f"Only {effective_sample_count} unique benchmark sample(s) are present in the {source}."
        )
    if manifest.get("split") != "test":
        limitations.append(f"Manifest split is {manifest.get('split', '<missing>')}; public claims require test.")
    missing_top = [
        s for s in top_systems_for_sota
        if s.lower() not in {x.lower() for x in systems + evidence_systems}
    ]
    if missing_top:
        limitations.append(
            "Not a SOTA/best-in-world claim; missing top-system measurements: "
            + ", ".join(missing_top)
            + "."
        )
    for baseline in ("mem0", "graphiti"):
        if baseline not in {s.lower() for s in systems}:
            limitations.append(f"{baseline} was not measured as a healthy in-harness competitor.")
    if errors:
        limitations.append(f"{len(errors)} row(s) contain runtime errors and must be excluded.")
    if not limitations:
        limitations.append(
            "Claim is limited to the measured harness systems, datasets, split, and run configuration."
        )

    return {
        "public_claim_scope": public_claim_scope,
        "measured_harness_systems": systems,
        "measured_external_systems": measured_external_systems,
        "external_system_evidence": external_system_evidence,
        "datasets": datasets,
        "runs": runs,
        "manifest_subset": manifest.get("subset"),
        "manifest_sample_count": manifest.get("sample_count"),
        "log_sample_count": log_sample_count,
        "effective_sample_count": effective_sample_count,
        "manifest_split": manifest.get("split"),
        "limitations": limitations,
    }


def write_claim_scope(out_dir: Path, report: dict, report_out: str = "claim_scope.json") -> Path:
    path = Path(out_dir) / report_out
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def _load_external_evidence(path: str) -> list[dict]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("--external-evidence-json must point at a JSON list of evidence objects")
    return [item for item in data if isinstance(item, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate claim_scope.json for release_gate.")
    ap.add_argument("--out", default="artifacts/bench")
    ap.add_argument("--scope", default="measured-harness-only")
    ap.add_argument("--external-systems", default="")
    ap.add_argument("--external-evidence-json", default="")
    ap.add_argument("--top-systems-for-sota", default="chronos,mastra,byterover,hindsight")
    ap.add_argument("--report-out", default="claim_scope.json")
    args = ap.parse_args()

    report = build_claim_scope(
        Path(args.out),
        public_claim_scope=args.scope,
        measured_external_systems=_csv(args.external_systems),
        external_system_evidence=_load_external_evidence(args.external_evidence_json),
        top_systems_for_sota=_csv(args.top_systems_for_sota),
    )
    path = write_claim_scope(Path(args.out), report, args.report_out)
    print(f"Claim scope -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
