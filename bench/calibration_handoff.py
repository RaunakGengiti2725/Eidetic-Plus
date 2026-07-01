"""Publish dev-split calibration into a public benchmark artifact.

The release gate intentionally refuses an enabled ABSTENTION_V2 threshold unless the public
artifact contains the dev calibration report that produced it. This module makes that handoff
explicit and testable instead of relying on a human to copy a JSON file and type an env var.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


class CalibrationHandoffError(ValueError):
    """Raised when a calibration report is not safe to publish."""


def _load_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CalibrationHandoffError(f"calibration report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationHandoffError(f"calibration report is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise CalibrationHandoffError("calibration report must be a JSON object")
    return data


def _finite_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationHandoffError(f"{field} must be numeric") from exc
    if not math.isfinite(out):
        raise CalibrationHandoffError(f"{field} must be finite")
    return out


def validate_abstention_v2_tau_report(
    report: dict[str, Any],
    *,
    expected_system: str | None = "eidetic-plus-full",
    min_samples: int = 1,
    min_target: float = 0.95,
) -> dict[str, Any]:
    """Validate the calibration report shape used by the public release gate."""
    if not report.get("ok"):
        raise CalibrationHandoffError("calibration report is not ok")
    if report.get("method") != "abstention_v2_tau":
        raise CalibrationHandoffError("calibration method must be abstention_v2_tau")
    if report.get("split") != "dev":
        raise CalibrationHandoffError("calibration split must be dev")
    if expected_system is not None and report.get("system") != expected_system:
        raise CalibrationHandoffError(
            f"calibration system must be {expected_system}, got {report.get('system')!r}"
        )

    tau = _finite_float(report.get("tau"), "tau")
    if tau < 0:
        raise CalibrationHandoffError("tau must be non-negative")
    try:
        n = int(report.get("n", 0))
    except (TypeError, ValueError) as exc:
        raise CalibrationHandoffError("n must be an integer") from exc
    if n < min_samples:
        raise CalibrationHandoffError(f"n must be >= {min_samples}, got {n}")

    target = _finite_float(report.get("target"), "target")
    if target < min_target:
        raise CalibrationHandoffError(
            f"target precision must be >= {min_target}, got {target}"
        )
    fingerprint = report.get("log_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("combined_sha256"):
        raise CalibrationHandoffError("log_fingerprint.combined_sha256 is required")
    return report


def env_vars_for_report(report: dict[str, Any]) -> dict[str, str]:
    """Return the env vars that make a benchmark manifest match this report."""
    return {
        "ABSTENTION_V2": "1",
        "ABSTENTION_V2_TAU": str(report["tau"]),
    }


def write_env_file(env_vars: dict[str, str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key}={value}\n" for key, value in sorted(env_vars.items()))
    path.write_text(body)
    return path


def copy_abstention_v2_tau_report(
    calibration_report: Path,
    out_dir: Path,
    *,
    env_out: Path | None = None,
    expected_system: str | None = "eidetic-plus-full",
    min_samples: int = 1,
    min_target: float = 0.95,
) -> dict[str, Any]:
    """Validate and copy ABSTENTION_V2_TAU calibration into a release artifact directory."""
    src = Path(calibration_report)
    report = validate_abstention_v2_tau_report(
        _load_report(src),
        expected_system=expected_system,
        min_samples=min_samples,
        min_target=min_target,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "abstention_v2_tau.json"
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)

    env_vars = env_vars_for_report(report)
    written_env = write_env_file(env_vars, Path(env_out)) if env_out else None
    return {
        "report": report,
        "dest": dest,
        "env_out": written_env,
        "env_vars": env_vars,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate/copy dev ABSTENTION_V2 calibration into a public artifact."
    )
    ap.add_argument("--calibration", required=True,
                    help="path to artifacts/cal_dev/abstention_v2_tau.json")
    ap.add_argument("--out", required=True,
                    help="public benchmark artifact directory")
    ap.add_argument("--env-out",
                    help="optional env file to write with ABSTENTION_V2 and ABSTENTION_V2_TAU")
    ap.add_argument("--system", default="eidetic-plus-full",
                    help="expected calibrated system; use empty string to skip this check")
    ap.add_argument("--min-samples", type=int, default=1,
                    help="minimum calibration rows required for handoff validation")
    ap.add_argument("--min-target", type=float, default=0.95,
                    help="minimum precision target required for handoff validation")
    args = ap.parse_args(argv)

    expected_system = args.system or None
    try:
        result = copy_abstention_v2_tau_report(
            Path(args.calibration),
            Path(args.out),
            env_out=Path(args.env_out) if args.env_out else None,
            expected_system=expected_system,
            min_samples=args.min_samples,
            min_target=args.min_target,
        )
    except CalibrationHandoffError as exc:
        print(f"Calibration handoff failed: {exc}", file=sys.stderr)
        return 2

    print(f"Copied ABSTENTION_V2 calibration -> {result['dest']}")
    if result["env_out"]:
        print(f"Wrote benchmark env -> {result['env_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
