"""Run a reproducible dev-split ablation bundle and build ``ablation_report.json``.

This is the producer-side companion to ``bench.build_ablation_report``. It keeps the measured
artifacts comparable by pinning one dev samples file and running full memory, metabolism-off,
regions-off, forgetting-off, and affect-off into separate directories before building the
release-gate sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .build_ablation_report import build_ablation_report, write_report
from .run import load_samples


METABOLISM_OFF_ENV = {
    "FULL_SLEEP": "0",
    "GIST_CHANNEL": "0",
    "COACTIVATION_CHANNEL": "0",
    "STRUCT_CHANNEL": "0",
    "EVENT_RANKING": "0",
    "EVENT_CHAIN_CONTEXT": "0",
    "GRAPH_VOCAB_SEEDING": "0",
    "EXTRACT_CHUNKING": "0",
    "MEMORY_TYPING": "0",
    "PREF_SENTENCE_SCAN": "0",
    "TEMPORAL_RERANK": "0",
    "CONFLICT_RESOLVER": "0",
}
FORGETTING_OFF_ENV = {
    # These are reversible index/context cost controls, not raw-record deletion. If a release run
    # uses a stronger pruning profile, pass it via --full-env and keep the off profile here at zero.
    "SALIENCE_PRUNE_THRESHOLD": "0",
    "DREAM_PRUNE_PERCENTILE": "0",
}
AFFECT_OFF_ENV = {
    "AFFECT_SALIENCE": "0",
}
REGIONS_OFF_ENV = {
    # Memory-region/cocoon routing is implemented through RAPTOR gist centroids. This ablation keeps
    # the rest of metabolism on so the measured delta is specific to region routing.
    "GIST_CHANNEL": "0",
}
FORGETTING_COST_DEFAULTS = {
    # Keep these in sync with eidetic.config.Settings. They are duplicated here deliberately so the
    # runner can validate the child-process environment without importing config and mutating
    # os.environ via METABOLISM_MODE overlays.
    "SALIENCE_PRUNE_THRESHOLD": "0.0",
    "DREAM_PRUNE_PERCENTILE": "5.0",
}
FORGETTING_COST_KEYS = tuple(FORGETTING_COST_DEFAULTS)


@dataclass(frozen=True)
class AblationRunSpec:
    role: str
    out_dir: str
    data_dir: str
    command: list[str]
    env_overrides: dict[str, str]


def _parse_env_pairs(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values or []:
        parts = [part.strip() for part in str(raw).split(",") if part.strip()]
        for part in parts:
            if "=" not in part:
                raise ValueError(f"env override must look like KEY=VALUE, got {part!r}")
            key, value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"env override has an empty key: {part!r}")
            out[key] = value.strip()
    return out


def _write_dev_samples_file(path: Path, *, dataset: str, subset: int, variant: str,
                            sample_offset: int, sample_strategy: str) -> Path:
    samples = load_samples(
        dataset,
        subset,
        variant,
        sample_offset,
        split="dev",
        sample_strategy=sample_strategy,
    )
    if not samples:
        raise ValueError("no dev samples loaded for ablation run")
    rows = [{"dataset": sample.dataset, "sample_id": sample.sample_id} for sample in samples]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return path


def _run_command(spec: AblationRunSpec, env: dict[str, str], cwd: Path) -> int:
    proc = subprocess.run(spec.command, env=env, cwd=cwd, check=False)
    return int(proc.returncode)


def _cost_float(key: str, value: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric for forgetting-cost ablation, got {value!r}") from exc


def _effective_forgetting_cost_profile(spec: AblationRunSpec,
                                       parent_env: dict[str, str] | None = None) -> dict[str, float]:
    parent = os.environ if parent_env is None else parent_env
    profile: dict[str, float] = {}
    for key, default in FORGETTING_COST_DEFAULTS.items():
        raw = spec.env_overrides.get(key, parent.get(key, default))
        profile[key] = _cost_float(key, raw)
    return profile


def _forgetting_cost_profiles(specs: list[AblationRunSpec],
                              parent_env: dict[str, str] | None = None) -> dict[str, dict[str, float]]:
    by_role = {spec.role: spec for spec in specs}
    return {
        role: _effective_forgetting_cost_profile(by_role[role], parent_env)
        for role in ("full", "forgetting_off")
        if role in by_role
    }


def _forgetting_profile_failures(specs: list[AblationRunSpec],
                                 parent_env: dict[str, str] | None = None) -> list[str]:
    try:
        profiles = _forgetting_cost_profiles(specs, parent_env)
    except ValueError as exc:
        return [f"forgetting_off:invalid_cost_profile:{exc}"]
    if "full" not in profiles or "forgetting_off" not in profiles:
        return ["forgetting_off:missing_full_or_forgetting_off_run_spec"]
    full = profiles["full"]
    off = profiles["forgetting_off"]
    inverted = [key for key in FORGETTING_COST_KEYS if off[key] > full[key]]
    if inverted:
        details = ",".join(f"{key}:full={full[key]:g},off={off[key]:g}" for key in inverted)
        return [f"forgetting_off:inverted_cost_profile:{details}"]
    stronger_full = [key for key in FORGETTING_COST_KEYS if full[key] > off[key]]
    if not stronger_full:
        details = ",".join(f"{key}={full[key]:g}" for key in FORGETTING_COST_KEYS)
        keys = ",".join(FORGETTING_COST_KEYS)
        return [f"forgetting_off:identical_cost_profile:{details}; set a larger full value for one of {keys}"]
    return []


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _effective_env_value(spec: AblationRunSpec, key: str,
                         parent_env: dict[str, str] | None = None) -> str:
    parent = os.environ if parent_env is None else parent_env
    return str(spec.env_overrides.get(key, parent.get(key, "")))


def _affect_profile_failures(specs: list[AblationRunSpec],
                             parent_env: dict[str, str] | None = None) -> list[str]:
    by_role = {spec.role: spec for spec in specs}
    missing = [role for role in ("full", "metabolism_off", "regions_off", "forgetting_off", "affect_off")
               if role not in by_role]
    if missing:
        return [f"affect_off:missing_run_spec:{','.join(missing)}"]

    failures: list[str] = []
    for role in ("full", "metabolism_off", "regions_off", "forgetting_off"):
        value = _effective_env_value(by_role[role], "AFFECT_SALIENCE", parent_env)
        if not _truthy(value):
            failures.append(f"{role}:AFFECT_SALIENCE:{value or '<missing>'}:expected:on")
    off_value = _effective_env_value(by_role["affect_off"], "AFFECT_SALIENCE", parent_env)
    if _truthy(off_value):
        failures.append(f"affect_off:AFFECT_SALIENCE:{off_value}:expected:off")
    return failures


def _region_profile_failures(specs: list[AblationRunSpec],
                             parent_env: dict[str, str] | None = None) -> list[str]:
    by_role = {spec.role: spec for spec in specs}
    missing = [role for role in ("full", "regions_off", "forgetting_off", "affect_off")
               if role not in by_role]
    if missing:
        return [f"regions_off:missing_run_spec:{','.join(missing)}"]

    failures: list[str] = []
    for role in ("full", "forgetting_off", "affect_off"):
        value = _effective_env_value(by_role[role], "GIST_CHANNEL", parent_env)
        if not _truthy(value):
            failures.append(f"{role}:GIST_CHANNEL:{value or '<missing>'}:expected:on")
    off_value = _effective_env_value(by_role["regions_off"], "GIST_CHANNEL", parent_env)
    if _truthy(off_value):
        failures.append(f"regions_off:GIST_CHANNEL:{off_value}:expected:off")
    return failures


def build_run_specs(*, out_root: Path, samples_file: Path, systems: str, dataset: str,
                    variant: str, runs: int, overwrite: bool,
                    common_env: dict[str, str] | None = None,
                    full_env: dict[str, str] | None = None,
                    metabolism_off_env: dict[str, str] | None = None,
                    regions_off_env: dict[str, str] | None = None,
                    forgetting_off_env: dict[str, str] | None = None,
                    affect_off_env: dict[str, str] | None = None) -> list[AblationRunSpec]:
    out_root = Path(out_root)
    samples_file = Path(samples_file)
    base_env = {
        "METABOLISM_MODE": "1",
        "AFFECT_SALIENCE": "1",
        "GIST_CHANNEL": "1",
        **(common_env or {}),
    }
    role_envs = {
        "full": {**base_env, **(full_env or {})},
        "metabolism_off": {**base_env, **METABOLISM_OFF_ENV, **(metabolism_off_env or {})},
        "regions_off": {**base_env, **REGIONS_OFF_ENV, **(regions_off_env or {})},
        "forgetting_off": {**base_env, **FORGETTING_OFF_ENV, **(forgetting_off_env or {})},
        "affect_off": {**base_env, **AFFECT_OFF_ENV, **(affect_off_env or {})},
    }
    specs: list[AblationRunSpec] = []
    for role, env_overrides in role_envs.items():
        out_dir = out_root / role
        data_dir = out_root / f"data_{role}"
        command = [
            sys.executable,
            "-m",
            "bench.run",
            "--systems",
            systems,
            "--dataset",
            dataset,
            "--variant",
            variant,
            "--samples-file",
            str(samples_file),
            "--split",
            "dev",
            "--holdout-profile",
            "dev",
            "--runs",
            str(runs),
            "--out",
            str(out_dir),
        ]
        if overwrite:
            command.append("--overwrite")
        specs.append(AblationRunSpec(
            role=role,
            out_dir=str(out_dir),
            data_dir=str(data_dir),
            command=command,
            env_overrides={**env_overrides, "DATA_DIR": str(data_dir)},
        ))
    return specs


def run_dev_ablation(*, out_root: Path, report_out: Path, systems: str = "eidetic-full",
                     system_under_test: str = "eidetic-plus-full", dataset: str = "both",
                     subset: int = 24, variant: str = "longmemeval_s", sample_offset: int = 0,
                     sample_strategy: str = "stratified", samples_file: Path | None = None,
                     runs: int = 1, overwrite: bool = False,
                     common_env: dict[str, str] | None = None,
                     full_env: dict[str, str] | None = None,
                     metabolism_off_env: dict[str, str] | None = None,
                     regions_off_env: dict[str, str] | None = None,
                     forgetting_off_env: dict[str, str] | None = None,
                     affect_off_env: dict[str, str] | None = None,
                     min_samples: int = 20,
                     min_metabolism_accuracy_delta_pp: float = 5.0,
                     min_region_accuracy_delta_pp: float = 2.0,
                     min_affect_accuracy_delta_pp: float = 2.0,
                     min_forgetting_cost_ratio: float = 1.05,
                     max_forgetting_accuracy_regression_pp: float = 1.0,
                     command_runner: Callable[[AblationRunSpec, dict[str, str], Path], int] = _run_command) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if samples_file is None:
        samples_file = out_root / "dev_ablation.samples.json"
        _write_dev_samples_file(
            samples_file,
            dataset=dataset,
            subset=subset,
            variant=variant,
            sample_offset=sample_offset,
            sample_strategy=sample_strategy,
        )
    specs = build_run_specs(
        out_root=out_root,
        samples_file=Path(samples_file),
        systems=systems,
        dataset=dataset,
        variant=variant,
        runs=runs,
        overwrite=overwrite,
        common_env=common_env,
        full_env=full_env,
        metabolism_off_env=metabolism_off_env,
        regions_off_env=regions_off_env,
        forgetting_off_env=forgetting_off_env,
        affect_off_env=affect_off_env,
    )
    try:
        forgetting_cost_profiles = _forgetting_cost_profiles(specs)
    except ValueError:
        forgetting_cost_profiles = {}
    preflight_failures = [
        *_forgetting_profile_failures(specs),
        *_affect_profile_failures(specs),
        *_region_profile_failures(specs),
    ]
    if preflight_failures:
        report = {
            "pass": False,
            "status": "FAIL",
            "generated_by": "bench.run_dev_ablation",
            "system": system_under_test,
            "split": "dev",
            "failures": preflight_failures,
            "run_specs": [asdict(spec) for spec in specs],
            "forgetting_cost_profiles": forgetting_cost_profiles,
            "samples_file": str(samples_file),
        }
        write_report(report, Path(report_out))
        return report
    failures: list[str] = []
    cwd = Path(__file__).resolve().parents[1]
    for spec in specs:
        env = os.environ.copy()
        env.update(spec.env_overrides)
        rc = command_runner(spec, env, cwd)
        if rc != 0:
            failures.append(f"{spec.role}:bench.run exited {rc}")
    if failures:
        report = {
            "pass": False,
            "status": "FAIL",
            "generated_by": "bench.run_dev_ablation",
            "system": system_under_test,
            "split": "dev",
            "failures": failures,
            "run_specs": [asdict(spec) for spec in specs],
            "forgetting_cost_profiles": forgetting_cost_profiles,
        }
    else:
        by_role = {spec.role: Path(spec.out_dir) for spec in specs}
        report = build_ablation_report(
            by_role["full"],
            by_role["metabolism_off"],
            by_role["forgetting_off"],
            by_role["affect_off"],
            by_role["regions_off"],
            system=system_under_test,
            split="dev",
            min_samples=min_samples,
            min_metabolism_accuracy_delta_pp=min_metabolism_accuracy_delta_pp,
            min_region_accuracy_delta_pp=min_region_accuracy_delta_pp,
            min_affect_accuracy_delta_pp=min_affect_accuracy_delta_pp,
            min_forgetting_cost_ratio=min_forgetting_cost_ratio,
            max_forgetting_accuracy_regression_pp=max_forgetting_accuracy_regression_pp,
        )
        report["generated_by"] = "bench.run_dev_ablation"
        report["run_specs"] = [asdict(spec) for spec in specs]
        report["forgetting_cost_profiles"] = forgetting_cost_profiles
        report["samples_file"] = str(samples_file)
    write_report(report, Path(report_out))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Run dev ablations and build ablation_report.json")
    ap.add_argument("--out-root", default="artifacts/dev_ablation")
    ap.add_argument("--report-out", default="",
                    help="default: <out-root>/ablation_report.json")
    ap.add_argument("--systems", default="eidetic-full")
    ap.add_argument("--system-under-test", default="eidetic-plus-full")
    ap.add_argument("--dataset", default="both",
                    choices=["longmemeval", "locomo", "memoryagentbench", "beam", "both", "all"])
    ap.add_argument("--subset", type=int, default=24)
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--sample-strategy", default="stratified",
                    choices=["contiguous", "stratified"])
    ap.add_argument("--samples-file", default="",
                    help="existing JSON list of {dataset, sample_id}; skips dev sample generation")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--common-env", action="append", default=[],
                    help="KEY=VALUE override applied to all ablation runs; may be comma-separated")
    ap.add_argument("--full-env", action="append", default=[],
                    help="KEY=VALUE override applied only to the full run")
    ap.add_argument("--metabolism-off-env", action="append", default=[],
                    help="KEY=VALUE override applied after the default metabolism-off profile")
    ap.add_argument("--regions-off-env", action="append", default=[],
                    help="KEY=VALUE override applied after the default regions-off profile")
    ap.add_argument("--forgetting-off-env", action="append", default=[],
                    help="KEY=VALUE override applied after the default forgetting-off profile")
    ap.add_argument("--affect-off-env", action="append", default=[],
                    help="KEY=VALUE override applied after the default affect-off profile")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--min-metabolism-accuracy-delta-pp", type=float, default=5.0)
    ap.add_argument("--min-region-accuracy-delta-pp", type=float, default=2.0)
    ap.add_argument("--min-affect-accuracy-delta-pp", type=float, default=2.0)
    ap.add_argument("--min-forgetting-cost-ratio", type=float, default=1.05)
    ap.add_argument("--max-forgetting-accuracy-regression-pp", type=float, default=1.0)
    args = ap.parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    report_out = Path(args.report_out) if args.report_out else Path(args.out_root) / "ablation_report.json"
    report = run_dev_ablation(
        out_root=Path(args.out_root),
        report_out=report_out,
        systems=args.systems,
        system_under_test=args.system_under_test,
        dataset=args.dataset,
        subset=args.subset,
        variant=args.variant,
        sample_offset=args.sample_offset,
        sample_strategy=args.sample_strategy,
        samples_file=Path(args.samples_file) if args.samples_file else None,
        runs=args.runs,
        overwrite=args.overwrite,
        common_env=_parse_env_pairs(args.common_env),
        full_env=_parse_env_pairs(args.full_env),
        metabolism_off_env=_parse_env_pairs(args.metabolism_off_env),
        regions_off_env=_parse_env_pairs(args.regions_off_env),
        forgetting_off_env=_parse_env_pairs(args.forgetting_off_env),
        affect_off_env=_parse_env_pairs(args.affect_off_env),
        min_samples=args.min_samples,
        min_metabolism_accuracy_delta_pp=args.min_metabolism_accuracy_delta_pp,
        min_region_accuracy_delta_pp=args.min_region_accuracy_delta_pp,
        min_affect_accuracy_delta_pp=args.min_affect_accuracy_delta_pp,
        min_forgetting_cost_ratio=args.min_forgetting_cost_ratio,
        max_forgetting_accuracy_regression_pp=args.max_forgetting_accuracy_regression_pp,
    )
    print(f"Dev ablation report: {report['status']} -> {report_out}")
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
