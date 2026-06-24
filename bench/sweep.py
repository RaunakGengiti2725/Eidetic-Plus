"""Coordinate-descent parameter sweep on a small benchmark subset.

The sweep tunes one benchmark-visible knob at a time. `--dry-run` enumerates the
plan and token estimate only, with no scoring and no invented numbers. A live run
requires a funded key, writes raw logs per trial, renders a scoreboard per trial,
and fails loudly if a trial produces no valid rows.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

# OtterTune-style knob blacklist: settings whose change forces an HNSW index rebuild
# must NEVER be flipped per-trial by an always-on tuner (a ruinous rebuild per arm).
# They are tuned only by the explicit offline rebuild path, never here. We ASSERT the
# stage grid is disjoint from this set so the exclusion is a documented mechanism, not
# an accident of omission.
REBUILD_KNOBS = {"HNSW_M", "HNSW_EF_CONSTRUCTION"}

# Coordinate-descent stages: (env var to set, candidate values). Two exclusions, both
# intentional: (1) product-only knobs such as READER_ROUTER / ABSTENTION_THRESHOLD that the
# neutral fixed-reader adapter does not exercise; (2) the REBUILD_KNOBS above (index rebuild).
# HNSW_EF_SEARCH is a query-time knob, so it stays.
STAGES = [
    ("READER_COT", ["0", "1"]),
    ("CONFLICT_RESOLVER", ["0", "1"]),
    ("COMPRESSION_RATIO", ["1.0", "0.75", "0.5"]),
    ("EXTRACT_LIGHT", ["0", "1"]),
    ("TEMPORAL_RERANK", ["0", "1"]),
    ("HIPPO2_SEEDING", ["0", "1"]),
    ("PERSISTENT_BM25", ["1", "0"]),
    ("DREAM_AB", ["0", "1"]),
    ("RERANK_ENABLED", ["1", "0"]),
    ("RERANK_DEPTH", ["50", "100"]),
    ("CONTEXT_TOKEN_BUDGET", ["8000", "6000", "3000"]),
    ("ANN_TOPK", ["100", "200"]),
    ("FINAL_TOPK", ["10", "15"]),
    ("RRF_W_BM25", ["0.6", "1.0"]),
    ("RRF_W_GRAPH", ["0.8", "1.2"]),
    ("RRF_W_RECENCY", ["0.3", "0.0"]),
    ("HNSW_EF_SEARCH", ["256", "500"]),
    # Layer-2 hot-path knobs (benchmark-visible via the neutral adapter's retrieve path).
    ("FUSION_METHOD", ["rrf", "dbsf"]),
    ("ADAPTIVE_K", ["0", "1"]),
    ("MMR_ENABLED", ["0", "1"]),
    ("RERANK_SKIP_MARGIN", ["0.0", "0.05"]),
    ("ADAPTIVE_EF", ["0", "1"]),
]

_DATASET_CHOICES = ["longmemeval", "locomo", "memoryagentbench", "beam", "both", "all"]
_REQUIRED_LOG_FIELDS = {"system", "dataset", "category", "sample_id", "correct", "run_idx"}

# Integrity wall: the sweep is an optimizer, so it may tune on the DEV split ONLY.
assert not (set(s for s, _ in STAGES) & REBUILD_KNOBS), \
    f"STAGES must not include rebuild knobs {REBUILD_KNOBS}"


def plan(subset: int, runs: int) -> list[dict]:
    """Return the ordered coordinate-descent trials."""
    trials = []
    for env_var, values in STAGES:
        for v in values:
            trials.append({"stage": env_var, "value": v, "subset": subset, "runs": runs})
    return trials


def estimate_tokens(n_trials: int, subset: int, runs: int) -> int:
    # Rough: each question ~= retrieval + reader + judge. This is only a budget warning.
    return n_trials * max(1, subset) * max(1, runs) * 6000


def _restore_env(original: dict[str, str | None]) -> None:
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _safe_value(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def stage_assignment(stage: str, value: str) -> dict[str, str]:
    if stage == "COMPRESSION_RATIO":
        if value == "1.0":
            return {"CONTEXT_COMPRESS": "0", "COMPRESSION_RATIO": "1.0"}
        return {"CONTEXT_COMPRESS": "1", "COMPRESSION_RATIO": value}
    return {stage: value}


def _stage_env_keys() -> set[str]:
    keys = set()
    for stage, values in STAGES:
        for value in values:
            keys.update(stage_assignment(stage, value))
    return keys


def _format_assignment(stage: str, value: str) -> str:
    return ", ".join(f"{key}={val}" for key, val in stage_assignment(stage, value).items())


def _validate_rows(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        raise RuntimeError(f"Sweep trial produced no rows in {out_dir}; refusing to score an empty run.")
    for i, row in enumerate(rows):
        missing = _REQUIRED_LOG_FIELDS - set(row)
        if missing:
            raise RuntimeError(
                f"Sweep trial row {i} in {out_dir} is missing fields: {sorted(missing)}"
            )


def _avg(items: list[float]) -> float | None:
    return mean(items) if items else None


def score_rows(rows: list[dict], system: str = "eidetic-plus") -> tuple[float, dict[str, dict[str, Any]]]:
    scoped = [r for r in rows if r.get("system") == system] or rows
    _validate_rows(scoped, Path("<memory>"))
    cats: dict[str, list[dict]] = {}
    for row in scoped:
        key = f"{row.get('dataset', 'unknown')}/{row.get('category', 'unknown')}"
        cats.setdefault(key, []).append(row)

    by_cat: dict[str, dict[str, Any]] = {}
    for key, vals in sorted(cats.items()):
        n = len(vals)
        by_cat[key] = {
            "accuracy": sum(1 for r in vals if r.get("correct")) / n,
            "n": n,
            "tokens_per_query": _avg([float(r["query_tokens"]) for r in vals if "query_tokens" in r]),
            "search_ms": _avg([float(r["search_ms"]) for r in vals if "search_ms" in r]),
            "e2e_ms": _avg([float(r["e2e_ms"]) for r in vals if "e2e_ms" in r]),
        }
    overall = sum(1 for r in scoped if r.get("correct")) / len(scoped)
    return overall, by_cat


def _format_categories(cats: dict[str, dict[str, Any]]) -> str:
    parts = []
    for name, item in cats.items():
        parts.append(
            f"{name}={item['accuracy']:.3f}(n={item['n']}, "
            f"tok={item['tokens_per_query']}, search_ms={item['search_ms']})"
        )
    return ", ".join(parts) if parts else "no rows"


def write_stage_comparison(
    control_dir: Path,
    experiment_dir: Path,
    *,
    stage: str,
    value: str,
    system: str = "eidetic-plus",
) -> tuple[Path, dict[str, Any]]:
    from .compare import compare_dirs, render_markdown

    result = compare_dirs(control_dir, experiment_dir, system=system)
    result["sweep_stage"] = stage
    result["sweep_value"] = value
    out_path = experiment_dir / "stage_ab.md"
    render_markdown(result, out_path)
    return out_path, result


def build_tpe_space() -> dict:
    """The TPE search space: every coordinate-descent stage becomes a categorical knob, so
    TPE searches the JOINT knob vector (modeling interactions coordinate descent misses)."""
    return {env: ("categorical", list(values)) for env, values in STAGES}


def _objectives_from_rows(rows: list[dict], system: str = "eidetic-plus") -> tuple[float, float, float]:
    """(accuracy, p95 end-to-end ms, mean tokens/query) for the multi-objective Pareto set."""
    import numpy as np
    scoped = [r for r in rows if r.get("system") == system] or rows
    acc = sum(1 for r in scoped if r.get("correct")) / len(scoped)
    e2e = [float(r["e2e_ms"]) for r in scoped if "e2e_ms" in r]
    tok = [float(r["query_tokens"]) for r in scoped if "query_tokens" in r]
    p95 = float(np.percentile(e2e, 95)) if e2e else 0.0
    return acc, p95, (mean(tok) if tok else 0.0)


def run_tpe_sweep(args, samples, judge, judge_desc) -> int:
    """Layer-1a/1b: a numpy TPE study over the joint knob space on the DEV split, recording
    the multi-objective Pareto set over (accuracy, p95 latency, tokens). Same fail-loud /
    no-fabrication discipline as the coordinate-descent path."""
    from eidetic.config import get_settings
    from eidetic.optim.pareto import pareto_front
    from eidetic.optim.tpe import TPESampler

    from . import run as bench_run
    from . import scoreboard
    from .harness import load_logs, run_system

    sampler = TPESampler(build_tpe_space(), n_startup=min(8, args.trials), seed=0)
    original_env = {k: os.environ.get(k) for k in _stage_env_keys()}
    trials_meta: list[dict] = []
    objectives: list[tuple] = []
    best: dict | None = None
    try:
        for t in range(args.trials):
            cfg = sampler.suggest()
            for env_var, value in cfg.items():
                for key, val in stage_assignment(env_var, value).items():
                    os.environ[key] = val
            get_settings.cache_clear()
            out_dir = Path(args.out) / f"tpe_trial{t}"
            for raw in args.systems.split(","):
                sysobj = bench_run.make_system(raw)
                run_system(sysobj, samples, judge, runs=args.runs, out_dir=out_dir,
                           run_offset=args.run_offset, overwrite=True)
            bench_run.write_manifest(out_dir, args, judge_desc, samples=samples)
            scoreboard.render(out_dir, judge_desc)
            rows = load_logs(out_dir)
            _validate_rows(rows, out_dir)
            acc, p95, mtok = _objectives_from_rows(rows)
            sampler.observe(cfg, -acc)                    # TPE minimizes loss = -accuracy
            objectives.append((-acc, p95, mtok))
            trials_meta.append({"trial": t, "config": dict(cfg), "accuracy": acc,
                                "p95_e2e_ms": p95, "tokens_per_query": mtok, "out_dir": str(out_dir)})
            if best is None or acc > best["accuracy"]:
                best = {"config": dict(cfg), "accuracy": acc, "p95_e2e_ms": p95,
                        "tokens_per_query": mtok}
            print(f"  TPE trial {t}: acc={acc:.3f} p95={p95:.0f}ms tok={mtok:.0f}")
    finally:
        _restore_env(original_env)
        get_settings.cache_clear()

    pareto = [trials_meta[i] for i in pareto_front(objectives)]
    best_env: dict[str, str] = {}
    if best:
        for env_var, value in best["config"].items():
            best_env.update(stage_assignment(env_var, value))
    out = Path(args.out) / "best_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sampler": "tpe", "split": "dev", "best_env": best_env,
                               "best": best, "pareto": pareto, "trials": trials_meta}, indent=2))
    print(f"\nTPE best -> {out}: {best_env}")
    print(f"Pareto front: {len(pareto)} configs (accuracy vs p95 latency vs tokens/query)")
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Coordinate-descent parameter sweep")
    ap.add_argument("--systems", default="eidetic")
    ap.add_argument("--dataset", default="locomo", choices=_DATASET_CHOICES)
    ap.add_argument("--subset", type=int, default=50)
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--run-offset", type=int, default=0)
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--out", default="artifacts/bench/sweep")
    ap.add_argument("--split", default="dev", choices=["dev"],
                    help="integrity wall: the sweep may tune on the DEV split ONLY. Reported "
                         "runs use --split test (bench.run / reproduce.sh). Locked to 'dev'.")
    ap.add_argument("--sampler", default="coord", choices=["coord", "tpe"],
                    help="coord = coordinate descent (default); tpe = numpy TPE study over the "
                         "joint knob space with a multi-objective Pareto set.")
    ap.add_argument("--trials", type=int, default=24, help="number of TPE trials (--sampler tpe)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing trial logs")
    ap.add_argument("--mem0-gate-out", help="real Mem0 gate log directory from a qwen-plus run")
    ap.add_argument("--mem0-gate-expected", help="published Mem0 LoCoMo reference JSON")
    ap.add_argument("--skip-mem0-gate", action="store_true",
                    help="skip the reproduction gate for local plumbing only")
    ap.add_argument("--dry-run", action="store_true", help="print the plan offline, no scoring")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    args.render_only = False
    if args.sample_offset < 0:
        raise SystemExit("--sample-offset must be >= 0")
    if args.run_offset < 0:
        raise SystemExit("--run-offset must be >= 0")
    if not args.dry_run and args.runs <= 0:
        raise SystemExit("--runs must be > 0 unless --dry-run is used")

    trials = plan(args.subset, args.runs)
    est = estimate_tokens(len(trials), args.subset, args.runs)
    print(f"Coordinate-descent sweep: {len(STAGES)} stages, {len(trials)} trials on a "
          f"{args.subset}-question subset x {args.runs} run(s), dataset={args.dataset}, "
          f"sample_offset={args.sample_offset}.")
    print(f"Rough cost ESTIMATE if run live: ~{est:,} output tokens.")
    for i, t in enumerate(trials):
        print(f"  [{i+1:>2}] set {_format_assignment(t['stage'], t['value'])}  "
              "(other knobs at running-best)")

    if args.dry_run:
        print("\n--dry-run: enumerated the plan only. No questions scored, nothing fabricated.")
        return 0

    from eidetic.config import get_settings

    if not get_settings().has_api_key:
        print("\nNo DASHSCOPE_API_KEY: refusing to fabricate sweep scores. Add a key, then "
              "re-run without --dry-run.")
        return 2
    if not args.skip_mem0_gate:
        if not args.mem0_gate_out or not args.mem0_gate_expected:
            raise SystemExit(
                "Live sweep requires a passing Mem0 reproduction gate first. Provide "
                "--mem0-gate-out and --mem0-gate-expected, or use --skip-mem0-gate only "
                "for local plumbing."
            )
        from .gate import run_gate

        gate = run_gate(Path(args.mem0_gate_out), Path(args.mem0_gate_expected))
        print(f"Mem0 reproduction gate: {gate['status']} ({gate.get('reason', '')})")
        if gate["status"] != "PASS":
            raise SystemExit("Mem0 reproduction gate did not pass; refusing to tune benchmark flags.")

    from . import run as bench_run
    from . import scoreboard
    from .datasets import category_counts
    from .harness import load_logs, run_system
    from .judge import Judge

    if args.split != "dev":
        raise SystemExit("Integrity wall: the sweep tunes on the DEV split only.")
    samples = bench_run.load_samples(args.dataset, args.subset, args.variant, args.sample_offset,
                                     split="dev")
    if not samples:
        raise SystemExit(
            "No DEV-split samples loaded. The sweep tunes on the private dev split only "
            "(integrity wall); check --dataset/--subset/--sample-offset. Reported numbers "
            "come from --split test via bench.run/reproduce.sh."
        )
    print(f"Loaded {len(samples)} DEV-split samples (integrity wall); "
          f"categories: {category_counts(samples)}")

    judge = Judge()
    judge_desc = judge.describe()

    if args.sampler == "tpe":
        return run_tpe_sweep(args, samples, judge, judge_desc)

    original_env = {env_var: os.environ.get(env_var) for env_var in _stage_env_keys()}
    best_env: dict[str, str] = {}
    best_score = -1.0
    results: list[dict[str, Any]] = []

    try:
        for env_var, values in STAGES:
            stage_best_val, stage_best_score = None, -1.0
            stage_control_dir: Path | None = None
            for value in values:
                for key, val in stage_assignment(env_var, value).items():
                    os.environ[key] = val
                for key, val in best_env.items():
                    os.environ[key] = val
                get_settings.cache_clear()

                out_dir = Path(args.out) / f"{env_var}={_safe_value(value)}"
                for raw in args.systems.split(","):
                    sysobj = bench_run.make_system(raw)
                    run_system(sysobj, samples, judge, runs=args.runs, out_dir=out_dir,
                               run_offset=args.run_offset, overwrite=args.overwrite)
                bench_run.write_manifest(out_dir, args, judge_desc, samples=samples)
                scoreboard.render(out_dir, judge_desc)

                rows = load_logs(out_dir)
                _validate_rows(rows, out_dir)
                acc, cats = score_rows(rows)
                comparison_md = None
                comparison_status = "control"
                if stage_control_dir is None:
                    stage_control_dir = out_dir
                else:
                    comparison_md, comparison = write_stage_comparison(
                        stage_control_dir, out_dir, stage=env_var, value=value
                    )
                    comparison_status = comparison["status"]
                print(f"  {_format_assignment(env_var, value)} -> overall accuracy {acc:.3f}; "
                      f"categories: {_format_categories(cats)}")
                results.append({
                    "stage": env_var,
                    "value": value,
                    "overall_accuracy": acc,
                    "categories": cats,
                    "out_dir": str(out_dir),
                    "stage_control_dir": str(stage_control_dir),
                    "stage_comparison": str(comparison_md) if comparison_md else None,
                    "stage_comparison_status": comparison_status,
                })
                if acc > stage_best_score:
                    stage_best_val, stage_best_score = value, acc
            if stage_best_val is None:
                raise RuntimeError(f"No scored rows for stage {env_var}")
            best_env.update(stage_assignment(env_var, stage_best_val))
            best_score = stage_best_score
            print(f"Stage {env_var}: best={stage_best_val} ({stage_best_score:.3f})")
    finally:
        _restore_env(original_env)
        get_settings.cache_clear()

    out = Path(args.out) / "best_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_env": best_env,
        "best_overall_accuracy": best_score,
        "dataset": args.dataset,
        "subset": args.subset,
        "sample_offset": args.sample_offset,
        "runs": args.runs,
        "run_offset": args.run_offset,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nBest config -> {out}: {best_env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
