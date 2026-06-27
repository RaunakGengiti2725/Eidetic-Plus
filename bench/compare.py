"""Compare two real benchmark log directories.

Use this for flag A/Bs: run a control board into one directory, run the flagged config
into another, then compare per-category accuracy, tokens, and latency from JSONL logs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from pydantic import BaseModel

from .scoreboard import _mcnemar_pvalue, _wilson_ci


class LogBundle(BaseModel):
    files: list[str]
    rows: list[dict[str, Any]]
    row_count: int


def _pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def _load_logs_strict(out_dir: Path) -> LogBundle:
    files = sorted(Path(out_dir).glob("*__run*.jsonl"))
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{lineno}: {e}") from e
            for field in ("system", "dataset", "category", "sample_id", "correct"):
                if field not in row:
                    raise ValueError(f"Missing '{field}' in {path}:{lineno}")
            if not isinstance(row["correct"], bool):
                raise ValueError(f"'correct' must be bool in {path}:{lineno}")
            for metric in ("query_tokens", "write_tokens", "search_ms", "e2e_ms"):
                if metric in row and row[metric] is not None and float(row[metric]) < 0:
                    raise ValueError(f"Negative '{metric}' in {path}:{lineno}")
            key = (
                row["system"], row["dataset"], row["category"],
                row["sample_id"], int(row.get("run_idx", 0)),
            )
            if key in seen:
                raise ValueError(f"Duplicate row key {key} in {path}:{lineno}")
            seen.add(key)
            rows.append(row)
    return LogBundle(files=[str(p) for p in files], rows=rows, row_count=len(rows))


def _mean_present(rows: list[dict], field: str) -> float | None:
    vals = [float(r[field]) for r in rows if field in r and r[field] is not None]
    return mean(vals) if vals else None


def _p_present(rows: list[dict], field: str, p: float) -> float | None:
    vals = [float(r[field]) for r in rows if field in r and r[field] is not None]
    return _pct(vals, p) if vals else None


def _write_mean(rows: list[dict]) -> float | None:
    vals: dict[str, float] = {}
    for r in rows:
        if "write_tokens" not in r or r["write_tokens"] is None:
            continue
        group = str(r["sample_id"]).split("_q")[0]
        vals[f"{r['system']}:{r['dataset']}:{group}:{r.get('run_idx', 0)}"] = float(r["write_tokens"])
    return mean(vals.values()) if vals else None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _group(rows: list[dict], system: str | None) -> dict[tuple[str, str, str], dict]:
    grouped: dict[tuple[str, str, str], dict] = {}
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if system and r.get("system") != system:
            continue
        buckets[(r["system"], r["dataset"], r["category"])].append(r)
    for key, vals in buckets.items():
        correct = sum(1 for r in vals if r.get("correct"))
        n = len(vals)
        lo, hi = _wilson_ci(correct, n)
        grouped[key] = {
            "correct": correct,
            "n": n,
            "runs": sorted({int(r.get("run_idx", 0)) for r in vals}),
            "accuracy": correct / n if n else 0.0,
            "ci95": [lo, hi],
            "tokens_per_query": _mean_present(vals, "query_tokens"),
            "write_tokens_per_conversation": _write_mean(vals),
            "search_p50": _p_present(vals, "search_ms", 50),
            "search_p95": _p_present(vals, "search_ms", 95),
            "e2e_p50": _p_present(vals, "e2e_ms", 50),
            "e2e_p95": _p_present(vals, "e2e_ms", 95),
        }
    return grouped


def _paired(control: list[dict], experiment: list[dict], key: tuple[str, str, str]) -> dict:
    system, dataset, category = key
    c = {
        (r["sample_id"], r.get("run_idx", 0)): bool(r.get("correct"))
        for r in control
        if r.get("system") == system and r.get("dataset") == dataset and r.get("category") == category
    }
    e = {
        (r["sample_id"], r.get("run_idx", 0)): bool(r.get("correct"))
        for r in experiment
        if r.get("system") == system and r.get("dataset") == dataset and r.get("category") == category
    }
    common = sorted(set(c) & set(e))
    control_only = experiment_only = both = neither = 0
    gained: list[dict] = []     # wrong in control -> right in experiment (the experiment fixed it)
    regressed: list[dict] = []  # right in control -> wrong in experiment (the experiment broke it)
    for k in common:
        cv, ev = c[k], e[k]
        if cv and ev:
            both += 1
        elif cv and not ev:
            control_only += 1
            regressed.append({"sample_id": k[0], "run_idx": k[1]})
        elif ev and not cv:
            experiment_only += 1
            gained.append({"sample_id": k[0], "run_idx": k[1]})
        else:
            neither += 1
    return {
        "paired_n": len(common),
        "control_only": control_only,
        "experiment_only": experiment_only,
        "both": both,
        "neither": neither,
        # Per-question flip attribution: the actual question ids that moved, so a judge can confirm
        # WHICH questions the ablated/added mechanism is responsible for (the experiment dir name
        # carries the mechanism; these lists carry the evidence).
        "gained": gained,
        "regressed": regressed,
        "unpaired_control_rows": len(set(c) - set(e)),
        "unpaired_experiment_rows": len(set(e) - set(c)),
        "p_mcnemar": _mcnemar_pvalue(control_only, experiment_only) if common else None,
    }


def compare_dirs(control_dir: Path, experiment_dir: Path,
                 *, system: str | None = None) -> dict:
    control_bundle = _load_logs_strict(control_dir)
    experiment_bundle = _load_logs_strict(experiment_dir)
    control_rows = control_bundle.rows
    experiment_rows = experiment_bundle.rows
    warnings: list[str] = []
    if not control_rows:
        warnings.append("no control logs")
    if not experiment_rows:
        warnings.append("no experiment logs")
    control = _group(control_rows, system)
    experiment = _group(experiment_rows, system)
    keys = sorted(set(control) | set(experiment))
    comparisons: dict[str, dict] = {}
    for key in keys:
        label = "|".join(key)
        c = control.get(key)
        e = experiment.get(key)
        if c is None or e is None:
            comparisons[label] = {"status": "missing-control" if c is None else "missing-experiment"}
            continue
        paired = _paired(control_rows, experiment_rows, key)
        item_status = "compared"
        if paired["paired_n"] == 0 or paired["unpaired_control_rows"] or paired["unpaired_experiment_rows"]:
            item_status = "unpaired"
        comparisons[label] = {
            "status": item_status,
            "control": c,
            "experiment": e,
            "delta_accuracy_points": (e["accuracy"] - c["accuracy"]) * 100.0,
            "delta_tokens_per_query": _delta(c["tokens_per_query"], e["tokens_per_query"]),
            "delta_write_tokens_per_conversation": _delta(
                c["write_tokens_per_conversation"], e["write_tokens_per_conversation"]
            ),
            "delta_search_p50_ms": _delta(c["search_p50"], e["search_p50"]),
            "delta_search_p95_ms": _delta(c["search_p95"], e["search_p95"]),
            "delta_e2e_p50_ms": _delta(c["e2e_p50"], e["e2e_p50"]),
            "delta_e2e_p95_ms": _delta(c["e2e_p95"], e["e2e_p95"]),
            "paired": paired,
        }
    status = "ok"
    if warnings:
        status = "no_logs"
    elif any(v["status"] == "unpaired" for v in comparisons.values()):
        status = "unpaired"
    elif any(v["status"] != "compared" for v in comparisons.values()):
        status = "partial"
    return {
        "status": status,
        "control_dir": str(control_dir),
        "experiment_dir": str(experiment_dir),
        "control_files": control_bundle.files,
        "experiment_files": experiment_bundle.files,
        "control_rows": control_bundle.row_count,
        "experiment_rows": experiment_bundle.row_count,
        "warnings": warnings,
        "system_filter": system,
        "comparisons": comparisons,
    }


def render_markdown(result: dict, out_path: Path) -> Path:
    lines = [
        "# Benchmark A/B comparison",
        "",
        f"Control: `{result['control_dir']}`",
        f"Experiment: `{result['experiment_dir']}`",
        "",
        f"Status: **{result['status']}**",
        "",
        "| system | dataset | category | n control | n experiment | acc delta pp | query tok delta | write tok delta | search p95 delta ms | paired n | unpaired control | unpaired experiment | McNemar p | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, item in sorted(result["comparisons"].items()):
        system, dataset, category = label.split("|")
        if item["status"] not in ("compared", "unpaired"):
            lines.append(
                f"| {system} | {dataset} | {category} | - | - | - | - | - | - | - | - | - | - | {item['status']} |"
            )
            continue
        paired = item["paired"]
        p = paired["p_mcnemar"]
        p_text = "-" if p is None else f"{p:.4f}"
        qtok = "-" if item["delta_tokens_per_query"] is None else f"{item['delta_tokens_per_query']:.0f}"
        wtok = (
            "-" if item["delta_write_tokens_per_conversation"] is None
            else f"{item['delta_write_tokens_per_conversation']:.0f}"
        )
        sp95 = "-" if item["delta_search_p95_ms"] is None else f"{item['delta_search_p95_ms']:.1f}"
        lines.append(
            f"| {system} | {dataset} | {category} | {item['control']['n']} | "
            f"{item['experiment']['n']} | {item['delta_accuracy_points']:.1f} | "
            f"{qtok} | {wtok} | {sp95} | {paired['paired_n']} | "
            f"{paired['unpaired_control_rows']} | {paired['unpaired_experiment_rows']} | "
            f"{p_text} | {item['status']} |"
        )

    # Per-question flip table -- the attribution evidence. For an ablation the experiment dir is the
    # ablated config, so a GAIN (wrong->right) names a question the mechanism is responsible for and a
    # REGRESSION (right->wrong) names one it costs. Only keys with a flip are listed.
    flip_rows = []
    for label, item in sorted(result["comparisons"].items()):
        if item.get("status") not in ("compared", "unpaired"):
            continue
        paired = item["paired"]
        gained = paired.get("gained", [])
        regressed = paired.get("regressed", [])
        if not gained and not regressed:
            continue
        system, dataset, category = label.split("|")

        def _ids(flips):
            return ", ".join(f["sample_id"] for f in flips) or "-"
        flip_rows.append(
            f"| {system} | {dataset} | {category} | {len(gained)} | {len(regressed)} | "
            f"{_ids(gained)} | {_ids(regressed)} |"
        )
    lines += ["", "## Per-question flips (attribution evidence)", ""]
    if flip_rows:
        lines += [
            "| system | dataset | category | gains | regressions | gained question ids | regressed question ids |",
            "|---|---|---|---:|---:|---|---|",
            *flip_rows,
        ]
    else:
        lines.append("_No per-question flips (identical correctness on every paired question)._")

    out_path.write_text("\n".join(lines) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two benchmark artifact directories")
    ap.add_argument("--control", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--out", default="artifacts/compare.md")
    args = ap.parse_args()
    result = compare_dirs(Path(args.control), Path(args.experiment),
                          system=args.system.strip() or None)
    out = render_markdown(result, Path(args.out))
    print(f"Comparison -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
