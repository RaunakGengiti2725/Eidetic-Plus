"""Mem0 reproduction gate.

Reads real harness logs and compares Mem0 LoCoMo category accuracy against a published
reference supplied as JSON. No logs or no reference means no pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .compare import _load_logs_strict
from .fingerprints import log_fingerprint


def _as_fraction(value: float) -> float:
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _load_expected(path: Path) -> tuple[dict[str, float], float, int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Expected reference file not found: {path}")
    data = json.loads(path.read_text())
    cats = data.get("categories") if isinstance(data, dict) else data
    if not isinstance(cats, dict) or not cats:
        raise ValueError("Expected reference must be a JSON object with a non-empty 'categories' map.")
    tolerance = float(data.get("tolerance_points", 2.0)) if isinstance(data, dict) else 2.0
    min_n = int(data.get("min_n", 50)) if isinstance(data, dict) else 50
    reader_model = str(data.get("reader_model", "qwen-plus")) if isinstance(data, dict) else "qwen-plus"
    return {str(k): _as_fraction(v) for k, v in cats.items()}, tolerance / 100.0, min_n, reader_model


def _load_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {out_dir}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("run_manifest.json must be a JSON object.")
    return data


def _observed(rows: list[dict], *, system: str, dataset: str) -> dict[str, dict]:
    by_cat: dict[str, list[bool]] = {}
    runs: dict[str, set[int]] = {}
    query_tokens: dict[str, list[float]] = {}
    write_tokens: dict[str, dict[str, float]] = {}
    search_ms: dict[str, list[float]] = {}
    e2e_ms: dict[str, list[float]] = {}
    for r in rows:
        if r.get("system") != system or r.get("dataset") != dataset:
            continue
        cat = str(r.get("category", "unknown"))
        by_cat.setdefault(cat, []).append(r["correct"])
        runs.setdefault(cat, set()).add(int(r.get("run_idx", 0)))
        if "query_tokens" in r and r["query_tokens"] is not None:
            query_tokens.setdefault(cat, []).append(float(r["query_tokens"]))
        if "write_tokens" in r and r["write_tokens"] is not None:
            group = str(r["sample_id"]).split("_q")[0]
            write_tokens.setdefault(cat, {})[f"{group}:{r.get('run_idx', 0)}"] = float(r["write_tokens"])
        if "search_ms" in r and r["search_ms"] is not None:
            search_ms.setdefault(cat, []).append(float(r["search_ms"]))
        if "e2e_ms" in r and r["e2e_ms"] is not None:
            e2e_ms.setdefault(cat, []).append(float(r["e2e_ms"]))

    def p95(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
        return values[idx]

    return {
        cat: {"n": len(vals), "accuracy": mean(1.0 if v else 0.0 for v in vals),
              "runs": sorted(runs.get(cat, set())),
              "tokens_per_query": mean(query_tokens.get(cat, [])) if query_tokens.get(cat) else None,
              "write_tokens_per_conversation": (
                  mean(write_tokens.get(cat, {}).values()) if write_tokens.get(cat) else None
              ),
              "search_p95": p95(search_ms.get(cat, [])),
              "e2e_p95": p95(e2e_ms.get(cat, []))}
        for cat, vals in by_cat.items()
    }


def run_gate(out_dir: Path, expected_path: Path, *, system: str = "mem0",
             dataset: str = "locomo") -> dict:
    expected, tolerance, min_n, required_reader = _load_expected(expected_path)
    manifest = _load_manifest(out_dir)
    fingerprint = log_fingerprint(out_dir)
    env = manifest.get("env", {}) if isinstance(manifest.get("env", {}), dict) else {}
    actual_reader = env.get("READER_MODEL", "")
    if actual_reader != required_reader:
        return {
            "status": "FAIL",
            "reason": f"reader mismatch: {actual_reader or '<unset>'} != {required_reader}",
            "system": system,
            "dataset": dataset,
            "manifest": manifest,
            "expected": expected,
            "log_fingerprint": fingerprint,
        }
    rows = _load_logs_strict(out_dir).rows
    obs = _observed(rows, system=system, dataset=dataset)
    if not rows:
        return {
            "status": "FAIL",
            "reason": "no logs",
            "system": system,
            "dataset": dataset,
            "manifest": manifest,
            "observed": obs,
            "expected": expected,
            "log_fingerprint": fingerprint,
        }

    missing = [cat for cat in expected if cat not in obs]
    total_n = sum(v["n"] for v in obs.values())
    comparisons: dict[str, dict] = {}
    ok = not missing and total_n >= min_n
    for cat, exp in expected.items():
        got = obs.get(cat)
        if got is None:
            comparisons[cat] = {"status": "missing", "expected": exp}
            continue
        delta = got["accuracy"] - exp
        pass_cat = abs(delta) <= tolerance
        ok = ok and pass_cat
        comparisons[cat] = {
            "status": "PASS" if pass_cat else "FAIL",
            "n": got["n"], "runs": got["runs"],
            "observed": got["accuracy"], "expected": exp, "delta": delta,
            "tokens_per_query": got.get("tokens_per_query"),
            "write_tokens_per_conversation": got.get("write_tokens_per_conversation"),
            "search_p95": got.get("search_p95"),
            "e2e_p95": got.get("e2e_p95"),
        }
    if total_n < min_n:
        reason = f"insufficient n: {total_n} < {min_n}"
    elif missing:
        reason = "missing categories"
    else:
        reason = "within tolerance" if ok else "outside tolerance"
    return {
        "status": "PASS" if ok else "FAIL",
        "reason": reason,
        "system": system,
        "dataset": dataset,
        "tolerance": tolerance,
        "min_n": min_n,
        "total_n": total_n,
        "observed": obs,
        "expected": expected,
        "manifest": manifest,
        "comparisons": comparisons,
        "log_fingerprint": fingerprint,
    }


def render_markdown(result: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Mem0 Reproduction Gate", ""]
    lines.append(f"Status: **{result['status']}**")
    lines.append(f"Reason: {result.get('reason', '')}")
    lines.append(f"System: `{result.get('system', '')}`")
    lines.append(f"Dataset: `{result.get('dataset', '')}`")
    lines.append("")
    lines.append("| category | status | n | observed | expected | delta pp | search p95 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for cat, item in sorted(result.get("comparisons", {}).items()):
        if item.get("status") == "missing":
            lines.append(f"| {cat} | missing | 0 | - | {item.get('expected', 0.0) * 100:.1f}% | - | - |")
            continue
        observed = float(item.get("observed", 0.0) or 0.0)
        expected = float(item.get("expected", 0.0) or 0.0)
        delta = float(item.get("delta", 0.0) or 0.0)
        sp95 = item.get("search_p95")
        sp95_s = "-" if sp95 is None else f"{float(sp95):.1f}"
        lines.append(f"| {cat} | {item.get('status', '')} | {item.get('n', 0)} | "
                     f"{observed * 100:.1f}% | {expected * 100:.1f}% | "
                     f"{delta * 100:.1f} | {sp95_s} |")
    out_path.write_text("\n".join(lines) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Mem0 reproduction gate from real logs")
    ap.add_argument("--out", default="artifacts/bench", help="directory containing *__run*.jsonl")
    ap.add_argument("--expected", required=True, help="JSON reference with published category scores")
    ap.add_argument("--system", default="mem0")
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--report-out", default="", help="write Markdown + JSON report")
    args = ap.parse_args()

    res = run_gate(Path(args.out), Path(args.expected), system=args.system, dataset=args.dataset)
    print("# Mem0 reproduction gate")
    print(f"system={res.get('system')} dataset={res.get('dataset')} status={res['status']}")
    print("per-category n:")
    for cat in sorted(res.get("expected", {})):
        got = res.get("observed", {}).get(cat)
        n = got["n"] if got else 0
        runs = got["runs"] if got else []
        print(f"  {cat}: n={n}, runs={runs}")
    print(json.dumps(res, indent=2))
    report_out = Path(args.report_out) if args.report_out else Path(args.out) / "mem0_gate.md"
    render_markdown(res, report_out)
    print(f"Report -> {report_out}")
    return 0 if res["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
