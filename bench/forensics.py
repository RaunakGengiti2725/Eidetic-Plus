"""Failure forensics for benchmark JSONL logs.

This is intentionally offline and deterministic: it reads the exact rows emitted by the harness,
groups failed/abstained/error rows, assigns a coarse bucket, and writes an audit report. It never
rescored answers or calls a model, so it is safe to run after every benchmark.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .harness import load_logs
from .scoreboard import consolidation_rollup


def _is_abstain_text(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in (
        "do not have that in memory",
        "don't have enough",
        "do not have enough",
        "insufficient evidence",
        "cannot answer",
    ))


def bucket_failure(row: dict) -> str:
    """Coarse, deterministic failure bucket from logged metadata."""
    if row.get("error"):
        return "infra"
    if row.get("abstained") or _is_abstain_text(row.get("predicted", "")):
        return "abstention"
    extra = row.get("extra") or {}
    if extra.get("verified") and not row.get("correct"):
        return "reader_error"
    coverage = extra.get("coverage")
    try:
        coverage_f = float(coverage)
    except (TypeError, ValueError):
        coverage_f = None
    if coverage_f is not None and coverage_f < 0.25:
        return "retrieval_miss"
    if coverage_f is not None and coverage_f >= 0.25:
        return "reader_error"
    if extra.get("post_answer_error"):
        return "infra"
    return "retrieval_miss"


def suggested_fix(bucket: str) -> str:
    return {
        "retrieval_miss": "raise lexical/graph/event recall depth; inspect candidate_memory_ids and query parsing",
        "reader_error": "inspect context blocks and reader scaffold; failure happened with plausible evidence present",
        "abstention": "calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals",
        "policy": "relax shared reader policy only if the question permits grounded inference",
        "infra": "fix transport/dependency error before counting this slice",
    }.get(bucket, "inspect manually")


def analyze(rows: Iterable[dict], *, system: str | None = None) -> dict:
    selected = [r for r in rows if system is None or r.get("system") == system]
    consolidation = consolidation_rollup(selected)
    failures = []
    by_bucket: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for r in selected:
        failed = bool(r.get("error")) or bool(r.get("abstained")) or not bool(r.get("correct"))
        if not failed:
            continue
        bucket = bucket_failure(r)
        by_bucket[bucket] += 1
        by_category[str(r.get("category", "unknown"))][bucket] += 1
        failures.append({
            "system": r.get("system", ""),
            "dataset": r.get("dataset", ""),
            "category": r.get("category", ""),
            "sample_id": r.get("sample_id", ""),
            "question": r.get("question", ""),
            "gold": r.get("gold", ""),
            "predicted": r.get("predicted", ""),
            "bucket": bucket,
            "suggested_fix": suggested_fix(bucket),
            "coverage": (r.get("extra") or {}).get("coverage"),
            "verified": (r.get("extra") or {}).get("verified"),
            "abstained": bool(r.get("abstained")),
            "error": r.get("error", ""),
        })
    return {
        "rows": len(selected),
        "failures": len(failures),
        "bucket_counts": dict(by_bucket),
        "category_bucket_counts": {k: dict(v) for k, v in sorted(by_category.items())},
        "consolidation": consolidation,
        "items": failures,
    }


def render_markdown(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Benchmark Failure Forensics", ""]
    lines.append(f"Rows inspected: {report['rows']}")
    lines.append(f"Failures/abstentions/errors: {report['failures']}")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    lines.append("| bucket | count | suggested fix |")
    lines.append("|---|---:|---|")
    for bucket, count in sorted(report["bucket_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {bucket} | {count} | {suggested_fix(bucket)} |")
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    lines.append("| category | buckets |")
    lines.append("|---|---|")
    for cat, counts in report["category_bucket_counts"].items():
        desc = ", ".join(f"{b}: {n}" for b, n in sorted(counts.items()))
        lines.append(f"| {cat} | {desc} |")
    lines.append("")
    consolidation = report.get("consolidation", {})
    if consolidation:
        lines.append("## Consolidation Health")
        lines.append("")
        lines.append("| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for sysname, c in sorted(consolidation.items()):
            lines.append(f"| {sysname} | {c.get('groups', 0)} | {c.get('pending_processed', 0)} | "
                         f"{c.get('facts_extracted', 0)} | {c.get('events_indexed', 0)} | "
                         f"{c.get('extraction_timed_out', 0)} | {c.get('extraction_deferred', 0)} | "
                         f"{c.get('extraction_windows_planned', 0)} | "
                         f"{c.get('extraction_windows_submitted', 0)} | "
                         f"{c.get('extraction_raw_only_bounded', 0)} | "
                         f"{c.get('record_raw_only_bounded', 0)} | "
                         f"{c.get('extraction_partial_bounded', 0)} | "
                         f"{c.get('long_haystack_bounded', 0)} | "
                         f"{c.get('long_haystack_raw_only', 0)} |")
        lines.append("")
    lines.append("## Failed Items")
    lines.append("")
    lines.append("| system | category | sample | bucket | question | gold | predicted |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in report["items"]:
        q = str(item["question"]).replace("|", "\\|")[:180]
        gold = str(item["gold"]).replace("|", "\\|")[:120]
        pred = str(item["predicted"]).replace("|", "\\|")[:120]
        lines.append(
            f"| {item['system']} | {item['category']} | {item['sample_id']} | "
            f"{item['bucket']} | {q} | {gold} | {pred} |"
        )
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze benchmark JSONL failures")
    ap.add_argument("--logs", default="artifacts/bench")
    ap.add_argument("--system", help="optional system filter, e.g. eidetic-plus-full")
    ap.add_argument("--out", default="artifacts/forensics/report.md")
    args = ap.parse_args()

    rows = load_logs(Path(args.logs))
    report = analyze(rows, system=args.system)
    out = render_markdown(report, Path(args.out))
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(report, indent=2))
    print(f"Forensics -> {out}")
    print(f"JSON      -> {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
