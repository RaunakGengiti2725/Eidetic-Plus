"""Cost report: bench run jsonl -> the verified-answers-per-token story in one table.

    python -m bench.cost_report artifacts/dev40_combined_off_codex artifacts/dev40_combined_on_codex

Reads every *__run0.jsonl (or an explicit .jsonl path) and prints a markdown table with
the honest cost fields: real query_tokens per row, real write-side model_usage tokens
deduplicated per conversation (the write_tokens column is a content-volume proxy that is
identical across write-path arms -- it is reported but never used for totals), structured
vs reader path split from extra.structured_recall, verified-correct (correct AND
entailment-proven), and the headline metric: total DashScope tokens per verified-correct
answer. Dev jsonl only from this branch; holdout claims live elsewhere.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _conversation_id(sample_id: str) -> str:
    return sample_id.rsplit("_q", 1)[0] if "_q" in sample_id else sample_id


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    qtoks = [int(r.get("query_tokens") or 0) for r in rows]
    correct = sum(1 for r in rows if r.get("correct"))
    verified_correct = sum(
        1 for r in rows if r.get("correct") and r.get("extra", {}).get("verified")
    )
    abstained_rows = [r for r in rows if r.get("abstained")]
    structured = sum(1 for r in rows if r.get("extra", {}).get("structured_recall"))
    operators = Counter(
        r["extra"]["smqe_operator"]
        for r in rows
        if r.get("extra", {}).get("smqe_operator")
    )

    write_real: dict[str, int] = {}
    write_calls: dict[str, int] = {}
    write_proxy: dict[str, int] = {}
    for r in rows:
        conv = _conversation_id(str(r.get("sample_id", "")))
        write_proxy.setdefault(conv, int(r.get("write_tokens") or 0))
        usage = (r.get("extra", {}).get("consolidate") or {}).get("model_usage") or {}
        if usage:
            write_real.setdefault(
                conv, int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
            )
            write_calls.setdefault(conv, int(usage.get("calls", 0)))

    query_total = sum(qtoks)
    write_total = sum(write_real.values()) if write_real else None
    total = query_total + write_total if write_total is not None else None
    return {
        "n": n,
        "correct": correct,
        "verified_correct": verified_correct,
        "abstained": len(abstained_rows),
        "abstained_e2e_ms": sorted(round(r.get("e2e_ms", 0)) for r in abstained_rows),
        "structured": structured,
        "operators": operators,
        "qtok_median": statistics.median(qtoks) if qtoks else 0,
        "qtok_mean": statistics.mean(qtoks) if qtoks else 0,
        "qtok_total": query_total,
        "write_real_total": write_total,
        "write_calls_total": sum(write_calls.values()) if write_calls else None,
        "write_proxy_total": sum(write_proxy.values()),
        "conversations": len(write_proxy),
        "total_tokens": total,
        "tokens_per_vc": (round(total / verified_correct) if total and verified_correct else None),
        "qtok_per_vc": (round(query_total / verified_correct) if verified_correct else None),
    }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.0f}"
    return f"{v:,}"


def render(results: dict[str, dict]) -> str:
    lines = [
        "| run | n | vc | structured | qtok med | qtok mean | qtok total | "
        "write tok (real) | write calls | total tok | tok/vc | qtok/vc | abstained |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, a in results.items():
        lines.append(
            f"| {name} | {a['n']} | {a['verified_correct']} | "
            f"{a['structured']}/{a['n']} | {_fmt(a['qtok_median'])} | {_fmt(a['qtok_mean'])} | "
            f"{_fmt(a['qtok_total'])} | {_fmt(a['write_real_total'])} | "
            f"{_fmt(a['write_calls_total'])} | {_fmt(a['total_tokens'])} | "
            f"{_fmt(a['tokens_per_vc'])} | {_fmt(a['qtok_per_vc'])} | {a['abstained']} |"
        )
    lines.append("")
    for name, a in results.items():
        ops = ", ".join(f"{k}={v}" for k, v in a["operators"].most_common()) or "none"
        abst = ", ".join(f"{ms}ms" for ms in a["abstained_e2e_ms"]) or "none"
        lines.append(
            f"- **{name}**: write proxy {_fmt(a['write_proxy_total'])} "
            f"(content volume, arm-invariant); operators: {ops}; abstained e2e: {abst}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="run dirs or jsonl files")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for raw in args.paths:
        path = Path(raw)
        jsonls = [path] if path.suffix == ".jsonl" else sorted(path.glob("*__run0.jsonl"))
        if not jsonls:
            raise SystemExit(f"no *__run0.jsonl under {path}")
        for jp in jsonls:
            label = f"{path.name}/{jp.stem}" if len(jsonls) > 1 else path.name
            results[label] = analyze(load_rows(jp))
    print(render(results))


if __name__ == "__main__":
    main()
