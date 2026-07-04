"""Bucket a bench run's misses into a forensics draft.

Reads one ``<system>__run0.jsonl`` and prints a markdown table grouping misses by
category and failure bucket (abstained / verified-wrong / unverified-wrong / error),
plus the per-row detail block forensics always starts from. Read-only; works on any
run artifact. Dev jsonl from the acceleration branch; the holdout session may point it
at holdout artifacts (sample IDs stay in the run dir, never in scanned docs).

Usage:
    python -m bench.miss_taxonomy artifacts/<run>/eidetic-plus-full__run0.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def bucket(row: dict) -> str:
    if row.get("error"):
        return "error"
    if row.get("abstained"):
        return "abstained"
    if row.get("extra", {}).get("verified"):
        return "verified-wrong"
    return "unverified-wrong"


def taxonomize(path: Path) -> str:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    misses = [r for r in rows if not r.get("correct")]
    ok = len(rows) - len(misses)
    lines = [f"## Miss taxonomy: {path.name} ({ok}/{len(rows)} correct)", ""]
    counts = Counter((r.get("category", "?"), bucket(r)) for r in misses)
    buckets = ["verified-wrong", "abstained", "unverified-wrong", "error"]
    cats = sorted({c for c, _b in counts})
    lines.append("| category | " + " | ".join(buckets) + " |")
    lines.append("|---|" + "---|" * len(buckets))
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(str(counts.get((c, b), 0)) for b in buckets) + " |")
    lines.append("")
    for b in buckets:
        group = [r for r in misses if bucket(r) == b]
        if not group:
            continue
        lines.append(f"### {b} ({len(group)})")
        for r in group:
            lines.append(f"- `{r.get('sample_id')}` [{r.get('category')}] Q: {str(r.get('question'))[:90]}")
            lines.append(f"  - gold: `{str(r.get('gold'))[:80]}`")
            lines.append(f"  - pred: `{str(r.get('predicted'))[:80]}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    print(taxonomize(Path(sys.argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
