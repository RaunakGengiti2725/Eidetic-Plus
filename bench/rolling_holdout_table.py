"""Rolling never-touched holdout table across rotation windows (r1..rN).

Reads each window's per-row jsonl for both systems and emits a markdown table +
totals. Every number is recomputed from the raw logs -- no hand-carried digits.

    .venv/bin/python -m bench.rolling_holdout_table \
        artifacts/holdout_rotation_r1_codex ... artifacts/holdout_rotation_r7_codex
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SYSTEMS = {
    "eidetic": "eidetic-plus-full__run0.jsonl",
    "mem0": "mem0__run0.jsonl",
}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _win(rows: list[dict]) -> dict:
    correct = sum(1 for r in rows if r.get("correct"))
    verified = sum(1 for r in rows if (r.get("extra") or {}).get("verified"))
    structured = sum(1 for r in rows if (r.get("extra") or {}).get("structured_recall"))
    qtoks = [r.get("query_tokens") for r in rows if isinstance(r.get("query_tokens"), (int, float))]
    return {
        "n": len(rows),
        "correct": correct,
        "verified": verified,
        "structured": structured,
        "qtok_median": statistics.median(qtoks) if qtoks else None,
    }


def build(dirs: list[Path]) -> dict:
    windows = []
    for d in dirs:
        entry: dict = {"dir": d.name}
        for name, fname in SYSTEMS.items():
            entry[name] = _win(_rows(d / fname))
        windows.append(entry)
    totals = {}
    for name in SYSTEMS:
        totals[name] = {
            "n": sum(w[name]["n"] for w in windows),
            "correct": sum(w[name]["correct"] for w in windows),
            "verified": sum(w[name]["verified"] for w in windows),
            "structured": sum(w[name]["structured"] for w in windows),
        }
    return {"windows": windows, "totals": totals}


def render(table: dict) -> str:
    lines = [
        "| window | eidetic correct | eidetic verified | eidetic structured | eidetic qtok med | mem0 correct | margin |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in table["windows"]:
        e, m = w["eidetic"], w["mem0"]
        margin = e["correct"] - m["correct"] if m["n"] else None
        lines.append(
            f"| {w['dir']} | {e['correct']}/{e['n']} | {e['verified']} | {e['structured']}/{e['n']} "
            f"| {e['qtok_median'] if e['qtok_median'] is not None else 'n/a'} "
            f"| {m['correct']}/{m['n']} " + (f"| +{margin} |" if margin and margin > 0 else f"| {margin if margin is not None else 'pending'} |")
        )
    te, tm = table["totals"]["eidetic"], table["totals"]["mem0"]
    lines.append(
        f"| **rolling** | **{te['correct']}/{te['n']}** | **{te['verified']}** | **{te['structured']}/{te['n']}** | "
        f"| **{tm['correct']}/{tm['n']}** | **{te['correct'] - tm['correct']:+d}** |"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json-out", help="optional JSON sidecar path")
    args = ap.parse_args()
    table = build([Path(d) for d in args.dirs])
    print(render(table))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
