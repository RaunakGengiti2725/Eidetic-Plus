"""Aggregate a live free-tier collection (bench/notebooklm_freetier_run.py) into a
labeled report. Re-scores prelim containment from the stored answers (so heuristic fixes
apply retroactively -- the collection is the expensive part, scoring is derived).

Every number is labeled: caller tokens are 0 BY CONSTRUCTION (Gemini free read);
prelim_contains_gold is a HEURISTIC, NOT the pinned qwen3-max judge -- the answers file is
judge-ready for the moment a funded key exists. Nothing here is merged into the benchmark
scoreboard.

    .venv/bin/python -m bench.notebooklm_freetier_report \
        artifacts/holdout_rotation_r14_codex/notebooklm_freetier.jsonl \
        --out artifacts/public_ship/notebooklm_freetier_r14.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .notebooklm_freetier_run import prelim_contains


def build(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    for r in ok:  # re-score with the current heuristic
        r["prelim_contains_gold"] = prelim_contains(r["gold"], r.get("nb_answer", ""))
    n = len(ok)
    cats = Counter(r.get("category") for r in ok)
    cat_hits = Counter(r.get("category") for r in ok if r["prelim_contains_gold"])
    ground = [r.get("grounding") or {} for r in ok]
    cited = [r.get("cited_sources") or {} for r in ok]
    cov = [g.get("answer_token_coverage") for g in ground
           if isinstance(g.get("answer_token_coverage"), (int, float))]
    unmatched_rows = sum(1 for g in ground if (g.get("quotes_unmatched") or 0) > 0)
    return {
        "collection": path.as_posix(),
        "n_answered": n,
        "n_errors": len(err),
        "caller_llm_tokens_total": 0,
        "caller_tokens_basis": "BY CONSTRUCTION -- every read ran on NotebookLM/Gemini "
                               "free tier; no metered key was set in the environment",
        "prelim_contains_gold": {
            "rate": round(sum(r["prelim_contains_gold"] for r in ok) / n, 3) if n else None,
            "by_category": {c: f"{cat_hits.get(c, 0)}/{cats[c]}" for c in sorted(cats)},
            "label": "HEURISTIC (prefix-tolerant gold-token containment). NOT the pinned "
                     "qwen3-max judge; NOT comparable to the benchmark scoreboard; the "
                     "jsonl is judge-ready for a funded key.",
        },
        "grounding": {
            "rows_with_any_unmatched_quote": unmatched_rows,
            "mean_answer_token_coverage": round(statistics.mean(cov), 3) if cov else None,
            "total_quotes": {
                "verbatim": sum(g.get("quotes_verbatim") or 0 for g in ground),
                "high_overlap": sum(g.get("quotes_high_overlap") or 0 for g in ground),
                "unmatched": sum(g.get("quotes_unmatched") or 0 for g in ground),
            },
            "label": "deterministic lexical check vs exported bytes; NOT NLI, NOT the gate",
        },
        "cited_sources": {
            "total_cited": sum(c.get("cited") or 0 for c in cited),
            "total_confirmed_in_eidetic": sum(c.get("confirmed_in_eidetic") or 0 for c in cited),
            "note": "confirmed = resolves to a real immutable record by content hash. "
                    "Packed raw-record sources are often cited by BODY text (no header "
                    "token in the quote), so quote-grounding above is the provenance "
                    "signal for those rows.",
        },
        "latency_s": {
            "median": round(statistics.median([r["latency_s"] for r in ok]), 1) if ok else None,
            "p90": round(sorted(r["latency_s"] for r in ok)[int(0.9 * n)], 1) if n else None,
        },
        "honest_boundaries": [
            "single window, single run -- NOT a multi-run gate; no SOTA/best claim",
            "Gemini-side answers -- NOT eidetic verify-or-abstain",
            "NOT a row in the fixed-qwen-reader benchmark table",
        ],
    }


def render_md(rep: dict) -> str:
    p = rep["prelim_contains_gold"]
    g = rep["grounding"]
    c = rep["cited_sources"]
    lines = [
        "# NotebookLM free-tier live collection — labeled report", "",
        f"Collection: `{rep['collection']}` — n={rep['n_answered']} answered, "
        f"{rep['n_errors']} errors, **caller LLM tokens: 0** ({rep['caller_tokens_basis']}).", "",
        f"## Preliminary containment (heuristic — NOT the judge)", "",
        f"- rate: **{p['rate']}**  ({p['label']})",
        f"- by category: {json.dumps(p['by_category'])}", "",
        "## Deterministic grounding", "",
        f"- rows with any unmatched (fabricated/altered) quote: **{g['rows_with_any_unmatched_quote']}**",
        f"- quotes: {json.dumps(g['total_quotes'])}",
        f"- mean answer token coverage: {g['mean_answer_token_coverage']}",
        f"- {g['label']}", "",
        "## Citation confirmation", "",
        f"- cited {c['total_cited']}, confirmed-in-eidetic {c['total_confirmed_in_eidetic']}",
        f"- {c['note']}", "",
        f"Latency: median {rep['latency_s']['median']}s, p90 {rep['latency_s']['p90']}s.", "",
        "## Honest boundaries", "",
    ] + [f"- {b}" for b in rep["honest_boundaries"]]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = build(Path(args.jsonl))
    print(json.dumps(rep, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_md(rep) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
