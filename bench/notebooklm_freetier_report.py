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
import re
import statistics
from collections import Counter
from pathlib import Path

from .notebooklm_freetier_run import prelim_contains

# CONFIRMED LoCoMo gold-label defects (extraction/judge fleet 2026-07-09, each independently
# re-verified against locomo10.json + live stores; evidence in
# artifacts/forensics/extraction_judge_fleet_20260709.json): 3 speaker misattributions
# (c4_q86, c9_q137, c7_q116 -- the gold's own evidence turn is spoken by a DIFFERENT person
# than the question's subject), 1 temporal-template instantiation bug (c5_q81), 1 annotator
# arithmetic error (c6_q64). These are DATASET defects no memory system can or should
# "recover"; judge v2 scores them into a separate quarantine bucket -- never silently
# excluded, never counted against any system.
QUARANTINED_GOLD_DEFECTS = frozenset({"c4_q86", "c5_q81", "c6_q64", "c9_q137", "c7_q116"})


# An insufficiency answer can still CONTAIN the gold's tokens in its "topics I did find"
# tour; a date-shaped gold can be token-contained by an answer asserting a DIFFERENT date
# (both false-positive shapes measured on the replay: c6_q29 tz-shifted date, gpt4_85da3956
# "could not find"). Neither may ever short-circuit -- they go to the LLM judge.
_V2_INSUFFICIENT_RE = re.compile(
    r"\b(?:no\s+information|not\s+(?:mentioned|specified|stated|provided|available)|"
    r"could\s+not\s+find|cannot\s+(?:determine|find|answer)|"
    r"unable\s+to\s+(?:determine|find|answer)|insufficient\s+(?:information|context))\b",
    re.I)
_V2_TEMPORAL_GOLD_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b|\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|"
    r"months?|years?)\b|\bago\b|\byesterday\b|\btomorrow\b", re.I)


def _gold_containment_correct(gold: str, answer: str) -> bool:
    """Deterministic judge pre-check (judge v2): item-level gold coverage on the FULL answer.
    The gold splits on commas/conjunctions into items; the answer is correct-by-containment
    when EVERY item's content tokens appear (prefix-tolerant, so 'candles' matches 'candle').
    Store-grounded EXTRAS never penalize -- the fleet's confirmed judge false-negatives were
    all superset/paraphrase answers ('candles and essential oils' + a music section judged
    wrong against gold 'candles, music, essential oils'). Fails closed on empty golds,
    insufficiency answers, and temporal-shaped golds (dates/durations demand value equality
    that containment cannot prove -- replay-measured false positives otherwise)."""
    if _V2_INSUFFICIENT_RE.search(answer or ""):
        return False
    if _V2_TEMPORAL_GOLD_RE.search(str(gold or "")):
        return False
    items = [p.strip() for p in re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+", str(gold or ""))
             if p.strip()]
    if not items:
        return False
    return all(prelim_contains(item, answer or "") for item in items)


def judge_rows_v2(path: Path, judge=None) -> dict:
    """Judge v2 = pinned qwen3-max PLUS two deterministic hardening layers, shipped as a NEW
    judge version with its own sidecar (.judged_v2.json) -- v1 sidecars and every published
    number stay untouched; comparisons across versions must re-judge both sides.

    Layer 1 QUARANTINE: rows in QUARANTINED_GOLD_DEFECTS score into a separate
    'quarantined_gold_defects' bucket, excluded from the accuracy denominator and reported.
    Layer 2 PRE-CHECK: deterministic item-level gold containment on the FULL answer
    short-circuits to correct=True (no LLM call, no LLM flakiness on superset answers);
    everything else falls through to the pinned LLM judge unchanged."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    live = [r for r in ok if r["sample_id"] not in QUARANTINED_GOLD_DEFECTS]
    quarantined = [r["sample_id"] for r in ok if r["sample_id"] in QUARANTINED_GOLD_DEFECTS]
    per = []
    correct = 0
    shortcircuit = 0
    needs_llm = []
    for r in live:
        if _gold_containment_correct(str(r.get("gold") or ""), str(r.get("nb_answer") or "")):
            per.append({"sample_id": r["sample_id"], "correct": True,
                        "category": r.get("category"), "via": "gold-containment"})
            correct += 1
            shortcircuit += 1
        else:
            needs_llm.append(r)
    if needs_llm:
        if judge is None:
            import os
            if not os.environ.get("DASHSCOPE_API_KEY"):
                return {"judged": False, "judge_version": "v2",
                        "reason": "DASHSCOPE_API_KEY not set; deterministic layer alone "
                                  f"decided {shortcircuit} rows, {len(needs_llm)} need the "
                                  "pinned LLM judge",
                        "shortcircuit_correct": shortcircuit,
                        "quarantined_gold_defects": quarantined}
            from .judge import Judge
            judge = Judge()
        for r in needs_llm:
            c = bool(judge.judge_locomo(r["question"], r["gold"], r.get("nb_answer", "")))
            per.append({"sample_id": r["sample_id"], "correct": c,
                        "category": r.get("category"), "via": "llm-judge"})
            correct += c
    out = {"judged": True, "judge_version": "v2",
           "judge": "pinned qwen3-max + deterministic gold-containment pre-check + "
                    "gold-defect quarantine (fleet-verified 2026-07-09)",
           "n": len(live), "correct": correct,
           "accuracy": round(correct / len(live), 3) if live else None,
           "shortcircuit_correct": shortcircuit,
           "quarantined_gold_defects": quarantined,
           "rows": per,
           "label": ("judge v2 accuracy: NOT comparable to v1-judged rows -- re-judge both "
                     "sides of any comparison with the same version. Quarantined rows are "
                     "confirmed dataset defects, reported, never counted for any system.")}
    side = path.with_suffix(".judged_v2.json")
    side.write_text(json.dumps(out, indent=2))
    return out


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


def judge_rows(path: Path, judge=None) -> dict:
    """THE decisive scoring step -- runs the SAME pinned qwen3-max judge the benchmark
    uses over the collected free-tier answers. Needs a funded DASHSCOPE key; refuses with
    a plain message otherwise. Writes a `judged` sidecar so the result is durable. This is
    the ONLY path that turns the collection into a judge-scored accuracy number; the
    heuristic rate above never substitutes for it."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    if judge is None:
        import os
        if not os.environ.get("DASHSCOPE_API_KEY"):
            return {"judged": False,
                    "reason": "DASHSCOPE_API_KEY not set -- the pinned qwen3-max judge "
                              "cannot run. Collection is judge-ready; set the key and "
                              "re-run with --judge."}
        from .judge import Judge
        judge = Judge()
    correct = 0
    per = []
    for r in ok:
        c = bool(judge.judge_locomo(r["question"], r["gold"], r.get("nb_answer", "")))
        per.append({"sample_id": r["sample_id"], "correct": c, "category": r.get("category")})
        correct += c
    out = {"judged": True, "judge": "pinned bench judge (same as scoreboard rows)",
           "n": len(ok), "correct": correct,
           "accuracy": round(correct / len(ok), 3) if ok else None,
           "rows": per,
           "label": ("judge-scored accuracy of the NotebookLM free-read tier on this "
                     "window. Different READER than the fixed-qwen table (off-meter "
                     "Gemini), so present it as its own labeled row, never merged into "
                     "the fixed-reader comparison.")}
    side = path.with_suffix(".judged.json")
    side.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--out")
    ap.add_argument("--judge", action="store_true",
                    help="ALSO judge-score with the pinned bench judge (needs DASHSCOPE key)")
    ap.add_argument("--judge-v2", action="store_true",
                    help="ALSO judge-score with judge v2 (quarantine + deterministic "
                         "gold-containment pre-check; own .judged_v2.json sidecar; NOT "
                         "comparable to v1 rows)")
    args = ap.parse_args()
    rep = build(Path(args.jsonl))
    if args.judge:
        rep["judge_scored"] = judge_rows(Path(args.jsonl))
    if args.judge_v2:
        rep["judge_scored_v2"] = judge_rows_v2(Path(args.jsonl))
    print(json.dumps(rep, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_md(rep) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
