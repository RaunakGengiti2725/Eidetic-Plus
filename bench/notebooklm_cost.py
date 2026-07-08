"""Billable-caller-token cost accounting: what does each memory system spend on the
OPERATOR'S OWN metered LLM to answer one query?

This is the honest metric behind the NotebookLM play. Two kinds of number here, never
conflated:

  MEASURED (from committed holdout logs, real qwen tokens on our DashScope meter):
    rag-vector, mem0, and eidetic's metered reader -- their `query_tokens` per row.

  BY CONSTRUCTION (a definitional property of the code, labeled as such, NOT a live
  measurement): eidetic + NotebookLM `routed_answer` spends 0 caller tokens whenever it
  takes the free-read tier, because that call goes to NotebookLM/Gemini, not to the
  caller's metered model. The structured tier is ~6-85 caller tokens; the metered-reader
  tier (~4034) is only reached under require_gate_verification.

What this is NOT: a live accuracy comparison (NotebookLM is a Gemini reader, off the
fixed-qwen benchmark -- run bench/notebooklm_costbench live with your own Google account
for that), and NOT a claim that the compute is free globally (Google spends it).

    .venv/bin/python -m bench.notebooklm_cost \
        artifacts/holdout_rotation_r12_codex artifacts/holdout_rotation_r13_codex \
        artifacts/holdout_rotation_r14_codex --out artifacts/public_ship/notebooklm_cost.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _measured(dirs: list[Path], system_file: str) -> dict:
    qt: list[float] = []
    for d in dirs:
        for r in _rows(d / system_file):
            v = r.get("query_tokens")
            if isinstance(v, (int, float)):
                qt.append(float(v))
    if not qt:
        return {"n": 0}
    return {"n": len(qt), "median": statistics.median(qt),
            "mean": round(sum(qt) / len(qt), 1), "total": int(sum(qt))}


def build_report(dirs: list[Path]) -> dict:
    """Per-query billable caller-tokens. rag-vector/mem0/eidetic-metered are MEASURED;
    the NotebookLM-routed row is BY-CONSTRUCTION (labeled)."""
    ragv = _measured(dirs, "rag-vector__run0.jsonl")
    mem0 = _measured(dirs, "mem0__run0.jsonl")
    eidetic = _measured(dirs, "eidetic-plus-full__run0.jsonl")
    # NotebookLM-routed caller-token model (by construction): free-read tier = 0, structured
    # tier ~6-85, metered tier only under require_gate_verification. Reported as a RANGE, not a
    # measured median -- because the tier mix depends on the query set and is measured live.
    return {
        "metric": "billable tokens on the OPERATOR'S OWN metered LLM, per query",
        "windows": [d.name for d in dirs],
        "systems": {
            "eidetic+notebooklm (routed, free-read tier)": {
                "caller_tokens_per_query": 0,
                "basis": "BY CONSTRUCTION -- the read runs on NotebookLM/Gemini, off the "
                         "caller's meter; routed_answer reports user_llm_tokens=0 on that tier",
                "verified": "provenance-mapped (Gemini-side, NOT gate-verified)",
            },
            "eidetic+notebooklm (routed, structured tier)": {
                "caller_tokens_per_query": "6-85",
                "basis": "structured_recall typed path (design-supplied range)",
                "verified": "gate-verified (verify-or-abstain)",
            },
            "mem0": {
                "caller_tokens_per_query": mem0.get("median"),
                "basis": "MEASURED (qwen reader tokens, committed logs)",
                "verified": "0 verified answers", **{k: mem0[k] for k in mem0 if k != "median"}},
            "rag-vector": {
                "caller_tokens_per_query": ragv.get("median"),
                "basis": "MEASURED (qwen reader tokens, committed logs)",
                "verified": "0 verified answers", **{k: ragv[k] for k in ragv if k != "median"}},
            "eidetic (metered reader, no notebooklm)": {
                "caller_tokens_per_query": eidetic.get("median"),
                "basis": "MEASURED (qwen reader tokens, committed logs)",
                "verified": "gate-verified",
                **{k: eidetic[k] for k in eidetic if k != "median"}},
        },
        "honest_claim": (
            "On billable tokens spent on the operator's own metered model, the NotebookLM "
            "free-read tier costs 0 per query -- below rag-vector's measured "
            f"~{ragv.get('median')} and mem0's ~{mem0.get('median')} -- because Google's "
            "free tier does the read. This is an operator-cost property (0 on YOUR meter), "
            "by construction; it is NOT free globally, the NotebookLM answer is Gemini-side "
            "provenance-mapped (not gate-verified), and it is NOT a row in the fixed-qwen "
            "benchmark accuracy table. Run bench/notebooklm_costbench live to prove the "
            "end-to-end head-to-head with your own Google account."),
    }


def render_md(rep: dict) -> str:
    lines = [f"# Billable caller-token cost — {rep['metric']}", "",
             f"Windows: {', '.join(rep['windows'])}", "",
             "| system | caller tokens / query | basis | verified |",
             "|---|---|---|---|"]
    for name, s in rep["systems"].items():
        lines.append(f"| {name} | {s.get('caller_tokens_per_query')} | {s['basis']} | {s['verified']} |")
    lines += ["", "## Honest claim", "", rep["honest_claim"]]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = build_report([Path(d) for d in args.dirs])
    print(json.dumps(rep, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_md(rep) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
