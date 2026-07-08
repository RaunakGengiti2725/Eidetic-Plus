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


def _measured_struct_tier(dirs: list[Path]) -> dict:
    """MEASURED caller tokens for the rows the structured (SMQE) tier actually answered in
    the committed holdout logs -- the rows routed_answer's Tier 1 takes. Filter:
    extra.policy contains 'smqe'. Real per-row `query_tokens` from the runs, so the
    structured tier's cost band is measured, not design-supplied."""
    qt: list[float] = []
    for d in dirs:
        for r in _rows(d / "eidetic-plus-full__run0.jsonl"):
            pol = ((r.get("extra") or {}).get("policy") or "")
            if "smqe" in pol:
                v = r.get("query_tokens")
                if isinstance(v, (int, float)):
                    qt.append(float(v))
    if not qt:
        return {"n": 0}
    return {"n": len(qt), "median": statistics.median(qt),
            "mean": round(sum(qt) / len(qt), 1), "max": max(qt), "min": min(qt)}


def build_report(dirs: list[Path]) -> dict:
    """Per-query billable caller-tokens. rag-vector/mem0/eidetic-metered are MEASURED;
    the NotebookLM-routed row is BY-CONSTRUCTION (labeled)."""
    ragv = _measured(dirs, "rag-vector__run0.jsonl")
    mem0 = _measured(dirs, "mem0__run0.jsonl")
    eidetic = _measured(dirs, "eidetic-plus-full__run0.jsonl")
    struct = _measured_struct_tier(dirs)
    # NotebookLM-routed caller-token model: free-read tier = 0 BY CONSTRUCTION; structured
    # tier MEASURED from the smqe-answered rows in the same committed logs; metered tier only
    # under require_gate_verification. The TIER MIX on an arbitrary query stream is still
    # unmeasured -- we report per-tier costs, never a blended figure.
    struct_cost = (f"median {struct.get('median')}, max {struct.get('max')} "
                   f"(n={struct.get('n')})") if struct.get("n") else "no smqe rows in logs"
    return {
        "metric": "billable tokens on the OPERATOR'S OWN metered LLM, per query",
        "windows": [d.name for d in dirs],
        "systems": {
            "eidetic+notebooklm (routed, free-read tier)": {
                "caller_tokens_per_query": 0,
                "basis": "BY CONSTRUCTION -- the read runs on NotebookLM/Gemini, off the "
                         "caller's meter; routed_answer reports user_llm_tokens=0 on that tier",
                "verified": "provenance-mapped + deterministic grounding check "
                            "(Gemini-side, NOT gate-verified)",
            },
            "eidetic+notebooklm (routed, structured tier)": {
                "caller_tokens_per_query": struct_cost,
                "basis": "MEASURED -- query_tokens of the smqe-answered rows in the same "
                         "committed holdout logs (the rows Tier 1 takes)",
                "verified": "gate-verified (verify-or-abstain)",
                **struct,
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
            "Under free-read routing (require_gate_verification=False) a query costs the "
            "caller EITHER a structured gate-verified answer -- MEASURED median "
            f"{struct.get('median')}, worst-case {struct.get('max')} tokens on the "
            f"n={struct.get('n')} smqe-answered rows of these windows -- OR a NotebookLM "
            "free read at 0 caller tokens (by construction). Both are below mem0's measured "
            f"median ~{mem0.get('median')} and rag-vector's ~{ragv.get('median')}. Honest "
            "scope: the tier MIX on an arbitrary query stream is unmeasured (per-tier costs "
            "only, no blended figure); this is an operator-cost property, NOT free globally; "
            "the free-read answer is Gemini-side provenance-mapped + deterministically "
            "grounded (lexical), NOT gate-verified; and this is NOT a row in the fixed-qwen "
            "benchmark accuracy table. Accuracy on the free-read tier is unmeasured."),
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
