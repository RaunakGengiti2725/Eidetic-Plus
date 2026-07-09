"""The >=10-run reproduce gate for the NotebookLM free-tier product row.

Pure aggregator over already-JUDGED run files (`*.judged.json` produced by
`notebooklm_freetier_report --judge`). The collection step is quota-bound and lives in
`notebooklm_freetier_run.py`; THIS renders the gate verdict the instant enough runs exist,
so nothing here spends a token or needs a key.

A run's accuracy = correct / answered on that run. The gate reports per-run accuracy, the
mean, a distribution-free spread (min/max/stdev), and a bootstrap 95% CI on the mean --
then a plain verdict: PASS only when there are >= min_runs runs AND the CI lower bound
clears the comparator. Anything less is PARTIAL, stated as such. No SOTA wording is emitted
here; that is an editorial decision for a human, gated on this verdict.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _run_accuracy(judged_path: Path) -> tuple[int, int]:
    """(correct, answered) for one judged run file."""
    d = json.loads(judged_path.read_text())
    rows = d.get("rows", [])
    answered = [r for r in rows if "error" not in r]
    correct = sum(1 for r in answered if r.get("correct"))
    return correct, len(answered)


def _bootstrap_ci(per_run_acc: list[float], *, iters: int = 10000,
                  seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the MEAN of per-run accuracies. Deterministic (fixed
    LCG seed) so the verdict is reproducible -- no Math.random equivalent leaks in."""
    # ORDER-INDEPENDENCE: the fixed-LCG index stream maps positionally onto the input, so an
    # unsorted list (argv/glob order) would give order-dependent CIs. Sort a copy first so the
    # CI depends only on the multiset of accuracies, not their file order.
    per_run_acc = sorted(per_run_acc)
    n = len(per_run_acc)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (per_run_acc[0], per_run_acc[0])
    state = seed
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF   # portable LCG
            s += per_run_acc[state % n]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (round(lo, 4), round(hi, 4))


def gate_report(judged_paths: list[Path], *, min_runs: int = 10,
                comparator_acc: float | None = None,
                comparator_name: str = "comparator",
                min_answered: int | None = None) -> dict:
    runs = []
    for p in judged_paths:
        c, a = _run_accuracy(p)
        runs.append({"file": p.name, "correct": c, "answered": a,
                     "accuracy": round(c / a, 4) if a else None})
    # COMPLETENESS FLOOR: a truncated run (quota cut it short) must NOT be weighted equally
    # with a full run -- 15 truncated questions could otherwise flip PARTIAL->PASS. Only runs
    # with answered >= floor count toward the gate; the rest are reported but excluded.
    # Default floor = 90% of the largest answered count in the set (an absolute --min-answered
    # overrides). A byte-truncated judged file is otherwise indistinguishable from a full one.
    max_answered = max((r["answered"] for r in runs), default=0)
    floor = min_answered if min_answered is not None else int(0.9 * max_answered)
    for r in runs:
        r["counts_toward_gate"] = (r["accuracy"] is not None and r["answered"] >= floor)
    excluded = [r for r in runs if not r["counts_toward_gate"] and r["accuracy"] is not None]
    accs = [r["accuracy"] for r in runs if r["counts_toward_gate"]]
    n = len(accs)
    mean = round(statistics.mean(accs), 4) if accs else None
    stdev = round(statistics.pstdev(accs), 4) if n >= 2 else 0.0
    lo, hi = _bootstrap_ci(accs) if accs else (None, None)
    has_runs = n >= min_runs
    # A single point has no real interval; require >=2 full runs for a CI-based PASS.
    ci_valid = n >= 2
    beats = (comparator_acc is None) or (ci_valid and lo is not None and lo > comparator_acc)
    verdict = "PASS" if (has_runs and beats) else "PARTIAL"
    why = []
    if not has_runs:
        why.append(f"{n}/{min_runs} full runs -- need {min_runs - n} more"
                   + (f" ({len(excluded)} truncated run(s) excluded, answered<{floor})"
                      if excluded else ""))
    if comparator_acc is not None:
        if not ci_valid:
            why.append(f"n={n} full run(s): no valid CI (need >=2) to clear {comparator_name}")
        elif not (lo is not None and lo > comparator_acc):
            why.append(f"CI lower bound {lo} does not clear {comparator_name} {comparator_acc}")
    return {
        "n_runs": n, "min_runs": min_runs, "answered_floor": floor,
        "runs_excluded_truncated": len(excluded),
        "per_run": runs,
        "mean_accuracy": mean, "stdev": stdev,
        "ci95_mean": [lo, hi] if ci_valid else None,
        "comparator": {"name": comparator_name, "accuracy": comparator_acc}
                      if comparator_acc is not None else None,
        "verdict": verdict,
        "verdict_reason": "; ".join(why) or (
            "meets run count and clears comparator CI" if comparator_acc is not None
            else f"meets run count ({n}>={min_runs}); no comparator supplied"),
        "honest_note": ("PASS here is a STATISTICAL gate (>= min_runs runs, bootstrap CI "
                        "clears the comparator). It is NOT itself a 'best/SOTA' claim -- that "
                        "wording additionally needs named-comparator breadth and remains a "
                        "human editorial decision. Different (Gemini) reader than the "
                        "fixed-qwen table; own product row."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("judged", nargs="+", help="*.judged.json run files")
    ap.add_argument("--min-runs", type=int, default=10)
    ap.add_argument("--comparator-acc", type=float, default=None)
    ap.add_argument("--comparator-name", default="comparator")
    ap.add_argument("--min-answered", type=int, default=None,
                    help="absolute completeness floor: runs with fewer answered questions are "
                         "excluded from the gate (default: 90%% of the largest run)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = gate_report([Path(p) for p in args.judged], min_runs=args.min_runs,
                      comparator_acc=args.comparator_acc,
                      comparator_name=args.comparator_name,
                      min_answered=args.min_answered)
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
