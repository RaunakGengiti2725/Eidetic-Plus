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
                comparator_name: str = "comparator") -> dict:
    runs = []
    for p in judged_paths:
        c, a = _run_accuracy(p)
        runs.append({"file": p.name, "correct": c, "answered": a,
                     "accuracy": round(c / a, 4) if a else None})
    accs = [r["accuracy"] for r in runs if r["accuracy"] is not None]
    n = len(accs)
    mean = round(statistics.mean(accs), 4) if accs else None
    stdev = round(statistics.pstdev(accs), 4) if n >= 2 else 0.0
    lo, hi = _bootstrap_ci(accs) if accs else (None, None)
    has_runs = n >= min_runs
    beats = (comparator_acc is None) or (lo is not None and lo > comparator_acc)
    verdict = "PASS" if (has_runs and beats) else "PARTIAL"
    why = []
    if not has_runs:
        why.append(f"{n}/{min_runs} runs -- need {min_runs - n} more")
    if comparator_acc is not None and not (lo is not None and lo > comparator_acc):
        why.append(f"CI lower bound {lo} does not clear {comparator_name} {comparator_acc}")
    return {
        "n_runs": n, "min_runs": min_runs,
        "per_run": runs,
        "mean_accuracy": mean, "stdev": stdev,
        "ci95_mean": [lo, hi],
        "comparator": {"name": comparator_name, "accuracy": comparator_acc}
                      if comparator_acc is not None else None,
        "verdict": verdict,
        "verdict_reason": "; ".join(why) or "meets run count and clears comparator CI",
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
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = gate_report([Path(p) for p in args.judged], min_runs=args.min_runs,
                      comparator_acc=args.comparator_acc,
                      comparator_name=args.comparator_name)
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
