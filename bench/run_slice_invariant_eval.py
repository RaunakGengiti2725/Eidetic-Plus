"""Run repeated stratified benchmark slices and write a release-gate sidecar."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .seed_policy import resolve_seed
from .run import load_samples


def _draw(samples: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    for sample in samples:
        buckets[str(sample.category)].append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    cats = sorted(buckets)
    out = []
    while len(out) < n and any(buckets.values()):
        for cat in cats:
            if buckets[cat] and len(out) < n:
                out.append(buckets[cat].pop())
    return out


def _scoreboard_pass(scoreboard_path: Path, system: str, required: int) -> dict:
    if not scoreboard_path.exists():
        return {"pass": False, "detail": "scoreboard.json missing"}
    data = json.loads(scoreboard_path.read_text())
    integrity = data.get("integrity") if isinstance(data.get("integrity"), dict) else {}
    for name, row in integrity.items():
        if str(name).lower() != system.lower() or not isinstance(row, dict):
            continue
        total = int(row.get("n", row.get("total", 0)) or 0)
        verified_correct = int(row.get("verified_correct", 0) or 0)
        has_verify = bool(row.get("has_verify", verified_correct > 0))
        passed = has_verify and total >= required and verified_correct >= total
        return {
            "pass": passed,
            "verified": has_verify,
            "verified_correct": verified_correct,
            "correct": verified_correct,
            "total": total,
            "verified_accuracy": (verified_correct / total) if total else 0.0,
        }
    rows = data.get("systems") or data.get("rows") or []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and str(row.get("system", "")).lower() == system.lower():
            total = int(row.get("total", row.get("n", required)) or required)
            verified_correct = int(row.get("verified_correct", 0) or 0)
            has_verify = bool(row.get("verified", row.get("has_verify", verified_correct > 0)))
            has_verified_score = has_verify and "verified_correct" in row
            passed = has_verified_score and total >= required and verified_correct >= total
            return {
                "pass": passed,
                "verified": has_verified_score,
                "verified_correct": verified_correct,
                "correct": verified_correct,
                "total": total,
                "verified_accuracy": (verified_correct / total) if total else 0.0,
                "detail": "verified scoreboard row" if has_verified_score else "scoreboard row lacks verified_correct proof",
            }
    return {"pass": False, "detail": f"system {system} not found"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="longmemeval", choices=["longmemeval", "locomo"])
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--subset", type=int, default=24)
    ap.add_argument("--seed", default=None, help="integer seed for reproducibility; omitted/auto/random draws a fresh seed")
    ap.add_argument("--systems", default="eidetic-full")
    ap.add_argument("--system-under-test", default="eidetic-plus-full")
    ap.add_argument("--out", default="artifacts/slice_invariant")
    ap.add_argument("--plan-only", action="store_true", help="write sample files and sidecar without executing runs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed, seed_mode = resolve_seed(args.seed)
    pool = load_samples(args.dataset, 0, args.variant, 0, split=args.split, sample_strategy="contiguous")
    pool_unique_sample_ids = len({s.sample_id for s in pool})
    required_unique_sample_ids = args.draws * args.subset
    if pool_unique_sample_ids < required_unique_sample_ids:
        sidecar = {
            "dataset": args.dataset,
            "split": args.split,
            "holdout_profile": "holdout",
            "draws": args.draws,
            "subset": args.subset,
            "seed": seed,
            "seed_mode": seed_mode,
            "draw_seeds": [seed + i for i in range(args.draws)],
            "pool_unique_sample_ids": pool_unique_sample_ids,
            "required_unique_sample_ids": required_unique_sample_ids,
            "unique_sample_ids": pool_unique_sample_ids,
            "system_under_test": args.system_under_test,
            "pass": False,
            "failures": [
                f"insufficient_pool_unique_samples:{pool_unique_sample_ids}<required:{required_unique_sample_ids}"
            ],
            "runs": [],
        }
        (out / "slice_invariant.json").write_text(json.dumps(sidecar, indent=2))
        print(json.dumps(sidecar, indent=2))
        return 0 if args.plan_only else 1
    runs = []
    available = list(pool)
    used_ids: set[str] = set()
    for i in range(args.draws):
        draw_seed = seed + i
        draw = _draw(available, args.subset, draw_seed)
        draw_ids = {s.sample_id for s in draw}
        used_ids.update(draw_ids)
        available = [s for s in available if s.sample_id not in draw_ids]
        samples_file = out / f"draw_{i + 1}.samples.json"
        samples_file.write_text(json.dumps([
            {"dataset": s.dataset, "sample_id": s.sample_id} for s in draw
        ], indent=2))
        draw_out = out / f"draw_{i + 1}"
        cmd = [
            sys.executable, "-m", "bench.run",
            "--systems", args.systems,
            "--dataset", args.dataset,
            "--variant", args.variant,
            "--split", args.split,
            "--samples-file", str(samples_file),
            "--holdout-profile", "holdout",
            "--out", str(draw_out),
            "--runs", "1",
            "--overwrite",
        ]
        entry = {
            "draw": i + 1,
            "seed": draw_seed,
            "samples_file": str(samples_file),
            "sample_ids": [s.sample_id for s in draw],
            "command": cmd,
            "executed": not args.plan_only,
        }
        if not args.plan_only:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            entry["returncode"] = proc.returncode
            entry["output_tail"] = proc.stdout[-4000:]
            entry["score"] = _scoreboard_pass(draw_out / "scoreboard.json", args.system_under_test, args.subset)
        runs.append(entry)
    failures = []
    for run in runs:
        if not run.get("executed"):
            failures.append(f"draw{run.get('draw')}:not_executed")
            continue
        rc = run.get("returncode", 1)
        if int(rc if rc is not None else 1) != 0:
            failures.append(f"draw{run.get('draw')}:returncode:{run.get('returncode')}")
            continue
        score = run.get("score") if isinstance(run.get("score"), dict) else {}
        if not score.get("pass"):
            failures.append(f"draw{run.get('draw')}:score:false")
    sidecar = {
        "dataset": args.dataset,
        "split": args.split,
        "holdout_profile": "holdout",
        "draws": args.draws,
        "subset": args.subset,
        "seed": seed,
        "seed_mode": seed_mode,
        "draw_seeds": [r["seed"] for r in runs],
        "pool_unique_sample_ids": pool_unique_sample_ids,
        "required_unique_sample_ids": required_unique_sample_ids,
        "unique_sample_ids": len(used_ids),
        "system_under_test": args.system_under_test,
        "pass": not failures,
        "failures": failures,
        "runs": runs,
    }
    (out / "slice_invariant.json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))
    return 0 if sidecar["pass"] or args.plan_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
