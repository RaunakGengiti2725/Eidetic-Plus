"""Rotating held-out evaluation slices over the test split.

The integrity wall (bench.datasets.split_of) already reserves a disjoint test split that no
optimizer may read. This module adds ROTATION on top of it: every evaluation epoch draws a fresh,
category-stratified slice of the test split that has never been used before, so a reported number
can never be the product of tuning against one fixed slice.

Mechanics:
  - The test-split samples are grouped by category, each group is shuffled with a seed derived
    from (dataset, epoch), and the groups are interleaved proportionally into one deterministic
    ring. Consecutive windows of the ring are the rotation slices -- disjoint by construction and
    each approximately category-balanced.
  - A committed ledger records every draw as a SHA-256 digest of the slice's sorted sample IDs
    (never the IDs themselves, so the no-holdout-leakage source audit stays meaningful). Drawing
    refuses to hand out a window twice; when the ring is exhausted the epoch advances, which
    reshuffles the ring, and the rollover is recorded.
  - The draw is written as a bench samples file ([{"dataset", "sample_id"}]) consumable by
    run_dev_ablation --samples / bench.run --samples-file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def _seed_int(dataset: str, epoch: int) -> int:
    return int(hashlib.sha256(f"{dataset}|epoch={epoch}".encode("utf-8")).hexdigest()[:16], 16)


def slice_digest(sample_ids: Iterable[str]) -> str:
    """Order-independent fingerprint of a slice. The ledger stores ONLY this."""
    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def stratified_ring(samples: list, dataset: str, epoch: int) -> list[str]:
    """Deterministic category-balanced ordering of the test-split sample IDs.

    Each category group is shuffled with the epoch seed, then groups are consumed by largest
    remaining fraction so every consecutive window mirrors the corpus category mix instead of
    clumping one category into one slice."""
    by_cat: dict[str, list[str]] = {}
    for s in samples:
        by_cat.setdefault(getattr(s, "category", "unknown"), []).append(s.sample_id)
    rng = random.Random(_seed_int(dataset, epoch))
    for cat in sorted(by_cat):
        by_cat[cat].sort()
        rng.shuffle(by_cat[cat])
    totals = {c: len(ids) for c, ids in by_cat.items()}
    taken = {c: 0 for c in by_cat}
    ring: list[str] = []
    remaining = sum(totals.values())
    while remaining:
        # next category = the one furthest behind its proportional share (ties: sorted name)
        cat = max(sorted(by_cat), key=lambda c: (totals[c] - taken[c]) / totals[c] if totals[c] else 0.0)
        if taken[cat] >= totals[cat]:
            break
        ring.append(by_cat[cat][taken[cat]])
        taken[cat] += 1
        remaining -= 1
    return ring


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "datasets": {}}


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def draw_slice(samples: list, *, dataset: str, n: int, state_path: Path,
               out_path: Optional[Path] = None) -> dict:
    """Draw the next unused rotation slice of size n from the test-split `samples`.

    Returns {"sample_ids", "epoch", "window", "digest", "rollover"}. Raises if n exceeds the
    test-split size. Every draw appends a digest entry to the ledger at state_path; a window
    whose digest is already in the ledger is never returned again."""
    if n <= 0:
        raise ValueError("slice size must be positive")
    ids_all = [s.sample_id for s in samples]
    if n > len(ids_all):
        raise ValueError(f"requested slice n={n} exceeds test split size {len(ids_all)}")
    state = _load_state(state_path)
    ds = state["datasets"].setdefault(dataset, {"epoch": 0, "next_window": 0, "draws": []})
    seen_digests = {d["digest"] for d in ds["draws"]}

    rollover = False
    while True:
        ring = stratified_ring(samples, dataset, ds["epoch"])
        windows = len(ring) // n
        if ds["next_window"] >= windows:
            # ring exhausted -> new epoch reshuffles; record the rollover honestly
            ds["epoch"] += 1
            ds["next_window"] = 0
            rollover = True
            continue
        start = ds["next_window"] * n
        chosen = ring[start:start + n]
        digest = slice_digest(chosen)
        ds["next_window"] += 1
        if digest in seen_digests:
            # possible only after an epoch rollover re-partitions the same small corpus;
            # skip forward rather than re-report a previously used slice
            continue
        break

    ds["draws"].append({
        "digest": digest, "epoch": ds["epoch"], "window": ds["next_window"] - 1,
        "n": n, "commit": _git_commit(), "rollover": rollover,
    })
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    if out_path is not None:
        rows = [{"dataset": dataset, "sample_id": sid} for sid in chosen]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n")
    return {"sample_ids": chosen, "epoch": ds["epoch"], "window": ds["next_window"] - 1,
            "digest": digest, "rollover": rollover}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["locomo", "longmemeval"])
    ap.add_argument("--n", type=int, required=True, help="slice size")
    ap.add_argument("--state", default="bench/holdout_rotation_state.json")
    ap.add_argument("--out", required=True, help="samples-file path to write")
    args = ap.parse_args()

    from bench.datasets import filter_split
    if args.dataset == "locomo":
        from bench.datasets.locomo import load
    else:
        from bench.datasets.longmemeval import load
    samples = filter_split(load(), "test")
    meta = draw_slice(samples, dataset=args.dataset, n=args.n,
                      state_path=Path(args.state), out_path=Path(args.out))
    print(json.dumps({k: meta[k] for k in ("epoch", "window", "digest", "rollover")}, indent=2))
    print(f"wrote {args.n} test-split samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
