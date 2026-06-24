"""Calibrate the abstention threshold from a held-out subset of REAL scored questions.

`conformal_threshold` is pure math on provided (signal, correct) pairs -- no model call, no
fabricated score. The CLI reads per-question logs (each carries a `coverage` signal in
`extra` + the judged `correct`), finds the lowest threshold whose answered-set precision
clears the target (~95%), and writes a suggested ABSTENTION_THRESHOLD. Producing the scored
logs needs a funded key (real run); computing the threshold from them does not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .datasets import split_of
from .harness import load_logs


def conformal_threshold(samples: list[dict], target_precision: float = 0.95) -> dict:
    """samples = [{'signal': float, 'correct': bool}] from REAL scored calibration questions.
    Returns the lowest threshold t such that precision over {signal >= t} >= target, with the
    coverage achieved. Falls back to the highest-precision operating point if none reach target."""
    pts = [(float(s["signal"]), bool(s["correct"])) for s in samples if "signal" in s]
    if not pts:
        return {"ok": False, "note": "no calibration samples with a 'signal' field"}
    n = len(pts)
    candidates = sorted({p[0] for p in pts})
    best_fallback = None
    for t in candidates:
        answered = [c for sig, c in pts if sig >= t]
        if not answered:
            continue
        precision = sum(answered) / len(answered)
        coverage = len(answered) / n
        if best_fallback is None or precision > best_fallback["precision"]:
            best_fallback = {"threshold": float(t), "precision": precision, "coverage": coverage}
        if precision >= target_precision:
            return {"ok": True, "threshold": float(t), "precision": precision,
                    "coverage": coverage, "n": n, "target": target_precision}
    return {"ok": True, "threshold": best_fallback["threshold"], "precision": best_fallback["precision"],
            "coverage": best_fallback["coverage"], "n": n, "target": target_precision,
            "note": "target precision not reachable; returning best-precision operating point"}


def _samples_from_logs(rows: list[dict], system: str, split: str = "dev") -> tuple[list[dict], int]:
    """Extract (signal, correct) calibration pairs from logs. Integrity wall: when split is
    'dev'/'test', only rows whose sample_id falls on that split are used, so an abstention
    threshold can never be fit to benchmark TEST items even if the log dir is mixed. Returns
    (samples, n_excluded_by_split)."""
    out, excluded = [], 0
    for r in rows:
        if r.get("system") != system:
            continue
        sig = (r.get("extra") or {}).get("coverage")
        if sig is None:
            continue
        sid = r.get("sample_id")
        if split in ("dev", "test") and sid is not None and split_of(str(sid)) != split:
            excluded += 1
            continue
        out.append({"signal": float(sig), "correct": bool(r.get("correct"))})
    return out, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the abstention threshold from real logs")
    ap.add_argument("--logs", default="artifacts/bench")
    ap.add_argument("--system", default="eidetic-plus")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="integrity wall: calibrate the abstention threshold on the DEV split "
                         "only (default). 'all' disables the firewall (ad-hoc only).")
    args = ap.parse_args()

    rows = load_logs(Path(args.logs))
    samples, excluded = _samples_from_logs(rows, args.system, split=args.split)
    if excluded:
        print(f"Integrity wall: excluded {excluded} non-{args.split}-split rows from calibration.")
    if not samples:
        print(f"No {args.split}-split scored logs with a coverage signal found. Run the harness "
              "with a funded key first (calibration computes a real threshold from real scored "
              "questions; it never fabricates one). Expected at: " + args.logs)
        return 2
    res = conformal_threshold(samples, args.target)
    out = Path(args.logs) / "abstention_threshold.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"Calibrated abstention threshold: {res}")
    print(f"Set ABSTENTION_THRESHOLD={res.get('threshold')} in .env to apply. Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
