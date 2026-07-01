"""Calibrate retrieval/abstention thresholds from a held-out DEV subset of REAL scored
questions. Two distinct, honestly-named procedures:

  * `precision_threshold` (was `conformal_threshold`): the lowest signal cutoff whose
    answered-set precision clears a target (~95%). This is a useful operating-point
    selector, but it is NOT split-conformal and carries NO distribution-free coverage
    guarantee. -> writes ABSTENTION_THRESHOLD.
  * `split_conformal_calibrate`: the genuine split-conformal q_hat (eidetic.optim.conformal),
    the ceil((n+1)(1-alpha))/n empirical quantile of the nonconformity scores. Gives the
    distribution-free coverage TARGET (caveat: retrieval scores are not exchangeable at
    finite depth, so it is a calibrated target, not a proof). -> writes CONFORMAL_QHAT.

Both are pure math on logged (signal, correct) pairs -- no model call, no fabricated score.
Integrity wall: only DEV-split log rows are used (see --split). Producing the scored logs
needs a funded key; computing the thresholds from them does not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from eidetic.abstention import pick_tau
from eidetic.optim.conformal import calibrate_qhat_from_pairs

from .datasets import split_of
from .fingerprints import log_fingerprint
from .harness import load_logs


def split_conformal_calibrate(samples: list[dict], alpha: float = 0.1) -> dict:
    """Genuine split-conformal q_hat for retrieval depth/abstention. samples =
    [{'signal': float, 'correct': bool}]; we calibrate on the CORRECT rows (where the true
    evidence was present), treating the coverage signal as the evidence-chunk similarity
    proxy until the harness logs the answer-chunk sim directly. Returns q_hat + the implied
    similarity cutoff for CONFORMAL_QHAT."""
    pairs = [{"answer_sim": float(s["signal"])} for s in samples
             if "signal" in s and bool(s.get("correct"))]
    if not pairs:
        return {"ok": False, "note": "no correct calibration rows with a coverage signal"}
    res = calibrate_qhat_from_pairs(pairs, alpha=alpha, sim_key="answer_sim")
    res["n_correct"] = len(pairs)
    res["signal_proxy"] = "coverage (max dense sim); replace with answer-chunk sim when logged"
    return res


def conformal_threshold(samples: list[dict], target_precision: float = 0.95) -> dict:
    """Precision-target operating-point selector (kept under its historical name for
    back-compat). NOT split-conformal -- no distribution-free coverage guarantee; use
    split_conformal_calibrate for that.

    samples = [{'signal': float, 'correct': bool}] from REAL scored calibration questions.
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


def calibrate_abstention_v2(samples: list[dict], precision_target: float = 0.95) -> dict:
    """Fit ABSTENTION_V2_TAU from logged confidence/correct pairs.

    `samples` = [{'confidence': float, 'correct': bool}] from the DEV split only. The returned tau
    is the lowest confidence cutoff whose answered set reaches the requested precision, maximizing
    coverage subject to that precision. If no cutoff reaches the target, pick_tau returns just above
    the max confidence, which safely abstains everything.
    """
    pts = [(float(s["confidence"]), bool(s["correct"])) for s in samples if "confidence" in s]
    if not pts:
        return {"ok": False, "note": "no calibration samples with a 'confidence' field"}
    tau = pick_tau(pts, precision_target=precision_target)
    answered = [ok for conf, ok in pts if conf >= tau]
    precision = (sum(answered) / len(answered)) if answered else 0.0
    coverage = len(answered) / len(pts)
    res = {
        "ok": True,
        "tau": float(tau),
        "n": len(pts),
        "precision_at_tau": precision,
        "coverage_at_tau": coverage,
        "target": precision_target,
    }
    if not answered:
        res["note"] = "target precision not reachable; tau abstains every item"
    return res


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


def _v2_samples_from_logs(rows: list[dict], system: str, split: str = "dev") -> tuple[list[dict], int]:
    """Extract (confidence, correct) pairs for ABSTENTION_V2 from answerable DEV rows."""
    out, excluded = [], 0
    for r in rows:
        if r.get("system") != system:
            continue
        cat = str(r.get("category", "")).lower()
        if "abstention" in cat or "unanswerable" in cat:
            continue
        sid = r.get("sample_id")
        if split in ("dev", "test") and sid is not None and split_of(str(sid)) != split:
            excluded += 1
            continue
        extra = r.get("extra") or {}
        conf = extra.get("confidence")
        if conf is None:
            continue
        out.append({"confidence": float(conf), "correct": bool(r.get("correct"))})
    return out, excluded


def attach_calibration_metadata(
    report: dict,
    *,
    logs: Path,
    method: str,
    system: str,
    split: str,
    excluded_by_split: int = 0,
) -> dict:
    """Bind a calibration result to its source logs and split.

    Public release checks need to distinguish a real dev-split calibration from a magic threshold
    typed into `.env`. The raw-log fingerprint proves which scored calibration corpus produced the
    threshold; the split/system fields prove the integrity wall.
    """
    out = dict(report)
    out.update({
        "method": method,
        "system": system,
        "split": split,
        "source_logs": str(Path(logs)),
        "excluded_by_split": int(excluded_by_split),
        "log_fingerprint": log_fingerprint(Path(logs)),
    })
    if method == "abstention_v2_tau" and out.get("tau") is not None:
        out["recommended_env"] = {
            "ABSTENTION_V2": "1",
            "ABSTENTION_V2_TAU": str(out["tau"]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the abstention threshold from real logs")
    ap.add_argument("--logs", default="artifacts/bench")
    ap.add_argument("--system", default="eidetic-plus")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="integrity wall: calibrate the abstention threshold on the DEV split "
                         "only (default). 'all' disables the firewall (ad-hoc only).")
    ap.add_argument("--method", default="precision", choices=["precision", "conformal", "v2", "all"],
                    help="precision = precision-target ABSTENTION_THRESHOLD (default); "
                         "conformal = split-conformal CONFORMAL_QHAT for calibrated depth; "
                         "v2 = ABSTENTION_V2_TAU; all = write every available calibration.")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="split-conformal miscoverage level (only used with --method conformal).")
    args = ap.parse_args()

    rows = load_logs(Path(args.logs))
    exit_code = 0

    def _warn_empty(kind: str) -> None:
        print(f"No {args.split}-split scored logs with a {kind} signal found. Run the harness "
              "with a funded key first (calibration computes real thresholds from real scored "
              f"questions; it never fabricates one). Expected at: {args.logs}")

    if args.method in ("precision", "conformal", "all"):
        samples, excluded = _samples_from_logs(rows, args.system, split=args.split)
        if excluded:
            print(f"Integrity wall: excluded {excluded} non-{args.split}-split rows from calibration.")
        if not samples:
            _warn_empty("coverage")
            if args.method != "all":
                return 2
            exit_code = 2
        elif args.method in ("precision", "all"):
            res = conformal_threshold(samples, args.target)
            res = attach_calibration_metadata(
                res,
                logs=Path(args.logs),
                method="precision_threshold",
                system=args.system,
                split=args.split,
                excluded_by_split=excluded,
            )
            out = Path(args.logs) / "abstention_threshold.json"
            out.write_text(json.dumps(res, indent=2))
            print(f"Calibrated abstention threshold: {res}")
            print(f"Set ABSTENTION_THRESHOLD={res.get('threshold')} in .env to apply. Saved -> {out}")

    if args.method in ("conformal", "all") and 'samples' in locals() and samples:
        res = split_conformal_calibrate(samples, alpha=args.alpha)
        res = attach_calibration_metadata(
            res,
            logs=Path(args.logs),
            method="split_conformal_qhat",
            system=args.system,
            split=args.split,
            excluded_by_split=excluded,
        )
        out = Path(args.logs) / "conformal_qhat.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"Split-conformal calibration: {res}")
        if res.get("ok"):
            print(f"Set CONFORMAL_QHAT={res.get('qhat')} and CONFORMAL_DEPTH=1 in .env to apply. "
                  f"Saved -> {out}")
    if args.method in ("v2", "all"):
        v2_samples, excluded = _v2_samples_from_logs(rows, args.system, split=args.split)
        if excluded:
            print(f"Integrity wall: excluded {excluded} non-{args.split}-split rows from v2 calibration.")
        if not v2_samples:
            _warn_empty("confidence")
            return 2 if args.method == "v2" else exit_code or 2
        res = calibrate_abstention_v2(v2_samples, precision_target=args.target)
        res = attach_calibration_metadata(
            res,
            logs=Path(args.logs),
            method="abstention_v2_tau",
            system=args.system,
            split=args.split,
            excluded_by_split=excluded,
        )
        out = Path(args.logs) / "abstention_v2_tau.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"Calibrated ABSTENTION_V2_TAU: {res}")
        if res.get("ok"):
            print(f"Set ABSTENTION_V2_TAU={res.get('tau')} in .env to apply. Saved -> {out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
