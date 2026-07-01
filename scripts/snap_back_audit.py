#!/usr/bin/env python
"""Snap-back fidelity audit over a real corpus (forgetting-machine plan: a number, not a demo).

For every content-addressed memory in the store, verify the immutable substrate still returns the
byte-identical original (sha256(get_raw(h)) == h). Forgetting/fading lowers only FSRS index
priority; the raw record is never mutated, so a faded memory snaps back losslessly. Run AFTER a
benchmark slice to report the guarantee over the actual ingested corpus.

    DATA_DIR=data/proof_multisession_h2h .venv/bin/python scripts/snap_back_audit.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eidetic.config import get_settings
from eidetic.engine import Engine


def build_report(audit: dict, data_dir: Path, *, min_records: int = 1) -> dict:
    total = int(audit.get("total", 0) or 0)
    lossless = int(audit.get("lossless", 0) or 0)
    rate = float(audit.get("rate", 0.0) or 0.0)
    raw_failures = audit.get("failures", []) or []
    failures = raw_failures if isinstance(raw_failures, list) else [{"error": "malformed_failures"}]
    audited_hashes = [
        str(h).strip()
        for h in (audit.get("audited_content_hashes", []) or [])
        if str(h).strip()
    ]
    ok = total >= min_records and lossless == total and rate >= 1.0 and not failures
    return {
        "status": "PASS" if ok else "FAIL",
        "data_dir": str(Path(data_dir).resolve()),
        "records_with_raw_blob": total,
        "lossless_byte_identical": lossless,
        "rate": rate,
        "rate_pct": round(rate * 100.0, 4),
        "min_records": int(min_records),
        "audited_content_hashes": sorted(dict.fromkeys(audited_hashes)),
        "failures": failures[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit immutable raw-memory snap-back fidelity")
    ap.add_argument("--out", help="optional JSON report path")
    ap.add_argument("--min-records", type=int, default=1,
                    help="minimum raw-backed records required for a passing release audit")
    args = ap.parse_args()

    settings = get_settings()
    eng = Engine(settings)
    report = build_report(eng.snap_back_audit(), settings.data_dir, min_records=args.min_records)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    # Non-zero exit if ANY record failed to snap back -- the guarantee is 100% or it is a bug.
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
