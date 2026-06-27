"""Snap-back fidelity audit over a real corpus (forgetting-machine plan: a number, not a demo).

For every content-addressed memory in the store, verify the immutable substrate still returns the
byte-identical original (sha256(get_raw(h)) == h). Forgetting/fading lowers only FSRS index
priority; the raw record is never mutated, so a faded memory snaps back losslessly. Run AFTER a
benchmark slice to report the guarantee over the actual ingested corpus.

    DATA_DIR=data/proof_multisession_h2h .venv/bin/python scripts/snap_back_audit.py
"""
from __future__ import annotations

import json
import sys

from eidetic.config import get_settings
from eidetic.engine import Engine


def main() -> int:
    settings = get_settings()
    eng = Engine(settings)
    audit = eng.snap_back_audit()
    pct = audit["rate"] * 100.0
    print(json.dumps({
        "data_dir": str(settings.data_dir),
        "records_with_raw_blob": audit["total"],
        "lossless_byte_identical": audit["lossless"],
        "rate_pct": round(pct, 4),
        "failures": audit["failures"][:20],
    }, indent=2))
    # Non-zero exit if ANY record failed to snap back -- the guarantee is 100% or it is a bug.
    return 0 if audit["rate"] >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
