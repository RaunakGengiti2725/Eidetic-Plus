#!/usr/bin/env python3
"""Seed a small, vivid demo corpus into the LIVE store so the web UI has content.

This uses the REAL Engine against the default DATA_DIR (./data) and makes REAL
embedding + extraction calls. It is NOT mocked: if no DashScope key is configured
it exits loudly (code 2) rather than fabricating anything.

Highlights of the seeded corpus:
  * A bi-temporal CONTRADICTION pair on the SAME subject+relation across time
    (Alice's employer a year ago vs. now), so the graph shows the old edge
    invalidated but retained.
  * A user-preference memory (window seats).
  * A couple of factual project memories with concrete numbers (good targets for
    Ask + NLI entailment verification).
  * One clearly time-stamped event.

Run:
    DATA_DIR=./data DASHSCOPE_API_KEY=... python scripts/seed_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the project importable regardless of where this is run from.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eidetic.config import get_settings  # noqa: E402
from eidetic.engine import Engine  # noqa: E402

DAY = 86400.0


def main() -> int:
    settings = get_settings()

    # REAL calls only. No key -> stop loudly, never mock.
    if not settings.has_api_key:
        print(
            "ERROR: no DashScope API key configured.\n"
            "  seed_demo.py makes REAL embedding + extraction calls and never mocks.\n"
            "  Set DASHSCOPE_API_KEY in your environment or in "
            f"{_PROJECT_ROOT}/.env and re-run.",
            file=sys.stderr,
        )
        return 2

    engine = Engine()
    now = time.time()

    # (text, source, valid_at) tuples. valid_at = now - days*DAY for older facts.
    memories = [
        # --- Bi-temporal contradiction pair (SAME subject + relation over time) ---
        # Old fact, ~1 year ago: should end up invalidated-but-retained.
        (
            "Alice works at Acme Corp as a data engineer.",
            "hr-records",
            now - 365 * DAY,
        ),
        # New fact, now: supersedes the old employer edge.
        (
            "Alice now works at Globex as a staff engineer.",
            "hr-records",
            now,
        ),

        # --- User preference ---
        (
            "The user prefers window seats when booking flights.",
            "travel-profile",
            now - 30 * DAY,
        ),

        # --- Factual project memories with concrete numbers (Ask + NLI targets) ---
        (
            "Project Eidetic reached 1024-dimensional embeddings and indexed "
            "12,500 memories in the latest benchmark run.",
            "engineering-notes",
            now - 14 * DAY,
        ),
        (
            "The signature demo simulated 30 years of memories and held recall@10 "
            "flat at 0.97 with p95 retrieval latency near 42 ms regardless of age.",
            "engineering-notes",
            now - 7 * DAY,
        ),
        (
            "The retrieval pipeline runs ANN top-100, fuses with Personalized "
            "PageRank via reciprocal rank fusion, then reranks to the final top-10.",
            "architecture-doc",
            now - 21 * DAY,
        ),

        # --- Clearly time-stamped event ---
        (
            "On 2026-03-15 the team shipped Eidetic-Plus v1.0.0 to production in "
            "the Singapore region.",
            "release-log",
            now - 100 * DAY,
        ),

        # --- A few more vivid facts to give the graph some texture ---
        (
            "Bob is the project lead for Eidetic-Plus and reports to Alice.",
            "org-chart",
            now - 60 * DAY,
        ),
        (
            "The immutable substrate stores every raw record as a write-once "
            "sha256-addressed blob and never deletes it.",
            "architecture-doc",
            now - 45 * DAY,
        ),
        (
            "Carol joined the Eidetic-Plus team as a research scientist focused on "
            "FSRS forgetting curves.",
            "hr-records",
            now - 5 * DAY,
        ),
    ]

    print(f"Seeding {len(memories)} demo memories into {settings.data_dir} ...\n")

    seeded = 0
    for i, (text, source, valid_at) in enumerate(memories, start=1):
        age_days = (now - valid_at) / DAY
        print(f"[{i:>2}/{len(memories)}] ({source}, valid {age_days:.0f}d ago) {text}")
        record = engine.ingest_text(
            text,
            source=source,
            valid_at=valid_at,
            extract_graph=True,
        )
        seeded += 1
        ents = ", ".join(record.entities) if record.entities else "-"
        print(
            f"        -> memory_id={record.memory_id} "
            f"salience={record.salience:.2f} entities=[{ents}]"
        )

    print(f"\nSeeded {seeded} memories.\n")
    print("Engine stats:")
    for k, v in engine.stats().items():
        print(f"  {k}: {v}")

    print(
        '\nOpen http://localhost:8000 and try asking '
        '"Where does Alice work now?" and "What seat does the user prefer?".'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
