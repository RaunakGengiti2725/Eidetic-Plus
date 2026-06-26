"""THE photographic-memory demo: every spoken answer points back to immutable bytes.

This is the "scream photographic memory" surface from the benchmark-dominance plan. For one
question it prints, side by side:

  1. The SPOKEN answer (what engine.ask emits, after NLI verification + abstention).
  2. The PROOF TREE (engine.prove): per citation -- the SHA-256 content_hash, the grounded
     snippet, and the NLI entailment label. Provenance-complete iff every citation carries a hash.
  3. The RAW RECORD (engine.get_raw(content_hash)): the exact stored bytes the answer rests on,
     fetched from the WORM substrate by hash -- proof the record is lossless, not a paraphrase.
  4. The INTEGRITY ROLLUP (engine.integrity_report): store-wide capture/provenance counters.

The doctrine in code: the agent either proves a claim against an immutable hash or abstains.
That is photographic memory in engineering terms -- the record is perfect; the voice is honest.

Real embeddings + real model calls only. No mocks. Requires DASHSCOPE_API_KEY with credit; if the
key is missing the demo fails loud (it never fabricates a proof).

Usage:  python scripts/photographic_demo.py ["your question"]
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

# Isolate the demo store BEFORE importing eidetic.
os.environ.setdefault("APP_ENV", "dev")
os.environ["DATA_DIR"] = os.environ.get("DEMO_DATA_DIR", "./data/photographic_demo")
os.environ.setdefault("VECTOR_BACKEND", "hnswlib")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eidetic.config import get_settings  # noqa: E402
from eidetic.engine import Engine  # noqa: E402
from eidetic.models import Scope  # noqa: E402

# A tiny, fully-known corpus so the proof is checkable by eye. Each line is one lossless record.
_MEMORIES = [
    "Maria adopted a rescue greyhound named Pixel on 2023-04-18.",
    "Maria moved from Lisbon to Berlin in September 2022 for a research job.",
    "Maria finished reading 'The Left Hand of Darkness' in early 2022.",
    "Maria runs every Sunday morning along the canal to de-stress.",
    "Maria's sister Ana lives in Porto and visits Berlin twice a year.",
]

_DEFAULT_QUESTION = "What is the name of Maria's dog and when did she get it?"


def _hr(label: str) -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_QUESTION

    if not get_settings().has_api_key:
        print("ERROR: DASHSCOPE_API_KEY is required (real embeddings + model calls, no mocks).",
              file=sys.stderr)
        return 2

    # Fresh isolated store so the demo is reproducible.
    data_dir = Path(os.environ["DATA_DIR"])
    if data_dir.exists():
        shutil.rmtree(data_dir)
    engine = Engine()
    scope = Scope(namespace="photographic-demo")

    _hr("1. CAPTURE  -- write lossless records (WORM substrate, SHA-256 addressed)")
    for i, text in enumerate(_MEMORIES):
        engine.ingest_text(text, source=f"demo-{i}", scope=scope, consolidate_now=False)
        print(f"  stored: {text}")
    engine.consolidate_pending(scope=scope, score_importance=False)

    _hr(f"2. RECALL   -- ask, verify, speak (or abstain)\n   Q: {question}")
    answer = engine.ask(question, scope=scope, verify=True)
    print(f"\n  SPOKEN ANSWER:\n    {textwrap.fill(answer.answer, 72, subsequent_indent='    ')}")
    print(f"\n  verified={answer.verified}  confidence={answer.confidence:.2f}  "
          f"note={answer.note or '(none)'}")

    _hr("3. PROVE    -- every claim points to an immutable hash (engine.prove)")
    proof = engine.prove(answer, with_paths=True)
    print(f"  claim: {proof.get('claim', '')}")
    print(f"  provenance_complete: {proof.get('provenance_complete')}")
    for j, ev in enumerate(proof.get("evidence", [])):
        ch = ev.get("content_hash", "")
        print(f"\n  [E{j}] content_hash = {ch}")
        print(f"       nli = {ev.get('nli_label')} (score {ev.get('nli_score')})  "
              f"grounded={ev.get('grounded')}")
        print(f"       snippet  = {ev.get('snippet', '')!r}")
        # 4. The exact stored bytes behind THIS citation -- the photograph itself.
        if ch:
            try:
                raw = engine.get_raw(ch).decode("utf-8", errors="replace")
                print(f"       get_raw  = {raw!r}")
            except Exception as e:        # a real lookup failure is informative, not swallowed
                print(f"       get_raw  = <unavailable: {e}>")

    _hr("4. INTEGRITY -- store-wide capture/provenance rollup (engine.integrity_report)")
    report = engine.integrity_report(scope)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\nDoctrine: prove against an immutable hash, or abstain. The record is perfect; "
          "the voice is honest.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
