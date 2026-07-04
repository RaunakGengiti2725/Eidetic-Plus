#!/usr/bin/env bash
# Replays the reader form floor over every rotation window's eidetic jsonl.
# Contract: kills may rise, flips must stay EMPTY on every window. Run before
# committing ANY change to eidetic/smqe/verify.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/.venv/bin/python" - <<'PYEOF'
import glob, json, sys
sys.path.insert(0, ".")
from eidetic.smqe.verify import reader_answer_form_credible as f
bad = False
for path in sorted(glob.glob("artifacts/holdout_rotation_r*_codex/eidetic-plus-full__run0.jsonl")):
    rows = [json.loads(l) for l in open(path)]
    kills = [r["sample_id"] for r in rows
             if not r["correct"] and r.get("extra", {}).get("verified")
             and not r.get("abstained") and not f(r["question"], r["predicted"])]
    flips = [r["sample_id"] for r in rows
             if r["correct"] and r.get("extra", {}).get("verified")
             and not f(r["question"], r["predicted"])]
    print(f"{path.split('/')[1]}: kills={len(kills)} flips={flips}")
    bad = bad or bool(flips)
sys.exit(1 if bad else 0)
PYEOF
