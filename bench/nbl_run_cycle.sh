#!/usr/bin/env bash
# One NBL reproduce-run cycle for the >=10-run gate: collect run N on a window, judge it
# with the pinned judge, re-aggregate the gate verdict. Exists because the judge tool's
# --out is the MARKDOWN path while the judged sidecar auto-writes <stem>.judged.json --
# pointing --out at the sidecar clobbers it (happened on r15 run1/run2; re-judged).
#
# Usage: bench/nbl_run_cycle.sh <window_dir> <run_n> [comparator_acc] [comparator_name]
# Example: bench/nbl_run_cycle.sh artifacts/holdout_rotation_r15_codex 3 0.575 "rag-vector (r15 fixed-reader)"
set -euo pipefail

WINDOW="${1:?window dir}"
N="${2:?run number}"
COMP_ACC="${3:-0.575}"
COMP_NAME="${4:-rag-vector (r15 fixed-reader)}"
PY=".venv/bin/python"
STEM="$WINDOW/notebooklm_freetier_run$N"

# 1. collect (resumes: retries errored rows if the jsonl already exists)
DATA_DIR="$WINDOW/data" "$PY" -m bench.notebooklm_freetier_run "$WINDOW" \
  --skip-export --out "$STEM.jsonl" 2>&1 | tee -a "$WINDOW/run${N}_collect.log" | tail -3

# 2. judge -- NEVER point --out at the .judged.json sidecar
"$PY" -m bench.notebooklm_freetier_report "$STEM.jsonl" --judge \
  --out "$STEM.report.md" | tail -2

# 3. gate over every judged run present for this window
"$PY" -m bench.notebooklm_gate "$WINDOW"/notebooklm_freetier*.judged.json \
  --comparator-acc "$COMP_ACC" --comparator-name "$COMP_NAME" \
  --out "$WINDOW/notebooklm_gate.json" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
runs = d.get('per_run') or d.get('runs') or []
for r in runs:
    print(f\"{r['file']:44} {r['correct']}/{r['answered']} acc={r['accuracy']:.3f} \"
          + ('counts' if r.get('counts_toward_gate') else 'EXCLUDED'))
print(f\"mean={d.get('mean_accuracy')} stdev={d.get('stdev')} ci95={d.get('ci95_mean')}\")
print(f\"VERDICT: {d.get('verdict')} -- {d.get('verdict_reason')}\")
"
