#!/usr/bin/env bash
# One-line reproduce command for the neutral benchmark (a number that doesn't reproduce
# doesn't exist). Runs Eidetic-Plus, RAG baselines, Mem0, and Graphiti through ONE fixed judge.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Prereqs (fail loud if absent):
#   1. DASHSCOPE_API_KEY in .env (real model calls; no mocks).
#   2. Baselines installed:  .venv/bin/pip install -r requirements-bench.txt
#   3. Graphiti needs Neo4j:  export NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=...
#      (free Neo4j AuraDB works; no Docker required).
#   4. Optional GPT-4o judge:  export JUDGE_BASE_URL=https://api.openai.com/v1 \
#      JUDGE_API_KEY=sk-... JUDGE_MODEL=gpt-4o   (else qwen3-max is the fixed judge).
# Pin Qwen snapshots via the *_MODEL env vars in .env so aliases don't rotate mid-study.

source .venv/bin/activate 2>/dev/null || true
OUT="${OUT:-artifacts/bench}"
mkdir -p "$OUT"
export METABOLISM_MODE="${METABOLISM_MODE:-1}"
export DASHSCOPE_MAX_CONCURRENCY="${DASHSCOPE_MAX_CONCURRENCY:-2}"
export DASHSCOPE_RPM="${DASHSCOPE_RPM:-30}"

# Full study: both datasets, all four LoCoMo categories, >=10 runs for variance.
# --split test: reported numbers come ONLY from the held-out test split. The optimizers
# (bench.sweep, bench.calibrate) tune on the disjoint --split dev partition, so no reported
# number is ever fit to the items it scores. This is the integrity wall.
python -m bench.run \
  --systems eidetic,eidetic-full,eidetic-product,rag-full,rag-vector,mem0,graphiti \
  --dataset both \
  --subset 0 \
  --runs 10 \
  --split test \
  --out "$OUT"

echo "== Release audit bundle =="
# Populate data/bench/holdout first from a real private holdout samples file, for example:
#   python -m bench.build_holdout_registry --samples-file /secure/holdout.samples.json
# The audit intentionally fails closed when the registry is empty.
AUDIT_STATUS=0
python -m bench.audit_no_holdout_leakage --roots eidetic bench tests docs \
  > "$OUT/holdout_audit.json" || AUDIT_STATUS=$?
cat "$OUT/holdout_audit.json"
if [[ "$AUDIT_STATUS" -ne 0 ]]; then
  echo "holdout leakage audit failed with status $AUDIT_STATUS; release_gate will fail closed." >&2
fi
python -m bench.forensics --logs "$OUT" --out "$OUT/forensics.md"

SNAP_STATUS=0
python scripts/snap_back_audit.py --out "$OUT/snap_back_audit.json" || SNAP_STATUS=$?
if [[ "$SNAP_STATUS" -ne 0 ]]; then
  echo "snap_back_audit failed with status $SNAP_STATUS; release_gate will fail closed." >&2
fi

python -m bench.claim_scope --out "$OUT"

ABLATION_STATUS=0
if [[ -f "$OUT/ablation_report.json" ]]; then
  echo "ablation_report.json already present; release_gate will validate it."
elif [[ -n "${ABLATION_FULL_DIR:-}" && -n "${ABLATION_METABOLISM_OFF_DIR:-}" && -n "${ABLATION_REGIONS_OFF_DIR:-}" && -n "${ABLATION_FORGETTING_OFF_DIR:-}" && -n "${ABLATION_AFFECT_OFF_DIR:-}" ]]; then
  python -m bench.build_ablation_report \
    --full "$ABLATION_FULL_DIR" \
    --metabolism-off "$ABLATION_METABOLISM_OFF_DIR" \
    --regions-off "$ABLATION_REGIONS_OFF_DIR" \
    --forgetting-off "$ABLATION_FORGETTING_OFF_DIR" \
    --affect-off "$ABLATION_AFFECT_OFF_DIR" \
    --system "${ABLATION_SYSTEM:-eidetic-plus-full}" \
    --out "$OUT/ablation_report.json" || ABLATION_STATUS=$?
else
  echo "ablation_report.json missing. Run python -m bench.run_dev_ablation --report-out \"$OUT/ablation_report.json\" or set ABLATION_FULL_DIR, ABLATION_METABOLISM_OFF_DIR, ABLATION_REGIONS_OFF_DIR, ABLATION_FORGETTING_OFF_DIR, and ABLATION_AFFECT_OFF_DIR; release_gate will fail closed." >&2
  ABLATION_STATUS=2
fi
if [[ "$ABLATION_STATUS" -ne 0 ]]; then
  echo "ablation evidence unavailable or failed with status $ABLATION_STATUS; release_gate will fail closed." >&2
fi

SIDECAR_STATUS=0
run_sidecar() {
  local label="$1"
  shift
  if "$@"; then
    return 0
  else
    local status=$?
    if [[ "$SIDECAR_STATUS" -eq 0 ]]; then
      SIDECAR_STATUS="$status"
    fi
    echo "$label failed with status $status; release_gate will fail closed." >&2
    return 0
  fi
}

run_sidecar affect_salience_invariant python -m bench.affect_salience_invariant \
  --cases "${AFFECT_SALIENCE_CASES:-24}" \
  --out "$OUT/affect_salience_invariant.json"
run_sidecar scratchpad_invariant python -m bench.scratchpad_invariant \
  --cases "${SCRATCHPAD_CASES:-24}" \
  --out "$OUT/scratchpad_invariant.json"
run_sidecar region_routing_invariant python -m bench.region_routing_invariant \
  --cases "${REGION_ROUTING_CASES:-24}" \
  --out "$OUT/region_routing_invariant.json"
run_sidecar reflex_recall_invariant python -m bench.reflex_recall_invariant \
  --cases "${REFLEX_RECALL_CASES:-24}" \
  --out "$OUT/reflex_recall_invariant.json"
run_sidecar smqe_synthetic_invariant python -m bench.smqe_synthetic_invariant \
  --cases "${SMQE_SYNTHETIC_CASES:-24}" \
  --out "$OUT/smqe_synthetic_invariant.json"
run_sidecar smqe_planner_invariant python -m bench.smqe_planner_invariant \
  --cases "${SMQE_PLANNER_CASES:-24}" \
  --out "$OUT/smqe_planner_invariant.json"
run_sidecar smqe_claim_coverage python -m bench.smqe_claim_coverage \
  --cases "${SMQE_CLAIM_COVERAGE_CASES:-24}" \
  --out "$OUT/smqe_claim_coverage.json"
run_sidecar smqe_fullpath_invariant python -m bench.smqe_fullpath_invariant \
  --cases "${SMQE_FULLPATH_CASES:-24}" \
  --out "$OUT/smqe_fullpath_invariant.json"
run_sidecar smqe_paraphrase_invariant python -m bench.smqe_paraphrase_invariant \
  --cases "${SMQE_PARAPHRASE_CASES:-24}" \
  --out "$OUT/smqe_paraphrase_invariant.json"
run_sidecar smqe_conflict_invariant python -m bench.smqe_conflict_invariant \
  --cases "${SMQE_CONFLICT_CASES:-24}" \
  --out "$OUT/smqe_conflict_invariant.json"
run_sidecar smqe_composition_invariant python -m bench.smqe_composition_invariant \
  --cases "${SMQE_COMPOSITION_CASES:-24}" \
  --out "$OUT/smqe_composition_invariant.json"
run_sidecar smqe_relative_phrase_invariant python -m bench.smqe_relative_phrase_invariant \
  --cases "${SMQE_RELATIVE_PHRASE_CASES:-24}" \
  --out "$OUT/smqe_relative_phrase_invariant.json"
run_sidecar smqe_temporal_window_invariant python -m bench.smqe_temporal_window_invariant \
  --cases "${SMQE_TEMPORAL_WINDOW_CASES:-24}" \
  --out "$OUT/smqe_temporal_window_invariant.json"
run_sidecar smqe_attribution_invariant python -m bench.smqe_attribution_invariant \
  --cases "${SMQE_ATTRIBUTION_CASES:-24}" \
  --out "$OUT/smqe_attribution_invariant.json"
run_sidecar smqe_abstention_invariant python -m bench.smqe_abstention_invariant \
  --cases "${SMQE_ABSTENTION_CASES:-24}" \
  --out "$OUT/smqe_abstention_invariant.json"
run_sidecar smqe_scope_invariant python -m bench.smqe_scope_invariant \
  --cases "${SMQE_SCOPE_CASES:-24}" \
  --out "$OUT/smqe_scope_invariant.json"
run_sidecar smqe_subscope_invariant python -m bench.smqe_subscope_invariant \
  --cases "${SMQE_SUBSCOPE_CASES:-24}" \
  --out "$OUT/smqe_subscope_invariant.json"
run_sidecar smqe_time_invariant python -m bench.smqe_time_invariant \
  --cases "${SMQE_TIME_CASES:-24}" \
  --out "$OUT/smqe_time_invariant.json"
run_sidecar smqe_invalidation_invariant python -m bench.smqe_invalidation_invariant \
  --cases "${SMQE_INVALIDATION_CASES:-24}" \
  --out "$OUT/smqe_invalidation_invariant.json"
run_sidecar smqe_dialogue_invariant python -m bench.smqe_dialogue_invariant \
  --cases "${SMQE_DIALOGUE_CASES:-24}" \
  --out "$OUT/smqe_dialogue_invariant.json"
run_sidecar smqe_lacuna_invariant python -m bench.smqe_lacuna_invariant \
  --cases "${SMQE_LACUNA_CASES:-24}" \
  --out "$OUT/smqe_lacuna_invariant.json"
run_sidecar crystal_demotion_invariant python -m bench.crystal_demotion_invariant \
  --cases "${CRYSTAL_DEMOTION_CASES:-20}" \
  --out "$OUT/crystal_demotion_invariant.json"
if [[ "$SIDECAR_STATUS" -ne 0 ]]; then
  echo "one or more invariant sidecars failed; release_gate will fail closed." >&2
fi

SLICE_STATUS=0
SLICE_ROOT="$OUT/slice_invariant"
mkdir -p "$SLICE_ROOT"
SLICE_SEED_ARGS=()
if [[ -n "${SLICE_INVARIANT_SEED:-}" ]]; then
  SLICE_SEED_ARGS=(--seed "$SLICE_INVARIANT_SEED")
fi
python -m bench.run_slice_invariant_eval \
  --dataset longmemeval \
  --variant longmemeval_s \
  --split test \
  --draws "${SLICE_INVARIANT_DRAWS:-5}" \
  --subset "${LME_SLICE_INVARIANT_SUBSET:-24}" \
  "${SLICE_SEED_ARGS[@]}" \
  --systems "${SLICE_INVARIANT_SYSTEMS:-eidetic-full}" \
  --system-under-test eidetic-plus-full \
  --out "$SLICE_ROOT/longmemeval" || SLICE_STATUS=$?
python -m bench.run_slice_invariant_eval \
  --dataset locomo \
  --variant longmemeval_s \
  --split test \
  --draws "${SLICE_INVARIANT_DRAWS:-5}" \
  --subset "${LOCOMO_SLICE_INVARIANT_SUBSET:-20}" \
  "${SLICE_SEED_ARGS[@]}" \
  --systems "${SLICE_INVARIANT_SYSTEMS:-eidetic-full}" \
  --system-under-test eidetic-plus-full \
  --out "$SLICE_ROOT/locomo" || SLICE_STATUS=$?
python - "$SLICE_ROOT" "$OUT/slice_invariant.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
reports = []
for name in ("longmemeval", "locomo"):
    path = root / name / "slice_invariant.json"
    if path.exists():
        reports.append(json.loads(path.read_text()))
    else:
        reports.append({"dataset": name, "pass": False, "runs": [], "missing": str(path)})
sidecar = {"pass": all(r.get("pass") for r in reports), "reports": reports}
out.write_text(json.dumps(sidecar, indent=2) + "\n")
print(json.dumps(sidecar, indent=2))
PY
if [[ "$SLICE_STATUS" -ne 0 ]]; then
  echo "slice-invariant run failed with status $SLICE_STATUS; release_gate will fail closed." >&2
fi

MEM0_GATE_EXPECTED="${MEM0_GATE_EXPECTED:-data/bench/mem0_locomo_published.json}"
if [[ -f "$MEM0_GATE_EXPECTED" ]]; then
  MEM0_STATUS=0
  python -m bench.gate \
    --out "$OUT" \
    --expected "$MEM0_GATE_EXPECTED" \
    --report-out "$OUT/mem0_gate.md" || MEM0_STATUS=$?
  if [[ "$MEM0_STATUS" -ne 0 ]]; then
    echo "mem0 reproduction gate failed with status $MEM0_STATUS; release_gate will fail closed." >&2
  fi
else
  echo "Mem0 reproduction reference not found at $MEM0_GATE_EXPECTED; release_gate will fail closed." >&2
fi

set +e
python -m bench.release_gate --out "$OUT" --report-out "$OUT/release_gate.md"
RELEASE_STATUS=$?
set -e

echo "Done. See $OUT/scoreboard.md, $OUT/forensics.md, $OUT/claim_scope.json, and $OUT/release_gate.md."
exit "$RELEASE_STATUS"
