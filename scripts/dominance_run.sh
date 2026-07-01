#!/usr/bin/env bash
# Master benchmark orchestrator for the dominance plan.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate 2>/dev/null || true

MODE="smoke"
OUT="artifacts/bench"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline) MODE="offline"; shift ;;
    --full) MODE="full"; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT" artifacts/forensics artifacts/guard
export DATA_DIR="${DATA_DIR:-data/bench}"

echo "== Offline verification =="
pytest tests/test_metabolism_mode.py tests/test_optimization.py tests/test_abstention.py \
  tests/test_temporal_indexing.py tests/test_bench_plumbing.py tests/test_bench_harness.py \
  tests/test_bench_eidetic_full.py tests/test_scope_isolation.py tests/test_capture_fidelity.py \
  tests/test_conflict_resolver.py tests/test_forensics.py \
  tests/test_bench_report.py tests/test_release_gate.py tests/test_snap_back_fidelity.py \
  tests/test_guard.py tests/test_calibrate_abstention.py tests/test_calibration_handoff.py -q
python -m bench.calibration_handoff --help >/dev/null
python -m bench.sweep --dry-run --subset 50 >/dev/null
python -m bench.calibrate --help >/dev/null
python -m bench.forensics --logs "$OUT" --out artifacts/forensics/latest.md
python -m bench.guard --help >/dev/null
python -m bench.claim_scope --help >/dev/null
python -m bench.release_gate --help >/dev/null

if [[ "$MODE" == "offline" ]]; then
  echo "Offline dominance plumbing is ready. Use --full when funded API credits are available."
  exit 0
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" && ! -f .env ]]; then
  echo "DASHSCOPE_API_KEY missing and .env not found; stopping before live model calls." >&2
  exit 2
fi

export METABOLISM_MODE="${METABOLISM_MODE:-1}"
export DASHSCOPE_MAX_CONCURRENCY="${DASHSCOPE_MAX_CONCURRENCY:-2}"
export DASHSCOPE_RPM="${DASHSCOPE_RPM:-30}"

echo "== Smoke benchmark =="
python -m bench.run \
  --systems eidetic,rag-full \
  --dataset locomo \
  --subset 5 \
  --split dev \
  --out artifacts/smoke \
  --overwrite
python -m bench.forensics --logs artifacts/smoke --out artifacts/forensics/smoke.md

if [[ "$MODE" != "full" ]]; then
  echo "Smoke complete. Use --full for dev calibration, guard, and reproduce gate."
  exit 0
fi

echo "== Dev calibration corpus =="
python -m bench.run \
  --systems eidetic,eidetic-full,rag-full,rag-vector,mem0 \
  --dataset longmemeval \
  --subset 100 \
  --split dev \
  --out artifacts/cal_dev \
  --overwrite
python -m bench.calibrate --logs artifacts/cal_dev --system eidetic-plus-full --split dev --method all
python -m bench.forensics --logs artifacts/cal_dev --out artifacts/forensics/cal_dev.md
CAL_TAU_JSON="${CAL_TAU_JSON:-artifacts/cal_dev/abstention_v2_tau.json}"
CAL_ENV="${CAL_ENV:-artifacts/guard/abstention.env}"
if [[ -f "$CAL_TAU_JSON" ]]; then
  python -m bench.calibration_handoff \
    --calibration "$CAL_TAU_JSON" \
    --out "$OUT" \
    --env-out "$CAL_ENV"
  set -a
  source "$CAL_ENV"
  set +a
else
  echo "ABSTENTION_V2 calibration report missing at $CAL_TAU_JSON; release gate will fail closed." >&2
fi

echo "== Sweep dry run and guard promotion hook =="
python -m bench.sweep --dry-run --subset 50
if [[ -d artifacts/champion_dev && -d artifacts/sweep/best_trial && -f artifacts/sweep/best_config.json ]]; then
  python -m bench.guard \
    --champion artifacts/champion_dev \
    --challenger artifacts/sweep/best_trial \
    --best-config artifacts/sweep/best_config.json \
    --champion-out artifacts/guard/champion.env || true
fi

echo "== Public reproduce gate =="
OUT="$OUT" bash bench/reproduce.sh
python -m bench.forensics --logs "$OUT" --out artifacts/forensics/final.md
MEM0_GATE_EXPECTED="${MEM0_GATE_EXPECTED:-data/bench/mem0_locomo_published.json}"
if [[ -f "$MEM0_GATE_EXPECTED" ]]; then
  python -m bench.gate \
    --out "$OUT" \
    --expected "$MEM0_GATE_EXPECTED" \
    --report-out "$OUT/mem0_gate.md"
else
  echo "Mem0 reproduction reference missing at $MEM0_GATE_EXPECTED; release gate will fail closed." >&2
fi

echo "== Snap-back audit =="
python scripts/snap_back_audit.py --out "$OUT/snap_back_audit.json"
python -m bench.claim_scope --out "$OUT"
python -m bench.release_gate --out "$OUT" --report-out artifacts/guard/release_gate.md

echo "Dominance run complete. Inspect $OUT/scoreboard.md, artifacts/forensics/final.md, and artifacts/guard/release_gate.md."
