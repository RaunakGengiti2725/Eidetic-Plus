#!/usr/bin/env bash
# Post-dev-gate sequence for the dominance program. Run each phase ONLY after the previous
# one passes; never tune on test-split failures (holdout discipline).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null || true; set +a
export DASHSCOPE_MAX_CONCURRENCY="${DASHSCOPE_MAX_CONCURRENCY:-2}"
export DASHSCOPE_RPM="${DASHSCOPE_RPM:-30}"
export DASHSCOPE_SLOT_TIMEOUT_SEC="${DASHSCOPE_SLOT_TIMEOUT_SEC:-240}"
BUNDLE="artifacts/holdout_dominance_20260701_codex"
PHASE="${1:?usage: dominance_next.sh <audit|lme_dev_health|slice_locomo|slice_lme|holdout_h2h>}"

case "$PHASE" in
  audit)
    .venv/bin/python -m bench.audit_no_holdout_leakage
    ;;
  lme_dev_health)
    # DEV-split LME health check before ANY test draw: verified-rate must be ~100% here first.
    .venv/bin/python -m bench.run --systems eidetic-full --dataset longmemeval \
      --subset 12 --sample-strategy stratified --split dev --holdout-profile dev --runs 1 \
      --out "$BUNDLE/lme_dev_health_wave_d" --overwrite
    ;;
  slice_locomo)
    # TEST split, 5 stratified draws x 20 LoCoMo questions; release-eligible only with
    # random seed mode and perfect verified_correct per draw.
    METABOLISM_MODE=1 AFFECT_SALIENCE=1 GIST_CHANNEL=1 CRYSTAL_SPAN_DEMOTION=1 \
    .venv/bin/python -m bench.run_slice_invariant_eval --dataset locomo --split test \
      --draws 5 --subset 20 --systems eidetic-full --system-under-test eidetic-plus-full \
      --out "$BUNDLE/slice_invariant_locomo"
    ;;
  slice_lme)
    METABOLISM_MODE=1 AFFECT_SALIENCE=1 GIST_CHANNEL=1 CRYSTAL_SPAN_DEMOTION=1 \
    .venv/bin/python -m bench.run_slice_invariant_eval --dataset longmemeval --split test \
      --draws 5 --subset 24 --systems eidetic-full --system-under-test eidetic-plus-full \
      --out "$BUNDLE/slice_invariant_lme"
    ;;
  holdout_h2h)
    # Frozen holdout head-to-head vs rag-vector + mem0. Raise the wall-clock deadline so the
    # Mem0 row is fair (its add() path needs more than 20s per call on long sessions).
    METABOLISM_MODE=1 AFFECT_SALIENCE=1 GIST_CHANNEL=1 CRYSTAL_SPAN_DEMOTION=1 \
    DASHSCOPE_REQUEST_TIMEOUT_SEC=120 \
    .venv/bin/python -m bench.run --systems eidetic-full,rag-vector,mem0 --dataset both \
      --subset 20 --sample-strategy stratified --split test --holdout-profile holdout --runs 1 \
      --out "$BUNDLE/holdout_h2h_wave_d" --overwrite
    ;;
esac
