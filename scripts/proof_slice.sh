#!/usr/bin/env bash
# Forgetting-machine proof slice: head-to-head + attribution ablation on ONE hard LongMemEval-S
# category, all through the ONE shared reader/judge. Directional (n small, 1 run) -- the full
# multi-run gate is bench/reproduce.sh. Every number lands with a manifest of every flag.
#
# Usage: scripts/proof_slice.sh <category_offset> <n> <tag>
#   e.g. scripts/proof_slice.sh 70 6 multisession   (multi-session block starts at offset 70)
# No `set -e`: one sub-run failing (e.g. a baseline lib hiccup) must not abort the whole batch;
# bench.run already isolates per-system failures and still renders the scoreboard.
set -uo pipefail
cd "$(dirname "$0")/.."
# eidetic/config.py load_dotenv() reads .env for DASHSCOPE_API_KEY directly, so no shell sourcing
# is needed (and sourcing under set -e was aborting the batch on quirky .env lines).

OFFSET="${1:?offset}"; N="${2:?n}"; TAG="${3:?tag}"
PY=.venv/bin/python
export DASHSCOPE_MAX_CONCURRENCY="${DASHSCOPE_MAX_CONCURRENCY:-4}"
OUT="artifacts/proof/${TAG}"
H2H="${OUT}/headtohead"
ABL="${OUT}/ablation_memory_off"

echo "=================================================================="
echo " PROOF SLICE  tag=${TAG}  offset=${OFFSET}  n=${N}  $(date +%T)"
echo "=================================================================="

# --- 1) Head-to-head: eidetic-metabolism vs rag-full vs rag-vector vs mem0 (shared reader on for all)
echo ">>> [1/3] head-to-head (METABOLISM_MODE=1 for all; shared Tier-A reader)"
METABOLISM_MODE=1 DATA_DIR="data/proof_${TAG}_h2h" \
  $PY -m bench.run --dataset longmemeval --sample-offset "$OFFSET" --subset "$N" \
  --systems eidetic-full,rag-full,rag-vector,mem0 --out "$H2H" --split all

# --- 2) Attribution ablation: SAME eidetic, memory/metabolism components OFF, reader+proof HELD ON.
#        This isolates the MEMORY layer (consolidation/dreaming/channels/capture/graph-temporal),
#        not the reader -- the reader scaffolds + proof gate stay identical to the head-to-head row.
echo ">>> [2/3] ablation: metabolism memory OFF, reader+proof fixed"
METABOLISM_MODE=1 DATA_DIR="data/proof_${TAG}_abl" \
  FULL_SLEEP=0 GIST_CHANNEL=0 COACTIVATION_CHANNEL=0 STRUCT_CHANNEL=0 EVENT_RANKING=0 \
  GRAPH_VOCAB_SEEDING=0 EXTRACT_CHUNKING=0 MEMORY_TYPING=0 PREF_SENTENCE_SCAN=0 \
  TEMPORAL_RERANK=0 CONFLICT_RESOLVER=0 \
  $PY -m bench.run --dataset longmemeval --sample-offset "$OFFSET" --subset "$N" \
  --systems eidetic-full --out "$ABL" --split all

# --- 3) Flip table: metabolism ON (head-to-head) vs OFF (ablation), per-question attribution.
echo ">>> [3/3] attribution flip table (control = memory OFF, experiment = memory ON)"
$PY -m bench.compare --control "$ABL" --experiment "$H2H" \
  --system eidetic-plus-full --out "${OUT}/attribution_compare.md"

echo "=================================================================="
echo " DONE  $(date +%T)   scoreboard: ${H2H}/scoreboard.md"
echo "                     attribution: ${OUT}/attribution_compare.md"
echo "=================================================================="
