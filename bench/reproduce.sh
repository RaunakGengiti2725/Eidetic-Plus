#!/usr/bin/env bash
# One-line reproduce command for the neutral benchmark (a number that doesn't reproduce
# doesn't exist). Runs Eidetic-Plus + Mem0 + Graphiti through ONE fixed judge.
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

# Full study: both datasets, all four LoCoMo categories, >=10 runs for variance.
# --split test: reported numbers come ONLY from the held-out test split. The optimizers
# (bench.sweep, bench.calibrate) tune on the disjoint --split dev partition, so no reported
# number is ever fit to the items it scores. This is the integrity wall.
python -m bench.run \
  --systems eidetic,mem0,graphiti \
  --dataset both \
  --subset 0 \
  --runs 10 \
  --split test \
  --out artifacts/bench

echo "Done. See artifacts/bench/scoreboard.md and the recall_vs_age / latency_vs_age PNGs."
