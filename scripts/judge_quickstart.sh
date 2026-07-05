#!/usr/bin/env bash
# Judge quickstart: verify the public claims from a fresh clone in ~5 minutes.
# Zero-API steps first (work with no key); the live demo needs DASHSCOPE_API_KEY in .env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

if [ ! -x .venv/bin/python ]; then
  step "Creating venv + installing (first run only)"
  python3 -m venv .venv
  .venv/bin/pip install -q -e ".[dev]"
fi
PY=.venv/bin/python

step "1/5 Holdout leakage audit (no benchmark IDs in source -- fails closed)"
$PY -m bench.audit_no_holdout_leakage | $PY -c "import json,sys; r=json.load(sys.stdin); print('PASS' if r['pass'] else 'FAIL', '-', r['needles_checked'], 'needles over', ', '.join(r['scan_roots'])); sys.exit(0 if r['pass'] else 1)"

step "2/5 War-room demo, fully offline (fake embeddings, zero API calls)"
bash bench/demo_war_room.sh

step "3/5 Rolling never-touched holdout table (recomputed from raw per-row logs)"
DIRS=$(ls -d artifacts/holdout_rotation_r*_codex 2>/dev/null | sort -V)
if [ -n "$DIRS" ]; then
  # shellcheck disable=SC2086
  $PY -m bench.rolling_holdout_table $DIRS
else
  echo "no holdout rotation artifacts found"
fi

step "4/5 Snap-back fidelity (immutable substrate: sha256(raw) == content hash)"
if [ -f artifacts/public_ship/snap_back_audit.json ]; then
  $PY -c "import json; r=json.load(open('artifacts/public_ship/snap_back_audit.json')); print(r['status'], '-', r['lossless_byte_identical'], '/', r['records_with_raw_blob'], 'records byte-identical (', r['rate_pct'], '%)')"
else
  echo "artifacts/public_ship/snap_back_audit.json not found -- run scripts/snap_back_audit.py over a corpus"
fi

step "5/5 Where the numbers live (every public number ties to a run_manifest.json)"
cat <<'EOF'
  docs/PUBLIC_CLAIMS.md                        -- the claims, each with its artifact path
  docs/claims.md                               -- what we refuse to claim, and why
  artifacts/holdout_rotation_r8_codex/         -- slice 8 (freshest): scoreboard.{md,json},
                                                  run_manifest.json, launch_A.log (pinned SHA +
                                                  product_cost profile), raw jsonl logs
  bench/COST_AB.md                             -- dev-split cost A/B ledger (median qtok 83 arm)
  python -m bench.release_gate --out <dir>     -- the fail-closed public-claim gate

Live demo (needs DASHSCOPE_API_KEY in .env): see docs/HACKATHON.md
EOF
