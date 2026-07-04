#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bench/run_cost_profile.sh <profile.json> <arm-name> [subset=20]" >&2
  echo "example: bench/run_cost_profile.sh bench/profiles/product_cost.json product_on 40" >&2
  exit 1
}

PROFILE="${1:-}"; ARM="${2:-}"; SUBSET="${3:-20}"
[ -n "$PROFILE" ] && [ -n "$ARM" ] || usage
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
INHERITED="$ROOT/artifacts/wave_i_mixed24_full_codex/inherited_env.json"
[ -f "$PROFILE" ] || { echo "profile not found: $PROFILE" >&2; exit 1; }
[ -f "$INHERITED" ] || { echo "inherited env not found: $INHERITED" >&2; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/artifacts/cost_${ARM}_${TS}"
mkdir -p "$OUT/data"

while IFS= read -r kv; do export "${kv?}"; done < <("$PY" - "$INHERITED" "$PROFILE" <<'EOF'
import json, sys
env = {}
for path in sys.argv[1:3]:
    for k, v in json.load(open(path)).items():
        if not k.startswith("_") and str(v).strip():
            env[k] = str(v)
for k, v in sorted(env.items()):
    print(f"{k}={v}")
EOF
)
export DATA_DIR="$OUT/data"

cp "$PROFILE" "$OUT/profile.json"
{
  echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
  echo "profile=$PROFILE"
  echo "arm=$ARM subset=$SUBSET split=dev ts=$TS"
} > "$OUT/launch.log"

CMD=("$PY" -m bench.run
  --systems eidetic-full --dataset locomo
  --subset "$SUBSET" --split dev --holdout-profile dev
  --sample-strategy stratified --runs 1
  --out "$OUT" --overwrite)

if [ "${COST_PROFILE_DRY:-0}" = "1" ]; then
  echo "DRY RUN -- resolved cost flags:"
  env | grep -E "^(ADAPTIVE_CONTEXT|EXTRACT_COMBINED|EXTRACT_RESULT_CACHE|DATA_DIR)=" | sort
  echo "cmd: ${CMD[*]}"
  rm -rf "$OUT"
  exit 0
fi

echo "launching cost arm '$ARM' -> $OUT"
exec "${CMD[@]}"
