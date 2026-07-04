#!/usr/bin/env bash
# Two-phase holdout runner (P3): Phase A eidetic-only -> forensics at +40 min;
# Phase B mem0 overnight into the SAME out dir; scoreboard merges via --render-only.
#
# Usage:
#   bench/phase_holdout.sh A <samples-file> <out-dir> [profile.json]  # eidetic-full now
#   bench/phase_holdout.sh B <samples-file> <out-dir> [profile.json]  # mem0, overnight
#   bench/phase_holdout.sh render <samples-file> <out-dir>            # rebuild scoreboard
#
# Both phases pin the code SHA into <out-dir>/launch_<phase>.log. The inherited wave-F
# profile env is replicated the same way the rotation launchers do it. Without a profile
# arg, unpromoted flags stay OFF (ADAPTIVE_CONTEXT explicitly unset). With a profile arg
# (promotion measurement, e.g. bench/profiles/product_cost.json), the profile's flags
# overlay the inherited env and the profile path + resolved flags land in the launch log.
# DASHSCOPE_REQUEST_TIMEOUT_SEC=120 for mem0 fairness.
set -euo pipefail

PHASE="${1:?phase A|B|render}"
SAMPLES="${2:?samples-file}"
OUT="${3:?out-dir}"
PROFILE="${4:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ -n "$PROFILE" ] && [ ! -f "$PROFILE" ]; then
  echo "profile not found: $PROFILE" >&2; exit 2
fi
mkdir -p "$OUT"

run_phase() {
  local systems="$1" phase="$2"
  local log="$OUT/launch_${phase}.log"
  {
    echo "phase=$phase systems=$systems"
    echo "sha=$(git -C "$ROOT" rev-parse HEAD)"
    echo "samples=$SAMPLES"
    echo "profile=${PROFILE:-none}"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | tee "$log"
  if [ -n "$PROFILE" ]; then cp "$PROFILE" "$OUT/profile_${phase}.json"; fi
  "$PY" - "$ROOT" "$systems" "$SAMPLES" "$OUT" "$PROFILE" <<'PYEOF' 2>&1 | tee -a "$log"
import json, os, subprocess, sys
root, systems, samples, out, profile = sys.argv[1:6]
env = dict(os.environ)
inherited = os.path.join(root, "artifacts/wave_i_mixed24_full_codex/inherited_env.json")
env.update({k: v for k, v in json.load(open(inherited)).items() if str(v).strip()})
env["DATA_DIR"] = os.path.join(out, "data")
env["DASHSCOPE_REQUEST_TIMEOUT_SEC"] = "120"
env.pop("ADAPTIVE_CONTEXT", None)
if profile:
    overlay = {k: str(v) for k, v in json.load(open(profile)).items()
               if not k.startswith("_") and str(v).strip()}
    env.update(overlay)
    print("profile overlay:", json.dumps(overlay, sort_keys=True), flush=True)
cmd = [sys.executable, "-m", "bench.run", "--systems", systems, "--dataset", "locomo",
       "--samples-file", samples, "--split", "test", "--holdout-profile", "holdout",
       "--runs", "1", "--out", out]
print("exec:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, env=env, cwd=root))
PYEOF
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
}

case "$PHASE" in
  A) run_phase "eidetic-full" A ;;
  B) run_phase "mem0" B ;;
  render)
    "$PY" -m bench.run --systems eidetic-full,mem0 --dataset locomo \
      --samples-file "$SAMPLES" --split test --holdout-profile holdout \
      --runs 1 --out "$OUT" --render-only ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
