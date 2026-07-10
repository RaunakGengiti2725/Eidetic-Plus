#!/usr/bin/env bash
# Detached quota-reset runner (2026-07-10): sleeps until ~23:25 PT, probes NLM quota, then
# executes the measurement queue in priority order. Written because the session cron is
# idle-gated and may be blocked; this runs regardless. Logs to artifacts/quota_queue_run.log.
# Reversible: kill $(cat /tmp/eidetic_quota_queue.pid)
cd /Users/raunakgengiti/Eidetic-Plus || exit 1
LOG=artifacts/quota_queue_run.log
echo "queue runner started $(date -u +%FT%TZ), sleeping until 23:25 PT" >> "$LOG"
# sleep until 23:25 local
target=$(date -j -f "%H:%M:%S" "23:25:00" +%s 2>/dev/null)
now=$(date +%s)
[ "$target" -le "$now" ] && target=$((target + 86400))
sleep $((target - now))
echo "waking $(date -u +%FT%TZ); probing quota" >> "$LOG"
for attempt in 1 2 3; do
  if .venv/bin/nlm notebook query 17c2096b-3107-46ac-b26f-2ffa5e9d3725 "probe" --json 2>&1 | grep -q '"answer"'; then
    echo "quota LIVE (attempt $attempt)" >> "$LOG"
    break
  fi
  echo "quota still walled (attempt $attempt); sleeping 30m" >> "$LOG"
  [ "$attempt" = 3 ] && { echo "giving up; task list carries the commands" >> "$LOG"; exit 1; }
  sleep 1800
done
{
  echo "=== runs 8-10 ==="
  bash bench/nbl_run_cycle.sh artifacts/holdout_rotation_r15_codex 8 0.575 "rag-vector (r15 fixed-reader)"
  bash bench/nbl_run_cycle.sh artifacts/holdout_rotation_r15_codex 9 0.575 "rag-vector (r15 fixed-reader)"
  bash bench/nbl_run_cycle.sh artifacts/holdout_rotation_r15_codex 10 0.575 "rag-vector (r15 fixed-reader)"
  echo "=== provenance revalidation ==="
  DATA_DIR=artifacts/lme_s_r1_codex/data .venv/bin/python bench/provenance_live_probe.py 26
  echo "=== LME-S lossless reproduce (fresh -v3 notebooks) ==="
  DATA_DIR=artifacts/lme_s_r1_codex/data NLM_NOTEBOOK_SUFFIX=-v3 .venv/bin/python -m bench.notebooklm_freetier_run artifacts/lme_s_r1_codex --out artifacts/lme_s_r1_codex/notebooklm_freetier_v2pack_run2.jsonl
  .venv/bin/python -m bench.notebooklm_freetier_report artifacts/lme_s_r1_codex/notebooklm_freetier_v2pack_run2.jsonl --judge --out artifacts/lme_s_r1_codex/notebooklm_freetier_v2pack_run2.report.md
  echo "=== queue complete $(date -u +%FT%TZ); results uncommitted -- next session commits with labels ==="
} >> "$LOG" 2>&1
