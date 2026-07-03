"""Five-role mixed-24 dev gate ablation on the wave-I build.

Same samples + profile as the wave-F ablation (common env inherited from the wave-F full-role
manifest; role deltas owned by bench.run_dev_ablation), fresh data dirs per role.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INHERITED = ROOT / "artifacts/wave_i_mixed24_full_codex/inherited_env.json"
SAMPLES = ROOT / ("artifacts/holdout_dominance_20260701_codex/"
                  "dev_ablation_mixed24_wave_c/dev_ablation.samples.json")
OUT = ROOT / "artifacts/wave_i_ablation_mixed24_codex"

ROLE_OWNED = {"METABOLISM_MODE", "AFFECT_SALIENCE", "GIST_CHANNEL", "CRYSTAL_SPAN_DEMOTION",
              "DATA_DIR"}
env_pairs = [f"{k}={v}" for k, v in json.loads(INHERITED.read_text()).items()
             if k not in ROLE_OWNED]

cmd = [sys.executable, "-m", "bench.run_dev_ablation",
       "--out-root", str(OUT),
       "--samples-file", str(SAMPLES),
       "--dataset", "both",
       "--variant", "longmemeval_s",
       "--runs", "1",
       "--overwrite"]
for pair in env_pairs:
    cmd += ["--common-env", pair]
print("launching five-role ablation:", OUT)
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
