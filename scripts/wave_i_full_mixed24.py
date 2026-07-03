"""Launch the wave-I mixed-24 FULL-profile dev measurement.

Replays the exact wave-F full-role environment (read from its run manifest, non-empty keys
only) over the same 24 dev samples with a FRESH data dir, so the result isolates the wave
G/H/I code changes. One system, one run - not the five-role ablation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / ("artifacts/holdout_dominance_20260701_codex/"
                   "dev_ablation_mixed24_wave_f/full/run_manifest.json")
SAMPLES = ROOT / ("artifacts/holdout_dominance_20260701_codex/"
                  "dev_ablation_mixed24_wave_c/dev_ablation.samples.json")
OUT = ROOT / "artifacts/wave_i_mixed24_full_codex"
DATA = OUT / "data_full"

manifest = json.loads(MANIFEST.read_text())
env = dict(os.environ)
inherited = {k: v for k, v in manifest["env"].items() if str(v).strip() != ""}
env.update(inherited)
env["DATA_DIR"] = str(DATA)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "inherited_env.json").write_text(json.dumps(inherited, indent=1) + "\n")

cmd = [
    sys.executable, "-m", "bench.run",
    "--systems", "eidetic-full",
    "--dataset", "both",
    "--split", "dev",
    "--samples-file", str(SAMPLES),
    "--holdout-profile", "dev",
    "--runs", "1",
    "--variant", "longmemeval_s",
    "--out", str(OUT),
    "--overwrite",
]
print("launching:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
