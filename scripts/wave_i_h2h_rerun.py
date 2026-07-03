"""Head-to-head on LoCoMo dev-20: eidetic-full vs rag-full vs rag-vector vs mem0.

Same profile env as the wave-I runs (inherited from the wave-F manifest); fresh data dir.
Mem0 fairness: DASHSCOPE_REQUEST_TIMEOUT_SEC=120 (its add() calls exceeded 20s in past runs).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INHERITED = ROOT / "artifacts/wave_i_mixed24_full_codex/inherited_env.json"
OUT = ROOT / "artifacts/wave_i_h2h_rerun_codex"
DATA = OUT / "data"

env = dict(os.environ)
env.update({k: v for k, v in json.loads(INHERITED.read_text()).items() if str(v).strip()})
env["DATA_DIR"] = str(DATA)
env["DASHSCOPE_REQUEST_TIMEOUT_SEC"] = "120"
OUT.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "bench.run",
    "--systems", "eidetic-full",
    "--dataset", "locomo",
    "--subset", "20",
    "--split", "dev",
    "--holdout-profile", "dev",
    "--sample-strategy", "stratified",
    "--runs", "1",
    "--out", str(OUT),
    "--overwrite",
]
print("launching h2h:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
