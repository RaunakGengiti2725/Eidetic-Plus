"""Rotation slice 2 (epoch 0, window 1, digest 20053b..): current build, never-touched window.

Measures whether today's nine verified-wrong kills GENERALIZE (slice 1 ran pre-fix code).
Same protocol as r1: eidetic-full vs mem0, holdout profile, unpromoted flags OFF, fresh
data dir, disjoint-by-construction sample window.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INHERITED = ROOT / "artifacts/wave_i_mixed24_full_codex/inherited_env.json"
OUT = ROOT / "artifacts/holdout_rotation_r2_codex"
SAMPLES = OUT / "holdout40.samples.json"

env = dict(os.environ)
env.update({k: v for k, v in json.loads(INHERITED.read_text()).items() if str(v).strip()})
env["DATA_DIR"] = str(OUT / "data")
env["DASHSCOPE_REQUEST_TIMEOUT_SEC"] = "120"
env.pop("ADAPTIVE_CONTEXT", None)
OUT.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "bench.run",
    "--systems", "eidetic-full,mem0",
    "--dataset", "locomo",
    "--samples-file", str(SAMPLES),
    "--split", "test",
    "--holdout-profile", "holdout",
    "--runs", "1",
    "--out", str(OUT),
    "--overwrite",
]
print("launching rotation-r2 h2h:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
