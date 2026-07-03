"""Rotation slice 1 (epoch 0, window 0, digest d96875..): release-grade held-out head-to-head.

eidetic-full vs mem0 on 40 NEVER-TOUCHED LoCoMo test-split questions drawn by
bench.rotating_holdout (committed ledger). Same inherited wave-F profile as every prior
head-to-head, but NO unpromoted flag overrides (ADAPTIVE_CONTEXT stays default-off) and
--holdout-profile holdout (forbids legacy rescue flags). Fresh data dir.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INHERITED = ROOT / "artifacts/wave_i_mixed24_full_codex/inherited_env.json"
OUT = ROOT / "artifacts/holdout_rotation_r1_codex"
SAMPLES = OUT / "holdout40.samples.json"

env = dict(os.environ)
env.update({k: v for k, v in json.loads(INHERITED.read_text()).items() if str(v).strip()})
env["DATA_DIR"] = str(OUT / "data")
env["DASHSCOPE_REQUEST_TIMEOUT_SEC"] = "120"   # mem0 fairness: its add() exceeds 20s
env.pop("ADAPTIVE_CONTEXT", None)              # unpromoted flags stay OFF on holdout runs
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
print("launching rotation-r1 h2h:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
