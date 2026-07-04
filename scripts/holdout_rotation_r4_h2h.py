"""Rotation slice 4 (epoch 0, window 3, digest dd0dc0..): wave-Q forensics build.

Measures on top of wave-N/O: type agreement both paths, k=1 ordinal anchors, next-X
future polarity, prefix-bridged term matching, synonym families, wrong-instance kills. Same protocol: eidetic-full vs mem0, holdout
profile, unpromoted flags OFF, fresh data dir. Capture protocol note: captions are now
included for EVERY system through the shared loader (slices 1-2 were caption-less).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INHERITED = ROOT / "artifacts/wave_i_mixed24_full_codex/inherited_env.json"
OUT = ROOT / "artifacts/holdout_rotation_r4_codex"
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
print("launching rotation-r4 h2h:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
