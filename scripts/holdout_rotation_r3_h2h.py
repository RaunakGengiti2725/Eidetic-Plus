"""Rotation slice 3 (epoch 0, window 2, digest b40402..): full wave-N/O build.

First never-touched window measuring together: photo-caption capture, reader form floor,
irregular-past claims, plural-list scaffold, last-monthname, bare-day, ordinal
interpolation, future-polarity floor. Same protocol: eidetic-full vs mem0, holdout
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
OUT = ROOT / "artifacts/holdout_rotation_r3_codex"
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
print("launching rotation-r3 h2h:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))
