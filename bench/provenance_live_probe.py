"""LIVE validation of the [n]->content_hash citation map on the LME-S retrieval-guided
notebooks (created this morning, sources already exported). Re-queries each notebook via
bridge.answer() -- which now returns citation_map -- and measures the goal bar: fraction of
rows with >=1 citation resolved to a content hash. Quota-frugal: --limit N rows."""
import json
import os
import subprocess
import sys

os.environ.setdefault("DATA_DIR", "artifacts/lme_s_r1_codex/data")

from eidetic.engine import Engine
from eidetic.integrations.notebooklm import CliBackend, NotebookLMBridge

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = "artifacts/lme_s_r1_codex/provenance_citation_map_live.json"

# sample_id -> notebook id from the live notebook list
raw = subprocess.run([".venv/bin/nlm", "notebook", "list"], capture_output=True, text=True).stdout
notebooks = {n["title"].replace("eidetic-rgi-", ""): n["id"]
             for n in json.loads(raw) if n.get("title", "").startswith("eidetic-rgi-")}
rows = [json.loads(l) for l in open("artifacts/lme_s_r1_codex/notebooklm_rg_injected.jsonl")
        if l.strip()]
rows = [r for r in rows if "question" in r and r["sample_id"] in notebooks]
print(f"matched {len(rows)} rows to live notebooks; probing {min(LIMIT, len(rows))}")

eng = Engine()
bridge = NotebookLMBridge(eng, CliBackend())
results = []
for r in rows[:LIMIT]:
    sid, ns, q = r["sample_id"], r["namespace"], r["question"]
    try:
        out = bridge.answer(ns, q, notebooks[sid])
        cmap = out.get("citation_map") or []
        resolved = [c for c in cmap if c.get("resolved")]
        rec = {"sample_id": sid, "n_references": len(cmap), "n_resolved": len(resolved),
               "row_resolved": bool(resolved),
               "matches": [c.get("match") for c in resolved],
               "reasons_unresolved": [c.get("reason") for c in cmap if not c.get("resolved")],
               "hashes": sorted({c["content_sha256"][:12] for c in resolved})}
    except Exception as e:
        rec = {"sample_id": sid, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    results.append(rec)
    print(json.dumps(rec))

ok_rows = [x for x in results if x.get("row_resolved")]
err_rows = [x for x in results if x.get("error")]
n_meas = len(results) - len(err_rows)
print(f"\nrows with >=1 resolved citation: {len(ok_rows)}/{n_meas} "
      f"({(100.0 * len(ok_rows) / n_meas) if n_meas else 0:.0f}%)  errors={len(err_rows)}  "
      f"goal bar: >80%")
json.dump(results, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
