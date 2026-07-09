"""Live-store validation of the _measure_type_sum scope-gate fix (task #42). Runs the
deterministic structured_recall path against the REAL built LME-S stores for the numeric-
aggregation questions -- proves the fix returns a sane total (bc149d6b -> ~70, not 1226.3)
on real data, not just synthetic atoms. Read-only over the stores; embeddings are cached."""
import json
import os
import re

os.environ.setdefault("DATA_DIR", "artifacts/lme_s_r1_codex/data")

from eidetic.engine import Engine
from eidetic.models import Scope

rows = [json.loads(l) for l in open("artifacts/lme_s_r1_codex/notebooklm_rg_injected.jsonl")
        if l.strip()]
NUM = re.compile(r"how (many|much)|total|weight|number of|\bcount\b", re.I)
targets = [r for r in rows if "question" in r and NUM.search(r["question"])]

eng = Engine()
print(f"DATA_DIR={os.environ['DATA_DIR']}  numeric questions={len(targets)}\n")
out = []
for r in targets:
    ns, q, gold = r["namespace"], r["question"], r["gold"]
    try:
        res = eng.structured_recall(q, scope=Scope(namespace=ns))
        ans = res.get("answer") or ""
        answered = res.get("answered")
        verified = res.get("verified")
        op = (res.get("plan") or {}).get("op") if isinstance(res.get("plan"), dict) else res.get("op")
    except Exception as e:
        ans, answered, verified, op = f"ERR:{type(e).__name__}:{str(e)[:60]}", None, None, None
    rec = {"sample_id": r["sample_id"], "op": op, "answered": answered,
           "verified": verified, "answer": ans, "gold": gold, "question": q}
    out.append(rec)
    print(f"{r['sample_id'][:14]:14} op={str(op):18} ans={ans[:48]!r:50} gold={gold[:32]!r}")

json.dump(out, open("artifacts/lme_s_r1_codex/measure_sum_live.json", "w"), indent=2)
print(f"\nwrote artifacts/lme_s_r1_codex/measure_sum_live.json ({len(out)} rows)")
# spotlight the fixed case
bc = [r for r in out if r["sample_id"].startswith("bc149d6b")]
if bc:
    print(f"\nBC149D6B (the 1226.3->70 case): answer={bc[0]['answer']!r} verified={bc[0]['verified']}")
