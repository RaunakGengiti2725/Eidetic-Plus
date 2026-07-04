#!/usr/bin/env bash
# The war-room story, offline (fake embeddings, zero API): problem -> hypothesis ->
# witness file -> decision -> structured verified answers -> as_of replay.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_DIR="$(mktemp -d)/data" VECTOR_BACKEND=numpy PROBLEM_CLAIMS=1 \
"$ROOT/.venv/bin/python" - <<'PYEOF'
import hashlib, re, sys, tempfile
sys.path.insert(0, ".")
import numpy as np
from eidetic.config import get_settings
from eidetic.engine import Engine
from eidetic.models import Scope
from eidetic import problems
from eidetic.smqe.executor import execute_plan
from eidetic.smqe.planner import plan_query

class FakeClient:
    def __init__(self, dim): self.dim = dim
    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    def embed_text(self, t): return self._e(t)
    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)

eng = Engine(get_settings(), client=FakeClient(get_settings().embed_dim))
p = problems.remember_problem(eng, "Checkout latency spikes above 2s at peak",
                              blockers=["no staging repro"], valid_at=100.0)
pid = p["problem_id"]
h = problems.add_hypothesis(eng, pid, "Connection pool exhaustion under burst traffic",
                            valid_at=200.0)
blob = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
blob.write(b"19:00 pool=64/64 saturated\n"); blob.close()
w = problems.add_witness(eng, pid, blob.name, note="pool saturation log", valid_at=250.0)
problems.resolve_hypothesis(eng, pid, h["hypothesis_id"], "confirmed",
                            rationale="pool at 100% during every spike",
                            evidence=[w["witness_memory_id"]], valid_at=300.0)
problems.update_problem(eng, pid, status="resolved",
                        decisions=[{"choice": "raise the pool size to 64",
                                    "rationale": "confirmed saturation hypothesis"}],
                        valid_at=400.0)

recs = eng.store.active_records_at(scope=Scope())
claims = eng.store.claims_in_scope(Scope())
print("== structured, verified-by-construction answers ==")
for q in ("What did we decide about the pool size?",
          "Why did we decide to raise the pool size?",
          "What hypotheses do we have?",
          "What is the status of the checkout problem?"):
    r = execute_plan(plan_query(q), q, records=recs, claims=claims)
    print(f"  {q}\n    -> {r.answer}   [{r.note}]")
print("== witness integrity ==")
print("  hash re-verifies:", eng.substrate.verify(w["content_hash"]))
print("== time travel (as of t=250, mid-investigation) ==")
past = problems.recall_problem(eng, problem_id=pid, as_of=250.0)
print(f"  status={past['status']}  hypothesis={past['hypotheses'][0]['status']}")
PYEOF
