"""Wings 7: collective problem memory -- bitemporal war-room state over the spine."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from eidetic import mcp_server, problems
from eidetic.config import get_settings
from eidetic.engine import Engine
from eidetic.models import Scope


class _FakeClient:
    def __init__(self, dim):
        self.dim = dim

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return (np.stack([self._e(t) for t in ts]) if ts
                else np.zeros((0, self.dim), np.float32))


@pytest.fixture()
def prob_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    get_settings.cache_clear()
    eng = Engine(get_settings(), client=_FakeClient(get_settings().embed_dim))
    monkeypatch.setattr(mcp_server, "_engine", eng)
    yield eng
    monkeypatch.setattr(mcp_server, "_engine", None)
    get_settings.cache_clear()


def test_problem_lifecycle_folds_revisions_bitemporally(prob_engine):
    out = mcp_server.remember_problem(goal="Checkout latency spikes above 2s at peak",
                                      blockers=["no staging repro"], valid_at=100.0)
    pid = out["problem_id"]
    assert out["state"]["status"] == "open"
    assert out["state"]["blockers"] == ["no staging repro"]

    hyp = mcp_server.add_hypothesis(pid, "Connection pool exhaustion under burst traffic",
                                    valid_at=200.0)
    hid = hyp["hypothesis_id"]
    assert hyp["state"]["hypotheses"][0]["status"] == "proposed"

    mcp_server.update_problem(pid, status="investigating",
                              handoffs=["night shift: check pool metrics"], valid_at=300.0)
    res = mcp_server.resolve_hypothesis(pid, hid, "confirmed",
                                        rationale="pool at 100% during every spike",
                                        valid_at=400.0)
    assert res["state"]["hypotheses"][0]["status"] == "confirmed"
    assert res["state"]["hypotheses"][0]["rationale"].startswith("pool at 100%")

    done = mcp_server.update_problem(
        pid, status="resolved",
        decisions=[{"choice": "raise pool size to 64", "rationale": "confirmed hypothesis",
                    "witnesses": ["night shift"]}], valid_at=500.0)
    st = done["state"]
    assert st["status"] == "resolved" and st["revisions"] == 5
    assert st["decisions"][0]["choice"] == "raise pool size to 64"

    # bitemporal replay: as of t=350 the hypothesis was still proposed, status investigating
    past = mcp_server.recall_problem(problem_id=pid, as_of=350.0)
    assert past["status"] == "investigating"
    assert past["hypotheses"][0]["status"] == "proposed"


def test_problem_query_recall_and_scope_isolation(prob_engine):
    a = mcp_server.remember_problem(goal="Flaky nightly build on arm64 runners")
    mcp_server.remember_problem(goal="Support tickets spike after billing change",
                                namespace="other-team")

    hit = mcp_server.recall_problem(query="nightly build arm64")
    assert hit["problem_id"] == a["problem_id"]

    with pytest.raises(RuntimeError, match="no matching problem"):
        mcp_server.recall_problem(query="billing tickets")   # other namespace invisible

    with pytest.raises(KeyError):
        problems.update_problem(prob_engine, a["problem_id"],
                                scope=Scope(namespace="other-team"), status="resolved")


def test_hypothesis_evidence_refs_are_scope_validated(prob_engine):
    rec = prob_engine.ingest_text("pool_metrics.png shows saturation at 19:00",
                                  scope=Scope(), consolidate_now=False)
    p = mcp_server.remember_problem(goal="API errors during evening peak")
    out = mcp_server.add_hypothesis(p["problem_id"], "Saturation starts at 19:00",
                                    evidence=[rec.memory_id])
    assert out["state"]["hypotheses"][0]["evidence"] == [rec.memory_id]

    with pytest.raises(ValueError, match="evidence refs not found"):
        mcp_server.add_hypothesis(p["problem_id"], "ghost evidence", evidence=["mem_nope"])

    with pytest.raises(ValueError):
        mcp_server.update_problem(p["problem_id"], decisions=[{"rationale": "no choice"}])

    with pytest.raises(ValueError):
        mcp_server.remember_problem(goal="bad status", status="done")


def test_witness_file_attaches_hash_checked_evidence(prob_engine, tmp_path):
    """Wings 8 scaffold: a witness file lands losslessly in the substrate; the problem's
    folded state carries its content hash; the raw bytes come back byte-identical and the
    hash re-verifies (the same tamper check the proof surface uses)."""
    blob = tmp_path / "pool_metrics.log"
    blob.write_bytes(b"19:00 pool=64/64 saturated\n19:05 pool=64/64 saturated\n")

    p = mcp_server.remember_problem(goal="API errors during evening peak")
    out = mcp_server.add_witness(p["problem_id"], str(blob), note="pool saturation log")
    assert out["content_hash"]
    st = out["state"]
    assert st["witnesses"][0]["content_hash"] == out["content_hash"]
    assert st["witnesses"][0]["note"] == "pool saturation log"

    raw = prob_engine.get_raw(out["content_hash"])
    assert raw == blob.read_bytes()
    assert prob_engine.substrate.verify(out["content_hash"]) is True

    hyp = mcp_server.add_hypothesis(p["problem_id"], "Pool saturation causes the errors",
                                    evidence=[out["witness_memory_id"]])
    assert hyp["state"]["hypotheses"][0]["evidence"] == [out["witness_memory_id"]]

    with pytest.raises(KeyError):
        mcp_server.add_witness("prob_nope", str(blob))
