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


def test_ask_problem_marks_revision_backed_citations(prob_engine, monkeypatch):
    """NL questions run through the SAME ask path; citations pointing into this problem's
    revision records are marked revision-backed, general memories are not, and the folded
    state rides along. as_of replays the state."""
    from eidetic.models import Answer, Citation, NLILabel

    p = mcp_server.remember_problem(goal="Checkout latency spikes at peak", valid_at=100.0)
    pid = p["problem_id"]
    mcp_server.update_problem(
        pid, decisions=[{"choice": "raise pool size to 64"}], valid_at=200.0)
    rev_ids = [r.memory_id for r in problems.problem_revisions(prob_engine, pid)]

    def fake_ask(query, scope=None, as_of=None, **kw):
        return Answer(question=query, answer="We decided to raise the pool size to 64.",
                      verified=True, confidence=0.9,
                      citations=[
                          Citation(memory_id=rev_ids[-1], content_hash="h", raw_uri="",
                                   source="problem", valid_at=200.0, snippet="raise pool",
                                   nli_label=NLILabel.ENTAILMENT, nli_score=0.95),
                          Citation(memory_id="mem_general", content_hash="h2", raw_uri="",
                                   source="user", valid_at=50.0, snippet="unrelated",
                                   nli_label=NLILabel.ENTAILMENT, nli_score=0.9),
                      ],
                      unverified_claims=[], generated_by="test", retrieved_count=2, note="")

    monkeypatch.setattr(prob_engine, "ask", fake_ask)
    out = mcp_server.ask_problem(pid, "What did we decide about the pool size?")
    assert out["verified"] is True and out["revision_backed_count"] == 1
    backed = {c["memory_id"]: c["revision_backed"] for c in out["citations"]}
    assert backed[rev_ids[-1]] is True and backed["mem_general"] is False
    assert out["state"]["decisions"][0]["choice"] == "raise pool size to 64"

    with pytest.raises(KeyError):
        mcp_server.ask_problem("prob_nope", "anything?")


def _claims_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("PROBLEM_CLAIMS", "1")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    get_settings.cache_clear()
    eng = Engine(get_settings(), client=_FakeClient(get_settings().embed_dim))
    return eng


def test_problem_claims_answer_structurally_when_flag_on(tmp_path, monkeypatch):
    """P6: with PROBLEM_CLAIMS=1 every revision emits typed claims into the SAME tier the
    executor reads, and war-room questions answer via a typed SELECT (:problem note) --
    decision, rationale, blocker, hypothesis, status -- each carrying the revision text as
    its proof atom. Flag off emits no problem claims, so the baseline is byte-identical
    by construction."""
    from eidetic.models import Scope as _S
    from eidetic.smqe.executor import execute_plan
    from eidetic.smqe.planner import plan_query

    eng = _claims_engine(tmp_path, monkeypatch)
    p = problems.remember_problem(eng, "Checkout latency spikes at peak",
                                  blockers=["no staging repro"], valid_at=100.0)
    pid = p["problem_id"]
    problems.add_hypothesis(eng, pid, "Connection pool exhaustion under burst traffic",
                            valid_at=150.0)
    problems.update_problem(eng, pid, decisions=[
        {"choice": "raise the pool size to 64",
         "rationale": "pool saturation confirmed at every spike"}], valid_at=200.0)

    claims = list(eng.store.claims_in_scope(_S()))
    assert {c.claim_type for c in claims} == {"problem"}
    recs = eng.store.active_records_at(scope=_S())

    def ask(q):
        return execute_plan(plan_query(q), q, records=recs, claims=claims)

    res = ask("What did we decide about the pool size?")
    assert res.answer == "raise the pool size to 64" and ":problem" in res.note
    assert "we decided: raise the pool size to 64" in res.supports[0].proof_atom

    assert ask("Why did we decide to raise the pool size?").answer == \
        "pool saturation confirmed at every spike"
    assert ask("What is blocking the checkout latency problem?").answer == "no staging repro"
    assert ask("What hypotheses do we have?").answer == \
        "Connection pool exhaustion under burst traffic"
    assert ask("What is the status of the checkout problem?").answer == "open"


def test_problem_claims_flag_off_emits_nothing(prob_engine):
    from eidetic.models import Scope as _S

    p = mcp_server.remember_problem(goal="Flaky nightly build on arm64 runners")
    mcp_server.update_problem(p["problem_id"], decisions=[{"choice": "pin the runner image"}])
    assert list(prob_engine.store.claims_in_scope(_S())) == []


def test_problem_extract_folds_conversation_into_war_room(tmp_path, monkeypatch):
    """PROBLEM_EXTRACT=1 (default off): explicit problem-shaped utterances at ingest fold
    into the war room -- goal opens the problem, blocker/hypothesis/decision/handoff
    attach to it with the record's valid_at, and root-cause becomes a hypothesis. With
    PROBLEM_CLAIMS also on, the decision then answers structurally."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("PROBLEM_EXTRACT", "1")
    monkeypatch.setenv("PROBLEM_CLAIMS", "1")
    monkeypatch.delenv("EIDETIC_NAMESPACE", raising=False)
    get_settings.cache_clear()
    eng = Engine(get_settings(), client=_FakeClient(get_settings().embed_dim))

    eng.ingest_text("problem: checkout latency spikes at peak", valid_at=100.0,
                    consolidate_now=False)
    eng.ingest_text("blocker: no staging repro", valid_at=150.0, consolidate_now=False)
    eng.ingest_text("After the metrics review we decided to raise the pool size to 64 "
                    "because saturation was confirmed.", valid_at=200.0,
                    consolidate_now=False)

    state = problems.recall_problem(eng, query="checkout latency")
    assert state is not None
    assert state["blockers"] == ["no staging repro"]
    assert state["decisions"][0]["choice"] == "raise the pool size to 64"
    assert state["decisions"][0]["rationale"].startswith("saturation was confirmed")

    from eidetic.models import Scope as _S
    from eidetic.smqe.executor import execute_plan
    from eidetic.smqe.planner import plan_query
    q = "What did we decide about the pool size?"
    res = execute_plan(plan_query(q), q,
                       records=eng.store.active_records_at(scope=_S()),
                       claims=eng.store.claims_in_scope(_S()))
    assert res is not None and res.answer == "raise the pool size to 64"
    get_settings.cache_clear()


def test_problem_extract_flag_off_is_inert(prob_engine):
    prob_engine.ingest_text("blocker: nothing should happen", consolidate_now=False)
    assert problems.recall_problem(prob_engine, query="nothing should happen") is None
