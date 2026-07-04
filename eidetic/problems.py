"""Wings 7: collective problem memory (war room) over the existing spine.

A PROBLEM is an ordinary immutable record whose metadata carries a typed revision of
problem state (goal, status, blockers, hypotheses, handoffs, decisions). Every update is
a NEW record referencing the same problem_id -- bitemporal by construction, nothing is
ever mutated or deleted, and the fold over revisions IS the current state. Hypotheses
carry evidence memory-id refs so `prove`/`truth_ledger` resolve them like any citation.

This module is pure state logic plus thin engine glue; no retrieval or verification
machinery is duplicated. MCP tools in mcp_server.py are doors over these functions.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from .models import MemoryRecord, Scope

_OPEN_STATUSES = {"open", "investigating", "blocked", "mitigated"}
_TERMINAL_STATUSES = {"resolved", "abandoned"}
VALID_STATUSES = _OPEN_STATUSES | _TERMINAL_STATUSES
HYPOTHESIS_STATUSES = {"proposed", "supported", "refuted", "confirmed"}


def _revision_payload(rec: MemoryRecord) -> Optional[dict]:
    payload = rec.metadata.get("problem")
    return payload if isinstance(payload, dict) else None


def problem_revisions(engine, problem_id: str, *, scope: Optional[Scope] = None) -> list[MemoryRecord]:
    """All revision records for a problem in-scope, oldest first. Scope-safe: a problem
    written in namespace A is invisible from namespace B (same rule as every read)."""
    scope = scope or Scope()
    out = [r for r in engine.store.all_records(scope)
           if (_revision_payload(r) or {}).get("problem_id") == problem_id]
    out.sort(key=lambda r: (r.valid_at or 0.0, r.created_at or 0.0))
    return out


def fold_state(revisions: list[MemoryRecord]) -> Optional[dict]:
    """Deterministic fold: later revisions override scalar fields; list fields append;
    hypotheses merge by id with the LATEST status/rationale winning (retraction-friendly,
    same latest-wins rule as the truth ledger)."""
    if not revisions:
        return None
    state: dict = {"problem_id": "", "goal": "", "status": "open", "blockers": [],
                   "hypotheses": [], "handoffs": [], "decisions": [], "witnesses": [],
                   "revisions": 0}
    hyps: dict[str, dict] = {}
    for rec in revisions:
        p = _revision_payload(rec) or {}
        state["problem_id"] = p.get("problem_id") or state["problem_id"]
        if p.get("goal"):
            state["goal"] = p["goal"]
        if p.get("status"):
            state["status"] = p["status"]
        for field in ("blockers", "handoffs", "decisions", "witnesses"):
            for item in p.get(field) or []:
                if item not in state[field]:
                    state[field].append(item)
        for h in p.get("hypotheses") or []:
            hid = h.get("hypothesis_id") or ""
            if not hid:
                continue
            merged = dict(hyps.get(hid) or {})
            merged.update({k: v for k, v in h.items() if v not in (None, "")})
            merged.setdefault("evidence", [])
            for ref in h.get("evidence") or []:
                if ref not in merged["evidence"]:
                    merged["evidence"].append(ref)
            hyps[hid] = merged
        state["revisions"] += 1
        state["as_of"] = rec.valid_at
    state["hypotheses"] = list(hyps.values())
    return state


def _write_revision(engine, payload: dict, *, scope: Optional[Scope],
                    valid_at: Optional[float], text: str) -> MemoryRecord:
    rec = engine.ingest_text(text, source="problem", valid_at=valid_at,
                             extract_graph=False, scope=scope, consolidate_now=False)
    engine.set_metadata(rec.memory_id, {"problem": payload}, scope=scope)
    if getattr(engine.settings, "problem_claims_enabled", False):
        engine.store.add_claims(_claims_for_revision(rec, payload, scope or Scope()))
    return engine.store.get_record(rec.memory_id) or rec


def _claims_for_revision(rec: MemoryRecord, payload: dict, scope: Scope) -> list:
    """PROBLEM_CLAIMS: every revision element becomes a typed claim in the SAME tier SMQE
    and verify already read -- goal/blocker/decision/handoff/status as `problem` claims,
    hypotheses with their id and status, witnesses as `witness` claims carrying the
    substrate content hash. Proof atoms quote the revision text verbatim so the anchor
    rule verifies them like any extracted claim."""
    from eidetic.models import ClaimRecord

    pid = payload.get("problem_id") or ""
    out: list[ClaimRecord] = []

    def claim(ctype, predicate, obj, extra_filters=None, value=None):
        filters = {"problem_id": pid}
        filters.update(extra_filters or {})
        out.append(ClaimRecord(
            claim_type=ctype, scope=scope, subject=pid, predicate=predicate,
            object=str(obj), value=value if value is not None else str(obj),
            filters=filters, valid_at=rec.valid_at,
            source_memory_id=rec.memory_id, proof_atom=rec.text or ""))

    if payload.get("goal"):
        claim("problem", "goal", payload["goal"])
    if payload.get("status"):
        claim("problem", "status", payload["status"])
    for b in payload.get("blockers") or []:
        claim("problem", "blocker", b)
    for h in payload.get("handoffs") or []:
        claim("problem", "handoff", h)
    for d in payload.get("decisions") or []:
        claim("problem", "decision", d.get("choice", ""),
              {"rationale": d.get("rationale", "")})
    for h in payload.get("hypotheses") or []:
        claim("problem", "hypothesis", h.get("claim") or h.get("rationale") or "",
              {"hypothesis_id": h.get("hypothesis_id", ""),
               "status": h.get("status", ""),
               "evidence": list(h.get("evidence") or [])})
    for w in payload.get("witnesses") or []:
        claim("witness", "witness", w.get("note") or w.get("content_hash", ""),
              {"content_hash": w.get("content_hash", ""),
               "memory_id": w.get("memory_id", "")})
    return out


def remember_problem(engine, goal: str, *, scope: Optional[Scope] = None,
                     status: str = "open", blockers: Optional[list[str]] = None,
                     valid_at: Optional[float] = None) -> dict:
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    problem_id = f"prob_{uuid.uuid4().hex[:12]}"
    payload = {"problem_id": problem_id, "goal": goal, "status": status,
               "blockers": list(blockers or [])}
    rec = _write_revision(engine, payload, scope=scope, valid_at=valid_at,
                          text=f"[problem {problem_id}] goal: {goal} (status: {status})")
    return {"problem_id": problem_id, "memory_id": rec.memory_id,
            "state": fold_state(problem_revisions(engine, problem_id, scope=scope))}


def update_problem(engine, problem_id: str, *, scope: Optional[Scope] = None,
                   status: Optional[str] = None, blockers: Optional[list[str]] = None,
                   handoffs: Optional[list[str]] = None,
                   decisions: Optional[list[dict]] = None,
                   valid_at: Optional[float] = None) -> dict:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    if not problem_revisions(engine, problem_id, scope=scope):
        raise KeyError(f"no such problem in scope: {problem_id}")
    for d in decisions or []:
        if not (isinstance(d, dict) and d.get("choice")):
            raise ValueError("each decision needs at least a 'choice'")
    payload = {"problem_id": problem_id}
    if status:
        payload["status"] = status
    if blockers:
        payload["blockers"] = list(blockers)
    if handoffs:
        payload["handoffs"] = list(handoffs)
    if decisions:
        payload["decisions"] = list(decisions)
    parts = [p for p in (f"status: {status}" if status else "",
                         f"blockers: {', '.join(blockers)}" if blockers else "",
                         f"handoffs: {', '.join(handoffs)}" if handoffs else "",
                         "; ".join(f"we decided: {d['choice']}"
                                   + (f" because {d['rationale']}" if d.get("rationale") else "")
                                   for d in decisions or []) if decisions else "") if p]
    rec = _write_revision(engine, payload, scope=scope, valid_at=valid_at,
                          text=f"[problem {problem_id}] update: {'; '.join(parts) or 'noted'}")
    return {"problem_id": problem_id, "memory_id": rec.memory_id,
            "state": fold_state(problem_revisions(engine, problem_id, scope=scope))}


def add_hypothesis(engine, problem_id: str, claim: str, *,
                   scope: Optional[Scope] = None,
                   evidence: Optional[list[str]] = None,
                   valid_at: Optional[float] = None) -> dict:
    if not problem_revisions(engine, problem_id, scope=scope):
        raise KeyError(f"no such problem in scope: {problem_id}")
    scope = scope or Scope()
    missing = [m for m in (evidence or []) if engine.get_record_in_scope(m, scope) is None]
    if missing:
        raise ValueError(f"evidence refs not found in scope: {missing}")
    hypothesis_id = f"hyp_{uuid.uuid4().hex[:10]}"
    payload = {"problem_id": problem_id, "hypotheses": [{
        "hypothesis_id": hypothesis_id, "claim": claim, "status": "proposed",
        "evidence": list(evidence or []),
    }]}
    rec = _write_revision(engine, payload, scope=scope, valid_at=valid_at,
                          text=f"[problem {problem_id}] hypothesis {hypothesis_id}: {claim}")
    return {"problem_id": problem_id, "hypothesis_id": hypothesis_id,
            "memory_id": rec.memory_id,
            "state": fold_state(problem_revisions(engine, problem_id, scope=scope))}


def resolve_hypothesis(engine, problem_id: str, hypothesis_id: str, status: str, *,
                       scope: Optional[Scope] = None, rationale: str = "",
                       evidence: Optional[list[str]] = None,
                       valid_at: Optional[float] = None) -> dict:
    if status not in HYPOTHESIS_STATUSES:
        raise ValueError(f"hypothesis status must be one of {sorted(HYPOTHESIS_STATUSES)}")
    state = fold_state(problem_revisions(engine, problem_id, scope=scope))
    if state is None:
        raise KeyError(f"no such problem in scope: {problem_id}")
    if hypothesis_id not in {h.get("hypothesis_id") for h in state["hypotheses"]}:
        raise KeyError(f"no such hypothesis on {problem_id}: {hypothesis_id}")
    payload = {"problem_id": problem_id, "hypotheses": [{
        "hypothesis_id": hypothesis_id, "status": status, "rationale": rationale,
        "evidence": list(evidence or []),
    }]}
    rec = _write_revision(engine, payload, scope=scope, valid_at=valid_at,
                          text=f"[problem {problem_id}] hypothesis {hypothesis_id} -> {status}"
                               + (f": {rationale}" if rationale else ""))
    return {"problem_id": problem_id, "hypothesis_id": hypothesis_id,
            "memory_id": rec.memory_id,
            "state": fold_state(problem_revisions(engine, problem_id, scope=scope))}


def add_witness(engine, problem_id: str, path: str, *, scope: Optional[Scope] = None,
                note: str = "", valid_at: Optional[float] = None) -> dict:
    """Wings 8 scaffold: attach an operational-truth WITNESS (screenshot, log, dump) to a
    problem. The file lands losslessly in the content-addressed substrate (byte-identical
    retrieval via get_raw); the problem gains a witness entry carrying the memory_id,
    content hash, and note, so any hypothesis or answer citing it is checkable down to
    the raw bytes."""
    if not problem_revisions(engine, problem_id, scope=scope):
        raise KeyError(f"no such problem in scope: {problem_id}")
    rec = engine.ingest_file(path, source=f"witness:{problem_id}", valid_at=valid_at,
                             extract_graph=False, scope=scope, consolidate_now=False)
    payload = {"problem_id": problem_id, "handoffs": [], "witnesses": [{
        "memory_id": rec.memory_id, "content_hash": rec.content_hash,
        "raw_uri": rec.raw_uri, "note": note,
    }]}
    marker = _write_revision(engine, payload, scope=scope, valid_at=valid_at,
                             text=f"[problem {problem_id}] witness {rec.content_hash[:12]}"
                                  + (f": {note}" if note else ""))
    return {"problem_id": problem_id, "witness_memory_id": rec.memory_id,
            "content_hash": rec.content_hash, "revision_memory_id": marker.memory_id,
            "state": fold_state(problem_revisions(engine, problem_id, scope=scope))}


def ask_problem(engine, problem_id: str, question: str, *,
                scope: Optional[Scope] = None,
                as_of: Optional[float] = None) -> dict:
    """Natural-language question against a problem's war-room history through the SAME
    verify-or-abstain ask path as every other answer. Revision records are ordinary
    memories, so the answer's citations resolve normally; the response marks which
    citations point INTO this problem's revisions (revision-backed) versus general
    memory, and carries the folded state for context."""
    scope = scope or Scope()
    revisions = problem_revisions(engine, problem_id, scope=scope)
    if not revisions:
        raise KeyError(f"no such problem in scope: {problem_id}")
    revision_ids = {r.memory_id for r in revisions}
    ans = engine.ask(question, scope=scope, as_of=as_of)
    citations = []
    for c in ans.citations or []:
        citations.append({
            "memory_id": c.memory_id,
            "snippet": c.snippet,
            "nli_label": getattr(c.nli_label, "value", str(c.nli_label)),
            "revision_backed": c.memory_id in revision_ids,
        })
    if as_of is not None:
        revisions = [r for r in revisions if (r.valid_at or 0.0) <= as_of]
    return {
        "problem_id": problem_id,
        "question": question,
        "answer": ans.answer,
        "verified": ans.verified,
        "abstained": ans.note.startswith("abstained") if ans.note else False,
        "citations": citations,
        "revision_backed_count": sum(1 for c in citations if c["revision_backed"]),
        "state": fold_state(revisions),
    }


def recall_problem(engine, *, problem_id: Optional[str] = None, query: str = "",
                   scope: Optional[Scope] = None,
                   as_of: Optional[float] = None) -> Optional[dict]:
    """Current folded state (or state AS OF a past moment -- revisions after as_of are
    invisible, same bitemporal rule as recall). With no problem_id, the newest problem
    whose goal/text matches the query terms wins; None when nothing matches."""
    scope = scope or Scope()
    if problem_id is None:
        terms = {t for t in (query or "").lower().split() if len(t) > 2}
        best: tuple[float, str] | None = None
        for rec in engine.store.all_records(scope):
            p = _revision_payload(rec)
            if not p or not p.get("goal"):
                continue
            hay = (p.get("goal", "") + " " + rec.text).lower()
            if terms and not all(t in hay for t in terms):
                continue
            key = (rec.valid_at or 0.0, p["problem_id"])
            if best is None or key > best:
                best = key
        if best is None:
            return None
        problem_id = best[1]
    revisions = problem_revisions(engine, problem_id, scope=scope)
    if as_of is not None:
        revisions = [r for r in revisions if (r.valid_at or 0.0) <= as_of]
    return fold_state(revisions)


_EXTRACT_RULES = (
    (re.compile(r"^\s*(?:problem|goal)\s*:\s*(.+)$", re.I | re.M), "goal"),
    (re.compile(r"^\s*blocker\s*:\s*(.+)$", re.I | re.M), "blocker"),
    (re.compile(r"^\s*hypothesis\s*:\s*(.+)$", re.I | re.M), "hypothesis"),
    (re.compile(r"\bwe\s+decided\s+(?:to\s+)?([^.;!?\n]+)", re.I), "decision"),
    (re.compile(r"\bhand(?:ing)?\s*(?:off|over)\s+to\s+([^.;!?\n]+)", re.I), "handoff"),
    (re.compile(r"\broot\s+cause(?:\s+is|:)\s*([^.;!?\n]+)", re.I), "root_cause"),
)


def extract_problem_signals(text: str) -> list[tuple[str, str]]:
    """Rules-first detection of problem-shaped utterances. Returns (kind, payload) pairs
    in document order; deliberately conservative -- explicit markers only, no inference."""
    out: list[tuple[str, str]] = []
    for pat, kind in _EXTRACT_RULES:
        for m in pat.finditer(text or ""):
            value = m.group(1).strip().rstrip(".")
            if value:
                out.append((kind, value))
    return out


def apply_extracted_signals(engine, rec, *, scope: Optional[Scope] = None) -> Optional[str]:
    """PROBLEM_EXTRACT hook: fold detected signals into the war room. A goal signal opens
    a new problem; every other signal attaches to the most recent problem in scope (none
    existing -> an implicit problem is opened from the signal itself). Revisions carry the
    triggering record's valid_at, so as_of replay stays truthful. Returns the problem_id
    touched, or None when the text carries no signals."""
    signals = extract_problem_signals(rec.text or "")
    if not signals:
        return None
    scope = scope or Scope()
    pid: Optional[str] = None
    for kind, value in signals:
        if kind == "goal":
            pid = remember_problem(engine, value, scope=scope,
                                   valid_at=rec.valid_at)["problem_id"]
            continue
        if pid is None:
            latest = recall_problem(engine, scope=scope)
            pid = latest["problem_id"] if latest else remember_problem(
                engine, f"(implicit) {value}", scope=scope,
                valid_at=rec.valid_at)["problem_id"]
        if kind == "blocker":
            update_problem(engine, pid, scope=scope, blockers=[value], valid_at=rec.valid_at)
        elif kind == "hypothesis":
            add_hypothesis(engine, pid, value, scope=scope, valid_at=rec.valid_at)
        elif kind == "decision":
            m = re.match(r"(.+?)\s+because\s+(.+)$", value, re.I)
            decision = ({"choice": m.group(1).strip(), "rationale": m.group(2).strip()}
                        if m else {"choice": value})
            update_problem(engine, pid, scope=scope, decisions=[decision], valid_at=rec.valid_at)
        elif kind == "handoff":
            update_problem(engine, pid, scope=scope, handoffs=[value], valid_at=rec.valid_at)
        elif kind == "root_cause":
            add_hypothesis(engine, pid, f"root cause: {value}", scope=scope,
                           valid_at=rec.valid_at)
    return pid
