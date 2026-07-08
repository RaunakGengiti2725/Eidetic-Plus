"""eidetic -> NotebookLM bridge: export VERIFIED, provenance-tracked memory into a
NotebookLM notebook so a user can query it / generate an audio overview over their
*trustworthy* memory substrate.

WHAT THIS IS (and is not), stated up front so nobody over-claims it:
- This is a CONSUMPTION bridge. eidetic stays the immutable, verified source of truth;
  NotebookLM is a downstream surface (query UI, podcast/audio, sharing). The unique bit
  is that every exported source carries eidetic's provenance header (content hash,
  validity window, source label, and any NLI-entailed claims) -- no other memory tool
  feeds a *verified* substrate into NotebookLM.
- This is NOT a token-cost win for our benchmark. NotebookLM runs on Google's Gemini;
  its tokens are on Google's meter, not eliminated. Answers it returns are Gemini-side
  and are NOT verified by eidetic's proof gate -- for a trustworthy, cited answer, use
  eidetic's own `recall`. `query()` here is a convenience pass-through, clearly labeled.
- It touches NOTHING in the engine/benchmark. Default-off, import-only, no global state.

AUTH (you supply credentials; nothing is hardcoded):
- Enterprise API: a GCP bearer token (env NOTEBOOKLM_ACCESS_TOKEN, e.g.
  `gcloud auth print-access-token`) + project number + location. Requires a NotebookLM
  Enterprise license.
- Community CLI: the `nlm` tool (github.com/jacob-bd/notebooklm-mcp-cli), cookie auth,
  works with a personal Google account. Set backend="cli".

The HTTP transport is injectable (`session=`) so the whole thing is unit-testable with
zero network access.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from eidetic.graph import CO_ACTIVATED, _norm
from eidetic.models import ClaimRecord, Edge, MemoryRecord, Scope

_ENTERPRISE_HOST_TMPL = "https://{loc}-discoveryengine.googleapis.com"
_API_VERSION = "v1alpha"


def _iso(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    try:
        return _dt.datetime.fromtimestamp(
            float(ts), _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(ts)


import re as _re

_EIDETIC_REF_RE = _re.compile(r"eidetic:([0-9a-zA-Z_\-]{4,32})")


def format_source(record: MemoryRecord, claims: Iterable[ClaimRecord] = ()) -> dict:
    """One NotebookLM source built from an eidetic memory, with a PROVENANCE HEADER that
    is the whole point of the bridge: the reader (human or NotebookLM) can trace every
    exported line back to an immutable, hash-addressed, time-scoped record, and see which
    of its facts eidetic verified. Returns {display_name, text_content}."""
    header_lines = [
        "--- EIDETIC VERIFIED MEMORY (provenance) ---",
        f"memory_id: {record.memory_id}",
        f"content_sha256: {record.content_hash}",
        f"valid_at: {_iso(record.valid_at)}",
        f"source: {record.source}",
    ]
    if record.invalid_at is not None:
        header_lines.append(f"invalidated_at: {_iso(record.invalid_at)}  (superseded)")
    claim_list = [c for c in claims if c is not None]
    if claim_list:
        header_lines.append("verified_claims:")
        for c in claim_list[:20]:
            subj = getattr(c, "subject", "") or ""
            pred = getattr(c, "predicate", "") or ""
            obj = getattr(c, "object", "") or ""
            header_lines.append(f"  - {subj} {pred} {obj}".rstrip())
    header_lines.append("--- END PROVENANCE ---")
    body = (record.text or record.summary or "").strip()
    text = "\n".join(header_lines) + "\n\n" + body
    name = f"eidetic:{record.memory_id[:16]} @ {_iso(record.valid_at)}"
    return {"display_name": name, "text_content": text}


# The four honest boundary labels. Carried VERBATIM on every graph source and every
# router return so the honesty can never be dropped downstream (docs/claims.md).
_HONESTY_BOUNDARIES = {
    "caller_token_cost": (
        "~0 tokens on YOUR metered model per recall IF read via NotebookLM/Gemini free "
        "tier -- WITH provenance (content-hash-mapped)."
    ),
    "not_free_globally": (
        "NOT free globally: Google spends real compute on the Gemini read; only the "
        "CALLER's metered model sees ~0 tokens."
    ),
    "not_verified": (
        "A NotebookLM/Gemini ANSWER over this source is Gemini-side and is NOT "
        "eidetic-verify-or-abstain (provenance-mapped only, not gate-verified). Use "
        "eidetic.recall() for a gate-verified cited answer."
    ),
    "not_a_benchmark_row": (
        "NOT a row in the fixed-qwen-reader benchmark table (different, off-meter reader). "
        "No SOTA / best / strongest claim is made or implied."
    ),
}

_GRAPH_HONESTY_BLOCK = "\n".join([
    "HONESTY: ~0 tokens on YOUR metered model IF read via NotebookLM/Gemini free tier",
    "  (Google spends real compute -- NOT free globally). This is the VERIFIED claim graph,",
    "  so the SOURCE is trustworthy; a NotebookLM/Gemini ANSWER over it is Gemini-side and",
    "  NOT eidetic-verify-or-abstain (provenance-mapped only).",
    "  NOT a row in the fixed-qwen-reader benchmark table (different, off-meter reader).",
    "  No SOTA/\"best\" claim.",
])


def _graph_ref(source_memory_id: str) -> str:
    """Short, regex-matching provenance token: eidetic:<memory_id[:16]> (within the
    _EIDETIC_REF_RE {4,32} window and character class)."""
    return "eidetic:" + (source_memory_id or "")[:16]


def format_graph_source(
    edges: "list[Edge]",
    records_by_id: "dict[str, MemoryRecord]",
    *,
    scope_label: str,
    at: Optional[float] = None,
    node_features: Optional[dict] = None,
    include_history: bool = True,
    max_entities: Optional[int] = None,
    include_inferred: bool = False,
) -> dict:
    """Serialize a VERIFIED claim graph into ONE compact, provenance-carrying NotebookLM
    source. Iterates raw `Edge` objects (NOT build_nx, which drops
    source_memory_id/valid_at/invalid_at/supersedes and collapses parallel edges);
    build_nx/node_features are used ONLY for hub ordering.

    Four labeled text regions: HONESTY, PROVENANCE LEGEND (short-id -> immutable
    content_hash, hoisted once), ACTIVE FACTS (entity blocks, hub-first), HISTORY
    (superseded facts with correct successor pointers). Returns
    {display_name, text_content, stats}. compression_ratio is MEASURED per call and
    surfaced -- never asserted (a sparse graph can invert)."""
    # ---- edge filtering (do this FIRST) ----
    survivors: list[Edge] = []
    for e in edges:
        if e.relation == CO_ACTIVATED:
            continue
        if getattr(e, "pruned", False):
            continue
        if e.inferred and not include_inferred:
            continue
        survivors.append(e)

    active = [e for e in survivors if e.is_active_at(at)]
    superseded = [
        e for e in survivors
        if e.invalid_at is not None and not e.is_active_at(at)
    ]

    # ---- supersession reverse index: successor.supersedes == closed_edge.edge_id ----
    successor_of = {e.supersedes: e for e in survivors if e.supersedes}

    # ---- referenced-id accumulator (drives the legend; keeps it minimal) ----
    referenced_ids: list[str] = []
    seen_ref: set[str] = set()

    def _note_ref(mid: str) -> None:
        if mid and mid not in seen_ref:
            seen_ref.add(mid)
            referenced_ids.append(mid)

    # ---- ACTIVE FACTS: group by _norm(src) (matches node_features keys) ----
    groups: dict[str, list[Edge]] = {}
    display_name_of: dict[str, str] = {}
    for e in active:
        key = _norm(e.src)
        groups.setdefault(key, []).append(e)
        display_name_of.setdefault(key, e.src)

    def _order_key(entity_key: str):
        if node_features:
            feat = node_features.get(entity_key, {})
            return (-float(feat.get("degree", 0.0)), -float(feat.get("ppr", 0.0)), entity_key)
        return (0.0, 0.0, entity_key)  # alphabetical fallback (last element breaks ties)

    ordered_keys = sorted(groups.keys(), key=_order_key)
    if max_entities is not None:
        ordered_keys = ordered_keys[:max_entities]

    active_lines: list[str] = []
    n_relations = 0
    for key in ordered_keys:
        active_lines.append(display_name_of[key])
        for e in sorted(groups[key], key=lambda x: (x.relation, x.dst)):
            _note_ref(e.source_memory_id)
            ref = _graph_ref(e.source_memory_id)
            active_lines.append(f"  {e.relation} -> {e.dst}   [{ref} @{_iso(e.valid_at)}]")
            n_relations += 1

    # ---- HISTORY: superseded facts with correct successor pointers ----
    history_lines: list[str] = []
    if include_history:
        for c in sorted(superseded, key=lambda x: (x.valid_at, x.edge_id)):
            _note_ref(c.source_memory_id)
            window = f"{_iso(c.valid_at)}..{_iso(c.invalid_at)}"
            succ = successor_of.get(c.edge_id)
            if succ is not None:
                _note_ref(succ.source_memory_id)
                tail = f"(superseded by {_graph_ref(succ.source_memory_id)})"
            else:
                tail = "(superseded)"
            history_lines.append(f"{c.src} {c.relation} {c.dst}   {window}   {tail}")

    # ---- PROVENANCE LEGEND: one line per DISTINCT referenced source_memory_id ----
    legend_lines: list[str] = []
    for mid in referenced_ids:
        rec = records_by_id.get(mid)
        if rec is None:
            continue
        legend_lines.append(
            f"{_graph_ref(mid)}  sha256={rec.content_hash}  source={rec.source}  "
            f"valid_at={_iso(rec.valid_at)}"
        )

    # ---- assemble text ----
    parts: list[str] = []
    parts.append("--- EIDETIC VERIFIED CLAIM GRAPH (provenance) ---")
    parts.append(f"scope: {scope_label}   as_of: {_iso(at)}")
    parts.append(_GRAPH_HONESTY_BLOCK)
    parts.append("--- END HONESTY ---")
    parts.append("")
    parts.append("--- PROVENANCE LEGEND (short-id -> immutable hash) ---")
    parts.extend(legend_lines)
    parts.append("--- END LEGEND ---")
    parts.append("")
    parts.append("--- ACTIVE FACTS (entity blocks) ---")
    parts.extend(active_lines)
    parts.append("--- END ACTIVE FACTS ---")
    if include_history:
        parts.append("")
        parts.append("--- HISTORY (superseded facts, temporal) ---")
        parts.extend(history_lines)
        parts.append("--- END HISTORY ---")
    text_content = "\n".join(parts)

    # Numerator = chars of ONLY the records actually rendered into this source (referenced,
    # post-max_entities). Summing over ALL records_by_id would inflate the ratio whenever the
    # graph is truncated or graph-sparse -- a big record that was never serialized would count
    # in the "raw" side though it never entered the compacted source. This makes the surfaced
    # compression_ratio a true measure of what was compacted.
    raw_record_chars = 0
    for mid in referenced_ids:
        r = records_by_id.get(mid)
        if r is not None:
            raw_record_chars += len((r.text or r.summary or ""))
    serialized_chars = len(text_content)
    stats = {
        "n_entities": len(ordered_keys),
        "n_relations": n_relations,
        "n_active": len(active),
        "n_superseded": len(superseded),
        "serialized_chars": serialized_chars,
        "raw_record_chars": raw_record_chars,
        "compression_ratio": raw_record_chars / max(1, serialized_chars),
    }
    display_name = f"eidetic:graph:{scope_label} @ {_iso(at)}"
    return {"display_name": display_name, "text_content": text_content, "stats": stats}


class NotebookLMError(RuntimeError):
    pass


@dataclass
class EnterpriseBackend:
    """Official NotebookLM Enterprise API (discoveryengine v1alpha). Auth = GCP bearer
    token; requires an enterprise license. `session` defaults to a requests.Session but is
    injectable for tests."""
    project_number: str
    location: str = "global"
    access_token: Optional[str] = None
    session: Any = None
    endpoint_location: str = "us"

    def __post_init__(self) -> None:
        self.access_token = self.access_token or os.environ.get("NOTEBOOKLM_ACCESS_TOKEN")
        if self.session is None:
            import requests  # local import so the module loads without network deps in tests
            self.session = requests.Session()

    def _headers(self) -> dict:
        if not self.access_token:
            raise NotebookLMError(
                "no bearer token: set NOTEBOOKLM_ACCESS_TOKEN (e.g. `gcloud auth "
                "print-access-token`) or pass access_token=")
        return {"Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"}

    def _base(self) -> str:
        host = _ENTERPRISE_HOST_TMPL.format(loc=self.endpoint_location)
        return (f"{host}/{_API_VERSION}/projects/{self.project_number}"
                f"/locations/{self.location}")

    def batch_create_sources(self, notebook_id: str, sources: list[dict]) -> dict:
        headers = self._headers()  # token check BEFORE we touch the session
        url = f"{self._base()}/notebooks/{notebook_id}/sources:batchCreate"
        payload = {"userContents": [
            {"content": s["text_content"], "displayName": s.get("display_name", "")}
            for s in sources]}
        resp = self.session.post(url, headers=headers, data=json.dumps(payload))
        if getattr(resp, "status_code", 200) >= 400:
            raise NotebookLMError(f"batchCreate {resp.status_code}: {getattr(resp,'text','')}")
        return resp.json()

    def query(self, notebook_id: str, question: str) -> dict:
        headers = self._headers()  # token check BEFORE we touch the session
        url = f"{self._base()}/notebooks/{notebook_id}:query"
        resp = self.session.post(url, headers=headers, data=json.dumps({"query": question}))
        if getattr(resp, "status_code", 200) >= 400:
            raise NotebookLMError(f"query {resp.status_code}: {getattr(resp,'text','')}")
        return resp.json()

    def doctor(self) -> dict:
        """Preflight: is a token present + what endpoints will we hit? Never raises."""
        has_token = bool(self.access_token)
        return {"backend": "enterprise",
                "token_present": has_token,
                "hint": None if has_token else
                        "set NOTEBOOKLM_ACCESS_TOKEN=$(gcloud auth print-access-token)",
                "base_url": self._base(),
                "commands": {
                    "add_source": f"POST {self._base()}/notebooks/<id>/sources:batchCreate",
                    "query": f"POST {self._base()}/notebooks/<id>:query"}}


@dataclass
class CliBackend:
    """Community `nlm` CLI (personal Google account, cookie auth). Shells out; the runner
    is injectable for tests. Uses undocumented internal APIs -- may break; ToS-gray."""
    notebook_id: Optional[str] = None
    runner: Callable[[list[str]], str] = None

    def __post_init__(self) -> None:
        if self.runner is None:
            if shutil.which("nlm") is None:
                raise NotebookLMError("`nlm` CLI not found on PATH "
                                      "(pip install / see github.com/jacob-bd/notebooklm-mcp-cli)")
            self.runner = lambda args: subprocess.run(
                ["nlm", *args], capture_output=True, text=True, check=True).stdout

    def batch_create_sources(self, notebook_id: str, sources: list[dict]) -> dict:
        # Real nlm syntax (notebooklm-mcp-cli): `nlm source add <notebook> --text "..."`.
        # notebook is POSITIONAL (no --notebook flag); text sources take no name flag, so
        # the provenance rides inside the text body (the header), not a display-name arg.
        created = 0
        for s in sources:
            self.runner(["source", "add", notebook_id, "--text", s["text_content"]])
            created += 1
        return {"created": created}

    def query(self, notebook_id: str, question: str) -> dict:
        # Real nlm syntax: `nlm notebook query <notebook> "question"`.
        out = self.runner(["notebook", "query", notebook_id, question])
        return {"answer": out.strip(), "backend": "nlm-cli (gemini-side, UNVERIFIED)"}

    def doctor(self) -> dict:
        """Preflight: is `nlm` reachable + logged in, and what commands will we run? Never
        raises -- returns a status dict so the user sees the plan before a live export."""
        status = {"backend": "cli", "reachable": False, "logged_in": None,
                  "commands": {
                      "add_source": "nlm source add <notebook> --text \"<provenance+body>\"",
                      "query": "nlm notebook query <notebook> \"<question>\"",
                      "login": "nlm login   (check: nlm login --check)"}}
        try:
            out = self.runner(["login", "--check"])
            status["reachable"] = True
            status["logged_in"] = "logged in" if "not" not in (out or "").lower() else "NOT logged in (run `nlm login`)"
        except Exception as exc:  # nlm missing OR not logged in
            status["reachable"] = shutil.which("nlm") is not None
            status["logged_in"] = f"unknown ({type(exc).__name__}); run `nlm login` first"
        return status


class NotebookLMBridge:
    """Ties an eidetic engine to a NotebookLM backend. Read-only on eidetic's side."""

    def __init__(self, engine, backend) -> None:
        self.engine = engine
        self.backend = backend

    def _records(self, namespace: str, limit: Optional[int]) -> list[MemoryRecord]:
        scope = Scope(namespace=namespace)
        recs = self.engine.store.active_records_at(None, scope)
        recs = [r for r in recs if (r.text or r.summary)]
        recs.sort(key=lambda r: r.valid_at or 0.0)
        return recs[:limit] if limit else recs

    def build_sources(self, namespace: str, limit: Optional[int] = None) -> list[dict]:
        out = []
        for rec in self._records(namespace, limit):
            try:
                claims = self.engine.store.claims_by_source(rec.memory_id)
            except Exception:
                claims = []
            out.append(format_source(rec, claims))
        return out

    def export_namespace(self, namespace: str, notebook_id: str, *,
                         limit: Optional[int] = None, batch_size: int = 20,
                         include_graph: bool = False) -> dict:
        """Push a namespace's verified memories into `notebook_id` as sources. Returns a
        summary; raises NotebookLMError on backend failure (no partial silent success).

        include_graph=True (opt-in, default off) appends the VERIFIED CLAIM GRAPH as one
        ADDITIONAL compact provenance source alongside the per-record sources."""
        sources = self.build_sources(namespace, limit)
        if include_graph:
            gsrc = self.build_graph_source(namespace)
            if gsrc["stats"]["n_relations"] > 0:
                sources = sources + [{k: gsrc[k] for k in ("display_name", "text_content")}]
        if not sources:
            return {"exported": 0, "notebook_id": notebook_id, "note": "no records"}
        exported = 0
        for i in range(0, len(sources), max(1, batch_size)):
            self.backend.batch_create_sources(notebook_id, sources[i:i + batch_size])
            exported += len(sources[i:i + batch_size])
        return {"exported": exported, "notebook_id": notebook_id,
                "namespace": namespace,
                "note": "sources carry eidetic provenance headers; NotebookLM answers over "
                        "them are Gemini-side and NOT eidetic-verified -- use eidetic recall "
                        "for cited answers."}

    def query(self, notebook_id: str, question: str) -> dict:
        res = self.backend.query(notebook_id, question)
        res.setdefault("caveat", "NotebookLM/Gemini answer -- NOT verified by eidetic's "
                                 "proof gate. For a cited, verify-or-abstain answer use "
                                 "eidetic.recall().")
        return res

    def _resolve_provenance(self, namespace: str, text: str) -> list[dict]:
        """Map NotebookLM answer/citation references (the `eidetic:<id>` tokens we stamped
        into each source's display_name + provenance header) back to the immutable records,
        so even a Gemini-side answer carries eidetic's content-hash provenance."""
        ids = {m.group(1) for m in _EIDETIC_REF_RE.finditer(text or "")}
        if not ids:
            return []
        scope = Scope(namespace=namespace)
        out = []
        # all_records (NOT active_records_at): the graph legend hoists BOTH active and
        # superseded record hashes, so the resolver must read the same record set or a
        # Gemini answer citing a history token would dangle. Widening is strictly more
        # permissive -- it maps a token to an immutable record regardless of validity.
        # EXACT match only: the tokens the serializer emits are always the full memory_id or
        # memory_id[:16] (16 chars, unique). An unanchored startswith on a token as short as 4
        # chars (the _EIDETIC_REF_RE floor) would match EVERY record sharing that prefix, so a
        # truncated/hallucinated Gemini citation like `eidetic:mem_` could be misattributed to
        # many (now also superseded) records with the wrong content hashes.
        for rec in self.engine.store.all_records(scope):
            short = rec.memory_id[:16]
            if rec.memory_id in ids or short in ids:
                out.append({"memory_id": rec.memory_id,
                            "content_sha256": rec.content_hash,
                            "valid_at": _iso(rec.valid_at)})
        return out

    def answer(self, namespace: str, question: str, notebook_id: str) -> dict:
        """NotebookLM READER MODE -- the token-efficient product path. NotebookLM answers
        over eidetic's exported VERIFIED sources using Gemini's free tier, so the answer
        costs ~0 of the caller's own LLM tokens (Google eats the compute). We then map the
        answer's `eidetic:<id>` references back to immutable content hashes, so the free
        answer still carries provenance.

        Returns {answer, provenance, user_llm_tokens, backend, caveat}. Honest labels:
        `user_llm_tokens: 0` means ZERO tokens on the CALLER's metered model -- it does NOT
        mean the compute was free globally, and the answer is Gemini-side, NOT run through
        eidetic's verify-or-abstain proof gate. Use `engine.ask(verify=True)` (the MCP
        `recall` tool) when you need a cited, gate-verified answer instead of a free one."""
        res = self.backend.query(notebook_id, question)
        answer_text = res.get("answer") or res.get("text") or json.dumps(res)
        provenance = self._resolve_provenance(namespace, answer_text
                                              + " " + json.dumps(res.get("citations", "")))
        return {
            "answer": answer_text,
            "provenance": provenance,
            "user_llm_tokens": 0,
            "backend": res.get("backend", "notebooklm"),
            "caveat": ("0 tokens on YOUR metered model (NotebookLM/Gemini free tier does the "
                       "read); answer is Gemini-side and NOT eidetic-verify-or-abstain. "
                       "Provenance below maps it back to immutable content hashes."),
        }

    # ---- graph-native serializer wiring -------------------------------------
    def build_graph_source(self, namespace: str, *, at: Optional[float] = None,
                           include_history: bool = True, max_entities: Optional[int] = None,
                           include_inferred: bool = False) -> dict:
        """Build (offline; works with backend=None) the ONE compact verified-claim-graph
        source for `namespace`. Hash-join uses all_records (NOT active_records_at): a
        superseded history edge can point to a later-invalidated record, so all_records is
        the only source that keeps every history token's sha256 resolvable."""
        scope = Scope(namespace=namespace)
        edges = self.engine.store.all_edges(scope, include_inferred=include_inferred)
        records = self.engine.store.all_records(scope)
        records_by_id = {r.memory_id: r for r in records}
        try:
            nf = self.engine.graph.node_features(at, scope)
        except Exception:
            nf = None
        return format_graph_source(
            edges, records_by_id, scope_label=namespace, at=at, node_features=nf,
            include_history=include_history, max_entities=max_entities,
            include_inferred=include_inferred,
        )

    def export_graph(self, namespace: str, notebook_id: str, *, at: Optional[float] = None,
                     include_history: bool = True, max_entities: Optional[int] = None) -> dict:
        """Push the verified claim graph as one ADDITIONAL compact provenance source."""
        src = self.build_graph_source(namespace, at=at, include_history=include_history,
                                      max_entities=max_entities)
        if src["stats"]["n_relations"] == 0:
            return {"exported": 0, "notebook_id": notebook_id, "note": "empty graph"}
        self.backend.batch_create_sources(
            notebook_id, [{k: src[k] for k in ("display_name", "text_content")}])
        return {
            "exported": 1, "notebook_id": notebook_id, "namespace": namespace,
            "stats": src["stats"],
            "note": ("compact VERIFIED claim graph source (provenance-carrying); a "
                     "NotebookLM/Gemini answer over it is Gemini-side and NOT "
                     "eidetic-verify-or-abstain -- NOT a row in the fixed-qwen-reader "
                     "benchmark table. No SOTA claim."),
        }

    # ---- router-aware answer path -------------------------------------------
    # Per-tier caller-token costs are DESIGN-SUPPLIED constants labeled as such; the P_*
    # hit-rate weights that would blend them are UNMEASURED (to be measured on a dev split).
    # We ship the FORMULA, never a specific blended figure.
    _COST_STRUCTURED = 45   # design-supplied midpoint of the ~6-85 structured band
    _COST_METERED = 4034    # design-supplied metered verified-reader cost

    def routed_answer(self, namespace: str, question: str, notebook_id: str, *,
                      require_gate_verification: bool = False,
                      struct_tau: float = 0.0) -> dict:
        """Route one question across four tiers, cheapest-verified first.

        Tier 0 reflex pre-filter (0 caller tokens, no model call) -> candidate cross-check.
        Tier 1 structured, cheap + gate-verified (answered+verified+immutable_proof and
                confidence>=struct_tau).
        Tier 2 free NotebookLM read (0 caller tokens) when NOT struct-ok AND not
                require_gate_verification -- Gemini-side, provenance-mapped, NOT gate-verified.
        Tier 3 metered verified reader (engine.ask verify=True) when NOT struct-ok AND
                require_gate_verification -- the ONLY path that runs the proof gate on a
                GENERATED answer.

        NOTE the asymmetry fix: Tier 3 gates on `not struct_ok`, never on 'abstained', so a
        verified-but-low-confidence answer under require_gate_verification escalates instead
        of falling through."""
        scope = Scope(namespace=namespace)

        # Tier 0 -- reflex pre-filter (emits no answer).
        packet = self.engine.reflex_recall(question, scope=scope)
        reflex_ids = list(packet.candidate_ids())

        def _honesty() -> dict:
            return dict(_HONESTY_BOUNDARIES)

        # Tier 1 -- structured, verify-or-abstain (itself gate-verified when it answers).
        s = self.engine.structured_recall(question, scope=scope)
        struct_ok = bool(
            s.get("answered") and s.get("verified") and s.get("immutable_proof")
            and float(s.get("confidence", 0.0)) >= struct_tau
        )
        if struct_ok:
            prov = []
            for cit in (s.get("citations") or []):
                if isinstance(cit, dict) and cit.get("memory_id"):
                    prov.append({"memory_id": cit.get("memory_id"),
                                 "content_sha256": cit.get("content_hash", ""),
                                 "valid_at": cit.get("valid_at", "")})
            return {
                "tier": 1,
                "answer": s.get("answer", ""),
                "provenance": prov,
                "caller_llm_tokens": self._COST_STRUCTURED,
                "gate_verified": True,
                "provenance_verb": "gate-verified",
                "reflex_cross_check": {
                    "candidate_ids": reflex_ids,
                    "intersection": [p["memory_id"] for p in prov if p["memory_id"] in reflex_ids],
                },
                "honesty": _honesty(),
            }

        # Tier 3 -- metered verified reader (proof gate on a generated answer).
        # Engine has NO `recall` method: its verify-or-abstain path is `ask()` (which runs the
        # NLI proof gate when verify=True) and its proof-tree method is `prove()`. We mirror
        # mcp_server.recall: ask(verify=True) -> Answer.model_dump(); citations are Citation
        # objects whose model_dump carries memory_id / content_hash / valid_at.
        if require_gate_verification:
            ans = self.engine.ask(question, verify=True, scope=scope)
            res = ans.model_dump()
            prov = []
            for cit in (res.get("citations") or []):
                if isinstance(cit, dict) and cit.get("memory_id"):
                    prov.append({"memory_id": cit.get("memory_id"),
                                 "content_sha256": cit.get("content_hash", ""),
                                 "valid_at": cit.get("valid_at", "")})
            return {
                "tier": 3,
                "answer": res.get("answer", ""),
                "provenance": prov,
                "caller_llm_tokens": self._COST_METERED,
                "gate_verified": True,
                "provenance_verb": "gate-verified",
                "reflex_cross_check": {
                    "candidate_ids": reflex_ids,
                    "intersection": [p["memory_id"] for p in prov if p["memory_id"] in reflex_ids],
                },
                "honesty": _honesty(),
            }

        # Tier 2 -- free NotebookLM read (0 caller tokens), provenance-mapped only.
        r = self.answer(namespace, question, notebook_id)
        resolved_ids = [p["memory_id"] for p in r.get("provenance", [])]
        return {
            "tier": 2,
            "answer": r.get("answer", ""),
            "provenance": r.get("provenance", []),
            "caller_llm_tokens": 0,
            "gate_verified": False,
            "provenance_verb": "provenance-mapped",
            "reflex_cross_check": {
                "candidate_ids": reflex_ids,
                "intersection": [i for i in resolved_ids if i in reflex_ids],
            },
            "honesty": _honesty(),
        }


class IncrementalSync:
    """Content-hash-diffed push of a namespace into a NotebookLM notebook. Dedupe key is
    record.content_hash (already stamped into each source header, so a diff needs no
    read-back from NotebookLM). The manifest is a per-(namespace, notebook_id) sidecar JSON
    of pushed hashes.

    Supersession policy = APPEND-ONLY (matches eidetic's write-once ethos): a changed fact
    is a NEW content_hash pushed as a NEW source; the superseded record's header already
    carries invalidated_at. TRADEOFF: NotebookLM source count grows monotonically (a ceiling
    concern at scale); pruning is a separate opt-in policy, deliberately NOT the default."""

    def __init__(self, bridge: "NotebookLMBridge", manifest_path: str) -> None:
        self.bridge = bridge
        self.manifest_path = manifest_path

    def _load(self) -> dict:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)

    def sync(self, namespace: str, notebook_id: str) -> dict:
        key = f"{namespace}::{notebook_id}"
        manifest = self._load()
        pushed_hashes = set(manifest.get(key, []))
        scope = Scope(namespace=namespace)
        records = self.bridge.engine.store.active_records_at(None, scope)
        records = [r for r in records if (r.text or r.summary)]
        to_push, skipped = [], 0
        new_hashes = []
        for rec in records:
            if rec.content_hash in pushed_hashes:
                skipped += 1
                continue
            try:
                claims = self.bridge.engine.store.claims_by_source(rec.memory_id)
            except Exception:
                claims = []
            to_push.append(format_source(rec, claims))
            new_hashes.append(rec.content_hash)
        if to_push:
            self.bridge.backend.batch_create_sources(notebook_id, to_push)
            pushed_hashes.update(new_hashes)
            manifest[key] = sorted(pushed_hashes)
            self._save(manifest)
        return {
            "pushed": len(to_push),
            "skipped": skipped,
            "total": len(records),
            "note": ("APPEND-ONLY: changed facts push as NEW content_hash sources; NotebookLM "
                     "source count grows monotonically (pruning is opt-in, not default). "
                     "Sources are Gemini-side reads, NOT eidetic-verify-or-abstain."),
        }


def _cli() -> int:  # pragma: no cover - thin argparse wrapper
    import argparse
    from eidetic.config import get_settings
    from eidetic.engine import Engine

    ap = argparse.ArgumentParser(description="Export eidetic verified memory into NotebookLM")
    ap.add_argument("action", choices=["export", "query", "preview", "preview-graph",
                                       "export-graph", "sync", "routed-answer", "doctor"])
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--notebook-id", default=os.environ.get("NOTEBOOKLM_NOTEBOOK_ID", ""))
    ap.add_argument("--backend", choices=["enterprise", "cli"], default="enterprise")
    ap.add_argument("--project-number", default=os.environ.get("NOTEBOOKLM_PROJECT_NUMBER", ""))
    ap.add_argument("--location", default=os.environ.get("NOTEBOOKLM_LOCATION", "global"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--question", default="")
    ap.add_argument("--at", type=float, default=None)
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--max-entities", type=int, default=None)
    ap.add_argument("--require-gate", action="store_true")
    ap.add_argument("--manifest", default=os.environ.get("NOTEBOOKLM_SYNC_MANIFEST",
                                                         "notebooklm_sync_manifest.json"))
    args = ap.parse_args()

    eng = Engine(get_settings())
    if args.action == "preview":
        bridge = NotebookLMBridge(eng, backend=None)
        srcs = bridge.build_sources(args.namespace, args.limit)
        print(json.dumps({"sources": len(srcs),
                          "first": srcs[0] if srcs else None}, indent=2))
        return 0
    if args.action == "preview-graph":
        bridge = NotebookLMBridge(eng, backend=None)
        src = bridge.build_graph_source(args.namespace, at=args.at,
                                        include_history=not args.no_history,
                                        max_entities=args.max_entities)
        preview = "\n".join(src["text_content"].splitlines()[:40])
        print(json.dumps({"stats": src["stats"], "display_name": src["display_name"],
                          "text_preview": preview}, indent=2))
        return 0
    if args.action == "doctor":
        # Preflight: never touches eidetic data; just reports backend reachability + the
        # EXACT commands/endpoints the tool will run, so the first live attempt isn't a guess.
        try:
            backend = (EnterpriseBackend(project_number=args.project_number, location=args.location)
                       if args.backend == "enterprise" else CliBackend(runner=None))
            print(json.dumps(backend.doctor(), indent=2))
        except NotebookLMError as exc:
            print(json.dumps({"backend": args.backend, "reachable": False,
                              "error": str(exc)}, indent=2))
        return 0
    backend = (EnterpriseBackend(project_number=args.project_number, location=args.location)
               if args.backend == "enterprise" else CliBackend())
    bridge = NotebookLMBridge(eng, backend)
    if args.action == "export":
        print(json.dumps(bridge.export_namespace(
            args.namespace, args.notebook_id, limit=args.limit), indent=2))
    elif args.action == "export-graph":
        print(json.dumps(bridge.export_graph(
            args.namespace, args.notebook_id, at=args.at,
            include_history=not args.no_history, max_entities=args.max_entities), indent=2))
    elif args.action == "sync":
        print(json.dumps(IncrementalSync(bridge, args.manifest).sync(
            args.namespace, args.notebook_id), indent=2))
    elif args.action == "routed-answer":
        print(json.dumps(bridge.routed_answer(
            args.namespace, args.question, args.notebook_id,
            require_gate_verification=args.require_gate), indent=2))
    else:
        print(json.dumps(bridge.query(args.notebook_id, args.question), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
