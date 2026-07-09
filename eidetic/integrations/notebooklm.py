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
from pathlib import Path
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


def find_notebook_id(raw_json: str, title: Optional[str] = None, *,
                     strict: bool = False) -> Optional[str]:
    """Recursively pull a NotebookLM notebook id out of whatever JSON shape `nlm notebook
    list/create --json` emits (array, {notebooks:[...]}, {notebook_id:...}, nested {data:{
    items:[...]}}). Prefers the id whose object's title/name matches `title`; else the first
    id found. Returns None if unparseable or absent -- callers never crash on it.

    strict=True: return ONLY an exact-title match (no first-found fallback). REQUIRED for
    per-namespace isolation flows -- the fallback once resolved every missing-title lookup
    to the same first notebook, silently mixing 10 namespaces' sources and questions into
    one notebook. The deterministic grounding check (unmatched foreign quotes) is what
    exposed it."""
    try:
        data = json.loads(raw_json)
    except Exception:
        return None
    id_keys = ("id", "notebook_id", "notebookId")
    found: list[tuple[str, str]] = []

    def walk(o):
        if isinstance(o, dict):
            nid = next((str(o[k]) for k in id_keys if o.get(k)), None)
            if nid and _re.search(r"[A-Za-z0-9_-]{8,}", nid):
                found.append((nid, str(o.get("title") or o.get("name") or "")))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    if title:
        for nid, t in found:
            if t.strip() == title.strip():
                return nid
    if strict:
        return None
    return found[0][0] if found else None


def parse_nlm_query_output(raw: str) -> dict:
    """Normalise `nlm notebook query --json` output. Live shape (v0.8.x): a JSON object
    whose `answer` field is ITSELF a JSON string carrying the clean answer + `references`
    (each with `cited_text` that still contains the eidetic:<id> tokens we stamped). Returns
    {answer, references, cited_text} -- answer is clean human text, cited_text concatenates
    every reference so provenance resolution sees the citations. Falls back to raw text."""
    try:
        d = json.loads(raw)
    except Exception:
        return {"answer": (raw or "").strip(), "references": [], "cited_text": raw or ""}
    if not isinstance(d, dict):
        return {"answer": str(d), "references": [], "cited_text": raw or ""}
    ans = d.get("answer")
    refs = d.get("references") or []
    if isinstance(ans, str) and ans.lstrip().startswith("{"):
        try:  # answer double-encoded as JSON
            inner = json.loads(ans)
            if isinstance(inner, dict):
                ans = inner.get("answer", ans)
                refs = inner.get("references") or refs
        except Exception:
            pass
    cited = " ".join(str(r.get("cited_text", "")) for r in refs if isinstance(r, dict))
    return {"answer": ans if isinstance(ans, str) else (raw or "").strip(),
            "references": refs, "cited_text": cited + " " + (raw or "")}


def _resolve_nlm() -> Optional[str]:
    """Locate the `nlm` binary. Order: $NLM_BIN, PATH, then this repo's own .venv/bin/nlm
    (so `.venv/bin/pip install notebooklm-mcp-cli` works without activating the venv or
    polluting a Homebrew-managed system Python)."""
    env = os.environ.get("NLM_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("nlm")
    if found:
        return found
    venv_nlm = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "nlm"
    return str(venv_nlm) if venv_nlm.exists() else None


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
    "qwen_memory_check": (
        "qwen (the model that BUILT these sources) ran post-hoc, per-claim NLI of Gemini's "
        "free answer against ONLY the exported qwen sources Gemini CITED. A self-consistency "
        "/ faithfulness AUDIT that FLAGS where the draft diverges from or is unsupported by "
        "the cited memory. NOT a correctness guarantee, NOT independent verification, NOT "
        "eidetic's verify-or-abstain generation gate; the answer is returned regardless. "
        "Premises = cited memories only, so a correct claim whose block Gemini did not cite "
        "can be flagged unsupported. On the current self-sourced benchmark it demonstrates 0 "
        "catches; value is prospective / out-of-distribution (non-self-sourced reads) only. "
        "NLI is metered qwen-plus (NOT 0 tokens); it is reported separately from generation."
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
        # Construction NEVER fails on a missing binary -- so `doctor` can still report the
        # install hint. The runner is wired only if nlm is present; operations raise otherwise.
        if self.runner is None:
            nlm = _resolve_nlm()
            if nlm is not None:
                self.runner = lambda args: subprocess.run(
                    [nlm, *args], capture_output=True, text=True, check=True).stdout

    def _require_runner(self):
        if self.runner is None:
            raise NotebookLMError(
                "`nlm` not found. Install the PyPI package 'notebooklm-mcp-cli' "
                "(it provides the `nlm` command): into this repo's venv with "
                "`.venv/bin/pip install notebooklm-mcp-cli`, or globally with "
                "`pipx install notebooklm-mcp-cli`. Set NLM_BIN to override the path.")
        return self.runner

    def batch_create_sources(self, notebook_id: str, sources: list[dict]) -> dict:
        # Real nlm syntax (notebooklm-mcp-cli): `nlm source add <notebook> --text "..."`.
        # notebook is POSITIONAL (no --notebook flag); text sources take no name flag, so
        # the provenance rides inside the text body (the header), not a display-name arg.
        self._require_runner()
        created = 0
        for s in sources:
            self.runner(["source", "add", notebook_id, "--text", s["text_content"]])
            created += 1
        return {"created": created}

    def query(self, notebook_id: str, question: str) -> dict:
        # Real nlm syntax: `nlm notebook query <notebook> "question" --json`. --json gives a
        # structured payload (answer + references w/ cited_text) instead of a wall of prose,
        # so we return a CLEAN answer + the citations (which still carry eidetic:<id> tokens).
        self._require_runner()
        out = self.runner(["notebook", "query", notebook_id, question, "--json"])
        parsed = parse_nlm_query_output(out)
        return {"answer": parsed["answer"], "references": parsed["references"],
                "cited_text": parsed["cited_text"],
                "backend": "nlm-cli (gemini-side, UNVERIFIED)"}

    def doctor(self) -> dict:
        """Preflight: is `nlm` reachable + logged in, and what commands will we run? Never
        raises -- returns a status dict so the user sees the plan before a live export."""
        nlm_path = _resolve_nlm()
        # A wired runner (real binary OR an injected test double) means we can probe login;
        # only when there's no runner AND no binary do we report the install hint.
        reachable = self.runner is not None or nlm_path is not None
        status = {"backend": "cli", "nlm_path": nlm_path, "reachable": reachable,
                  "logged_in": None,
                  "install_hint": None if reachable else
                      ".venv/bin/pip install notebooklm-mcp-cli   (or: pipx install notebooklm-mcp-cli)",
                  "commands": {
                      "add_source": "nlm source add <notebook> --text \"<provenance+body>\"",
                      "query": "nlm notebook query <notebook> \"<question>\"",
                      "login": "nlm login   (check: nlm login --check)"}}
        if not reachable:
            status["logged_in"] = "n/a -- install notebooklm-mcp-cli first"
            return status
        try:
            out = self._require_runner()(["login", "--check"])
            status["logged_in"] = ("logged in" if "not" not in (out or "").lower()
                                   else "NOT logged in (run `nlm login`)")
        except Exception as exc:
            status["logged_in"] = f"NOT logged in / error ({type(exc).__name__}); run `nlm login`"
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

    # ---- deterministic grounding check (no model calls, no caller tokens) ----
    _GROUND_TOKEN_RE = _re.compile(r"[a-z0-9][a-z0-9'-]{2,}")
    _GROUND_STOP = frozenset({
        "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
        "was", "were", "are", "been", "being", "not", "but", "you", "your",
    })
    _QUOTE_MIN_CHARS = 12          # below this a "quote" is too short to mean anything
    _QUOTE_OVERLAP_FLOOR = 0.8     # token-overlap floor for a non-verbatim quote match

    def _exported_corpus(self, namespace: str) -> str:
        """Deterministically rebuild every byte this bridge would export for `namespace`
        (graph source + per-record provenance sources), whitespace-normalized + lowercased.
        The grounding check compares NotebookLM's quoted references and answer tokens
        against THIS -- the caller's own immutable store -- so a fabricated or altered
        quote cannot match. Read-only, zero model calls."""
        parts: list[str] = []
        try:
            parts.append(self.build_graph_source(namespace)["text_content"])
        except Exception:
            pass  # no graph surface on this store (fine: records below still ground quotes)
        try:
            parts.extend(s["text_content"] for s in self.build_sources(namespace))
        except Exception:
            pass
        return _re.sub(r"\s+", " ", "\n".join(parts).lower()).strip()

    def verify_grounding(self, namespace: str, answer_text: str,
                         references: list) -> dict:
        """Deterministic grounding report for a NotebookLM answer. Two checks, both against
        the exported-source bytes rebuilt from the immutable store:

        1. QUOTE FAITHFULNESS -- each reference's `cited_text` must appear in the exported
           text (whitespace-normalized substring => "verbatim"; else content-token overlap
           >= 0.8 => "high-overlap"; else "unmatched" -- NotebookLM altered or fabricated
           the quote).
        2. ANSWER TOKEN COVERAGE -- fraction of the answer's content tokens present in the
           exported text (a low number flags Gemini-side additions beyond your memories).

        HONEST LABEL: this is a deterministic lexical check -- NOT NLI entailment and NOT
        eidetic's verify-or-abstain proof gate. It can catch fabricated quotes and alien
        answer content; it cannot certify the reasoning is correct."""
        corpus = self._exported_corpus(namespace)
        corpus_tokens = set(self._GROUND_TOKEN_RE.findall(corpus))
        verdicts: list[str] = []
        for ref in references or []:
            if not isinstance(ref, dict):
                continue
            quote = _EIDETIC_REF_RE.sub("", str(ref.get("cited_text", "")))
            # the ref tokens ride inside [brackets] in cited_text; removing the token leaves
            # empty bracket husks that would break an otherwise-verbatim substring match
            quote = _re.sub(r"[\[\]()]+", " ", quote)
            qn = _re.sub(r"\s+", " ", quote.lower()).strip()
            if len(qn) < self._QUOTE_MIN_CHARS:
                verdicts.append("too-short")
                continue
            if qn in corpus:
                verdicts.append("verbatim")
                continue
            toks = {t for t in self._GROUND_TOKEN_RE.findall(qn)
                    if t not in self._GROUND_STOP}
            overlap = (sum(1 for t in toks if t in corpus_tokens) / len(toks)) if toks else 0.0
            verdicts.append("high-overlap" if overlap >= self._QUOTE_OVERLAP_FLOOR
                            else "unmatched")
        ans_toks = {t for t in self._GROUND_TOKEN_RE.findall((answer_text or "").lower())
                    if t not in self._GROUND_STOP}
        coverage = (round(sum(1 for t in ans_toks if t in corpus_tokens) / len(ans_toks), 3)
                    if ans_toks else None)
        return {
            "references_checked": len(verdicts),
            "quotes_verbatim": verdicts.count("verbatim"),
            "quotes_high_overlap": verdicts.count("high-overlap"),
            "quotes_unmatched": verdicts.count("unmatched"),
            "quotes_too_short": verdicts.count("too-short"),
            "answer_token_coverage": coverage,
            "method": ("deterministic lexical check vs the exported source text rebuilt "
                       "from the immutable store (normalized substring + token overlap). "
                       "NOT NLI, NOT eidetic's proof gate: catches fabricated/altered "
                       "quotes and alien answer content, does not certify reasoning. "
                       "Gemini's connective prose lowers answer_token_coverage -- read a "
                       "low number as a flag to inspect, not a fabrication verdict; "
                       "quotes_unmatched>0 is the strong signal."),
        }

    # ---- qwen post-hoc faithfulness AUDIT of the Gemini free answer -----------
    # split on sentence enders / newlines / semicolons, but NOT after common abbreviations
    # (Dr. Mr. Mrs. Ms. St. etc.) which would strip the claim's subject/object into sub-min
    # fragments that never reach the audit.
    _ABBREV = r"(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bvs)(?<!\bNo)"
    _CLAIM_SPLIT_RE = _re.compile(_ABBREV + r"(?<=[.!?])\s+|\n+|;\s+")

    def qwen_memory_check(self, namespace: str, question: str, answer_text: str,
                          cited_memory_ids: list[str]) -> dict:
        """Post-hoc, NON-BLOCKING, per-claim NLI of Gemini's free answer against ONLY the
        qwen-built sources Gemini CITED. This is an AUDITOR, not an arbiter: the answer was
        already emitted; this observes whether it stays faithful to the cited memory. qwen
        (verify_model) is the model that built those sources, so it checks self-consistency
        against its own prior claims. Scheme (per the red-team spec): decompose -> substantive
        filter -> per claim {Tier0 free lexical, Tier1 per-block NLI early-stop, Tier2 union
        rescue}; gate on ENTAILMENT; contradiction hard-fails. NLI is metered qwen (NOT 0)."""
        from eidetic.smqe.verify import reader_answer_form_credible
        from eidetic.models import NLILabel

        r = self.engine.retriever
        s = self.engine.settings
        # premises = ONLY cited records' bodies (honest scope: cited_only)
        blocks: list[tuple[str, str]] = []   # (memory_id, text)
        for mid in cited_memory_ids:
            rec = self.engine.store.get_record(mid)
            body = (rec.text or rec.summary or "").strip() if rec else ""
            if body:
                blocks.append((mid, body))
        min_chars = getattr(s, "span_nli_min_chars", 12)
        # decompose + substantive filter (a chatty-but-correct answer false-negatives without it)
        claims = [c.strip() for c in self._CLAIM_SPLIT_RE.split(answer_text or "") if c.strip()]
        claims = [c for c in claims
                  if len(c) >= min_chars and reader_answer_form_credible(question, c)]

        def _lexical_entails(claim: str) -> bool:  # Tier 0, free -- POSITIVE claims only
            # A negation cue makes lexical overlap UNSOUND (it cannot tell "Bob is the
            # manager" from "Bob is NOT the manager"), so a negated claim MUST fall through
            # to NLI where a contradiction can be caught. (Bug: Tier0 was masking contradictions.)
            if _re.search(r"\b(?:not|no|never|none|n't|without|neither|nor|isn't|aren't|"
                          r"wasn't|weren't|didn't|doesn't|don't|can't|won't)\b", claim.lower()):
                return False
            toks = {t for t in _re.findall(r"[a-z0-9][a-z0-9'-]{2,}", claim.lower())
                    if t not in self._GROUND_STOP}
            if not toks:
                return False
            for _mid, body in blocks:
                # WORD-BOUNDARY membership (not substring: 'art' must not match 'apart')
                btoks = set(_re.findall(r"[a-z0-9][a-z0-9'-]{2,}", body.lower()))
                if sum(1 for t in toks if t in btoks) / len(toks) >= 0.9:
                    return True
            return False

        per_claim = []
        nli_calls = 0
        concat = "\n\n".join(b for _m, b in blocks)[:6000]
        for claim in claims:
            verdict, tier, by_mid, conf = "neutral", None, None, 0.0
            if _lexical_entails(claim):
                verdict, tier = "entailed", 0
            else:
                # Tier 1: scan ALL cited blocks -- contradiction HARD-FAILS and dominates
                # entailment (per the docstring), so we must not early-stop on entailment and
                # let a later contradicting block go unseen.
                entail_mid = entail_conf = None
                contra_mid = contra_conf = None
                for mid, body in blocks:
                    label, c = r.verify(body, claim)
                    nli_calls += 1
                    if label == NLILabel.CONTRADICTION and contra_mid is None:
                        contra_mid, contra_conf = mid, c
                    elif label == NLILabel.ENTAILMENT and entail_mid is None:
                        entail_mid, entail_conf = mid, c
                if contra_mid is not None:                     # contradiction dominates
                    verdict, by_mid, conf = "contradicted", contra_mid, contra_conf
                elif entail_mid is not None:
                    verdict, tier, by_mid, conf = "entailed", 1, entail_mid, entail_conf
                elif concat:                                   # Tier 2: union rescue per-claim
                    label, c = r.verify(concat, claim)
                    nli_calls += 1
                    if label == NLILabel.ENTAILMENT:
                        verdict, tier, conf = "entailed", 2, c
                    elif label == NLILabel.CONTRADICTION:
                        verdict, conf = "contradicted", c
            per_claim.append({"claim": claim[:200], "verdict": verdict, "tier": tier,
                              "by_memory_id": by_mid, "confidence": round(float(conf), 3)})

        entailed = sum(1 for p in per_claim if p["verdict"] == "entailed")
        neutral = sum(1 for p in per_claim if p["verdict"] == "neutral")
        contradicted = sum(1 for p in per_claim if p["verdict"] == "contradicted")
        if not per_claim:
            decision = "no_checkable_claims"    # honest: nothing substantive to audit (was
            # silently reported "consistent" -- an abstention/pleasantry is not "consistent")
        elif contradicted:
            decision = "diverges_from_cited_memory"
        elif neutral:
            decision = "unsupported_by_cited_memory"
        else:
            decision = "consistent_with_cited_memory"
        return {
            "decision": decision,
            "claims_total": len(per_claim),
            "entailed": entailed, "neutral": neutral, "contradicted": contradicted,
            "per_claim": per_claim,
            "premise_scope": "cited_only",
            "cost": {
                "qwen_nli_calls": nli_calls,
                "nli_input_tokens_est": (sum(len(b) for _m, b in blocks) // 4) * max(1, len(claims)),
                "note": "NLI, not generation, not a gate; metered qwen-plus, NOT 0 tokens. "
                        "Some calls may be verify() LRU cache hits (cost 0).",
            },
            "note": _HONESTY_BOUNDARIES["qwen_memory_check"],
        }

    def answer(self, namespace: str, question: str, notebook_id: str,
               verify_with_qwen: bool = False) -> dict:
        """NotebookLM READER MODE -- the token-efficient product path. NotebookLM answers
        over eidetic's exported VERIFIED sources using Gemini's free tier, so the answer
        costs ~0 of the caller's own LLM tokens (Google eats the compute). We then map the
        answer's `eidetic:<id>` references back to immutable content hashes, so the free
        answer still carries provenance.

        Returns {answer, provenance, user_llm_tokens, backend, caveat}. Honest labels:
        `user_llm_tokens: 0` means ZERO tokens on the CALLER's metered model -- it does NOT
        mean the compute was free globally, and the answer is Gemini-side, NOT run through
        eidetic's verify-or-abstain proof gate. Use `engine.ask(verify=True)` (the MCP
        `recall` tool) when you need a cited, gate-verified answer instead of a free one.

        verify_with_qwen (default False, opt-in): after Gemini answers, qwen (the model that
        BUILT the sources) runs a post-hoc, NON-BLOCKING per-claim faithfulness AUDIT of the
        draft against the cited memory (adds `qwen_memory_check`). It does NOT change the
        answer and does NOT make it gate-verified. `user_llm_tokens` stays 0 (generation is
        still free); the NLI cost is metered qwen, reported ONLY inside `qwen_memory_check.cost`."""
        res = self.backend.query(notebook_id, question)
        answer_text = res.get("answer") or res.get("text") or json.dumps(res)
        # cited_text is where the eidetic:<id> tokens live (references' cited_text); fall back
        # to the answer + any citations blob for backends that don't split them out.
        cited_text = res.get("cited_text") or (
            answer_text + " " + json.dumps(res.get("citations", "")))
        cited_ids = {m.group(1) for m in _EIDETIC_REF_RE.finditer(cited_text)}
        provenance = self._resolve_provenance(namespace, cited_text)
        # SOURCE VERIFICATION: which of the answer's cited eidetic tokens map to a REAL,
        # content-hash-addressable record in this store? Confirms the Gemini answer cited
        # genuine eidetic memories (not hallucinated ones) -- provenance you can check, even
        # though the Gemini REASONING itself is not gate-verified.
        confirmed = {p["memory_id"] for p in provenance} | {p["memory_id"][:16] for p in provenance}
        n_confirmed = len([i for i in cited_ids if i in confirmed])
        # GROUNDING: deterministic quote-faithfulness + answer-token-coverage vs the exported
        # bytes rebuilt from the immutable store. Free (no model call), catches fabricated or
        # altered quotes; labeled honestly as lexical, not NLI/gate.
        grounding = self.verify_grounding(namespace, answer_text,
                                          res.get("references") or [])
        # OPT-IN qwen faithfulness AUDIT (default off). Non-blocking, post-hoc: the free
        # answer above is returned regardless. Its NLI cost is reported inside the block,
        # NEVER folded into user_llm_tokens (which stays 0 -- generation only).
        qwen_check = None
        if verify_with_qwen:
            cited_mids = [p["memory_id"] for p in provenance]
            qwen_check = self.qwen_memory_check(namespace, question, answer_text, cited_mids)
        return {
            "answer": answer_text,
            "provenance": provenance,
            "cited_sources": {
                "cited": len(cited_ids),
                "confirmed_in_eidetic": n_confirmed,
                "note": "each confirmed citation resolves to a real eidetic memory by content "
                        "hash; unconfirmed ones are Gemini's and are NOT backed by your store."},
            "grounding": grounding,
            **({"qwen_memory_check": qwen_check} if qwen_check is not None else {}),
            # trimmed raw references: lets any caller inspect exactly WHAT NotebookLM
            # quoted when a grounding verdict says unmatched (diagnosability -- the
            # contamination incident was only debuggable via a live re-ask without these)
            "references": [str(r.get("cited_text", ""))[:400]
                           for r in (res.get("references") or [])[:12]
                           if isinstance(r, dict)],
            "user_llm_tokens": 0,
            "backend": res.get("backend", "notebooklm"),
            "caveat": ("0 tokens on YOUR metered model (NotebookLM/Gemini free tier does the "
                       "read); answer is Gemini-side and NOT eidetic-verify-or-abstain. "
                       "Provenance + cited_sources below let you CHECK which sources are real "
                       "eidetic memories, even though the reasoning is not gate-verified."),
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
            "cited_sources": r.get("cited_sources", {}),
            "grounding": r.get("grounding", {}),
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
    ap.add_argument("action", choices=["export", "query", "answer", "preview", "preview-graph",
                                       "export-graph", "sync", "routed-answer", "doctor",
                                       "seed", "find-notebook-id"])
    ap.add_argument("--title", default="")
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
    ap.add_argument("--verify-with-qwen", action="store_true",
                    help="opt-in: qwen post-hoc per-claim faithfulness AUDIT of the free "
                         "Gemini answer against cited memory (non-blocking; metered NLI, "
                         "reported separately; user_llm_tokens stays 0)")
    ap.add_argument("--manifest", default=os.environ.get("NOTEBOOKLM_SYNC_MANIFEST",
                                                         "notebooklm_sync_manifest.json"))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", ""),
                    help="eidetic store to read (default $DATA_DIR). Point at your live "
                         "store, e.g. ~/.eidetic-plus/data, or the CLI sees an empty store.")
    args = ap.parse_args()

    if args.action == "find-notebook-id":
        # Read `nlm notebook list/create --json` from stdin, print the id (or nothing).
        import sys as _sys
        nid = find_notebook_id(_sys.stdin.read(), args.title or None)
        if nid:
            print(nid)
        return 0
    if args.data_dir:
        os.environ["DATA_DIR"] = os.path.expanduser(args.data_dir)
    eng = Engine(get_settings())
    if args.action == "seed":
        # Populate a namespace with sample linked facts (real ingest+consolidate -> graph
        # edges) so the NotebookLM loop can be tested end-to-end without pre-existing memories.
        # Uses your DASHSCOPE_API_KEY (the same write path the benchmark uses).
        from eidetic.models import Scope as _Scope
        sc = _Scope(namespace=args.namespace)
        facts = [
            "I moved to Berlin in March 2021 for a job at Acme Robotics.",
            "At Acme Robotics I lead the perception team building lidar pipelines.",
            "I adopted a beagle named Biscuit in July 2023.",
            "In 2024 I switched from Acme Robotics to Nova Labs as a staff engineer.",
        ]
        for f in facts:
            eng.ingest_text(f, extract_graph=True, consolidate_now=True, scope=sc)
        edges = eng.store.all_edges(sc)
        print(json.dumps({"seeded_facts": len(facts), "graph_edges": len(edges),
                          "namespace": args.namespace,
                          "note": "sample data ingested + consolidated; now export-graph it"},
                         indent=2))
        return 0
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
        res = bridge.export_graph(
            args.namespace, args.notebook_id, at=args.at,
            include_history=not args.no_history, max_entities=args.max_entities)
        if not res.get("exported"):
            # Empty graph (no consolidated edges yet) -> fall back to per-record export so
            # the notebook still gets the verified memories (provenance headers intact).
            fb = bridge.export_namespace(args.namespace, args.notebook_id, limit=args.limit)
            res = {"graph_export": res, "fell_back_to_per_record": fb,
                   "note": "graph had no edges (needs consolidation); exported raw records "
                           "instead. `remember` + let it consolidate for a graph."}
        print(json.dumps(res, indent=2))
    elif args.action == "sync":
        print(json.dumps(IncrementalSync(bridge, args.manifest).sync(
            args.namespace, args.notebook_id), indent=2))
    elif args.action == "answer":
        # Free NotebookLM read + provenance + cited-source check + deterministic grounding.
        # Needs NO metered model key unless --verify-with-qwen (adds a metered qwen NLI audit).
        print(json.dumps(bridge.answer(
            args.namespace, args.question, args.notebook_id,
            verify_with_qwen=args.verify_with_qwen), indent=2))
    elif args.action == "routed-answer":
        print(json.dumps(bridge.routed_answer(
            args.namespace, args.question, args.notebook_id,
            require_gate_verification=args.require_gate), indent=2))
    else:
        print(json.dumps(bridge.query(args.notebook_id, args.question), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
