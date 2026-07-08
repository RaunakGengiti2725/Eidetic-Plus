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

from eidetic.models import ClaimRecord, MemoryRecord, Scope

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
        created = 0
        for s in sources:
            self.runner(["source", "add", "--notebook", notebook_id,
                         "--text", s["text_content"], "--name", s.get("display_name", "")])
            created += 1
        return {"created": created}

    def query(self, notebook_id: str, question: str) -> dict:
        out = self.runner(["query", "--notebook", notebook_id, question])
        return {"answer": out.strip(), "backend": "nlm-cli (gemini-side, UNVERIFIED)"}


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
                         limit: Optional[int] = None, batch_size: int = 20) -> dict:
        """Push a namespace's verified memories into `notebook_id` as sources. Returns a
        summary; raises NotebookLMError on backend failure (no partial silent success)."""
        sources = self.build_sources(namespace, limit)
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
        for rec in self.engine.store.active_records_at(None, scope):
            short = rec.memory_id[:16]
            if rec.memory_id in ids or short in ids or any(rec.memory_id.startswith(i) for i in ids):
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
        eidetic's verify-or-abstain proof gate. Use `engine.recall()` when you need a cited,
        gate-verified answer instead of a free one."""
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


def _cli() -> int:  # pragma: no cover - thin argparse wrapper
    import argparse
    from eidetic.config import get_settings
    from eidetic.engine import Engine

    ap = argparse.ArgumentParser(description="Export eidetic verified memory into NotebookLM")
    ap.add_argument("action", choices=["export", "query", "preview"])
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--notebook-id", default=os.environ.get("NOTEBOOKLM_NOTEBOOK_ID", ""))
    ap.add_argument("--backend", choices=["enterprise", "cli"], default="enterprise")
    ap.add_argument("--project-number", default=os.environ.get("NOTEBOOKLM_PROJECT_NUMBER", ""))
    ap.add_argument("--location", default=os.environ.get("NOTEBOOKLM_LOCATION", "global"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--question", default="")
    args = ap.parse_args()

    eng = Engine(get_settings())
    if args.action == "preview":
        bridge = NotebookLMBridge(eng, backend=None)
        srcs = bridge.build_sources(args.namespace, args.limit)
        print(json.dumps({"sources": len(srcs),
                          "first": srcs[0] if srcs else None}, indent=2))
        return 0
    backend = (EnterpriseBackend(project_number=args.project_number, location=args.location)
               if args.backend == "enterprise" else CliBackend())
    bridge = NotebookLMBridge(eng, backend)
    if args.action == "export":
        print(json.dumps(bridge.export_namespace(
            args.namespace, args.notebook_id, limit=args.limit), indent=2))
    else:
        print(json.dumps(bridge.query(args.notebook_id, args.question), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
