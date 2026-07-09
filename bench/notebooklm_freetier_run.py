"""Collect LIVE NotebookLM free-tier answers for a committed holdout window -- key-free.

What this does (and what it refuses to claim):

  COLLECTS, live, per holdout question: the free Gemini answer over that conversation's
  OWN exported verified-claim-graph notebook, plus eidetic's deterministic checks
  (citation hash-confirmation + quote grounding). Caller LLM tokens: 0 -- no DashScope
  key is needed or used.

  SCORES only PRELIMINARILY: a normalized gold-containment heuristic, clearly labeled.
  It is NOT the pinned qwen3-max judge, so the number is NOT comparable to the benchmark
  scoreboard and is never merged into it. The answers file is judge-ready: the moment a
  funded key exists, `bench.judge` can score the same rows properly.

  ISOLATION: one notebook per conversation namespace (matching the harness's per-group
  namespaces), so a question can only see its own conversation's memories -- the same
  isolation the benchmark enforces.

    DATA_DIR=artifacts/holdout_rotation_r14_codex/data \
    .venv/bin/python -m bench.notebooklm_freetier_run \
        artifacts/holdout_rotation_r14_codex --limit 3   # smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

from eidetic.engine import Engine
from eidetic.integrations.notebooklm import CliBackend, NotebookLMBridge, find_notebook_id

SYS_FILE = "eidetic-plus-full__run0.jsonl"


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def sample_to_namespace(window: Path, sqlite_path: str) -> dict:
    """Map each logged sample -> its per-conversation namespace via the memory_ids the
    original run recorded (namespace isolation makes a cited/candidate memory live in the
    sample's own namespace). Same technique the offline verified-precision probe used."""
    mid2ns: dict[str, str] = {}
    con = sqlite3.connect(sqlite_path)
    for mid, ns in con.execute("select memory_id, namespace from memories"):
        mid2ns[mid] = ns
        mid2ns[mid[:16]] = ns
    out: dict[str, str] = {}
    for row in _rows(window / SYS_FILE):
        ex = row.get("extra") or {}
        ids = (ex.get("entailed_memory_ids") or []) + (ex.get("candidate_memory_ids") or [])
        ns = next((mid2ns.get(i) or mid2ns.get(str(i)[:16]) for i in ids
                   if (mid2ns.get(i) or mid2ns.get(str(i)[:16]))), None)
        if ns:
            out[row["sample_id"]] = ns
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def prelim_contains(gold: str, answer: str) -> bool:
    """PRELIMINARY heuristic only (labeled everywhere): all gold content tokens present in
    the answer, prefix-tolerant for inflection (intense/intensity). NOT the benchmark
    judge; over- and under-credits differently than the judge would."""
    stop = {"and", "the", "for", "with", "was", "were", "are", "his", "her",
            "they", "them", "that", "this", "from", "has", "had", "have", "not"}
    g = [t for t in _norm(gold).split() if len(t) > 2 and t not in stop]
    a = set(_norm(answer).split())

    def hit(t: str) -> bool:
        if t in a:
            return True
        return len(t) >= 5 and any(x.startswith(t[:5]) or t.startswith(x[:5])
                                   for x in a if len(x) >= 5)
    return bool(g) and all(hit(t) for t in g)


def pack_record_sources(bridge: NotebookLMBridge, namespace: str,
                        max_sources: int = 24, max_chars: int = 12_000) -> list[dict]:
    """Pack per-record provenance sources into combined sources that fit the free tier's
    per-notebook source cap. Each packed source concatenates whole record sources --
    provenance headers ride INSIDE the text, so citation resolution and quote grounding
    are unaffected. The graph source gives Gemini the compact verified facts; these give
    it the raw evidence the graph compaction drops (affect, detail, phrasing).

    NLM_CHUNK_CHARS>0 (default off): split long records into per-chunk sources so a fact
    buried in a long LongMemEval-S turn is separately surfaceable (see format_source_chunks).
    Off => byte-identical to the shipped behavior, so LoCoMo numbers are untouched."""
    import os as _os
    chunk_chars = int(_os.environ.get("NLM_CHUNK_CHARS", "0") or "0")
    if chunk_chars > 0:
        from eidetic.integrations.notebooklm import format_source_chunks
        singles = []
        for rec in bridge._records(namespace, None):  # noqa: SLF001 - same records as build_sources
            try:
                claims = bridge.engine.store.claims_by_source(rec.memory_id)
            except Exception:
                claims = []
            singles.extend(format_source_chunks(rec, claims, chunk_chars=chunk_chars))
    else:
        singles = bridge.build_sources(namespace)
    packed: list[dict] = []
    buf: list[str] = []
    size = 0
    for s in singles:
        t = s["text_content"]
        if buf and (size + len(t) > max_chars or len(packed) >= max_sources - 1):
            packed.append({"display_name": f"eidetic-records:{namespace[-6:]}:{len(packed)}",
                           "text_content": "\n\n=====\n\n".join(buf)})
            buf, size = [], 0
        buf.append(t)
        size += len(t)
    if buf:
        packed.append({"display_name": f"eidetic-records:{namespace[-6:]}:{len(packed)}",
                       "text_content": "\n\n=====\n\n".join(buf)})
    return packed[:max_sources]


def ensure_notebook(backend: CliBackend, title: str) -> str:
    """STRICT per-title notebook resolution. The non-strict fallback ('any first id')
    once resolved every missing title to the same notebook, silently mixing all 10
    namespaces' sources+questions into one pot -- cross-conversation contamination that
    the grounding check exposed (foreign quotes -> unmatched). Isolation is the whole
    point of per-namespace notebooks, so: exact title match or create, never fallback."""
    run = backend._require_runner()  # noqa: SLF001 - same pinned nlm CLI the backend uses
    nb = find_notebook_id(run(["notebook", "list", "--json"]), title, strict=True)
    if nb:
        return nb
    created = run(["notebook", "create", title, "--json"])
    nb = (find_notebook_id(created, title, strict=True)
          or find_notebook_id(created)  # create output carries only the new notebook
          or find_notebook_id(run(["notebook", "list", "--json"]), title, strict=True))
    if not nb:
        raise RuntimeError(f"could not create/find notebook {title!r}")
    return nb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("window")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke)")
    ap.add_argument("--out", default="", help="output jsonl (default <window>/notebooklm_freetier.jsonl)")
    ap.add_argument("--skip-export", action="store_true",
                    help="REPEAT RUNS: reuse the already-exported per-namespace notebooks "
                         "(re-exporting would duplicate sources and change what Gemini sees "
                         "between runs -- variance measurement needs identical notebooks)")
    args = ap.parse_args()
    window = Path(args.window)
    out_path = Path(args.out) if args.out else window / "notebooklm_freetier.jsonl"

    eng = Engine()
    bridge = NotebookLMBridge(eng, CliBackend())
    s2ns = sample_to_namespace(window, str(eng.settings.sqlite_path))
    rows = _rows(window / SYS_FILE)
    if args.limit:
        rows = rows[: args.limit]

    # resume: skip answered rows, RETRY errored ones (a transient nlm failure must not
    # permanently hole the collection)
    done = ({r["sample_id"] for r in _rows(out_path) if "error" not in r}
            if out_path.exists() else set())
    exported: dict[str, str] = {}   # namespace -> notebook_id (graph exported this run)
    n_ok = n_err = 0
    with open(out_path, "a") as fh:
        for row in rows:
            sid = row["sample_id"]
            if sid in done:
                continue
            ns = s2ns.get(sid)
            if not ns:
                continue
            try:
                if ns not in exported:
                    nb = ensure_notebook(bridge.backend, f"eidetic-{window.name}-{ns[-6:]}")
                    if not args.skip_export:
                        bridge.export_graph(ns, nb)  # compact verified facts (may be empty)
                        packed = pack_record_sources(bridge, ns)
                        if packed:  # raw evidence the graph compaction drops
                            bridge.backend.batch_create_sources(nb, packed)
                    exported[ns] = nb
                t0 = time.time()
                ans = bridge.answer(ns, row["question"], exported[ns])
                rec = {
                    "sample_id": sid, "category": row.get("category"),
                    "question": row["question"], "gold": row["gold"],
                    "namespace": ns, "notebook_id": exported[ns],
                    "nb_answer": ans.get("answer", ""),
                    "cited_sources": ans.get("cited_sources", {}),
                    "grounding": {k: v for k, v in (ans.get("grounding") or {}).items()
                                  if k != "method"},
                    "references": ans.get("references") or [],
                    "caller_llm_tokens": 0,
                    "latency_s": round(time.time() - t0, 1),
                    "prelim_contains_gold": prelim_contains(row["gold"], ans.get("answer", "")),
                    "labels": ("caller tokens 0 by construction (Gemini free read); "
                               "prelim_contains_gold is a HEURISTIC, NOT the qwen3-max judge; "
                               "not comparable to the benchmark scoreboard"),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_ok += 1
                print(f"[{n_ok}] {sid} contains_gold={rec['prelim_contains_gold']} "
                      f"confirmed={rec['cited_sources'].get('confirmed_in_eidetic')}"
                      f"/{rec['cited_sources'].get('cited')} {rec['latency_s']}s")
            except Exception as exc:  # noqa: BLE001 - per-question resilience, like the harness
                n_err += 1
                fh.write(json.dumps({"sample_id": sid, "error": f"{type(exc).__name__}: {exc}"[:300]}) + "\n")
                fh.flush()
                print(f"[ERR] {sid}: {exc}")
    print(f"\ndone ok={n_ok} err={n_err} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
