"""Preflight 'doctor': prove the live key + model IDs actually work, one real call per capability.

Run:
    python -m eidetic.doctor                 # human + JSON summary, exit code by health

It is the acceptance test for "the moment a real DASHSCOPE_API_KEY is in .env the product path
works for real": it makes ONE tiny real call per capability (embed / chat / rerank / multimodal /
document) against the CONFIGURED model IDs and reports pass/fail + latency. It never returns a
green check on a dead call, and it tells quota-exhaustion apart from a bad key / renamed model so a
403'd free tier does not read as "everything is broken".

No key -> every capability is reported `skipped: no_key` (honest, not a fake pass). It never
fabricates a result.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from time import perf_counter
from typing import Callable, Optional

import numpy as np

from .dashscope_client import ModelCallError

# A real 1x1 PNG (so embed_image gets a genuinely decodable image, not a dead fixture).
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _classify_error(msg: str) -> str:
    """Tell the failure modes apart so a quota block does not look like a dead capability."""
    m = (msg or "").lower()
    if "quota" in m or "free tier" in m or "allocationquota" in m.replace(" ", ""):
        return "quota_exhausted"
    if "model" in m and ("not exist" in m or "not found" in m or "invalid" in m or "unknown" in m):
        return "bad_model_id"
    if "api key" in m or "apikey" in m or "unauthor" in m or "invalid key" in m or "access denied" in m:
        return "auth"
    return "error"


def _tmpfile(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def preflight(engine=None, *, include_document: bool = True) -> dict:
    """Make one real call per capability and return a structured report. Read-only on storage."""
    from .engine import Engine
    eng = engine if engine is not None else Engine()
    s = eng.settings
    client = eng.client
    caps: list[dict] = []

    def run(name: str, model: str, fn: Callable[[], dict]) -> dict:
        if not s.has_api_key:
            return {"capability": name, "model": model, "ok": False, "skipped": True,
                    "error_class": "no_key", "error": "DASHSCOPE_API_KEY not set"}
        t0 = perf_counter()
        try:
            detail = fn()
            return {"capability": name, "model": model, "ok": True,
                    "latency_ms": round((perf_counter() - t0) * 1000.0, 1), "detail": detail}
        except ModelCallError as e:
            return {"capability": name, "model": model, "ok": False,
                    "latency_ms": round((perf_counter() - t0) * 1000.0, 1),
                    "error_class": _classify_error(str(e)), "error": str(e)[:300]}
        except Exception as e:  # a LOCAL problem (fixture/decode), NOT a dead remote capability
            return {"capability": name, "model": model, "ok": False,
                    "error_class": "local", "error": f"{type(e).__name__}: {e}"[:300]}

    def _embed() -> dict:
        v = np.asarray(client.embed_texts(["preflight ping"]))
        dim = int(v.shape[-1])
        if dim != s.embed_dim:
            raise ValueError(f"embed dim {dim} != EMBED_DIM {s.embed_dim}")
        return {"dim": dim}

    def _chat() -> dict:
        out = client.chat(s.salience_model, "Reply with exactly: ok", "ping",
                          temperature=0.0, max_tokens=8)
        if not isinstance(out, str) or not out.strip():
            raise ValueError("empty chat response")
        return {"sample": out.strip()[:40]}

    def _rerank() -> dict:
        r = client.rerank("apple fruit", ["an apple is a fruit", "the sky is blue"], 2)
        if not r:
            raise ValueError("empty rerank response")
        return {"results": len(r), "top_score": round(float(r[0][1]), 4)}

    def _image() -> dict:
        path = _tmpfile(".png", _TINY_PNG)
        try:
            v = np.asarray(client.embed_image(path))
            return {"dim": int(v.shape[-1])}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _document() -> dict:
        path = _tmpfile(".txt", b"Preflight probe. The capital of France is Paris.")
        try:
            txt = client.read_document(path)
            if not isinstance(txt, str):
                raise ValueError("read_document did not return text")
            return {"chars": len(txt)}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    caps.append(run("embed", s.text_embed_model, _embed))
    caps.append(run("chat", s.salience_model, _chat))
    caps.append(run("rerank", s.rerank_model, _rerank))
    caps.append(run("embed_image", s.multimodal_embed_model, _image))
    if include_document:
        doc = run("read_document", s.doc_model, _document)
        doc["optional"] = True          # document reading needs a long-context reader the account
        caps.append(doc)                # may not have; it never marks the CORE path degraded

    # Optional capabilities are reported but excluded from the health headline.
    failing = [c for c in caps if not c["ok"] and not c.get("skipped") and not c.get("optional")]
    if not s.has_api_key:
        summary = "no_key"
    elif not failing:
        summary = "healthy"
    elif all(c.get("error_class") == "quota_exhausted" for c in failing):
        summary = "quota_exhausted"     # the key is valid; the free tier is just exhausted
    else:
        summary = "degraded"
    return {"has_api_key": s.has_api_key, "region": s.region, "summary": summary,
            "failing": [c["capability"] for c in failing], "capabilities": caps}


_EXIT = {"healthy": 0, "degraded": 1, "quota_exhausted": 2, "no_key": 3}


def main(argv: Optional[list] = None) -> int:
    report = preflight()
    print(json.dumps(report, indent=2))
    sym = {"healthy": "OK", "quota_exhausted": "QUOTA", "degraded": "FAIL", "no_key": "NO-KEY"}
    print(f"\npreflight: {sym.get(report['summary'], report['summary'])} "
          f"({report['summary']})", file=sys.stderr)
    for c in report["capabilities"]:
        mark = "ok " if c["ok"] else ("-- " if c.get("skipped") else "XX ")
        extra = c.get("detail") or c.get("error_class", "")
        print(f"  {mark}{c['capability']:<14} {c['model']:<28} {extra}", file=sys.stderr)
    return _EXIT.get(report["summary"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
