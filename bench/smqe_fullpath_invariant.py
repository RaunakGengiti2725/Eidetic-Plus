"""Rotating full-path invariant eval for SMQE.

This sidecar exercises the real benchmark product adapter path:
ingest_session -> consolidate -> EideticFullSystem.answer. It deliberately uses invented
conversations from the synthetic SMQE generator and a deterministic no-API client, then fails if the
fixed reader is reached. Passing here proves the public adapter wiring serves verified structured
recall cheaply; it does not tune or inspect any benchmark question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from statistics import quantiles
from typing import Optional

import numpy as np

from bench.adapters.eidetic_adapter import EideticFullSystem
from bench.seed_policy import resolve_seed
from bench.smqe_synthetic_invariant import _answer_matches, generate_cases
from eidetic.config import get_settings
from eidetic.engine import Engine


_LATENCY_BUDGET_MS = 100.0
_CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class _NoApiClient:
    def __init__(self, dim: int):
        self.dim = dim
        self.reader_models: list[str] = []

    def _e(self, text: str):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, text: str):
        return self._e(text)

    def embed_texts(self, texts: list[str]):
        return np.stack([self._e(text) for text in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def extract_edges(self, text: str) -> list[dict]:
        return []

    def extract_edges_bounded(self, text: str, *, max_windows: int = 0) -> list[dict]:
        return []

    def extract_claims(self, text: str) -> list[dict]:
        return []

    def extract_claims_bounded(self, text: str, *, max_windows: int = 0) -> list[dict]:
        return []

    def chat(self, model: str, system: str, user: str, **kw) -> str:
        self.reader_models.append(model)
        return "I do not have that in memory."

    def nli(self, premise: str, hypothesis: str):
        prem = re.sub(r"\s+", " ", (premise or "").lower()).strip()
        hyp = re.sub(r"\s+", " ", (hypothesis or "").lower()).strip()
        return ("entailment", 1.0) if hyp and hyp in prem else ("neutral", 0.0)


@contextmanager
def _adapter_env():
    """Keep the invariant independent of the caller's benchmark flags."""
    overrides = {
        "FULL_SLEEP": "0",
        "DREAM_AB": "0",
        "INGEST_GRANULARITY": "session",
    }
    old = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _split_role_content(text: str) -> tuple[str, str]:
    if ":" not in text:
        return "user", text
    role, content = text.split(":", 1)
    return (role.strip() or "user"), content.strip()


def _engine(tmp: Path) -> tuple[Engine, _NoApiClient]:
    settings = replace(
        get_settings(),
        data_dir=tmp / "data",
        vector_backend="numpy",
        rerank_enabled=False,
        semantic_cache_enabled=False,
        user_evidence_context_enabled=True,
        extract_chunking_enabled=False,
        consolidation_extract_deadline_sec=0.0,
        consolidation_extract_call_budget=0,
        consolidation_long_haystack_raw_only=False,
        consolidation_raw_only_window_threshold=0,
    )
    client = _NoApiClient(settings.embed_dim)
    engine = Engine(settings, client=client)
    return engine, client


def _consolidate_claims(report: dict) -> int:
    pending = report.get("consolidate_pending", report)
    if not isinstance(pending, dict):
        return 0
    return int(pending.get("claims_extracted", 0) or 0)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(quantiles(values, n=20, method="inclusive")[18])


def _immutable_proof_ok(extra: dict) -> bool:
    memory_ids = [str(mid) for mid in (extra.get("entailed_memory_ids") or []) if str(mid)]
    hashes = {
        str(value).strip().lower()
        for value in (extra.get("entailed_content_hashes") or [])
        if _CONTENT_HASH_RE.fullmatch(str(value).strip().lower())
    }
    raw_uris = [str(value).strip().lower() for value in (extra.get("entailed_raw_uris") or [])]
    return bool(memory_ids) and bool(hashes) and any(
        uri.startswith("cas://") and uri.removeprefix("cas://") in hashes
        for uri in raw_uris
    )


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures: list[dict] = []
    operator_counts: Counter[str] = Counter()
    case_operator_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    proof_tokens = 0
    context_tokens = 0
    latencies: list[float] = []
    latency_budget_checks = 0
    proof_link_checks = 0
    verified = 0
    structured = 0
    claims_extracted = 0
    # P0 fail-closed (2026-07-09): DERIVED count/sum cases no longer answer in the structured
    # adapter -- they fall through the tiers to the reader, which abstains. Count those consults
    # explicitly rather than hide them, so the "adapter runs without reader calls" claim stays
    # honestly scoped to the ops the adapter actually still answers.
    reader_consults = 0

    from bench import reader as bench_reader

    old_get_client = bench_reader.get_client
    with tempfile.TemporaryDirectory(prefix="smqe-fullpath-") as tmp_str, _adapter_env():
        engine, client = _engine(Path(tmp_str))
        bench_reader.get_client = lambda: client
        try:
            system = EideticFullSystem(engine=engine)
            for case in generated:
                namespace = f"smqe-fullpath-{case.case_id}"
                system.reset(namespace)
                for idx, (text, valid_at) in enumerate(case.rows):
                    role, content = _split_role_content(text)
                    system.ingest_session(
                        namespace,
                        f"{case.case_id}-s{idx}",
                        [{"role": role, "content": content}],
                        session_time=valid_at,
                    )
                claims_extracted += _consolidate_claims(system.consolidate(namespace))
                answer = system.answer(namespace, case.question, as_of=1_800_000_000)
                extra = answer.extra or {}
                policy = str(extra.get("policy", "") or "")
                op = str(extra.get("smqe_operator", "") or "")
                backend = str(extra.get("smqe_backend", "") or "")
                case_operator_counts[case.op] += 1
                if op:
                    operator_counts[op] += 1
                if backend:
                    backend_counts[backend] += 1
                if policy:
                    policy_counts[policy] += 1
                proof_tokens += int(extra.get("proof_surface_tokens", answer.context_tokens) or 0)
                context_tokens += int(answer.context_tokens or 0)
                e2e_ms = float(answer.e2e_ms or 0.0)
                latencies.append(e2e_ms)
                if e2e_ms <= _LATENCY_BUDGET_MS:
                    latency_budget_checks += 1
                verified += 1 if extra.get("verified") is True else 0
                structured += 1 if bool(extra.get("structured_recall")) or policy.startswith("smqe:") else 0
                proof_link_ok = _immutable_proof_ok(extra)
                if proof_link_ok:
                    proof_link_checks += 1
                proof = " ".join(str(c) for c in extra.get("entailed_memory_ids", []) or [])
                if getattr(case, "expect_abstain", False):
                    # P0 fail-closed (2026-07-09): a DERIVED count/sum no longer verifies in the
                    # structured adapter (eidetic/smqe/verify.py). It falls through the tiers and
                    # the product ABSTAINS -- correct-or-silent. The structured path no longer
                    # answers these, so the zero-reader-call invariant does not apply to them; a
                    # reader-tier consult that itself abstains is the designed fallback.
                    # The reader consult is REQUIRED (adversarial review): abstention alone
                    # cannot distinguish designed fail-closed from SMQE never running at all --
                    # a broken adapter that abstains everywhere would keep this gate green.
                    ok = (extra.get("verified") is not True and bool(answer.abstained)
                          and len(client.reader_models) > 0)
                else:
                    ok = (
                        extra.get("verified") is True
                        and not answer.abstained
                        and policy.startswith("smqe:")
                        and proof_link_ok
                        and _answer_matches(answer.answer, case.expected)
                        and not client.reader_models
                    )
                if not ok:
                    failures.append({
                        "case_id": case.case_id,
                        "case_op": case.op,
                        "question": case.question,
                        "expected": case.expected,
                        "actual": answer.answer,
                        "policy": policy,
                        "operator": op,
                        "backend": backend,
                        "verified": bool(extra.get("verified")),
                        "abstained": bool(answer.abstained),
                        "reader_calls": len(client.reader_models),
                        "content_hashes": len(extra.get("entailed_content_hashes", []) or []),
                        "raw_uris": len(extra.get("entailed_raw_uris", []) or []),
                        "proof": proof[:500],
                    })
                reader_consults += len(client.reader_models)
                client.reader_models.clear()
        finally:
            bench_reader.get_client = old_get_client

    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        # P0 fail-closed (2026-07-09): derived count/sum cases assert abstention; published so
        # the release gate can scope its all-cases-verified checks to the answerable remainder.
        "expected_abstain_cases": sum(
            1 for case in generated if getattr(case, "expect_abstain", False)
        ),
        "correct": cases - len(failures),
        "verified": verified,
        "structured_recall": structured,
        "reader_calls": 0 if not failures else sum(int(f.get("reader_calls", 0)) for f in failures),
        "reader_consults": reader_consults,
        "proof_link_checks": proof_link_checks,
        "claim_backend_correct": backend_counts.get("claim", 0),
        "claims_extracted": claims_extracted,
        "avg_claims_per_case": round(claims_extracted / cases, 2),
        "failures": failures,
        "operator_counts": dict(sorted(operator_counts.items())),
        "case_operator_counts": dict(sorted(case_operator_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "avg_proof_tokens": round(proof_tokens / cases, 2),
        "avg_context_tokens": round(context_tokens / cases, 2),
        "latency_budget_checks": latency_budget_checks,
        "latency_budget_ms": _LATENCY_BUDGET_MS,
        "p95_latency_ms": round(_p95(latencies), 6),
        "max_latency_ms": round(max(latencies, default=0.0), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="repro seed; omitted means random")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    report = run_eval(seed=args.seed, cases=args.cases)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
