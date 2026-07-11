from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from eidetic.models import ABSTENTION_TEXT, AnswerStatus

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPLAY_VERSION = "phase-a-v1"
_GUARD_PROJECTION_VERSION = "policy-v2-guards-v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    else:
        text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def _source_id(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _load_source(path: Path) -> tuple[dict, list[dict]]:
    raw = path.read_bytes()
    rows: list[dict] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{_source_id(path)}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{_source_id(path)}:{line_number}: row is not a JSON object")
        rows.append(row)
    return {
        "source_id": _source_id(path),
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "rows": len(rows),
    }, rows


def _positive_number(value: object) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _proof_metadata_issues(extra: dict) -> list[str]:
    issues: list[str] = []
    citations = extra.get("citations")
    if not _positive_number(citations):
        issues.append("citations")
    memory_ids = extra.get("entailed_memory_ids")
    if not isinstance(memory_ids, list) or not any(str(item).strip() for item in memory_ids):
        issues.append("entailed_memory_ids")
    hashes = extra.get("entailed_content_hashes")
    valid_hashes = [str(item).strip().lower() for item in hashes or []
                    if _SHA256_RE.fullmatch(str(item).strip().lower())]
    if not valid_hashes:
        issues.append("entailed_content_hashes")
    raw_uris = extra.get("entailed_raw_uris")
    valid_uris = [str(item).strip() for item in raw_uris or []
                  if str(item).strip().startswith(("cas://", "oss://"))]
    if not valid_uris:
        issues.append("entailed_raw_uris")
    if valid_hashes and valid_uris:
        cas_hashes = {uri.removeprefix("cas://").lower() for uri in valid_uris
                      if uri.startswith("cas://")}
        if cas_hashes and not set(valid_hashes).issubset(cas_hashes):
            issues.append("content_hash_raw_uri_mismatch")
    if not _positive_number(extra.get("proof_surface_tokens")):
        issues.append("proof_surface_tokens")
    return issues


def _original_status(row: dict) -> str:
    if bool(row.get("abstained")):
        return AnswerStatus.ABSTAINED.value
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    if bool(extra.get("verified")):
        return AnswerStatus.VERIFIED.value
    return "UNVERIFIED_DELIVERED"


def _replay_row(row: dict, source: dict, row_number: int) -> tuple[dict, dict]:
    replayed = copy.deepcopy(row)
    extra = replayed.get("extra") if isinstance(replayed.get("extra"), dict) else {}
    replayed["extra"] = extra
    original_status = _original_status(row)
    original_answer = str(row.get("predicted", "") or "")
    original_correct = bool(row.get("correct"))
    proof_issues = _proof_metadata_issues(extra) if original_status == AnswerStatus.VERIFIED.value else []
    preserved = original_status == AnswerStatus.VERIFIED.value and not proof_issues
    if preserved:
        status = AnswerStatus.VERIFIED.value
        reason = "preserved_verified_with_complete_proof_metadata"
        replayed["abstained"] = False
        extra["verified"] = True
        extra["status"] = status
    else:
        status = AnswerStatus.ABSTAINED.value
        if original_status == "UNVERIFIED_DELIVERED":
            reason = "converted_unverified_delivery"
        elif original_status == AnswerStatus.VERIFIED.value:
            reason = "converted_invalid_verified_proof_metadata"
        else:
            reason = "canonicalized_existing_abstention"
        replayed["predicted"] = ABSTENTION_TEXT
        replayed["correct"] = False
        replayed["abstained"] = True
        extra["verified"] = False
        extra["status"] = status
        extra["confidence"] = 0.0
        extra["citations"] = 0
        extra["entailed_memory_ids"] = []
        extra["entailed_content_hashes"] = []
        extra["entailed_raw_uris"] = []
        extra["proof_surface_tokens"] = 0
    replayed["status"] = status
    extra["replay"] = {
        "version": _REPLAY_VERSION,
        "mode": "mechanical_policy_replay",
        "source_id": source["source_id"],
        "source_sha256": source["sha256"],
        "source_row": row_number,
        "original_status": original_status,
        "original_answer": original_answer,
        "original_correct": original_correct,
        "result_status": status,
        "reason": reason,
        "proof_metadata_issues": proof_issues,
        "provider_calls": 0,
        "nli_calls": 0,
        "generation_calls": 0,
    }
    audit = {
        "original_status": original_status,
        "result_status": status,
        "original_correct": original_correct,
        "result_correct": bool(replayed.get("correct")),
        "preserved": preserved,
        "reason": reason,
        "proof_metadata_issues": proof_issues,
    }
    return replayed, audit


def replay_sources(paths: list[Path]) -> tuple[list[dict], dict]:
    loaded: list[tuple[dict, list[dict]]] = []
    seen: set[str] = set()
    for path in paths:
        source, rows = _load_source(Path(path))
        if source["sha256"] in seen:
            raise ValueError(f"duplicate source content: {source['source_id']}")
        seen.add(source["sha256"])
        loaded.append((source, rows))
    loaded.sort(key=lambda item: item[0]["source_id"])
    replayed_rows: list[dict] = []
    windows: list[dict] = []
    for source, rows in loaded:
        audits: list[dict] = []
        for row_number, row in enumerate(rows, start=1):
            replayed, audit = _replay_row(row, source, row_number)
            replayed_rows.append(replayed)
            audits.append(audit)
        original_verified = [item for item in audits
                             if item["original_status"] == AnswerStatus.VERIFIED.value]
        replay_verified = [item for item in audits
                           if item["result_status"] == AnswerStatus.VERIFIED.value]
        converted_unverified = [item for item in audits
                                if item["reason"] == "converted_unverified_delivery"]
        invalid_verified = [item for item in audits
                            if item["reason"] == "converted_invalid_verified_proof_metadata"]
        windows.append({
            **source,
            "original": {
                "correct": sum(item["original_correct"] for item in audits),
                "verified_answered": len(original_verified),
                "verified_correct": sum(item["original_correct"] for item in original_verified),
                "abstained": sum(item["original_status"] == AnswerStatus.ABSTAINED.value
                                 for item in audits),
                "unverified_answered": sum(item["original_status"] == "UNVERIFIED_DELIVERED"
                                           for item in audits),
            },
            "replay": {
                "correct": sum(item["result_correct"] for item in audits),
                "verified_answered": len(replay_verified),
                "verified_correct": sum(item["result_correct"] for item in replay_verified),
                "abstained": sum(item["result_status"] == AnswerStatus.ABSTAINED.value
                                 for item in audits),
                "unverified_answered": 0,
            },
            "converted_unverified": len(converted_unverified),
            "converted_unverified_correct": sum(item["original_correct"]
                                                 for item in converted_unverified),
            "converted_unverified_wrong": sum(not item["original_correct"]
                                               for item in converted_unverified),
            "invalid_verified_proof_rows": len(invalid_verified),
            "invalid_verified_samples": [
                str(rows[index].get("sample_id", ""))
                for index, item in enumerate(audits)
                if item["reason"] == "converted_invalid_verified_proof_metadata"
            ],
        })
    aggregate = {
        "sources": len(windows),
        "rows": sum(window["rows"] for window in windows),
        "original_correct": sum(window["original"]["correct"] for window in windows),
        "original_verified_answered": sum(window["original"]["verified_answered"]
                                          for window in windows),
        "original_verified_correct": sum(window["original"]["verified_correct"]
                                         for window in windows),
        "original_abstained": sum(window["original"]["abstained"] for window in windows),
        "original_unverified_answered": sum(window["original"]["unverified_answered"]
                                            for window in windows),
        "replay_correct": sum(window["replay"]["correct"] for window in windows),
        "replay_verified_answered": sum(window["replay"]["verified_answered"]
                                        for window in windows),
        "replay_verified_correct": sum(window["replay"]["verified_correct"]
                                       for window in windows),
        "replay_abstained": sum(window["replay"]["abstained"] for window in windows),
        "replay_unverified_answered": 0,
        "converted_unverified": sum(window["converted_unverified"] for window in windows),
        "converted_unverified_correct": sum(window["converted_unverified_correct"]
                                             for window in windows),
        "converted_unverified_wrong": sum(window["converted_unverified_wrong"]
                                           for window in windows),
        "invalid_verified_proof_rows": sum(window["invalid_verified_proof_rows"]
                                           for window in windows),
    }
    regressions = (aggregate["original_verified_correct"]
                   - aggregate["replay_verified_correct"])
    verified_wrong_rows = [
        row for row in replayed_rows
        if row.get("status") == AnswerStatus.VERIFIED.value and not bool(row.get("correct"))
    ]
    by_category = Counter(str(row.get("category", "unknown")) for row in verified_wrong_rows)
    by_policy = Counter(
        str((row.get("extra") or {}).get("policy") or "missing")
        for row in verified_wrong_rows
    )
    by_operator = Counter(
        str((row.get("extra") or {}).get("smqe_operator") or "reader")
        for row in verified_wrong_rows
    )
    verified_wrong_summary = {
        "rows": len(verified_wrong_rows),
        "rate_among_verified": round(
            len(verified_wrong_rows) / aggregate["replay_verified_answered"], 6
        ) if aggregate["replay_verified_answered"] else 0.0,
        "by_category": dict(sorted(by_category.items())),
        "by_policy": dict(sorted(by_policy.items())),
        "by_operator": dict(sorted(by_operator.items())),
        "sample_ids": sorted(str(row.get("sample_id", "")) for row in verified_wrong_rows),
    }
    checks = {
        "all_sources_nonempty": bool(windows) and all(window["rows"] > 0 for window in windows),
        "row_count_preserved": aggregate["rows"] == len(replayed_rows),
        "no_unverified_answered": aggregate["replay_unverified_answered"] == 0,
        "no_invalid_verified_proof_metadata": aggregate["invalid_verified_proof_rows"] == 0,
        "zero_verified_correct_regression": regressions == 0,
        "zero_provider_calls": True,
    }
    implementation_bytes = Path(__file__).read_bytes()
    report = {
        "artifact_type": "burned_window_mechanical_replay",
        "version": _REPLAY_VERSION,
        "implementation": {
            "path": "bench/replay.py",
            "bytes": len(implementation_bytes),
            "sha256": _sha256(implementation_bytes),
        },
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": "mechanical_policy_replay",
        "provider_calls": 0,
        "nli_calls": 0,
        "generation_calls": 0,
        "checks": checks,
        "aggregate": aggregate,
        "verified_wrong": verified_wrong_summary,
        "verified_correct_regressions": regressions,
        "windows": windows,
        "limitations": [
            "Reuses frozen historical correctness and verification labels; it does not rerun retrieval, generation, NLI, or judges.",
            "Validates immutable proof metadata linkage in logs but cannot re-fetch historical CAS bytes without the original data stores.",
            "Measures the exact policy effect of forbidding unverified delivery, not the full quality of the current runtime.",
        ],
    }
    return replayed_rows, report


def _row_guard_floor(row: dict) -> Optional[str]:
    """Apply the CURRENT runtime deterministic guards to one frozen benchmark row.

    Pure mechanical projection: reader rows face `reader_answer_form_credible`, structured
    rows face `structured_answer_form_floor` -- the exact functions the runtime enforces,
    imported from eidetic.smqe.verify (whose bytes the manifest binds). Returns the floor
    name that would reject the answer today, or None when the row survives.
    """
    from eidetic.models import StructuredAnswerResult
    from eidetic.smqe.verify import (reader_answer_form_credible,
                                     structured_answer_form_floor)

    query = str(row.get("question", "") or "")
    answer = str(row.get("predicted", "") or "")
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    policy = str(extra.get("policy") or "")
    if policy.startswith("smqe:"):
        parts = policy.split(":")
        op = parts[1] if len(parts) > 1 else ""
        result = StructuredAnswerResult(answer=answer, op=op, note=policy)
        return structured_answer_form_floor(query, result)
    if not reader_answer_form_credible(query, answer):
        return "reader_form"
    return None


def project_guard_policy(paths: list[Path]) -> dict:
    """Mechanically project the CURRENT deterministic guard policy over frozen rows.

    For every row that historically shipped VERIFIED, decide whether today's pure form
    floors would keep it VERIFIED or convert it to ABSTAINED. No provider, NLI, judge, or
    generation calls -- correctness labels are the frozen judge labels. The projection
    answers one question with exact counts: what does the guard policy convert, split into
    wrong-converted (gain) and correct-lost (cost), with every loss enumerated.
    """
    loaded: list[tuple[dict, list[dict]]] = []
    seen: set[str] = set()
    for path in paths:
        source, rows = _load_source(Path(path))
        if source["sha256"] in seen:
            raise ValueError(f"duplicate source content: {source['source_id']}")
        seen.add(source["sha256"])
        loaded.append((source, rows))
    loaded.sort(key=lambda item: item[0]["source_id"])

    verified_rows = 0
    verified_wrong_before = 0
    converted_wrong: Counter = Counter()
    lost_correct: Counter = Counter()
    residual_wrong: Counter = Counter()
    losses: list[dict] = []
    conversions: list[dict] = []
    sources: list[dict] = []
    for source, rows in loaded:
        sources.append(source)
        for row_number, row in enumerate(rows, start=1):
            if _original_status(row) != AnswerStatus.VERIFIED.value:
                continue
            if _proof_metadata_issues(row.get("extra") or {}):
                continue
            verified_rows += 1
            correct = bool(row.get("correct"))
            extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
            policy = str(extra.get("policy") or "reader")
            family = policy.split(":")[1] if policy.startswith("smqe:") else "reader"
            if not correct:
                verified_wrong_before += 1
            floor = _row_guard_floor(row)
            record = {
                "source_id": source["source_id"],
                "source_row": row_number,
                "sample_id": str(row.get("sample_id", "")),
                "family": family,
                "floor": floor,
                "question": str(row.get("question", ""))[:160],
                "predicted": str(row.get("predicted", ""))[:160],
                "gold": str(row.get("gold", ""))[:160],
            }
            if floor is not None and not correct:
                converted_wrong[floor] += 1
                conversions.append(record)
            elif floor is not None and correct:
                lost_correct[floor] += 1
                losses.append(record)
            elif not correct:
                residual_wrong[family] += 1
    checks = {
        "all_sources_nonempty": bool(sources),
        "zero_provider_calls": True,
        "all_losses_enumerated": len(losses) == sum(lost_correct.values()),
        "all_conversions_enumerated": len(conversions) == sum(converted_wrong.values()),
    }
    implementation_files = {
        "bench/replay.py": Path(__file__).resolve(),
        "eidetic/smqe/verify.py": Path(__file__).resolve().parent.parent
        / "eidetic" / "smqe" / "verify.py",
    }
    implementations = []
    for rel, path in sorted(implementation_files.items()):
        data = path.read_bytes()
        implementations.append({"path": rel, "bytes": len(data), "sha256": _sha256(data)})
    return {
        "artifact_type": "burned_window_guard_projection",
        "version": _GUARD_PROJECTION_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": "mechanical_guard_projection",
        "provider_calls": 0,
        "nli_calls": 0,
        "generation_calls": 0,
        "checks": checks,
        "implementations": implementations,
        "sources": sources,
        "aggregate": {
            "verified_rows": verified_rows,
            "verified_wrong_before": verified_wrong_before,
            "wrong_converted_to_abstain": sum(converted_wrong.values()),
            "correct_lost_to_abstain": sum(lost_correct.values()),
            "verified_wrong_after": verified_wrong_before - sum(converted_wrong.values()),
            "net_verified_delta": -(sum(converted_wrong.values()) + sum(lost_correct.values())),
        },
        "wrong_converted_by_floor": dict(sorted(converted_wrong.items())),
        "correct_lost_by_floor": dict(sorted(lost_correct.items())),
        "residual_wrong_by_family": dict(sorted(residual_wrong.items())),
        "conversions": conversions,
        "losses": losses,
        "limitations": [
            "Projects only the PURE deterministic form floors; support-dependent floors (aggregate citation, category anchoring, premise position) and the relative_temporal ambiguity guard need live candidate sets and are NOT projected here.",
            "Reuses frozen historical correctness labels; it does not rerun retrieval, generation, NLI, or judges.",
            "Correct-lost rows are enumerated in full; accepting them is a policy decision, not a mechanical one.",
        ],
    }


def render_guard_projection_markdown(report: dict) -> str:
    agg = report["aggregate"]
    lines = [
        "# Burned-Window Guard Projection (policy v2)",
        "",
        f"Status: **{report['status']}**",
        "",
        "Applies the CURRENT deterministic runtime guards to frozen historically-VERIFIED rows.",
        "No provider, NLI, judge, or generation calls.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| verified rows | {agg['verified_rows']} |",
        f"| verified wrong before | {agg['verified_wrong_before']} |",
        f"| wrong converted to abstain | {agg['wrong_converted_to_abstain']} |",
        f"| correct lost to abstain | {agg['correct_lost_to_abstain']} |",
        f"| verified wrong after | {agg['verified_wrong_after']} |",
        "",
        "## Wrong converted, by floor",
        "",
    ]
    for floor, count in report["wrong_converted_by_floor"].items():
        lines.append(f"- **{floor}**: {count}")
    lines.extend(["", "## Correct lost, by floor (every row enumerated)", ""])
    for floor, count in report["correct_lost_by_floor"].items():
        lines.append(f"- **{floor}**: {count}")
    for loss in report["losses"]:
        lines.append(f"  - `{loss['sample_id']}` [{loss['floor']}] Q: {loss['question'][:80]}"
                     f" | P: {loss['predicted'][:80]}")
    lines.extend(["", "## Residual verified-wrong, by family", ""])
    for family, count in report["residual_wrong_by_family"].items():
        lines.append(f"- **{family}**: {count}")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- **Boundary**: {limitation}")
    return "\n".join(lines) + "\n"


def write_guard_projection_artifact(paths: list[Path], out_dir: Path) -> dict:
    report = project_guard_policy(paths)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_bytes = _canonical_json(report, pretty=True)
    (out_dir / "guard_projection.json").write_bytes(report_bytes)
    markdown_bytes = render_guard_projection_markdown(report).encode("utf-8")
    (out_dir / "guard_projection.md").write_bytes(markdown_bytes)
    manifest = {
        "artifact_type": report["artifact_type"],
        "version": report["version"],
        "status": report["status"],
        "algorithm": "sha256",
        "implementations": report["implementations"],
        "sources": report["sources"],
        "outputs": [
            {"path": "guard_projection.json", "bytes": len(report_bytes),
             "sha256": _sha256(report_bytes)},
            {"path": "guard_projection.md", "bytes": len(markdown_bytes),
             "sha256": _sha256(markdown_bytes)},
        ],
    }
    (out_dir / "guard_projection_manifest.json").write_bytes(
        _canonical_json(manifest, pretty=True))
    return manifest


def render_markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Burned-Window Mechanical Replay",
        "",
        f"Status: **{report['status']}**",
        "",
        "This artifact performs no provider, generation, NLI, or judge calls. It replays the Phase A output policy over frozen rows.",
        "",
        "| metric | original | replay |",
        "|---|---:|---:|",
        f"| rows | {aggregate['rows']} | {aggregate['rows']} |",
        f"| correct | {aggregate['original_correct']} | {aggregate['replay_correct']} |",
        f"| verified answered | {aggregate['original_verified_answered']} | {aggregate['replay_verified_answered']} |",
        f"| verified correct | {aggregate['original_verified_correct']} | {aggregate['replay_verified_correct']} |",
        f"| abstained | {aggregate['original_abstained']} | {aggregate['replay_abstained']} |",
        f"| unverified answered | {aggregate['original_unverified_answered']} | {aggregate['replay_unverified_answered']} |",
        f"| verified wrong | {report['verified_wrong']['rows']} | {report['verified_wrong']['rows']} |",
        "",
        "| source | rows | original correct | replay correct | converted unverified | invalid verified proof |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window in report["windows"]:
        lines.append(
            f"| {window['source_id']} | {window['rows']} | {window['original']['correct']} "
            f"| {window['replay']['correct']} | {window['converted_unverified']} "
            f"| {window['invalid_verified_proof_rows']} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- **{name}**: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- **Boundary**: {limitation}")
    return "\n".join(lines) + "\n"


def write_replay_artifact(paths: list[Path], out_dir: Path) -> dict:
    rows, report = replay_sources(paths)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_bytes = b"".join(_canonical_json(row) for row in rows)
    rows_path = out_dir / "replay_rows.jsonl"
    rows_path.write_bytes(rows_bytes)
    report["replay_rows"] = {
        "path": rows_path.name,
        "bytes": len(rows_bytes),
        "sha256": _sha256(rows_bytes),
    }
    report_bytes = _canonical_json(report, pretty=True)
    report_path = out_dir / "replay_report.json"
    report_path.write_bytes(report_bytes)
    markdown_bytes = render_markdown(report).encode("utf-8")
    markdown_path = out_dir / "replay_report.md"
    markdown_path.write_bytes(markdown_bytes)
    manifest = {
        "artifact_type": report["artifact_type"],
        "version": report["version"],
        "status": report["status"],
        "algorithm": "sha256",
        "implementation": report["implementation"],
        "sources": [
            {key: window[key] for key in ("source_id", "bytes", "sha256", "rows")}
            for window in report["windows"]
        ],
        "outputs": [
            {"path": rows_path.name, "bytes": len(rows_bytes), "sha256": _sha256(rows_bytes)},
            {"path": report_path.name, "bytes": len(report_bytes), "sha256": _sha256(report_bytes)},
            {"path": markdown_path.name, "bytes": len(markdown_bytes),
             "sha256": _sha256(markdown_bytes)},
        ],
    }
    manifest_bytes = _canonical_json(manifest, pretty=True)
    (out_dir / "replay_manifest.json").write_bytes(manifest_bytes)
    return manifest


def verify_replay_artifact(out_dir: Path, repo_root: Optional[Path] = None) -> dict:
    out_dir = Path(out_dir)
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    manifest_path = out_dir / "replay_manifest.json"
    failures: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "failures": [f"manifest:{exc}"]}
    implementation = manifest.get("implementation") or {}
    implementation_path = repo_root / str(implementation.get("path", ""))
    if not implementation_path.is_file():
        failures.append("implementation:missing")
    else:
        data = implementation_path.read_bytes()
        if len(data) != implementation.get("bytes") or _sha256(data) != implementation.get("sha256"):
            failures.append("implementation:hash_mismatch")
    source_root = out_dir.parent
    source_rows = 0
    for source in manifest.get("sources") or []:
        path = source_root / str(source.get("source_id", ""))
        if not path.is_file():
            failures.append(f"source:{source.get('source_id')}:missing")
            continue
        data = path.read_bytes()
        rows = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
        source_rows += rows
        if (len(data) != source.get("bytes") or _sha256(data) != source.get("sha256")
                or rows != source.get("rows")):
            failures.append(f"source:{source.get('source_id')}:hash_or_row_mismatch")
    for output in manifest.get("outputs") or []:
        path = out_dir / str(output.get("path", ""))
        if not path.is_file():
            failures.append(f"output:{output.get('path')}:missing")
            continue
        data = path.read_bytes()
        if len(data) != output.get("bytes") or _sha256(data) != output.get("sha256"):
            failures.append(f"output:{output.get('path')}:hash_mismatch")
    replay_rows_path = out_dir / "replay_rows.jsonl"
    replay_rows = []
    if replay_rows_path.is_file():
        try:
            replay_rows = [json.loads(line) for line in replay_rows_path.read_text().splitlines()
                           if line.strip()]
        except json.JSONDecodeError as exc:
            failures.append(f"replay_rows:invalid_json:{exc}")
    if len(replay_rows) != source_rows:
        failures.append(f"replay_rows:count:{len(replay_rows)}!=sources:{source_rows}")
    for index, row in enumerate(replay_rows, start=1):
        status = str(row.get("status", ""))
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        if status not in {AnswerStatus.VERIFIED.value, AnswerStatus.ABSTAINED.value}:
            failures.append(f"replay_rows:{index}:invalid_status:{status or '<missing>'}")
            continue
        if status == AnswerStatus.VERIFIED.value:
            if not bool(extra.get("verified")) or _proof_metadata_issues(extra):
                failures.append(f"replay_rows:{index}:invalid_verified_proof")
        elif (not bool(row.get("abstained")) or bool(extra.get("verified"))
              or row.get("citations") or extra.get("citations")
              or extra.get("entailed_memory_ids") or extra.get("entailed_content_hashes")
              or extra.get("entailed_raw_uris") or extra.get("proof_surface_tokens")):
            failures.append(f"replay_rows:{index}:invalid_abstention")
    try:
        report = json.loads((out_dir / "replay_report.json").read_text())
        if report.get("status") != "PASS" or not all((report.get("checks") or {}).values()):
            failures.append("report:not_pass")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"report:{exc}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "artifact": str(out_dir),
        "sources": len(manifest.get("sources") or []),
        "rows": len(replay_rows),
        "failures": failures,
    }


def _input_paths(values: list[str], filename: str) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        paths.append(path / filename if path.is_dir() else path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing replay sources: " + ", ".join(missing))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--filename", default="eidetic-plus-full__run0.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--verify-artifact")
    parser.add_argument("--project-guards", action="store_true",
                        help="emit the mechanical guard-policy projection instead of the "
                             "phase-a replay")
    args = parser.parse_args()
    if args.verify_artifact:
        result = verify_replay_artifact(Path(args.verify_artifact))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if not args.sources or not args.out:
        parser.error("sources and --out are required unless --verify-artifact is used")
    if args.project_guards:
        manifest = write_guard_projection_artifact(
            _input_paths(args.sources, args.filename),
            Path(args.out),
        )
        print(json.dumps({
            "status": manifest["status"],
            "sources": len(manifest["sources"]),
            "manifest": str(Path(args.out) / "guard_projection_manifest.json"),
        }, sort_keys=True))
        return 0 if manifest["status"] == "PASS" else 2
    manifest = write_replay_artifact(
        _input_paths(args.sources, args.filename),
        Path(args.out),
    )
    print(json.dumps({
        "status": manifest["status"],
        "sources": len(manifest["sources"]),
        "rows": sum(source["rows"] for source in manifest["sources"]),
        "manifest": str(Path(args.out) / "replay_manifest.json"),
    }, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
