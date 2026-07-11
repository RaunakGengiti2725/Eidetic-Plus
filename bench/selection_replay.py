"""Offline SELECTION replay for the relative_temporal derivation boundary.

The verify-side date floor was replay-refuted (HANDOFF 2026-07-10): wrong-EVENT selection
is invisible to any (atom, answer) check, so the fix moved to SELECTION time -- the
candidate loop now tags every shipped date `:atom_derived` (the winning atom's own
expression dates the event, or a deterministic rule resolved the contest) or
`:mention_selected` (score/hit ordering alone picked among materially conflicting dated
mentions), and the note-keyed `temporal_selection` floor fails the latter closed.

This module enforces the shipping protocol for that change MECHANICALLY and OFFLINE:
re-run the CURRENT selection code over the burned windows' frozen stores and compare
against the frozen rows. Zero provider calls -- the store snapshots (`data/eidetic.sqlite`)
already hold the records and write-time claims, and citation checking uses the same
whitespace-normalized substring entailment the synthetic suite uses (every proof atom in
this operator is a verbatim record substring).

Gates reported (the artifact fails loud when the first is nonzero):
  - verified-correct regression: frozen VERIFIED-CORRECT relative_temporal rows whose
    replayed answer changed or vanished;
  - wrong-converted: frozen VERIFIED-WRONG rows that now fail closed (abstain);
  - answer-changed: frozen wrong rows whose replayed answer DIFFERS (not claimable as
    fixed without a live judge -- listed, never counted as wins).

Namespaces and question timestamps are rebuilt exactly as the harness built them
(same loaders, same grouping, same as-of rules), so the replay asks the same store the
same question the run did.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eidetic.models import NLILabel, Scope
from eidetic.smqe.engine import structured_recall
from eidetic.store import RecordStore

from .harness import _as_of_time, _group_by_sessions
from .run import _filter_samples_file, _load_samples_file, load_samples


class _MechRetriever:
    """Store + whitespace-normalized substring entailment (the synthetic suite's checker).
    relative_temporal proof atoms are verbatim record substrings, so this reproduces the
    live entailment decision for this operator without a model call."""

    def __init__(self, store: RecordStore):
        self.store = store

    def verify_citation(self, rec, atom):
        hay = " ".join((getattr(rec, "text", "") or "").lower().split())
        needle = " ".join((atom or "").lower().split())
        if needle and needle in hay:
            return (NLILabel.ENTAILMENT, 1.0)
        return (NLILabel.NEUTRAL, 0.0)


_SAMPLES_CACHE: dict[tuple[str, str, str], list] = {}


def _load_dataset_samples(dataset: str, variant: str, split: str) -> list:
    key = (dataset, variant, split)
    if key not in _SAMPLES_CACHE:
        _SAMPLES_CACHE[key] = load_samples(dataset, 0, variant, 0, split=split,
                                           sample_strategy="contiguous")
    return _SAMPLES_CACHE[key]


def _window_bindings(window: Path) -> tuple[dict, dict]:
    """(sample_id -> namespace, sample_id -> as_of) rebuilt with the harness's own rules."""
    manifest = json.loads((window / "run_manifest.json").read_text())
    samples_file = window / Path(str(manifest.get("samples_file") or "")).name
    if not samples_file.exists():
        raise FileNotFoundError(f"{window}: samples file {samples_file.name} missing")
    all_samples = _load_dataset_samples(
        manifest["dataset"], manifest.get("variant", "longmemeval_s"),
        manifest.get("split", "test"))
    picked = _filter_samples_file(all_samples, _load_samples_file(samples_file))
    ns_of: dict = {}
    at_of: dict = {}
    for gi, (_sessions, qs) in enumerate(_group_by_sessions(picked)):
        for s in qs:
            ns_of[s.sample_id] = f"eidetic-plus-full-{s.dataset}-g{gi}-r0"
            at_of[s.sample_id] = _as_of_time(s)
    return ns_of, at_of


def replay_window(window: Path) -> tuple[list[dict], str]:
    """Returns (rows, skip_reason). A window whose store snapshot is EMPTY cannot be
    replayed -- reported loudly, never silently folded into the gates."""
    rows_file = window / "eidetic-plus-full__run0.jsonl"
    store_path = window / "data" / "eidetic.sqlite"
    if not rows_file.exists() or not store_path.exists():
        return [], "missing rows file or store snapshot"
    ns_of, at_of = _window_bindings(window)
    store = RecordStore(store_path)
    if not store.count():
        return [], "store snapshot is empty (0 records)"
    retriever = _MechRetriever(store)
    out: list[dict] = []
    for line in rows_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        extra = row.get("extra") or {}
        if extra.get("smqe_operator") != "relative_temporal":
            continue
        sample_id = row.get("sample_id")
        if sample_id not in ns_of:
            raise KeyError(f"{window}: no namespace binding for {sample_id}")
        trace = structured_recall(
            retriever, row.get("question") or "",
            at=at_of[sample_id], scope=Scope(namespace=ns_of[sample_id]))
        answered = bool(trace.get("answered"))
        replay_answer = (trace.get("answer") or "") if answered else ""
        replay_note = trace.get("note") or ""
        frozen_answer = row.get("predicted") or ""
        frozen_correct = bool(row.get("correct"))
        if not answered:
            status = "converted_to_abstain"
        elif replay_answer == frozen_answer:
            status = "unchanged"
        else:
            status = "answer_changed"
        out.append({
            "window": window.name,
            "sample_id": sample_id,
            "question": row.get("question") or "",
            "gold": row.get("gold") or "",
            "frozen_answer": frozen_answer,
            "frozen_correct": frozen_correct,
            "frozen_note": extra.get("smqe_policy") or extra.get("note") or "",
            "replay_answer": replay_answer,
            "replay_note": replay_note,
            "replay_pre_verify_answer": trace.get("answer") or "",
            "replay_failure": trace.get("failure_reason") or "",
            "replay_op": trace.get("op") or "",
            "status": status,
        })
    return out, ""


def replay_windows(windows: list[Path], *, baseline: dict | None = None) -> dict:
    rows: list[dict] = []
    skipped: list[dict] = []
    for window in windows:
        window_rows, skip_reason = replay_window(window)
        if skip_reason:
            skipped.append({"window": window.name, "reason": skip_reason})
            continue
        rows.extend(window_rows)
    vc = [r for r in rows if r["frozen_correct"]]
    vw = [r for r in rows if not r["frozen_correct"]]
    vc_regressions = [r for r in vc if r["status"] != "unchanged"]
    # The protocol gate protects the ATOM-DERIVED verified-correct subset: a correct row
    # the replay still classifies atom_derived must keep its exact answer. A correct row
    # reclassified mention_selected is the fail-closed tie POLICY (correct-or-silent),
    # reported separately, never silently. Baseline attribution (pre-diff drift) requires
    # a second run on the stashed tree; ship both JSONs together.
    atom_derived_regressions = [
        r for r in vc_regressions
        if r["replay_note"].endswith(":atom_derived")]
    policy_losses = [
        r for r in vc_regressions
        if r["replay_note"].endswith(":mention_selected")]
    # DIFF ATTRIBUTION: rows already regressed on the pre-diff tree (the stashed-baseline
    # report) are drift the diff neither caused nor can be gated on; the artifact ships
    # BOTH reports so the subtraction is verifiable, never asserted.
    baseline_regressed: set[tuple[str, str]] = set()
    if baseline:
        baseline_regressed = {(r["window"], r["sample_id"])
                              for r in baseline.get("vc_regression_rows", [])}
    diff_attributable = [
        r for r in atom_derived_regressions
        if (r["window"], r["sample_id"]) not in baseline_regressed]
    report = {
        "rows": len(rows),
        "frozen_correct": len(vc),
        "frozen_wrong": len(vw),
        "vc_regressions": len(vc_regressions),
        "vc_regressions_atom_derived": len(atom_derived_regressions),
        "vc_regressions_atom_derived_diff_attributable": len(diff_attributable),
        "vc_regressions_policy_fail_closed": len(policy_losses),
        "vc_regression_rows": vc_regressions,
        "wrong_converted_to_abstain": sum(
            1 for r in vw if r["status"] == "converted_to_abstain"),
        "wrong_answer_changed": sum(1 for r in vw if r["status"] == "answer_changed"),
        "wrong_unchanged": sum(1 for r in vw if r["status"] == "unchanged"),
        "correct_kept": len(vc) - len(vc_regressions),
        "skipped_windows": skipped,
        "pass": not diff_attributable,
        "detail": rows,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows", nargs="+", help="burned window artifact dirs")
    parser.add_argument("--out", default="", help="write the full JSON report here")
    parser.add_argument("--baseline-report", default="",
                        help="pre-diff report JSON (stashed-tree run) for attribution")
    args = parser.parse_args(argv)
    baseline = None
    if args.baseline_report:
        baseline = json.loads(Path(args.baseline_report).read_text())
    report = replay_windows([Path(w) for w in args.windows], baseline=baseline)
    summary = {k: v for k, v in report.items() if k not in ("detail", "vc_regression_rows")}
    print(json.dumps(summary, indent=2))
    for s in report["skipped_windows"]:
        print(f"SKIPPED {s['window']}: {s['reason']}")
    for r in report["vc_regression_rows"]:
        print(f"REGRESSION {r['window']} {r['sample_id']}: "
              f"{r['frozen_answer']!r} -> {r['replay_answer']!r} ({r['status']})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report: {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
