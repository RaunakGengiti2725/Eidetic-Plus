"""Build holdout leakage-audit registries from real benchmark sample IDs.

The registry is intentionally small and non-answer-bearing: it records which sample IDs belong to
the release holdout slice, so source audits can fail if those identifiers are copied into code.
Questions, answers, and transcripts stay in the benchmark data or a private samples file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import _filter_samples_file, _load_samples_file, load_samples


REGISTRY_FILES = {
    "longmemeval": "longmemeval_test_holdout.json",
    "locomo": "locomo_test_holdout.json",
}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_requested_samples(
    *,
    dataset: str,
    variant: str,
    split: str,
    samples_file: str,
):
    load_dataset = "both" if dataset == "both" else dataset
    samples = load_samples(load_dataset, 0, variant, 0, split=split, sample_strategy="contiguous")
    if samples_file:
        rows = _load_samples_file(samples_file)
        samples = _filter_samples_file(samples, rows)
    return samples


def build_registry(
    *,
    out_dir: Path,
    dataset: str = "both",
    variant: str = "longmemeval_s",
    split: str = "test",
    samples_file: str = "",
) -> dict:
    samples = _load_requested_samples(
        dataset=dataset,
        variant=variant,
        split=split,
        samples_file=samples_file,
    )
    by_dataset: dict[str, list[str]] = {name: [] for name in REGISTRY_FILES}
    for sample in samples:
        if sample.dataset in by_dataset:
            by_dataset[sample.dataset].append(sample.sample_id)

    requested = set(REGISTRY_FILES) if dataset == "both" else {dataset}
    empty_requested = sorted(name for name in requested if not by_dataset.get(name))
    if empty_requested:
        raise ValueError("no holdout samples selected for: " + ", ".join(empty_requested))

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, filename in REGISTRY_FILES.items():
        ids = sorted(set(by_dataset.get(name, [])))
        counts[name] = len(ids)
        _write_json(out_dir / filename, ids)

    leaked_path = out_dir / "leaked_sample_ids.json"
    if not leaked_path.exists():
        _write_json(leaked_path, [])

    manifest = {
        "version": 1,
        "registry_kind": "holdout_sample_id_needles",
        "description": (
            "Source-audit registry for release-grade memory evaluation. Contains sample IDs "
            "only; keep questions, answers, and transcripts out of this file."
        ),
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "samples_file": samples_file,
        "counts": counts,
        "files": ["leaked_sample_ids.json", *REGISTRY_FILES.values()],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "pass": True,
        "out_dir": str(out_dir),
        "dataset": dataset,
        "split": split,
        "counts": counts,
        "total": sum(counts.values()),
        "samples_file": samples_file,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/bench/holdout")
    ap.add_argument("--dataset", default="both", choices=["longmemeval", "locomo", "both"])
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--samples-file", default="",
                    help="Optional JSON list of {dataset, sample_id}; exact rows become registry IDs.")
    args = ap.parse_args()
    result = build_registry(
        out_dir=Path(args.out),
        dataset=args.dataset,
        variant=args.variant,
        split=args.split,
        samples_file=args.samples_file,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
