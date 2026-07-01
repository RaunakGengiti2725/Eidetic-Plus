"""Deterministic fingerprints for benchmark raw-log artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


def log_fingerprint(out_dir: Path) -> dict:
    """Return a stable SHA-256 fingerprint for all benchmark JSONL logs in a directory.

    The release gate uses this to bind rendered reports back to the exact raw logs they
    summarize. Hashing raw bytes, rather than parsed rows, catches append/edit/reorder drift.
    """
    out_dir = Path(out_dir)
    files: list[dict] = []
    combined = hashlib.sha256()
    for path in sorted(out_dir.glob("*__run*.jsonl"), key=lambda p: p.name):
        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        rel = path.name
        files.append({"path": rel, "bytes": len(data), "sha256": file_hash})
        combined.update(rel.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(len(data)).encode("ascii"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\0")
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "combined_sha256": combined.hexdigest(),
        "files": files,
    }
