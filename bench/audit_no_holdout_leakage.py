"""Fail closed if holdout identifiers or forbidden legacy policies appear in source."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_HOLDOUT_DIR = Path("data/bench/holdout")
DEFAULT_SCAN_ROOTS = ("eidetic", "bench", "tests", "docs")
FORBIDDEN_POLICY_STRINGS = (
    "product-" + "source-scan",
    "long" + "memeval-direct",
    "locomo-" + "direct-fact",
    "open-domain-bridge-" + "source-scan",
    "direct-fact-" + "source-scan",
)
FORBIDDEN_FIXED_ANSWER_STRINGS = (
    "Middle-class" + " or wealthy",
    "Hairless pets" + ", such as hairless cats or pigs",
    "Cook " + "dog treats",
    "battery-saving " + "mode",
    "fully " + "charged",
    "Atmospheric distillation, fluid catalytic cracking" + " (FCC), alkylation, and hydrotreating",
    "Eternal Sunshine of the " + "Spotless Mind",
    "Rev" + "ell",
    "Tam" + "iya",
    "Spit" + "fire",
    "B-" + "29",
    "Cam" + "aro",
)
FORBIDDEN_RUNTIME_SYMBOLS = (
    "_long" + "memeval_source_scan",
    "_locomo" + "_fact_source_scan",
    "_relative_date" + "_source_scan",
    "_answer_from" + "_source_scan",
    "_dataset" + "_source_scans_enabled",
    "_extract_" + "direct_fact_match",
    "_extract_" + "open_domain_bridge_match",
    "_extract_" + "profile_fact_answer",
    "_extract_" + "relationship_status_answer",
    "_extract_" + "user_slot_answer",
    "_verified_" + "direct_citations",
    "_verified_" + "atom_citations",
    "_compact_" + "temporal_slot_answer",
    "_known_title" + "_alias_entailment",
    "_model" + "_kit_count_answer",
    "_SCALE" + "_MODEL_RE",
    "_camera" + "_accessory_answer",
    "_martial" + "_arts_answer",
    "_MARTIAL" + "_ARTS",
    "_plant" + "_count_answer",
    "_PLANT" + "_HINTS",
    "from\\s+" + "(?:my\\s+|the\\s+)?garden",
    '"utensil", "holder", "countertop"' + ', "granite", "sink"',
    (
        '"phone", "battery"'
        + ', "power"'
        + ', "bank"'
        + ', "charging", "charged"'
    ),
    (
        '"paint", "painting"'
        + ', "paintings"'
        + ', "inspiration"'
        + ', "tutorial", "tutorials"'
        + ', "online", "social", "media"'
        + ', "challenge"'
        + ', "flower"'
        + ', "flowers"'
    ),
    (
        '"colleague", "colleagues"'
        + ', "socialize"'
        + ', "coffee"'
        + ', "team"'
        + ', "activities"'
        + ', "groups"'
    ),
    (
        '"enjoy", "writing"'
        + ', "reading"'
        + ', "movies"'
        + ', "nature"'
        + ', "friends"'
    ),
)


def _needle_pos(hay: str, target: str) -> int:
    """First match position, or -1. A needle ending in a digit must not be
    immediately followed by another digit: an ID ending in '_q4' inside a
    different ID ending in '_q42' is decimal-numbering collision, not a leak."""
    if not target or not target[-1].isdigit():
        return hay.find(target)
    start = 0
    while True:
        pos = hay.find(target, start)
        if pos < 0:
            return -1
        end = pos + len(target)
        if end >= len(hay) or not hay[end].isdigit():
            return pos
        start = pos + 1


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _strings_from_obj(obj, *, min_len: int = 8) -> set[str]:
    out: set[str] = set()
    if obj is None:
        return out
    if isinstance(obj, str):
        value = obj.strip()
        if len(value) >= min_len:
            out.add(value)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in {"sample_id", "sample_ids"}:
                out.update(_strings_from_obj(value, min_len=4))
            elif str(key).lower() in {"question", "questions", "answer", "answers", "leaked_strings"}:
                out.update(_strings_from_obj(value, min_len=8))
            else:
                out.update(_strings_from_obj(value, min_len=min_len))
    elif isinstance(obj, list):
        for item in obj:
            out.update(_strings_from_obj(item, min_len=min_len))
    return out


def load_holdout_needles(holdout_dir: Path) -> set[str]:
    needles: set[str] = set()
    for name in (
        "leaked_sample_ids.json",
        "longmemeval_test_holdout.json",
        "locomo_test_holdout.json",
    ):
        needles.update(_strings_from_obj(_load_json(holdout_dir / name), min_len=4))
    manifest = _load_json(holdout_dir / "manifest.json")
    if isinstance(manifest, dict):
        needles.update(_strings_from_obj(manifest.get("sample_ids"), min_len=4))
        for key in ("questions", "answers", "leaked_strings"):
            needles.update(_strings_from_obj(manifest.get(key), min_len=8))
    return {n for n in needles if len(n) >= 4}


def iter_files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".toml"}:
                if any(part in {".git", "__pycache__"} for part in path.parts):
                    continue
                yield path


def audit(
    holdout_dir: Path,
    roots: list[Path],
    *,
    include_legacy_policy: bool = True,
    require_holdout_needles: bool = True,
) -> dict:
    holdout_needles = load_holdout_needles(holdout_dir)
    needles = set(holdout_needles)
    if include_legacy_policy:
        needles.update(FORBIDDEN_POLICY_STRINGS)
        needles.update(FORBIDDEN_FIXED_ANSWER_STRINGS)
        needles.update(FORBIDDEN_RUNTIME_SYMBOLS)
    findings = []
    for path in iter_files(roots):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        low = text.lower()
        for needle in sorted(needles, key=len, reverse=True):
            if not needle:
                continue
            case_insensitive = needle in FORBIDDEN_POLICY_STRINGS or needle in FORBIDDEN_FIXED_ANSWER_STRINGS
            hay = low if case_insensitive else text
            target = needle.lower() if case_insensitive else needle
            pos = _needle_pos(hay, target)
            if pos >= 0:
                line = text[:pos].count("\n") + 1
                findings.append({"path": str(path), "needle": needle, "line": line})
    registry_error = ""
    if require_holdout_needles and not holdout_needles:
        registry_error = "holdout registry is empty"
    return {
        "pass": not findings and not registry_error,
        "findings": findings,
        "needles_checked": len(needles),
        "holdout_needles_checked": len(holdout_needles),
        "legacy_policy_scan_enabled": bool(include_legacy_policy),
        "forbidden_policy_strings_checked": len(FORBIDDEN_POLICY_STRINGS) if include_legacy_policy else 0,
        "forbidden_fixed_answer_strings_checked": len(FORBIDDEN_FIXED_ANSWER_STRINGS) if include_legacy_policy else 0,
        "forbidden_runtime_symbols_checked": len(FORBIDDEN_RUNTIME_SYMBOLS) if include_legacy_policy else 0,
        "scan_roots": [str(root) for root in roots],
        "registry_error": registry_error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR))
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_SCAN_ROOTS))
    ap.add_argument("--allow-legacy-policy-strings", action="store_true")
    ap.add_argument("--allow-empty-holdout-registry", action="store_true")
    args = ap.parse_args()
    result = audit(
        Path(args.holdout_dir),
        [Path(r) for r in args.roots],
        include_legacy_policy=not args.allow_legacy_policy_strings,
        require_holdout_needles=not args.allow_empty_holdout_registry,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
