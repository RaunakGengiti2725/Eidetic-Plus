"""MIRIX-style role-specialized memory types (PDF Theme 2c; MIRIX 85.4% LoCoMo, +24pt multi-hop).

MIRIX splits memory into six structured types coordinated by a controller. Eidetic-Plus already
realizes several implicitly (episodic = raw records, semantic = consolidated gists, core =
preferences, resource = non-text modalities). This makes the typing EXPLICIT with a
deterministic classifier so the coordinator can route retrieval/consolidation by type.

The classifier is pure and offline-testable. An optional LLM typing pass (more accurate) is
left as a gated upgrade; the deterministic classifier is the default and the test target.
"""
from __future__ import annotations

import re
from enum import Enum


class MemoryType(str, Enum):
    CORE = "core"                     # identity / persona / standing preferences
    EPISODIC = "episodic"             # time-stamped experiences (raw turns)
    SEMANTIC = "semantic"             # consolidated general knowledge / facts
    PROCEDURAL = "procedural"         # how-to / steps / instructions
    RESOURCE = "resource"             # documents / images / files
    KNOWLEDGE_VAULT = "knowledge_vault"  # sensitive structured data (ids, keys, credentials)


_RESOURCE_MODALITIES = {"pdf", "image", "audio", "video", "binary"}
_VAULT_RE = re.compile(
    r"\b(password|passcode|api[\s_-]?key|secret\s*key|ssn|social security|credit\s*card|"
    r"account\s*(number|no)|routing\s*number|pin\b|private\s*key|access\s*token)\b", re.I)
_PREF_RE = re.compile(
    r"\b(i (prefer|like|love|hate|enjoy|favou?rite)|my favou?rite|i'?m allergic|"
    r"i (always|usually|never)|please (always|never)|call me)\b", re.I)
_PROC_RE = re.compile(
    r"\b(how to|step\s*\d|first\b.*\bthen\b|procedure|instructions?|recipe|"
    r"to (install|configure|set ?up|deploy|build))\b", re.I)


def classify_memory_type(text: str, *, modality: str = "text", is_preference: bool = False,
                         consolidated: bool = False) -> MemoryType:
    """Route a memory to one of the six MIRIX types from cheap deterministic signals."""
    t = text or ""
    if modality in _RESOURCE_MODALITIES:
        return MemoryType.RESOURCE
    if _VAULT_RE.search(t):
        return MemoryType.KNOWLEDGE_VAULT
    if is_preference or _PREF_RE.search(t):
        return MemoryType.CORE
    if _PROC_RE.search(t):
        return MemoryType.PROCEDURAL
    if consolidated:
        return MemoryType.SEMANTIC
    return MemoryType.EPISODIC


def classify_record(record) -> MemoryType:
    """Convenience over a MemoryRecord."""
    modality = getattr(getattr(record, "modality", None), "value", "text")
    return classify_memory_type(record.text or record.summary or "", modality=modality,
                                consolidated=bool(getattr(record, "consolidated", False)))


# ---- coordinator: which types to prioritize for a query class ---------------
_PRIORITY = {
    "preference": [MemoryType.CORE, MemoryType.SEMANTIC, MemoryType.EPISODIC],
    "procedural": [MemoryType.PROCEDURAL, MemoryType.SEMANTIC, MemoryType.EPISODIC],
    "factual": [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.CORE],
    "temporal": [MemoryType.EPISODIC, MemoryType.SEMANTIC],
    "default": [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.CORE,
                MemoryType.PROCEDURAL, MemoryType.RESOURCE, MemoryType.KNOWLEDGE_VAULT],
}


def type_priority(query_class: str = "default") -> list[MemoryType]:
    """The coordinator's deterministic type-priority order for a query class."""
    return _PRIORITY.get(query_class, _PRIORITY["default"])
