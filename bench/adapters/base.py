"""The system-under-test interface. Every system (Eidetic-Plus, Mem0, Graphiti) conforms
to this so the harness drives all three IDENTICALLY -- same ingest loop, same answerer
call, same judge. That "by construction" identity is what makes the comparison neutral.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


def approx_tokens(text: str) -> int:
    """Uniform token estimate (~4 chars/token) applied IDENTICALLY to all three systems,
    so tokens/write and tokens/query are an apples-to-apples comparison of how much text
    each system feeds the models -- not a vendor-reported figure."""
    return max(0, len(text or "") // 4)


@dataclass
class WriteResult:
    tokens: int = 0          # approx tokens this system spent ingesting (context + prompts)
    ms: float = 0.0          # wall-clock for the write/ingest


@dataclass
class AnswerResult:
    answer: str = ""
    context_tokens: int = 0  # approx tokens of retrieved context fed to the reader
    search_ms: float = 0.0   # retrieval latency (before the answer LLM)
    e2e_ms: float = 0.0      # end-to-end latency including the answer LLM
    abstained: bool = False
    extra: dict = field(default_factory=dict)


class MemorySystem(ABC):
    """A memory backend under test. Implementations MUST be real (no mocks) and fail loud
    when a dependency or key is missing."""

    name: str = "system"

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Prepare a fresh, isolated scope for one conversation/question set."""

    @abstractmethod
    def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                       session_time: Optional[float] = None) -> WriteResult:
        """Ingest one session's turns. turns = [{'role','content','timestamp'?}]."""

    def consolidate(self, namespace: str) -> None:
        """Optional async/offline build step (graph, facts). Default no-op."""
        return None

    @abstractmethod
    def answer(self, namespace: str, question: str,
               as_of: Optional[float] = None) -> AnswerResult:
        """Answer using only the memory in `namespace` (the one fixed reader path)."""

    def teardown(self) -> None:
        return None
