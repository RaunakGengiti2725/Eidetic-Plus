"""The ONE fixed answerer shared by every adapter, so the scoreboard measures MEMORY
quality (what each system retrieves), not answerer quality. Each system retrieves its own
context; this single reader model + the fixed reader prompt turn that context into an
answer identically for all three. (Eidetic-Plus's cascade/cache remain in the product and
are reflected in the cost/latency tables, but the accuracy comparison pins one reader.)
"""
from __future__ import annotations

import os

from eidetic.config import get_settings
from eidetic.dashscope_client import get_client

from .judge import FIXED_READER_PHOTOGRAPHIC_PROMPT, FIXED_READER_PROMPT

# Pin one reader model across all systems (override with READER_MODEL; pin a snapshot).
READER_MODEL = os.environ.get("READER_MODEL", "").strip() or "qwen-plus"

# READER_MODE selects the shared answer prompt. "default" (the default) keeps FIXED_READER_PROMPT
# byte-identical; "photographic"/"extractive" switches to the verbatim-quoting prompt. Applied to
# every system (shared reader), so the comparison stays fair.
READER_MODE = os.environ.get("READER_MODE", "default").strip().lower()
_READER_PROMPT = (FIXED_READER_PHOTOGRAPHIC_PROMPT
                  if READER_MODE in ("photographic", "extractive") else FIXED_READER_PROMPT)

# Per-block char cap fed to the reader. Default 3000 = byte-identical to the historical harness;
# raise (e.g. 8000) so a retrieved session whose key fact sits past char 3000 reaches the reader.
# Applied EQUALLY to every system (it lives in the shared fixed reader), so the comparison stays
# fair -- baselines benefit from the larger window too.
READER_BLOCK_CHARS = int(os.environ.get("READER_BLOCK_CHARS", "3000"))

FIXED_READER_COT_PROMPT = (
    _READER_PROMPT
    + "\n\nBefore answering, write brief evidence notes for each useful source. "
      "Reply ONLY as JSON: {\"notes\":[{\"source\":\"S0\",\"relevant\":true,"
      "\"note\":\"...\"}],\"answer\":\"...\"}. The answer must contain only the final "
      "answer text with source citations. If the context does not contain the answer, "
      "set answer to \"I do not have that in memory.\""
)


def answer_with_fixed_reader(question: str, context_blocks: list[str]) -> str:
    client = get_client()
    ctx = "\n\n".join(f"[S{i}] {b[:READER_BLOCK_CHARS]}" for i, b in enumerate(context_blocks))
    if get_settings().reader_cot_enabled:
        data = client.chat_json(READER_MODEL, FIXED_READER_COT_PROMPT,
                                f"Question: {question}\n\nMemory/context:\n{ctx}",
                                temperature=0.1, max_tokens=1536)
        answer = data.get("answer") if isinstance(data, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            from eidetic.dashscope_client import ModelCallError
            raise ModelCallError("Fixed reader COT response did not include a non-empty JSON answer.")
        return answer.strip()
    return client.chat(READER_MODEL, _READER_PROMPT,
                       f"Question: {question}\n\nMemory/context:\n{ctx}",
                       temperature=0.1, max_tokens=512)
