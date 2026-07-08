# NotebookLM bridge — free reads + provenance

`eidetic/integrations/notebooklm.py` connects eidetic's verified memory to Google
NotebookLM. Two modes, one idea: **eidetic supplies the trustworthy, provenance-tagged
sources; NotebookLM (Gemini) does the reading — on its free tier, that read costs zero
of *your* metered LLM tokens.**

## Why this is the token-efficient path

When you query a NotebookLM notebook, Google's Gemini generates the answer over your
uploaded sources. On a personal NotebookLM account that generation is **free** — it
does not draw down your DashScope / OpenAI / Anthropic keys. So an agent that answers
recall queries *through* NotebookLM pays **0 user LLM tokens per read**, while eidetic
keeps the part no one else has: every source carries a provenance header (content
hash + validity window + verified claims), and the answer's `eidetic:<id>` references
map back to immutable records.

Net: **free reads + provenance** — a genuinely cheap, trustworthy memory agent.

## The two modes

**1. Export (`export_namespace`)** — push a namespace's active memories into a notebook
as sources, each stamped with its eidetic provenance header:

```bash
# preview what would be exported (offline, no auth):
python -m eidetic.integrations.notebooklm preview --namespace raunak-main --limit 5

# push into a notebook (needs auth, see below):
python -m eidetic.integrations.notebooklm export \
    --namespace raunak-main --notebook-id <NOTEBOOK_ID> --backend cli
```

**2. Reader mode (`NotebookLMBridge.answer`)** — the 0-token recall path:

```python
from eidetic.integrations.notebooklm import NotebookLMBridge, CliBackend
bridge = NotebookLMBridge(engine, CliBackend())
out = bridge.answer("raunak-main", "Where did Priya move?", notebook_id)
# -> {"answer": ..., "provenance": [{content_sha256, valid_at, memory_id}...],
#     "user_llm_tokens": 0, "caveat": "..."}
```

## Auth (you provide credentials; nothing is hardcoded)

| backend | auth | account | stability |
|---|---|---|---|
| `enterprise` | GCP bearer token `NOTEBOOKLM_ACCESS_TOKEN` (`gcloud auth print-access-token`) + `NOTEBOOKLM_PROJECT_NUMBER` | NotebookLM **Enterprise** license (paid) | official, stable API |
| `cli` | the `nlm` tool ([notebooklm-mcp-cli](https://github.com/jacob-bd/notebooklm-mcp-cli)), browser-cookie login | **personal** Google account (free tier works) | undocumented internal APIs — may break; ToS-gray |

The free-read trick uses the `cli` backend (personal free account). The code never
stores or transmits your credentials anywhere except the backend you choose.

## Honest boundaries (so nobody over-claims it)

- **`user_llm_tokens: 0` means zero on *your* metered model** — Google still spends
  compute; it is free *to you* on the personal tier, not free globally, and subject to
  Google's quotas and Terms of Service.
- **A NotebookLM answer is Gemini-side and is NOT run through eidetic's verify-or-abstain
  proof gate.** It is a free, provenance-mapped answer, not a gate-verified one. For a
  cited, abstain-when-unsure answer, call `engine.recall()` — that path is what the
  benchmark measures.
- **This is a product-cost win, not a benchmark-table row.** The rotating-holdout
  benchmark compares every system through one fixed qwen reader so it measures *memory*,
  not *answerer*. NotebookLM is a different (Gemini) reader whose tokens are off that
  meter, so it cannot be an apples-to-apples line in that table. We report the free-read
  advantage as a product mode, separately and labeled.
- The `cli` backend depends on an unofficial tool; treat it as best-effort. The
  `enterprise` backend is the stable, supported path.
