# LongMemEval-S measurement — the honest "beat #1" gate (2026-07-09)

Chronos (PwC, arXiv 2603.16862) reports **95.6% on LongMemEval-S** — the number this goal
targets. Every prior eidetic number was LoCoMo. This is the first eidetic run on LongMemEval-S:
never-touched ledgered slice (window 0, digest 988f7ba6, n=30), shared qwen-plus reader +
qwen3-max judge (inline), fresh data dir.

| system | LME-S accuracy (judged) | median caller-tokens/q |
|---|---|---|
| rag-vector | **19/30 = 63.3%** | 1941 |
| eidetic fixed-reader (eidetic-plus-full) | **16/30 = 53.3%** | 6096 |
| Chronos (PwC, **reported, NOT reproduced**) | 95.6% | its own agentic reader (Cohere rerank + multi-LLM + ReAct); no code |

## The honest reading
- **On the target benchmark, the eidetic fixed-reader LOSES to vector RAG on BOTH accuracy
  (53.3% vs 63.3%) and cost (6096 vs 1941 tokens).** The LoCoMo finding replicates exactly:
  the neutral fixed-reader path is not an accuracy or cost winner.
- **We are ~42 points below Chronos's reported 95.6%**, and Chronos is unreproducible
  head-to-head (no code; its number comes from its own agentic reader, not the shared qwen
  reader). "Beat #1" is not close on measured evidence.
- The failure is broad, not one lever: multi-session aggregation, temporal reasoning, recall
  gaps (honest abstentions on answerable questions), and a verified-precision hole
  (10/30 verified-wrong).

## What is still true (and measured)
- eidetic's **cost** win is real ONLY on its non-fixed-reader paths: the structured tier
  (~20-146 tokens, measured on LoCoMo) and the NotebookLM free read (0 caller tokens). The
  metered fixed-reader is the expensive path and is not the product story.
- eidetic's **uniqueness** — content-hash-verifiable provenance + qwen post-hoc audit — stands.
- The only realistic path to a COMPETITIVE LongMemEval-S accuracy number is the NotebookLM
  free-read config (85% vs 55% fixed-reader on LoCoMo), which is quota-gated (task #37) and,
  honestly, would still likely land below 95.6%.

## Bottom line
This measurement is the point of measuring: it refutes a "most powerful / beats #1" claim.
On LongMemEval-S the current eidetic path is **53.3%, below rag-vector, far below Chronos.**
No SOTA claim is made or supported. The defensible, measured claims remain cost (on the free/
structured paths) and verifiable provenance — not accuracy leadership.

## Update — NotebookLM free-read on LME-S = 25.0% (judged), the export bug (2026-07-09)

The free-read config (85% on LoCoMo, the hoped competitive path) was run on the SAME LME-S
slice and judged by the pinned qwen3-max judge: **7/28 = 25.0%** (2 nlm errors). Complete
honest LME-S table:

| path | LME-S accuracy | note |
|---|---|---|
| Chronos (PwC) | 95.6% | reported, unreproduced, own agentic reader, no code |
| rag-vector | 63.3% | shared qwen reader |
| eidetic fixed-reader | 53.3% | shared qwen reader |
| **eidetic NotebookLM free-read** | **25.0%** | 0 caller tokens, but the long-session export bug |

**Every eidetic path loses to rag-vector on LME-S; the free-read collapses from its LoCoMo
85%.** Root cause (read-only store check): the answer facts ARE in the store but buried in
long LongMemEval-S records (a fact inside a multi-paragraph "write a blog post about..."
turn), so NotebookLM reads over one big block and answers "no information" (only 3/… cited
tokens confirmed). LoCoMo turns are short, so this only bites long-session benchmarks.

Fix built this session (commit 5e6f23529, flag-gated default-off): `format_source_chunks` +
`NLM_CHUNK_CHARS` split long records into per-chunk sources (provenance on each) so buried
facts surface. UNMEASURED end-to-end — a fresh chunked re-run is task #40. Honest ceiling
even if it works: a different-reader product row, still below Chronos 95.6%.

**Bottom line, unchanged and now complete: "most powerful / beat #1" is refuted on every
measured path.** Defensible measured claims remain cost (free/structured paths) + verifiable
provenance — not accuracy leadership.
