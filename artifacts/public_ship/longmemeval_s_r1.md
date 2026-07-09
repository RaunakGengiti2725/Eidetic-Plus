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
