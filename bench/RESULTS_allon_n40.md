# Full-scale all-architecture run — LoCoMo c0, n=40 (real numbers)

**Config:** maximal valid flag bundle — 52 of 60 boolean flags ON (photographic reader,
READER_BLOCK_CHARS=8000, all retrieval channels, CoVe, span-NLI, abstention-v2, reflex/flow/cascade,
FULL_SLEEP). Single run, LoCoMo conversation `c0`, first 40 questions, `--split all`, judge
`qwen3-max`, reader `qwen-plus`. Date 2026-06-26. Excluded by design: DEBATE / MEMORY_MANAGER
(crash stubs), DREAM_REPAIR / DREAM_REPAIR_APPLY / DREAM_USE_LLM_NLI (O(corpus×LLM) — never finished).

## Overall accuracy (correct / scored)

| System | Correct/Scored | Accuracy | Tokens/query |
|--------|----------------|----------|--------------|
| **eidetic-plus-full** (all-on) | **28/39** | **71.8%** | 3476 |
| rag-full (stuff everything) | 23/40 | 57.5% | 15511 |
| rag-vector (chunk+embed+top-k) | 17/40 | 42.5% | 1882 |
| eidetic-product (engine.ask) | _running ~32s/q_ | _pending_ | 3273 |

eidetic-plus-full beats full-context RAG by **+14.3pp** at **~22% of its tokens**, and beats vector
RAG by **+29.3pp**. (1 transport error on eidetic excluded from its denominator.)

## Per category (n=40)

| Category (n) | eidetic-plus-full | rag-full | rag-vector |
|--------------|-------------------|----------|------------|
| temporal (20) | **15/20 (75%)** | 8/20 (40%) | 5/20 (25%) |
| multi-hop (15) | 11/15 (73%) | 11/15 (73%) | 9/15 (60%) |
| open-domain (4–5) | 2/4 (50%) | 4/5 (80%) | 3/5 (60%) |

## Statistical significance (McNemar, paired, single run)

| Comparison (category) | a-only | b-only | McNemar p | Verdict |
|-----------------------|--------|--------|-----------|---------|
| eidetic vs rag-full (temporal) | 8 | 1 | **0.039** | **eidetic wins, significant** |
| eidetic vs rag-vector (temporal) | 11 | 1 | **0.006** | **eidetic wins, significant + CI-clear** |
| eidetic vs rag-full (multi-hop) | 1 | 1 | 1.000 | tie |
| eidetic vs rag-full (open-domain) | 0 | 1 | 1.000 | eidetic lost 1 (parametric-inference refusal) |

## Integrity (verified recall — the moat)

| System | Verified accuracy (/n) | Unproven-answer rate | Abstention rate |
|--------|------------------------|----------------------|-----------------|
| eidetic-plus-full | 53.8% | 30.8% | 2.6% |
| rag-full / rag-vector | N/A (no verify step) | N/A | 0% |

## What this actually says (honest)

- **This is the strongest measured result to date and it is real.** At n=40 on the same c0 slice the
  prior tuned config scored 60–65%; **all-on with the photographic reader + 8k blocks reaches 71.8%**,
  and the temporal win over full-context RAG is **statistically significant (p=0.039)**. The session's
  capture/reader/channel work shows up exactly where predicted (temporal 40% → 75%).
- **Still one run, one conversation.** McNemar per-category is significant, but "slice survival" needs
  ≥2 runs and the headline public claim needs the 10-run `--split test` gate. Do NOT yet say "beats RAG"
  unqualified — say "beats full-context RAG on temporal, p=0.039, single run, c0 n=40."
- **eidetic loses open-domain** (2/4 vs rag-full 4/5) — the documented parametric-inference refusal; it
  won't guess world-knowledge the memory doesn't state. Honest trade, not a bug.
- **Token efficiency is a clean win:** 3476 vs 15511 tok/query — same or better accuracy at ~1/4.5 the
  context cost. This is the LongMemEval thesis in miniature.
- **Not every flag could run.** 2 crash by design, 3 dream-LLM sweeps are O(corpus×LLM) and never
  finished — 52 of 60 flags ran. "Literally every bit" is impossible by the code's own gating.
