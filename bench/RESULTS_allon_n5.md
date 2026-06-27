# All-architecture run — LoCoMo c0, n=5 (real numbers)

**Config:** maximal valid flag bundle (`/tmp/allon.env`) — 55 of 60 boolean flags ON + photographic
reader + READER_BLOCK_CHARS=8000 + FULL_SLEEP + all retrieval channels + CoVe + span-NLI + abstention-v2
+ reflex/flow/cascade. Single run, LoCoMo conversation `c0`, first 5 questions, `--split all`,
judge `qwen3-max`, reader `qwen-plus`. Date 2026-06-26.

## Overall accuracy (correct / scored)

| System | Correct/Scored | % | Tokens/query | Notes |
|--------|----------------|---|--------------|-------|
| **eidetic-plus-full** (all-on, fixed-reader + NLI verify) | **5/5** | **100%** | 3547 | photographic reader applies here |
| rag-full (stuff everything) | 4/5 | 80% | 15511 | no retrieval, no verify |
| eidetic-product (all-on, engine.ask: reflex/flow/cascade/CoVe/span) | 3/4 | 75% | 3206 | 1 transport error (q4); engine.ask reader, NOT photographic |
| rag-vector (chunk+embed+top-k) | 3/5 | 60% | 1842 | no graph, no verify |

## Per category (n=5)

| Category (n) | eidetic-plus-full | eidetic-product | rag-full | rag-vector |
|--------------|-------------------|-----------------|----------|------------|
| multi-hop (2) | 100% | 100% | 100% | 50% |
| temporal (2) | 100% | 50% | 50% | 50% |
| open-domain (1) | 100% | 100% | 100% | 100% |

## Integrity (verified recall — the moat)

| System | Verified accuracy (/n) | Unproven-answer rate | Abstention rate |
|--------|------------------------|----------------------|-----------------|
| eidetic-plus-full | 80% (4/5 proven) | 20% | 0% |
| eidetic-product | 75% (3/4) | 0% | 0% |
| rag-full | N/A (no verify step) | N/A | 0% |
| rag-vector | N/A (no verify step) | N/A | 0% |

## Honest caveats — READ THESE

- **n=5 is NOT significant.** One question swings it 20pp. The documented larger slice (n=40, same c0)
  is **60–65%** for eidetic-full — do NOT read this 100% as a robust score. It is a working
  end-to-end run on a tiny slice, not proof of "best agent."
- **"Every single bit of architecture" is not literally achievable** — by the code's own design:
  - `DEBATE`, `MEMORY_MANAGER` **raise FeatureNotImplementedError** when enabled → would crash `ask()`.
  - `DREAM_REPAIR`, `DREAM_REPAIR_APPLY`, `DREAM_USE_LLM_NLI` make consolidation **O(corpus × LLM)** —
    1.5h elapsed with **zero answers** before they were excluded. Impractical at any scale (which is
    why they are gated default-off). The other **55 flags ran**.
- **No single row uses all architecture.** Photographic/extractive reader + READER_BLOCK_CHARS only
  bind on the neutral fixed-reader rows (eidetic-plus-full). reflex/flow/cascade/CoVe/span bind on the
  product `engine.ask` row (eidetic-product). Observable: on the temporal q0, eidetic-plus-full
  (photographic reader → absolute date) was correct while eidetic-product (engine.ask reader → answered
  "yesterday") was wrong. Run both rows to cover everything.
- **1 transport error** on eidetic-product q4 (empty prediction) — flagged and excluded, not counted
  as wrong (n scored = 4).
- All-flags-on is **not the optimal config** (the plan rejects it: lost-in-the-middle). This run honors
  the "use all architecture" request; it is not a tuning recommendation.
