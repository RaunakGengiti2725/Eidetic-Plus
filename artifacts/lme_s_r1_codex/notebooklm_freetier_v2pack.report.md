# NotebookLM free-tier live collection — labeled report

Collection: `artifacts/lme_s_r1_codex/notebooklm_freetier_v2pack.jsonl` — n=30 answered, 0 errors, **caller LLM tokens: 0** (BY CONSTRUCTION -- every read ran on NotebookLM/Gemini free tier; no metered key was set in the environment).

## Preliminary containment (heuristic — NOT the judge)

- rate: **0.467**  (HEURISTIC (prefix-tolerant gold-token containment). NOT the pinned qwen3-max judge; NOT comparable to the benchmark scoreboard; the jsonl is judge-ready for a funded key.)
- by category: {"knowledge-update": "2/5", "multi-session": "2/7", "single-session-assistant": "3/4", "single-session-preference": "0/2", "single-session-user": "4/4", "temporal-reasoning": "3/8"}

## Deterministic grounding

- rows with any unmatched (fabricated/altered) quote: **0**
- quotes: {"verbatim": 64, "high_overlap": 79, "unmatched": 0}
- mean answer token coverage: 0.94
- deterministic lexical check vs exported bytes; NOT NLI, NOT the gate

## Citation confirmation

- cited 0, confirmed-in-eidetic 0
- confirmed = resolves to a real immutable record by content hash. Packed raw-record sources are often cited by BODY text (no header token in the quote), so quote-grounding above is the provenance signal for those rows.

Latency: median 18.1s, p90 36.3s.

## Honest boundaries

- single window, single run -- NOT a multi-run gate; no SOTA/best claim
- Gemini-side answers -- NOT eidetic verify-or-abstain
- NOT a row in the fixed-qwen-reader benchmark table
