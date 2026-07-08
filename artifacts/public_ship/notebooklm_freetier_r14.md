# NotebookLM free-tier live collection — labeled report

Collection: `artifacts/holdout_rotation_r14_codex/notebooklm_freetier.jsonl` — n=40 answered, 0 errors, **caller LLM tokens: 0** (BY CONSTRUCTION -- every read ran on NotebookLM/Gemini free tier; no metered key was set in the environment).

## Preliminary containment (heuristic — NOT the judge)

- rate: **0.675**  (HEURISTIC (prefix-tolerant gold-token containment). NOT the pinned qwen3-max judge; NOT comparable to the benchmark scoreboard; the jsonl is judge-ready for a funded key.)
- by category: {"multi-hop": "5/7", "open-domain": "1/3", "single-hop": "18/22", "temporal": "3/8"}

## Deterministic grounding

- rows with any unmatched (fabricated/altered) quote: **0**
- quotes: {"verbatim": 0, "high_overlap": 215, "unmatched": 0}
- mean answer token coverage: 0.785
- deterministic lexical check vs exported bytes; NOT NLI, NOT the gate

## Citation confirmation

- cited 82, confirmed-in-eidetic 82
- confirmed = resolves to a real immutable record by content hash. Packed raw-record sources are often cited by BODY text (no header token in the quote), so quote-grounding above is the provenance signal for those rows.

Latency: median 15.3s, p90 23.4s.

## Honest boundaries

- single window, single run -- NOT a multi-run gate; no SOTA/best claim
- Gemini-side answers -- NOT eidetic verify-or-abstain
- NOT a row in the fixed-qwen-reader benchmark table
