# NotebookLM free-tier live collection — labeled report

Collection: `artifacts/holdout_rotation_r15_codex/notebooklm_freetier_run5.jsonl` — n=40 answered, 0 errors, **caller LLM tokens: 0** (BY CONSTRUCTION -- every read ran on NotebookLM/Gemini free tier; no metered key was set in the environment).

## Preliminary containment (heuristic — NOT the judge)

- rate: **0.575**  (HEURISTIC (prefix-tolerant gold-token containment). NOT the pinned qwen3-max judge; NOT comparable to the benchmark scoreboard; the jsonl is judge-ready for a funded key.)
- by category: {"multi-hop": "2/8", "open-domain": "1/2", "single-hop": "14/21", "temporal": "6/9"}

## Deterministic grounding

- rows with any unmatched (fabricated/altered) quote: **0**
- quotes: {"verbatim": 4, "high_overlap": 299, "unmatched": 0}
- mean answer token coverage: 0.772
- deterministic lexical check vs exported bytes; NOT NLI, NOT the gate

## Citation confirmation

- cited 102, confirmed-in-eidetic 102
- confirmed = resolves to a real immutable record by content hash. Packed raw-record sources are often cited by BODY text (no header token in the quote), so quote-grounding above is the provenance signal for those rows.

Latency: median 15.9s, p90 21.5s.

## Honest boundaries

- single window, single run -- NOT a multi-run gate; no SOTA/best claim
- Gemini-side answers -- NOT eidetic verify-or-abstain
- NOT a row in the fixed-qwen-reader benchmark table
