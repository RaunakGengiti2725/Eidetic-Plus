# Billable caller-token cost — billable tokens on the OPERATOR'S OWN metered LLM, per query

Windows: holdout_rotation_r9_codex, holdout_rotation_r10_codex, holdout_rotation_r11_codex, holdout_rotation_r12_codex, holdout_rotation_r13_codex, holdout_rotation_r14_codex

| system | caller tokens / query | basis | verified |
|---|---|---|---|
| eidetic+notebooklm (routed, free-read tier) | 0 | BY CONSTRUCTION -- the read runs on NotebookLM/Gemini, off the caller's meter; routed_answer reports user_llm_tokens=0 on that tier | provenance-mapped + deterministic grounding check (Gemini-side, NOT gate-verified) |
| eidetic+notebooklm (routed, structured tier) | median 20.5, max 146.0 (n=78) | MEASURED -- query_tokens of the smqe-answered rows in the same committed holdout logs (the rows Tier 1 takes) | gate-verified (verify-or-abstain) |
| mem0 | 381.0 | MEASURED (qwen reader tokens, committed logs) | 0 verified answers |
| rag-vector | 1892.0 | MEASURED (qwen reader tokens, committed logs) | 0 verified answers |
| eidetic (metered reader, no notebooklm) | 4030.5 | MEASURED (qwen reader tokens, committed logs) | gate-verified |

## Honest claim

Under free-read routing (require_gate_verification=False) a query costs the caller EITHER a structured gate-verified answer -- MEASURED median 20.5, worst-case 146.0 tokens on the n=78 smqe-answered rows of these windows -- OR a NotebookLM free read at 0 caller tokens (by construction). Both are below mem0's measured median ~381.0 and rag-vector's ~1892.0. Honest scope: the tier MIX on an arbitrary query stream is unmeasured (per-tier costs only, no blended figure); this is an operator-cost property, NOT free globally; the free-read answer is Gemini-side provenance-mapped + deterministically grounded (lexical), NOT gate-verified; and this is NOT a row in the fixed-qwen benchmark accuracy table. Accuracy on the free-read tier is unmeasured.
