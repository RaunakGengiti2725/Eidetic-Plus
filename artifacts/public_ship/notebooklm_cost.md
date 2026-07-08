# Billable caller-token cost — billable tokens on the OPERATOR'S OWN metered LLM, per query

Windows: holdout_rotation_r9_codex, holdout_rotation_r10_codex, holdout_rotation_r12_codex, holdout_rotation_r13_codex, holdout_rotation_r14_codex

| system | caller tokens / query | basis | verified |
|---|---|---|---|
| eidetic+notebooklm (routed, free-read tier) | 0 | BY CONSTRUCTION -- the read runs on NotebookLM/Gemini, off the caller's meter; routed_answer reports user_llm_tokens=0 on that tier | provenance-mapped (Gemini-side, NOT gate-verified) |
| eidetic+notebooklm (routed, structured tier) | 6-85 | structured_recall typed path (design-supplied range) | gate-verified (verify-or-abstain) |
| mem0 | 381.0 | MEASURED (qwen reader tokens, committed logs) | 0 verified answers |
| rag-vector | 1880.0 | MEASURED (qwen reader tokens, committed logs) | 0 verified answers |
| eidetic (metered reader, no notebooklm) | 4030.0 | MEASURED (qwen reader tokens, committed logs) | gate-verified |

## Honest claim

On billable tokens spent on the operator's own metered model, the NotebookLM free-read tier costs 0 per query -- below rag-vector's measured ~1880.0 and mem0's ~381.0 -- because Google's free tier does the read. This is an operator-cost property (0 on YOUR meter), by construction; it is NOT free globally, the NotebookLM answer is Gemini-side provenance-mapped (not gate-verified), and it is NOT a row in the fixed-qwen benchmark accuracy table. Run bench/notebooklm_costbench live to prove the end-to-end head-to-head with your own Google account.
