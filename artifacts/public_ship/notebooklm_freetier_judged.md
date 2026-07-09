# Free-tier NotebookLM recall — JUDGE-SCORED, three disjoint windows

Pinned qwen3-max judge (same judge as every scoreboard row). Windows drawn from the
one-time rotation ledger; **r15 was drawn AFTER all NotebookLM code existed — fully
prospective, no possibility of tuning to the window.**

| window | eidetic+NotebookLM free tier | rag-vector (same window/judge) | eidetic fixed-reader (same window/judge) |
|---|---|---|---|
| r14 | **31/40 = 77.5%** | 26/40 = 65.0% | 22/40 = 55.0% |
| r13 | **37/40 = 92.5%** | 25/40 = 62.5% | 26/40 = 65.0% |
| r15 (prospective) | **34/40 = 85.0%** | 23/40 = 57.5% | 22/40 = 55.0% |
| **all three** | **102/120 = 85.0%** | 74/120 = 61.7% | 70/120 = 58.3% |

**Caller cost of every free-tier answer: 0 tokens** (Gemini free read — collections ran
with no DASHSCOPE key in the shell). Provenance across all three windows: **222/222 cited
eidetic tokens resolve to immutable records by content hash; 0 quoted spans unmatched**
(deterministic grounding, ~690 quotes checked). 120/120 questions answered, 0 errors.

Sidecars: `artifacts/holdout_rotation_r1{3,4,5}_codex/notebooklm_freetier.judged.json`.

## What this is
The PRODUCT configuration measured honestly: eidetic's verified claim graph + provenance-
stamped records exported to NotebookLM, read by Gemini's free tier, every citation
verifiable against the immutable store. +23.3 points over rag-vector across 120 judged
held-out questions at zero caller token cost.

## Honest boundaries (unchanged, non-negotiable)
- **Different reader** (Gemini) than the fixed-qwen benchmark table — this is its own
  labeled product row, never merged into the fixed-reader comparison.
- Three windows, ONE run each — **NOT the ≥10-run reproduce gate**; variance across
  repeated runs is unmeasured; **no SOTA / "best in world" claim**.
- Gemini-side answers are provenance-mapped + lexically grounded, NOT verify-or-abstain
  gate-verified. eidetic `recall` remains the gate-verified path.
- Free tier: Google spends the compute (not free globally); subject to Google quotas/ToS.
