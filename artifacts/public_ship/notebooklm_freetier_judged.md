# Free-tier NotebookLM recall — JUDGE-SCORED (pinned qwen3-max, same judge as every scoreboard row)

Two disjoint rotation windows, collected live and key-free, then scored by the SAME pinned
judge the benchmark uses. Different READER than the fixed-qwen table (off-meter Gemini), so
this is its own labeled product row — never merged into the fixed-reader comparison.

| window | eidetic+NotebookLM free tier | rag-vector (same window/judge) | eidetic fixed-reader (same window/judge) |
|---|---|---|---|
| r14 | **31/40 = 77.5%** | 26/40 = 65.0% | 22/40 = 55.0% |
| r13 | **37/40 = 92.5%** | 25/40 = 62.5% | 26/40 = 65.0% |
| **both** | **68/80 = 85.0%** | 51/80 = 63.75% | 48/80 = 60.0% |

Per-category (both windows): single-hop 39/43, multi-hop 12/15, temporal 13/17, open-domain 4/5.

**Caller cost of every free-tier answer: 0 tokens** (Gemini free read; collection ran with
no DASHSCOPE key in the shell). Provenance: **141/141 cited eidetic tokens resolve to
immutable records by content hash; 0/453 quoted spans unmatched** (deterministic grounding).

Sidecars: `artifacts/holdout_rotation_r1{3,4}_codex/notebooklm_freetier.judged.json`.

## Honest boundaries
- Different reader (Gemini) than the fixed-qwen benchmark table — labeled product row only.
- Two windows, ONE run each — NOT the ≥10-run reproduce gate; no SOTA/"best in world" claim.
- Gemini-side answers are provenance-mapped + lexically grounded, NOT verify-or-abstain
  gate-verified. eidetic `recall` remains the gate-verified path.
- Free tier: Google spends the compute; subject to Google quotas/ToS. Not free globally.
