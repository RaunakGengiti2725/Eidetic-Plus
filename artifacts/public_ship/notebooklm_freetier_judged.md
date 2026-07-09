# Free-tier NotebookLM recall — JUDGE-SCORED, three disjoint windows

Pinned qwen3-max judge (same judge as every scoreboard row). Windows drawn from the
one-time rotation ledger; **r15 was drawn AFTER all NotebookLM code existed — fully
prospective, no possibility of tuning to the window.**

| window | eidetic+NotebookLM free tier | rag-vector (same window/judge) | eidetic fixed-reader (same window/judge) |
|---|---|---|---|
| r14 | **31/40 = 77.5%** | 23/40 = 57.5% | 22/40 = 55.0% |
| r13 | **37/40 = 92.5%** | 25/40 = 62.5% | 26/40 = 65.0% |
| r15 (prospective) | **34/40 = 85.0%** | 23/40 = 57.5% | 22/40 = 55.0% |
| **all three** | **102/120 = 85.0%** | 71/120 = 59.2% | 70/120 = 58.3% |

(Correction 2026-07-08: an earlier draft mis-transcribed r14 rag-vector as 26/40; the
committed jsonl is 23/40. Corrected here — it widens eidetic's lead, i.e. NOT a
self-serving edit. Every cell above is recomputed directly from the committed files.)

**Statistical significance (paired, exact McNemar over the 120 questions):** eidetic+
NotebookLM vs rag-vector — 33 questions only NotebookLM got right, 2 only rag-vector got
right → **p < 1e-5**. vs eidetic-fixed-reader — 36 vs 5 → **p < 1e-5**. The +25.8-point
gap over rag-vector is not noise at this n.

**Caller cost of every free-tier answer: 0 tokens** (Gemini free read — collections ran
with no DASHSCOPE key in the shell). Provenance across all three windows: **222/222 cited
eidetic tokens resolve to immutable records by content hash; 0 quoted spans unmatched**
(deterministic grounding, ~690 quotes checked). 120/120 questions answered, 0 errors.

Sidecars: `artifacts/holdout_rotation_r1{3,4,5}_codex/notebooklm_freetier.judged.json`.

### r15 four-system snapshot (all systems, same window + judge)
| system | r15 accuracy | median caller tokens/q |
|---|---|---|
| **eidetic + NotebookLM (free)** | **34/40 = 85.0%** | **0** |
| rag-vector | 23/40 = 57.5% | 1902 |
| eidetic fixed-reader | 22/40 = 55.0% | 4967 |
| mem0 | 12/40 = 30.0% | 401 |

r15 is the fully-prospective window (drawn after all code existed). On this window the
free-tier product row leads every comparator while spending zero caller tokens. mem0 was
only run on r15 (not r13/r14), so it is a single-window comparator, labeled as such.

### Variance (toward the ≥10-run gate — still partial)
r15 has two runs over identical notebooks: run 1 = 85.0% (40/40 answered), run 2 = 84.8%
(28/33; 7 rows hit the free-tier daily quota, recorded as errors, retry-on-reset). Two
partial runs is NOT the gate; it is an early variance signal (tight so far).

## What this is
The PRODUCT configuration measured honestly: eidetic's verified claim graph + provenance-
stamped records exported to NotebookLM, read by Gemini's free tier, every citation
verifiable against the immutable store. +25.8 points over rag-vector (85.0% vs 59.2%)
across 120 judged held-out questions at zero caller token cost — paired McNemar p < 1e-5.

## Honest boundaries (unchanged, non-negotiable)
- **Different reader** (Gemini) than the fixed-qwen benchmark table — this is its own
  labeled product row, never merged into the fixed-reader comparison.
- Three windows, ONE run each — **NOT the ≥10-run reproduce gate**; variance across
  repeated runs is unmeasured; **no SOTA / "best in world" claim**.
- Gemini-side answers are provenance-mapped + lexically grounded, NOT verify-or-abstain
  gate-verified. eidetic `recall` remains the gate-verified path.
- Free tier: Google spends the compute (not free globally); subject to Google quotas/ToS.
