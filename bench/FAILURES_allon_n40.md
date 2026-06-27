# Failure forensics — all-architecture run, LoCoMo c0, n=40

eidetic-plus-full scored **28/39 (71.8%)**. This maps **every one of the 12 losses** to a root
cause, so we know exactly where to spend effort. Source:
[artifacts/bench_allon_n40/eidetic-plus-full__run0.jsonl].

## The headline: we are losing at the READER, not the memory

`verified=True` on a wrong answer means an NLI-entailed citation supported it — i.e. **the correct
evidence WAS retrieved**, and the reader still produced the wrong final string. 5 of 11 real losses
are like this. The memory is mostly doing its job; the answer-formatting layer is the weak point.

| Bucket | # | Failing IDs | Memory or reader? |
|--------|---|-------------|-------------------|
| Reader: wrong selection / over-generation / format | 5 | q5, q23, q24, q34, q37 | **reader** (evidence retrieved, verified=True) |
| Inference refusal (open-domain needs a leap) | 3 | q22, q30, (q27 erred) | **policy** (integrity discipline too strict) |
| Retrieval miss (fact never surfaced) | 1–2 | q26, (q23 partial) | **memory** |
| Wrongful abstention (uncalibrated τ) | 1 | q10 | **policy** (abstention-v2 misfired) |
| Off-by-one date arithmetic | 1 | q16 | reader/event interval |
| Transport/API error (not a model failure) | 1 | q27 | infra |

## Per-failure detail

### Reader errors — the right fact was retrieved, the answer was still wrong (5)
| ID | Question | Gold | We said | Root cause |
|----|----------|------|---------|-----------|
| q5 | charity race when? | **Sunday** before 25 May | "last **Saturday** [2023-05-25]" | relative-date: off-by-one weekday + gave absolute where gold is relative |
| q23 | books Melanie read? | "Nothing is Impossible", "Charlotte's Web" | only "Charlotte's Web" | incomplete list — 1 of 2 (2nd title not surfaced/selected) |
| q24 | destress activities? | Running, pottery | "runs, paints, reads, plays violin" | over-generation + **missed pottery** |
| q34 | events to help children? | mentoring, school speech | "pride event, mentorship, ..." | wrong inclusion (pride event) + missed "school speech" |
| q37 | painted **recently**? | sunset | "a horse" | wrong fact among many paintings — "recent" not disambiguated to latest |

### Inference refusals — open-domain questions that REQUIRE a leap (3)
| ID | Question | Gold | We said | Root cause |
|----|----------|------|---------|-----------|
| q22 | Dr. Seuss on her shelf? | Yes (collects classic children's books) | described collection, never said "yes" | won't make world-knowledge leap (Dr. Seuss = classic child book) |
| q30 | Melanie LGBTQ member? | Likely no (never self-refers) | "I do not have that in memory" | won't make **negative** inference from absence |
| q27 | pursue writing career? | Likely no | (empty — API error) | infra error, not the model |

### Memory / policy losses (3)
| ID | Question | Gold | We said | Root cause |
|----|----------|------|---------|-----------|
| q26 | when read "Nothing is Impossible"? | 2022 | "I do not have that in memory" | **retrieval miss** — book+date buried, never surfaced (same fact as q23) |
| q10 | how long has Caroline had friends? | 4 years | abstained | **wrongful abstention** — uncalibrated abstention-v2 τ suppressed an answerable Q |
| q16 | pottery signup when? | 2 July 2023 | 2023-07-**03** | off-by-one day in event-interval arithmetic |

## Why we are not at 100% — and what is actually fixable

**The honest ceiling first:** open-ended QA graded by an LLM judge **cannot** hit 100%. Judges
disagree, gold is sometimes ambiguous (q24's "pottery" vs our "running+painting" both defensible),
and paraphrase/over-listing get marked wrong. Chasing literal 100% judge accuracy is the wrong goal.
The right goals: (a) win every loss that is a real defect, (b) keep 100% *integrity* (prove or abstain).

**Ranked by expected accuracy lift (where to spend effort):**

1. **Reader selection + format (≈5 questions, biggest lever).** The evidence is retrieved; the reader
   mis-selects or mis-formats. Three concrete sub-fixes:
   - *Relative-date answers* (q5, q31-class): when gold is "the week before X" / "Sunday before X",
     the reader must answer in that relative frame, not an absolute date. The photographic reader
     forces absolute dates — that HELPS exact-date questions but HURTS relative ones. Needs a
     relative-expression mode keyed off the question phrasing.
   - *List exactly what is asked* (q24, q34): stop over-generating; answer the minimal set the gold
     wants. "What does X do to destress" → only the destress activities, not every hobby.
   - *Disambiguate "recent"/"current"* (q37): "painted recently" must pick the **latest** painting —
     this is exactly what `CONFLICT_RESOLVER`/event-recency is for; it did not fire here. Verify it
     is actually engaging on superlative/recency questions.

2. **Open-domain inference policy (≈3 questions).** Our integrity discipline ("never state what the
   sources don't say") backfires when the question REQUIRES a parametric or negative inference
   (q22 "yes", q30 "likely no"). We refuse and lose. Needs a *gated* open-domain mode: when the
   question is world-knowledge/opinion and memory supplies the premise, allow the reasoned leap with
   a hedge ("likely…"). This trades a little integrity purity for real open-domain accuracy — apply
   it to ALL systems for fairness.

3. **Calibrate abstention (≈1 question, free).** q10 was answerable; uncalibrated abstention-v2 τ
   killed it. Run `bench.calibrate` on dev to set τ; all-on shipped the gate at its default. Expect
   the wrongful-abstention rate to drop toward 0.

4. **Buried-fact retrieval (≈1–2 questions).** q26/q23's "Nothing is Impossible / 2022" was never
   surfaced despite EXTRACT_CHUNKING — the title sits deep in a long session. Needs turn-level or
   gist indexing of that span (INGEST_GRANULARITY=hybrid or a working GIST_CHANNEL), then re-measure.

5. **Off-by-one dates (≈1).** q16 (July 3 vs 2). Event-interval day rounding; tolerable but tightening
   `_event_epochs` day math would catch it.

## The strategic read

- ~5 of 11 losses are **reader**, ~4 are **policy** (refusal/abstention), ~2 are **memory**.
  **The memory layer is the strongest part; the reader and the answer-policy are where we bleed.**
- That is good news: the reader is a prompt/selection problem (cheap to iterate) and the policy is a
  calibration problem (cheap), versus rebuilding retrieval (expensive). The highest-leverage next
  move is a **smarter answer layer**, not more memory channels.
- It also explains why "all channels on" only got us to 71.8%: piling on retrieval channels does not
  fix a reader that mis-selects from already-good evidence. The plan's "all-on is not optimal"
  prediction holds — the bottleneck moved downstream to the reader.
