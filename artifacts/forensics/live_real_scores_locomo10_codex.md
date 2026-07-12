# Benchmark Failure Forensics

Rows inspected: 40
Failures/abstentions/errors: 18

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| retrieval_miss | 15 | raise lexical/graph/event recall depth; inspect candidate_memory_ids and query parsing |
| abstention | 2 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |
| reader_error | 1 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |

## Categories

| category | buckets |
|---|---|
| multi-hop | abstention: 1, retrieval_miss: 1 |
| temporal | abstention: 1, reader_error: 1, retrieval_miss: 14 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 1 | 19 | 75 | 75 | 0 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | temporal | c0_q5 | reader_error | When did Melanie run a charity race? | The sunday before 25 May 2023 | Saturday 2023-05-20 [S0] "I ran a charity race for mental health last Saturday"; session date is 2023-05-25, so 'last Sa |
| eidetic-plus-full | temporal | c0_q9 | abstention | When did Caroline meet up with her friends, family, and mentors? | The week before 9 June 2023 | I don't have enough verified evidence in memory to answer that confidently. |
| mem0 | temporal | c0_q0 | retrieval_miss | When did Caroline go to the LGBTQ support group? | 7 May 2023 | Caroline went to the LGBTQ support group on June 27, 2026 [S1]. |
| mem0 | temporal | c0_q1 | retrieval_miss | When did Melanie paint a sunrise? | 2022 | Melanie painted a sunrise last year (2025) [S1]. |
| mem0 | temporal | c0_q5 | retrieval_miss | When did Melanie run a charity race? | The sunday before 25 May 2023 | Saturday, June 27, 2026 [S0] |
| mem0 | temporal | c0_q6 | retrieval_miss | When is Melanie planning on going camping? | June 2023 | next month (July 2026) [S0] |
| mem0 | multi-hop | c0_q7 | abstention | What is Caroline's relationship status? | Single | I do not have that in memory. |
| mem0 | temporal | c0_q8 | retrieval_miss | When did Caroline give a speech at a school? | The week before 9 June 2023 | Caroline gave a speech at a school around June 21, 2026 [S0]. |
| mem0 | temporal | c0_q9 | retrieval_miss | When did Caroline meet up with her friends, family, and mentors? | The week before 9 June 2023 | Caroline met up with her friends, family, and mentors around June 21, 2026, during a picnic [S1], and also around June 2 |
| rag-full | temporal | c0_q0 | retrieval_miss | When did Caroline go to the LGBTQ support group? | 7 May 2023 | Caroline went to the LGBTQ support group yesterday [S0] 'I went to a LGBTQ support group yesterday and it was so powerfu |
| rag-full | temporal | c0_q5 | retrieval_miss | When did Melanie run a charity race? | The sunday before 25 May 2023 | Melanie ran a charity race for mental health last Saturday [S1]. |
| rag-full | temporal | c0_q6 | retrieval_miss | When is Melanie planning on going camping? | June 2023 | Melanie is planning on going camping next month [S1] 'We're thinking about going camping next month.' |
| rag-vector | temporal | c0_q0 | retrieval_miss | When did Caroline go to the LGBTQ support group? | 7 May 2023 | Caroline went to the LGBTQ support group yesterday [S0] 'I went to a LGBTQ support group yesterday'. |
| rag-vector | multi-hop | c0_q3 | retrieval_miss | What did Caroline research? | Adoption agencies | Caroline researched counseling and mental health work [S1] 'looking into counseling or mental health work more'; [S4] 'k |
| rag-vector | temporal | c0_q5 | retrieval_miss | When did Melanie run a charity race? | The sunday before 25 May 2023 | Last Saturday [S0] |
| rag-vector | temporal | c0_q6 | retrieval_miss | When is Melanie planning on going camping? | June 2023 | Melanie is planning on going camping next month [S1]. |
| rag-vector | temporal | c0_q8 | retrieval_miss | When did Caroline give a speech at a school? | The week before 9 June 2023 | Last week [S0] |
| rag-vector | temporal | c0_q9 | retrieval_miss | When did Caroline meet up with her friends, family, and mentors? | The week before 9 June 2023 | Last week [S0] |
