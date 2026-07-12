# Benchmark Failure Forensics

Rows inspected: 10
Failures/abstentions/errors: 3

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| abstention | 2 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |
| reader_error | 1 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |

## Categories

| category | buckets |
|---|---|
| multi-hop | abstention: 1 |
| temporal | abstention: 1, reader_error: 1 |

## Consolidation Health

| system | groups | pending processed | facts | events | extraction timed out | extraction deferred |
|---|---:|---:|---:|---:|---:|---:|
| eidetic-plus-full | 1 | 19 | 73 | 73 | 0 | 0 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | multi-hop | c0_q4 | abstention | What is Caroline's identity? | Transgender woman | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | temporal | c0_q5 | reader_error | When did Melanie run a charity race? | The sunday before 25 May 2023 | Saturday 2023-05-20 [S0] "I ran a charity race for mental health last Saturday"; [S19] "I ran a charity race for mental  |
| eidetic-plus-full | temporal | c0_q9 | abstention | When did Caroline meet up with her friends, family, and mentors? | The week before 9 June 2023 | I don't have enough verified evidence in memory to answer that confidently. |
