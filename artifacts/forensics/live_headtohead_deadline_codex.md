# Benchmark Failure Forensics

Rows inspected: 3
Failures/abstentions/errors: 2

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| retrieval_miss | 2 | raise lexical/graph/event recall depth; inspect candidate_memory_ids and query parsing |

## Categories

| category | buckets |
|---|---|
| multi-session | retrieval_miss: 2 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| rag-full | multi-session | 36b9f61e | retrieval_miss | What is the total amount I spent on luxury items in the past few months? | $2,500 | The total amount spent on luxury items in the past few months is $2,000: $800 for the luxury evening gown [S13] and $1,2 |
| rag-vector | multi-session | 36b9f61e | retrieval_miss | What is the total amount I spent on luxury items in the past few months? | $2,500 | The only two luxury item amounts explicitly stated in memory are $800 for a luxury evening gown [S5] and $500 for boots  |
