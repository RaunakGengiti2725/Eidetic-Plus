# Benchmark Failure Forensics

Rows inspected: 1
Failures/abstentions/errors: 1

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| reader_error | 1 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |

## Categories

| category | buckets |
|---|---|
| multi-session | reader_error: 1 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | multi-session | 36b9f61e | reader_error | What is the total amount I spent on luxury items in the past few months? | $2,500 | $500 + $1,200 = $1,700. The total amount spent on luxury items in the past few months is $1,700 [S0] "a pair of leather  |
