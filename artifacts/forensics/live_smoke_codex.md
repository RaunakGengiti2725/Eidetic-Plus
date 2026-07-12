# Benchmark Failure Forensics

Rows inspected: 6
Failures/abstentions/errors: 4

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| abstention | 3 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |
| reader_error | 1 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |

## Categories

| category | buckets |
|---|---|
| single-session-user | abstention: 3, reader_error: 1 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | single-session-user | 58bf7951 | abstention | What play did I attend at the local community theater? | The Glass Menagerie | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | single-session-user | 1e043500 | abstention | What is the name of the playlist I created on Spotify? | Summer Vibes | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus | single-session-user | 1e043500 | reader_error | What is the name of the playlist I created on Spotify? | Summer Vibes | Summer Vibes [S4] |
| rag-full | single-session-user | 58bf7951 | abstention | What play did I attend at the local community theater? | The Glass Menagerie | I do not have that in memory. |
