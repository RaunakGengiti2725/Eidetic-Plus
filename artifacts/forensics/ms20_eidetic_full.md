# Benchmark Failure Forensics

Rows inspected: 20
Failures/abstentions/errors: 16

## Buckets

| bucket | count | suggested fix |
|---|---:|---|
| abstention | 8 | calibrate ABSTENTION_V2_TAU on dev logs; inspect entailment/proof signals |
| reader_error | 8 | inspect context blocks and reader scaffold; failure happened with plausible evidence present |

## Categories

| category | buckets |
|---|---|
| multi-session | abstention: 8, reader_error: 8 |

## Failed Items

| system | category | sample | bucket | question | gold | predicted |
|---|---|---|---|---|---|---|
| eidetic-plus-full | multi-session | 0a995998 | reader_error | How many items of clothing do I need to pick up or return from a store? | 3 | You need to pick up 1 item of clothing: your navy blue blazer [S7]. |
| eidetic-plus-full | multi-session | 6d550036 | abstention | How many projects have I led or am currently leading? | 2 | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | multi-session | gpt4_59c863d7 | reader_error | How many model kits have I worked on or bought? | I have worked on or bought five model kits. The scales of the models are: Revell F-15 Eagle (scale not mentioned), Tamiy | You have worked on or bought at least three model kits: the 1/72 scale B-29 bomber, the 1/24 scale '69 Camaro, and the R |
| eidetic-plus-full | multi-session | b5ef892d | abstention | How many days did I spend on camping trips in the United States this year? | 8 days. | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | multi-session | 3a704032 | reader_error | How many plants did I acquire in the last month? | 3 | You acquired 2 plants — a peace lily and a succulent — two weeks ago, on or around 2023-05-20 [S4] 'I got from the nurse |
| eidetic-plus-full | multi-session | gpt4_d84a3211 | reader_error | How much total money have I spent on bike-related expenses since the start of the year? | $185 | You have spent $65 on bike-related expenses since the start of the year: $25 for chain replacement and $40 for bike ligh |
| eidetic-plus-full | multi-session | aae3761f | abstention | How many hours in total did I spend driving to my three road trip destinations combined? | 15 hours for getting to the three destinations (or 30 hours for the round trip) | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | multi-session | dd2973ad | abstention | What time did I go to bed on the day before I had a doctor's appointment? | 2 AM | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | multi-session | c4a1ceb8 | abstention | How many different types of citrus fruits have I used in my cocktail recipes? | 3 | I do not have that in memory. |
| eidetic-plus-full | multi-session | gpt4_a56e767c | abstention | How many movie festivals that I attended? | I attended four movie festivals. | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | multi-session | 46a3abf7 | abstention | How many tanks do I currently have, including the one I set up for my friend's kid? | 3 | I don't have enough verified evidence in memory to answer that confidently. |
| eidetic-plus-full | multi-session | 36b9f61e | reader_error | What is the total amount I spent on luxury items in the past few months? | $2,500 | $1,700 [S0] "a pair of leather boots from a high-end Italian designer that I got for $500"; [S6] "that designer handbag  |
| eidetic-plus-full | multi-session | 28dc39ac | reader_error | How many hours have I spent playing games in total? | 140 hours | You have spent at least 30 hours playing games in total, based on completing The Last of Us Part II in 25 hours [S6] and |
| eidetic-plus-full | multi-session | 2e6d26dc | reader_error | How many babies were born to friends and family members in the last few months? | 5 | Two babies were born to friends and family members in the last few months: Charlotte (Mike and Emma's daughter) and Jasp |
| eidetic-plus-full | multi-session | gpt4_15e38248 | reader_error | How many pieces of furniture did I buy, assemble, sell, or fix in the past few months? | 4 | You fixed 1 piece of furniture (kitchen table leg) and assembled 1 piece of furniture (IKEA bookshelf) in the past few m |
| eidetic-plus-full | multi-session | 88432d0a | abstention | How many times did I bake something in the past two weeks? | 4 | I don't have enough verified evidence in memory to answer that confidently. |
