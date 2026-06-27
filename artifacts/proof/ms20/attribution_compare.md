# Benchmark A/B comparison

Control: `artifacts/proof/ms20/ablation_memory_off`
Experiment: `artifacts/proof/ms20/headtohead`

Status: **ok**

| system | dataset | category | n control | n experiment | acc delta pp | query tok delta | write tok delta | search p95 delta ms | paired n | unpaired control | unpaired experiment | McNemar p | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eidetic-plus-full | longmemeval | multi-session | 20 | 20 | 10.0 | -0 | 0 | 139.6 | 20 | 0 | 0 | 0.6250 | compared |

## Per-question flips (attribution evidence)

_Distinct questions that flipped; a question flipping in multiple runs is counted once._

| system | dataset | category | gained questions | regressed questions | gained question ids | regressed question ids |
|---|---|---|---:|---:|---|---|
| eidetic-plus-full | longmemeval | multi-session | 3 | 1 | b5ef892d, gpt4_2f8be40d, gpt4_f2262a51 | dd2973ad |
