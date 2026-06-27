# Benchmark A/B comparison

Control: `artifacts/proof/multisession/ablation_memory_off`
Experiment: `artifacts/proof/multisession/headtohead`

Status: **ok**

| system | dataset | category | n control | n experiment | acc delta pp | query tok delta | write tok delta | search p95 delta ms | paired n | unpaired control | unpaired experiment | McNemar p | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eidetic-plus-full | longmemeval | multi-session | 6 | 6 | 16.7 | -0 | 0 | 261.2 | 6 | 0 | 0 | 1.0000 | compared |

## Per-question flips (attribution evidence)

_Distinct questions that flipped; a question flipping in multiple runs is counted once._

| system | dataset | category | gained questions | regressed questions | gained question ids | regressed question ids |
|---|---|---|---:|---:|---|---|
| eidetic-plus-full | longmemeval | multi-session | 1 | 0 | b5ef892d | - |
