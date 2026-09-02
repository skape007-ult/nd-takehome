# numbers.md — every number in the writeup, with its source

Each row: the claimed number, the command/file that produces it, seed/hardware.
Filled in as milestones land. Nothing here is hand-typed from memory.

## Dataset (Stage 1)
| number | value | source | notes |
|---|---|---|---|
| train examples | TBD | `data/train.jsonl` | after dedup + degenerate filter |
| held-out examples | TBD | `data/heldout.jsonl` | theorem-disjoint from train |
| distinct theorems (train) | TBD | `scripts/gen_data.py` stats | up to atom renaming |
| held-out renaming-overlap with train | TBD | `scripts/gen_data.py` stats | should be ~0 by construction |
| length histogram (emitted) | TBD | `figures/hist_length.png` | lengths 2–6 |
| emitted vs effective length | TBD | `figures/hist_emitted_vs_effective.png` | padding audit |
| rule-usage histogram | TBD | `figures/hist_rules.png` | |
| premise-count histogram | TBD | `figures/hist_premises.png` | |

## Model / training
| number | value | source | notes |
|---|---|---|---|
| params | TBD (~3.3M target) | `nd/model.py` printout | 4 layers, d=256 |
| train steps / batch | TBD | `nd/train.py` args | |
| wall-clock, hardware | TBD | training log | device recorded |

## Evaluation (Stage 1)
| number | value | source | notes |
|---|---|---|---|
| greedy held-out solve rate | TBD | `nd/eval.py` | Wilson CI |
| solve rate by length 2..6 | TBD | `figures/solve_by_length.png` | Wilson CIs |
| P (max length solved, pre-train) | TBD | `nd/eval.py` | at chosen accuracy |
| top failure reasons | TBD | `verify_cli.py --reasons` | |

## Stage 2 (stretch, only if attempted)
| number | value | source | notes |
|---|---|---|---|
| L (max length solved, RL) | TBD | | frozen-model control required |
| L − P | TBD | | in effective length |
