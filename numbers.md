# numbers.md — every number in the writeup, with its source

Each row: the claimed number, the command/file that produces it, seed/hardware.
Filled in as milestones land. Nothing here is hand-typed from memory.

## Dataset (Stage 1)
Reproduce: `PYTHONPATH=. python3 scripts/gen_data.py --n 300000 --seed 0`
then `scripts/check_dataset.py` and `scripts/make_figures.py`. Wall-clock: gen 48s,
check 4s (CPU, macOS arm64). All numbers below from `data/stats.json`.

| number | value | source | notes |
|---|---|---|---|
| sampling attempts | 300,000 | seed 0 | |
| pool accepted (verified) | 284,320 | stats.pool_in | 95% yield |
| pruned (had dead lines) | 177,633 | stats.pruned_changed | 62% of pool was padded |
| dropped degenerate | 4,133 | stats.dropped_degenerate | concl∈prem, F⊢·, A⊢(A v A) |
| dropped = validation_36 (±renaming) | 25,984 | stats.dropped_forbidden | never enters either split |
| distinct theorems | 71,892 | stats.distinct_theorems | up to renaming + premise order |
| train examples | 63,270 | `data/train.jsonl` | tight (emitted==effective) |
| held-out examples | 8,622 | `data/heldout.jsonl` | theorem-disjoint |
| held-out↔train theorem overlap | 0 | stats.heldout_train_theorem_overlap | 0 by construction (split on renaming/order-invariant key) |
| validation_36 leakage into splits | 0 | `scripts/check_dataset.py` | asserted |
| train length hist (2..6) | 9524 / 21148 / 19938 / 9586 / 3074 | stats.length_hist_train | `figures/hist_length.png` |
| held-out length hist (2..6) | 1344 / 2890 / 2709 / 1258 / 421 | stats.length_hist_heldout | |
| emitted vs effective (raw) | see figure | `figures/hist_emitted_vs_effective.png` | padding audit |
| rule-usage histogram | see figure | `figures/hist_rules.png` | all 15 rules present |
| premise-count histogram | see figure | `figures/hist_premises.png` | |

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
