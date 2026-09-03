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
| train examples | 63,270 | `data/train.jsonl` | tight by `effective_length`; NOT tight in fact — see box-padding audit |
| held-out examples | 8,622 | `data/heldout.jsonl` | theorem-disjoint |
| held-out↔train theorem overlap | 0 | stats.heldout_train_theorem_overlap | 0 by construction (split on renaming/order-invariant key) |
| validation_36 leakage into splits | 0 | `scripts/check_dataset.py` | asserted |
| train length hist (2..6) | 9524 / 21148 / 19938 / 9586 / 3074 | stats.length_hist_train | `figures/hist_length.png` |
| held-out length hist (2..6) | 1344 / 2890 / 2709 / 1258 / 421 | stats.length_hist_heldout | |
| emitted vs effective (raw) | see figure | `figures/hist_emitted_vs_effective.png` | padding audit |
| rule-usage histogram | see figure | `figures/hist_rules.png` | **14 of 15 rules present; `R` = 0** (earlier "all 15" counted surface tokens: atom `R` == rule `R`) |
| starved rules (of 71,892 theorems) | R 0, NEGI 8, BOTE 97, ORE 197, DN 433, NEGE 516, IMPE 836 | stats.rule_hist_all | vs ORI1 39,583 / ORI2 39,379 / ANDI 26,938 |
| premise-count histogram | see figure | `figures/hist_premises.png` | |

## Model / training
| number | value | source | notes |
|---|---|---|---|
| params | 3.251M | `nd/model.py` printout | 4 layers, d=256, 8 heads, tied head |
| train steps / batch | 4000 / 128 | `nd/train.py` --seed 0 | |
| optimizer | AdamW β=(0.9, 0.95), wd 0.1 | `nd/train.py:103`, `optim_groups` | no decay on biases/norms |
| LR schedule | 3e-4 peak → 3e-5, warmup 200 + cosine | `nd/train.py:77-79`, `lr_at` | |
| dropout / grad clip | 0.1 / 1.0 | `nd/train.py:81`, `nd/model.py:23` | |
| wall-clock, hardware | 65 min | Apple M-series MPS | ~0.75s/step |
| block size / decode budget | 256 / 176 | | 176 covers longest gold body (154) |

## Evaluation (Stage 1) — greedy, ckpt/stage1.pt, `scripts/eval_stage1.py`
| number | value | source | notes |
|---|---|---|---|
| greedy held-out solve rate | 0.869 (7493/8622) | `data/eval_stage1.json` | 95% CI [0.862, 0.876]; verify_cli agrees |
| solve rate len 2 | 0.982 (1320/1344) | eval_stage1.json | [0.974, 0.988] |
| solve rate len 3 | 0.967 (2795/2890) | eval_stage1.json | [0.960, 0.973] |
| solve rate len 4 | 0.851 (2305/2709) | eval_stage1.json | [0.837, 0.864] |
| solve rate len 5 | 0.653 (821/1258) | eval_stage1.json | [0.626, 0.678] |
| solve rate len 6 | 0.599 (252/421) | eval_stage1.json | [0.551, 0.644] |
| **P (pre-train frontier, target 85%)** | **3** | eval_stage1.json + box_padding_audit.json | 4 only on labelled point estimate (len-4 0.851 vs bar 0.850); Wilson LB 0.837 → 3; relabelled 0.848 → 3 |
| solve-by-length figure | — | `figures/solve_by_length.png` | with Wilson CIs |
| top failure reasons | rule check failed: ANDI 529, ORI1 206, ORI2 134, ANDE1 91, IMPI 45, ANDE2 44, IMPE 17 | `verify_cli.py data/heldout_attempts.jsonl --reasons` | line claimed that rule and did not follow by it |
| non-rule failures | 48 wrong final formula, 3 bad premise block, 2 wrong ref count | same | zero truncation failures |

## Box-interior padding audit
Reproduce: `PYTHONPATH=. python3 scripts/audit_box_padding.py`
(writes `data/box_padding_audit.json`, `figures/box_padding_audit.png`,
`figures/solve_by_true_length.png`). Deterministic, no model involved.

`effective_length` resolves a cited box to its whole index span, so `prune` keeps
dead lines at depth ≥ 1. "Strict" = a box contributes its AS line, its end line,
and the end line's transitive citations. Every strict-pruned proof is re-verified.

| number | value | source | notes |
|---|---|---|---|
| train proofs containing a box | 13,336 | audit.train.boxed | |
| …with dead interior lines | **4,738 (35.5%)** | audit.train.padded | = 7.5% of 63,270 examples |
| held-out proofs with dead interior lines | **650 (7.5%)** | audit.heldout.padded | |
| strict-prune re-verify failures | **0 / 8,622** | audit.strict_prune_reverify_failures | the removed lines were genuinely dead |
| labelled 5 → true 3/4/5 | 120 / 215 / 923 | audit.heldout_relabel_matrix | 26.6% mislabelled |
| labelled 6 → true 3/4/5/6 | 35 / 151 / 129 / **106** | audit.heldout_relabel_matrix | **74.8% mislabelled** |
| solve rate by TRUE length 2..6 | 0.982 / 0.953 / 0.848 / 0.591 / **0.387** | audit.solve_by_true_length | vs labelled 0.982 / 0.967 / 0.851 / 0.653 / 0.599 |
| len-6 overstatement | **21.2 pp** | 0.599 − 0.387 | Wilson CIs [55.1, 64.4] vs [30.0, 48.2] do not overlap |
| P, labelled vs true (85% bar) | **4 vs 3** | audit.P_by_bar | also 4/4 at 80% and 75%; 3/3 at 90% |
| model's accepted proofs with dead interior lines | 117 / 7,493 (1.6%) | audit.model_accepted_with_dead_interior | pre-RL reward-hack surface |
| model's accepted 6-line outputs, emitted → strict | 58 → 42 | audit.model_written_length* | |

## Evaluation — validation_36 cross-check (eval only, never trained on)
| number | value | source | notes |
|---|---|---|---|
| overall | 2/36 = 0.056 | `eval_targets.py --by min_lines_ub` | 95% CI 1.5–18.1%; 24/36 need >6 lines (OOD by L=6 cap) |
| **≤6 bin** | **2/12 = 0.167** | eval_targets.py | 95% CI 4.7–44.8%; in-dist 0.869 vs curated 0.167 = coverage gap |
| barrier | R=0, NEGI=8, ... | `data/stats.json` rule_hist | generator starves goal-directed rules |

## Test leaderboard — ONE-SHOT, run once, no tuning (`score_test.py`)
| set | value | source | notes |
|---|---|---|---|
| test_short | 31.5% (84/267) | `data/test_short_out.jsonl` | 95% CI 26.2–37.3% |
| test_long | 1.5% (8/532) | `data/test_long_out.jsonl` | 95% CI 0.8–2.9%; long (>6) proofs = Stage-2 territory |

Note on reproducibility: `data/` was generated by `nd/generator.py` at commit 268e468
(the template-free forward generator). `nd/generator_templates.py` is a later
rebalance exploration that was **NOT** used for the shipped model (see log.md).

## Stage 2 (stretch, only if attempted)
| number | value | source | notes |
|---|---|---|---|
| L (max length solved, RL) | TBD | | frozen-model control required |
| L − P | **not measured** | | Stage 2 not attempted; score in STRICT effective length when it is |
