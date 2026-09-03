# Natural-deduction prover — take-home submission

**Sahil Paliwal** · Stage 1 complete, Stage 2 not attempted.
The original take-home brief is preserved verbatim in [`BRIEF.md`](BRIEF.md).

**Read [`writeup.md`](writeup.md) first** — it opens with a ≤600-word executive
summary and three figures. This file is the reproduction guide the brief asks for:
every command, seed, and wall-clock needed to regenerate every number.

## Headline

| | value | 95% CI |
|---|---|---|
| Stage-1 held-out, greedy | **86.9%** (7,493/8,622) | 86.2–87.6% |
| **P** (pre-trained frontier, 85% bar) | **3** | 4 on the labelled point estimate, by 0.1pp |
| **L − P** | **not measured** | Stage 2 not attempted |
| `validation_36`, ≤6 bin | 16.7% (2/12) | 4.7–44.8% |
| test_short (one-shot) | 31.5% (84/267) | 26.2–37.3% |
| test_long (one-shot) | 1.5% (8/532) | 0.8–2.9% |

Three findings, each with a figure in the write-up: the barrier is **rule
coverage**, not model capacity (`R` occurs 0 times in 71,892 theorems); `prune`
**misses dead lines inside subproof boxes**, so 7.5% of the shipped data is padded
where the metric cannot see it; and **length is a weak difficulty proxy** — 75% of
the "length 6" held-out theorems are shorter than their label.

## Where things live

| path | what |
|---|---|
| [`writeup.md`](writeup.md) | the submission write-up (exec summary, method, results, limitations, next steps) |
| [`numbers.md`](numbers.md) | every number in the write-up with the command/file it came from |
| [`log.md`](log.md) | dated work log, dead ends included |
| [`BRIEF.md`](BRIEF.md) | the original take-home brief, unmodified |
| `nd/` | library: formula utils, effective length, generator, pruning, dataset, tokenizer, model, train, eval |
| `scripts/` | runnable pipeline steps (data, figures, checks, eval, audit) |
| `data/` | shipped train/held-out sets, stats, eval outputs |
| `ckpt/stage1.pt` | the Stage-1 checkpoint every number below was produced from |
| `nd_verify/` | the provided verifier — **unmodified**, and the only ground truth |
| `prove.py` | matches `submission_template/prove.py` exactly (args, output schema, no verifier in the loop) |

## Environment

- Python 3.10, `pip install -r requirements.txt` (torch, numpy, matplotlib).
- Hardware used: Apple M-series (arm64), Torch MPS. Device is auto-selected
  (`cuda` > `mps` > `cpu`); results are seed-fixed, but exact solve counts can
  vary slightly across GPU backends. Greedy decoding is deterministic per model.
- `nd_verify/verify.py` is unmodified and is the only ground truth. Every proof
  counted anywhere in this repo was accepted by it.

## 1. Dataset (CPU only, ~1 min)

```
PYTHONPATH=. python3 scripts/gen_data.py --n 300000 --seed 0
PYTHONPATH=. python3 scripts/check_dataset.py   # all verify, <=6 lines, disjoint, no val leak
PYTHONPATH=. python3 scripts/make_figures.py    # figures/hist_*.png
```

Outputs `data/train.jsonl` (63,270), `data/heldout.jsonl` (8,622),
`data/stats.json`. Wall-clock: generation 48s, check 4s.

> The shipped `data/` and `ckpt/stage1.pt` are the exact pair the final model was
> trained on, and both are committed — you do **not** need to regenerate anything
> to reproduce the eval numbers. The command above reproduces the dataset with the
> template-free `nd/generator.py` at commit `268e468`.
> `nd/generator_templates.py` is a later coverage-rebalance exploration that was
> **NOT** used for the shipped model (see `log.md`).

## 2. Train Stage-1 (MPS ~65 min; T4 ~20 min)

```
PYTHONPATH=. python3 nd/train.py --steps 4000 --batch 128 --warmup 200 \
    --eval-every 500 --eval-n 1000 --seed 0 --out ckpt/stage1.pt
```

Saves the checkpoint with the best greedy held-out probe. 3.251M parameters
(4 layers, d=256, 8 heads, weight-tied head), ~0.75s/step.

## 3. Evaluate Stage-1 (defines P)

```
PYTHONPATH=. python3 scripts/eval_stage1.py --ckpt ckpt/stage1.pt --target 0.85
```

Prints overall and per-length greedy solve rate with Wilson CIs, defines P, and
writes `figures/solve_by_length.png`, `data/eval_stage1.json`,
`data/heldout_attempts.jsonl`. Failure reasons:

```
PYTHONPATH=. python3 verify_cli.py data/heldout_attempts.jsonl --reasons
```

## 4. Validation cross-check (curated 36; eval only, never trained on)

```
python3 prove.py --ckpt ckpt/stage1.pt --in targets/validation_36.jsonl \
    --out data/val36_out.jsonl --greedy
python3 eval_targets.py --proofs data/val36_out.jsonl --by min_lines_ub
```

## 5. Test leaderboard (RUN ONCE — not tuned against)

```
python3 prove.py --ckpt ckpt/stage1.pt --in targets/test_short_prompts.jsonl \
    --out data/test_short_out.jsonl --greedy
python3 score_test.py data/test_short_out.jsonl
python3 prove.py --ckpt ckpt/stage1.pt --in targets/test_long_prompts.jsonl \
    --out data/test_long_out.jsonl --greedy
python3 score_test.py data/test_long_out.jsonl
```

## 6. Box-interior padding audit (CPU only, ~1 min)

```
PYTHONPATH=. python3 scripts/audit_box_padding.py
```

Reproduces finding 2 and finding 3: measures how much of the shipped dataset is
still padded once a cited subproof box contributes only the lines it actually
uses, relabels the held-out set by true minimal length, and recomputes the
solve-rate curve and P under both labellings. Deterministic, no model involved.
Writes `data/box_padding_audit.json`, `figures/box_padding_audit.png`,
`figures/solve_by_true_length.png`.

Every strict-pruned proof is re-verified with `verify_text`: 0 failures over
8,622, which is what makes the relabelling a fact rather than an argument.

## 7. Tests

```
PYTHONPATH=. python3 tests/test_effective_length.py
PYTHONPATH=. python3 tests/test_generator.py
```

`test_generator.py` pins the rule-coverage barrier (`R` is never generated), so a
future generator change makes it visible rather than silent.

## Known issues, stated up front

- **`nd/effective_length.py` under-counts padding inside subproof boxes.** A cited
  box resolves to its whole index span, so `prune` keeps dead lines at depth ≥ 1.
  7.5% of the shipped dataset is padded that way. This is left **unfixed on
  purpose** so that `data/` stays byte-reproducible from the commit that produced
  it; §6 above measures it and `writeup.md` §7 gives the fix.
- **`scripts/check_dataset.py`'s tightness assertion inherits that blind spot.** It
  proves "no dead lines at depth 0", not "no dead lines". Its docstring says so.
- **The generator never emits `R`.** `'R'` is listed in `generator.LOCAL_RULES` but
  `_local_candidates` has no branch producing it. This is the diagnosed coverage
  barrier, not an oversight left unnoticed — see `writeup.md` §6.
