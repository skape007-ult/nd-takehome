# Reproducing every number

Exact commands, seeds, and hardware for every figure and number in `writeup.md`.
`README.md` is the original take-home brief; this file is the submission's
reproduction guide. All numbers are also indexed in `numbers.md`.

## Environment
- Python 3.10, `pip install -r requirements.txt` (torch, numpy, matplotlib).
- Hardware used: Apple M-series (arm64), Torch MPS. Device is auto-selected
  (`cuda` > `mps` > `cpu`); results are seed-fixed but exact solve counts can
  vary slightly across GPU backends. Greedy decoding is deterministic per model.
- The verifier `nd_verify/verify.py` is unmodified and is the only ground truth.

## 1. Dataset  (CPU only, ~1 min)
```
PYTHONPATH=. python3 scripts/gen_data.py --n 300000 --seed 0
PYTHONPATH=. python3 scripts/check_dataset.py      # strict: all verify, <=6, tight, disjoint, no val leak
PYTHONPATH=. python3 scripts/make_figures.py       # figures/hist_*.png
```
Outputs: `data/train.jsonl`, `data/heldout.jsonl`, `data/stats.json`.

## 2. Train Stage-1  (MPS ~50 min; T4 ~20 min)
```
PYTHONPATH=. python3 nd/train.py --steps 4000 --batch 128 --warmup 200 \
    --eval-every 500 --eval-n 1000 --seed 0 --out ckpt/stage1.pt
```
Saves the checkpoint with the best greedy held-out probe.

## 3. Evaluate Stage-1  (defines P)
```
PYTHONPATH=. python3 scripts/eval_stage1.py --ckpt ckpt/stage1.pt --target 0.85
```
Prints overall + per-length greedy solve rate with Wilson CIs, defines P,
writes `figures/solve_by_length.png`, `data/eval_stage1.json`, and
`data/heldout_attempts.jsonl`. Failure reasons also via:
```
PYTHONPATH=. python3 verify_cli.py data/heldout_attempts.jsonl --reasons
```

## 4. Validation cross-check  (curated 36; eval only, never trained on)
```
python3 prove.py --ckpt ckpt/stage1.pt --in targets/validation_36.jsonl \
    --out data/val36_out.jsonl --greedy
python3 eval_targets.py --proofs data/val36_out.jsonl --by min_lines_ub
```

## 5. Test leaderboard  (RUN ONCE — do not tune against it)
```
python3 prove.py --ckpt ckpt/stage1.pt --in targets/test_short_prompts.jsonl \
    --out data/test_short_out.jsonl --greedy
python3 score_test.py data/test_short_out.jsonl
python3 prove.py --ckpt ckpt/stage1.pt --in targets/test_long_prompts.jsonl \
    --out data/test_long_out.jsonl --greedy
python3 score_test.py data/test_long_out.jsonl
```
