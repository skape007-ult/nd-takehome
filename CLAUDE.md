# Notes for an assistant reading this repo

Take-home submission: pre-train a small transformer to write natural-deduction
proofs of ≤ 6 lines (Stage 1), then use RL to push past that length (Stage 2).
**Stage 1 is complete; Stage 2 was not attempted.** Start with `writeup.md`;
`README.md` has every reproduction command; `numbers.md` maps each claimed number
to the file or command it came from; `log.md` is the dated work log.

## Ground truth

`nd_verify/verify.py` is provided, **unmodified**, and is the only arbiter. Every
proof counted anywhere in this repo was accepted by `verify_text`. If you are
asked whether something is a valid proof, run it through `verify_text` rather than
reasoning about it — the notation is easy to get subtly wrong.

Quick check of any proof string:

```
PYTHONPATH=. python3 -c "from nd_verify import verify_text; print(verify_text('THM ... QED'))"
```

## Layout

| path | role |
|---|---|
| `nd/formula.py` | formula tuples, rendering, atom canonicalization. Reuses the verifier's parser — there is exactly one parser in the project |
| `nd/effective_length.py` | citation-DAG reachability; `emitted` vs `effective` length. **Has a known blind spot — see below** |
| `nd/generator.py` | the forward generator that produced the shipped data |
| `nd/generator_templates.py` | **NOT used for the shipped model.** A later coverage-rebalance exploration, kept for the record |
| `nd/prune.py` | trims a proof to its reachable core, re-verifies |
| `nd/dataset.py` | dedup up to renaming, degenerate filter, theorem-disjoint split |
| `nd/tokenizer.py`, `nd/model.py`, `nd/train.py`, `nd/eval.py` | 101-token vocab, 3.251M-param GPT, training, greedy eval with Wilson CIs |
| `scripts/` | runnable steps; `audit_box_padding.py` is the post-hoc audit behind findings 2 and 3 |
| `prove.py` | matches `submission_template/prove.py` exactly. No verifier in the loop |

## Three things that will otherwise trip you up

1. **The atom `R` and the rule `R` are the same surface token.** Any counter that
   matches proof tokens against a set of rule names will report reiteration as
   covered when it is not. Always read the parsed `rule` field
   (`parse_proof_tokens(body)[i]['rule']`). This collision hid the main finding for
   most of the project; `tests/test_generator.py` now pins it.

2. **`effective_length` under-counts padding inside subproof boxes.**
   `_line_deps` resolves a cited box `(s, e)` to the whole index span `s..e`, so
   every interior line counts as reachable whether or not it feeds the box's end
   line. Consequence: `prune` removes a dead line at depth 0 and keeps the
   identical dead line at depth 1, and `scripts/check_dataset.py`'s tightness
   assertion proves only "no dead lines at depth 0". 7.5% of the shipped dataset
   is padded this way.

   This is **left unfixed on purpose** so `data/` stays byte-reproducible from the
   commit that produced it. `scripts/audit_box_padding.py` measures the gap without
   touching the shipped pipeline. The fix is `writeup.md` §7 item 2. Do not
   "helpfully" patch `nd/effective_length.py` or `nd/prune.py` — that would
   invalidate `data/stats.json`, the figures, and every number in `numbers.md`.

3. **Length labels in `data/` are emitted length, which overstates difficulty.**
   75% of the "length 6" held-out theorems have a shorter verifier-accepted proof.
   When reasoning about difficulty, use the true minimal lengths in
   `data/box_padding_audit.json`, not `n_lines`.

## Conventions

- Run everything from the repo root with `PYTHONPATH=.`.
- Seeds are fixed (`--seed 0`) and splits hash with `hashlib`, not Python's salted
  `hash()`, so they reproduce across processes.
- `data/` and `ckpt/stage1.pt` are the exact pair the reported numbers come from.
  Regenerating is not needed to reproduce any evaluation number.
- Figures are generated, never hand-edited: `scripts/make_figures.py` and
  `scripts/audit_box_padding.py`.

## Headline numbers

86.9% greedy held-out (7,493/8,622, CI 86.2–87.6%); **P = 3** at an 85% bar;
`L − P` not measured; `validation_36` ≤6 bin 2/12; test_short 31.5% (84/267),
test_long 1.5% (8/532), both run once.
