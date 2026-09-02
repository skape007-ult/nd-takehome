# Work log

Honest, dated, in-order. Dead ends included on purpose.

---

## 2026-09-02

### Setup / recon
- Read the ground truth: `nd_verify/verify.py` (`verify_text` → `(ok, reason, n_lines)`),
  `spec.md`, `submission_template/prove.py`, examples, target files.
- Confirmed verifier runs: `nd_verify/scratch_check.py` → `(True, 'ok', 8)`.
- Env: macOS arm64 (Apple Silicon → Torch MPS available), Python 3.10.12, 8 cores.
  No torch/numpy installed yet — data + eval pipeline built CPU-only; training will
  be device-agnostic (MPS locally or Colab T4).
- Targets: `validation_36.jsonl` (NEVER train on), `test_short` (267),
  `test_long` (532). Test sets are a leaderboard — run `prove.py` once at the end.

### Formula utils + effective-length metric (`nd/formula.py`, `nd/effective_length.py`)
- Reused the verifier's own `parse_formula`/`parse_proof_tokens` so there is one
  parser in the project; added rendering + atom canonicalization + the citation-DAG
  metric on top.
- `effective_length`: build citation DAG, reverse-traverse from the final line,
  count reachable lines; a cited box (IMPI/NEGI/ORE) pulls in all interior lines.
- Tests (`tests/test_effective_length.py`, all green against `verify_text`):
  - 21 example proofs: emitted == verifier n_lines, 1 ≤ effective ≤ emitted.
  - Padding case: sound proof with 1 unused premise + 1 dead line → emitted 5,
    effective 3. Metric removes exactly the dead lines. This is the anti-reward-hack
    tool for Stage 2 and the difficulty-gradient guard for Stage 1.

### Forward generator (`nd/generator.py`)
- Local forward walk (workhorse) + IMPI insertion + NEGI/ORE templates that stay
  ≤ 6 lines. Every candidate gated through `verify_text`.
- Stress test (`tests/test_generator.py`, seed 0, 20k attempts): 90% yield, all
  18,022 accepted proofs re-verify, all 15 rules covered.
  - Gotcha: my first rule-coverage counter over-counted `R` because the atom `R`
    renders as the bare token `R`, same as the R-rule name. The generator does not
    emit the (useless) R rule. Real stats count the parsed rule field instead.
- Observed: 57% of raw generated proofs carry dead lines (effective < emitted).
  This is the padding problem from the brief in the wild.

### Pruning (`nd/prune.py`)
- Reduce a verified proof to its reachable sub-proof: keep reachable lines (cited
  boxes keep all interior lines), drop unused premises, renumber, re-render,
  re-verify. Falls back to the original if anything fails to re-verify.
- Validated on 13.5k generated proofs (seed 1): 57.7% tightened, 0 re-verify
  failures, 0 still-padded afterwards. Training proofs will be tight
  (emitted == effective), so length is an honest difficulty measure.
- Cost: pruning collapses many proofs to length 2 → dataset builder must
  oversample + balance to keep enough genuine length-5/6 proofs (6 defines P).

### Dataset build (`nd/dataset.py`, `scripts/gen_data.py`, `scripts/check_dataset.py`)
- Pipeline: sample pool → prune → drop degenerate → drop validation_36 matches →
  dedup (shortest proof per prompt) → theorem-disjoint split.
- Two theorem keys: `theorem_prompt_key` (renaming-invariant, order-sensitive) for
  dedup of prompts; `theorem_split_key` (renaming- AND premise-order-invariant) for
  the split, so a theorem can't leak across train/held-out via relabelling or
  premise reorder. hashlib (not salted `hash()`) → reproducible split.
- Long-bias sampling (half the attempts target length 5–6) to survive pruning
  collapse, so length 6 (which defines P) is well represented.
- Run (n=300k, seed 0, 48s CPU): pool 284,320; 62% were padded; dropped 4,133
  degenerate + 25,984 validation-matching; 71,892 distinct theorems →
  train 63,270 / held-out 8,622; overlap 0; leakage 0.
  - Note: 25,984 dropped-as-validation is large but expected — a random generator
    hits canonical short theorems (modus ponens, disj. syllogism = the val set)
    constantly; all their renamings are excluded from training. Good, not a bug.
- `check_dataset.py` (strict, the exam re-checks this): every train+held-out proof
  verifies for its exact sequent, ≤6 lines, tight (emitted==effective),
  prompt+proof==text, splits disjoint, no validation overlap. All green.
- Figures (`scripts/make_figures.py`): length, rule-usage, premise-count,
  emitted-vs-effective. Padding audit clearly shows emitted-5/6 proofs are mostly
  effective-2/4 in the raw pool → justifies pruning.

<!-- next: tokenizer, then model (4L d=256 ~3.3M) + training (needs torch/MPS) -->


