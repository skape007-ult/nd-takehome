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

<!-- next: dataset build — dedup up-to-renaming, degenerate filter, theorem-disjoint split, stats/histograms -->

