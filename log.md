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

### Tokenizer (`nd/tokenizer.py`)
- Fixed whitespace vocab, one id per surface symbol, atomic N1..N64. 101 tokens.
- Round-trip check over all 71,892 dataset examples: 0 mismatches. Max sequence
  length 193 (< block_size 256). All target prompts (validation_36, test_short,
  test_long) encode with no unknown tokens.

### Model (`nd/model.py`)
- Pre-norm GPT, 4 layers, d=256, 8 heads, weight-tied head. 3.251M params
  (reference ~3.3M). Forward + generate (greedy/sampled, per-row QED/EOS stop).

### Training (`nd/train.py`) + perf investigation
- Prompt-masked next-token loss (only the proof body + EOS is scored), AdamW,
  warmup 200 + cosine, periodic greedy held-out probe, saves best checkpoint.
- Perf dead-end + fix: first MPS run showed 1.47s/step. Cause = per-step
  `loss.item()` forcing a GPU sync every step. Fixed: accumulate loss on-device,
  sync once per 100 steps. Benchmark (40 steps, proper mps.synchronize):
  CPU 2.09s/step vs MPS 0.53s/step → MPS wins clearly for this tiny model once
  the sync is gone. 4000 steps ≈ 35 min on this M-series.
- Launched full run: steps 4000, batch 128, seed 0.

### Diagnostic @ step 500 (loss 0.54, greedy held-out solve 0.1%)
- Low loss but ~0% solve looked alarming — checked whether eval was buggy. It is
  NOT: inspecting raw greedy output shows well-formed proofs. The failures are
  semantic, and systematic:
  - "premise block does not match declared premises": the model does not yet copy
    the premise formulas verbatim into the PR lines (it paraphrases/hallucinates a
    simpler premise).
  - "final formula is not the conclusion": it writes a plausible forward proof but
    doesn't steer to the exact requested conclusion.
- Both are the in-context copy/align capability (induction heads) that typically
  emerges as a phase transition, consistent with loss still being high. The
  reference model is this exact size and the brief targets ≥85%, so the working
  hypothesis is undertraining, not a design flaw. Decision: watch the 1000/1500
  probes before any intervention rather than tuning blind.

### Training trajectory (phase transition confirmed)
- Greedy held-out probe (1k subset, decode budget 80): 500→0.1%, 1000→36.9%,
  1500→54.9%, 2000→65.6%, 2500→72.8%, 3000→74.8%, 3500→78.9%, 4000→80.6%.
  Undertraining hypothesis confirmed; still climbing at 4000. 65 min on MPS.

### Decode-budget barrier (fixed)
- First full eval: 75.8% overall, P=3. Failure tally had lots of parse errors
  (eof in formula, missing ), missing QED) → suspicious. Checked gold body token
  counts: len-6 proofs need up to 154 tokens, len-5 up to 125; I was decoding
  only 80 new tokens. 1,411 held-out gold proofs literally could not finish.
  Raised max_new_tokens 80→176 (prompt+body stays < block_size 256). No retrain.
- Re-eval (budget 176): **86.9%** overall (7493/8622), target met. Per length:
  len2 0.982, len3 0.967, len4 0.851, len5 0.653, len6 0.599. P=4 (point) / P=3
  (Wilson-LB, since len-4 LB 0.837 < 0.85). Failures now purely semantic
  (ANDI 529, ORI1 206, ORI2 134 dominate). `verify_cli --reasons` agrees (86.9%).
  Figure: figures/solve_by_length.png.

### THE barrier: generator rule coverage (validation_36 cross-check)
- Held-out is in-distribution. On the mentor's curated validation_36 (eval only,
  never trained on): 2/36 overall — but 24/36 need >6 lines (Stage-2 territory,
  OOD by the L=6 cap), so the honest number is the **≤6 bin: 2/12 = 16.7%**.
- Big in-dist vs curated gap. Root cause found in the dataset rule histogram:
  **R (reiteration) = 0**, NEGI = 8, ORE = 197, BOTE = 97, DN = 433, NEGE = 516,
  IMPE = 836, while ORI1/ORI2/ANDI = 39k/39k/27k. The forward-random generator
  over-samples "free" introduction rules (always applicable) and starves the
  consuming / goal-directed rules that canonical theorems need:
  - positive_paradox `Q ⊢ (P>Q)`, absorption: need R (reiteration) → impossible.
  - modus_tollens, negative_paradox, consequentia_mirabilis: need goal-directed
    NEGI/BOTE chains the generator underproduces.
  - Also malformed output on OOD prompts (dn_intro dropped a paren; identity
    emitted a garbage `N2 ;` line).
- This is the non-obvious barrier: **coverage, not model size**. The model is
  86.9% in-distribution and 16.7% on curated ≤6 because the training distribution
  is narrow. Fixable by rebalancing the generator (add R; bias toward
  consuming/discharging rules; richer NEGI/ORE beyond templates).

<!-- decision point: finalize clean in-dist Phase 1 vs invest in generator rebalance + retrain before the one-shot test run -->




