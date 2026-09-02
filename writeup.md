<!--
  writeup.md — SKELETON ONLY.
  All narrative prose is written by the author (Sahil), by agreement:
  LLM-written writeups read badly and communication is 40% of the grade.
  The coding agent fills ONLY figures, tables, and numbers referenced here.
  Section headings below are a scaffold; rewrite/reorder freely.
-->

# Bootstrapping a natural-deduction prover past its training length

## Executive summary
<!-- <= 600 words, author-written. Lead with the headline: what P is, and (if
     Stage 2) what L and L-P are, in effective length, greedy, with CIs. -->

<!-- FIGURE: figures/solve_by_length.png -->

## 1. Setup and the metric that matters
<!-- The verifier checks soundness not relevance; effective_length; why length
     padding would corrupt L-P. Author prose; numbers from numbers.md. -->

## 2. Stage 1 — data generation
<!-- Forward generator, rule coverage, dedup up to renaming, degenerate filter,
     theorem-disjoint split, renaming-overlap estimate. -->
<!-- FIGURES: hist_length.png, hist_rules.png, hist_premises.png,
     hist_emitted_vs_effective.png -->

## 3. Stage 1 — tokenizer and model
<!-- Whitespace one-token-per-symbol tokenizer + why; 4L d=256 ~3.3M params. -->

## 4. Stage 1 — evaluation
<!-- Greedy held-out solve rate overall and by length 2..6 with Wilson CIs;
     define P; failure-mode analysis from verify_cli --reasons. -->

## 5. Stage 2 — bootstrapping past length 6  (stretch; only if attempted)
<!-- Rejection-sampling / expert iteration vs verifier reward. REQUIRED if
     attempted: frozen-model control (same sample budget), a transfer set RL
     never samples, found-proof-length histogram, in-distribution tracking. -->

## 6. Limitations

## 7. What I'd do with another week

## 8. Disclosure of AI-assistant use
<!-- Which parts the coding agent wrote (code) vs author (all prose, judgment
     calls, what to measure/believe). -->
