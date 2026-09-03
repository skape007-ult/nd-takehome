#!/usr/bin/env python3
"""Audit: dead lines INSIDE subproof boxes survive `nd/prune.py`.

    PYTHONPATH=. python3 scripts/audit_box_padding.py

Why this exists
---------------
`nd/effective_length.py` resolves a box citation (IMPI/NEGI/ORE) to the whole
index span `s..e`, so every line between the assumption and the box's end line
counts as reachable whether or not it feeds anything. `prune` keeps exactly that
set, so a dead line at depth 0 is removed and the identical dead line at depth 1
is not. The shipped dataset is therefore tight by the loose metric but not tight
in fact.

This script measures the gap under STRICT box semantics -- a cited box
contributes its AS line, its end line, and the end line's transitive citations,
and nothing else -- by strict-pruning every proof and RE-VERIFYING the result.
A strict-pruned proof that still verifies is proof that the removed lines were
genuinely dead.

Nothing here changes the shipped pipeline. `nd/effective_length.py` and
`nd/prune.py` are deliberately left as they were used to build `data/`, so the
dataset remains byte-reproducible; the fix is written up as future work.

Outputs
-------
    data/box_padding_audit.json      every number quoted in writeup.md
    figures/box_padding_audit.png    labelled length -> true minimal length
    figures/solve_by_true_length.png solve rate, labelled vs relabelled
"""
import collections
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from nd_verify import verify_text                       # noqa: E402
from nd_verify.verify import parse_proof_tokens         # noqa: E402
from nd.formula import render                           # noqa: E402
from nd.prune import _split, _render_line               # noqa: E402
from nd.effective_length import _line_deps              # noqa: E402
from nd.eval import wilson                              # noqa: E402

DATA, FIG = os.path.join(HERE, 'data'), os.path.join(HERE, 'figures')
ACCENT, ACCENT2, WARN = '#2f6fed', '#e8823a', '#c0392b'
plt.rcParams.update({'figure.dpi': 130, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.axisbelow': True, 'font.size': 11})


# --------------------------------------------------------------------------- #
# Loose (shipped) vs strict (honest) reachability
# --------------------------------------------------------------------------- #
def _closure(lines, dep):
    by = {ln['idx']: ln for ln in lines}
    seen, stack = set(), [lines[-1]['idx']]
    while stack:
        i = stack.pop()
        if i in seen or i not in by:
            continue
        seen.add(i)
        stack.extend(dep(by[i]))
    return seen


def loose_reachable(lines):
    """What nd/effective_length.py counts: a cited box pulls in its whole span."""
    idx0, n = lines[0]['idx'], len(lines)
    return _closure(lines, lambda ln: _line_deps(ln, idx0, n))


def strict_reachable(lines):
    """A cited box pulls in only the lines it actually uses.

    Following each line's own `refs` is enough: IMPI/NEGI cite (s, e) and ORE
    cites (j, s1, e1, s2, e2), so the assumption and end line of every cited box
    are kept, and the traversal then follows whatever the end line genuinely
    depends on. Interior lines nothing cites drop out.
    """
    return _closure(lines, lambda ln: list(ln['refs']))


def strict_prune(text):
    """Strict-prune a full 'THM ... QED'. Returns (text, changed, n_kept)."""
    premises, concl, body = _split(text)
    lines = parse_proof_tokens(body)
    keep = strict_reachable(lines)
    kept = [ln for ln in lines if ln['idx'] in keep]
    if len(kept) == len(lines):
        return text, False, len(kept)
    remap = {ln['idx']: k + 1 for k, ln in enumerate(kept)}
    new_prems = [ln['formula'] for ln in kept if ln['rule'] == 'PR']
    head = 'THM '
    if new_prems:
        head += ' , '.join(render(p) for p in new_prems) + ' '
    head += 'SEQ ' + render(concl) + ' PRF'
    body_str = ' '.join(
        _render_line(remap[ln['idx']], ln['depth'], ln['formula'], ln['rule'],
                     [remap[r] for r in ln['refs']]) for ln in kept)
    return head + ' ' + body_str + ' QED', True, len(kept)


def has_dead_interior(proof_body):
    lines = parse_proof_tokens(proof_body.split())
    return len(loose_reachable(lines)) > len(strict_reachable(lines))


# --------------------------------------------------------------------------- #
# 1. How much of the shipped dataset is still padded?
# --------------------------------------------------------------------------- #
def audit_split(path):
    boxed = padded = total = 0
    by_len = collections.Counter()
    padded_by_len = collections.Counter()
    for raw in open(path):
        d = json.loads(raw)
        total += 1
        by_len[d['n_lines']] += 1
        lines = parse_proof_tokens(d['proof'].split())
        if any(ln['rule'] in ('IMPI', 'NEGI', 'ORE') for ln in lines):
            boxed += 1
        if len(loose_reachable(lines)) > len(strict_reachable(lines)):
            padded += 1
            padded_by_len[d['n_lines']] += 1
    return {'total': total, 'boxed': boxed, 'padded': padded,
            'by_len': dict(by_len), 'padded_by_len': dict(padded_by_len)}


# --------------------------------------------------------------------------- #
# 2. True minimal length of every held-out theorem
# --------------------------------------------------------------------------- #
def relabel_heldout():
    true_len, labelled, fails = {}, {}, 0
    matrix = collections.defaultdict(collections.Counter)
    for raw in open(os.path.join(DATA, 'heldout.jsonl')):
        d = json.loads(raw)
        out, changed, _ = strict_prune(d['text'])
        ok, _, nl = verify_text(out)
        if changed and not ok:
            fails += 1
            nl = d['n_lines']
        elif not changed:
            nl = d['n_lines']
        labelled[d['prompt']] = d['n_lines']
        true_len[d['prompt']] = nl
        matrix[d['n_lines']][nl] += 1
    return true_len, labelled, matrix, fails


# --------------------------------------------------------------------------- #
# 3. Solve rate under both labellings
# --------------------------------------------------------------------------- #
def solve_rates(true_len, labelled):
    lab_n, lab_k = collections.Counter(), collections.Counter()
    tru_n, tru_k = collections.Counter(), collections.Counter()
    written, written_strict = collections.Counter(), collections.Counter()
    model_padded = 0
    for raw in open(os.path.join(DATA, 'heldout_attempts.jsonl')):
        a = json.loads(raw)
        p = a['prompt']
        ok, _, nl = verify_text(p + ' ' + a['proof'])
        lab_n[labelled[p]] += 1
        tru_n[true_len[p]] += 1
        if ok:
            lab_k[labelled[p]] += 1
            tru_k[true_len[p]] += 1
            written[nl] += 1
            out, _, _ = strict_prune(p + ' ' + a['proof'])
            ok2, _, nl2 = verify_text(out)
            written_strict[nl2 if ok2 else nl] += 1
            if has_dead_interior(a['proof']):
                model_padded += 1
    def table(n, k):
        return {L: dict(zip(('lo', 'rate', 'hi'), wilson(k[L], n[L])),
                        **{'k': k[L], 'n': n[L]}) for L in sorted(n)}
    return (table(lab_n, lab_k), table(tru_n, tru_k),
            dict(written), dict(written_strict), model_padded,
            sum(lab_k.values()))


def frontier(tbl, bar):
    best = 0
    for L in sorted(tbl):
        if tbl[L]['rate'] >= bar:
            best = L
    return best


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_padding(matrix):
    labels = sorted(matrix)
    trues = sorted({t for row in matrix.values() for t in row})
    cmap = plt.get_cmap('viridis')
    colors = {t: cmap(i / max(1, len(trues) - 1)) for i, t in enumerate(trues)}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, L in enumerate(labels):
        bottom = 0
        for t in trues:
            v = matrix[L].get(t, 0)
            if not v:
                continue
            ax.bar(i, v, 0.62, bottom=bottom, color=colors[t],
                   edgecolor='white', linewidth=0.6)
            if v / sum(matrix[L].values()) > 0.12:
                ax.text(i, bottom + v / 2, str(v), ha='center', va='center',
                        fontsize=9, color='white')
            bottom += v
        mis = sum(c for t, c in matrix[L].items() if t < L)
        if mis:
            ax.text(i, bottom + 60, f'{100*mis/sum(matrix[L].values()):.0f}% '
                    'mislabelled', ha='center', fontsize=9, color=WARN)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in trues]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 3250)
    ax.set_xlabel('length as labelled in the shipped held-out set')
    ax.set_ylabel('theorems')
    ax.set_title('Dead lines inside boxes survive pruning: 650 of 8,622 held-out\n'
                 'proofs are shorter than their label, all of it at length 5-6')
    ax.legend(handles, [f'truly {t}' for t in trues], fontsize=9, ncol=2,
              title='true minimal length', loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'box_padding_audit.png'))
    plt.close(fig)


def fig_solve(lab, tru):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tbl, name, color, marker, dy in ((lab, 'as labelled', ACCENT2, 's', 12),
                                         (tru, 'true minimal length', ACCENT,
                                          'o', -18)):
        xs = sorted(tbl)
        ys = [100 * tbl[L]['rate'] for L in xs]
        lo = [100 * (tbl[L]['rate'] - tbl[L]['lo']) for L in xs]
        hi = [100 * (tbl[L]['hi'] - tbl[L]['rate']) for L in xs]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=marker, capsize=4,
                    color=color, label=name, linewidth=1.8)
        for L in xs:
            ax.annotate(f"n={tbl[L]['n']}", (L, 100 * tbl[L]['rate']),
                        textcoords='offset points', xytext=(0, dy),
                        ha='center', fontsize=8, color=color)
    ax.axhline(85, ls='--', lw=1.2, color=WARN)
    ax.text(2.05, 86.5, 'target 85%', color=WARN, fontsize=9)
    ax.annotate('P = 4 as labelled (85.1%)\nP = 3 relabelled (84.8%)',
                xy=(4, 85), xytext=(2.15, 47), fontsize=9, color=WARN,
                arrowprops=dict(arrowstyle='->', color=WARN, lw=1.1))
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xlabel('proof length (lines)')
    ax.set_ylabel('greedy solve rate (%)')
    ax.set_ylim(25, 105)
    ax.set_title('Held-out solve rate, Wilson 95% CIs: relabelling costs 21pp\n'
                 'at length 6 and tips P from 4 to 3 at the 85% bar')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'solve_by_true_length.png'))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    train = audit_split(os.path.join(DATA, 'train.jsonl'))
    held = audit_split(os.path.join(DATA, 'heldout.jsonl'))
    true_len, labelled, matrix, fails = relabel_heldout()
    lab, tru, written, written_strict, model_padded, solved = \
        solve_rates(true_len, labelled)

    out = {
        'train': train, 'heldout': held,
        'strict_prune_reverify_failures': fails,
        'heldout_relabel_matrix': {str(k): dict(v) for k, v in matrix.items()},
        'solve_by_labelled_length': lab,
        'solve_by_true_length': tru,
        'model_written_length': written,
        'model_written_length_strict': written_strict,
        'model_accepted': solved,
        'model_accepted_with_dead_interior': model_padded,
        'P_by_bar': {f'{int(b*100)}%': {'labelled': frontier(lab, b),
                                        'true': frontier(tru, b)}
                     for b in (0.90, 0.85, 0.80, 0.75, 0.60)},
    }
    with open(os.path.join(DATA, 'box_padding_audit.json'), 'w') as f:
        json.dump(out, f, indent=2)

    fig_padding(matrix)
    fig_solve(lab, tru)

    p = train['padded'] / train['total']
    pb = train['padded'] / train['boxed']
    print(f"train:    {train['padded']}/{train['boxed']} boxed proofs padded "
          f"({100*pb:.1f}%) = {100*p:.1f}% of {train['total']} examples")
    print(f"held-out: {held['padded']}/{held['total']} padded "
          f"({100*held['padded']/held['total']:.1f}%)")
    print(f"strict-prune re-verify failures: {fails}")
    print(f"model wrote {model_padded}/{solved} accepted proofs with dead "
          f"interior lines ({100*model_padded/solved:.1f}%)")
    print('\nlabelled -> true minimal length (held-out)')
    for L in sorted(matrix):
        row = ' '.join(f'{t}:{c}' for t, c in sorted(matrix[L].items()))
        print(f'  {L}: {row}')
    print('\nP by accuracy bar (labelled / true):')
    for bar, v in out['P_by_bar'].items():
        print(f"  {bar:>4}: {v['labelled']} / {v['true']}")
    print('\nwrote data/box_padding_audit.json, '
          'figures/box_padding_audit.png, figures/solve_by_true_length.png')


if __name__ == '__main__':
    main()
