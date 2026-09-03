#!/usr/bin/env python3
"""Render Stage-1 dataset figures from data/stats.json into figures/.

    PYTHONPATH=. python3 scripts/make_figures.py

Figures (all deliverables referenced by writeup.md):
  hist_length.png                 length distribution, train vs held-out
  hist_rules.png                  fraction of proofs using each rule
  hist_premises.png               premise-count distribution
  hist_emitted_vs_effective.png   padding audit on the RAW generated pool
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, 'figures')
ACCENT, ACCENT2, WARN = '#2f6fed', '#e8823a', '#c0392b'
plt.rcParams.update({'figure.dpi': 130, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.axisbelow': True,
                     'font.size': 11})


def load_stats():
    with open(os.path.join(HERE, 'data', 'stats.json')) as f:
        return json.load(f)


def fig_length(s):
    lens = [2, 3, 4, 5, 6]
    tr = [s['length_hist_train'].get(str(L), 0) for L in lens]
    ho = [s['length_hist_heldout'].get(str(L), 0) for L in lens]
    x = range(len(lens))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], tr, 0.4, label='train', color=ACCENT)
    ax.bar([i + 0.2 for i in x], ho, 0.4, label='held-out', color=ACCENT2)
    ax.set_xticks(list(x)); ax.set_xticklabels(lens)
    ax.set_xlabel('proof length (lines, = effective length)')
    ax.set_ylabel('number of theorems')
    ax.set_title('Stage-1 dataset: proof-length distribution')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'hist_length.png')); plt.close(fig)


ALL_RULES = ['PR', 'AS', 'ANDI', 'ANDE1', 'ANDE2', 'IMPI', 'IMPE', 'ORI1',
             'ORI2', 'ORE', 'NEGI', 'NEGE', 'BOTE', 'DN', 'R']


def fig_rules(s):
    """Per-proof rule presence, log scale, over ALL 15 rules.

    Plotted over the fixed rule list rather than over the keys present in the
    histogram: a rule the generator never emits has no key, and omitting it is
    exactly how `R = 0` stayed invisible. Zeros are drawn as a labelled stub so
    the coverage hole is the first thing the figure shows.
    """
    total = s['distinct_theorems']
    counts = {r: s['rule_hist_all'].get(r, 0) for r in ALL_RULES}
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    frac = [v / total for _, v in items]
    starved = 0.02                       # < 2% of proofs = effectively unseen
    colors = [ACCENT if f >= starved else WARN for f in frac]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    floor = 2e-5
    ax.bar(names, [max(f, floor) for f in frac], color=colors)
    ax.set_yscale('log')
    ax.set_ylim(floor, 2.0)
    ax.axhline(starved, ls='--', lw=1.1, color=WARN)
    ax.text(len(names) - 0.4, starved * 1.25, 'starved (<2% of proofs)',
            ha='right', fontsize=9, color=WARN)
    for i, (n, v) in enumerate(items):
        if frac[i] < starved:
            ax.text(i, max(frac[i], floor) * 1.9, '0' if v == 0 else str(v),
                    ha='center', fontsize=8, color=WARN)
    ax.set_ylabel('fraction of proofs using rule (log)')
    ax.set_title('Stage-1 dataset: rule coverage over all 15 rules\n'
                 'R is never generated; NEGI appears in 8 of 71,892 theorems')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'hist_rules.png')); plt.close(fig)


def fig_premises(s):
    ph = s['prem_hist_all']
    ks = sorted(int(k) for k in ph)
    vals = [ph[str(k)] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([str(k) for k in ks], vals, color=ACCENT)
    ax.set_xlabel('number of premises'); ax.set_ylabel('number of theorems')
    ax.set_title('Stage-1 dataset: premise-count distribution')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'hist_premises.png')); plt.close(fig)


def fig_padding(s):
    """Stacked bars: for each emitted length in the RAW pool, how the effective
    length breaks down. Shorter effective within a taller emitted = padding."""
    joint = s['raw_emitted_vs_effective']            # {emitted: {effective: count}}
    emitted = sorted(int(e) for e in joint)
    effs = sorted({int(k) for e in joint for k in joint[e]})
    cmap = plt.get_cmap('viridis')
    colors = {ef: cmap(i / max(1, len(effs) - 1)) for i, ef in enumerate(effs)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottoms = [0] * len(emitted)
    for ef in effs:
        vals = [joint[str(e)].get(str(ef), 0) for e in emitted]
        ax.bar([str(e) for e in emitted], vals, bottom=bottoms,
               color=colors[ef], label=f'eff {ef}')
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xlabel('emitted length (raw generated pool)')
    ax.set_ylabel('number of proofs')
    ax.set_title('Padding audit: effective length within each emitted length')
    ax.legend(title='effective', ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'hist_emitted_vs_effective.png')); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    s = load_stats()
    fig_length(s); fig_rules(s); fig_premises(s); fig_padding(s)
    print('wrote figures to', FIG)
    for f in sorted(os.listdir(FIG)):
        print('  ', f)


if __name__ == '__main__':
    main()
