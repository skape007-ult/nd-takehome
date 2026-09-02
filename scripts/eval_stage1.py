#!/usr/bin/env python3
"""Evaluate a Stage-1 checkpoint on the held-out set.

    PYTHONPATH=. python3 scripts/eval_stage1.py --ckpt ckpt/stage1.pt

Reports greedy held-out solve rate overall and per proof length 2..6 with Wilson
95% CIs, defines P (max length whose lower CI bound clears the accuracy target),
writes figures/solve_by_length.png, and tallies failure reasons (the mechanism
behind verify_cli.py --reasons). Greedy is the reported baseline; a temperature
run can be added with --temperature/--seed but greedy is what defines P.
"""
import argparse, json, os, sys, collections
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nd.tokenizer import Tokenizer
from nd.model import load_ckpt
from nd.eval import solve_rate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def define_P(per_length, target, use_ci=True):
    """Largest length L (contiguous from 2) whose solve rate clears `target`.
    With use_ci, require the Wilson lower bound >= target (conservative)."""
    P = None
    for L in sorted(per_length):
        val = per_length[L]['lo'] if use_ci else per_length[L]['rate']
        if val >= target:
            P = L
        else:
            break
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(HERE, 'ckpt', 'stage1.pt'))
    ap.add_argument('--data-dir', default=os.path.join(HERE, 'data'))
    ap.add_argument('--target', type=float, default=0.85, help='accuracy for P')
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'data', 'eval_stage1.json'))
    ap.add_argument('--attempts-out', default=os.path.join(HERE, 'data', 'heldout_attempts.jsonl'))
    a = ap.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    tk = Tokenizer()
    model, blob = load_ckpt(a.ckpt, device)
    print(f'loaded {a.ckpt} (trained {blob.get("step")} steps), device={device}')

    heldout = [json.loads(l) for l in open(os.path.join(a.data_dir, 'heldout.jsonl'))]
    greedy = a.temperature is None
    r = solve_rate(model, tk, heldout, device, greedy=greedy,
                   temperature=(a.temperature or 0.0), seed=a.seed)

    mode = 'greedy' if greedy else f'temp={a.temperature} seed={a.seed}'
    print(f'\n[{mode}] held-out solve: {r["solved"]}/{r["n"]} = '
          f'{r["rate"]:.4f}  95% CI [{r["lo"]:.4f}, {r["hi"]:.4f}]')
    print('\n per length   k / n      rate    95% CI')
    for L in sorted(r['per_length']):
        d = r['per_length'][L]
        print(f'   len {L}    {d["k"]:5d}/{d["n"]:-5d}   {d["rate"]:.3f}  '
              f'[{d["lo"]:.3f}, {d["hi"]:.3f}]')

    P_ci = define_P(r['per_length'], a.target, use_ci=True)
    P_point = define_P(r['per_length'], a.target, use_ci=False)
    print(f'\n P (target {a.target:.0%}, Wilson-LB) = {P_ci}   '
          f'(point-estimate P = {P_point})')

    reasons = collections.Counter()
    for res in r['results']:
        if not res['ok']:
            reasons[res['reason'].split(' (line')[0]] += 1
    print('\n top failure reasons:')
    for reason, c in reasons.most_common(10):
        print(f'   {c:5d}  {reason}')

    # figure: solve rate by length with Wilson CI error bars
    Ls = sorted(r['per_length'])
    rates = [r['per_length'][L]['rate'] for L in Ls]
    los = [r['per_length'][L]['rate'] - r['per_length'][L]['lo'] for L in Ls]
    his = [r['per_length'][L]['hi'] - r['per_length'][L]['rate'] for L in Ls]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(Ls, rates, yerr=[los, his], marker='o', capsize=4,
                color='#2f6fed', lw=2)
    ax.axhline(a.target, ls='--', color='#888', label=f'target {a.target:.0%}')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(Ls)
    ax.set_xlabel('proof length (lines)')
    ax.set_ylabel('greedy solve rate (held-out)')
    ax.set_title(f'Stage-1 held-out solve rate by length ({mode})')
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    figpath = os.path.join(HERE, 'figures', 'solve_by_length.png')
    fig.savefig(figpath); plt.close(fig)
    print(f'\n wrote {figpath}')

    # persist numbers + an attempts file for verify_cli.py --reasons
    summary = {'mode': mode, 'ckpt_step': blob.get('step'),
               'overall': {k: r[k] for k in ('n', 'solved', 'rate', 'lo', 'hi')},
               'per_length': r['per_length'], 'P_ci': P_ci, 'P_point': P_point,
               'target': a.target, 'failure_reasons': dict(reasons.most_common())}
    with open(a.out, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(a.attempts_out, 'w') as f:
        for res in r['results']:
            f.write(json.dumps({'prompt': res['prompt'], 'proof': res['proof']}) + '\n')
    print(f' wrote {a.out} and {a.attempts_out}')


if __name__ == '__main__':
    main()
