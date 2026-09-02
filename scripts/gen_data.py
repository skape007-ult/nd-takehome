#!/usr/bin/env python3
"""Generate the Stage-1 dataset.

    PYTHONPATH=. python3 scripts/gen_data.py --n 200000 --seed 0

Produces (under --out-dir, default data/):
  train.jsonl, heldout.jsonl  -- {text, prompt, proof, n_lines, n_prem, rules}
  stats.json                  -- all counts + histograms, incl. the raw
                                 emitted-vs-effective joint (padding audit)

Never emits any theorem that matches validation_36.jsonl (up to renaming and
premise reordering); those are counted under stats.dropped_forbidden.
"""
import argparse, json, os, random, collections

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd_verify import verify_text
from nd.generator import sample_proof
from nd.effective_length import lengths_from_text
from nd.dataset import build, theorem_split_key
from nd.prune import _split

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def forbidden_from_validation():
    """split_keys of every validation_36 theorem (renaming/order invariant)."""
    path = os.path.join(HERE, 'targets', 'validation_36.jsonl')
    keys = set()
    for line in open(path):
        rec = json.loads(line)
        prompt = rec['prompt'].strip()
        # prompt has no body; append a dummy QED-less body is unnecessary — parse header only
        premises, concl, _ = _split(prompt + ' PRF ' if not prompt.endswith('PRF') else prompt)
        keys.add(theorem_split_key(premises, concl))
    return keys


def make_pool(n, seed):
    rng = random.Random(seed)
    pool = []
    joint = collections.defaultdict(collections.Counter)   # emitted -> effective -> count
    for i in range(n):
        # Half the attempts are length-biased toward 5-6 so that, after pruning,
        # long proofs are well represented (length 6 defines P).
        t = sample_proof(rng, long_bias=(i % 2 == 0))
        if not t:
            continue
        pool.append(t)
        emitted, eff = lengths_from_text(t)
        joint[emitted][eff] += 1
    return pool, joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200000, help='sampling attempts')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'data'))
    ap.add_argument('--heldout-frac', type=float, default=0.12)
    ap.add_argument('--per-length-cap', type=int, default=None,
                    help='max proofs per emitted length (balancing)')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    forbidden = forbidden_from_validation()
    print(f'forbidden validation_36 theorem keys: {len(forbidden)}')

    print(f'sampling pool (attempts={a.n}, seed={a.seed}) ...')
    pool, joint = make_pool(a.n, a.seed)
    print(f'  pool accepted: {len(pool)}')

    out = build(pool, heldout_frac=a.heldout_frac,
                per_length_cap=a.per_length_cap, forbidden_keys=forbidden)
    stats = out['stats']
    stats['raw_emitted_vs_effective'] = {
        str(e): dict(sorted(joint[e].items())) for e in sorted(joint)}

    def dump(recs, name):
        fields = ('text', 'prompt', 'proof', 'n_lines', 'n_prem', 'rules')
        with open(os.path.join(a.out_dir, name), 'w') as f:
            for r in recs:
                f.write(json.dumps({k: r[k] for k in fields}) + '\n')

    dump(out['train'], 'train.jsonl')
    dump(out['heldout'], 'heldout.jsonl')
    with open(os.path.join(a.out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # final guard: NOTHING in train/heldout may verify against a validation theorem
    leak = sum(1 for r in out['train'] + out['heldout']
               if theorem_split_key(*_split(r['text'])[:2]) in forbidden)
    print(f'  leakage into splits (must be 0): {leak}')
    assert leak == 0

    print(json.dumps({k: stats[k] for k in (
        'pool_in', 'pruned_changed', 'dropped_degenerate', 'dropped_forbidden',
        'distinct_theorems', 'train_size', 'heldout_size',
        'heldout_train_theorem_overlap', 'length_hist_train',
        'length_hist_heldout')}, indent=2))


if __name__ == '__main__':
    main()
