#!/usr/bin/env python3
"""Strict integrity check on the shipped dataset. The exam checks this too.

    PYTHONPATH=. python3 scripts/check_dataset.py

Asserts, for every record in data/train.jsonl and data/heldout.jsonl:
  * verify_text accepts it for exactly its prompted sequent,
  * n_lines <= 6 (the hard cap L),
  * the proof is tight (emitted == effective) -- no padding,
  * prompt+proof reconstruct the stored text,
and across the sets:
  * no theorem (up to renaming/premise-order) appears in both train and heldout,
  * no theorem matches validation_36.jsonl.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd_verify import verify_text
from nd.effective_length import lengths_from_text
from nd.dataset import theorem_split_key
from nd.prune import _split

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data')


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name)) if l.strip()]


def forbidden():
    keys = set()
    for line in open(os.path.join(HERE, 'targets', 'validation_36.jsonl')):
        p = json.loads(line)['prompt'].strip()
        prem, concl, _ = _split(p if p.endswith('PRF') else p + ' PRF')
        keys.add(theorem_split_key(prem, concl))
    return keys


def check_set(recs, name):
    for i, r in enumerate(recs):
        text = r['text']
        assert r['prompt'].strip() + ' ' + r['proof'].strip() == text, f'{name}[{i}] prompt+proof != text'
        ok, reason, nl = verify_text(text)
        assert ok, f'{name}[{i}] does not verify: {reason}'
        assert nl == r['n_lines'] <= 6, f'{name}[{i}] length {nl} > 6 or mismatched'
        emitted, eff = lengths_from_text(text)
        assert emitted == eff, f'{name}[{i}] not tight: emitted {emitted} eff {eff}'
    print(f'  {name}: {len(recs)} records, all verify, all <= 6 lines, all tight')


def main():
    train, heldout = load('train.jsonl'), load('heldout.jsonl')
    check_set(train, 'train')
    check_set(heldout, 'heldout')

    def keys(recs):
        return {theorem_split_key(*_split(r['text'])[:2]) for r in recs}
    tk, hk, fk = keys(train), keys(heldout), forbidden()
    assert not (tk & hk), f'{len(tk & hk)} theorems shared between train and heldout'
    assert not (tk & fk), f'{len(tk & fk)} train theorems match validation_36'
    assert not (hk & fk), f'{len(hk & fk)} heldout theorems match validation_36'
    print(f'  disjoint train/heldout: OK ({len(tk)} vs {len(hk)} theorem keys)')
    print(f'  no overlap with validation_36: OK')
    print('DATASET OK')


if __name__ == '__main__':
    main()
