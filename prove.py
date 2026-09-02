#!/usr/bin/env python3
"""REQUIRED INTERFACE — we run this on private targets you never see.

    python prove.py --ckpt <path> --in targets.jsonl --out proofs.jsonl [--greedy | --temperature T --seed S]

Input : jsonl, one record per theorem, with at least {"name": str, "prompt": "THM ... PRF"}.
Output: jsonl, same order, ONE proof per theorem: {"name": ..., "prompt": ..., "proof": "<string>"}
        where the string is the proof BODY only (from the first "N<i>" through "QED"),
        whitespace-separated tokens in the format of spec.md.
--greedy       : deterministic decoding (this is the default if no temperature is given).
--temperature  : sample once per prompt at this temperature; --seed makes the run reproducible.
Must run on one GPU (or CPU) with only the dependencies in your requirements.txt, and finish
1,000 theorems in under 5 minutes on a T4-class GPU.
Do not call the verifier inside prove.py to filter, retry, or re-sample: return the raw output.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd.tokenizer import Tokenizer
from nd.model import load_ckpt
from nd.eval import batched_generate


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    device = pick_device()
    tk = Tokenizer()
    model, _ = load_ckpt(a.ckpt, device)

    records = [json.loads(l) for l in open(a.inp) if l.strip()]
    prompts = [r['prompt'] for r in records]

    greedy = a.temperature is None or a.greedy
    proofs = batched_generate(
        model, tk, prompts, device,
        max_new_tokens=80,
        greedy=greedy,
        temperature=(a.temperature or 0.0),
        seed=a.seed,
        batch_size=256,
    )

    with open(a.out, 'w') as f:
        for r, proof in zip(records, proofs):
            f.write(json.dumps({'name': r.get('name'), 'prompt': r['prompt'],
                                'proof': proof}) + '\n')


if __name__ == '__main__':
    main()
