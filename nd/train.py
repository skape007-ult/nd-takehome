#!/usr/bin/env python3
"""Train the Stage-1 prover from scratch.

    PYTHONPATH=. python3 nd/train.py --steps 4000 --batch 128 --out ckpt/stage1.pt

Loss is next-token cross-entropy, masked so the model is only scored on the
proof body + EOS (never on the prompt). A greedy held-out solve-rate probe runs
periodically so training is watched by the metric that matters, not just loss.
Everything is seeded; the checkpoint stores the config so eval/prove reload it.
"""
import argparse, json, math, os, random, time
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict
from nd.tokenizer import Tokenizer, build_example
from nd.model import GPT, GPTConfig
from nd.eval import solve_rate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(arg):
    if arg != 'auto':
        return arg
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_examples(tk, path):
    ex = []
    for line in open(path):
        r = json.loads(line)
        ids, plen = build_example(tk, r['prompt'], r['proof'])
        ex.append((ids, plen))
    return ex


def make_batch(examples, idxs, pad_id, device):
    seqs = [examples[i][0] for i in idxs]
    plens = [examples[i][1] for i in idxs]
    maxlen = max(len(s) for s in seqs)
    B = len(seqs)
    inp = torch.full((B, maxlen - 1), pad_id, dtype=torch.long)
    tgt = torch.full((B, maxlen - 1), -100, dtype=torch.long)
    for b, (s, pl) in enumerate(zip(seqs, plens)):
        L = len(s)
        inp[b, :L - 1] = torch.tensor(s[:-1])
        tgt[b, pl - 1:L - 1] = torch.tensor(s[pl:])       # body + EOS only
    return inp.to(device), tgt.to(device)


def lr_at(step, base_lr, warmup, total, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def optim_groups(model, wd):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{'params': decay, 'weight_decay': wd},
            {'params': no_decay, 'weight_decay': 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(HERE, 'data'))
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--min-lr', type=float, default=3e-5)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--eval-n', type=int, default=1000)
    ap.add_argument('--out', default=os.path.join(HERE, 'ckpt', 'stage1.pt'))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = pick_device(a.device)
    tk = Tokenizer()
    print(f'device={device}  vocab={tk.vocab_size}')

    train_ex = load_examples(tk, os.path.join(a.data_dir, 'train.jsonl'))
    heldout = [json.loads(l) for l in open(os.path.join(a.data_dir, 'heldout.jsonl'))]
    probe = heldout[:a.eval_n]
    print(f'train examples: {len(train_ex)}  held-out probe: {len(probe)}')

    cfg = GPTConfig(vocab_size=tk.vocab_size, dropout=a.dropout, pad_id=tk.pad_id)
    model = GPT(cfg).to(device)
    print(f'params: {model.num_params()/1e6:.3f}M')
    opt = torch.optim.AdamW(optim_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.95))

    rng = random.Random(a.seed)
    N = len(train_ex)
    t0 = time.time()
    running = torch.zeros((), device=device)   # keep on-device; avoid per-step sync
    best = -1.0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def save(tag):
        torch.save({'model_state': model.state_dict(), 'cfg': asdict(cfg),
                    'step': step, 'args': vars(a)}, a.out)

    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.lr, a.warmup, a.steps, a.min_lr)
        idxs = [rng.randrange(N) for _ in range(a.batch)]
        inp, tgt = make_batch(train_ex, idxs, tk.pad_id, device)
        model.train()
        _, loss = model(inp, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.detach()

        if step % 100 == 0:
            sps = (time.time() - t0) / step
            print(f'step {step:5d}  loss {running.item()/100:.3f}  '
                  f'lr {opt.param_groups[0]["lr"]:.2e}  {sps:.3f}s/step')
            running.zero_()

        if step % a.eval_every == 0 or step == a.steps:
            r = solve_rate(model, tk, probe, device, greedy=True)
            print(f'  [probe @ {step}] greedy held-out solve '
                  f'{r["solved"]}/{r["n"]} = {r["rate"]:.3f} '
                  f'[{r["lo"]:.3f},{r["hi"]:.3f}]')
            if r['rate'] >= best:
                best = r['rate']
                save('best')
                print(f'  saved checkpoint -> {a.out} (best {best:.3f})')

    print(f'done in {(time.time()-t0)/60:.1f} min. best probe solve rate {best:.3f}')


if __name__ == '__main__':
    main()
