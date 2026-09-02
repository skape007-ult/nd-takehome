"""Generation + evaluation against the verifier.

`batched_generate` groups prompts by token length so each batch is uniform and
needs no padding (absolute positions stay correct, no pad-leakage through
attention). Greedy is the reported baseline; temperature/seed are recorded for
anything sampled. `solve_rate` decodes, verifies each proof against its own
prompt, and reports Wilson confidence intervals overall and per proof length.
"""
import math
import collections
import torch

from nd_verify import verify_text
from .tokenizer import Tokenizer


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, mid, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), p, min(1.0, centre + half))


def _proof_body(text):
    """Extract the proof body (first 'N..' through 'QED') from a decoded string."""
    toks = text.split()
    # find first line-index token after PRF
    if 'PRF' in toks:
        start = toks.index('PRF') + 1
    else:
        start = 0
    body = toks[start:]
    if 'QED' in body:
        body = body[:body.index('QED') + 1]
    return ' '.join(body)


@torch.no_grad()
def batched_generate(model, tk, prompts, device, max_new_tokens=80,
                     greedy=True, temperature=0.0, seed=0, batch_size=256):
    """Return a list of proof-body strings, one per prompt, in input order."""
    model.eval()
    gen = None
    if not greedy:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
    # group by encoded length so batches are uniform (no padding needed)
    by_len = collections.defaultdict(list)
    enc = []
    for i, p in enumerate(prompts):
        ids = tk.encode(p, add_bos=True, add_eos=False)
        enc.append(ids)
        by_len[len(ids)].append(i)

    out = [None] * len(prompts)
    for L, idxs in by_len.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            batch = torch.tensor([enc[i] for i in chunk], device=device)
            if gen is not None:
                cpu_ids = model.generate(batch.cpu(), max_new_tokens,
                                         tk.qed_id, tk.eos_id, temperature=temperature,
                                         greedy=False, generator=gen)
                full = cpu_ids
            else:
                full = model.generate(batch, max_new_tokens, tk.qed_id, tk.eos_id,
                                      greedy=True)
            for row, i in enumerate(chunk):
                out[i] = _proof_body(tk.decode(full[row].tolist()))
    return out


def solve_rate(model, tk, records, device, greedy=True, temperature=0.0,
               seed=0, max_new_tokens=80, batch_size=256):
    """records: list of dicts with 'prompt' and (optional) 'n_lines'.
    Returns dict with overall + per-reference-length solve stats."""
    prompts = [r['prompt'] for r in records]
    proofs = batched_generate(model, tk, prompts, device, max_new_tokens,
                              greedy, temperature, seed, batch_size)
    by_len_k = collections.Counter()
    by_len_n = collections.Counter()
    solved = 0
    results = []
    for r, proof in zip(records, proofs):
        ok, reason, nl = verify_text(r['prompt'].strip() + ' ' + proof.strip())
        solved += ok
        L = r.get('n_lines')
        if L is not None:
            by_len_n[L] += 1
            by_len_k[L] += ok
        results.append({'prompt': r['prompt'], 'proof': proof, 'ok': bool(ok),
                        'reason': reason, 'n_lines_out': nl, 'ref_len': L})
    n = len(records)
    lo, mid, hi = wilson(solved, n)
    per_length = {}
    for L in sorted(by_len_n):
        klo, kmid, khi = wilson(by_len_k[L], by_len_n[L])
        per_length[L] = {'k': by_len_k[L], 'n': by_len_n[L],
                         'rate': kmid, 'lo': klo, 'hi': khi}
    return {'n': n, 'solved': solved, 'rate': mid, 'lo': lo, 'hi': hi,
            'per_length': per_length, 'results': results}
