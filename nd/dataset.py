"""Turn a pool of verified proofs into tight, deduped, split train/held-out sets.

Pipeline (all keyed off the verifier + effective_length + canonical renaming):
  1. prune each proof to its reachable sub-proof (emitted == effective).
  2. drop degenerate theorems (conclusion is a premise; F among premises; A|-AvA).
  3. dedup: one proof per exact prompt (shortest wins on ties).
  4. theorem-disjoint split: assign each theorem to train/held-out by a stable
     hash of a renaming- AND premise-order-invariant key, so held-out is disjoint
     from train even under atom relabelling or premise reordering.
  5. report the measured held-out<->train renaming overlap (0 by construction).

hashlib (not Python's salted hash) makes the split reproducible across runs.
"""
import hashlib
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens, BOT
from .formula import render, canonical_rename
from .prune import prune, _split


# --------------------------------------------------------------------------- #
# Record extraction
# --------------------------------------------------------------------------- #
def split_prompt_proof(text):
    """Split 'THM ... PRF N1 ... QED' into (prompt 'THM ... PRF', proof 'N1 ... QED')."""
    marker = ' PRF '
    k = text.index(marker)
    return text[:k + 4], text[k + 5:]


def record(text):
    """Build a dataset record from a verified full-string proof."""
    ok, reason, nl = verify_text(text)
    assert ok, reason
    premises, concl, body = _split(text)
    lines = parse_proof_tokens(body)
    rules = [ln['rule'] for ln in lines]
    prompt, proof = split_prompt_proof(text)
    return {
        'text': text,
        'prompt': prompt,
        'proof': proof,
        'n_lines': nl,
        'n_prem': len(premises),
        'rules': rules,
        'thm_key': theorem_prompt_key(premises, concl),
        'split_key': theorem_split_key(premises, concl),
    }


# --------------------------------------------------------------------------- #
# Theorem keys
# --------------------------------------------------------------------------- #
def theorem_prompt_key(premises, conclusion):
    """Renaming-invariant, premise-ORDER-sensitive key (identifies a prompt)."""
    canon = canonical_rename(list(premises) + [conclusion])
    return (tuple(render(p) for p in canon[:-1]), render(canon[-1]))


def theorem_split_key(premises, conclusion):
    """Renaming- AND order-invariant key (identifies a theorem for splitting):
    premise reorderings and atom relabellings collapse to one key, so a theorem
    cannot leak across the train/held-out boundary via either transform."""
    canon = canonical_rename(list(premises) + [conclusion])
    prem = tuple(sorted(render(p) for p in canon[:-1]))
    return (prem, render(canon[-1]))


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def is_degenerate(premises, conclusion):
    if conclusion in premises:                    # conclusion literally a premise
        return True
    if BOT in premises:                           # F |- anything (ex falso)
        return True
    if conclusion[0] == 'or' and conclusion[1] == conclusion[2]:   # A |- ( A v A )
        return True
    return False


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(pool_texts, heldout_frac=0.12, per_length_cap=None, salt='nd-v1',
          forbidden_keys=None):
    """pool_texts: iterable of verified full-string proofs.

    forbidden_keys: set of split_keys (e.g. validation_36 theorems + renamings)
    that must never appear in train OR held-out.

    Returns dict(train=[rec], heldout=[rec], stats=dict). Proofs are pruned,
    filtered, deduped (shortest per prompt), balanced (optional per-length cap),
    and split theorem-disjointly."""
    forbidden_keys = forbidden_keys or set()
    best = {}                       # thm_key -> record (shortest kept)
    n_in = n_pruned = n_degenerate = n_forbidden = 0
    for text in pool_texts:
        n_in += 1
        pruned, changed = prune(text)
        if changed:
            n_pruned += 1
        premises, concl, _ = _split(pruned)
        if is_degenerate(premises, concl):
            n_degenerate += 1
            continue
        if theorem_split_key(premises, concl) in forbidden_keys:
            n_forbidden += 1
            continue
        rec = record(pruned)
        key = rec['thm_key']
        cur = best.get(key)
        if cur is None or rec['n_lines'] < cur['n_lines']:
            best[key] = rec

    recs = list(best.values())

    # optional per-length balancing (stable order: shortest hash first)
    if per_length_cap:
        recs.sort(key=lambda r: _digest(str(r['thm_key']), salt))
        capped, seen = [], {}
        for r in recs:
            L = r['n_lines']
            if seen.get(L, 0) < per_length_cap:
                capped.append(r)
                seen[L] = seen.get(L, 0) + 1
        recs = capped

    # theorem-disjoint split by split_key hash
    train, heldout = [], []
    for r in recs:
        h = _digest(str(r['split_key']), salt)
        (heldout if (h % 10_000) / 10_000 < heldout_frac else train).append(r)

    stats = _stats(recs, train, heldout, n_in, n_pruned, n_degenerate)
    stats['dropped_forbidden'] = n_forbidden       # validation_36 + renamings
    return {'train': train, 'heldout': heldout, 'stats': stats}


def _digest(s, salt):
    return int(hashlib.md5((salt + '|' + s).encode()).hexdigest(), 16)


def _stats(recs, train, heldout, n_in, n_pruned, n_degenerate):
    import collections
    def length_hist(rs):
        c = collections.Counter(r['n_lines'] for r in rs)
        return dict(sorted(c.items()))
    def rule_hist(rs):
        c = collections.Counter()
        for r in rs:
            for rule in set(r['rules']):        # per-proof presence
                c[rule] += 1
        return dict(c.most_common())
    def prem_hist(rs):
        c = collections.Counter(r['n_prem'] for r in rs)
        return dict(sorted(c.items()))

    train_split = {r['split_key'] for r in train}
    held_split = {r['split_key'] for r in heldout}
    overlap = len(train_split & held_split)

    return {
        'pool_in': n_in,
        'pruned_changed': n_pruned,
        'dropped_degenerate': n_degenerate,
        'distinct_theorems': len(recs),
        'train_size': len(train),
        'heldout_size': len(heldout),
        'heldout_train_theorem_overlap': overlap,     # 0 by construction
        'length_hist_all': length_hist(recs),
        'length_hist_train': length_hist(train),
        'length_hist_heldout': length_hist(heldout),
        'rule_hist_all': rule_hist(recs),
        'prem_hist_all': prem_hist(recs),
    }
