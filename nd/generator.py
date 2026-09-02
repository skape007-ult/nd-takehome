"""Forward generator for verifier-accepted natural-deduction proofs (<= 6 lines).

Strategy (in the brief's order):
  * local walk   -- the workhorse: sample premises, then repeatedly fire a local
                    rule whose inputs already exist. Covers PR, R, ANDI, ANDE1,
                    ANDE2, IMPE, ORI1, ORI2, NEGE, BOTE, DN. All depth 0.
  * IMPI          -- forward walk with one subproof inserted: assume A, do a few
                    local steps inside, close to ( A > B ). Adds depth.
  * NEGI / ORE    -- parametric templates that stay <= 6 lines (these are the
                    hardest to hit by random walk, so we construct them directly).

EVERY candidate is gated through nd_verify.verify_text; nothing else is trusted.
The conclusion of a proof is simply its last (depth-0) derived line, so theorems
are non-degenerate by construction (we still filter the rare A |- A cases in
dataset.py). Rendering and formula tuples match the verifier exactly.
"""
import random

from nd_verify import verify_text
from nd_verify.verify import BOT
from .formula import render, atoms, size, ATOMS

MAX_FORMULA_SIZE = 14   # cap node count so proofs stay readable / real


# --------------------------------------------------------------------------- #
# Random formulas
# --------------------------------------------------------------------------- #
def rand_formula(rng, max_depth, p_atom=0.5, allow_f=False):
    """Random formula up to `max_depth`. Connective mix favours & and > so that
    consuming rules (ANDE/IMPE) have inputs to fire on."""
    if max_depth <= 0 or rng.random() < p_atom:
        if allow_f and rng.random() < 0.08:
            return BOT
        return ('atom', rng.choice(ATOMS))
    op = rng.choices(['and', 'imp', 'or', 'not'], weights=[3, 3, 2, 2])[0]
    if op == 'not':
        return ('not', rand_formula(rng, max_depth - 1, p_atom, allow_f))
    return (op,
            rand_formula(rng, max_depth - 1, p_atom, allow_f),
            rand_formula(rng, max_depth - 1, p_atom, allow_f))


def rand_premise(rng):
    """A premise with some structure (depth up to 2), so ANDE/IMPE/NEGE/DN fire."""
    for _ in range(6):
        f = rand_formula(rng, 2, p_atom=0.4)
        if f != BOT and size(f) <= MAX_FORMULA_SIZE:
            return f
    return ('atom', rng.choice(ATOMS))


# --------------------------------------------------------------------------- #
# Proof builder
# --------------------------------------------------------------------------- #
class Builder:
    """Accumulates proof lines and renders the full 'THM ... QED' string.

    Indices start at 1 and are consecutive; `depth` is the box nesting of a line.
    Scoping/citability is managed by the caller via explicit `avail` lists, which
    is why this class stays dumb.
    """

    def __init__(self):
        self.lines = []   # list of dict(idx, depth, formula, rule, refs)

    def add(self, depth, formula, rule, refs):
        idx = len(self.lines) + 1
        self.lines.append({'idx': idx, 'depth': depth, 'formula': formula,
                           'rule': rule, 'refs': list(refs)})
        return idx, formula

    def _render_line(self, ln):
        bars = '| ' * ln['depth']
        ref = ''.join(f" N{r}" for r in ln['refs'])
        return f"N{ln['idx']} {bars}{render(ln['formula'])} : {ln['rule']}{ref} ;"

    def text(self, premises, conclusion):
        head = 'THM '
        if premises:
            head += ' , '.join(render(p) for p in premises) + ' '
        head += 'SEQ ' + render(conclusion) + ' PRF'
        body = ' '.join(self._render_line(ln) for ln in self.lines)
        return head + ' ' + body + ' QED'


# --------------------------------------------------------------------------- #
# One local forward step
# --------------------------------------------------------------------------- #
def _local_candidates(avail, rng, rules):
    """All local-rule moves firing on `avail` = list of (idx, formula).
    Returns list of (rule, refs, result_formula)."""
    cands = []
    have = {}
    for i, f in avail:
        have.setdefault(f, i)   # first index carrying each formula

    for i, f in avail:
        tag = f[0]
        if 'ANDE1' in rules and tag == 'and':
            cands.append(('ANDE1', [i], f[1]))
        if 'ANDE2' in rules and tag == 'and':
            cands.append(('ANDE2', [i], f[2]))
        if 'DN' in rules and tag == 'not' and f[1][0] == 'not':
            cands.append(('DN', [i], f[1][1]))
        if 'IMPE' in rules and tag == 'imp':
            ante = f[1]
            if ante in have:
                cands.append(('IMPE', [i, have[ante]], f[2]))
        if 'NEGE' in rules:                       # a: A, b: ( ~ A ) -> F
            neg = ('not', f)
            if neg in have:
                cands.append(('NEGE', [i, have[neg]], BOT))
        if 'BOTE' in rules and f == BOT:
            cands.append(('BOTE', [i], ('atom', rng.choice(ATOMS))))

    # introducing rules: always available, kept lower-weight
    if avail:
        if 'ANDI' in rules:
            (ia, fa), (ib, fb) = rng.choice(avail), rng.choice(avail)
            g = ('and', fa, fb)
            if size(g) <= MAX_FORMULA_SIZE:
                cands.append(('ANDI', [ia, ib], g))
        if 'ORI1' in rules:
            ia, fa = rng.choice(avail)
            g = ('or', fa, rand_formula(rng, 1))
            if size(g) <= MAX_FORMULA_SIZE:
                cands.append(('ORI1', [ia], g))
        if 'ORI2' in rules:
            ia, fa = rng.choice(avail)
            g = ('or', rand_formula(rng, 1), fa)
            if size(g) <= MAX_FORMULA_SIZE:
                cands.append(('ORI2', [ia], g))
    return cands


CONSUMING = {'ANDE1', 'ANDE2', 'DN', 'IMPE', 'NEGE', 'BOTE'}


def local_step(b, avail, depth, rng, rules):
    """Fire one local rule; append the line; return (idx, formula) or None.
    Prefers consuming rules so proofs make real inferences rather than only
    ballooning conjunctions/disjunctions."""
    cands = _local_candidates(avail, rng, rules)
    if not cands:
        return None
    # avoid deriving a formula already present (would be a dead duplicate)
    present = {f for _, f in avail}
    fresh = [c for c in cands if c[2] not in present]
    pool = fresh or cands
    weights = [3.0 if c[0] in CONSUMING else 1.0 for c in pool]
    rule, refs, g = rng.choices(pool, weights=weights)[0]
    return b.add(depth, g, rule, refs)


LOCAL_RULES = {'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'ORI1', 'ORI2',
               'NEGE', 'BOTE', 'DN', 'R'}


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def gen_local(rng, target_len, rules=LOCAL_RULES):
    """Flat depth-0 local proof of length ~target_len (2..6)."""
    b = Builder()
    n_prem = rng.choices([1, 2, 3], weights=[3, 3, 2])[0]
    prems = [rand_premise(rng) for _ in range(n_prem)]
    avail = [b.add(0, p, 'PR', []) for p in prems]
    for _ in range(max(0, target_len - n_prem)):
        r = local_step(b, avail, 0, rng, rules)
        if r is None:
            break
        avail.append(r)
    if len(b.lines) <= n_prem:            # no real inference happened
        return None
    return b.text(prems, b.lines[-1]['formula'])


def gen_impi(rng, target_len, rules=LOCAL_RULES):
    """Forward proof containing one IMPI subproof: assume A, derive B, close."""
    b = Builder()
    n_prem = rng.choices([0, 1, 2], weights=[2, 3, 2])[0]
    prems = [rand_premise(rng) for _ in range(n_prem)]
    outer = [b.add(0, p, 'PR', []) for p in prems]

    hyp = rand_formula(rng, rng.choice([0, 1]))          # assumption A
    as_idx, _ = b.add(1, hyp, 'AS', [])
    inner = outer + [(as_idx, hyp)]
    n_inner = rng.randint(0, max(0, target_len - n_prem - 2))
    last = (as_idx, hyp)
    for _ in range(n_inner):
        r = local_step(b, inner, 1, rng, rules)
        if r is None:
            break
        inner.append(r)
        last = r
    e_idx, b_formula = last
    concl = ('imp', hyp, b_formula)
    if size(concl) > MAX_FORMULA_SIZE:
        return None
    b.add(0, concl, 'IMPI', [as_idx, e_idx])
    return b.text(prems, concl)


def gen_negi(rng):
    """NEGI template: from ( A > ( ~ A ) ) derive ( ~ A ) in 5 lines.

        N1 ( A > ( ~ A ) ) : PR ;
        N2 |  A            : AS ;
        N3 |  ( ~ A )      : IMPE N1 N2 ;
        N4 |  F            : NEGE N2 N3 ;   (positive A first, then ~A)
        N5 ( ~ A )         : NEGI N2 N4 ;
    """
    A = rand_formula(rng, rng.choice([0, 1]))
    if size(A) > 5:
        return None
    notA = ('not', A)
    prem = ('imp', A, notA)
    b = Builder()
    p1, _ = b.add(0, prem, 'PR', [])
    a2, _ = b.add(1, A, 'AS', [])
    n3, _ = b.add(1, notA, 'IMPE', [p1, a2])
    n4, _ = b.add(1, BOT, 'NEGE', [a2, n3])
    b.add(0, notA, 'NEGI', [a2, n4])
    return b.text([prem], notA)


def gen_ore(rng):
    """ORE templates that stay <= 6 lines. Two shapes:
      (a) commutativity: ( A v B ) |- ( B v A )      (6 lines)
      (b) idempotent:    ( A v A ) |- A               (4 lines)
    """
    A = rand_formula(rng, rng.choice([0, 1]))
    B = rand_formula(rng, rng.choice([0, 1]))
    b = Builder()
    if rng.random() < 0.5 and A != B:
        # commutativity
        prem = ('or', A, B)
        goal = ('or', B, A)
        if size(goal) > MAX_FORMULA_SIZE:
            return None
        p1, _ = b.add(0, prem, 'PR', [])
        s1, _ = b.add(1, A, 'AS', [])
        e1, _ = b.add(1, goal, 'ORI2', [s1])        # ( B v A ) from A
        s2, _ = b.add(1, B, 'AS', [])
        e2, _ = b.add(1, goal, 'ORI1', [s2])        # ( B v A ) from B
        b.add(0, goal, 'ORE', [p1, s1, e1, s2, e2])
        return b.text([prem], goal)
    else:
        # idempotent ( A v A ) |- A
        prem = ('or', A, A)
        p1, _ = b.add(0, prem, 'PR', [])
        s1, _ = b.add(1, A, 'AS', [])
        s2, _ = b.add(1, A, 'AS', [])
        b.add(0, A, 'ORE', [p1, s1, s1, s2, s2])
        return b.text([prem], A)


# --------------------------------------------------------------------------- #
# Top-level sampling
# --------------------------------------------------------------------------- #
STRATEGIES = ['local', 'impi', 'negi', 'ore']


def sample_proof(rng, weights=(6, 3, 1, 1)):
    """Draw one strategy, produce a candidate, verify it. Returns the accepted
    text or None. Length is checked to be in [2, 6]."""
    strat = rng.choices(STRATEGIES, weights=weights)[0]
    if strat == 'local':
        text = gen_local(rng, rng.randint(2, 6))
    elif strat == 'impi':
        text = gen_impi(rng, rng.randint(3, 6))
    elif strat == 'negi':
        text = gen_negi(rng)
    else:
        text = gen_ore(rng)
    if text is None:
        return None
    ok, reason, nl = verify_text(text)
    if not ok or not (2 <= nl <= 6):
        return None
    return text
