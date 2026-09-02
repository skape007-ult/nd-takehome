"""Formula tree helpers.

Formulas use the SAME tuple representation as the verifier (nd_verify.verify),
so anything we build here can be handed straight to `verify_text` via `render`:

    ('atom', 'P')          atom
    ('bot',)               falsum F
    ('not', a)             ( ~ a )
    ('and', a, b)          ( a & b )
    ('or',  a, b)          ( a v b )
    ('imp', a, b)          ( a > b )

We deliberately reuse the verifier's own `parse_formula` for parsing so there is
exactly one parser in the project; only rendering and analysis live here.
"""
from nd_verify.verify import parse_formula, BOT

ATOMS = ('P', 'Q', 'R', 'S')
_BINOP_TOK = {'and': '&', 'or': 'v', 'imp': '>'}


def render(f):
    """Render a formula tuple to the whitespace-separated token string."""
    tag = f[0]
    if tag == 'atom':
        return f[1]
    if tag == 'bot':
        return 'F'
    if tag == 'not':
        return '( ~ ' + render(f[1]) + ' )'
    if tag in _BINOP_TOK:
        return '( ' + render(f[1]) + ' ' + _BINOP_TOK[tag] + ' ' + render(f[2]) + ' )'
    raise ValueError(f'bad formula {f!r}')


def parse(text):
    """Parse a formula string (or token list) to a tuple."""
    toks = text.split() if isinstance(text, str) else list(text)
    f, i = parse_formula(toks, 0)
    if i != len(toks):
        raise ValueError(f'trailing tokens in formula: {text!r}')
    return f


def atoms(f):
    """Set of atom names occurring in f (excludes falsum F)."""
    tag = f[0]
    if tag == 'atom':
        return {f[1]}
    if tag == 'bot':
        return set()
    if tag == 'not':
        return atoms(f[1])
    return atoms(f[1]) | atoms(f[2])


def size(f):
    """Number of nodes in the formula tree (a rough complexity measure)."""
    tag = f[0]
    if tag in ('atom', 'bot'):
        return 1
    if tag == 'not':
        return 1 + size(f[1])
    return 1 + size(f[1]) + size(f[2])


def subst(f, mapping):
    """Rename atoms in f according to `mapping` (name -> name). F is untouched."""
    tag = f[0]
    if tag == 'atom':
        return ('atom', mapping.get(f[1], f[1]))
    if tag == 'bot':
        return f
    if tag == 'not':
        return ('not', subst(f[1], mapping))
    return (tag, subst(f[1], mapping), subst(f[2], mapping))


def canonical_rename(formulas):
    """Return a copy of the formula list with atoms renamed to P, Q, R, S in
    order of first appearance (left-to-right, premises then conclusion).

    This is the key for dedup-up-to-renaming and theorem-disjoint splitting:
    two sequents that differ only by a consistent atom relabelling map to the
    same canonical tuple. Falsum F is a constant and never renamed. If a
    sequent uses more than 4 distinct atoms it cannot occur in this logic, so
    we never need names beyond S.
    """
    mapping = {}
    order = iter(ATOMS)

    def walk(f):
        tag = f[0]
        if tag == 'atom':
            if f[1] not in mapping:
                mapping[f[1]] = next(order)
        elif tag == 'not':
            walk(f[1])
        elif tag != 'bot':
            walk(f[1]); walk(f[2])

    for f in formulas:
        walk(f)
    return tuple(subst(f, mapping) for f in formulas)


def theorem_key(premises, conclusion):
    """Canonical, order-sensitive key for a sequent (premises |- conclusion),
    invariant under atom renaming. Used to dedup theorems and to guarantee
    train/held-out splits are disjoint by theorem."""
    canon = canonical_rename(list(premises) + [conclusion])
    prem = canon[:-1]
    concl = canon[-1]
    return (tuple(render(p) for p in prem), render(concl))
