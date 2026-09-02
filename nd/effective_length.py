"""Effective length of a proof.

The verifier checks SOUNDNESS, not RELEVANCE: it accepts unused premises and
"dead" lines (lines that are derived but never cited on the way to the
conclusion). Because the headline metric L - P is measured in *length*, this
matters a lot:

  * A padded 6-line training example may contain only 3 real inference steps,
    which corrupts the difficulty gradient the model learns from.
  * In Stage 2, an RL policy can append junk lines to *look* like it writes
    longer proofs -- reward hacking that fakes the L - P number.

The effective length is the number of lines that actually contribute to the
final conclusion. We build the citation DAG and reverse-traverse from the last
line, counting only reachable lines. A cited box counts ALL of its interior
lines (that is what "citing a box" means -- you are using the whole subproof).

`emitted_length` is what the verifier reports (total `;` lines). Always report
both; `effective_length <= emitted_length`.
"""
from nd_verify.verify import parse_proof_tokens, ParseError


def _line_deps(line, idx0, n):
    """Line indices that `line` directly depends on, by rule.

    IMPI/NEGI cite a box (s, e): the dependency is every line s..e.
    ORE cites a disjunction line j plus two boxes (s1,e1),(s2,e2).
    Every other rule's refs are plain line citations.
    PR/AS have no refs and thus no dependencies.
    """
    rule, refs = line['rule'], line['refs']
    lo, hi = idx0, idx0 + n - 1

    def box(s, e):
        if s > e:
            s, e = e, s
        return [k for k in range(s, e + 1) if lo <= k <= hi]

    if rule in ('IMPI', 'NEGI'):
        s, e = refs
        return box(s, e)
    if rule == 'ORE':
        j, s1, e1, s2, e2 = refs
        return [j] + box(s1, e1) + box(s2, e2)
    return list(refs)


def reachable_lines(proof_toks):
    """Set of line indices reachable from the final line via the citation DAG.

    Assumes the proof already parses (verify it first in real use). Raises
    ParseError otherwise.
    """
    lines = parse_proof_tokens(proof_toks)
    if not lines:
        return set(), []
    idx0 = lines[0]['idx']
    n = len(lines)
    by_idx = {ln['idx']: ln for ln in lines}

    final = lines[-1]['idx']
    seen = set()
    stack = [final]
    while stack:
        i = stack.pop()
        if i in seen or i not in by_idx:
            continue
        seen.add(i)
        for d in _line_deps(by_idx[i], idx0, n):
            if d not in seen:
                stack.append(d)
    return seen, lines


def effective_length(proof_toks):
    """Number of lines that contribute to the conclusion (>= 1)."""
    seen, _ = reachable_lines(proof_toks)
    return len(seen)


def lengths_from_text(text):
    """Return (emitted_length, effective_length) for a full 'THM ... QED' string.

    emitted_length matches the verifier's n_lines. Returns (0, 0) if the body
    does not parse.
    """
    toks = text.split() if isinstance(text, str) else list(text)
    try:
        i = toks.index('PRF')
    except ValueError:
        return 0, 0
    body = toks[i + 1:]
    emitted = sum(1 for t in body if t == ';')
    try:
        eff = effective_length(body)
    except ParseError:
        return emitted, 0
    return emitted, eff
