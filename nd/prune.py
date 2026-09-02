"""Prune a verified proof to its reachable sub-proof.

The generator (like the verifier) tolerates dead lines and unused premises. For
training we want *tight* proofs, where emitted length == effective length, so
that "length" is an honest measure of difficulty. `prune` keeps only the lines
reachable from the conclusion (a cited box keeps all its interior lines),
renumbers them consecutively, drops premises that are no longer cited, and
re-renders. The result is re-checked with verify_text; on any surprise we fall
back to the original string rather than emit something unverified.
"""
from nd_verify import verify_text
from nd_verify.verify import parse_formula
from .formula import render
from .effective_length import reachable_lines


def _split(text):
    """Return (premises, conclusion, body_tokens) for a full 'THM ... QED'."""
    toks = text.split() if isinstance(text, str) else list(text)
    assert toks[0] == 'THM'
    i = 1
    premises = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            premises.append(f)
            if toks[i] == ',':
                i += 1
                continue
            break
    assert toks[i] == 'SEQ'
    concl, i = parse_formula(toks, i + 1)
    assert toks[i] == 'PRF'
    return premises, concl, toks[i + 1:]


def _render_line(idx, depth, formula, rule, refs):
    bars = '| ' * depth
    ref = ''.join(f" N{r}" for r in refs)
    return f"N{idx} {bars}{render(formula)} : {rule}{ref} ;"


def prune(text):
    """Return (pruned_text, changed: bool). Falls back to `text` if pruning
    would not verify (should not happen for a valid input)."""
    ok, _, _ = verify_text(text)
    if not ok:
        return text, False
    premises, concl, body = _split(text)
    reachable, lines = reachable_lines(body)
    kept = [ln for ln in lines if ln['idx'] in reachable]
    if len(kept) == len(lines):
        return text, False                      # already tight

    remap = {ln['idx']: k + 1 for k, ln in enumerate(kept)}
    new_prems = [ln['formula'] for ln in kept if ln['rule'] == 'PR']
    body_str = ' '.join(
        _render_line(remap[ln['idx']], ln['depth'], ln['formula'], ln['rule'],
                     [remap[r] for r in ln['refs']])
        for ln in kept)
    head = 'THM '
    if new_prems:
        head += ' , '.join(render(p) for p in new_prems) + ' '
    head += 'SEQ ' + render(concl) + ' PRF'
    pruned = head + ' ' + body_str + ' QED'

    if not verify_text(pruned)[0]:
        return text, False                      # safety: never emit unverified
    return pruned, True
