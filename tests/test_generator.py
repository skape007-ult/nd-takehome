"""Generator smoke test: yield, length spread, rule coverage, re-verify.

Run: PYTHONPATH=. python3 tests/test_generator.py
"""
import random, collections
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens
from nd.generator import sample_proof
from nd.effective_length import lengths_from_text

RULES = {'PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI',
         'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN'}


def rules_used(text):
    """Rules used, read from the PARSED rule field of each line.

    Do NOT match surface tokens against RULES: the atom `R` renders as the bare
    token `R`, identical to the reiteration rule, so a token-level counter
    reports R as covered when the generator never emits it. That false green is
    what hid the rule-coverage barrier (see log.md, writeup.md §2).
    """
    body = text.split('PRF', 1)[1].split()
    return {ln['rule'] for ln in parse_proof_tokens(body)}


def main(n_attempts=20000, seed=0):
    rng = random.Random(seed)
    accepted = []
    tries = 0
    for _ in range(n_attempts):
        tries += 1
        t = sample_proof(rng)
        if t:
            accepted.append(t)

    print(f'attempts: {tries}  accepted: {len(accepted)}  '
          f'yield: {len(accepted)/tries:.2f}')

    # every accepted proof must independently verify
    bad = sum(1 for t in accepted if not verify_text(t)[0])
    assert bad == 0, f'{bad} accepted proofs failed re-verification!'
    print(f'  re-verify: {len(accepted)}/{len(accepted)} OK')

    lens = collections.Counter()
    effs = collections.Counter()
    rulecov = collections.Counter()
    padded = 0
    for t in accepted:
        _, _, nl = verify_text(t)
        emitted, eff = lengths_from_text(t)
        lens[nl] += 1
        effs[eff] += 1
        if eff < emitted:
            padded += 1
        for r in rules_used(t):
            rulecov[r] += 1

    print('  emitted length:', dict(sorted(lens.items())))
    print('  effective length:', dict(sorted(effs.items())))
    print(f'  proofs with dead lines (eff < emitted): {padded} '
          f'({padded/len(accepted):.1%})')
    print('  rule coverage:')
    for r in sorted(RULES):
        print(f'    {r:6s} {rulecov[r]:6d}')
    missing = RULES - set(rulecov)
    print('  rules never generated:', sorted(missing) if missing else 'none')
    # Known and deliberate: 'R' is listed in generator.LOCAL_RULES but
    # _local_candidates has no branch that emits it, so the corpus contains zero
    # reiteration steps. This is the coverage barrier diagnosed in writeup.md §6;
    # the test pins it so a future generator change is visible rather than silent.
    assert missing == {'R'}, f'rule coverage changed: missing={sorted(missing)}'


if __name__ == '__main__':
    main()
