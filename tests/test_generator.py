"""Generator smoke test: yield, length spread, rule coverage, re-verify.

Run: PYTHONPATH=. python3 tests/test_generator.py
"""
import random, collections
from nd_verify import verify_text
from nd.generator import sample_proof
from nd.effective_length import lengths_from_text

RULES = {'PR', 'AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI',
         'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'BOTE', 'DN'}


def rules_used(text):
    body = text.split('PRF', 1)[1]
    toks = body.split()
    return {t for t in toks if t in RULES}


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


if __name__ == '__main__':
    main()
