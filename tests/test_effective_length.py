"""Effective-length metric: sanity + padding detection, all against the verifier.

Run: PYTHONPATH=. python3 tests/test_effective_length.py
"""
import os
from nd_verify import verify_text
from nd.effective_length import lengths_from_text
from nd.formula import render, parse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_examples():
    path = os.path.join(HERE, 'examples', 'proofs_2_to_8.txt')
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith('THM'):
            out.append(line)
    return out


def test_examples_metric():
    ex = load_examples()
    assert ex, 'no examples found'
    for s in ex:
        ok, reason, nl = verify_text(s)
        assert ok, f'example did not verify: {reason}\n{s}'
        emitted, eff = lengths_from_text(s)
        assert emitted == nl, f'emitted {emitted} != verifier n_lines {nl}'
        assert 1 <= eff <= emitted, f'bad effective length {eff} vs emitted {emitted}'
    print(f'  {len(ex)} verifier-accepted examples: emitted == n_lines, 1 <= eff <= emitted  OK')


def test_padding_detected():
    """A sound proof padded with a dead premise and a dead line must have
    effective_length strictly below emitted length."""
    # Real proof of Q from ( P > Q ), P is 3 lines. We pad it: add an unused
    # premise R and a dead ANDI line that the conclusion never cites.
    padded = ('THM ( P > Q ) , P , R SEQ Q PRF '
              'N1 ( P > Q ) : PR ; '
              'N2 P : PR ; '
              'N3 R : PR ; '
              'N4 ( P & R ) : ANDI N2 N3 ; '   # dead: never cited by the conclusion
              'N5 Q : IMPE N1 N2 ; '
              'QED')
    ok, reason, nl = verify_text(padded)
    assert ok, f'padded proof should still verify (soundness): {reason}'
    emitted, eff = lengths_from_text(padded)
    assert emitted == 5, emitted
    # reachable from N5: N5 -> N1, N2. Effective = {N5, N1, N2} = 3.
    assert eff == 3, f'expected effective 3, got {eff}'
    print(f'  padded proof: emitted={emitted}  effective={eff}  (2 dead lines removed)  OK')


def test_box_counts_interior():
    """Citing a box counts all its interior lines."""
    s = ('THM SEQ ( ( ~ ( ~ Q ) ) > Q ) PRF '
         'N1 | ( ~ ( ~ Q ) ) : AS ; '
         'N2 | Q : DN N1 ; '
         'N3 ( ( ~ ( ~ Q ) ) > Q ) : IMPI N1 N2 ; '
         'QED')
    ok, reason, nl = verify_text(s)
    assert ok, reason
    emitted, eff = lengths_from_text(s)
    # N3 cites box (N1,N2); both interior lines count. Effective = 3 = emitted.
    assert emitted == 3 and eff == 3, (emitted, eff)
    print(f'  box cite pulls in interior: emitted={emitted}  effective={eff}  OK')


if __name__ == '__main__':
    test_examples_metric()
    test_padding_detected()
    test_box_counts_interior()
    print('effective_length: all tests passed')
