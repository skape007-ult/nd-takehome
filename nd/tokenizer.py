"""Whitespace tokenizer: one model token per surface symbol.

The proof language is tiny and already whitespace-separated, so the simplest
defensible choice is a fixed vocabulary with exactly one id per surface token.
No BPE, no digit-splitting: line indices N1..N64 are atomic tokens. This keeps
the mapping between model output and verifier input transparent (decode is
"join with spaces"), which matters because the verifier is the judge.

Vocabulary (fixed, data-independent, ~110 tokens):
  specials : <pad> <bos> <eos>
  atoms    : P Q R S F
  logic    : ~ & v > ( )
  struct   : THM , SEQ PRF QED | : ;
  rules    : 15 rule names
  indices  : N1 .. N64

The `<eos>` follows QED. Training masks the loss on the prompt (THM..PRF) so the
model is only scored on producing the proof body.
"""
MAX_LINE_INDEX = 64

SPECIALS = ['<pad>', '<bos>', '<eos>']
ATOMS = ['P', 'Q', 'R', 'S', 'F']
LOGIC = ['~', '&', 'v', '>', '(', ')']
STRUCT = ['THM', ',', 'SEQ', 'PRF', 'QED', '|', ':', ';']
RULES = ['ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE',
         'NEGE', 'NEGI', 'BOTE', 'DN', 'PR', 'AS', 'R']
INDICES = [f'N{i}' for i in range(1, MAX_LINE_INDEX + 1)]

VOCAB = SPECIALS + ATOMS + LOGIC + STRUCT + RULES + INDICES


class Tokenizer:
    def __init__(self):
        self.itos = list(VOCAB)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi['<pad>']
        self.bos_id = self.stoi['<bos>']
        self.eos_id = self.stoi['<eos>']
        self.qed_id = self.stoi['QED']

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text, add_bos=True, add_eos=True):
        ids = [self.stoi[t] for t in text.split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t in ('<pad>', '<bos>'):
                continue
            if t == '<eos>':
                break
            out.append(t)
        return ' '.join(out)

    def prompt_len(self, text, add_bos=True):
        """Number of token ids up to and including PRF (the conditioning prompt),
        counting the optional BOS. Used to mask the loss on the prompt."""
        toks = text.split()
        upto = toks.index('PRF') + 1
        return upto + (1 if add_bos else 0)


def build_example(tokenizer, prompt, proof):
    """Return (ids, prompt_len) for one training example: BOS prompt proof EOS."""
    text = prompt.strip() + ' ' + proof.strip()
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return ids, tokenizer.prompt_len(text, add_bos=True)
